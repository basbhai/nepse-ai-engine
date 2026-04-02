"""
modules/telegram_bot.py
───────────────────────
Nepal paper trading journal bot via Telegram.

Architecture rules (project-wide):
  - from sheets import ...        ← all DB reads/writes
  - from db.connection import _db ← ONLY for multi-table atomic transactions
  - Never raw psycopg2 outside _db()

Paper trading tables (add to schema.prisma, run codegen + migrations):
  paper_portfolio   — open/closed positions with WACC averaging
  paper_capital     — single row (id=1) capital state
  paper_trade_log   — immutable log of every BUY/SELL event

Commands
────────
  /buy  SYMBOL SHARES PRICE   — open or average into position
  /sell SYMBOL SHARES PRICE   — close or partial sell
  /cancel                     — cancel pending confirmation
  /status                     — open positions
  /pnl                        — closed trades + win rate
  /capital                    — paper capital state
  /signal                     — latest AI signals from market_log
  /mode                       — PAPER or LIVE mode
  /pause  /resume             — circuit breaker
  /reset                      — ⚠ wipe paper tables, restart NPR 1,00,000
  /help

Natural language works: "bought nabil 10 at 380", "sell nabil 20 at 400"
Gemini parses and corrects input — typos and mixed language handled.

Market gate: /buy and /sell are blocked outside NEPSE trading hours
  (Sun–Thu 10:45 AM – 3:00 PM NST, non-holiday days only).
  Uses calendar_guard.is_open() + get_status() for reason + next open time.

ENV vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DATABASE_URL,
  GEMINI_API_KEY, GEMINI_MODEL, PAPER_MODE
"""

import os
import json
import logging
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters, ConversationHandler,
)

from sheets import (
    read_tab, read_tab_where, write_row, upsert_row,
    update_row, run_raw_sql, get_setting,
)
from db.connection import _db
from calendar_guard import is_open as market_is_open, get_status as market_status

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    level=logging.INFO)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
NST              = ZoneInfo("Asia/Kathmandu")
STARTING_CAPITAL = Decimal("100000.00")
MAX_POSITIONS    = 15
CGT_RATE         = Decimal("0.075")   # 7.5% on profit only
SEBON_PCT        = Decimal("0.00015") # 0.015%
DP_FEE           = Decimal("25")      # flat per trade

# Conversation states
CONFIRM_BUY   = 1
CONFIRM_SELL  = 2
CONFIRM_RESET = 3

# ─── ENV ─────────────────────────────────────────────────────────────────────
TOKEN          = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID        = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))
PAPER_MODE     = os.environ.get("PAPER_MODE", "true").lower() == "true"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_circuit_breaker = os.environ.get("CIRCUIT_BREAKER", "false").lower() == "true"


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI NLP PARSER
# Parses natural language → structured trade dict
# ═══════════════════════════════════════════════════════════════════════════

PARSE_SYSTEM = """
You are a parser for a Nepal stock market Telegram trading bot.
The user sends a raw message — a buy or sell command — possibly with
typos, mixed Nepali-English, abbreviations, or wrong word order.

Extract fields and return ONLY valid JSON. No markdown. No backticks. No explanation.

For BUY:  {"action":"BUY","symbol":"NABIL","shares":10,"price":380.0,"error":null}
For SELL: {"action":"SELL","symbol":"NABIL","shares":10,"price":400.0,"error":null}

Rules:
- symbol: uppercase 2-8 chars. Fix obvious typos: "nabl"→"NABIL", "nbl"→"NABIL"
- shares: positive integer
- price: positive float, NPR per share
- "k" = ×1000  e.g. "1.2k" = 1200
- word numbers: "ten"→10, "five"→5, "twenty"→20
- Missing price → set error. Missing shares → set error. Never guess.
- If unclear: {"error":"Cannot parse: <reason>"}
"""

def parse_trade_command(raw: str) -> dict:
    """Call Gemini to parse a natural language trade command. Returns dict."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}
    try:
        import google.genai as genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=raw,
            config=types.GenerateContentConfig(
                system_instruction=PARSE_SYSTEM,
                temperature=0.1,
            ),
        )
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "AI returned invalid response. Please rephrase."}
    except Exception as e:
        log.error("Gemini parse error: %s", e)
        return {"error": f"AI error: {str(e)[:80]}"}


# ═══════════════════════════════════════════════════════════════════════════
# TIME HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def nst_now() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

def nst_today() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d")

def hold_days(date_str: str) -> int:
    try:
        return (date.today() - date.fromisoformat(date_str)).days
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# FEE CALCULATIONS  (Nepal NEPSE)
# ═══════════════════════════════════════════════════════════════════════════

def _brokerage(gross: Decimal) -> Decimal:
    """NEPSE tiered brokerage commission."""
    if gross <= Decimal("2500"):
        return Decimal("10")
    elif gross <= Decimal("50000"):
        rate = Decimal("0.0036")
    elif gross <= Decimal("500000"):
        rate = Decimal("0.0033")
    elif gross <= Decimal("2000000"):
        rate = Decimal("0.0031")
    elif gross <= Decimal("10000000"):
        rate = Decimal("0.0027")
    else:
        rate = Decimal("0.0024")
    return (gross * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)

def calc_buy_fees(price: Decimal, shares: Decimal) -> dict:
    gross      = (price * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    brokerage  = _brokerage(gross)
    sebon      = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    total_fees = brokerage + sebon + DP_FEE
    return {
        "gross":       gross,
        "brokerage":   brokerage,
        "sebon":       sebon,
        "dp_fee":      DP_FEE,
        "total_fees":  total_fees,
        "total_cost":  gross + total_fees,   # what leaves capital
    }

def calc_sell_fees(price: Decimal, shares: Decimal, wacc: Decimal) -> dict:
    gross        = (price * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    brokerage    = _brokerage(gross)
    sebon        = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    total_fees   = brokerage + sebon + DP_FEE
    cost_basis   = (wacc * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    gross_profit = gross - cost_basis
    cgt = (gross_profit * CGT_RATE).quantize(Decimal("0.01"), ROUND_HALF_UP) \
          if gross_profit > 0 else Decimal("0")
    net_proceeds = gross - total_fees - cgt
    net_pnl      = net_proceeds - cost_basis
    return {
        "gross":        gross,
        "brokerage":    brokerage,
        "sebon":        sebon,
        "dp_fee":       DP_FEE,
        "total_fees":   total_fees,
        "gross_profit": gross_profit,
        "cgt":          cgt,
        "net_proceeds": net_proceeds,
        "net_pnl":      net_pnl,
        "cost_basis":   cost_basis,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PAPER TABLE READ HELPERS  (via sheets.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_paper_capital() -> dict:
    """Return the single paper_capital row (id=1)."""
    rows = read_tab_where("paper_capital", {"id": "1"})
    if not rows:
        _seed_capital_row()
        rows = read_tab_where("paper_capital", {"id": "1"})
    return rows[0] if rows else {}

def get_paper_position(symbol: str) -> Optional[dict]:
    rows = read_tab_where("paper_portfolio", {"symbol": symbol.upper(), "status": "OPEN"})
    return rows[0] if rows else None

def get_all_open_positions() -> list[dict]:
    return read_tab_where("paper_portfolio", {"status": "OPEN"},
                          order_by="id", desc=False)

def count_open_positions() -> int:
    rows = run_raw_sql(
        "SELECT COUNT(*) AS cnt FROM paper_portfolio WHERE status = 'OPEN'"
    )
    return int(rows[0]["cnt"]) if rows else 0

def _seed_capital_row():
    """Insert id=1 capital row if missing. Called on first use."""
    now = nst_now()
    # Use _db directly — upsert_row needs TABLE_COLUMNS which requires codegen first
    with _db() as cur:
        cur.execute("""
            INSERT INTO paper_capital
                (id, starting_capital, current_capital, total_realised_pnl,
                 total_fees_paid, total_cgt_paid, total_trades,
                 total_wins, total_losses, last_updated)
            VALUES (1,%s,%s,'0','0','0','0','0','0',%s)
            ON CONFLICT (id) DO NOTHING
        """, (str(STARTING_CAPITAL), str(STARTING_CAPITAL), now))


# ═══════════════════════════════════════════════════════════════════════════
# RESET HELPER
# ═══════════════════════════════════════════════════════════════════════════

def reset_paper_tables():
    """
    DROP and recreate paper_portfolio, paper_capital, paper_trade_log.
    Uses _db() directly because schema.prisma manages these tables via migrations —
    this is a destructive admin operation, not a normal write.
    """
    with _db() as cur:
        cur.execute("DROP TABLE IF EXISTS paper_trade_log")
        cur.execute("DROP TABLE IF EXISTS paper_portfolio")
        cur.execute("DROP TABLE IF EXISTS paper_capital")
        cur.execute("""
            CREATE TABLE paper_capital (
                id                  SERIAL PRIMARY KEY,
                starting_capital    TEXT DEFAULT '100000',
                current_capital     TEXT DEFAULT '100000',
                total_realised_pnl  TEXT DEFAULT '0',
                total_fees_paid     TEXT DEFAULT '0',
                total_cgt_paid      TEXT DEFAULT '0',
                total_trades        TEXT DEFAULT '0',
                total_wins          TEXT DEFAULT '0',
                total_losses        TEXT DEFAULT '0',
                last_updated        TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE paper_portfolio (
                id              SERIAL PRIMARY KEY,
                symbol          TEXT NOT NULL,
                status          TEXT DEFAULT 'OPEN',
                total_shares    TEXT,
                wacc            TEXT,
                total_cost      TEXT,
                first_buy_date  TEXT,
                last_buy_date   TEXT,
                buy_count       TEXT DEFAULT '1',
                exit_date       TEXT,
                exit_price      TEXT,
                exit_shares     TEXT,
                gross_pnl       TEXT,
                sell_fees       TEXT,
                cgt_paid        TEXT,
                net_pnl         TEXT,
                result          TEXT,
                created_at      TEXT,
                updated_at      TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE paper_trade_log (
                id             SERIAL PRIMARY KEY,
                symbol         TEXT NOT NULL,
                action         TEXT NOT NULL,
                shares         TEXT,
                price          TEXT,
                gross_amount   TEXT,
                brokerage      TEXT,
                sebon          TEXT,
                dp_fee         TEXT,
                cgt            TEXT,
                total_fees     TEXT,
                net_amount     TEXT,
                capital_before TEXT,
                capital_after  TEXT,
                wacc_before    TEXT,
                wacc_after     TEXT,
                note           TEXT,
                created_at     TEXT
            )
        """)
        # Seed capital row
        cur.execute("""
            INSERT INTO paper_capital
                (id, starting_capital, current_capital, total_realised_pnl,
                 total_fees_paid, total_cgt_paid, total_trades,
                 total_wins, total_losses, last_updated)
            VALUES (1,%s,%s,'0','0','0','0','0','0',%s)
        """, (str(STARTING_CAPITAL), str(STARTING_CAPITAL), nst_now()))
    log.info("Paper tables reset. Capital = NPR 100,000.")


# ═══════════════════════════════════════════════════════════════════════════
# ATOMIC BUY TRANSACTION
# Uses _db() to keep 3-table update in one commit.
# ═══════════════════════════════════════════════════════════════════════════

def execute_buy(symbol: str, shares: Decimal, price: Decimal) -> dict:
    """
    Opens new position or averages into existing.
    All three table updates (paper_portfolio, paper_capital, paper_trade_log)
    happen in one _db() transaction — atomic.
    Raises ValueError on insufficient capital or position limit.
    """
    fees      = calc_buy_fees(price, shares)
    total_out = fees["total_cost"]

    cap       = get_paper_capital()
    available = Decimal(str(cap.get("current_capital", "0")))

    if total_out > available:
        raise ValueError(
            f"Insufficient capital.\n"
            f"Required:  NPR {total_out:,.2f}\n"
            f"Available: NPR {available:,.2f}"
        )

    existing = get_paper_position(symbol)
    now      = nst_now()
    today    = nst_today()

    with _db() as cur:
        if existing:
            old_shares = Decimal(str(existing["total_shares"]))
            old_cost   = Decimal(str(existing["total_cost"]))
            new_shares = old_shares + shares
            new_cost   = old_cost + total_out
            new_wacc   = (new_cost / new_shares).quantize(Decimal("0.0001"), ROUND_HALF_UP)
            old_wacc   = Decimal(str(existing["wacc"]))

            cur.execute("""
                UPDATE paper_portfolio SET
                    total_shares  = %s,
                    wacc          = %s,
                    total_cost    = %s,
                    last_buy_date = %s,
                    buy_count     = (buy_count::int + 1)::text,
                    updated_at    = %s
                WHERE id = %s
            """, (str(new_shares), str(new_wacc), str(new_cost),
                  today, now, existing["id"]))
            wacc_after = new_wacc
        else:
            if count_open_positions() >= MAX_POSITIONS:
                raise ValueError(
                    f"Max {MAX_POSITIONS} open positions reached. Close one first."
                )
            new_shares = shares
            new_cost   = total_out
            new_wacc   = (total_out / shares).quantize(Decimal("0.0001"), ROUND_HALF_UP)
            old_wacc   = Decimal("0")
            wacc_after = new_wacc

            cur.execute("""
                INSERT INTO paper_portfolio
                    (symbol, status, total_shares, wacc, total_cost,
                     first_buy_date, last_buy_date, buy_count, created_at, updated_at)
                VALUES (%s,'OPEN',%s,%s,%s,%s,%s,'1',%s,%s)
            """, (symbol, str(new_shares), str(new_wacc), str(new_cost),
                  today, today, now, now))

        new_capital = available - total_out
        cur.execute("""
            UPDATE paper_capital SET
                current_capital  = %s,
                total_fees_paid  = (total_fees_paid::numeric + %s)::text,
                last_updated     = %s
            WHERE id = 1
        """, (str(new_capital), str(fees["total_fees"]), now))

        cur.execute("""
            INSERT INTO paper_trade_log
                (symbol, action, shares, price, gross_amount,
                 brokerage, sebon, dp_fee, cgt, total_fees, net_amount,
                 capital_before, capital_after, wacc_before, wacc_after,
                 note, created_at)
            VALUES (%s,'BUY',%s,%s,%s,%s,%s,%s,'0',%s,%s,%s,%s,%s,%s,%s,%s)
        """, (symbol, str(shares), str(price), str(fees["gross"]),
              str(fees["brokerage"]), str(fees["sebon"]), str(DP_FEE),
              str(fees["total_fees"]), str(-total_out),
              str(available), str(new_capital),
              str(old_wacc), str(wacc_after),
              f"BUY {shares:.0f} @ {price}", now))

    return {
        "symbol":    symbol,
        "shares":    shares,
        "price":     price,
        "fees":      fees,
        "total_out": total_out,
        "new_wacc":  wacc_after,
        "averaged":  existing is not None,
        "cap_after": new_capital,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ATOMIC SELL TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════

def execute_sell(symbol: str, shares: Decimal, price: Decimal) -> dict:
    """
    Full or partial close. CGT deducted immediately on profit.
    All three table updates in one _db() transaction — atomic.
    Raises ValueError if no position or insufficient shares.
    """
    pos = get_paper_position(symbol)
    if not pos:
        raise ValueError(f"No open position for {symbol}.")

    held = Decimal(str(pos["total_shares"]))
    if shares > held:
        raise ValueError(
            f"You hold {held:.0f} shares of {symbol}. Cannot sell {shares:.0f}."
        )

    wacc       = Decimal(str(pos["wacc"]))
    fees       = calc_sell_fees(price, shares, wacc)
    cap        = get_paper_capital()
    available  = Decimal(str(cap.get("current_capital", "0")))
    now        = nst_now()
    today      = nst_today()
    remaining  = held - shares
    result_str = ("WIN" if fees["net_pnl"] > 0
                  else "LOSS" if fees["net_pnl"] < 0
                  else "BREAKEVEN")

    with _db() as cur:
        if remaining <= 0:
            # Full close
            cur.execute("""
                UPDATE paper_portfolio SET
                    status      = 'CLOSED',
                    exit_date   = %s,
                    exit_price  = %s,
                    exit_shares = %s,
                    gross_pnl   = %s,
                    sell_fees   = %s,
                    cgt_paid    = %s,
                    net_pnl     = %s,
                    result      = %s,
                    updated_at  = %s
                WHERE id = %s
            """, (today, str(price), str(shares),
                  str(fees["gross_profit"]), str(fees["total_fees"]),
                  str(fees["cgt"]), str(fees["net_pnl"]),
                  result_str, now, pos["id"]))
        else:
            # Partial close — reduce shares, keep WACC
            new_cost = (wacc * remaining).quantize(Decimal("0.01"), ROUND_HALF_UP)
            cur.execute("""
                UPDATE paper_portfolio SET
                    total_shares = %s,
                    total_cost   = %s,
                    updated_at   = %s
                WHERE id = %s
            """, (str(remaining), str(new_cost), now, pos["id"]))

        new_capital = available + fees["net_proceeds"]
        is_win  = fees["net_pnl"] > 0
        is_loss = fees["net_pnl"] < 0

        cur.execute("""
            UPDATE paper_capital SET
                current_capital    = %s,
                total_realised_pnl = (total_realised_pnl::numeric + %s)::text,
                total_fees_paid    = (total_fees_paid::numeric + %s)::text,
                total_cgt_paid     = (total_cgt_paid::numeric + %s)::text,
                total_trades       = (total_trades::int + 1)::text,
                total_wins         = (total_wins::int + %s)::text,
                total_losses       = (total_losses::int + %s)::text,
                last_updated       = %s
            WHERE id = 1
        """, (str(new_capital),
              str(fees["net_pnl"]),
              str(fees["total_fees"]),
              str(fees["cgt"]),
              1 if is_win else 0,
              1 if is_loss else 0,
              now))

        cur.execute("""
            INSERT INTO paper_trade_log
                (symbol, action, shares, price, gross_amount,
                 brokerage, sebon, dp_fee, cgt, total_fees, net_amount,
                 capital_before, capital_after, wacc_before, wacc_after,
                 note, created_at)
            VALUES (%s,'SELL',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (symbol, str(shares), str(price), str(fees["gross"]),
              str(fees["brokerage"]), str(fees["sebon"]), str(DP_FEE),
              str(fees["cgt"]), str(fees["total_fees"]),
              str(fees["net_proceeds"]),
              str(available), str(new_capital),
              str(wacc), str(wacc),
              f"SELL {shares:.0f} @ {price} | {result_str}", now))

    return {
        "symbol":    symbol,
        "shares":    shares,
        "price":     price,
        "wacc":      wacc,
        "fees":      fees,
        "result":    result_str,
        "remaining": remaining,
        "cap_after": new_capital,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def mode_label() -> str:
    return "📄 PAPER" if PAPER_MODE else "🔴 LIVE"

def circuit_label() -> str:
    return "🔴 PAUSED" if _circuit_breaker else "🟢 ACTIVE"

def fmt_npr(val, sign=False) -> str:
    v = float(val)
    prefix = ("+" if v >= 0 else "") if sign else ""
    return f"NPR {prefix}{v:,.2f}"

def win_rate_str(cap: dict) -> str:
    total = int(cap.get("total_trades", "0"))
    wins  = int(cap.get("total_wins", "0"))
    if total == 0:
        return "No trades yet"
    return f"{wins}/{total} ({wins/total*100:.1f}%)"

def market_gate_message() -> Optional[str]:
    """
    Returns a human-readable block message if market is closed, else None.
    Uses calendar_guard.get_status() for reason + next open time.
    """
    if market_is_open():
        return None
    s = market_status()
    return (
        f"🔒 *Market is closed* — trading blocked.\n\n"
        f"{s['message']}\n\n"
        f"Next open: *{s['next_open']}*\n"
        f"Use /status or /capital anytime."
    )


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"*Nepal Paper Trading Bot* | {mode_label()} | {circuit_label()}\n\n"
        "*Trading* _(market hours only: Sun–Thu 10:45AM–3:00PM NST)_\n"
        "`/buy SYMBOL SHARES PRICE`\n"
        "`/sell SYMBOL SHARES PRICE`\n"
        "`/cancel`\n\n"
        "*Status* _(available anytime)_\n"
        "`/status` — open positions\n"
        "`/pnl` — closed trades + win rate\n"
        "`/capital` — paper capital state\n"
        "`/signal` — latest AI signals\n\n"
        "*Control*\n"
        "`/pause` `/resume` — circuit breaker\n"
        "`/reset` ⚠️ — wipe paper data, restart NPR 1,00,000\n\n"
        "_Natural language works:_\n"
        "`bought nabil 10 at 380`\n"
        "`sell nabil 20 at 400`",
        parse_mode="Markdown",
    )


async def cmd_capital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cap    = get_paper_capital()
    start  = Decimal(str(cap.get("starting_capital", "100000")))
    curr   = Decimal(str(cap.get("current_capital", "100000")))
    pnl    = Decimal(str(cap.get("total_realised_pnl", "0")))
    fees   = Decimal(str(cap.get("total_fees_paid", "0")))
    cgt    = Decimal(str(cap.get("total_cgt_paid", "0")))
    growth = ((curr - start) / start * 100).quantize(Decimal("0.01")) if start else Decimal("0")

    locked_rows = run_raw_sql(
        "SELECT COALESCE(SUM(total_cost::numeric),0) AS lk FROM paper_portfolio WHERE status='OPEN'"
    )
    locked = Decimal(str(locked_rows[0]["lk"])) if locked_rows else Decimal("0")

    await update.message.reply_text(
        f"*Paper Capital* | {mode_label()}\n\n"
        f"Starting:      {fmt_npr(start)}\n"
        f"Current cash:  {fmt_npr(curr)}\n"
        f"Locked trades: {fmt_npr(locked)}\n\n"
        f"Realised P&L:  {fmt_npr(pnl, sign=True)}\n"
        f"Fees paid:     {fmt_npr(fees)}\n"
        f"CGT paid:      {fmt_npr(cgt)}\n\n"
        f"Overall growth: {growth:+.2f}%\n"
        f"Win rate:       {win_rate_str(cap)}\n"
        f"Total trades:   {cap.get('total_trades','0')}",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    positions = get_all_open_positions()
    cap       = get_paper_capital()
    curr      = Decimal(str(cap.get("current_capital", "0")))

    if not positions:
        await update.message.reply_text(
            f"No open positions | {mode_label()}\n"
            f"Available cash: {fmt_npr(curr)}"
        )
        return

    lines = [f"*Open Positions* ({len(positions)}/{MAX_POSITIONS}) | {mode_label()}\n"]
    for pos in positions:
        sym    = pos["symbol"]
        shares = pos.get("total_shares", "0")
        wacc   = pos.get("wacc", "0")
        cost   = pos.get("total_cost", "0")
        days   = hold_days(pos.get("first_buy_date") or nst_today())
        buys   = pos.get("buy_count", "1")
        lines.append(
            f"📌 *{sym}*\n"
            f"  Shares: {float(shares):.0f} | WACC: {fmt_npr(wacc)}\n"
            f"  Cost basis: {fmt_npr(cost)} | Held: {days}d | Avg: {buys}×\n"
        )
    lines.append(f"\nAvailable cash: {fmt_npr(curr)}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    cap    = get_paper_capital()
    trades = read_tab_where("paper_portfolio", {"status": "CLOSED"},
                            order_by="id", desc=True, limit=20)

    lines = [
        f"*Closed Trades* | {mode_label()}\n",
        f"Win rate:      {win_rate_str(cap)}",
        f"Realised P&L:  {fmt_npr(cap.get('total_realised_pnl','0'), sign=True)}",
        f"Total fees:    {fmt_npr(cap.get('total_fees_paid','0'))}",
        f"Total CGT:     {fmt_npr(cap.get('total_cgt_paid','0'))}\n",
    ]
    if not trades:
        lines.append("No closed trades yet.")
    else:
        lines.append("*Recent (last 20):*")
        for t in trades:
            em  = "🟢" if t.get("result") == "WIN" else ("🔴" if t.get("result") == "LOSS" else "⚪")
            pnl = float(t.get("net_pnl") or 0)
            cgt = float(t.get("cgt_paid") or 0)
            lines.append(
                f"{em} *{t['symbol']}* | {t.get('exit_date','?')}\n"
                f"  {float(t.get('exit_shares') or 0):.0f}sh @ "
                f"{fmt_npr(t.get('exit_price','0'))} | "
                f"Net: {fmt_npr(pnl, sign=True)} | CGT: {fmt_npr(cgt)}"
            )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    signals = read_tab_where("market_log", {"outcome": "PENDING"},
                             order_by="id", desc=True, limit=5)
    if not signals:
        await update.message.reply_text("No pending signals in market_log.")
        return
    lines = ["*Latest AI Signals*\n"]
    for s in signals:
        em = "🟢" if s.get("action") == "BUY" else ("🔴" if s.get("action") == "AVOID" else "⚪")
        lines.append(
            f"{em} *{s['symbol']}* — {s.get('action')} | {s.get('date')}\n"
            f"  Conf: {s.get('confidence')}% | Entry: {s.get('entry_price')}\n"
            f"  _{str(s.get('reasoning',''))[:100]}..._\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"{mode_label()} | {circuit_label()}")


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _circuit_breaker
    _circuit_breaker = True
    await update.message.reply_text(
        "🔴 *Circuit breaker ACTIVATED*\nUse /resume to deactivate.",
        parse_mode="Markdown"
    )


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _circuit_breaker
    _circuit_breaker = False
    await update.message.reply_text("🟢 *Circuit breaker DEACTIVATED*", parse_mode="Markdown")


# ═══════════════════════════════════════════════════════════════════════════
# RESET FLOW
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "⚠️ *This will DELETE all paper trades and reset capital to NPR 1,00,000.*\n\n"
        "Type /yes to confirm or /no to cancel.",
        parse_mode="Markdown"
    )
    return CONFIRM_RESET

async def confirm_reset_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        reset_paper_tables()
        await update.message.reply_text(
            "✅ Paper tables reset.\nCapital: NPR 1,00,000\nAll positions cleared."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Reset failed: {e}")
    return ConversationHandler.END

async def confirm_reset_no(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Reset cancelled. Nothing changed.")
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# BUY FLOW
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    if _circuit_breaker:
        await update.message.reply_text("🔴 Circuit breaker ACTIVE. Use /resume first.")
        return ConversationHandler.END

    gate = market_gate_message()
    if gate:
        await update.message.reply_text(gate, parse_mode="Markdown")
        return ConversationHandler.END

    raw = " ".join(ctx.args) if ctx.args else update.message.text.strip()
    if not raw:
        await update.message.reply_text(
            "Usage: `/buy SYMBOL SHARES PRICE`\nExample: `/buy NABIL 10 380`",
            parse_mode="Markdown"
        )
        return ConversationHandler.END

    await update.message.reply_text("⏳ Parsing...")
    parsed = parse_trade_command(f"BUY: {raw}")

    if parsed.get("error"):
        await update.message.reply_text(f"❌ {parsed['error']}")
        return ConversationHandler.END
    if parsed.get("action") != "BUY":
        await update.message.reply_text("❌ That looks like a SELL. Use /sell.")
        return ConversationHandler.END

    try:
        symbol = parsed["symbol"].upper()
        shares = Decimal(str(int(parsed["shares"])))
        price  = Decimal(str(parsed["price"]))
    except (KeyError, InvalidOperation, ValueError) as e:
        await update.message.reply_text(f"❌ Parse error: {e}")
        return ConversationHandler.END

    fees      = calc_buy_fees(price, shares)
    total_out = fees["total_cost"]
    available = Decimal(str(get_paper_capital().get("current_capital", "0")))
    existing  = get_paper_position(symbol)

    avg_note = ""
    if existing:
        old_shares = Decimal(str(existing["total_shares"]))
        old_cost   = Decimal(str(existing["total_cost"]))
        new_shares = old_shares + shares
        new_wacc   = ((old_cost + total_out) / new_shares).quantize(Decimal("0.0001"), ROUND_HALF_UP)
        avg_note   = (
            f"\n_Averaging: hold {old_shares:.0f}sh @ WACC {fmt_npr(existing['wacc'])}_\n"
            f"_New WACC after: {fmt_npr(new_wacc)}_"
        )

    ctx.user_data["pending_buy"] = {
        "symbol": symbol, "shares": str(shares), "price": str(price)
    }

    await update.message.reply_text(
        f"*Confirm BUY* | {mode_label()}\n\n"
        f"*{symbol}* | {shares:.0f} shares @ {fmt_npr(price)}\n"
        f"────────────────────\n"
        f"Gross:      {fmt_npr(fees['gross'])}\n"
        f"Brokerage:  {fmt_npr(fees['brokerage'])}\n"
        f"SEBON:      {fmt_npr(fees['sebon'])}\n"
        f"DP fee:     {fmt_npr(DP_FEE)}\n"
        f"Total fees: {fmt_npr(fees['total_fees'])}\n"
        f"────────────────────\n"
        f"*Total out: {fmt_npr(total_out)}*\n"
        f"Available:  {fmt_npr(available)}\n"
        f"After buy:  {fmt_npr(available - total_out)}"
        f"{avg_note}\n\n"
        f"Type /yes to confirm or /no to cancel.",
        parse_mode="Markdown"
    )
    return CONFIRM_BUY


async def confirm_buy_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    pending = ctx.user_data.get("pending_buy")
    if not pending:
        await update.message.reply_text("No pending BUY.")
        return ConversationHandler.END
    try:
        r = execute_buy(pending["symbol"], Decimal(pending["shares"]), Decimal(pending["price"]))
        avg_txt = "_(averaged)_" if r["averaged"] else "_(new position)_"
        await update.message.reply_text(
            f"✅ *BUY recorded* {avg_txt}\n\n"
            f"*{r['symbol']}* — {r['shares']:.0f}sh @ {fmt_npr(r['price'])}\n"
            f"WACC: {fmt_npr(r['new_wacc'])}\n"
            f"Capital remaining: {fmt_npr(r['cap_after'])}",
            parse_mode="Markdown"
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        log.error("BUY error: %s", e)
        await update.message.reply_text(f"❌ DB error: {e}")
    ctx.user_data.pop("pending_buy", None)
    return ConversationHandler.END


async def confirm_buy_no(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.pop("pending_buy", None)
    await update.message.reply_text("❌ BUY cancelled.")
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# SELL FLOW
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_sell(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    gate = market_gate_message()
    if gate:
        await update.message.reply_text(gate, parse_mode="Markdown")
        return ConversationHandler.END

    raw = " ".join(ctx.args) if ctx.args else update.message.text.strip()
    if not raw:
        await update.message.reply_text(
            "Usage: `/sell SYMBOL SHARES PRICE`\nExample: `/sell NABIL 20 400`",
            parse_mode="Markdown"
        )
        return ConversationHandler.END

    await update.message.reply_text("⏳ Parsing...")
    parsed = parse_trade_command(f"SELL: {raw}")

    if parsed.get("error"):
        await update.message.reply_text(f"❌ {parsed['error']}")
        return ConversationHandler.END
    if parsed.get("action") != "SELL":
        await update.message.reply_text("❌ That looks like a BUY. Use /buy.")
        return ConversationHandler.END

    try:
        symbol = parsed["symbol"].upper()
        shares = Decimal(str(int(parsed["shares"])))
        price  = Decimal(str(parsed["price"]))
    except (KeyError, InvalidOperation, ValueError) as e:
        await update.message.reply_text(f"❌ Parse error: {e}")
        return ConversationHandler.END

    pos = get_paper_position(symbol)
    if not pos:
        await update.message.reply_text(
            f"❌ No open position for *{symbol}*. Use /status.",
            parse_mode="Markdown"
        )
        return ConversationHandler.END

    held = Decimal(str(pos["total_shares"]))
    if shares > held:
        await update.message.reply_text(
            f"❌ You hold {held:.0f} shares of {symbol}. Cannot sell {shares:.0f}."
        )
        return ConversationHandler.END

    wacc      = Decimal(str(pos["wacc"]))
    fees      = calc_sell_fees(price, shares, wacc)
    remaining = held - shares
    em        = "🟢" if fees["net_pnl"] > 0 else "🔴"
    partial   = f"\n_{remaining:.0f} shares remain at WACC {fmt_npr(wacc)}_" if remaining > 0 else ""

    ctx.user_data["pending_sell"] = {
        "symbol": symbol, "shares": str(shares), "price": str(price)
    }

    await update.message.reply_text(
        f"*Confirm SELL* | {mode_label()}\n\n"
        f"*{symbol}* | {shares:.0f} shares @ {fmt_npr(price)}\n"
        f"Entry WACC: {fmt_npr(wacc)}\n"
        f"────────────────────\n"
        f"Gross sale:  {fmt_npr(fees['gross'])}\n"
        f"Brokerage:   {fmt_npr(fees['brokerage'])}\n"
        f"SEBON:       {fmt_npr(fees['sebon'])}\n"
        f"DP fee:      {fmt_npr(DP_FEE)}\n"
        f"CGT (7.5%):  {fmt_npr(fees['cgt'])}\n"
        f"Total fees:  {fmt_npr(fees['total_fees'])}\n"
        f"────────────────────\n"
        f"{em} *Net P&L: {fmt_npr(fees['net_pnl'], sign=True)}*\n"
        f"Proceeds back to capital: {fmt_npr(fees['net_proceeds'])}"
        f"{partial}\n\n"
        f"Type /yes to confirm or /no to cancel.",
        parse_mode="Markdown"
    )
    return CONFIRM_SELL


async def confirm_sell_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    pending = ctx.user_data.get("pending_sell")
    if not pending:
        await update.message.reply_text("No pending SELL.")
        return ConversationHandler.END
    try:
        r  = execute_sell(pending["symbol"], Decimal(pending["shares"]), Decimal(pending["price"]))
        em = "🟢" if r["result"] == "WIN" else "🔴"
        partial = f"\n_{r['remaining']:.0f} shares still open_" if r["remaining"] > 0 else ""
        await update.message.reply_text(
            f"✅ *SELL recorded*\n\n"
            f"{em} *{r['symbol']}* — {r['result']}\n"
            f"Net P&L:   {fmt_npr(r['fees']['net_pnl'], sign=True)}\n"
            f"CGT paid:  {fmt_npr(r['fees']['cgt'])}\n"
            f"Capital now: {fmt_npr(r['cap_after'])}"
            f"{partial}",
            parse_mode="Markdown"
        )
    except ValueError as e:
        await update.message.reply_text(f"❌ {e}")
    except Exception as e:
        log.error("SELL error: %s", e)
        await update.message.reply_text(f"❌ DB error: {e}")
    ctx.user_data.pop("pending_sell", None)
    return ConversationHandler.END


async def confirm_sell_no(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.pop("pending_sell", None)
    await update.message.reply_text("❌ SELL cancelled.")
    return ConversationHandler.END


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.pop("pending_buy", None)
    ctx.user_data.pop("pending_sell", None)
    await update.message.reply_text("❌ Cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# NLP FALLBACK
# ═══════════════════════════════════════════════════════════════════════════

async def nlp_fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    buy_kw  = ["buy", "bought", "purchase", "kineko", "kine"]
    sell_kw = ["sell", "sold", "becho", "becheko"]
    if any(k in text for k in buy_kw):
        ctx.args = text.split()
        return await cmd_buy(update, ctx)
    elif any(k in text for k in sell_kw):
        ctx.args = text.split()
        return await cmd_sell(update, ctx)
    else:
        await update.message.reply_text("Use /help to see available commands.")


# ═══════════════════════════════════════════════════════════════════════════
# PUSH ALERTS  (called externally by claude_analyst)
# ═══════════════════════════════════════════════════════════════════════════

async def push_buy_signal(app: Application, symbol: str, confidence: int,
                           entry: float, stop: float, target: float, reason: str):
    if CHAT_ID == 0:
        return
    await app.bot.send_message(
        chat_id=CHAT_ID,
        text=(
            f"🚨 *BUY Signal* | {mode_label()}\n\n"
            f"*{symbol}* | Conf: {confidence}%\n"
            f"Entry: {fmt_npr(entry)} | SL: {fmt_npr(stop)} | Target: {fmt_npr(target)}\n\n"
            f"_{reason[:200]}_\n\n"
            f"Use `/buy {symbol} <shares> {entry}` to log."
        ),
        parse_mode="Markdown"
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Ensure capital row exists on startup
    try:
        _seed_capital_row()
    except Exception as e:
        log.warning("Could not seed capital row (tables may not exist yet): %s", e)

    log.info("Starting bot | %s | %s", mode_label(), circuit_label())
    app = Application.builder().token(TOKEN).build()

    buy_conv = ConversationHandler(
        entry_points=[CommandHandler("buy", cmd_buy)],
        states={CONFIRM_BUY: [
            CommandHandler("yes", confirm_buy_yes),
            CommandHandler("no",  confirm_buy_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )
    sell_conv = ConversationHandler(
        entry_points=[CommandHandler("sell", cmd_sell)],
        states={CONFIRM_SELL: [
            CommandHandler("yes", confirm_sell_yes),
            CommandHandler("no",  confirm_sell_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )
    reset_conv = ConversationHandler(
        entry_points=[CommandHandler("reset", cmd_reset)],
        states={CONFIRM_RESET: [
            CommandHandler("yes", confirm_reset_yes),
            CommandHandler("no",  confirm_reset_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(buy_conv)
    app.add_handler(sell_conv)
    app.add_handler(reset_conv)
    app.add_handler(CommandHandler("cancel",  cmd_cancel))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("pnl",     cmd_pnl))
    app.add_handler(CommandHandler("capital", cmd_capital))
    app.add_handler(CommandHandler("signal",  cmd_signal))
    app.add_handler(CommandHandler("mode",    cmd_mode))
    app.add_handler(CommandHandler("pause",   cmd_pause))
    app.add_handler(CommandHandler("resume",  cmd_resume))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("start",   cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, nlp_fallback))

    log.info("Bot polling started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()