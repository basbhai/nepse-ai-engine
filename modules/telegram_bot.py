"""
modules/telegram_bot.py
───────────────────────
Nepal paper trading journal bot via Telegram — MULTI-USER EDITION.

Architecture rules (project-wide):
  - from sheets import ...        ← all DB reads/writes
  - from db.connection import _db ← ONLY for multi-table atomic transactions
  - Never raw psycopg2 outside _db()

Multi-user model:
  - Anyone can /register — creates a PENDING row in paper_users
  - Admin (TELEGRAM_CHAT_ID in .env) approves via /approve <telegram_id>
  - Approved users get isolated capital, portfolio, and trade log rows
  - All paper tables keyed by telegram_id (TEXT) — full isolation
  - Push alerts (BUY signals) go to ALL approved users
  - /reset is admin-only: /reset <telegram_id>

Paper trading tables (add to schema.prisma, run codegen + migrations):
  paper_users       — registration + approval state per telegram_id
  paper_portfolio   — open/closed positions per telegram_id
  paper_capital     — one row per telegram_id (capital state)
  paper_trade_log   — immutable log of every BUY/SELL event per telegram_id

Commands (all users)
────────────────────
  /register                   — request access
  /buy  SYMBOL SHARES PRICE   — open or average into position
  /sell SYMBOL SHARES PRICE   — close or partial sell
  /cancel                     — cancel pending confirmation
  /status                     — open positions (your own)
  /pnl                        — closed trades + win rate (your own)
  /capital                    — paper capital state (your own)
  /signal                     — latest AI signals from market_log
  /mode                       — PAPER or LIVE mode
  /pause  /resume             — circuit breaker (your session only)
  /help

Admin-only commands (TELEGRAM_CHAT_ID only)
───────────────────────────────────────────
  /approve <telegram_id>      — approve a pending registration
  /deny    <telegram_id>      — deny / revoke access
  /users                      — list all registered users + status
  /reset   <telegram_id>      — wipe one user's paper data, restart NPR 1L

Natural language works: "bought nabil 10 at 380", "sell nabil 20 at 400"
Gemini parses and corrects input — typos and mixed language handled.

Market gate: /buy and /sell are blocked outside NEPSE trading hours
  (Sun–Thu 10:45 AM – 3:00 PM NST, non-holiday days only).

ENV vars:
  TELEGRAM_BOT_TOKEN  — bot token from BotFather
  TELEGRAM_CHAT_ID    — admin's Telegram ID (system alerts + admin commands)
  DATABASE_URL        — Neon PostgreSQL
  GEMINI_API_KEY      — for NLP parsing
  GEMINI_MODEL        — default: gemini-2.5-flash
  PAPER_MODE          — "true" | "false"

  cli for sandbox

  python -m modules.telegram_bot --sandbox
"""

import os
import sys
import json
import logging
import argparse
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
    read_tab, write_row, upsert_row,
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
CGT_RATE         = Decimal("0.075")    # 7.5% on profit only
SEBON_PCT        = Decimal("0.00015")  # 0.015%
DP_FEE           = Decimal("25")       # flat per trade

# Conversation states
CONFIRM_BUY   = 1
CONFIRM_SELL  = 2
CONFIRM_RESET = 3

# ─── ENV ─────────────────────────────────────────────────────────────────────
TOKEN          = os.environ["TELEGRAM_BOT_TOKEN"]
ADMIN_CHAT_ID  = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))   # admin only
PAPER_MODE     = os.environ.get("PAPER_MODE", "true").lower() == "true"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# ─── Sandbox mode (set by --sandbox CLI flag) ────────────────────────────────
# Bypasses calendar_guard so you can test /buy and /sell on weekends / after hours.
# All DB writes are tagged test_mode='true' — purge with:
#   DELETE FROM paper_trade_log  WHERE test_mode = 'true';
#   DELETE FROM paper_portfolio  WHERE test_mode = 'true';
#   DELETE FROM paper_capital    WHERE test_mode = 'true' AND telegram_id IN (
#       SELECT telegram_id FROM paper_capital WHERE test_mode = 'true');
# Run:  python modules/telegram_bot.py --sandbox
SANDBOX_MODE: bool = False   # overwritten by parse_args() in main()

# Per-session circuit breakers keyed by telegram_id
_circuit_breakers: dict[int, bool] = {}


# ═══════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def get_user(telegram_id: int) -> Optional[dict]:
    """Fetch paper_users row for this telegram_id, or None."""
    rows = run_raw_sql(
        "SELECT * FROM paper_users WHERE telegram_id = %s",
        (str(telegram_id),)
    )
    return rows[0] if rows else None

def is_approved(telegram_id: int) -> bool:
    user = get_user(telegram_id)
    return user is not None and user.get("status") == "APPROVED"

def is_admin(telegram_id: int) -> bool:
    return telegram_id == ADMIN_CHAT_ID

def get_all_approved_ids() -> list[int]:
    """Return telegram_ids of all APPROVED users (for broadcast alerts)."""
    rows = run_raw_sql(
        "SELECT telegram_id FROM paper_users WHERE status = 'APPROVED'"
    )
    return [int(r["telegram_id"]) for r in rows]

def register_user(telegram_id: int, username: str, full_name: str) -> str:
    """
    Register a new user as PENDING.
    Returns: 'already_approved' | 'already_pending' | 'registered'
    """
    existing = get_user(telegram_id)
    if existing:
        if existing["status"] == "APPROVED":
            return "already_approved"
        if existing["status"] == "PENDING":
            return "already_pending"
        # BLOCKED — re-register attempt
        return "blocked"
    now = nst_now()
    with _db() as cur:
        cur.execute("""
            INSERT INTO paper_users
                (telegram_id, username, full_name, status,
                 registered_at, approved_at, approved_by)
            VALUES (%s, %s, %s, 'PENDING', %s, NULL, NULL)
            ON CONFLICT (telegram_id) DO NOTHING
        """, (str(telegram_id), username or "", full_name or "", now))
    return "registered"

def approve_user(telegram_id: int, approved_by: int) -> bool:
    """Approve a PENDING user and seed their capital row. Returns True if done."""
    user = get_user(telegram_id)
    if not user:
        return False
    now = nst_now()
    with _db() as cur:
        cur.execute("""
            UPDATE paper_users
            SET status = 'APPROVED', approved_at = %s, approved_by = %s
            WHERE telegram_id = %s
        """, (now, str(approved_by), str(telegram_id)))
    # Seed capital row for this user
    _seed_capital_row(telegram_id)
    return True

def deny_user(telegram_id: int) -> bool:
    """Set user status to BLOCKED."""
    user = get_user(telegram_id)
    if not user:
        return False
    with _db() as cur:
        cur.execute("""
            UPDATE paper_users SET status = 'BLOCKED'
            WHERE telegram_id = %s
        """, (str(telegram_id),))
    return True


# ─── Auth guard ─────────────────────────────────────────────────────────────

async def guard(update: Update) -> bool:
    """
    Call at the top of every trading/status command.
    Returns True if the user is approved and can proceed.
    Sends an appropriate reply and returns False otherwise.
    """
    uid = update.effective_user.id
    user = get_user(uid)
    if user is None:
        await update.message.reply_text(
            "You are not registered.\n\nUse /register to request access."
        )
        return False
    status = user.get("status")
    if status == "PENDING":
        await update.message.reply_text(
            "⏳ Your registration is *pending admin approval*.\n"
            "You will be notified when approved.",
            parse_mode="Markdown"
        )
        return False
    if status == "BLOCKED":
        await update.message.reply_text("🚫 Your access has been denied.")
        return False
    return True  # APPROVED


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI NLP PARSER
# ═══════════════════════════════════════════════════════════════════════════

PARSE_SYSTEM = """
You are a parser for a Nepal stock market Telegram trading bot.
The user sends a raw message — a buy/purchase or sell command — can be at any order, formats 
orders, laungauge. you need to make sense of it and
Extract fields and return ONLY valid JSON. No markdown. No backticks. No explanation.

For BUY:  {"action":"BUY","symbol":"NABIL","shares":10,"price":380.0,"error":null}
For SELL: {"action":"SELL","symbol":"NABIL","shares":10,"price":400.0,"error":null}

Rules:
- symbol: uppercase 2-8 chars. Fix obvious typos: "nabl"→"NABIL", "nbl"→"NABIL"
- symbol must match symbols of listed shares in NEPSE only search if you must. if not matched with typos correction, throw error with reason accordingly.
- shares: positive integer
- price: positive float, NPR per share
- "k" = ×1000  e.g. "1.2k" = 1200
- word numbers: "ten"→10, "five"→5, "twenty"→20
- Missing price → set error. Missing shares → set error. Never guess.
- If unclear: {"error":"Cannot parse: <reason>"}
"""

def parse_trade_command(raw: str) -> dict:
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
        "total_cost":  gross + total_fees,
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
# PAPER TABLE HELPERS  — all keyed by telegram_id
# ═══════════════════════════════════════════════════════════════════════════

def get_paper_capital(telegram_id: int) -> dict:
    rows = run_raw_sql(
        "SELECT * FROM paper_capital WHERE telegram_id = %s",
        (str(telegram_id),)
    )
    if not rows:
        _seed_capital_row(telegram_id)
        rows = run_raw_sql(
            "SELECT * FROM paper_capital WHERE telegram_id = %s",
            (str(telegram_id),)
        )
    return rows[0] if rows else {}

def get_paper_position(telegram_id: int, symbol: str) -> Optional[dict]:
    rows = run_raw_sql(
        "SELECT * FROM paper_portfolio WHERE telegram_id = %s AND symbol = %s AND status = 'OPEN'",
        (str(telegram_id), symbol.upper())
    )
    return rows[0] if rows else None

def get_all_open_positions(telegram_id: int) -> list[dict]:
    return run_raw_sql(
        "SELECT * FROM paper_portfolio WHERE telegram_id = %s AND status = 'OPEN' ORDER BY id",
        (str(telegram_id),)
    )

def count_open_positions(telegram_id: int) -> int:
    rows = run_raw_sql(
        "SELECT COUNT(*) AS cnt FROM paper_portfolio WHERE telegram_id = %s AND status = 'OPEN'",
        (str(telegram_id),)
    )
    return int(rows[0]["cnt"]) if rows else 0

def lookup_symbols(query: str) -> list[str]:
    """
    Look up symbols in share_sectors.
    Returns exact match as a single-item list, or partial ILIKE matches (up to 8).
    Returns [] if nothing found.
    """
    exact = run_raw_sql(
        "SELECT symbol FROM share_sectors WHERE UPPER(symbol) = UPPER(%s)",
        (query,),
    )
    if exact:
        return [r["symbol"] for r in exact]
    partial = run_raw_sql(
        "SELECT symbol FROM share_sectors WHERE UPPER(symbol) LIKE UPPER(%s) ORDER BY symbol LIMIT 8",
        (f"%{query}%",),
    )
    return [r["symbol"] for r in partial]

def _seed_capital_row(telegram_id: int):
    """Insert capital row for a newly approved user. Safe to call multiple times."""
    now  = nst_now()
    tmode = "true" if SANDBOX_MODE else "false"
    with _db() as cur:
        cur.execute("""
            INSERT INTO paper_capital
                (telegram_id, starting_capital, current_capital,
                 total_realised_pnl, total_fees_paid, total_cgt_paid,
                 total_trades, total_wins, total_losses, last_updated, test_mode)
            VALUES (%s,%s,%s,'0','0','0','0','0','0',%s,%s)
            ON CONFLICT (telegram_id) DO NOTHING
        """, (str(telegram_id), str(STARTING_CAPITAL), str(STARTING_CAPITAL), now, tmode))


# ═══════════════════════════════════════════════════════════════════════════
# RESET HELPER  — admin only, per user
# ═══════════════════════════════════════════════════════════════════════════

def reset_user_paper_data(telegram_id: int):
    """
    Wipe all paper_portfolio, paper_capital, paper_trade_log rows
    for a specific telegram_id and re-seed capital at NPR 1,00,000.
    Uses _db() for atomicity.
    """
    now = nst_now()
    with _db() as cur:
        cur.execute("DELETE FROM paper_trade_log WHERE telegram_id = %s", (str(telegram_id),))
        cur.execute("DELETE FROM paper_portfolio WHERE telegram_id = %s", (str(telegram_id),))
        cur.execute("DELETE FROM paper_capital WHERE telegram_id = %s", (str(telegram_id),))
        cur.execute("""
            INSERT INTO paper_capital
                (telegram_id, starting_capital, current_capital,
                 total_realised_pnl, total_fees_paid, total_cgt_paid,
                 total_trades, total_wins, total_losses, last_updated)
            VALUES (%s,%s,%s,'0','0','0','0','0','0',%s)
        """, (str(telegram_id), str(STARTING_CAPITAL), str(STARTING_CAPITAL), now))
    log.info("Paper data reset for telegram_id=%s", telegram_id)


# ═══════════════════════════════════════════════════════════════════════════
# ATOMIC BUY TRANSACTION
# ═══════════════════════════════════════════════════════════════════════════

def execute_buy(telegram_id: int, symbol: str, shares: Decimal, price: Decimal) -> dict:
    fees      = calc_buy_fees(price, shares)
    total_out = fees["total_cost"]

    cap       = get_paper_capital(telegram_id)
    available = Decimal(str(cap.get("current_capital", "0")))

    if total_out > available:
        raise ValueError(
            f"Insufficient capital.\n"
            f"Required:  NPR {total_out:,.2f}\n"
            f"Available: NPR {available:,.2f}"
        )

    existing = get_paper_position(telegram_id, symbol)
    now      = nst_now()
    today    = nst_today()
    tmode    = "true" if SANDBOX_MODE else "false"

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
            if count_open_positions(telegram_id) >= MAX_POSITIONS:
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
                    (telegram_id, symbol, status, total_shares, wacc, total_cost,
                     first_buy_date, last_buy_date, buy_count, created_at, updated_at,
                     test_mode)
                VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s,'1',%s,%s,%s)
            """, (str(telegram_id), symbol, str(new_shares), str(new_wacc),
                  str(new_cost), today, today, now, now, tmode))

        new_capital = available - total_out
        cur.execute("""
            UPDATE paper_capital SET
                current_capital  = %s,
                total_fees_paid  = (total_fees_paid::numeric + %s)::text,
                last_updated     = %s
            WHERE telegram_id = %s
        """, (str(new_capital), str(fees["total_fees"]), now, str(telegram_id)))

        cur.execute("""
            INSERT INTO paper_trade_log
                (telegram_id, symbol, action, shares, price, gross_amount,
                 brokerage, sebon, dp_fee, cgt, total_fees, net_amount,
                 capital_before, capital_after, wacc_before, wacc_after,
                 note, created_at, test_mode)
            VALUES (%s,%s,'BUY',%s,%s,%s,%s,%s,%s,'0',%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (str(telegram_id), symbol, str(shares), str(price), str(fees["gross"]),
              str(fees["brokerage"]), str(fees["sebon"]), str(DP_FEE),
              str(fees["total_fees"]), str(-total_out),
              str(available), str(new_capital),
              str(old_wacc), str(wacc_after),
              f"BUY {shares:.0f} @ {price}", now, tmode))

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

def execute_sell(telegram_id: int, symbol: str, shares: Decimal, price: Decimal) -> dict:
    pos = get_paper_position(telegram_id, symbol)
    if not pos:
        raise ValueError(f"No open position for {symbol}.")

    held = Decimal(str(pos["total_shares"]))
    if shares > held:
        raise ValueError(
            f"You hold {held:.0f} shares of {symbol}. Cannot sell {shares:.0f}."
        )

    wacc       = Decimal(str(pos["wacc"]))
    fees       = calc_sell_fees(price, shares, wacc)
    cap        = get_paper_capital(telegram_id)
    available  = Decimal(str(cap.get("current_capital", "0")))
    now        = nst_now()
    today      = nst_today()
    remaining  = held - shares
    tmode      = "true" if SANDBOX_MODE else "false"
    result_str = ("WIN" if fees["net_pnl"] > 0
                  else "LOSS" if fees["net_pnl"] < 0
                  else "BREAKEVEN")

    with _db() as cur:
        if remaining <= 0:
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
            WHERE telegram_id = %s
        """, (str(new_capital),
              str(fees["net_pnl"]), str(fees["total_fees"]), str(fees["cgt"]),
              1 if is_win else 0, 1 if is_loss else 0,
              now, str(telegram_id)))

        cur.execute("""
            INSERT INTO paper_trade_log
                (telegram_id, symbol, action, shares, price, gross_amount,
                 brokerage, sebon, dp_fee, cgt, total_fees, net_amount,
                 capital_before, capital_after, wacc_before, wacc_after,
                 note, created_at, test_mode)
            VALUES (%s,%s,'SELL',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (str(telegram_id), symbol, str(shares), str(price), str(fees["gross"]),
              str(fees["brokerage"]), str(fees["sebon"]), str(DP_FEE),
              str(fees["cgt"]), str(fees["total_fees"]),
              str(fees["net_proceeds"]),
              str(available), str(new_capital),
              str(wacc), str(wacc),
              f"SELL {shares:.0f} @ {price} | {result_str}", now, tmode))

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
    base = "📄 PAPER" if PAPER_MODE else "🔴 LIVE"
    return f"{base} | 🧪 SANDBOX" if SANDBOX_MODE else base

def circuit_label(telegram_id: int) -> str:
    return "🔴 PAUSED" if _circuit_breakers.get(telegram_id) else "🟢 ACTIVE"

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

def sandbox_label() -> str:
    return " | 🧪 SANDBOX" if SANDBOX_MODE else ""

def market_gate_message() -> Optional[str]:
    """
    Returns a block message if the market is closed, else None.
    Returns None unconditionally when SANDBOX_MODE is active —
    all calendar checks are bypassed during sandbox testing.
    Trades are still written to DB (tagged test_mode='true') so you
    can purge them before official paper trading begins.
    """
    if SANDBOX_MODE:
        return None
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
# REGISTRATION COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_register(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid      = update.effective_user.id
    username = update.effective_user.username or ""
    name     = update.effective_user.full_name or ""

    result = register_user(uid, username, name)

    if result == "already_approved":
        await update.message.reply_text(
            "✅ You are already approved and can trade. Use /help."
        )
    elif result == "already_pending":
        await update.message.reply_text(
            "⏳ Your registration is still *pending approval*.\n"
            "The admin will review your request soon.",
            parse_mode="Markdown"
        )
    elif result == "blocked":
        await update.message.reply_text(
            "🚫 Your access request has been denied. Contact the admin."
        )
    else:
        await update.message.reply_text(
            f"✅ *Registration received!*\n\n"
            f"Your request is pending admin approval.\n"
            f"You will be notified here once approved.\n\n"
            f"Your Telegram ID: `{uid}`",
            parse_mode="Markdown"
        )
        # Notify admin
        if ADMIN_CHAT_ID:
            await ctx.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=(
                    f"📬 *New registration request*\n\n"
                    f"Name: {name}\n"
                    f"Username: @{username}\n"
                    f"Telegram ID: `{uid}`\n\n"
                    f"Use `/approve {uid}` or `/deny {uid}`"
                ),
                parse_mode="Markdown"
            )


# ═══════════════════════════════════════════════════════════════════════════
# ADMIN COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_approve(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: `/approve <telegram_id>`", parse_mode="Markdown")
        return
    try:
        target_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid Telegram ID.")
        return

    ok = approve_user(target_id, update.effective_user.id)
    if not ok:
        await update.message.reply_text(f"❌ User `{target_id}` not found.", parse_mode="Markdown")
        return

    await update.message.reply_text(f"✅ User `{target_id}` approved.", parse_mode="Markdown")
    try:
        await ctx.bot.send_message(
            chat_id=target_id,
            text=(
                "🎉 *Your registration has been approved!*\n\n"
                f"Starting capital: NPR 1,00,000 (paper)\n\n"
                "Use /help to see all commands.\n"
                "Use /buy to record your first trade."
            ),
            parse_mode="Markdown"
        )
    except Exception as e:
        log.warning("Could not notify user %s: %s", target_id, e)


async def cmd_deny(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: `/deny <telegram_id>`", parse_mode="Markdown")
        return
    try:
        target_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid Telegram ID.")
        return

    ok = deny_user(target_id)
    if not ok:
        await update.message.reply_text(f"❌ User `{target_id}` not found.", parse_mode="Markdown")
        return
    await update.message.reply_text(f"🚫 User `{target_id}` denied/blocked.", parse_mode="Markdown")
    try:
        await ctx.bot.send_message(
            chat_id=target_id,
            text="🚫 Your registration request has been denied."
        )
    except Exception:
        pass


async def cmd_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return
    rows = run_raw_sql(
        "SELECT telegram_id, username, full_name, status, registered_at FROM paper_users ORDER BY id"
    )
    if not rows:
        await update.message.reply_text("No registered users yet.")
        return
    lines = ["*Registered Users*\n"]
    status_emoji = {"APPROVED": "✅", "PENDING": "⏳", "BLOCKED": "🚫"}
    for r in rows:
        em = status_emoji.get(r["status"], "❓")
        lines.append(
            f"{em} `{r['telegram_id']}` — @{r['username'] or 'no_username'}\n"
            f"   {r['full_name']} | {r['status']} | {r['registered_at']}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
async def cmd_gate_review(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show all PENDING gate proposals for the latest review week."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return

    rows = run_raw_sql(
        """
        SELECT proposal_number, parameter_name, current_value,
               proposed_value, reasoning, false_block_rate, sample_size,
               review_week
        FROM gate_proposals
        WHERE status = 'PENDING'
        ORDER BY review_week DESC, proposal_number::int ASC
        LIMIT 10
        """
    )

    if not rows:
        await update.message.reply_text(
            "✅ No pending gate proposals.\n\n"
            "Proposals are generated by GPT on Sunday review when a filter "
            "gate has >40% false block rate with sufficient samples."
        )
        return

    week = rows[0].get("review_week", "?")
    lines = [f"🔧 *Pending Gate Proposals — Week {week}*\n"]

    for r in rows:
        num       = r.get("proposal_number", "?")
        param     = r.get("parameter_name", "?")
        curr      = r.get("current_value", "?")
        proposed  = r.get("proposed_value", "?")
        reasoning = r.get("reasoning", "")[:200]
        fb_rate   = r.get("false_block_rate", "?")
        n         = r.get("sample_size", "?")

        try:
            fb_pct = f"{float(fb_rate)*100:.0f}%" if fb_rate not in ("?", None, "") else "?"
        except (ValueError, TypeError):
            fb_pct = str(fb_rate)

        lines.append(
            f"*#{num}* `{param}`\n"
            f"  {curr} → *{proposed}*\n"
            f"  False block rate: {fb_pct} (n={n})\n"
            f"  _{reasoning}_\n\n"
            f"  `/approve_{num}` or `/reject_{num}`\n"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_gate_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show gate miss false block rates by category (last 90 days)."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return

    rows = run_raw_sql(
        """
        SELECT
            gate_category,
            COUNT(*) as total,
            COUNT(outcome) as stamped,
            SUM(CASE WHEN outcome = 'FALSE_BLOCK'   THEN 1 ELSE 0 END) as false_blocks,
            SUM(CASE WHEN outcome = 'CORRECT_BLOCK' THEN 1 ELSE 0 END) as correct_blocks,
            SUM(CASE WHEN outcome IS NULL           THEN 1 ELSE 0 END) as pending
        FROM gate_misses
        WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY gate_category
        ORDER BY total DESC
        """
    )

    if not rows:
        await update.message.reply_text(
            "📊 No gate miss data yet.\n\n"
            "Gate misses are recorded from the trading loop. "
            "Check back after a few trading days."
        )
        return

    lines = ["📊 *Gate Miss Stats — Last 90 Days*\n"]

    for r in rows:
        cat      = r.get("gate_category", "?") or "?"
        total    = int(r.get("total", 0) or 0)
        stamped  = int(r.get("stamped", 0) or 0)
        fb       = int(r.get("false_blocks", 0) or 0)
        cb       = int(r.get("correct_blocks", 0) or 0)
        pending  = int(r.get("pending", 0) or 0)

        fb_rate = round(fb / stamped * 100, 1) if stamped > 0 else 0.0
        cb_rate = round(cb / stamped * 100, 1) if stamped > 0 else 0.0

        # Warning emoji if false block rate is high
        if fb_rate >= 50:
            flag = "🚨"
        elif fb_rate >= 40:
            flag = "⚠️"
        else:
            flag = "✅"

        lines.append(
            f"{flag} *{cat}*\n"
            f"  Total: {total} | Pending: {pending} | Stamped: {stamped}\n"
            f"  FALSE: {fb} ({fb_rate}%) | CORRECT: {cb} ({cb_rate}%)\n"
        )

    # Total summary
    total_all   = sum(int(r.get("total",        0) or 0) for r in rows)
    fb_all      = sum(int(r.get("false_blocks", 0) or 0) for r in rows)
    stamped_all = sum(int(r.get("stamped",      0) or 0) for r in rows)
    overall_fb  = round(fb_all / stamped_all * 100, 1) if stamped_all > 0 else 0.0

    lines.append(
        f"\n📈 *Overall* — {total_all} blocks tracked | "
        f"overall false block rate: {overall_fb}%"
    )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_approve_proposal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /approve_1, /approve_2 etc — approve a gate proposal and apply to settings.
    Extracts N from the command text.
    """
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return

    text = update.message.text or ""
    # Extract number after /approve_
    try:
        n = int(text.strip().split("_", 1)[1].split()[0])
    except (IndexError, ValueError):
        await update.message.reply_text(
            "❌ Usage: `/approve_1`, `/approve_2` etc.\n"
            "Use /gate_review to see pending proposals.",
            parse_mode="Markdown"
        )
        return

    # Fetch the proposal — latest review week, matching proposal_number
    rows = run_raw_sql(
        """
        SELECT id, review_week, parameter_name, current_value,
               proposed_value, reasoning
        FROM gate_proposals
        WHERE proposal_number = %s
          AND status = 'PENDING'
        ORDER BY review_week DESC
        LIMIT 1
        """,
        (str(n),)
    )

    if not rows:
        await update.message.reply_text(
            f"❌ No pending proposal #{n} found.\n"
            "Use /gate_review to see current proposals."
        )
        return

    proposal    = rows[0]
    proposal_id = proposal["id"]
    param       = proposal["parameter_name"]
    current_val = proposal["current_value"]
    new_val     = proposal["proposed_value"]
    week        = proposal["review_week"]

    # Apply to settings table
    try:
        from sheets import update_setting
        update_setting(param, new_val, set_by="gate_proposal_approved")
    except Exception as e:
        await update.message.reply_text(
            f"❌ Failed to apply setting `{param}` = `{new_val}`\n"
            f"Error: {e}",
            parse_mode="Markdown"
        )
        return

    # Mark proposal as APPROVED
    now = nst_now()
    try:
        run_raw_sql(
            """
            UPDATE gate_proposals
            SET status = 'APPROVED', decided_at = %s, applied_at = %s
            WHERE id = %s
            """,
            (now, now, str(proposal_id))
        )
    except Exception as e:
        log.error("Failed to mark proposal approved: %s", e)

    await update.message.reply_text(
        f"✅ *Proposal #{n} APPROVED and applied*\n\n"
        f"Parameter: `{param}`\n"
        f"Changed:   `{current_val}` → `{new_val}`\n"
        f"Week:       {week}\n\n"
        f"_Setting is now active in filter_engine.py from next trading cycle._",
        parse_mode="Markdown"
    )
    log.info(
        "Gate proposal #%d approved by admin %s: %s = %s (was %s)",
        n, update.effective_user.id, param, new_val, current_val
    )


async def cmd_reject_proposal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /reject_1, /reject_2 etc — reject a gate proposal without changing settings.
    """
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("🚫 Admin only command.")
        return

    text = update.message.text or ""
    try:
        n = int(text.strip().split("_", 1)[1].split()[0])
    except (IndexError, ValueError):
        await update.message.reply_text(
            "❌ Usage: `/reject_1`, `/reject_2` etc.\n"
            "Use /gate_review to see pending proposals.",
            parse_mode="Markdown"
        )
        return

    rows = run_raw_sql(
        """
        SELECT id, review_week, parameter_name, current_value, proposed_value
        FROM gate_proposals
        WHERE proposal_number = %s
          AND status = 'PENDING'
        ORDER BY review_week DESC
        LIMIT 1
        """,
        (str(n),)
    )

    if not rows:
        await update.message.reply_text(
            f"❌ No pending proposal #{n} found.\n"
            "Use /gate_review to see current proposals."
        )
        return

    proposal    = rows[0]
    proposal_id = proposal["id"]
    param       = proposal["parameter_name"]
    current_val = proposal["current_value"]
    proposed    = proposal["proposed_value"]
    week        = proposal["review_week"]

    now = nst_now()
    try:
        run_raw_sql(
            """
            UPDATE gate_proposals
            SET status = 'REJECTED', decided_at = %s
            WHERE id = %s
            """,
            (now, str(proposal_id))
        )
    except Exception as e:
        log.error("Failed to mark proposal rejected: %s", e)
        await update.message.reply_text(f"❌ DB error: {e}")
        return

    await update.message.reply_text(
        f"🚫 *Proposal #{n} REJECTED*\n\n"
        f"Parameter: `{param}`\n"
        f"Proposed:  `{current_val}` → `{proposed}` _(not applied)_\n"
        f"Week:       {week}\n\n"
        f"_Setting unchanged. GPT may re-propose if evidence persists._",
        parse_mode="Markdown"
    )
    log.info(
        "Gate proposal #%d rejected by admin %s: %s (was %s → %s)",
        n, update.effective_user.id, param, current_val, proposed
    )

async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if not is_admin(uid):
        await update.message.reply_text("🚫 Admin only command.")
        return ConversationHandler.END

    if not ctx.args:
        await update.message.reply_text(
            "Usage: `/reset <telegram_id>`\n\nUse /users to see all IDs.",
            parse_mode="Markdown"
        )
        return ConversationHandler.END

    try:
        target_id = int(ctx.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid Telegram ID.")
        return ConversationHandler.END

    user = get_user(target_id)
    if not user:
        await update.message.reply_text(f"❌ User `{target_id}` not found.", parse_mode="Markdown")
        return ConversationHandler.END

    ctx.user_data["reset_target_id"] = target_id
    await update.message.reply_text(
        f"⚠️ *This will DELETE all paper trades for user* `{target_id}` "
        f"({user.get('full_name')}) *and reset their capital to NPR 1,00,000.*\n\n"
        f"Type /yes to confirm or /no to cancel.",
        parse_mode="Markdown"
    )
    return CONFIRM_RESET


async def confirm_reset_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    target_id = ctx.user_data.get("reset_target_id")
    if not target_id:
        await update.message.reply_text("No pending reset.")
        return ConversationHandler.END
    try:
        reset_user_paper_data(target_id)
        await update.message.reply_text(
            f"✅ Paper data reset for user `{target_id}`.\n"
            f"Capital: NPR 1,00,000. All positions cleared.",
            parse_mode="Markdown"
        )
        try:
            await ctx.bot.send_message(
                chat_id=target_id,
                text=(
                    "🔄 *Your paper trading account has been reset by admin.*\n\n"
                    "Starting capital: NPR 1,00,000\nAll positions cleared."
                ),
                parse_mode="Markdown"
            )
        except Exception:
            pass
    except Exception as e:
        await update.message.reply_text(f"❌ Reset failed: {e}")
    ctx.user_data.pop("reset_target_id", None)
    return ConversationHandler.END


async def confirm_reset_no(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.pop("reset_target_id", None)
    await update.message.reply_text("Reset cancelled. Nothing changed.")
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# USER COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    admin_section = ""
    if is_admin(uid):
        admin_section = (
            "\n*Admin Commands*\n"
            "`/approve <id>` — approve a user\n"
            "`/deny <id>` — deny/block a user\n"
            "`/users` — list all registered users\n"
            "`/reset <id>` — reset one user's paper data\n"
            "\n*Gate Tuning*\n"
            "`/gate_review` — pending threshold proposals\n"
            "`/gate_stats` — false block rates by category\n"
            "`/approve_N` — apply proposal N to settings\n"
            "`/reject_N` — reject proposal N\n"
        )
    await update.message.reply_text(
        f"*Nepal Paper Trading Bot* | {mode_label()}\n\n"
        "*Registration*\n"
        "`/register` — request access\n\n"
        "*Trading* _(market hours only: Sun–Thu 10:45AM–3:00PM NST)_\n"
        "`/buy SYMBOL SHARES PRICE`\n"
        "`/sell SYMBOL SHARES PRICE`\n"
        "`/cancel`\n\n"
        "*Status* _(available anytime)_\n"
        "`/status` — your open positions\n"
        "`/pnl` — your closed trades + win rate\n"
        "`/capital` — your paper capital state\n"
        "`/signal` — latest AI signals\n\n"
        "*Control*\n"
        "`/pause` `/resume` — your circuit breaker\n"
        + admin_section +
        "\n_Natural language works:_\n"
        "`bought nabil 10 at 380`\n"
        "`sell nabil 20 at 400`",
        parse_mode="Markdown",
    )


async def cmd_capital(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    uid = update.effective_user.id
    cap    = get_paper_capital(uid)
    start  = Decimal(str(cap.get("starting_capital", "100000")))
    curr   = Decimal(str(cap.get("current_capital", "100000")))
    pnl    = Decimal(str(cap.get("total_realised_pnl", "0")))
    fees   = Decimal(str(cap.get("total_fees_paid", "0")))
    cgt    = Decimal(str(cap.get("total_cgt_paid", "0")))
    growth = ((curr - start) / start * 100).quantize(Decimal("0.01")) if start else Decimal("0")

    locked_rows = run_raw_sql(
        "SELECT COALESCE(SUM(total_cost::numeric),0) AS lk FROM paper_portfolio "
        "WHERE telegram_id = %s AND status='OPEN'",
        (str(uid),)
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
    if not await guard(update):
        return
    uid       = update.effective_user.id
    positions = get_all_open_positions(uid)
    cap       = get_paper_capital(uid)
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
    if not await guard(update):
        return
    uid    = update.effective_user.id
    cap    = get_paper_capital(uid)
    trades = run_raw_sql(
        "SELECT * FROM paper_portfolio WHERE telegram_id = %s AND status = 'CLOSED' "
        "ORDER BY id DESC LIMIT 20",
        (str(uid),)
    )

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
    if not await guard(update):
        return
    signals = run_raw_sql(
        "SELECT * FROM market_log WHERE outcome = 'PENDING' ORDER BY id DESC LIMIT 5"
    )
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
    uid = update.effective_user.id
    await update.message.reply_text(f"{mode_label()} | {circuit_label(uid)}")


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    uid = update.effective_user.id
    _circuit_breakers[uid] = True
    await update.message.reply_text(
        "🔴 *Circuit breaker ACTIVATED* for your session.\nUse /resume to deactivate.",
        parse_mode="Markdown"
    )


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    uid = update.effective_user.id
    _circuit_breakers[uid] = False
    await update.message.reply_text("🟢 *Circuit breaker DEACTIVATED*", parse_mode="Markdown")


# ═══════════════════════════════════════════════════════════════════════════
# BUY FLOW
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    if not await guard(update):
        return ConversationHandler.END
    uid = update.effective_user.id

    if _circuit_breakers.get(uid):
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

    matches = lookup_symbols(symbol)
    if not matches:
        await update.message.reply_text(f"❌ Symbol *{symbol}* not found in NEPSE.", parse_mode="Markdown")
        return ConversationHandler.END
    if matches[0].upper() != symbol:
        suggestion = ", ".join(f"`{s}`" for s in matches)
        await update.message.reply_text(
            f"❌ Symbol *{symbol}* not found. Did you mean: {suggestion}?",
            parse_mode="Markdown",
        )
        return ConversationHandler.END
    symbol = matches[0]

    fees      = calc_buy_fees(price, shares)
    total_out = fees["total_cost"]
    available = Decimal(str(get_paper_capital(uid).get("current_capital", "0")))
    existing  = get_paper_position(uid, symbol)

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
    uid     = update.effective_user.id
    pending = ctx.user_data.get("pending_buy")
    log.info('buy has been set to yes')
    if not pending:
        await update.message.reply_text("No pending BUY.")
        return ConversationHandler.END
    try:
        r = execute_buy(uid, pending["symbol"], Decimal(pending["shares"]), Decimal(pending["price"]))
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
    if not await guard(update):
        return ConversationHandler.END
    uid = update.effective_user.id

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

    matches = lookup_symbols(symbol)
    if not matches:
        await update.message.reply_text(f"❌ Symbol *{symbol}* not found in NEPSE.", parse_mode="Markdown")
        return ConversationHandler.END
    if matches[0].upper() != symbol:
        suggestion = ", ".join(f"`{s}`" for s in matches)
        await update.message.reply_text(
            f"❌ Symbol *{symbol}* not found. Did you mean: {suggestion}?",
            parse_mode="Markdown",
        )
        return ConversationHandler.END
    symbol = matches[0]

    pos = get_paper_position(uid, symbol)
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
    uid     = update.effective_user.id
    pending = ctx.user_data.get("pending_sell")
    if not pending:
        await update.message.reply_text("No pending SELL.")
        return ConversationHandler.END
    try:
        r  = execute_sell(uid, pending["symbol"], Decimal(pending["shares"]), Decimal(pending["price"]))
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
    await update.message.reply_text("Use /help to see available commands.")


# ═══════════════════════════════════════════════════════════════════════════
# PUSH ALERTS  — broadcast to all approved users
# Called externally by claude_analyst.py
# ═══════════════════════════════════════════════════════════════════════════

async def push_buy_signal(
    app: Application,
    symbol: str,
    confidence: int,
    entry: float,
    stop: float,
    target: float,
    reason: str
):
    """Broadcast a BUY signal alert to ALL approved users."""
    approved_ids = get_all_approved_ids()
    if not approved_ids:
        return

    text = (
        f"🚨 *BUY Signal* | {mode_label()}\n\n"
        f"*{symbol}* | Conf: {confidence}%\n"
        f"Entry: {fmt_npr(entry)} | SL: {fmt_npr(stop)} | Target: {fmt_npr(target)}\n\n"
        f"_{reason[:200]}_\n\n"
        f"Use `/buy {symbol} <shares> {entry}` to log."
    )
    for uid in approved_ids:
        try:
            await app.bot.send_message(chat_id=uid, text=text, parse_mode="Markdown")
        except Exception as e:
            log.warning("Could not push signal to %s: %s", uid, e)


async def push_system_alert(app: Application, message: str):
    """Send a system alert to admin only (circuit breaker, errors, etc.)."""
    if ADMIN_CHAT_ID:
        try:
            await app.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=f"⚙️ *System Alert*\n\n{message}",
                parse_mode="Markdown"
            )
        except Exception as e:
            log.error("Could not send system alert: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA NOTE
# Add these models to schema.prisma, then run codegen + migrations.
# ═══════════════════════════════════════════════════════════════════════════
#
# model paper_users {
#   id            Serial
#   telegram_id   Text!  @unique
#   username      Text
#   full_name     Text
#   status        Text   @default("PENDING")  // PENDING | APPROVED | BLOCKED
#   registered_at Text
#   approved_at   Text
#   approved_by   Text
#   @@index([status])
# }
#
# model paper_capital {
#   id                  Serial
#   telegram_id         Text!  @unique   // one row per approved user
#   starting_capital    Text  @default("100000")
#   current_capital     Text  @default("100000")
#   total_realised_pnl  Text  @default("0")
#   total_fees_paid     Text  @default("0")
#   total_cgt_paid      Text  @default("0")
#   total_trades        Text  @default("0")
#   total_wins          Text  @default("0")
#   total_losses        Text  @default("0")
#   last_updated        Text
#   test_mode           Text  @default("false")  // "true" when seeded via --sandbox
#   @@index([telegram_id])
# }
#
# model paper_portfolio {
#   id              Serial
#   telegram_id     Text!
#   symbol          Text!
#   status          Text  @default("OPEN")
#   total_shares    Text
#   wacc            Text
#   total_cost      Text
#   first_buy_date  Text
#   last_buy_date   Text
#   buy_count       Text  @default("1")
#   exit_date       Text
#   exit_price      Text
#   exit_shares     Text
#   gross_pnl       Text
#   sell_fees       Text
#   cgt_paid        Text
#   net_pnl         Text
#   result          Text
#   created_at      Text
#   updated_at      Text
#   test_mode       Text  @default("false")  // "true" = written during --sandbox run
#   @@index([telegram_id])
#   @@index([telegram_id, status])
#   @@index([symbol])
#   @@index([test_mode])
# }
#
# model paper_trade_log {
#   id             Serial
#   telegram_id    Text!
#   symbol         Text!
#   action         Text!
#   shares         Text
#   price          Text
#   gross_amount   Text
#   brokerage      Text
#   sebon          Text
#   dp_fee         Text
#   cgt            Text
#   total_fees     Text
#   net_amount     Text
#   capital_before Text
#   capital_after  Text
#   wacc_before    Text
#   wacc_after     Text
#   note           Text
#   created_at     Text
#   test_mode      Text  @default("false")  // "true" = written during --sandbox run
#   @@index([telegram_id])
#   @@index([symbol])
#   @@index([test_mode])
# }
#
# ── Purge sandbox data before official paper trading begins ──────────────────
# DELETE FROM paper_trade_log WHERE test_mode = 'true';
# DELETE FROM paper_portfolio  WHERE test_mode = 'true';
# UPDATE paper_capital SET
#     current_capital='100000', total_realised_pnl='0',
#     total_fees_paid='0',      total_cgt_paid='0',
#     total_trades='0',         total_wins='0',    total_losses='0',
#     test_mode='false'
# WHERE test_mode = 'true';
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global SANDBOX_MODE

    parser = argparse.ArgumentParser(description="NEPSE Paper Trading Telegram Bot")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help=(
            "Bypass calendar_guard — allow /buy and /sell outside market hours. "
            "All DB writes are tagged test_mode='true' for easy purge before live trading. "
            "Use this for code/flow testing only."
        ),
    )
    args = parser.parse_args()
    SANDBOX_MODE = args.sandbox

    if SANDBOX_MODE:
        log.warning("=" * 60)
        log.warning("  SANDBOX MODE ACTIVE — calendar checks bypassed")
        log.warning("  All trades written with test_mode='true'")
        log.warning("  Purge before official paper trading:")
        log.warning("    DELETE FROM paper_trade_log WHERE test_mode='true';")
        log.warning("    DELETE FROM paper_portfolio  WHERE test_mode='true';")
        log.warning("    UPDATE paper_capital SET")
        log.warning("      current_capital='100000', total_realised_pnl='0',")
        log.warning("      total_fees_paid='0', total_cgt_paid='0',")
        log.warning("      total_trades='0', total_wins='0', total_losses='0'")
        log.warning("    WHERE test_mode='true';")
        log.warning("=" * 60)

    log.info("Starting multi-user bot | %s", mode_label())
    app = Application.builder().token(TOKEN).build()

    buy_conv = ConversationHandler(
        entry_points=[
            CommandHandler("buy", cmd_buy),
            MessageHandler(
                filters.TEXT & ~filters.COMMAND &
                filters.Regex(r'(?i)\b(buy\w*|bought|purchas\w*|kineko\w*|kine\w*)'),
                cmd_buy,
            ),
        ],
        states={CONFIRM_BUY: [
            CommandHandler("yes", confirm_buy_yes),
            CommandHandler("no",  confirm_buy_no),
            MessageHandler(filters.Regex(r'(?i)^yes$'), confirm_buy_yes),
            MessageHandler(filters.Regex(r'(?i)^no$'),  confirm_buy_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    sell_conv = ConversationHandler(
        entry_points=[
            CommandHandler("sell", cmd_sell),
            MessageHandler(
                filters.TEXT & ~filters.COMMAND &
                filters.Regex(r'(?i)\b(sell\w*|sold|becho\w*)'),
                cmd_sell,
            ),
        ],
        states={CONFIRM_SELL: [
            CommandHandler("yes", confirm_sell_yes),
            CommandHandler("no",  confirm_sell_no),
            MessageHandler(filters.Regex(r'(?i)^yes$'), confirm_sell_yes),
            MessageHandler(filters.Regex(r'(?i)^no$'),  confirm_sell_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )
    reset_conv = ConversationHandler(
        entry_points=[CommandHandler("reset", cmd_reset)],
        states={CONFIRM_RESET: [
            CommandHandler("yes", confirm_reset_yes),
            CommandHandler("no",  confirm_reset_no),
            MessageHandler(filters.Regex(r'(?i)^yes$'), confirm_reset_yes),
            MessageHandler(filters.Regex(r'(?i)^no$'),  confirm_reset_no),
        ]},
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(buy_conv)
    app.add_handler(sell_conv)
    app.add_handler(reset_conv)

    app.add_handler(CommandHandler("register", cmd_register))
    app.add_handler(CommandHandler("approve",  cmd_approve))
    app.add_handler(CommandHandler("deny",     cmd_deny))
    app.add_handler(CommandHandler("users",    cmd_users))
    app.add_handler(CommandHandler("cancel",   cmd_cancel))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("pnl",      cmd_pnl))
    app.add_handler(CommandHandler("capital",  cmd_capital))
    app.add_handler(CommandHandler("signal",   cmd_signal))
    app.add_handler(CommandHandler("mode",     cmd_mode))
    app.add_handler(CommandHandler("pause",    cmd_pause))
    app.add_handler(CommandHandler("resume",   cmd_resume))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("start",    cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, nlp_fallback))
        # Gate tuning commands (admin only)
    app.add_handler(CommandHandler("gate_review", cmd_gate_review))
    app.add_handler(CommandHandler("gate_stats",  cmd_gate_stats))

    # Pattern-matched /approve_N and /reject_N
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^/approve_\d+'),
        cmd_approve_proposal,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^/reject_\d+'),
        cmd_reject_proposal,
    ))

    log.info("Bot polling started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()