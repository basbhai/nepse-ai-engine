"""
telegram_bot.py
───────────────
Nepal stock market journal bot — paper & live trading via Telegram.

Commands
────────
Trading:
  /buy  SYMBOL SHARES PRICE   — open position (natural language OK)
  /sell SYMBOL PRICE          — close position
  /cancel                     — cancel any pending confirmation

Status:
  /status                     — open positions + unrealised P&L
  /pnl                        — closed trades, win rate, fees paid
  /mode                       — show PAPER or LIVE mode
  /signal                     — latest market_log recommendations

Control:
  /pause                      — activate circuit breaker
  /resume                     — deactivate circuit breaker
  /help                       — show all commands

Auto-alerts (push):
  • Daily EOD summary at 3:15 PM NST
  • BUY signal detected by claude_analyst

Architecture:
  • OpenRouter (free model) parses & corrects messy input
  • Always confirms before writing to DB
  • Writes to: portfolio, trade_journal
  • Reads from:  portfolio, trade_journal, market_log
  • Never touches: learning_hub (owned by auditor.py + ChatGPT)

ENV vars required:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID          — your personal chat id for push alerts
  DATABASE_URL              — Neon PostgreSQL connection string
  OPENROUTER_API_KEY
  PAPER_MODE                — "true" or "false"
  CIRCUIT_BREAKER           — "false" by default, bot can flip to "true"
"""

import os
import json
import asyncio
import logging
import httpx
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP
from AI.gemini import ask_ai_text

import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
NST = ZoneInfo("Asia/Kathmandu")


# Nepal broker fee structure (approximate — adjust to your broker)
def _calc_brokerage(amount: float) -> float:
    """NEPSE tiered brokerage commission."""
    if amount <= 2_500:
        return 10.0
    elif amount <= 50_000:
        return amount * 0.0036
    elif amount <= 500_000:
        return amount * 0.0033
    elif amount <= 2_000_000:
        return amount * 0.0031
    elif amount <= 10_000_000:
        return amount * 0.0027
    else:
        return amount * 0.0024

FEE_BUY_PCT   = Decimal("0.004")    # 0.40% brokerage on buy
FEE_SELL_PCT  = Decimal("0.004")    # 0.40% brokerage on sell
SEBON_PCT     = Decimal("0.00015")  # 0.015% SEBON levy
DP_FEE        = Decimal("25")       # Rs 25 flat DP fee per trade

# OpenRouter — free model (change model slug as needed)
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"   # free tier

# Conversation states
CONFIRM_BUY  = 1
CONFIRM_SELL = 2

# ─── ENV ─────────────────────────────────────────────────────────────────────
TOKEN        = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID      = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))
DATABASE_URL = os.environ["DATABASE_URL"]
OR_KEY       = os.environ["OPENROUTER_API_KEY"]
PAPER_MODE   = os.environ.get("PAPER_MODE", "true").lower() == "true"

# Runtime mutable state (persisted via simple env-style file for restarts)
_circuit_breaker = os.environ.get("CIRCUIT_BREAKER", "false").lower() == "true"


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def nst_now() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")


def nst_today() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d")


def execute_one(sql: str, params: tuple = ()):
    """Run INSERT/UPDATE, return nothing."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def fetch_all(sql: str, params: tuple = ()) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_one(sql: str, params: tuple = ()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


# ═══════════════════════════════════════════════════════════════════════════
# FEE CALCULATIONS  (Nepal)
# ═══════════════════════════════════════════════════════════════════════════

def calc_buy_fees(price: Decimal, shares: int) -> Decimal:
    """Total extra cost on top of share price when buying."""
    gross = price * shares
    brokerage = (gross * _calc_brokerage(gross)).quantize(Decimal("0.01"), ROUND_HALF_UP)
    sebon     = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return brokerage + sebon + DP_FEE


def calc_sell_fees(price: Decimal, shares: int) -> Decimal:
    """Total cost deducted from proceeds when selling."""
    gross = price * shares
    brokerage = (gross * FEE_SELL_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    sebon     = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return brokerage + sebon + DP_FEE


def calc_pnl(entry_price: Decimal, exit_price: Decimal, shares: int) -> dict:
    """
    Returns dict with:
      gross_pnl, buy_fees, sell_fees, total_fees, net_pnl, return_pct
    """
    buy_fees  = calc_buy_fees(entry_price, shares)
    sell_fees = calc_sell_fees(exit_price, shares)
    gross     = (exit_price - entry_price) * shares
    net       = gross - buy_fees - sell_fees
    total_cost = entry_price * shares + buy_fees
    ret_pct   = (net / total_cost * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return {
        "gross_pnl":  float(gross),
        "buy_fees":   float(buy_fees),
        "sell_fees":  float(sell_fees),
        "total_fees": float(buy_fees + sell_fees),
        "net_pnl":    float(net),
        "return_pct": float(ret_pct),
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPENROUTER — NLP CORRECTION LAYER
# ═══════════════════════════════════════════════════════════════════════════

PARSE_SYSTEM = """
You are a parser for a Nepal stock market Telegram trading bot.
The user will send a raw message that is a buy or sell command — possibly with
typos, mixed Nepali-English, abbreviations, or wrong order.

Your job: extract the fields and return ONLY valid JSON, no markdown, no explanation.

For BUY commands return:
{"action":"BUY","symbol":"NABIL","shares":10,"price":1240.0,"error":null}

For SELL commands return:
{"action":"SELL","symbol":"NABIL","price":1290.0,"error":null}

Rules:
- symbol: uppercase, 2-8 chars (Nepal stock symbols)
- shares: positive integer
- price: positive float, NPR
- If you cannot parse confidently, return {"error":"Cannot parse — what did you mean?"}
- Never guess a price from nothing. If price is missing, set error.
- Common shorthand: "nabil" → "NABIL", "ten" → 10, "k" = 1000 (e.g. "1.2k" = 1200)
"""




# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO DB OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_open_positions() -> list:
    return fetch_all(
        "SELECT * FROM portfolio WHERE status = 'OPEN' ORDER BY entry_date DESC"
    )


def get_open_position(symbol: str):
    return fetch_one(
        "SELECT * FROM portfolio WHERE symbol = %s AND status = 'OPEN' LIMIT 1",
        (symbol.upper(),),
    )


def insert_portfolio(data: dict) -> int:
    """Insert new open position. Returns new row id."""
    sql = """
        INSERT INTO portfolio (
            symbol, entry_date, entry_price, shares, total_cost,
            current_price, current_value, pnl_npr, pnl_pct,
            peak_price, stop_type, stop_level, trail_active,
            trail_stop, status, exit_date, exit_price, exit_reason
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, 'OPEN', NULL, NULL, NULL
        ) RETURNING id
    """
    ep    = Decimal(str(data["price"]))
    sh    = int(data["shares"])
    fees  = calc_buy_fees(ep, sh)
    total = ep * sh + fees

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                data["symbol"],
                nst_today(),
                str(ep),
                str(sh),
                str(total.quantize(Decimal("0.01"))),
                str(ep),                          # current_price = entry at open
                str((ep * sh).quantize(Decimal("0.01"))),
                "0.00",                           # pnl_npr at open
                "0.00",                           # pnl_pct at open
                str(ep),                          # peak_price
                "FIXED",                          # stop_type default
                str((ep * Decimal("0.95")).quantize(Decimal("0.01"))),  # 5% stop default
                "false",
                "0",
            ))
            new_id = cur.fetchone()["id"]
        conn.commit()
    return new_id


def close_portfolio(portfolio_id: int, exit_price: Decimal, exit_reason: str):
    sql = """
        UPDATE portfolio
        SET status      = 'CLOSED',
            exit_date   = %s,
            exit_price  = %s,
            exit_reason = %s
        WHERE id = %s
    """
    execute_one(sql, (nst_today(), str(exit_price), exit_reason, portfolio_id))


def insert_trade_journal(data: dict):
    """Write a closed trade to trade_journal (filled by bot, auditor enriches later)."""
    sql = """
        INSERT INTO trade_journal (
            created_at, symbol, sector, paper_mode,
            entry_date, entry_price, shares, allocation_npr,
            primary_signal, secondary_signal, candle_pattern, confidence_at_entry,
            rsi_entry, macd_hist_entry, bb_signal_entry, ema_trend_entry,
            obv_trend_entry, conf_score_entry, volume_ratio_entry, atr_pct_entry,
            market_state_entry, geo_score_entry, nepal_score_entry,
            combined_geo_entry, nepse_index_entry,
            stop_loss_planned, target_planned, hold_days_planned,
            exit_date, exit_price, exit_reason,
            market_state_exit, geo_score_exit, nepal_score_exit,
            combined_geo_exit, nepse_index_exit,
            hold_days_actual, return_pct, pnl_npr, result,
            geo_delta, nepal_delta, combined_geo_delta,
            nepse_return_pct, alpha_vs_nepse,
            loss_cause, lesson_ids
        ) VALUES (
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s
        )
    """
    p = data
    execute_one(sql, (
        nst_now(),
        p["symbol"], p.get("sector","UNKNOWN"), p["paper_mode"],
        p["entry_date"], p["entry_price"], p["shares"], p.get("allocation_npr",""),
        p.get("primary_signal","MANUAL"), p.get("secondary_signal","NONE"),
        p.get("candle_pattern","NONE"), p.get("confidence_at_entry",""),
        # technical fields — blank at this stage, auditor enriches
        "","","","",
        "","","","",
        p.get("market_state_entry",""), p.get("geo_score_entry",""),
        p.get("nepal_score_entry",""), p.get("combined_geo_entry",""),
        p.get("nepse_index_entry",""),
        p.get("stop_loss_planned",""), p.get("target_planned",""),
        p.get("hold_days_planned",""),
        p["exit_date"], p["exit_price"], p["exit_reason"],
        # exit context — blank, auditor fills
        "","","","","",
        p["hold_days_actual"], p["return_pct"], p["pnl_npr"], p["result"],
        # deltas — blank, auditor fills
        "","","","","",
        p.get("loss_cause",""), "",
    ))


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def mode_label() -> str:
    return "📄 PAPER" if PAPER_MODE else "🔴 LIVE"


def circuit_label() -> str:
    return "🔴 PAUSED" if _circuit_breaker else "🟢 ACTIVE"


def calc_unrealised(pos, current_price: float) -> dict:
    ep = Decimal(str(pos["entry_price"]))
    cp = Decimal(str(current_price))
    sh = int(pos["shares"])
    pnl = calc_pnl(ep, cp, sh)
    return pnl


def hold_days(entry_date_str: str) -> int:
    try:
        ed = date.fromisoformat(entry_date_str)
        return (date.today() - ed).days
    except Exception:
        return 0


def format_npr(val: float) -> str:
    prefix = "+" if val >= 0 else ""
    return f"{prefix}NPR {val:,.2f}"


def result_emoji(pnl: float) -> str:
    return "🟢" if pnl >= 0 else "🔴"


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = (
        f"*Nepal Stock Bot* | {mode_label()} | {circuit_label()}\n\n"
        "*Trading*\n"
        "`/buy SYMBOL SHARES PRICE` — open position\n"
        "`/sell SYMBOL PRICE` — close position\n"
        "`/cancel` — cancel pending confirmation\n\n"
        "*Status*\n"
        "`/status` — open positions + unrealised P&L\n"
        "`/pnl` — closed trades summary + win rate\n"
        "`/signal` — latest claude_analyst signals\n"
        "`/mode` — current mode\n\n"
        "*Control*\n"
        "`/pause` — activate circuit breaker\n"
        "`/resume` — deactivate circuit breaker\n\n"
        "_Natural language works too:_\n"
        "`bought nabil 10 shars at 124O`\n"
        "`sold nabil at 1290`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"{mode_label()} | {circuit_label()}\n\n"
        f"PAPER_MODE = `{'true' if PAPER_MODE else 'false'}`\n"
        f"CIRCUIT_BREAKER = `{'true' if _circuit_breaker else 'false'}`",
        parse_mode="Markdown",
    )


async def cmd_pause(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _circuit_breaker
    _circuit_breaker = True
    await update.message.reply_text(
        "🔴 *Circuit breaker ACTIVATED*\n"
        "No new positions will be opened.\n"
        "Use /resume to deactivate.",
        parse_mode="Markdown",
    )


async def cmd_resume(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global _circuit_breaker
    _circuit_breaker = False
    await update.message.reply_text(
        "🟢 *Circuit breaker DEACTIVATED*\n"
        "System is active.",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    positions = get_open_positions()
    if not positions:
        await update.message.reply_text(
            f"No open positions. | {mode_label()}"
        )
        return

    lines = [f"*Open Positions* | {mode_label()}\n"]
    for pos in positions:
        ep  = float(pos["entry_price"])
        cp  = float(pos.get("current_price") or ep)  # fallback to entry if not updated
        sh  = int(pos["shares"])
        pnl = calc_pnl(Decimal(str(ep)), Decimal(str(cp)), sh)
        days = hold_days(pos["entry_date"] or "")
        em  = result_emoji(pnl["net_pnl"])
        lines.append(
            f"{em} *{pos['symbol']}*\n"
            f"  Entry: NPR {ep:,.2f} × {sh} shares\n"
            f"  Current: NPR {cp:,.2f} ({days}d held)\n"
            f"  Unrealised: {format_npr(pnl['net_pnl'])} ({pnl['return_pct']:+.2f}%)\n"
            f"  Fees paid: NPR {pnl['buy_fees']:,.2f}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    paper_val = "true" if PAPER_MODE else "false"
    trades = fetch_all(
        """
        SELECT symbol, entry_date, exit_date, entry_price, exit_price,
               shares, pnl_npr, return_pct, result
        FROM trade_journal
        WHERE paper_mode = %s
        ORDER BY exit_date DESC
        LIMIT 20
        """,
        (paper_val,),
    )
    if not trades:
        await update.message.reply_text(
            f"No closed trades yet. | {mode_label()}"
        )
        return

    wins   = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    total  = len(trades)
    win_rate = (wins / total * 100) if total else 0
    total_pnl = sum(float(t["pnl_npr"] or 0) for t in trades)

    lines = [
        f"*P&L Summary* | {mode_label()}\n",
        f"Trades: {total} | Wins: {wins} | Losses: {losses}",
        f"Win rate: {win_rate:.1f}%",
        f"Total P&L: {format_npr(total_pnl)}\n",
        "*Recent trades:*",
    ]
    for t in trades[:10]:
        em  = "🟢" if t["result"] == "WIN" else ("🔴" if t["result"] == "LOSS" else "⚪")
        pnl = float(t["pnl_npr"] or 0)
        ret = float(t["return_pct"] or 0)
        lines.append(
            f"{em} {t['symbol']} | {t['entry_date']} → {t['exit_date']} | "
            f"{format_npr(pnl)} ({ret:+.2f}%)"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    signals = fetch_all(
        """
        SELECT symbol, date, time, action, confidence, entry_price,
               stop_loss, target, reasoning
        FROM market_log
        WHERE outcome = 'PENDING'
        ORDER BY date DESC, time DESC
        LIMIT 5
        """,
    )
    if not signals:
        await update.message.reply_text("No pending signals in market_log.")
        return

    lines = ["*Latest Signals* (from claude_analyst)\n"]
    for s in signals:
        em = "🟢" if s["action"] == "BUY" else ("🔴" if s["action"] == "AVOID" else "⚪")
        lines.append(
            f"{em} *{s['symbol']}* — {s['action']} | {s['date']} {s['time']}\n"
            f"  Confidence: {s['confidence']}%\n"
            f"  Entry: NPR {s['entry_price']} | SL: {s['stop_loss']} | Target: {s['target']}\n"
            f"  Reason: {str(s['reasoning'])[:120]}...\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ═══════════════════════════════════════════════════════════════════════════
# BUY FLOW  (with NLP + confirmation)
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    if _circuit_breaker:
        await update.message.reply_text(
            "🔴 Circuit breaker is ACTIVE. Use /resume first."
        )
        return ConversationHandler.END

    # Check position limit (max 3)
    open_pos = get_open_positions()
    if len(open_pos) >= 3:
        await update.message.reply_text(
            f"⚠️ Already at max 3 open positions.\n"
            f"Close an existing position before opening a new one.\n"
            f"Use /status to see open positions."
        )
        return ConversationHandler.END

    # Extract text after /buy
    raw = " ".join(ctx.args) if ctx.args else ""
    if not raw:
        await update.message.reply_text(
            "Usage: `/buy SYMBOL SHARES PRICE`\n"
            "Example: `/buy NABIL 10 1240`\n"
            "Or natural language: `bought nabil ten shares at 1240`",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    await update.message.reply_text("⏳ Parsing your command...")
    parsed = await ask_ai_text(f"BUY: {raw}")

    if parsed.get("error"):
        await update.message.reply_text(f"❌ {parsed['error']}\nPlease try again.")
        return ConversationHandler.END

    if parsed.get("action") != "BUY":
        await update.message.reply_text(
            "❌ That didn't look like a BUY. Use /sell for selling."
        )
        return ConversationHandler.END

    symbol = parsed["symbol"].upper()
    shares = int(parsed["shares"])
    price  = Decimal(str(parsed["price"]))
    fees   = calc_buy_fees(price, shares)
    total  = price * shares + fees

    # Check for existing open position in same symbol
    existing = get_open_position(symbol)
    if existing:
        await update.message.reply_text(
            f"⚠️ You already have an open position in *{symbol}*.\n"
            f"Close it first with `/sell {symbol} <price>`.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    # Store parsed data for confirmation step
    ctx.user_data["pending_buy"] = {
        "symbol": symbol,
        "shares": shares,
        "price":  str(price),
        "fees":   str(fees),
        "total":  str(total),
    }

    confirm_text = (
        f"*Confirm BUY* | {mode_label()}\n\n"
        f"Symbol: *{symbol}*\n"
        f"Shares: *{shares}*\n"
        f"Price:  NPR *{price:,.2f}*\n"
        f"────────────────\n"
        f"Gross:  NPR {(price * shares):,.2f}\n"
        f"Fees:   NPR {fees:,.2f}\n"
        f"*Total: NPR {total:,.2f}*\n\n"
        f"Type /yes to confirm or /no to cancel."
    )
    await update.message.reply_text(confirm_text, parse_mode="Markdown")
    return CONFIRM_BUY


async def confirm_buy_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    pending = ctx.user_data.get("pending_buy")
    if not pending:
        await update.message.reply_text("No pending BUY. Use /buy to start.")
        return ConversationHandler.END

    try:
        new_id = insert_portfolio(pending)
        await update.message.reply_text(
            f"✅ *BUY recorded* | {mode_label()}\n\n"
            f"*{pending['symbol']}* — {pending['shares']} shares @ NPR {float(pending['price']):,.2f}\n"
            f"Portfolio ID: `{new_id}`\n\n"
            f"Use /status to see your open positions.",
            parse_mode="Markdown",
        )
        log.info(f"BUY recorded: {pending} | paper={PAPER_MODE}")
    except Exception as e:
        log.error(f"DB error on BUY: {e}")
        await update.message.reply_text(f"❌ DB error: {e}")

    ctx.user_data.pop("pending_buy", None)
    return ConversationHandler.END


async def confirm_buy_no(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    ctx.user_data.pop("pending_buy", None)
    await update.message.reply_text("❌ BUY cancelled.")
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# SELL FLOW  (with NLP + confirmation)
# ═══════════════════════════════════════════════════════════════════════════

async def cmd_sell(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    raw = " ".join(ctx.args) if ctx.args else ""
    if not raw:
        await update.message.reply_text(
            "Usage: `/sell SYMBOL PRICE`\n"
            "Example: `/sell NABIL 1290`",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    await update.message.reply_text("⏳ Parsing your command...")
    parsed = await ask_ai_text(f"SELL: {raw}")

    if parsed.get("error"):
        await update.message.reply_text(f"❌ {parsed['error']}\nPlease try again.")
        return ConversationHandler.END

    symbol     = parsed["symbol"].upper()
    exit_price = Decimal(str(parsed["price"]))

    # Find open position
    pos = get_open_position(symbol)
    if not pos:
        await update.message.reply_text(
            f"❌ No open position found for *{symbol}*.\n"
            f"Use /status to see your open positions.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    ep    = Decimal(str(pos["entry_price"]))
    sh    = int(pos["shares"])
    pnl   = calc_pnl(ep, exit_price, sh)
    days  = hold_days(pos["entry_date"] or "")
    result_str = "WIN" if pnl["net_pnl"] >= 0 else "LOSS"

    ctx.user_data["pending_sell"] = {
        "portfolio_id": pos["id"],
        "symbol":       symbol,
        "entry_price":  str(ep),
        "exit_price":   str(exit_price),
        "shares":       sh,
        "pnl":          pnl,
        "hold_days":    days,
        "result":       result_str,
        "entry_date":   pos["entry_date"],
    }

    em = result_emoji(pnl["net_pnl"])
    confirm_text = (
        f"*Confirm SELL* | {mode_label()}\n\n"
        f"Symbol: *{symbol}*\n"
        f"Shares: *{sh}*\n"
        f"Entry:  NPR {ep:,.2f}\n"
        f"Exit:   NPR {exit_price:,.2f}\n"
        f"────────────────\n"
        f"Held: {days} days\n"
        f"Gross P&L: {format_npr(pnl['gross_pnl'])}\n"
        f"Total fees: NPR {pnl['total_fees']:,.2f}\n"
        f"{em} *Net P&L: {format_npr(pnl['net_pnl'])} ({pnl['return_pct']:+.2f}%)*\n"
        f"Result: *{result_str}*\n\n"
        f"Type /yes to confirm or /no to cancel."
    )
    await update.message.reply_text(confirm_text, parse_mode="Markdown")
    return CONFIRM_SELL


async def confirm_sell_yes(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    pending = ctx.user_data.get("pending_sell")
    if not pending:
        await update.message.reply_text("No pending SELL. Use /sell to start.")
        return ConversationHandler.END

    try:
        exit_price = Decimal(pending["exit_price"])
        close_portfolio(pending["portfolio_id"], exit_price, "MANUAL")

        # Write to trade_journal
        pnl = pending["pnl"]
        paper_val = "true" if PAPER_MODE else "false"
        insert_trade_journal({
            "symbol":       pending["symbol"],
            "paper_mode":   paper_val,
            "entry_date":   pending["entry_date"],
            "entry_price":  pending["entry_price"],
            "shares":       str(pending["shares"]),
            "exit_date":    nst_today(),
            "exit_price":   pending["exit_price"],
            "exit_reason":  "MANUAL",
            "hold_days_actual": str(pending["hold_days"]),
            "return_pct":   str(pnl["return_pct"]),
            "pnl_npr":      str(round(pnl["net_pnl"], 2)),
            "result":       pending["result"],
            # auditor.py will backfill all other fields from market_log context
        })

        em = result_emoji(pnl["net_pnl"])
        await update.message.reply_text(
            f"✅ *SELL recorded* | {mode_label()}\n\n"
            f"{em} *{pending['symbol']}* closed\n"
            f"Net P&L: {format_npr(pnl['net_pnl'])} ({pnl['return_pct']:+.2f}%)\n"
            f"Result: *{pending['result']}*\n\n"
            f"trade_journal updated ✓\n"
            f"auditor.py will enrich with full context at EOD.",
            parse_mode="Markdown",
        )
        log.info(f"SELL recorded: {pending['symbol']} | pnl={pnl['net_pnl']} | paper={PAPER_MODE}")

    except Exception as e:
        log.error(f"DB error on SELL: {e}")
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
    await update.message.reply_text(
        "❌ Cancelled. No changes made.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ═══════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE FALLBACK
# (catches "bought nabil ten shares at 1240" without a command prefix)
# ═══════════════════════════════════════════════════════════════════════════

async def nlp_fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    buy_keywords  = ["buy", "bought", "purchase", "kineko", "kine"]
    sell_keywords = ["sell", "sold", "becho", "becheko", "close"]

    if any(k in text for k in buy_keywords):
        ctx.args = text.split()
        return await cmd_buy(update, ctx)
    elif any(k in text for k in sell_keywords):
        ctx.args = text.split()
        return await cmd_sell(update, ctx)
    else:
        await update.message.reply_text(
            "I didn't understand that. Use /help to see available commands."
        )


# ═══════════════════════════════════════════════════════════════════════════
# PUSH ALERTS  (called externally or via scheduler)
# ═══════════════════════════════════════════════════════════════════════════

async def push_eod_summary(app: Application):
    """
    Send daily EOD summary at 3:15 PM NST.
    Call this from a scheduler (APScheduler or cron).
    """
    if CHAT_ID == 0:
        log.warning("TELEGRAM_CHAT_ID not set — cannot push EOD summary.")
        return

    positions = get_open_positions()
    paper_val = "true" if PAPER_MODE else "false"
    today_trades = fetch_all(
        "SELECT * FROM trade_journal WHERE exit_date = %s AND paper_mode = %s",
        (nst_today(), paper_val),
    )

    lines = [
        f"📊 *EOD Summary* — {nst_today()} | {mode_label()}\n",
        f"Open positions: {len(positions)}/3",
    ]

    if today_trades:
        lines.append(f"\n*Today's closed trades:*")
        for t in today_trades:
            em  = "🟢" if t["result"] == "WIN" else "🔴"
            pnl = float(t["pnl_npr"] or 0)
            lines.append(f"{em} {t['symbol']} → {format_npr(pnl)}")

    if positions:
        lines.append(f"\n*Still open:*")
        for p in positions:
            days = hold_days(p["entry_date"] or "")
            lines.append(f"• {p['symbol']} | {days}d | Entry NPR {float(p['entry_price']):,.2f}")

    await app.bot.send_message(
        chat_id=CHAT_ID,
        text="\n".join(lines),
        parse_mode="Markdown",
    )


async def push_buy_signal(app: Application, symbol: str, confidence: int,
                           entry: float, stop: float, target: float, reason: str):
    """
    Called by claude_analyst.py (or a watcher script) when a BUY signal fires.
    """
    if CHAT_ID == 0:
        return
    text = (
        f"🚨 *BUY Signal* | {mode_label()}\n\n"
        f"Symbol: *{symbol}*\n"
        f"Confidence: {confidence}%\n"
        f"Entry: NPR {entry:,.2f}\n"
        f"Stop Loss: NPR {stop:,.2f}\n"
        f"Target: NPR {target:,.2f}\n\n"
        f"_{reason[:200]}_\n\n"
        f"Use `/buy {symbol} <shares> {entry}` to log your trade."
    )
    await app.bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="Markdown")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log.info(f"Starting bot | mode={mode_label()} | circuit={circuit_label()}")

    app = Application.builder().token(TOKEN).build()

    # BUY conversation
    buy_conv = ConversationHandler(
        entry_points=[CommandHandler("buy", cmd_buy)],
        states={
            CONFIRM_BUY: [
                CommandHandler("yes", confirm_buy_yes),
                CommandHandler("no",  confirm_buy_no),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    # SELL conversation
    sell_conv = ConversationHandler(
        entry_points=[CommandHandler("sell", cmd_sell)],
        states={
            CONFIRM_SELL: [
                CommandHandler("yes", confirm_sell_yes),
                CommandHandler("no",  confirm_sell_no),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
    )

    app.add_handler(buy_conv)
    app.add_handler(sell_conv)
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pnl",    cmd_pnl))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("mode",   cmd_mode))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("start",  cmd_help))

    # Natural language fallback for plain text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, nlp_fallback))

    log.info("Bot polling started...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
