"""
modules/trading_core.py
───────────────────────
Platform-agnostic trading logic extracted from telegram_bot.py.
All DB operations, fee calculations, user management, and display
helpers live here so they can be shared across Telegram, Discord,
and any other bot surface without duplication.

Do NOT import telegram/discord-specific packages here.
"""

import os
import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional

from sheets import run_raw_sql
from db.connection import _db
from calendar_guard import is_open as market_is_open, get_status as market_status

log = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
NST              = ZoneInfo("Asia/Kathmandu")
STARTING_CAPITAL = Decimal("100000.00")
MAX_POSITIONS    = 15
CGT_RATE_LONG    = Decimal("0.075")    # 7.5% on profit, held > 365 days
CGT_RATE_SHORT   = Decimal("0.10")     # 10% on profit, held <= 365 days
SEBON_PCT        = Decimal("0.00015")  # 0.015%
DP_FEE           = Decimal("25")       # flat per trade

# ─── Runtime flags (set by CLI in telegram_bot.main / discord_bot.main) ──────
PAPER_MODE:   bool = os.environ.get("PAPER_MODE", "true").lower() == "true"
SANDBOX_MODE: bool = False   # overwritten by parse_args() in each bot's main()

# Per-session circuit breakers keyed by telegram_id (or discord user id)
_circuit_breakers: dict[int, bool] = {}


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

def cgt_rate_for_hold(hold_days_count: int) -> Decimal:
    """7.5% if held > 365 days (long-term), else 10% (short-term)."""
    return CGT_RATE_LONG if hold_days_count > 365 else CGT_RATE_SHORT

def calc_sell_fees(price: Decimal, shares: Decimal, wacc: Decimal,
                    hold_days_count: int = 0) -> dict:
    gross        = (price * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    brokerage    = _brokerage(gross)
    sebon        = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    total_fees   = brokerage + sebon + DP_FEE
    cost_basis   = (wacc * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    gross_profit = gross - cost_basis
    cgt_rate     = cgt_rate_for_hold(hold_days_count)
    cgt = (gross_profit * cgt_rate).quantize(Decimal("0.01"), ROUND_HALF_UP) \
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
        "cgt_rate":     cgt_rate,
        "cgt":          cgt,
        "net_proceeds": net_proceeds,
        "net_pnl":      net_pnl,
        "cost_basis":   cost_basis,
    }


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

def get_all_approved_ids() -> list[int]:
    """Return telegram_ids of all APPROVED users (for broadcast alerts)."""
    rows = run_raw_sql(
        "SELECT telegram_id FROM paper_users WHERE status = 'APPROVED'"
    )
    return [int(r["telegram_id"]) for r in rows]

def register_user(telegram_id: int, username: str, full_name: str) -> str:
    """
    Register a new user as PENDING.
    Returns: 'already_approved' | 'already_pending' | 'registered' | 'blocked'
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

def count_open_positions(telegram_id: int) -> int:
    rows = run_raw_sql(
        "SELECT COUNT(*) AS cnt FROM paper_portfolio WHERE telegram_id = %s AND status = 'OPEN'",
        (str(telegram_id),)
    )
    return int(rows[0]["cnt"]) if rows else 0

def get_all_open_positions(telegram_id: int) -> list[dict]:
    return run_raw_sql(
        "SELECT * FROM paper_portfolio WHERE telegram_id = %s AND status = 'OPEN' ORDER BY id",
        (str(telegram_id),)
    )

def lookup_symbols(query: str) -> list[str]:
    """
    Look up symbols in share_sectors.
    Returns exact match as single-item list, or partial ILIKE matches (up to 8).
    Returns [query.upper()] as fallback if share_sectors table is empty —
    symbol validation is best-effort, not a hard gate.
    """
    try:
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
        if partial:
            return [r["symbol"] for r in partial]
        count = run_raw_sql("SELECT COUNT(*) AS cnt FROM share_sectors")
        if count and int(count[0].get("cnt", 0)) == 0:
            log.warning("share_sectors table is empty — skipping symbol validation for %s", query)
            return [query.upper()]
        return []
    except Exception as e:
        log.warning("lookup_symbols error for %s: %s — skipping validation", query, e)
        return [query.upper()]

def _seed_capital_row(telegram_id: int):
    """Insert capital row for a newly approved user. Safe to call multiple times."""
    now   = nst_now()
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
    held_days  = hold_days(pos["first_buy_date"])
    fees       = calc_sell_fees(price, shares, wacc, held_days)
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
