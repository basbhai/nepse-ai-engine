"""
briefing.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Morning market briefing sent to Telegram at 10:30 AM NST,
          before market opens.

Paper trading mode (PAPER_MODE=true):
  - Does NOT call meroshare.py (no real portfolio to sync)
  - Reads paper_portfolio + paper_capital keyed by telegram_id
  - Fetches all APPROVED users from paper_users
  - Sends a personalised brief to EACH user's telegram_id
  - Each user sees only their own positions and capital

Live trading mode (PAPER_MODE=false):
  - Reads real portfolio table via meroshare.get_portfolio_summary()
  - Sends one brief to TELEGRAM_CHAT_ID (admin / single user)

CLI:
  python -m workflows.briefing        → build and send to Telegram
  python -m workflows.briefing print  → print to console only (no send)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from sheets import get_setting, read_tab, get_latest_geo, get_latest_pulse
from db.connection import _db

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIEFING] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NST            = timezone(timedelta(hours=5, minutes=45))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_CHAT_ID  = os.getenv("TELEGRAM_CHAT_ID", "")   # admin / live trading recipient
TELEGRAM_URL   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
MAX_MSG_LENGTH = 4000


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SHARED DATA HELPERS (same for paper and live)
# ─────────────────────────────────────────────────────────────────────────────

def _get_system_kpis() -> dict:
    """Read win rate and confidence from financials table."""
    try:
        rows = read_tab("financials")
        kpis = {r.get("kpi_name", ""): r.get("current_value", "") for r in rows}
        return {
            "win_rate":    kpis.get("overall_win_rate_pct", "?"),
            "total_trades": kpis.get("total_trades", "0"),
            "loss_streak": kpis.get("current_loss_streak", "0"),
        }
    except Exception:
        return {"win_rate": "?", "total_trades": "0", "loss_streak": "0"}


def _get_latest_allocation() -> dict:
    """Read last capital allocation recommendation."""
    try:
        rows = read_tab("financial_advisor", limit=1)
        if rows:
            return rows[0]
    except Exception:
        pass
    return {}


def _get_latest_signals() -> list:
    """Read any pending BUY signals from today's market_log."""
    try:
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        rows  = read_tab("market_log", limit=10)
        return [r for r in rows
                if r.get("date") == today and r.get("outcome") in ("PENDING", None, "")]
    except Exception:
        return []


def _get_market_snapshot() -> dict:
    """NEPSE index + breadth from latest market_breadth row."""
    snap = {"nepse_index": "", "nepse_chg": "", "breadth_sig": ""}
    try:
        rows = read_tab("market_breadth", limit=1)
        if rows:
            snap["nepse_index"] = rows[0].get("nepse_index", "")
            snap["nepse_chg"]   = rows[0].get("nepse_change_pct", "")
            snap["breadth_sig"] = rows[0].get("market_signal", "")
    except Exception:
        pass
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PAPER TRADING DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_approved_users() -> list:
    """Return all APPROVED users from paper_users table."""
    try:
        with _db() as cur:
            cur.execute("""
                SELECT telegram_id, username, full_name
                FROM paper_users
                WHERE status = 'APPROVED'
                ORDER BY telegram_id
            """)
            return cur.fetchall()
    except Exception as e:
        log.error("Failed to fetch approved users: %s", e)
        return []


def _get_paper_portfolio(telegram_id: str) -> list:
    """Return open paper positions for one user."""
    try:
        with _db() as cur:
            cur.execute("""
                SELECT symbol, total_shares, wacc, total_cost,
                       first_buy_date, updated_at
                FROM paper_portfolio
                WHERE telegram_id = %s AND status = 'OPEN'
                ORDER BY first_buy_date ASC
            """, (str(telegram_id),))
            return cur.fetchall()
    except Exception as e:
        log.error("paper_portfolio fetch failed for %s: %s", telegram_id, e)
        return []


def _get_paper_capital(telegram_id: str) -> dict:
    """Return capital state for one user."""
    try:
        with _db() as cur:
            cur.execute("""
                SELECT current_capital, total_realised_pnl,
                       total_trades, total_wins, total_losses,
                       total_fees_paid, total_cgt_paid
                FROM paper_capital
                WHERE telegram_id = %s
            """, (str(telegram_id),))
            row = cur.fetchone()
            return dict(row) if row else {}
    except Exception as e:
        log.error("paper_capital fetch failed for %s: %s", telegram_id, e)
        return {}


def _get_paper_closed_today(telegram_id: str) -> list:
    """Return trades closed today for one user (for EOD-style summary in morning brief)."""
    try:
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        with _db() as cur:
            cur.execute("""
                SELECT symbol, net_pnl, result, exit_date
                FROM paper_portfolio
                WHERE telegram_id = %s
                  AND status = 'CLOSED'
                  AND exit_date = %s
                ORDER BY updated_at DESC
            """, (str(telegram_id), today))
            return cur.fetchall()
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MESSAGE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_market_header(nst_now: datetime) -> list:
    """Shared header block — market state, geo, nepal, NEPSE index."""
    geo          = get_latest_geo()   or {}
    nepal        = get_latest_pulse() or {}
    market_state = get_setting("MARKET_STATE", "UNKNOWN")
    snap         = _get_market_snapshot()

    sep      = "━" * 22
    day_name = nst_now.strftime("%A")
    date_str = nst_now.strftime("%Y-%m-%d")

    lines = [
        sep,
        f"📊 *NEPSE MORNING BRIEF*",
        f"{day_name} {date_str} | {nst_now.strftime('%H:%M')} NST",
        sep,
        "",
        "📈 *MARKET*",
    ]

    if snap["nepse_index"]:
        nepse_str = f"NEPSE: {float(snap['nepse_index']):,.0f}"
        if snap["nepse_chg"]:
            try:
                nepse_str += f" ({float(snap['nepse_chg']):+.1f}%)"
            except ValueError:
                pass
        lines.append(nepse_str)

    lines.append(f"State:  {market_state}")

    geo_score   = int(float(geo.get("geo_score",   0) or 0))
    nepal_score = int(float(nepal.get("nepal_score", 0) or 0))
    combined    = geo_score + nepal_score
    geo_emoji   = "🟢" if combined >= 2 else ("🟡" if combined >= -1 else "🔴")
    lines.append(
        f"Geo:    {geo_emoji} {combined:+d}/10  |  "
        f"VIX: {geo.get('vix','?')}  [{geo.get('vix_level','?')}]"
    )
    lines.append(
        f"Nifty:  {float(geo.get('nifty_change_pct', 0) or 0):+.1f}%  |  "
        f"Crude: ${geo.get('crude_price','?')}"
    )

    return lines


def _build_paper_brief_for_user(
    telegram_id: str,
    username: str,
    nst_now: datetime,
) -> str:
    """
    Build a personalised morning brief for one paper trading user.
    Shows only their own positions and capital.
    """
    starting_capital = 100_000.0   # matches telegram_bot.py STARTING_CAPITAL

    cap       = _get_paper_capital(telegram_id)
    positions = _get_paper_portfolio(telegram_id)
    kpis      = _get_system_kpis()
    signals   = _get_latest_signals()

    current_capital  = float(cap.get("current_capital",    starting_capital) or starting_capital)
    realised_pnl     = float(cap.get("total_realised_pnl", 0) or 0)
    total_trades     = int(  cap.get("total_trades",        0) or 0)
    total_wins       = int(  cap.get("total_wins",          0) or 0)
    total_losses     = int(  cap.get("total_losses",        0) or 0)
    fees_paid        = float(cap.get("total_fees_paid",     0) or 0)

    # Unrealised P&L — sum across open positions
    # (paper_portfolio doesn't store current_price; use wacc as cost basis,
    #  no live price available in briefing — show cost basis only)
    total_invested = sum(float(p.get("total_cost", 0) or 0) for p in positions)
    net_worth      = current_capital + total_invested  # liquid + invested at cost

    user_win_rate = (
        round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0.0
    )

    lines = _build_market_header(nst_now)

    # ── Paper mode banner ──────────────────────────────────────────────────────
    lines += [
        "",
        "📄 *PAPER TRADING MODE*",
        f"Hello {username or 'Trader'} 👋",
    ]

    # ── Capital state ──────────────────────────────────────────────────────────
    lines += [
        "",
        "💰 *YOUR CAPITAL*",
        f"Liquid:    NPR {current_capital:,.0f}",
        f"Invested:  NPR {total_invested:,.0f}",
        f"Net worth: NPR {net_worth:,.0f}",
    ]

    pnl_emoji = "📈" if realised_pnl >= 0 else "📉"
    lines.append(f"Realised P&L: {pnl_emoji} NPR {realised_pnl:+,.0f}")

    if fees_paid > 0:
        lines.append(f"Total fees paid: NPR {fees_paid:,.0f}")

    # ── Open positions ─────────────────────────────────────────────────────────
    lines += ["", "💼 *YOUR POSITIONS*"]
    if positions:
        for p in positions:
            shares = int(float(p.get("total_shares", 0) or 0))
            wacc   = float(p.get("wacc", 0) or 0)
            cost   = float(p.get("total_cost", 0) or 0)
            since  = p.get("first_buy_date", "?")
            lines.append(
                f"  📌 *{p.get('symbol')}* | {shares} sh @ NPR {wacc:,.2f}"
                f" | Cost: NPR {cost:,.0f} | Since: {since}"
            )
    else:
        lines.append("  No open positions")

    # ── Performance ────────────────────────────────────────────────────────────
    if total_trades > 0:
        lines += [
            "",
            "🏆 *YOUR PERFORMANCE*",
            f"Trades: {total_trades}  |  W: {total_wins}  L: {total_losses}",
            f"Win rate: {user_win_rate:.1f}%",
        ]

    # ── Today's AI signals ─────────────────────────────────────────────────────
    if signals:
        lines += ["", "🤖 *TODAY'S SIGNALS*"]
        for s in signals[:3]:
            confidence = s.get("confidence_score", s.get("conf_score", "?"))
            lines.append(
                f"  🔔 *{s.get('symbol')}* — {s.get('primary_signal', '?')} "
                f"| conf={confidence}"
            )
    else:
        lines += ["", "🤖 *SIGNALS*", "  No signals fired yet today"]

    # ── System KPIs (from trade_journal — all paper trades combined) ───────────
    if kpis.get("total_trades", "0") != "0":
        loss_streak = int(kpis.get("loss_streak", 0) or 0)
        streak_note = f"  ⚠️ Loss streak: {loss_streak} — review required!" if loss_streak >= 5 else ""
        lines += [
            "",
            "📊 *SYSTEM STATS* (all paper traders)",
            f"Trades logged: {kpis.get('total_trades', '0')} | "
            f"Win rate: {kpis.get('win_rate', '?')}%",
        ]
        if streak_note:
            lines.append(streak_note)

    # ── Circuit breaker warning ────────────────────────────────────────────────
    circuit = get_setting("CIRCUIT_BREAKER", "")
    if circuit and circuit.lower() == "true":
        lines += [
            "",
            "🚨 *CIRCUIT BREAKER ACTIVE*",
            "All new BUY signals blocked. Manual review required.",
        ]

    lines += ["", "━" * 22]

    msg = "\n".join(lines)
    # Truncate if needed — Telegram 4096 char limit
    if len(msg) > MAX_MSG_LENGTH:
        msg = msg[:MAX_MSG_LENGTH - 20] + "\n\n_[truncated]_"
    return msg


def _build_live_brief(nst_now: datetime) -> str:
    """
    Morning brief for live trading mode.
    Reads real portfolio via meroshare.get_portfolio_summary().
    """
    try:
        from modules.meroshare import get_portfolio_summary
        portfolio = get_portfolio_summary()
    except Exception as e:
        log.error("meroshare.get_portfolio_summary failed: %s", e)
        portfolio = None

    kpis       = _get_system_kpis()
    allocation = _get_latest_allocation()
    signals    = _get_latest_signals()

    lines = _build_market_header(nst_now)

    # ── Portfolio ──────────────────────────────────────────────────────────────
    lines += ["", "💼 *PORTFOLIO*"]
    if portfolio and portfolio.holdings:
        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
        liquid        = max(0, total_capital - portfolio.total_cost_npr)
        lines.append(
            f"Value:   NPR {portfolio.total_value_npr:,.0f}  "
            f"({portfolio.total_pnl_pct:+.1f}%)"
        )
        lines.append(f"Invested: NPR {portfolio.total_cost_npr:,.0f}")
        lines.append(f"Liquid:  NPR {liquid:,.0f}")
        lines.append("")
        for h in sorted(portfolio.holdings, key=lambda x: x.pnl_pct, reverse=True):
            arrow = "📈" if h.pnl_pct >= 0 else "📉"
            lines.append(
                f"  {arrow} {h.symbol}: {h.pnl_pct:+.1f}%  (NPR {h.pnl_npr:+,.0f})"
            )
    else:
        lines.append("No open positions")

    # ── Capital allocation ────────────────────────────────────────────────────
    if allocation:
        alloc  = allocation.get("recommendation_json", "")
        urgent = allocation.get("urgent_actions", "")
        if urgent and urgent not in ("No urgent actions", ""):
            lines += ["", f"⚡ *ACTION*", f"{urgent[:200]}"]

    # ── Signals ────────────────────────────────────────────────────────────────
    if signals:
        lines += ["", "🤖 *TODAY'S SIGNALS*"]
        for s in signals[:3]:
            lines.append(
                f"  🔔 *{s.get('symbol')}* — {s.get('primary_signal', '?')}"
            )

    # ── System KPIs ────────────────────────────────────────────────────────────
    lines += [
        "",
        f"🤖 Win rate: {kpis.get('win_rate', '?')}% | "
        f"Trades: {kpis.get('total_trades', '0')}",
        "━" * 22,
    ]

    msg = "\n".join(lines)
    if len(msg) > MAX_MSG_LENGTH:
        msg = msg[:MAX_MSG_LENGTH - 20] + "\n\n_[truncated]_"
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TELEGRAM SENDER
# ─────────────────────────────────────────────────────────────────────────────

def _send_telegram(chat_id: str, message: str) -> bool:
    """Send message to a specific chat_id."""
    if not TELEGRAM_TOKEN or not chat_id:
        log.error("Telegram credentials missing — token=%s chat_id=%s",
                  bool(TELEGRAM_TOKEN), bool(chat_id))
        return False
    try:
        payload = {
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "Markdown",
        }
        r = requests.post(TELEGRAM_URL, json=payload, timeout=15)
        if r.status_code == 200:
            log.info("Brief sent to chat_id=%s", chat_id)
            return True
        else:
            log.error("Telegram send failed for %s: HTTP %d — %s",
                      chat_id, r.status_code, r.text[:200])
            return False
    except Exception as exc:
        log.error("Telegram send error for %s: %s", chat_id, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(print_only: bool = False) -> bool:
    """
    Build and send morning briefing.

    PAPER MODE:
      Fetches all approved users → sends personalised brief to each user's telegram_id.

    LIVE MODE:
      Builds one brief from real portfolio → sends to TELEGRAM_CHAT_ID.

    Args:
        print_only: If True, prints to console only (no Telegram send).
    Returns True if at least one message sent successfully.
    """
    nst_now   = datetime.now(tz=NST)
    paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"

    log.info("BRIEFING starting — %s NST | paper=%s",
             nst_now.strftime("%Y-%m-%d %H:%M"), paper_mode)

    if paper_mode:
        # ── Paper trading: personalised brief per user ─────────────────────────
        users = _get_approved_users()
        if not users:
            log.warning("No approved paper trading users found — nothing to send.")
            return False

        log.info("Sending paper briefs to %d approved user(s).", len(users))
        results = []
        for user in users:
            telegram_id = str(user.get("telegram_id", ""))
            username    = user.get("username") or user.get("full_name") or telegram_id

            if not telegram_id:
                log.warning("Skipping user with no telegram_id: %s", user)
                continue

            message = _build_paper_brief_for_user(telegram_id, username, nst_now)

            if print_only:
                print(f"\n{'='*60}")
                print(f"  Brief for: {username} ({telegram_id})")
                print(f"{'='*60}")
                print(message)
                results.append(True)
            else:
                ok = _send_telegram(telegram_id, message)
                results.append(ok)

        return any(results)

    else:
        # ── Live trading: single brief to admin ────────────────────────────────
        message = _build_live_brief(nst_now)

        if print_only:
            print(message)
            return True

        return _send_telegram(ADMIN_CHAT_ID, message)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
#   python -m workflows.briefing        → build and send to Telegram
#   python -m workflows.briefing print  → print to console only
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    arg        = sys.argv[1] if len(sys.argv) > 1 else ""
    print_only = (arg == "print")

    success = run(print_only=print_only)
    sys.exit(0 if success else 1)