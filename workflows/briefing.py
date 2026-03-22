"""
briefing.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Morning market briefing sent to Telegram at 10:30 AM NST,
          before market opens.

What this sends (plain English):
  One Telegram message covering everything you need to know before trading:
  - Portfolio health (total value, P&L, open positions)
  - Market context (NEPSE, geo score, nepal score)
  - Capital allocation recommendation from capital_allocator.py
  - Any urgent actions (positions to clear, FD to open)
  - Key event of the day (bandh? IPO drain? Nifty crash?)
  - System confidence

Example message:
  ━━━━━━━━━━━━━━━━━━━━━━
  📊 NEPSE MORNING BRIEF
  Sunday 2026-03-16 | 10:30 AM NST
  ━━━━━━━━━━━━━━━━━━━━━━

  💼 PORTFOLIO
  Value:  NPR 1,14,200 (+14.2%)
  Stocks: NABIL +8.2% | HBL +3.1%
  Liquid: NPR 42,000

  📈 MARKET
  NEPSE: 2,742 (+0.8%) | ABOVE 200DMA
  Geo:   +3 POSITIVE | Nepal: +2 POSITIVE
  State: CAUTIOUS BULL

  🎯 TODAY'S ACTION
  Hold NABIL, HBL — both performing well
  Watch NICA — RSI 36, volume surge

  ⚡ KEY EVENT
  RBI rate cut expected — positive for NEPSE

  🤖 System Confidence: 67% | Win Rate: 64%
  ━━━━━━━━━━━━━━━━━━━━━━

─────────────────────────────────────────────────────────────────────────────

SOP — STANDARD OPERATING PROCEDURE
───────────────────────────────────
WHEN IT RUNS:
  morning_brief.yml at 10:30 AM NST (before market opens at 10:45)

HOW TO TEST:
  python briefing.py        → generate and send to Telegram
  python briefing.py print  → generate and print to console (no Telegram send)

COMMON ERRORS AND FIXES:
  "Telegram send failed"    → Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
  "No portfolio data"       → Run meroshare.py first
  "No geo data"             → Run geo_sentiment.py first
  Message too long          → Telegram limit is 4096 chars. briefing.py truncates.

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
from modules.meroshare import get_portfolio_summary

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BRIEFING] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST              = timezone(timedelta(hours=5, minutes=45))
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_URL     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

MAX_MSG_LENGTH   = 4000  # Telegram limit is 4096, keeping buffer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GATHER DATA
# ══════════════════════════════════════════════════════════════════════════════

def _get_system_kpis() -> dict:
    """Read win rate and confidence from financials table."""
    try:
        rows = read_tab("financials")
        kpis = {r.get("kpi_name", ""): r.get("current_value", "") for r in rows}
        return {
            "win_rate":   kpis.get("Win_Rate_30d", "?"),
            "drawdown":   kpis.get("Max_Drawdown_Pct", "?"),
            "loss_streak": kpis.get("Current_Loss_Streak", "0"),
        }
    except Exception:
        return {"win_rate": "?", "drawdown": "?", "loss_streak": "0"}


def _get_latest_allocation() -> dict:
    """Read last capital allocation recommendation from financial_advisor."""
    try:
        rows = read_tab("financial_advisor", limit=1)
        if rows:
            return rows[0]
    except Exception:
        pass
    return {}


def _get_latest_signals() -> list[dict]:
    """Read any pending signals from today's market_log."""
    try:
        from datetime import date
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        rows  = read_tab("market_log", limit=5)
        return [r for r in rows if r.get("date") == today and r.get("outcome") == "PENDING"]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_message() -> str:
    """
    Build the full morning briefing message.
    Assembles all data into a clean Telegram-formatted string.
    """
    nst_now      = datetime.now(tz=NST)
    portfolio    = get_portfolio_summary()
    geo          = get_latest_geo()   or {}
    nepal        = get_latest_pulse() or {}
    kpis         = _get_system_kpis()
    allocation   = _get_latest_allocation()
    market_state = get_setting("MARKET_STATE", "UNKNOWN")
    nepse_index  = ""

    # Try to get NEPSE index from market_breadth
    try:
        rows = read_tab("market_breadth", limit=1)
        if rows:
            nepse_index = rows[0].get("nepse_index", "")
            nepse_chg   = rows[0].get("nepse_change_pct", "")
            breadth_sig = rows[0].get("market_signal", "")
    except Exception:
        nepse_chg = ""
        breadth_sig = ""

    day_name = nst_now.strftime("%A")
    date_str = nst_now.strftime("%Y-%m-%d")

    sep = "━" * 22

    lines = [
        f"{sep}",
        f"📊 *NEPSE MORNING BRIEF*",
        f"{day_name} {date_str} | {nst_now.strftime('%H:%M')} NST",
        f"{sep}",
    ]

    # ── Portfolio ──
    lines.append("")
    lines.append("💼 *PORTFOLIO*")
    if portfolio and portfolio.holdings:
        lines.append(f"Value:   NPR {portfolio.total_value_npr:,.0f}  ({portfolio.total_pnl_pct:+.1f}%)")
        lines.append(f"Invested: NPR {portfolio.total_cost_npr:,.0f}")
        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
        liquid = max(0, total_capital - portfolio.total_cost_npr)
        lines.append(f"Liquid:  NPR {liquid:,.0f}")
        lines.append("")
        for h in sorted(portfolio.holdings, key=lambda x: x.pnl_pct, reverse=True):
            arrow = "📈" if h.pnl_pct >= 0 else "📉"
            lines.append(f"  {arrow} {h.symbol}: {h.pnl_pct:+.1f}%  (NPR {h.pnl_npr:+,.0f})")
    else:
        lines.append("No open positions")

    # ── Market ──
    lines.append("")
    lines.append("📈 *MARKET*")
    if nepse_index:
        nepse_str = f"NEPSE: {float(nepse_index):,.0f}"
        if nepse_chg:
            nepse_str += f" ({float(nepse_chg):+.1f}%)"
        lines.append(nepse_str)
    lines.append(f"State:  {market_state}")

    geo_score   = int(geo.get("geo_score", 0) or 0)
    nepal_score = int(nepal.get("nepal_score", 0) or 0)
    combined    = geo_score + nepal_score
    geo_emoji   = "🟢" if combined >= 2 else "🟡" if combined >= -1 else "🔴"
    lines.append(f"Geo:    {geo_emoji} {combined:+d}/10  |  VIX: {geo.get('vix','?')}  [{geo.get('vix_level','?')}]")
    lines.append(f"Nifty:  {float(geo.get('nifty_change_pct', 0) or 0):+.1f}%  |  Crude: ${geo.get('crude_price','?')}")

    # ── Key Event ──
    key_event = nepal.get("key_event") or geo.get("key_event") or ""
    bandh     = nepal.get("bandh_today", "NO")
    ipo       = nepal.get("ipo_fpo_active", "NO")

    if bandh == "YES":
        lines.append("")
        lines.append(f"🚨 *BANDH ALERT*: {nepal.get('bandh_detail','Nepal bandh today')[:80]}")
    elif ipo == "YES":
        lines.append("")
        lines.append(f"⚠️ *IPO DRAIN*: {nepal.get('ipo_fpo_detail','IPO/FPO open today')[:80]}")
    elif key_event:
        lines.append("")
        lines.append(f"⚡ *KEY EVENT*: {key_event[:100]}")

    # ── Capital Allocation ──
    if allocation:
        lines.append("")
        lines.append("🎯 *ALLOCATION ADVICE*")
        stocks  = allocation.get("capital_in_stocks_pct", "?")
        fd      = allocation.get("capital_in_fd_pct", "?")
        savings = allocation.get("capital_in_savings_pct", "?")
        lines.append(f"Stocks: {stocks}%  |  FD: {fd}%  |  Savings: {savings}%")
        outlook = allocation.get("three_month_outlook", "")
        if outlook:
            lines.append(f"Outlook: {outlook[:120]}")

    # ── System health ──
    lines.append("")
    lines.append("🤖 *SYSTEM*")
    loss_streak = int(kpis.get("loss_streak", 0) or 0)
    streak_str  = f"  ⚠️ Loss streak: {loss_streak}" if loss_streak >= 3 else ""
    lines.append(f"Win rate: {kpis['win_rate']}%  |  Drawdown: {kpis['drawdown']}%{streak_str}")

    lines.append("")
    lines.append(f"{sep}")

    message = "\n".join(lines)

    # Truncate if too long
    if len(message) > MAX_MSG_LENGTH:
        message = message[:MAX_MSG_LENGTH - 50] + "\n...(truncated)\n" + sep

    return message


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SEND TO TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    """
    Send message to Telegram.
    Uses MarkdownV2 for formatting.
    Returns True on success.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Telegram credentials not set in .env")
        return False

    try:
        payload = {
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }
        r = requests.post(TELEGRAM_URL, json=payload, timeout=15)

        if r.status_code == 200:
            log.info("✅ Morning brief sent to Telegram")
            return True
        else:
            log.error("Telegram send failed: HTTP %d — %s", r.status_code, r.text[:200])
            return False

    except Exception as exc:
        log.error("Telegram send error: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(print_only: bool = False) -> bool:
    """
    Build and send morning briefing.
    Called by morning_brief.yml at 10:30 AM NST.

    Args:
        print_only: If True, prints to console only — no Telegram send.
    Returns True on success.
    """
    nst_now = datetime.now(tz=NST)
    log.info("BRIEFING starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))

    message = build_message()

    if print_only:
        print(message)
        return True

    return send_telegram(message)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python briefing.py        → build and send to Telegram
#   python briefing.py print  → print to console only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    print_only = (arg == "print")

    success = run(print_only=print_only)
    sys.exit(0 if success else 1)
