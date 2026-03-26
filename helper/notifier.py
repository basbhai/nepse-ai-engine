"""
notifier.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 5
Purpose : Telegram (primary) + Email (fallback) notification dispatcher.
          Single place for all outbound alerts.

What it sends:
  BUY signals     — full trade details (entry, stop, target, allocation)
  WAIT/AVOID      — brief context log (debug only, not sent by default)
  EOD summary     — daily P&L, win/loss count
  Error alerts    — when something breaks
  Heartbeat       — silent confirmation every trading loop (Telegram only)
  Morning brief   — assembled by briefing.py, sent here

Configuration:
  TELEGRAM_ENABLED = true/false  (settings table)
  EMAIL_ENABLED    = true/false  (settings table, optional)

All sends are fire-and-forget — failures are logged but never crash the pipeline.

─────────────────────────────────────────────────────────────────────────────
CLI:
  python notifier.py test              → send test message to Telegram
  python notifier.py heartbeat         → send silent heartbeat
  python notifier.py signal NABIL BUY  → send test BUY signal
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ── Credentials ───────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_USER       = os.getenv("EMAIL_USER", "")
EMAIL_APP_PASS   = os.getenv("EMAIL_APP_PASS", "")
EMAIL_RECEIVERS  = [e.strip() for e in os.getenv("EMAIL_RECEIVER", "").split(",") if e.strip()]

TELEGRAM_API     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
TELEGRAM_TIMEOUT = 15
MAX_TG_LENGTH    = 4000    # Telegram limit is 4096, keep buffer

# ── Paper mode prefix ─────────────────────────────────────────────────────────
PAPER_PREFIX = "[SIMULATION] "


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _is_paper_mode() -> bool:
    """Check if system is in paper trading mode."""
    try:
        from sheets import get_setting
        return get_setting("PAPER_MODE", "true").lower() == "true"
    except Exception:
        return True   # safe default


def _telegram_enabled() -> bool:
    """Check if Telegram sending is enabled in settings."""
    try:
        from sheets import get_setting
        return get_setting("TELEGRAM_ENABLED", "false").lower() == "true"
    except Exception:
        return bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(
    message: str,
    parse_mode: str = "Markdown",
    silent: bool = False,
) -> bool:
    """
    Send a message to Telegram.
    Returns True on success, False on failure.
    Never raises.

    Args:
        message:    Text to send (Markdown formatted)
        parse_mode: "Markdown" or "HTML" or "" (plain)
        silent:     If True, sends without notification sound (heartbeat)
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram: credentials not configured")
        return False

    if not _telegram_enabled():
        logger.debug("Telegram: disabled in settings (TELEGRAM_ENABLED=false)")
        return False

    # Truncate if needed
    if len(message) > MAX_TG_LENGTH:
        message = message[:MAX_TG_LENGTH - 50] + "\n...(truncated)"

    payload = {
        "chat_id":              TELEGRAM_CHAT_ID,
        "text":                 message,
        "disable_notification": silent,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(TELEGRAM_API, json=payload, timeout=TELEGRAM_TIMEOUT)
        if resp.status_code == 200:
            logger.info("Telegram sent (%d chars, silent=%s)", len(message), silent)
            return True
        else:
            # Parse mode error — retry as plain text
            if resp.status_code == 400 and "parse" in resp.text.lower():
                logger.warning("Telegram: parse mode error — retrying as plain text")
                payload.pop("parse_mode", None)
                resp2 = requests.post(TELEGRAM_API, json=payload, timeout=TELEGRAM_TIMEOUT)
                return resp2.status_code == 200
            logger.error("Telegram: HTTP %d — %s", resp.status_code, resp.text[:200])
            return False
    except requests.exceptions.Timeout:
        logger.error("Telegram: request timed out")
        return False
    except Exception as exc:
        logger.error("Telegram: unexpected error — %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EMAIL (optional fallback)
# ══════════════════════════════════════════════════════════════════════════════

def send_email(subject: str, body: str) -> bool:
    """
    Send email via Gmail SMTP.
    Returns True on success. Silent failure.
    """
    if not EMAIL_USER or not EMAIL_APP_PASS or not EMAIL_RECEIVERS:
        logger.debug("Email: not configured — skipping")
        return False

    try:
        msg             = MIMEMultipart()
        msg["From"]     = EMAIL_USER
        msg["To"]       = ", ".join(EMAIL_RECEIVERS)
        msg["Subject"]  = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_APP_PASS)
        server.sendmail(EMAIL_USER, EMAIL_RECEIVERS, msg.as_string())
        server.quit()

        logger.info("Email sent to %d recipient(s)", len(EMAIL_RECEIVERS))
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error("Email: authentication failed — check EMAIL_APP_PASS")
        return False
    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIGNAL FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def _format_buy_signal(result) -> str:
    """
    Format an AnalystResult BUY for Telegram.
    Paper mode prefix automatically added.
    """
    paper    = PAPER_PREFIX if _is_paper_mode() else ""
    nst_now  = datetime.now(tz=NST)
    sep      = "━" * 22

    # Fee calculations
    try:
        from budget import calc_true_profit, calc_breakeven
        profit  = calc_true_profit(result.entry_price, result.target, result.shares)
        net     = profit["net_profit"]
        be      = result.breakeven if result.breakeven > 0 else calc_breakeven(result.entry_price, result.shares)
    except Exception:
        net = 0.0
        be  = result.breakeven

    lines = [
        f"{sep}",
        f"{paper}📈 *BUY SIGNAL — {result.symbol}*",
        f"{nst_now.strftime('%H:%M NST')} | {result.sector} | Conf: {result.confidence}%",
        f"{sep}",
        f"🎯 Entry:      NPR {result.entry_price:,.0f}",
        f"🛑 Stop Loss:  NPR {result.stop_loss:,.0f}",
        f"✅ Target:     NPR {result.target:,.0f}",
        f"⚖️  Breakeven:  NPR {be:,.0f}",
        f"",
        f"📊 Shares:     {result.shares}",
        f"💰 Allocation: NPR {result.allocation_npr:,.0f}",
        f"📈 Net profit: NPR {net:+,.0f}",
        f"⚡ R/R Ratio:  {result.risk_reward:.1f}x",
        f"📅 Hold:       ~{result.suggested_hold} days",
        f"",
        f"Signal: {result.primary_signal} | Urgency: {result.urgency}",
        f"",
        f"💡 {result.reasoning[:200]}",
    ]

    if result.lesson_applied and result.lesson_applied not in ("", "NONE"):
        lines.append(f"📚 Lesson: {result.lesson_applied[:100]}")

    if result.candle_pattern:
        lines.append(f"🕯 Candle: {result.candle_pattern}")

    lines.append(sep)
    return "\n".join(lines)


def _format_eod_summary(
    winning: int,
    losing: int,
    pending: int,
    total_pnl: float,
    win_rate: float,
    market_comment: str = "",
) -> str:
    """Format EOD summary for Telegram."""
    paper   = PAPER_PREFIX if _is_paper_mode() else ""
    nst_now = datetime.now(tz=NST)
    sep     = "━" * 22
    emoji   = "✅" if total_pnl >= 0 else "🔴"

    lines = [
        sep,
        f"{paper}📊 *EOD SUMMARY — {nst_now.strftime('%Y-%m-%d')}*",
        sep,
        f"{emoji} P&L Today:   NPR {total_pnl:+,.0f}",
        f"✅ Wins:         {winning}",
        f"🔴 Losses:       {losing}",
        f"⏳ Pending:      {pending}",
        f"📈 Win rate:     {win_rate:.0f}%",
    ]
    if market_comment:
        lines.append(f"")
        lines.append(f"🌐 {market_comment[:150]}")
    lines.append(sep)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PUBLIC SEND FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def send_buy_signal(result) -> bool:
    """
    Send a BUY signal to Telegram (and optionally email).
    result: AnalystResult from claude_analyst.py
    """
    if result.action != "BUY":
        return False

    message = _format_buy_signal(result)
    tg_ok   = send_telegram(message, parse_mode="Markdown")

    # Email as backup for BUY signals
    if EMAIL_USER:
        email_body = message.replace("*", "").replace("━", "-")
        send_email(
            subject=f"NEPSE {'[PAPER] ' if _is_paper_mode() else ''}BUY: {result.symbol} @ NPR {result.entry_price:.0f}",
            body=email_body,
        )

    if tg_ok:
        logger.info("BUY signal sent: %s @ NPR %.0f", result.symbol, result.entry_price)
    return tg_ok


def send_avoid_signal(result, verbose: bool = False) -> bool:
    """
    Optionally send WAIT/AVOID to Telegram (debug mode only).
    Not sent in production — too noisy.
    """
    if not verbose:
        logger.debug("AVOID %s: %s", result.symbol, result.reasoning[:80])
        return True

    msg = (
        f"⏸ *{result.action} — {result.symbol}*\n"
        f"Conf: {result.confidence}% | {result.primary_signal}\n"
        f"{result.reasoning[:150]}"
    )
    return send_telegram(msg, parse_mode="Markdown", silent=True)


def send_heartbeat() -> bool:
    """
    Silent heartbeat every trading loop.
    Confirms system is running. No notification sound.
    """
    nst_now = datetime.now(tz=NST)
    msg     = f"💓 {nst_now.strftime('%H:%M')} — system running"
    return send_telegram(msg, parse_mode="", silent=True)


def send_error_alert(module: str, error: str, critical: bool = False) -> bool:
    """
    Send error alert to Telegram.
    critical=True adds email as well (system down scenario).
    """
    nst_now = datetime.now(tz=NST)
    emoji   = "🚨" if critical else "⚠️"
    msg = (
        f"{emoji} *NEPSE ENGINE ERROR*\n"
        f"Module: {module}\n"
        f"Time: {nst_now.strftime('%Y-%m-%d %H:%M NST')}\n"
        f"Error: {error[:300]}"
    )
    tg_ok = send_telegram(msg, parse_mode="Markdown")

    if critical and EMAIL_USER:
        send_email(
            subject=f"🚨 NEPSE ENGINE CRITICAL: {module}",
            body=f"Critical error in {module}:\n\n{error}",
        )

    return tg_ok


def send_eod_summary(
    winning: int = 0,
    losing: int = 0,
    pending: int = 0,
    total_pnl: float = 0,
    win_rate: float = 0,
    market_comment: str = "",
) -> bool:
    """
    Send end-of-day summary. Called by auditor.py at 3:15 PM NST.
    """
    msg = _format_eod_summary(winning, losing, pending, total_pnl, win_rate, market_comment)
    return send_telegram(msg, parse_mode="Markdown")


def send_morning_brief(message: str) -> bool:
    """
    Send the morning briefing assembled by briefing.py.
    Called at 10:30 AM NST.
    """
    return send_telegram(message, parse_mode="Markdown")


def send_circuit_breaker(reason: str) -> bool:
    """Alert when circuit breaker is triggered."""
    nst_now = datetime.now(tz=NST)
    paper   = PAPER_PREFIX if _is_paper_mode() else ""
    msg = (
        f"🚨 {paper}*CIRCUIT BREAKER TRIGGERED*\n"
        f"{nst_now.strftime('%H:%M NST')}\n"
        f"Reason: {reason}\n"
        f"Action: All new BUY signals blocked until manual review."
    )
    tg_ok = send_telegram(msg, parse_mode="Markdown")
    if EMAIL_USER:
        send_email(subject="🚨 NEPSE Circuit Breaker", body=reason)
    return tg_ok


def send_bandh_alert(detail: str) -> bool:
    """Alert when bandh is detected by nepal_pulse.py."""
    paper = PAPER_PREFIX if _is_paper_mode() else ""
    msg = (
        f"🚨 {paper}*BANDH ALERT*\n"
        f"Market liquidity severely impacted.\n"
        f"Details: {detail[:200]}\n"
        f"Action: All BUY signals auto-blocked today."
    )
    return send_telegram(msg, parse_mode="Markdown")


def send_position_update(
    symbol: str,
    event: str,
    detail: str,
) -> bool:
    """
    Send position event: trailing stop moved, target hit, stop hit.
    Called by recommendation_tracker.py.
    """
    paper = PAPER_PREFIX if _is_paper_mode() else ""
    icons = {
        "TARGET_HIT":     "🎯",
        "STOP_HIT":       "🛑",
        "TRAILING_MOVED": "📈",
        "PARTIAL_CLOSE":  "✂️",
    }
    icon = icons.get(event, "📊")
    msg = (
        f"{icon} {paper}*{event.replace('_', ' ')}: {symbol}*\n"
        f"{detail[:200]}"
    )
    return send_telegram(msg, parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BATCH SENDER (for multiple signals in one loop)
# ══════════════════════════════════════════════════════════════════════════════

def send_analysis_results(results: list) -> dict:
    """
    Process and send all AnalystResult objects from one trading loop.
    Sends BUYs immediately. Counts WAIT/AVOID silently.

    Returns:
        {buys_sent, waits_skipped, avoids_skipped, errors}
    """
    sent    = {"buys_sent": 0, "waits_skipped": 0, "avoids_skipped": 0, "errors": 0}

    for r in results:
        try:
            if r.action == "BUY":
                ok = send_buy_signal(r)
                if ok:
                    sent["buys_sent"] += 1
                else:
                    sent["errors"] += 1
            elif r.action == "WAIT":
                sent["waits_skipped"] += 1
                logger.info("WAIT %s (not sent — normal)", r.symbol)
            else:
                sent["avoids_skipped"] += 1
                logger.info("AVOID %s (not sent — normal)", r.symbol)
        except Exception as exc:
            logger.error("send_analysis_results: error on %s — %s",
                         getattr(r, "symbol", "?"), exc)
            sent["errors"] += 1

    logger.info(
        "Notifications: %d BUYs sent | %d WAIT | %d AVOID | %d errors",
        sent["buys_sent"], sent["waits_skipped"], sent["avoids_skipped"], sent["errors"]
    )
    return sent


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m helper.notifier test           → send test message
#   python -m helper.notifier heartbeat      → silent heartbeat
#   python -m helper.notifier error          → test error alert
#   python -m helper.notifier signal NABIL   → test BUY signal (fake data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [NOTIFIER] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "test"

    if cmd == "heartbeat":
        ok = send_heartbeat()
        print(f"  Heartbeat: {'✅ sent' if ok else '❌ failed'}")

    elif cmd == "error":
        ok = send_error_alert(
            module="test",
            error="This is a test error alert from notifier.py CLI",
            critical=False,
        )
        print(f"  Error alert: {'✅ sent' if ok else '❌ failed'}")

    elif cmd == "signal":
        # Create fake AnalystResult for testing
        from dataclasses import dataclass as dc, field as f

        @dc
        class FakeResult:
            symbol        : str   = args[1].upper() if len(args) > 1 else "NABIL"
            action        : str   = "BUY"
            confidence    : int   = 78
            entry_price   : float = 1240.0
            stop_loss     : float = 1190.0
            target        : float = 1380.0
            allocation_npr: float = 12000.0
            shares        : int   = 9
            breakeven     : float = 1258.0
            risk_reward   : float = 2.8
            suggested_hold: int   = 17
            reasoning     : str   = "MACD bullish cross + BB support + Non-Life Insurance sector"
            lesson_applied: str   = "MACD 17 day hold confirmed"
            primary_signal: str   = "MACD"
            sector        : str   = "non-life insurance"
            candle_pattern: str   = "Hammer"
            urgency       : str   = "NORMAL"
            geo_score     : int   = 2
            rsi_14        : float = 42.3

        fake = FakeResult()
        ok   = send_buy_signal(fake)
        print(f"  Signal: {'✅ sent' if ok else '❌ failed (is TELEGRAM_ENABLED=true?)'}")

    elif cmd == "eod":
        ok = send_eod_summary(
            winning=3, losing=1, pending=2,
            total_pnl=4500, win_rate=75,
            market_comment="NEPSE +1.2% today. Strong breadth."
        )
        print(f"  EOD summary: {'✅ sent' if ok else '❌ failed'}")

    else:  # test
        nst_now = datetime.now(tz=NST)
        paper   = PAPER_PREFIX if _is_paper_mode() else ""
        msg = (
            f"✅ {paper}*NEPSE AI ENGINE — Connection Test*\n"
            f"Time: {nst_now.strftime('%Y-%m-%d %H:%M NST')}\n"
            f"Telegram: Connected\n"
            f"Status: All systems ready\n"
            f"Mode: {'PAPER TRADING' if _is_paper_mode() else '🔴 LIVE TRADING'}"
        )
        ok = send_telegram(msg, parse_mode="Markdown")
        print(f"  Test message: {'✅ sent' if ok else '❌ failed'}")
        if not ok:
            print(f"  Check:")
            print(f"    TELEGRAM_BOT_TOKEN set: {bool(TELEGRAM_TOKEN)}")
            print(f"    TELEGRAM_CHAT_ID set:   {bool(TELEGRAM_CHAT_ID)}")
            print(f"    TELEGRAM_ENABLED=true in settings table")
