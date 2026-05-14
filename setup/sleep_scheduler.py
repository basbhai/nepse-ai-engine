#!/usr/bin/env python3
"""
setup/sleep_scheduler.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Manages machine wake/sleep schedule for NEPSE trading hours.

Schedule (NST) — Mon–Fri:
    Wake  10:45 AM  → morning workflow + trading loop
    Sleep  3:05 PM  → trading loop exits ~3:00 PM, nepse-sleep fires at 3:05
    Wake   9:00 PM  → summary_workflow (EOD + broker flow + summarizer)
    Sleep  9:30 PM  → summary_workflow suspends machine directly

Sunday (Weekly Review):
    Wake   5:30 PM  → Review starts 5:45 PM
    Sleep  6:15 PM  → after weekly workflow completes

Wake sequence logic (Mon–Fri):
    Called at end of summary_workflow → always sets next 10:45 AM wake.
    Called by nepse-sleep.timer at 3:05 PM → sets 9:00 PM wake today.

─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import subprocess
import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

NST          = ZoneInfo("Asia/Kathmandu")
TRADING_DAYS = frozenset({0, 1, 2, 3, 4})   # Mon–Fri

# Key times (NST)
WAKE_MORNING  = time(10, 45)   # trading starts
WAKE_SUMMARY  = time(21,  0)   # 9:00 PM — summary workflow
SLEEP_AFTER_EOD = time( 3,  5) # 3:05 PM — after market closes


def _now_nst() -> datetime:
    return datetime.now(NST)


# ─────────────────────────────────────────────────────────────────────────────
# WAKE TIME CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def _next_trading_day(from_dt: datetime) -> datetime:
    """Return the next Mon–Fri date at WAKE_MORNING from a given datetime."""
    candidate = from_dt.date() + timedelta(days=1)
    while candidate.weekday() not in TRADING_DAYS:
        candidate += timedelta(days=1)
    return datetime(
        candidate.year, candidate.month, candidate.day,
        WAKE_MORNING.hour, WAKE_MORNING.minute,
        tzinfo=NST,
    )


def _next_wake_time() -> datetime:
    """
    Determine the next RTC wake time from current NST time.

    Logic:
      If called at 3:05 PM (after market close, before 9 PM)
        → wake at 9:00 PM tonight
      If called at 9:00–9:30 PM (end of summary_workflow)
        → wake at 10:45 AM next trading day
      Any other time → conservative: next trading day 10:45 AM
    """
    now = _now_nst()
    current_time = now.time()

    # Between market close (3:05 PM) and summary start (9:00 PM)
    # → need to wake at 9 PM tonight
    if time(15, 5) <= current_time < time(21, 0):
        wake = datetime(
            now.year, now.month, now.day,
            WAKE_SUMMARY.hour, WAKE_SUMMARY.minute,
            tzinfo=NST,
        )
        log.info("Next wake: 9:00 PM tonight (%s)", wake.strftime("%Y-%m-%d %H:%M NST"))
        return wake

    # After 9 PM (summary workflow ending) → next trading day 10:45 AM
    if current_time >= time(21, 0):
        wake = _next_trading_day(now)
        log.info("Next wake: %s (next trading day)", wake.strftime("%Y-%m-%d %H:%M NST"))
        return wake

    # Before 3:05 PM (shouldn't happen normally — conservative fallback)
    wake = _next_trading_day(now)
    log.info("Next wake (fallback): %s", wake.strftime("%Y-%m-%d %H:%M NST"))
    return wake


# ─────────────────────────────────────────────────────────────────────────────
# RTC WAKE SCHEDULING
# ─────────────────────────────────────────────────────────────────────────────

def _schedule_wake(wake_dt: datetime) -> bool:
    """
    Set RTC hardware alarm using rtcwake.
    wake_dt must be timezone-aware (NST).
    Returns True on success.
    """
    # Convert NST → UTC for rtcwake
    wake_utc = wake_dt.astimezone(ZoneInfo("UTC"))
    wake_ts  = int(wake_utc.timestamp())

    log.info(
        "Setting RTC alarm: %s NST (%s UTC)",
        wake_dt.strftime("%Y-%m-%d %H:%M"),
        wake_utc.strftime("%Y-%m-%d %H:%M"),
    )

    try:
        # --mode no: set alarm without suspending (we suspend separately)
        result = subprocess.run(
            ["sudo", "rtcwake", "--mode", "no", "--time", str(wake_ts)],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            log.info("RTC alarm set successfully")
            return True
        else:
            log.error("rtcwake failed: %s", result.stderr.strip())
            return False
    except FileNotFoundError:
        log.warning("rtcwake not found — skipping (dev environment?)")
        return False
    except subprocess.TimeoutExpired:
        log.error("rtcwake timed out")
        return False
    except Exception as e:
        log.error("_schedule_wake error: %s", e)
        return False


def _suspend_now() -> None:
    """
    Suspend machine to RAM (s2ram).
    Called at end of summary_workflow after RTC alarm is set.
    """
    log.info("Suspending machine (s2ram)...")
    try:
        subprocess.run(
            ["sudo", "systemctl", "suspend"],
            timeout=10,
        )
    except Exception as e:
        log.error("Suspend failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: SLEEP AFTER MARKET CLOSE (called by nepse-sleep.timer at 3:05 PM)
# ─────────────────────────────────────────────────────────────────────────────

def sleep_after_market_close() -> None:
    """
    Called by nepse-sleep.timer at 3:05 PM NST.
    Sets RTC wake for 9:00 PM tonight, then suspends.
    """
    now  = _now_nst()
    wake = datetime(
        now.year, now.month, now.day,
        WAKE_SUMMARY.hour, WAKE_SUMMARY.minute,
        tzinfo=NST,
    )
    log.info("Market closed — sleeping until 9:00 PM NST")
    ok = _schedule_wake(wake)
    if ok:
        _suspend_now()
    else:
        log.error("RTC alarm failed — NOT suspending (safety)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEPSE Sleep Scheduler")
    parser.add_argument(
        "action",
        nargs="?",
        default="auto",
        choices=["auto", "after-market", "status"],
        help=(
            "auto         → compute next wake + set RTC (default)\n"
            "after-market → set 9 PM wake + suspend now\n"
            "status       → show next computed wake time, no action"
        ),
    )
    args = parser.parse_args()

    if args.action == "after-market":
        sleep_after_market_close()

    elif args.action == "status":
        wake = _next_wake_time()
        print(f"Next wake: {wake.strftime('%Y-%m-%d %H:%M NST')}")

    else:  # auto
        wake = _next_wake_time()
        ok   = _schedule_wake(wake)
        if ok:
            print(f"RTC alarm set for {wake.strftime('%Y-%m-%d %H:%M NST')}")
        else:
            print("RTC alarm failed — check sudo permissions")
            sys.exit(1)
