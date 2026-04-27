#!/usr/bin/env python3
"""
setup/sleep_scheduler.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Manages machine wake/sleep schedule for NEPSE trading hours.
UPDATED: April 2026 — added 9 PM NST summary wake

Schedule (NST):
    Monday–Friday (Trading):
        Wake  10:00 AM  → morning workflow starts 10:30 AM
        Sleep  3:45 PM  → after EOD workflow completes
        Wake   9:00 PM  → nepal_pulse + summarizer + backup
        Sleep  9:30 PM  → after summary workflow completes
    Sunday (Weekly Review):
        Wake   5:30 PM  → Review starts 5:45 PM
        Sleep  6:15 PM  → after weekly workflow completes

Wake sequence logic (Mon-Fri):
    If current time < 3:45 PM  → next wake = 9 PM today
    If current time >= 3:45 PM and < 9:30 PM → next wake = 10 AM tomorrow
    sleep_scheduler is called by nepse-sleep.timer at 3:45 PM
    and by nepse-summary.timer at 9:30 PM — each time it sets the
    correct next RTC alarm before suspending.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import logging
import argparse
import subprocess
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")

# NEPSE 2026: Monday(0) through Friday(4)
TRADING_DAYS = frozenset({0, 1, 2, 3, 4})


def _now_nst():
    """Returns current time in Nepal Standard Time."""
    return datetime.now(NST)


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _next_wake_time():
    """
    Calculates the next hardware RTC wake time.

    Mon-Fri logic:
      - If called at 3:45 PM  → next wake = 9:00 PM same day (summary run)
      - If called at 9:30 PM  → next wake = 10:00 AM next trading day
    Sunday logic:
      - After weekly review   → next wake = 10:00 AM Monday
    """
    now = _now_nst()

    for i in range(0, 8):
        check_date = (now + timedelta(days=i)).date()
        wd = datetime.combine(check_date, time(0), tzinfo=NST).weekday()

        if wd in TRADING_DAYS:
            # 9 PM summary wake — only offer if we haven't passed it yet
            summary_wake = datetime.combine(check_date, time(21, 0), tzinfo=NST)
            if summary_wake > now:
                return summary_wake

            # 10 AM morning wake
            morning_wake = datetime.combine(check_date, time(10, 0), tzinfo=NST)
            if morning_wake > now:
                return morning_wake

        # Sunday review wake
        if wd == 6:
            wake = datetime.combine(check_date, time(17, 30), tzinfo=NST)
            if wake > now:
                return wake

    return now + timedelta(days=1)


def _next_sleep_time():
    """Determines when the machine should suspend today."""
    now = _now_nst()
    wd  = now.weekday()

    if wd in TRADING_DAYS:
        # Called at 3:45 PM → sleep now
        # Called at 9:30 PM → sleep now
        # In both cases sleep_scheduler is triggered by a timer at the right time
        eod_sleep     = datetime.combine(now.date(), time(15, 45), tzinfo=NST)
        summary_sleep = datetime.combine(now.date(), time(21, 30), tzinfo=NST)

        if now >= summary_sleep:
            return summary_sleep   # 9:30 PM sleep
        return eod_sleep           # 3:45 PM sleep

    # Sunday: sleep after weekly review
    if wd == 6:
        return datetime.combine(now.date(), time(18, 15), tzinfo=NST)

    # Saturday: sleep immediately
    return now


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _schedule_wake(wake_dt, dry_run=False):
    """Sets the hardware RTC alarm via rtcwake command."""
    utc_ts = int(wake_dt.timestamp())
    cmd    = ["sudo", "rtcwake", "-m", "no", "-t", str(utc_ts)]

    log.info("Scheduling RTC wake for %s NST...", wake_dt.strftime("%Y-%m-%d %H:%M"))

    if dry_run:
        log.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        return True

    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        log.error("Failed to set RTC wake: %s", e)
        return False


def _suspend_now(dry_run=False):
    """Issues the system command to suspend the machine to RAM."""
    cmd = ["sudo", "systemctl", "suspend"]

    if dry_run:
        log.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        return True

    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        log.error("Suspend failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEPSE AI Machine Power Scheduler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a trial run without scheduling or suspending")
    parser.add_argument("--status",  action="store_true",
                        help="Display the calculated schedule and exit")
    args = parser.parse_args()

    now        = _now_nst()
    next_wake  = _next_wake_time()
    sleep_time = _next_sleep_time()

    if args.status:
        print("-" * 45)
        print(f"CURRENT TIME: {now.strftime('%Y-%m-%d %H:%M:%S NST')}")
        print(f"SLEEP TODAY:  {sleep_time.strftime('%H:%M:%S NST')}")
        print(f"NEXT WAKE:    {next_wake.strftime('%Y-%m-%d %H:%M:%S NST')}")
        print("-" * 45)
        return

    # 1. Always schedule next wake first
    if _schedule_wake(next_wake, args.dry_run):
        # 2. Suspend if at or past the sleep window
        if now >= sleep_time:
            log.info("Sleep window reached. Suspending...")
            _suspend_now(args.dry_run)
        else:
            diff_mins = int((sleep_time - now).total_seconds() / 60)
            log.info(
                "Target sleep is at %s NST. Staying awake for %d more min.",
                sleep_time.strftime("%H:%M"), diff_mins,
            )


if __name__ == "__main__":
    main()
