#!/usr/bin/env python3
"""
setup/sleep_scheduler.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Manages machine wake/sleep schedule for NEPSE trading hours.
UPDATED: April 2026 Schedule (Mon-Fri Trading + Sunday Review)

Schedule (NST):
    Monday–Friday (Trading):
        Wake  10:00 AM  → morning workflow starts 10:30 AM
        Sleep  3:45 PM  → after EOD workflow completes
    Sunday (Weekly Review):
        Wake   5:30 PM  → Review starts 5:45 PM
        Sleep  6:15 PM  → after weekly workflow completes
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
    """Calculates the next hardware RTC wake time based on the 2026 schedule."""
    now = _now_nst()
    
    # Check current day and the next 7 days to find the next event
    for i in range(0, 8):
        check_date = (now + timedelta(days=i)).date()
        wd = (now + timedelta(days=i)).weekday()

        # 1. TRADING DAY WAKE (Mon-Fri) @ 10:00 AM (UPDATED)
        if wd in TRADING_DAYS:
            wake = datetime.combine(check_date, time(10, 00), tzinfo=NST)
            if wake > now:
                return wake

        # 2. SUNDAY REVIEW WAKE @ 5:30 PM (as requested)
        if wd == 6: 
            wake = datetime.combine(check_date, time(17, 30), tzinfo=NST)
            if wake > now:
                return wake

    return now + timedelta(days=1)

def _next_sleep_time():
    """Determines when the machine should suspend today."""
    now = _now_nst()
    wd = now.weekday()

    # Mon-Fri: Sleep after EOD workflow (3:45 PM)
    if wd in TRADING_DAYS:
        return datetime.combine(now.date(), time(15, 45), tzinfo=NST)
    
    # Sunday: Sleep after Weekly Review (6:15 PM)
    if wd == 6:
        return datetime.combine(now.date(), time(18, 15), tzinfo=NST)

    # Saturday: Sleep immediately if script is triggered
    return now

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _schedule_wake(wake_dt, dry_run=False):
    """Sets the hardware RTC alarm via rtcwake command."""
    # Convert NST to UTC timestamp for the rtcwake system utility
    utc_ts = int(wake_dt.timestamp())
    
    # -m no: set the alarm without suspending immediately
    cmd = ["sudo", "rtcwake", "-m", "no", "-t", str(utc_ts)]
    
    log.info("Scheduling RTC wake for %s...", wake_dt.strftime("%Y-%m-%d %H:%M NST"))
    
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
    # Setup Argument Parser to match project CLI standards
    parser = argparse.ArgumentParser(description="NEPSE AI Machine Power Scheduler")
    
    # --dry-run: logs the intent but does not touch hardware or suspend
    parser.add_argument("--dry-run", action="store_true", 
                        help="Perform a trial run without scheduling or suspending")
    
    # --status: prints the calculated sleep/wake times and exits
    parser.add_argument("--status", action="store_true", 
                        help="Display the calculated schedule and exit")
    
    args = parser.parse_args()

    now        = _now_nst()
    next_wake  = _next_wake_time()
    sleep_time = _next_sleep_time()

    # Handle --status CLI
    if args.status:
        print("-" * 40)
        print(f"CURRENT TIME: {now.strftime('%Y-%m-%d %H:%M:%S NST')}")
        print(f"SLEEP TODAY:  {sleep_time.strftime('%H:%M:%S NST')}")
        print(f"NEXT WAKE:    {next_wake.strftime('%Y-%m-%d %H:%M:%S NST')}")
        print("-" * 40)
        return

    # Execution Flow
    # 1. Always schedule the next wake time first (safety measure)
    if _schedule_wake(next_wake, args.dry_run):
        
        # 2. Check if the current time is at or past the scheduled sleep time
        if now >= sleep_time:
            log.info("Sleep window reached. Suspending...")
            _suspend_now(args.dry_run)
        else:
            diff_mins = int((sleep_time - now).total_seconds() / 60)
            log.info("Target sleep is at %s. Staying awake for %d more mins.", 
                     sleep_time.strftime("%H:%M"), diff_mins)

if __name__ == "__main__":
    main()