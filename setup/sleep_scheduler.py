#!/usr/bin/env python3
"""
setup/sleep_scheduler.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Manages machine wake/sleep schedule for NEPSE trading hours.

Run by systemd after each workflow completes to schedule the next wake time.
Uses rtcwake to set the hardware RTC alarm — machine wakes from suspend
automatically.

Wake/sleep schedule (NST):
    Sunday–Thursday:
        Wake  10:20 AM  → morning workflow starts 10:30 AM
        Sleep  3:45 PM  → after EOD workflow completes
    Sunday extra:
        Wake  11:45 AM (if sleeping after EOD) → weekly review 5:45 PM
        (machine stays on from EOD to weekly review on Sundays)
    Friday–Saturday: stay asleep (no market)

This script is called:
    1. At boot (to schedule first wake if needed)
    2. After EOD workflow (to schedule next morning wake)
    3. After weekly workflow (to schedule next Monday wake)

─────────────────────────────────────────────────────────────────────────────
Usage:
    python setup/sleep_scheduler.py              → schedule next wake and sleep
    python setup/sleep_scheduler.py --status     → show next wake time
    python setup/sleep_scheduler.py --dry-run    → print plan, don't execute
─────────────────────────────────────────────────────────────────────────────
IMPORTANT: Requires sudo for rtcwake.
Add to /etc/sudoers.d/nepse:
    nepse ALL=(ALL) NOPASSWD: /usr/sbin/rtcwake
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import subprocess
import sys
import logging
from datetime import datetime, timedelta, timezone, time as dtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SLEEP] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST          = timezone(timedelta(hours=5, minutes=45))
UTC          = timezone.utc
TRADING_DAYS = frozenset({6, 0, 1, 2, 3})   # Sun=6, Mon=0, ..., Thu=3

# Wake 10 min before morning workflow (10:30 AM NST)
WAKE_TIME_NST  = dtime(10, 20)
# Sleep after EOD completes (3:45 PM NST — gives 30 min buffer after 3:15 PM EOD)
SLEEP_TIME_NST = dtime(15, 45)
# Weekly review wake — only needed Sunday if you want to stay on
# We keep machine on from market open (10:20 AM) through weekly review (5:45 PM) on Sundays
# So Sunday sleep is 6:15 PM NST
SUNDAY_SLEEP_NST = dtime(18, 15)


def _now_nst() -> datetime:
    return datetime.now(tz=NST)


def _next_wake_time() -> datetime:
    """Calculate the next time the machine should wake."""
    now = _now_nst()
    d   = now.date()

    # Check today first
    for offset in range(8):  # look up to a week ahead
        candidate = d + timedelta(days=offset)
        if candidate.weekday() in TRADING_DAYS:
            wake_dt = datetime.combine(candidate, WAKE_TIME_NST, tzinfo=NST)
            if wake_dt > now + timedelta(minutes=5):
                return wake_dt

    # Fallback: next Monday
    candidate = d + timedelta(days=1)
    while candidate.weekday() not in TRADING_DAYS:
        candidate += timedelta(days=1)
    return datetime.combine(candidate, WAKE_TIME_NST, tzinfo=NST)


def _next_sleep_time() -> datetime:
    """Calculate when the machine should sleep today."""
    now = _now_nst()
    d   = now.date()
    # Sunday: sleep at 6:15 PM (after weekly review)
    if d.weekday() == 6:
        return datetime.combine(d, SUNDAY_SLEEP_NST, tzinfo=NST)
    return datetime.combine(d, SLEEP_TIME_NST, tzinfo=NST)


def _schedule_wake(wake_dt: datetime, dry_run: bool) -> bool:
    """Use rtcwake to schedule hardware wake from suspend."""
    # Convert to UTC unix timestamp for rtcwake
    wake_utc    = wake_dt.astimezone(UTC)
    wake_unix   = int(wake_utc.timestamp())
    wake_str    = wake_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    wake_nst    = wake_dt.strftime("%Y-%m-%d %H:%M NST")

    log.info("Scheduling wake: %s (%s)", wake_nst, wake_str)

    if dry_run:
        log.info("[DRY-RUN] Would run: sudo rtcwake -m no -t %d", wake_unix)
        return True

    try:
        # -m no = set RTC alarm but don't actually suspend yet
        result = subprocess.run(
            ["sudo", "rtcwake", "-m", "no", "-t", str(wake_unix)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("RTC wake alarm set for %s", wake_nst)
            return True
        else:
            log.error("rtcwake failed: %s", result.stderr.strip())
            return False
    except subprocess.TimeoutExpired:
        log.error("rtcwake timed out")
        return False
    except FileNotFoundError:
        log.error("rtcwake not found — install with: sudo apt install util-linux")
        return False
    except Exception as e:
        log.error("rtcwake error: %s", e)
        return False


def _suspend_now(dry_run: bool) -> bool:
    """Suspend the machine to RAM (S3 sleep)."""
    log.info("Suspending machine to RAM...")
    if dry_run:
        log.info("[DRY-RUN] Would run: sudo systemctl suspend")
        return True
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "suspend"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception as e:
        log.error("suspend failed: %s", e)
        return False


def run(dry_run: bool = False, status_only: bool = False) -> None:
    now        = _now_nst()
    next_wake  = _next_wake_time()
    sleep_time = _next_sleep_time()

    log.info("Current time:  %s NST", now.strftime("%Y-%m-%d %H:%M"))
    log.info("Next wake:     %s NST", next_wake.strftime("%Y-%m-%d %H:%M"))
    log.info("Sleep today:   %s NST", sleep_time.strftime("%H:%M"))

    if status_only:
        mins_to_wake = int((next_wake - now).total_seconds() / 60)
        log.info("Time to next wake: %d min", mins_to_wake)
        return

    # Set the RTC alarm for next wake
    _schedule_wake(next_wake, dry_run)

    # Sleep if we're past the sleep time for today
    if now >= sleep_time:
        log.info("Past sleep time (%s NST) — suspending now",
                 sleep_time.strftime("%H:%M"))
        _suspend_now(dry_run)
    else:
        mins_left = int((sleep_time - now).total_seconds() / 60)
        log.info("Will sleep at %s NST (%d min from now)",
                 sleep_time.strftime("%H:%M"), mins_left)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPSE wake/sleep scheduler")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--status",   action="store_true", dest="status_only")
    args = parser.parse_args()
    run(dry_run=args.dry_run, status_only=args.status_only)
