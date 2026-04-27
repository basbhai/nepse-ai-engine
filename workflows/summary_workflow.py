"""
workflows/summary_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Nightly summary sequence. Runs at 9:00 PM NST after machine wakes from
3:45 PM suspend.

Sequence:
    1. calendar_guard          → exit if today was not a trading day
    2. nepal_pulse             → fresh EOD headlines (not 3 PM stale news)
    3. daily_context_summarizer→ collapse intraday data to one clean row
    4. backup_sync             → sync updated tables to Neon
    5. sleep_scheduler         → set RTC alarm for 10 AM, suspend machine

Why separate from eod_workflow:
    EOD runs at 3:15 PM with market data fresh but news stale.
    Summary runs at 9 PM with evening news cycle complete — better
    headlines for daily_context_log which GPT reads every Sunday.

─────────────────────────────────────────────────────────────────────────────
Run:
    python -m workflows.summary_workflow
    python -m workflows.summary_workflow --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SUMMARY] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
NST = timezone(timedelta(hours=5, minutes=45))


def _step(name: str, fn, dry_run: bool, *args, **kwargs):
    log.info("── %s ...", name)
    if dry_run:
        log.info("   [DRY-RUN] skipped")
        return True
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        log.info("   ✅ done (%.1fs)", time.time() - t0)
        return True
    except Exception as e:
        log.error("   ❌ FAILED (%.1fs): %s", time.time() - t0, e)
        return False


def run(dry_run: bool = False, skip_guard: bool = False) -> int:
    now = datetime.now(tz=NST)
    log.info("=" * 65)
    log.info("NEPSE SUMMARY WORKFLOW — %s NST", now.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not skip_guard:
        try:
            from calendar_guard import is_trading_day, today_nst
            if not is_trading_day(today_nst()):
                log.info("Not a trading day — summary workflow skipped")
                return 0
        except Exception as e:
            log.error("calendar_guard failed: %s — aborting", e)
            return 2

    results = {}

    # ── Step 1: Nepal pulse (fresh 9 PM headlines) ────────────────────────────
    def _nepal():
        from modules.nepal_pulse import run as run_pulse
        run_pulse()
    results["nepal_pulse"] = _step(
        "nepal_pulse (9 PM fresh headlines)", _nepal, dry_run
    )

    # ── Step 2: Daily context summarizer ─────────────────────────────────────
    def _summarizer():
        from analysis.daily_context_summarizer import run as run_summarizer
        run_summarizer()
    results["summarizer"] = _step(
        "daily_context_summarizer", _summarizer, dry_run
    )

    # ── Step 3: Backup sync ───────────────────────────────────────────────────
    def _backup():
        from analysis.backup_sync import run as run_backup
        run_backup()
    results["backup"] = _step(
        "backup_sync (local → Neon)", _backup, dry_run
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("Summary workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        log.warning("Failed steps: %s", ", ".join(k for k, v in results.items() if not v))

    # ── Step 4: Sleep (always — even if steps failed) ─────────────────────────
    log.info("── sleep_scheduler (set RTC 10 AM, suspend) ...")
    if not dry_run:
        try:
            from setup.sleep_scheduler import _schedule_wake, _suspend_now, _next_wake_time
            next_wake = _next_wake_time()
            _schedule_wake(next_wake)
            log.info("   RTC alarm set for %s NST", next_wake.strftime("%Y-%m-%d %H:%M"))
            _suspend_now()
        except Exception as e:
            log.error("   sleep_scheduler failed: %s", e)
    else:
        log.info("   [DRY-RUN] Would set RTC alarm and suspend")

    log.info("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    parser = argparse.ArgumentParser(description="NEPSE nightly summary workflow")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, skip_guard=args.skip_guard))
