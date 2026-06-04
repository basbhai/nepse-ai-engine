"""
workflows/summary_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Unified EOD + nightly summary sequence. Runs at 9:00 PM NST.
Called by systemd timer: nepse-summary.timer

Previously two separate workflows:
  - eod_workflow.py (3:15 PM) — RETIRED, moved to archives/
  - summary_workflow.py (9:00 PM) — this file, now absorbs EOD steps

Why merged at 9 PM (not 3:15 PM):
  - Floorsheet data is complete by 9 PM (settlement confirmed)
  - Evening news cycle complete → better headlines for daily_context_log
  - Broker flow data stable (no intraday noise)
  - One fewer wake/sleep cycle — machine sleeps at ~3:05 PM, wakes 9 PM

Sequence:
    calendar_guard              → exit if today was not a trading day (guard, not a step)
    1.  nepse_indices               → scrape latest index values (backfill 5d)
    2.  history_bootstrap           → today's EOD OHLCV → price_history
    3.  candle_detector             → detect 15 patterns on completed EOD OHLC → candle_signals
    4.  recommendation_tracker      → stamp WAIT/AVOID outcomes
    5.  auditor                     → close trades, causal attribution, KPIs
    6.  gate_miss_tracker           → stamp FALSE_BLOCK/CORRECT_BLOCK
    7.  floorsheet                  → today's full floorsheet scrape
    8.  broker_flow_scraper         → smart money accumulation/distribution
    9.  stealth_scanner             → hidden accumulation / stealth signals
    10. nepal_pulse                 → fresh 9 PM headlines
    11. daily_context_summarizer    → collapse intraday data to one clean row
    12. backup_sync                 → sync updated tables local → Neon
    13. pg_backup                   → local PostgreSQL schema + full dump
    14. sleep_scheduler             → set RTC alarm 10:45 AM, suspend

─────────────────────────────────────────────────────────────────────────────
Run:
    python -m workflows.summary_workflow
    python -m workflows.summary_workflow --dry-run
    python -m workflows.summary_workflow --skip-guard
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


def _step(name: str, fn, dry_run: bool, *args, **kwargs) -> bool:
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

    # ── Calendar guard ────────────────────────────────────────────────────────
    if not skip_guard:
        try:
            from calendar_guard import is_trading_day, today_nst
            if not is_trading_day(today_nst()):
                log.info("Not a trading day — summary workflow skipped")
                return 0
        except Exception as e:
            log.error("calendar_guard failed: %s — aborting", e)
            return 2

    # ── Detect paper/live mode ────────────────────────────────────────────────
    paper_mode = True
    try:
        from sheets import get_setting
        paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
    except Exception:
        pass
    log.info("Mode: %s", "PAPER" if paper_mode else "LIVE")

    results = {}

    # ── Step 1: NEPSE indices ─────────────────────────────────────────────────
    def _nepse_indices():
        from modules.sharehub_scraper import run as run_indices
        from datetime import date
        from_date = (datetime.now() - timedelta(days=5)).date()
        run_indices(from_date=from_date, dry_run=False)
    results["nepse_indices"] = _step("nepse_indices", _nepse_indices, dry_run)

    # ── Step 2: History bootstrap (today's EOD data) ──────────────────────────
    def _bootstrap():
        from modules.history_bootstrap import scrape_and_upsert
        scrape_and_upsert(dry_run=False)
    results["history"] = _step("history_bootstrap (today's EOD → price_history)", _bootstrap, dry_run)

    # ── Step 3: Candle detector (EOD patterns on completed OHLC) ─────────────
    def _candles():
        from modules.candle_detector import run as run_candles
        run_candles()
    results["candles"] = _step("candle_detector (15 patterns, EOD OHLC)", _candles, dry_run)

    # ── Step 4: Recommendation tracker ───────────────────────────────────────
    def _rec_tracker():
        from analysis.recommendation_tracker import run as run_tracker
        run_tracker(dry_run=False)
    results["rec_tracker"] = _step("recommendation_tracker", _rec_tracker, dry_run)

    # ── Step 5: Auditor ───────────────────────────────────────────────────────
    def _auditor():
        from analysis.auditor import run_eod_audit, run_paper_audit
        if paper_mode:
            run_paper_audit(dry_run=False)
        else:
            run_eod_audit(dry_run=False)
    results["auditor"] = _step(
        f"auditor ({'paper' if paper_mode else 'live'})", _auditor, dry_run
    )

    # ── Step 6: Gate miss tracker ─────────────────────────────────────────────
    def _gate_tracker():
        from analysis.gate_miss_tracker import run_eod
        run_eod(dry_run=False)
    results["gate_tracker"] = _step("gate_miss_tracker", _gate_tracker, dry_run)

    # ── Step 7: Floorsheet ────────────────────────────────────────────────────
    def _floorsheet():
        from modules.floorsheet_scraper import run_daily
        from datetime import date
        run_daily(target_date=date.today())
    results["floorsheet"] = _step("floorsheet (today)", _floorsheet, dry_run)

    # ── Step 8: Broker flow (smart money) ────────────────────────────────────
    def _broker_flow():
        from modules.broker_flow_scraper import run as run_broker_flow
        run_broker_flow(dry_run=False)
    results["broker_flow"] = _step(
        "broker_flow_scraper (accumulation/distribution)", _broker_flow, dry_run
    )

    # ── Step 9: Stealth accumulation scanner ─────────────────────────────────
    def _stealth_scanner():
        from modules.hidden_accum_scanner import run_scanner
        run_scanner(dry_run=False)
    results["stealth_scanner"] = _step(
        "hidden_accum_scanner (stealth signals)", _stealth_scanner, dry_run
    )

    # ── Step 10: Nepal pulse (fresh 9 PM headlines) ──────────────────────────
    def _nepal():
        from modules.nepal_pulse import run as run_pulse
        run_pulse()
    results["nepal_pulse"] = _step(
        "nepal_pulse (9 PM fresh headlines)", _nepal, dry_run
    )

    # ── Step 11: Daily context summarizer ────────────────────────────────────
    def _summarizer():
        from analysis.daily_context_summarizer import run as run_summarizer
        run_summarizer()
    results["summarizer"] = _step("daily_context_summarizer", _summarizer, dry_run)

    # ── Step 12: Backup sync ──────────────────────────────────────────────────
    def _backup():
        from analysis.backup_sync import run as run_backup
        run_backup()
    results["backup"] = _step("backup_sync (local → Neon)", _backup, dry_run)

    # ── Step 13: Local PostgreSQL backup ──────────────────────────────────────
    def _pg_backup():
        import subprocess, os
        backup_dir = os.path.expanduser("~/nepse-backup")
        subprocess.run(
            f"pg_dump postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine "
            f"--schema-only -f {backup_dir}/schema.sql && "
            f"pg_dump postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine "
            f"-f {backup_dir}/full.sql",
            shell=True, check=True
        )
    results["pg_backup"] = _step("pg_backup (schema + full → ~/nepse-backup)", _pg_backup, dry_run)

    # ── Summary log ───────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("Summary workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        log.warning("Failed steps: %s",
                    ", ".join(k for k, v in results.items() if not v))



    # ── Step 14: Sleep (always runs — even if steps failed) ───────────────────
    log.info("── sleep_scheduler (set RTC 10:45 AM, suspend) ...")
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
    parser = argparse.ArgumentParser(description="NEPSE unified EOD + summary workflow")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, skip_guard=args.skip_guard))
