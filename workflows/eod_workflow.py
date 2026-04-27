"""
workflows/eod_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
End-of-day sequence. Runs at 3:15 PM NST after market closes.
Called by systemd timer: nepse-eod.timer

Sequence:
    1. calendar_guard          → exit if today was not a trading day
    2. nepse_indices           → scrape latest index values
    3. recommendation_tracker  → stamp WAIT/AVOID outcomes
    4. auditor.py              → close trades, causal attribution, KPI refresh
    5. gate_miss_tracker       → stamp FALSE_BLOCK/CORRECT_BLOCK

NOTE: daily_context_summarizer is NOT run here.
      It runs at 9:00 PM NST via nepse-summary.timer after a fresh
      nepal_pulse run — so it gets EOD headlines, not 3 PM stale ones.

--paper flag routes auditor to paper_portfolio instead of portfolio.
─────────────────────────────────────────────────────────────────────────────
Run:
    python -m workflows.eod_workflow
    python -m workflows.eod_workflow --dry-run
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
    format="%(asctime)s [EOD] %(levelname)s: %(message)s",
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
    log.info("NEPSE EOD WORKFLOW — %s NST", now.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    if not skip_guard:
        try:
            from calendar_guard import is_trading_day, today_nst
            if not is_trading_day(today_nst()):
                log.info("Not a trading day — EOD workflow skipped")
                return 0
        except Exception as e:
            log.error("calendar_guard failed: %s — aborting", e)
            return 2

    paper_mode = True
    try:
        from sheets import get_setting
        paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
    except Exception:
        pass
    log.info("Mode: %s", "PAPER" if paper_mode else "LIVE")

    results = {}

    # ── Step 0: NEPSE indices scrape ─────────────────────────────────────────
    def _nepse_indices():
        from modules.sharehub_scraper import run as run_indices
        from datetime import datetime, timedelta
        from_date = (datetime.now() - timedelta(days=5)).date()
        run_indices(from_date=from_date, dry_run=False)
    results["nepse_indices"] = _step("nepse_indices", _nepse_indices, dry_run)

    # ── Step 1: Recommendation tracker ───────────────────────────────────────
    def _rec_tracker():
        from analysis.recommendation_tracker import run as run_tracker
        run_tracker(dry_run=False)
    results["rec_tracker"] = _step(
        "recommendation_tracker", _rec_tracker, dry_run
    )

    # ── Step 2: Auditor ───────────────────────────────────────────────────────
    def _auditor():
        from analysis.auditor import run_eod_audit, run_paper_audit
        if paper_mode:
            run_paper_audit(dry_run=False)
        else:
            run_eod_audit(dry_run=False)
    results["auditor"] = _step(
        f"auditor ({'paper' if paper_mode else 'live'})", _auditor, dry_run
    )

    # ── Step 3: Gate miss tracker ─────────────────────────────────────────────
    def _gate_tracker():
        from analysis.gate_miss_tracker import run_eod
        run_eod(dry_run=False)
    results["gate_tracker"] = _step("gate_miss_tracker", _gate_tracker, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("EOD workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        log.warning("Failed steps: %s", ", ".join(k for k, v in results.items() if not v))
    log.info("Next: machine sleeps at 3:45 PM, wakes at 9:00 PM for summary.")
    log.info("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    parser = argparse.ArgumentParser(description="NEPSE EOD workflow")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, skip_guard=args.skip_guard))
