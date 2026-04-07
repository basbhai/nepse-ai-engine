"""
workflows/eod_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
End-of-day sequence. Runs at 3:15 PM NST after market closes.
Called by systemd timer: nepse-eod.timer

Sequence:
    1. calendar_guard          → exit if today was not a trading day
    2. recommendation_tracker  → stamp WAIT/AVOID outcomes (if built)
    3. auditor.py              → close trades, causal attribution, KPI refresh
    4. daily_context_summarizer→ collapse intraday data to one row (if built)

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
                return 1
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

    # ── Step 1: Recommendation tracker ───────────────────────────────────────
    def _rec_tracker():
        from analysis.recommendation_tracker import run as run_tracker
        run_tracker(dry_run=False)
    results["rec_tracker"] = _step(
        "recommendation_tracker", _rec_tracker, dry_run
    )

    # ── Step 2: Auditor ───────────────────────────────────────────────────────
    def _auditor():
        from analysis.auditor import run_eod
        run_eod(paper_mode=paper_mode, dry_run=False)
    results["auditor"] = _step(
        f"auditor ({'paper' if paper_mode else 'live'})", _auditor, dry_run
    )

    # ── Step 3: Daily context summarizer (if built) ───────────────────────────
    def _summarizer():
        from analysis.daily_context_summarizer import run as run_summary
        run_summary()
    results["summarizer"] = _step("daily_context_summarizer", _summarizer, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("EOD workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        log.warning("Failed steps: %s", ", ".join(k for k, v in results.items() if not v))
    log.info("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPSE EOD workflow")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, skip_guard=args.skip_guard))
