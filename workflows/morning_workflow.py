"""
workflows/morning_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Morning preparation sequence. Runs at 10:30 AM NST before market opens.
Called by systemd timer: nepse-morning.timer

Sequence (in order, each step fail-safe):
    1. calendar_guard      → exit if today is holiday
    2. meroshare.sync()    → sync real portfolio (live mode only)
    3. history_bootstrap   → append yesterday OHLCV to price_history
    4. indicators.py       → compute RSI/EMA/MACD/BB for all symbols (FROZEN)
    5. candle_detector.py  → detect 15 patterns → candle_signals
    6. geo_sentiment.run() → DXY → geo_score
    7. nepal_pulse.run()   → news + Gemini → nepal_score
    8. capital_allocator   → wealth management advice
    9. briefing.run()      → morning Telegram message

Any step failure is logged but does NOT abort the rest.
─────────────────────────────────────────────────────────────────────────────
Run:
    python -m workflows.morning_workflow
    python -m workflows.morning_workflow --dry-run
    python -m workflows.morning_workflow --skip-guard
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
    format="%(asctime)s [MORNING] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
NST = timezone(timedelta(hours=5, minutes=45))


def _step(name: str, fn, dry_run: bool, *args, **kwargs):
    """Run one morning step — logs result, never aborts on failure."""
    log.info("── %s ...", name)
    if dry_run:
        log.info("   [DRY-RUN] skipped")
        return True
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info("   ✅ done (%.1fs)", elapsed)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        log.error("   ❌ FAILED (%.1fs): %s", elapsed, e)
        return False


def run(dry_run: bool = False, skip_guard: bool = False) -> int:
    now = datetime.now(tz=NST)
    log.info("=" * 65)
    log.info("NEPSE MORNING WORKFLOW — %s NST", now.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    # ── Step 0: Calendar guard ────────────────────────────────────────────────
    if not skip_guard:
        try:
            from calendar_guard import is_trading_day, today_nst
            if not is_trading_day(today_nst()):
                log.info("Not a trading day — morning workflow skipped")
                return 1
        except Exception as e:
            log.error("calendar_guard failed: %s — aborting", e)
            return 2

    # Read mode once
    paper_mode = True
    try:
        from sheets import get_setting
        paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
    except Exception:
        pass
    log.info("Mode: %s", "PAPER" if paper_mode else "LIVE")

    results = {}

    # ── Step 1: Meroshare sync (live only) ────────────────────────────────────
    if not paper_mode:
        def _meroshare():
            from modules.meroshare import sync
            sync()
        results["meroshare"] = _step("meroshare.sync()", _meroshare, dry_run)
    else:
        log.info("── meroshare.sync() — skipped (paper mode)")

    # ── Step 2: History bootstrap ─────────────────────────────────────────────
    def _bootstrap():
        from modules.history_bootstrap import scrape_and_upsert
        scrape_and_upsert(dry_run=False)
    results["history"] = _step("history_bootstrap", _bootstrap, dry_run)

    # ── Step 3: Indicators (frozen daily) ─────────────────────────────────────
    def _indicators():
        from modules.indicators import run as run_indicators
        run_indicators()
    results["indicators"] = _step("indicators (RSI/MACD/BB/EMA)", _indicators, dry_run)

    # ── Step 4: Candle detector ───────────────────────────────────────────────
    def _candles():
        from modules.candle_detector import run as run_candles
        run_candles()
    results["candles"] = _step("candle_detector (15 patterns)", _candles, dry_run)

    # ── Step 5: Geo sentiment ─────────────────────────────────────────────────
    def _geo():
        from modules.geo_sentiment import run as run_geo
        run_geo()
    results["geo"] = _step("geo_sentiment (DXY)", _geo, dry_run)

    # ── Step 6: Nepal pulse ───────────────────────────────────────────────────
    def _nepal():
        from modules.nepal_pulse import run as run_nepal
        run_nepal()
    results["nepal"] = _step("nepal_pulse (news + Gemini)", _nepal, dry_run)

    # ── Step 7: Capital allocator ─────────────────────────────────────────────
    def _allocator():
        from workflows.capital_allocator import run as run_alloc
        run_alloc()
    results["allocator"] = _step("capital_allocator (wealth mgmt)", _allocator, dry_run)

    # ── Step 8: Morning briefing ──────────────────────────────────────────────
    def _briefing():
        from workflows.briefing import run as run_brief
        run_brief()
    results["briefing"] = _step("briefing (Telegram message)", _briefing, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("Morning workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        failed_names = [k for k, v in results.items() if not v]
        log.warning("Failed steps: %s", ", ".join(failed_names))
    log.info("Market opens at 10:45 AM NST. Trading loop starts then.")
    log.info("=" * 65)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPSE morning workflow")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-guard", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run, skip_guard=args.skip_guard))
