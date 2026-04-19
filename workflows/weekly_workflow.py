"""
workflows/weekly_workflow.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Sunday evening review. Runs at 5:45 PM NST (after market closes Sunday).
Called by systemd timer: nepse-weekly.timer

Sequence:
    1. learning_hub.py     → GPT reviews trade_journal + rec_tracker (if built)
    2. capital_allocator   → deep weekly wealth management analysis
    3. interest_scraper    → update FD rates (monthly, skips if already done)
    4. Telegram weekly summary

─────────────────────────────────────────────────────────────────────────────
Run:
    python -m workflows.weekly_workflow
    python -m workflows.weekly_workflow --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

from analysis.monthly_council import _is_first_sunday_of_month

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WEEKLY] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
NST = timezone(timedelta(hours=5, minutes=45))


def _step(name: str, fn, dry_run: bool):
    log.info("── %s ...", name)
    if dry_run:
        log.info("   [DRY-RUN] skipped")
        return True
    t0 = time.time()
    try:
        fn()
        log.info("   ✅ done (%.1fs)", time.time() - t0)
        return True
    except Exception as e:
        log.error("   ❌ FAILED (%.1fs): %s", time.time() - t0, e)
        return False


def _is_quarterly_review_month() -> bool:
    """True on the first Sunday of March, June, September, or December (NST)."""
    now = datetime.now(tz=NST)
    return now.month in (3, 6, 9, 12) and _is_first_sunday_of_month()


def _should_run_interest_scraper() -> bool:
    """Only run interest_scraper if last run was > 25 days ago (monthly task)."""
    try:
        from sheets import get_setting
        last = get_setting("LAST_FD_SCRAPE", "")
        if not last:
            return True
        last_dt = datetime.strptime(last[:10], "%Y-%m-%d").replace(tzinfo=NST)
        return (datetime.now(tz=NST) - last_dt).days >= 25
    except Exception:
        return True


def run(dry_run: bool = False) -> int:
    now = datetime.now(tz=NST)
    log.info("=" * 65)
    log.info("NEPSE WEEKLY WORKFLOW — %s NST", now.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Sunday review — Learning Hub + Capital Allocation")
    log.info("=" * 65)

    results = {}

    # ── Step 1: Learning hub (GPT weekly review) ──────────────────────────────
    def _learning_hub():
        from analysis.learning_hub import run as run_hub
        run_hub()
    results["learning_hub"] = _step("learning_hub (GPT review)", _learning_hub, dry_run)

    # ── Step 1b: Monthly council (first Sunday of month only) ────────────────
    if _is_first_sunday_of_month():
        def _council():
            from analysis.monthly_council import run as run_council
            run_council()
        results["monthly_council"] = _step("monthly_council", _council, dry_run)

        # ── Step 1c: Quarterly lesson weight review (Mar/Jun/Sep/Dec only) ──
        if _is_quarterly_review_month():
            def _weight_review():
                from analysis.monthly_council import _run_weight_review
                _run_weight_review(dry_run=dry_run)
            results["weight_review"] = _step(
                "lesson_weight_review (DeepSeek quarterly)", _weight_review, dry_run,
            )
        else:
            log.info("── weight_review — skipped (not quarterly month)")
    else:
        log.info("── monthly_council — skipped (not first Sunday of month)")

    # ── Step 2: Capital allocator (deep weekly) ───────────────────────────────
    def _allocator():
        from workflows.capital_allocator import run as run_alloc
        run_alloc()
    results["allocator"] = _step("capital_allocator (weekly deep)", _allocator, dry_run)

    # ── Step 3: Interest scraper (monthly, self-throttled) ────────────────────
    if _should_run_interest_scraper():
        def _interest():
            from modules.interest_scraper import run as run_interest
            run_interest()
            from sheets import update_setting
            update_setting("LAST_FD_SCRAPE", now.strftime("%Y-%m-%d"))
        results["interest"] = _step("interest_scraper (FD rates)", _interest, dry_run)
    else:
        log.info("── interest_scraper — throttled (< 25 days since last run)")

    # ── Step 4: Weekly Telegram summary ──────────────────────────────────────
    def _weekly_summary():
        _send_weekly_telegram()
    results["telegram"] = _step("weekly Telegram summary", _weekly_summary, dry_run)

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    log.info("─" * 65)
    log.info("Weekly workflow complete — %d/%d steps OK", passed, passed + failed)
    if failed:
        log.warning("Failed: %s", ", ".join(k for k, v in results.items() if not v))
    log.info("=" * 65)
    return 0


def _send_weekly_telegram():
    """Build and send a weekly performance summary to Telegram."""
    import os
    import requests
    from sheets import read_tab, get_setting

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("Telegram not configured — skipping weekly summary")
        return

    now = datetime.now(tz=NST)

    try:
        kpi_rows = read_tab("financials")
        kpis = {r.get("kpi_name", ""): r.get("current_value", "?") for r in kpi_rows}
    except Exception:
        kpis = {}

    paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
    mode_label = "📝 PAPER" if paper_mode else "💰 LIVE"

    lines = [
        f"📊 *NEPSE AI — Weekly Review*",
        f"_{now.strftime('%Y-%m-%d')} | {mode_label}_",
        "",
        "🏆 *System Performance*",
        f"Win rate:     {kpis.get('overall_win_rate_pct', '?')}%",
        f"Profit factor:{kpis.get('profit_factor', '?')}",
        f"Total trades: {kpis.get('total_trades', '?')}",
        f"Loss streak:  {kpis.get('current_loss_streak', '0')}",
        "",
        "📌 Learning Hub updated. Check lessons in DB.",
        "━" * 22,
    ]

    msg = "\n".join(lines)
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=15,
        )
        if r.status_code == 200:
            log.info("Weekly Telegram summary sent")
        else:
            log.error("Telegram send failed: HTTP %d", r.status_code)
    except Exception as e:
        log.error("Telegram send error: %s", e)


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    parser = argparse.ArgumentParser(description="NEPSE weekly workflow")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
