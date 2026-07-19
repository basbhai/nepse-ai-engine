"""
workflows/intraday_indicator_refresh.py
────────────────────────────────────────
Standalone 1:00 PM NST re-run of the indicators pipeline, separate from the
10:15 AM morning job (nepse-morning.service).

Sequence:
  1. Archive today's current `indicators` rows into `indicators_intraday`
     (tagged with a snapshot_time label) so the 10:15 snapshot isn't lost.
  2. Recompute indicators from live prices and overwrite `indicators` via
     the normal modules.indicators.run() path (upsert on symbol+date).

filter_engine.py / gemini_filter.py / claude_analyst.py are NOT changed by
this script -- they keep reading `indicators` (now containing the 13:00
snapshot) exactly as they already do. `indicators_intraday` is history-only,
not wired into any live decision path.

CLI:
  python -m workflows.intraday_indicator_refresh
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

NST = ZoneInfo("Asia/Kathmandu")
log = logging.getLogger(__name__)


def _archive_current_snapshot(date: str, snapshot_label: str) -> int:
    """Copy today's indicators rows into indicators_intraday before they get overwritten."""
    from sheets import run_raw_sql, execute_dml

    rows = run_raw_sql(
        "SELECT * FROM indicators WHERE date = %s",
        (date,),
    )
    if not rows:
        log.warning("No existing indicators rows for %s -- nothing to archive", date)
        return 0

    cols = [
        "symbol", "date", "volume", "history_days", "rsi_14", "rsi_signal",
        "ema_20", "ema_50", "ema_200", "ema_trend", "ema_20_50_cross", "ema_50_200_cross",
        "macd_line", "macd_signal", "macd_histogram", "macd_cross",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b", "bb_signal",
        "atr_14", "atr_pct", "obv", "obv_trend", "tech_score", "tech_signal",
        "timestamp", "support_level", "resistance_level",
    ]

    archived = 0
    for row in rows:
        values = [snapshot_label] + [row.get(c) for c in cols]
        col_sql = "snapshot_time, " + ", ".join(f'"{c}"' for c in cols)
        val_sql = ", ".join("%s" for _ in values)
        ok = execute_dml(
            f'INSERT INTO indicators_intraday ({col_sql}) VALUES ({val_sql})',
            tuple(values),
        )
        if ok:
            archived += 1

    log.info("Archived %d/%d rows into indicators_intraday (snapshot_time=%s)",
              archived, len(rows), snapshot_label)
    return archived


def run() -> None:
    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    now_label = datetime.now(tz=NST).strftime("%H:%M")

    log.info("=" * 60)
    log.info("intraday_indicator_refresh — %s %s NST", today, now_label)

    # Step 1: archive whatever is currently in `indicators` for today
    archived = _archive_current_snapshot(today, now_label)
    log.info("Step 1 complete: %d rows archived", archived)

    # Step 2: recompute + overwrite `indicators` (normal daily path, upserts)
    from modules.indicators import run as run_indicators
    log.info("Step 2: recomputing indicators from live prices...")
    run_indicators()
    log.info("Step 2 complete: indicators table refreshed for %s", today)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [INTRADAY-REFRESH] %(levelname)s: %(message)s",
    )
    run()
