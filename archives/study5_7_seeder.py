"""
study5_7_seeder.py — Pre-load Study 5 (EMA20-dip) / Study 7 (uptrend-pullback)
into learning_hub, ahead of the automated weekly review.
NEPSE AI Engine | Same pattern as archives/learning_seeder.py.

USAGE:
  python -m archives.study5_7_seeder --dry-run    # Preview, no DB writes
  python -m archives.study5_7_seeder --execute    # Insert/upsert into learning_hub
  python -m archives.study5_7_seeder --status     # Show current hub stats for these seeds

RULES:
  - ALWAYS dry-run first and review output before executing.
  - Seeds use source='research_2026' — distinguishes from the original
    source='research_paper' seed set (archives/learning_seeder.py).
  - confidence_level='MEDIUM' to start (matches learning_seeder.py's own rule:
    upgraded to HIGH after 15+ live trades confirm the pattern).
  - trade_count/win_rate use the ex-2025 (current-regime) figures, not the
    inflated full-history numbers — same discipline used throughout this
    research arc.
  - Re-run --execute is safe: upsert on (lesson_type, source, condition).
  - Does NOT touch composite scoring — EMA_PULLBACK_WEIGHT / UPTREND_PULLBACK_WEIGHT
    stay at 0 on the dashboard until deliberately raised. This only makes the
    two findings visible to claude_analyst.py's lesson-loading immediately,
    instead of waiting for the weekly automated review.

IMPORT RULE: from sheets import ... — NEVER from db import ...
"""

import sys
import argparse
import logging
from datetime import datetime

from config import NST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("study5_7_seeder")


def _nst_now() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")


try:
    from sheets import write_row, read_tab, execute_dml
except ImportError as e:
    log.error("Cannot import sheets: %s", e)
    log.error("Run from project root where sheets.py exists.")
    sys.exit(1)


def _build_seeds() -> list[dict]:
    now = _nst_now()

    return [
        {
            # Study 5 — EMA20-dip
            "lesson_type":        "SIGNAL_FILTER",
            "source":             "research_2026",
            "symbol":             "MARKET",
            "sector":             "ALL",
            "applies_to":         "ALL",
            "consumer":           "ALL",
            "condition": "ema20_dip_fired = true",
            "finding":            (
                "EMA20-dip (Study 5): price >=4.9% below its 20-day EMA while still "
                "within 8.6% of its 200-day EMA. 58.6% win rate over 6 years "
                "(n=11,339); 58.0% in the current regime excluding the 2025 "
                "political-crisis year (n=5,705 — the conservative figure quoted "
                "here). Held up worse than Study 7 during 2025 itself (37.6%). "
                "Not yet weighted into composite scoring — dashboard "
                "EMA_PULLBACK_WEIGHT is 0 by default until validated live. "
                "Factor into reasoning as a positive signal when ema20_dip_fired "
                "is true; do not treat as a hard entry trigger by itself."
            ),
            "action":             "ADD_TO_REASONING",
            "trade_count":        "5705",
            "win_count":          "3309",
            "loss_count":         "2396",
            "win_rate":           "58.0",
            "avg_return_pct":     "0",
            "avg_pnl_npr":        "0",
            "confidence_level":   "MEDIUM",
            "loss_cause_primary": "NULL",
            "geo_delta_avg":      "0",
            "nepal_delta_avg":    "0",
            "alpha_vs_nepse_avg": "0",
            "active":             "true",
            "superseded_by":      "",
            "last_validated":     now[:10],
            "validation_count":   "1",
            "trade_journal_ids":  "",
            "created_at":         now,
        },
        {
            # Study 7 — uptrend-pullback
            "lesson_type":        "SIGNAL_FILTER",
            "source":             "research_2026",
            "symbol":             "MARKET",
            "sector":             "ALL",
            "applies_to":         "ALL",
            "consumer":           "ALL",
            "condition": "uptrend_pullback_fired = true",
            "finding":            (
                "Uptrend-pullback (Study 7): price >11.87% above its 200-day EMA "
                "(confirmed uptrend) and >=5.41% below its 20-day EMA (sharp dip "
                "within it). 71.3% win rate over 6 years (n=2,739); 59.6% ex-2025 "
                "(n=995) — the strongest, most crisis-resilient finding of the "
                "research arc (stayed at 55.8% even during the 2025 political-"
                "crisis year, vs. 37.6% for EMA20-dip in the same year). "
                "Not yet weighted into composite scoring — dashboard "
                "UPTREND_PULLBACK_WEIGHT is 0 by default until validated live. "
                "When both ema20_dip_fired and uptrend_pullback_fired are true on "
                "the same candidate, this is the stronger, more selective signal — "
                "weight it as primary in reasoning, do not double-count both."
            ),
            "action":             "ADD_TO_REASONING",
            "trade_count":        "995",
            "win_count":          "593",
            "loss_count":         "402",
            "win_rate":           "59.6",
            "avg_return_pct":     "0",
            "avg_pnl_npr":        "0",
            "confidence_level":   "MEDIUM",
            "loss_cause_primary": "NULL",
            "geo_delta_avg":      "0",
            "nepal_delta_avg":    "0",
            "alpha_vs_nepse_avg": "0",
            "active":             "true",
            "superseded_by":      "",
            "last_validated":     now[:10],
            "validation_count":   "1",
            "trade_journal_ids":  "",
            "created_at":         now,
        },
    ]


def _print_seed(i: int, seed: dict) -> None:
    print(f"\n{'─' * 70}")
    print(f"  Seed {i} | {seed['lesson_type']:15} | {seed['source']:15} | {seed['confidence_level']}")
    print(f"  Condition : {seed['condition']}")
    print(f"  Action    : {seed['action']}")
    print(f"  Trades    : {seed['trade_count']} (win_rate={seed['win_rate']}%)")
    print(f"  Finding   : {seed['finding'][:140]}...")


def cmd_dry_run() -> None:
    seeds = _build_seeds()
    print("\n" + "═" * 70)
    print(f"  STUDY 5/7 SEEDER — DRY RUN ({len(seeds)} seeds)")
    print("═" * 70)
    for i, s in enumerate(seeds, 1):
        _print_seed(i, s)
    print(f"\n{'═' * 70}")
    print("  ⚠  DRY RUN — nothing written to DB.")
    print("  Run: python -m archives.study5_7_seeder --execute  to insert.\n")


def _existing_id(source: str, condition: str) -> int | None:
    """
    learning_hub has no unique constraint on (lesson_type, source, condition),
    so ON CONFLICT upsert isn't usable here. Also: sheets.upsert_row() filters
    row_data down to TABLE_COLUMNS in schema.py, which deliberately excludes
    "id" (the auto-generated primary key) — so passing conflict_columns=["id"]
    silently drops "id" from the INSERT, Postgres assigns a fresh id every
    time, ON CONFLICT never matches, and every "upsert" is actually a new
    row. (Confirmed by testing: it duplicated rows 116/117 into 118/119 on
    a second --execute run before this fix — cleaned up, not left in prod.)

    Do idempotency manually instead: look up by (source, condition), then
    UPDATE via execute_dml (raw SQL, not schema-filtered) if found, else
    INSERT via write_row.
    """
    from sheets import run_raw_sql
    rows = run_raw_sql(
        "SELECT id FROM learning_hub WHERE source = %s AND condition = %s LIMIT 1",
        (source, condition),
    )
    return int(rows[0]["id"]) if rows else None


def cmd_execute() -> None:
    seeds = _build_seeds()
    inserted, updated, errors = 0, 0, 0
    log.info("Starting seed insertion — %d seeds", len(seeds))

    for seed in seeds:
        try:
            seed["date"] = _nst_now()[:10]
            existing_id = _existing_id(seed["source"], seed["condition"])
            if existing_id:
                cols = [c for c in seed.keys() if c not in ("source", "condition")]
                set_sql = ", ".join(f'"{c}" = %s' for c in cols)
                params  = tuple(seed[c] for c in cols) + (existing_id,)
                execute_dml(f'UPDATE learning_hub SET {set_sql} WHERE id = %s', params)
                log.info("✅ Updated   | id=%s | %s", existing_id, seed["condition"])
                updated += 1
            else:
                write_row("learning_hub", seed)
                log.info("✅ Inserted  | %-20s | %s", seed["lesson_type"], seed["condition"])
                inserted += 1
        except Exception as e:
            log.error("❌ FAILED    | %-20s | %s", seed["lesson_type"], str(e))
            errors += 1

    print(f"\n{'═' * 60}")
    print("  SEED INSERTION COMPLETE")
    print(f"  ✅ Inserted : {inserted}")
    print(f"  ✅ Updated  : {updated}")
    print(f"  ❌ Errors   : {errors}")
    print(f"{'═' * 60}\n")


def cmd_status() -> None:
    try:
        rows = read_tab("learning_hub")
    except Exception as e:
        print(f"\n❌ Cannot read learning_hub: {e}\n")
        return
    ours = [r for r in rows if r.get("source") == "research_2026"]
    print(f"\n  research_2026 seeds present: {len(ours)}/2")
    for r in ours:
        print(f"    - {r.get('condition')} | active={r.get('active')} | confidence={r.get('confidence_level')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Study 5/7 findings into learning_hub")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        cmd_dry_run()
    elif args.execute:
        cmd_execute()
    elif args.status:
        cmd_status()


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    main()
