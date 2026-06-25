# -*- coding: utf-8 -*-
"""
analysis/learning_hub_audit.py -- NEPSE AI Engine
==================================================
Read-only diagnostic script: audits learning_hub lessons written between
2026-05-12 and 2026-06-25 for two categories of suspected mislabelling:

  1. SYMBOL FLAGS — lesson condition or finding references any of the known
     mislabelled symbols (BHCL, AHL, SALICO, USHEC, NIL, RLFL, MSHL, TPC,
     CHCL, KSBBL, MANDU, SHPC, TSHL).

  2. ACTION FLAGS — lesson action contains a WAIT-reinforcement keyword written
     in that window: WAIT_FOR_CONFIRMATION, REDUCE_CONFIDENCE, BLOCK_ENTRY.

No writes. No deletes. Print-only.

Run as:
    python -m analysis.learning_hub_audit
"""

import logging
import sys
from datetime import datetime

from sheets import run_raw_sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

AUDIT_START = "2026-05-12"
AUDIT_END   = "2026-06-25"

MISLABELLED_SYMBOLS = {
    "BHCL", "AHL", "SALICO", "USHEC", "NIL", "RLFL",
    "MSHL", "TPC", "CHCL", "KSBBL", "MANDU", "SHPC", "TSHL",
}

WAIT_REINFORCE_ACTIONS = {
    "WAIT_FOR_CONFIRMATION",
    "REDUCE_CONFIDENCE",
    "BLOCK_ENTRY",
}


def _flag_reason(row: dict) -> list[str]:
    """Return list of flag reasons for a lesson row. Empty = not flagged."""
    reasons = []

    condition = (row.get("condition") or "").upper()
    finding   = (row.get("finding")   or "").upper()
    action    = (row.get("action")    or "").upper()
    symbol    = (row.get("symbol")    or "").upper().strip()

    # Symbol check — direct symbol field or mentioned in condition/finding text
    for sym in MISLABELLED_SYMBOLS:
        if symbol == sym:
            reasons.append(f"SYMBOL_FIELD={sym}")
            break
        if sym in condition or sym in finding:
            reasons.append(f"SYMBOL_IN_TEXT={sym}")
            break

    # Action check
    for kw in WAIT_REINFORCE_ACTIONS:
        if kw in action:
            reasons.append(f"WAIT_REINFORCE_ACTION={kw}")
            break

    return reasons


def run_audit() -> None:
    log.info("Learning Hub Audit — window %s to %s", AUDIT_START, AUDIT_END)

    try:
        rows = run_raw_sql(
            """
            SELECT id, lesson_type, condition, finding, action,
                   confidence_level, symbol, sector, source,
                   review_week, inserted_at, active
            FROM learning_hub
            WHERE inserted_at >= %s
              AND inserted_at <= %s
            ORDER BY inserted_at ASC
            """,
            (AUDIT_START + " 00:00:00", AUDIT_END + " 23:59:59"),
        )
    except Exception as exc:
        log.error("Query failed: %s", exc)
        sys.exit(1)

    if not rows:
        print(f"\nNo lessons found in window {AUDIT_START} to {AUDIT_END}.\n")
        return

    total     = len(rows)
    flagged   = []
    flagged_ids = []

    print()
    print("=" * 90)
    print(f"LEARNING HUB AUDIT — {AUDIT_START} to {AUDIT_END}")
    print(f"Total lessons in window: {total}")
    print("=" * 90)

    for row in rows:
        reasons = _flag_reason(row)
        if not reasons:
            continue

        lid = row.get("id", "?")
        flagged.append(row)
        flagged_ids.append(lid)

        print()
        print(f"--- FLAGGED LESSON id={lid} ---")
        print(f"  lesson_type     : {row.get('lesson_type', '')}")
        print(f"  symbol          : {row.get('symbol', '')}")
        print(f"  sector          : {row.get('sector', '')}")
        print(f"  source          : {row.get('source', '')}")
        print(f"  review_week     : {row.get('review_week', '')}")
        print(f"  inserted_at     : {row.get('inserted_at', '')}")
        print(f"  active          : {row.get('active', '')}")
        print(f"  confidence_level: {row.get('confidence_level', '')}")
        print(f"  action          : {row.get('action', '')}")
        print(f"  condition       : {row.get('condition', '')}")
        print(f"  finding         : {row.get('finding', '')}")
        print(f"  FLAG_REASON     : {' | '.join(reasons)}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  Total lessons in window : {total}")
    print(f"  Total flagged           : {len(flagged)}")
    if flagged_ids:
        print(f"  Flagged IDs             : {flagged_ids}")
    else:
        print("  No lessons flagged.")
    print()


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
