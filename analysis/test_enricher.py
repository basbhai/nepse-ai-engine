#!/usr/bin/env python3
"""
analysis/test_enricher.py
=========================
Smoke-test the agenda enricher against live DB data for run_month='2026-06'.
Loads items, calls _enrich_agenda_with_free_model, prints before/after.
Does NOT write back to DB.

Run:
    python analysis/test_enricher.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheets import run_raw_sql
from analysis.monthly_council import _enrich_agenda_with_free_model

TEST_RUN_MONTH = "2026-06"
ENRICHER_TAG   = "2026-06-TEST"


def main() -> None:
    rows = run_raw_sql(
        "SELECT agenda_item FROM monthly_council_agenda "
        "WHERE run_month = %s ORDER BY item_number",
        (TEST_RUN_MONTH,),
    ) or []

    items = [r["agenda_item"] for r in rows]

    if not items:
        print(f"No agenda items found for run_month={TEST_RUN_MONTH!r}. "
              "Check the DB or run the council preview first.")
        sys.exit(1)

    print(f"Loaded {len(items)} items for run_month={TEST_RUN_MONTH!r}")
    print("─" * 70)
    for i, item in enumerate(items, 1):
        print(f"[{i}] BEFORE: {item}")
    print("─" * 70)
    print(f"Running enricher (tag={ENRICHER_TAG!r}) — expect 10-12s sleep per item …")
    print()

    enriched = _enrich_agenda_with_free_model(items, ENRICHER_TAG)

    print()
    print("─" * 70)
    print("RESULTS:")
    print("─" * 70)
    for i, (orig, enr) in enumerate(zip(items, enriched), 1):
        if orig != enr:
            appended = enr[len(orig):]
            print(f"[{i}] ORIGINAL : {orig}")
            print(f"     APPENDED : {appended.strip()}")
        else:
            print(f"[{i}] NO CHANGE: {enr}")
        print()

    changed = sum(1 for o, e in zip(items, enriched) if o != e)
    print(f"Enrichment summary: {changed}/{len(items)} items enriched")


if __name__ == "__main__":
    main()
