"""
apply_fixes.py — NEPSE AI Engine
=================================
Run this ONCE to apply all 3 fixes:

  Fix 1 — Patch gemini_filter.py to flush NearMiss objects to gate_misses DB
           after every run_filter() call (upsert: same symbol+date = overwrite)

  Fix 2 — Patch gate_miss_tracker.py to persist computed gate_proposals to DB
           (upsert: same parameter_name+review_week = overwrite)

  Fix 3 — Add headlines_political / headlines_economy / headlines_nepse columns
           to daily_context_log table AND save them from daily_context_summarizer.py

Usage:
    python apply_fixes.py             # apply all fixes
    python apply_fixes.py --check     # show what would change, no writes
    python apply_fixes.py --fix 1     # apply only fix 1
    python apply_fixes.py --fix 2     # apply only fix 2
    python apply_fixes.py --fix 3     # apply only fix 3
"""

import argparse
import os
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust paths if your project layout differs
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_FILTER_PATH        = "gemini_filter.py"
GATE_MISS_TRACKER_PATH    = "analysis/gate_miss_tracker.py"
DAILY_CONTEXT_PATH        = "analysis/daily_context_summarizer.py"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _backup(path: str) -> None:
    backup_path = path + ".bak"
    content = _read(path)
    _write(backup_path, content)
    print(f"  [backup] {backup_path}")


def _check_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        print(f"          Edit the CONFIG section at top of apply_fixes.py to set correct path.")
        return False
    return True


def _patch(path: str, old: str, new: str, label: str, check_only: bool) -> bool:
    content = _read(path)
    if old not in content:
        print(f"  [SKIP] {label} — anchor not found (already patched or file changed)")
        return False
    if check_only:
        print(f"  [WOULD PATCH] {label}")
        return True
    _backup(path)
    _write(path, content.replace(old, new, 1))
    print(f"  [PATCHED] {label}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 — flush NearMiss list to gate_misses table after run_filter()
#
# WHERE:  gemini_filter.py — wherever it calls filter_engine.run_filter()
# WHAT:   After candidates = run_filter(...), call flush_near_misses_to_db()
#
# The flush function is injected at the top of gemini_filter.py and called
# immediately after run_filter() returns.
#
# Upsert key: (symbol, date) — same symbol blocked on same day = overwrite
# ─────────────────────────────────────────────────────────────────────────────

_FIX1_INJECT_ANCHOR = "from filter_engine import run_filter"

_FIX1_INJECT_CODE = '''\
from filter_engine import run_filter, get_last_near_misses


def flush_near_misses_to_db() -> None:
    """
    Write NearMiss objects captured by the last run_filter() call to gate_misses.
    Upsert on (symbol, date) — re-running same day overwrites, never duplicates.
    Called automatically after every run_filter() in this module.
    """
    from sheets import upsert_row
    misses = get_last_near_misses()
    if not misses:
        return
    written = 0
    for m in misses:
        try:
            upsert_row(
                "gate_misses",
                {
                    "symbol":                   m.symbol,
                    "sector":                   m.sector,
                    "date":                     m.date,
                    "gate_reason":              m.gate_reason,
                    "gate_category":            m.gate_category,
                    "price_at_block":           str(m.price_at_block) if m.price_at_block else None,
                    "market_state":             m.market_state,
                    "tech_score":               str(m.tech_score),
                    "conf_score":               str(m.conf_score),
                    "composite_score_would_be": str(m.composite_score_would_be),
                    "outcome":                  None,   # stamped later by gate_miss_tracker
                    "tracking_days":            "0",
                },
                conflict_columns=["symbol", "date"],
            )
            written += 1
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "flush_near_misses_to_db: failed for %s — %s", m.symbol, exc
            )
    import logging
    logging.getLogger(__name__).info(
        "flush_near_misses_to_db: wrote %d/%d near-misses to gate_misses", written, len(misses)
    )

'''

# Anchor in gemini_filter.py where run_filter() is called — patch to add flush after it
# Looks for the line that calls run_filter and adds flush call on the next line.
# Anchor must be the actual call pattern in gemini_filter.py — adjust if needed.
_FIX1_CALL_ANCHOR = "candidates = run_filter("

_FIX1_CALL_REPLACEMENT_TEMPLATE = """\
candidates = run_filter(
    # flush near-misses to DB immediately after filter run (Fix 1)
    # The actual run_filter call continues below — we wrap the result.
"""

# We use a regex-based approach for the call site since the args span multiple lines
_FIX1_FLUSH_CALL = "\n    flush_near_misses_to_db()  # Fix 1 — persist gate_misses\n"


def apply_fix1(check_only: bool) -> None:
    print("\n── Fix 1: flush NearMiss → gate_misses DB ──")

    if not _check_file(GEMINI_FILTER_PATH):
        return

    content = _read(GEMINI_FILTER_PATH)

    # Step A: inject flush function after import line
    if "def flush_near_misses_to_db" in content:
        print("  [SKIP] flush_near_misses_to_db already present")
    else:
        old_import = "from filter_engine import run_filter"
        if old_import not in content:
            print(f"  [ERROR] Could not find import anchor: '{old_import}'")
            print(f"          Check GEMINI_FILTER_PATH and import style.")
            return
        if check_only:
            print("  [WOULD PATCH] Inject flush_near_misses_to_db() function")
        else:
            _backup(GEMINI_FILTER_PATH)
            content = content.replace(old_import, _FIX1_INJECT_CODE, 1)
            print("  [PATCHED] Injected flush_near_misses_to_db()")

    # Step B: add flush call after run_filter() result assignment
    # Find pattern: variable = run_filter(   and insert flush after the closing )
    pattern = r'(candidates\s*=\s*run_filter\([^)]*\))'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("  [SKIP] run_filter() call site not found — add flush_near_misses_to_db() manually after your run_filter() call")
        if not check_only:
            _write(GEMINI_FILTER_PATH, content)
        return

    call_str = match.group(1)
    if "flush_near_misses_to_db()" in content[match.end():match.end()+80]:
        print("  [SKIP] flush call already present after run_filter()")
    else:
        if check_only:
            print("  [WOULD PATCH] Add flush_near_misses_to_db() after run_filter() call")
        else:
            new_call = call_str + "\n    flush_near_misses_to_db()  # Fix 1"
            content = content.replace(call_str, new_call, 1)
            _write(GEMINI_FILTER_PATH, content)
            print("  [PATCHED] flush_near_misses_to_db() added after run_filter()")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2 — persist gate_proposals to DB from gate_miss_tracker.py
#
# WHERE:  gate_miss_tracker.py — end of get_summary_for_gpt()
# WHAT:   After building proposal_candidates list, upsert each to gate_proposals
#
# Upsert key: (parameter_name, review_week) — same param same week = overwrite
# review_week = ISO week string e.g. "2026-W15"
# ─────────────────────────────────────────────────────────────────────────────

_FIX2_ANCHOR = '    return {\n        "total_misses":          total_misses,'

_FIX2_NEW = '''\
    # ── Fix 2: persist proposals to gate_proposals table ─────────────────────
    _write_gate_proposals(proposal_candidates)

    return {
        "total_misses":          total_misses,'''

_FIX2_FUNCTION = '''

def _write_gate_proposals(proposals: list[dict]) -> None:
    """
    Upsert gate_proposals rows.
    Conflict key: (parameter_name, review_week) — overwrites if re-run same week.
    review_week format: "2026-W15"
    """
    if not proposals:
        return
    from sheets import upsert_row
    review_week = datetime.now(NST).strftime("%G-W%V")   # ISO week
    written = 0
    for i, p in enumerate(proposals, start=1):
        try:
            upsert_row(
                "gate_proposals",
                {
                    "review_week":      review_week,
                    "proposal_number":  str(i),
                    "parameter_name":   p["parameter"],
                    "current_value":    str(p["current"]),
                    "proposed_value":   str(p["suggested"]),
                    "evidence":         p["evidence"],
                    "false_block_rate": str(p["false_block_rate"]),
                    "sample_size":      str(p["sample_size"]),
                    "status":           "PENDING",
                    "created_at":       datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S"),
                },
                conflict_columns=["parameter_name", "review_week"],
            )
            written += 1
        except Exception as exc:
            log.warning("_write_gate_proposals: failed for %s — %s", p["parameter"], exc)
    log.info("_write_gate_proposals: wrote %d/%d proposals to gate_proposals", written, len(proposals))

'''

# Inject _write_gate_proposals before get_summary_for_gpt definition
_FIX2_INJECT_ANCHOR = "\ndef get_summary_for_gpt("
_FIX2_INJECT_WITH   = _FIX2_FUNCTION + "\ndef get_summary_for_gpt("


def apply_fix2(check_only: bool) -> None:
    print("\n── Fix 2: persist gate_proposals to DB ──")

    if not _check_file(GATE_MISS_TRACKER_PATH):
        return

    content = _read(GATE_MISS_TRACKER_PATH)

    # Step A: inject _write_gate_proposals function
    if "_write_gate_proposals" in content:
        print("  [SKIP] _write_gate_proposals already present")
    else:
        if _FIX2_INJECT_ANCHOR not in content:
            print(f"  [ERROR] Anchor 'def get_summary_for_gpt(' not found")
            return
        if check_only:
            print("  [WOULD PATCH] Inject _write_gate_proposals() function")
        else:
            _backup(GATE_MISS_TRACKER_PATH)
            content = content.replace(_FIX2_INJECT_ANCHOR, _FIX2_INJECT_WITH, 1)
            print("  [PATCHED] Injected _write_gate_proposals()")

    # Step B: call it before the return statement
    if "_write_gate_proposals(proposal_candidates)" in content:
        print("  [SKIP] _write_gate_proposals() call already present")
    else:
        if _FIX2_ANCHOR not in content:
            print("  [ERROR] return-statement anchor not found in get_summary_for_gpt()")
            return
        if check_only:
            print("  [WOULD PATCH] Add _write_gate_proposals(proposal_candidates) before return")
        else:
            content = content.replace(_FIX2_ANCHOR, _FIX2_NEW, 1)
            _write(GATE_MISS_TRACKER_PATH, content)
            print("  [PATCHED] _write_gate_proposals() called before return")
            return

    if not check_only:
        _write(GATE_MISS_TRACKER_PATH, content)


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3a — Add 3 columns to daily_context_log via ALTER TABLE
# Fix 3b — Save headlines in daily_context_summarizer.py row dict
#
# Columns added:
#   headlines_political TEXT
#   headlines_economy   TEXT
#   headlines_nepse     TEXT
#
# Upsert key for daily_context_log is already "date" — no change needed there.
# ─────────────────────────────────────────────────────────────────────────────

_FIX3_ROW_ANCHOR = '        # Note: headlines_political / headlines_economy / headlines_nepse are NOT\n        # in the daily_context_log schema. They are embedded in key_events_summary.\n        # Gemini still generates them for structured extraction if schema is extended.'

_FIX3_ROW_REPLACEMENT = '''\
        # Fix 3 — headlines now saved to dedicated columns
        "headlines_political": _join_headlines("headlines_political"),
        "headlines_economy":   _join_headlines("headlines_economy"),
        "headlines_nepse":     _join_headlines("headlines_nepse"),'''


def _run_alter_table(check_only: bool) -> None:
    """Add the 3 headline columns to daily_context_log if they don't exist."""
    if check_only:
        print("  [WOULD RUN] ALTER TABLE daily_context_log ADD COLUMN headlines_political TEXT")
        print("  [WOULD RUN] ALTER TABLE daily_context_log ADD COLUMN headlines_economy TEXT")
        print("  [WOULD RUN] ALTER TABLE daily_context_log ADD COLUMN headlines_nepse TEXT")
        return
    try:
        from sheets import run_raw_sql
        for col in ("headlines_political", "headlines_economy", "headlines_nepse"):
            try:
                run_raw_sql(
                    f"ALTER TABLE daily_context_log ADD COLUMN IF NOT EXISTS {col} TEXT"
                )
                print(f"  [DB] Added column: {col}")
            except Exception as e:
                print(f"  [DB WARN] {col}: {e}")
    except ImportError:
        print("  [ERROR] Could not import sheets — run this from inside your project venv")


def apply_fix3(check_only: bool) -> None:
    print("\n── Fix 3: headlines columns in daily_context_log ──")

    # 3a: DB schema
    _run_alter_table(check_only)

    # 3b: patch summarizer row dict
    if not _check_file(DAILY_CONTEXT_PATH):
        return

    content = _read(DAILY_CONTEXT_PATH)

    if '"headlines_political"' in content:
        print("  [SKIP] headlines already in row dict")
    else:
        if _FIX3_ROW_ANCHOR not in content:
            print("  [ERROR] Row dict anchor comment not found in daily_context_summarizer.py")
            print("          Manually add these 3 keys to the 'row' dict in build_daily_context():")
            print('            "headlines_political": _join_headlines("headlines_political"),')
            print('            "headlines_economy":   _join_headlines("headlines_economy"),')
            print('            "headlines_nepse":     _join_headlines("headlines_nepse"),')
            return
        if check_only:
            print("  [WOULD PATCH] Add 3 headline keys to row dict")
        else:
            _backup(DAILY_CONTEXT_PATH)
            content = content.replace(_FIX3_ROW_ANCHOR, _FIX3_ROW_REPLACEMENT, 1)
            _write(DAILY_CONTEXT_PATH, content)
            print("  [PATCHED] 3 headline keys added to row dict")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEPSE AI — apply bug fixes")
    parser.add_argument("--check", action="store_true",
                        help="Show what would change without writing anything")
    parser.add_argument("--fix", type=int, choices=[1, 2, 3],
                        help="Apply only a specific fix (1, 2, or 3)")
    args = parser.parse_args()

    check_only = args.check
    mode = "CHECK ONLY" if check_only else "APPLYING"
    print(f"\n{'='*55}")
    print(f"  NEPSE AI — apply_fixes.py [{mode}]")
    print(f"{'='*55}")

    if args.fix:
        {1: apply_fix1, 2: apply_fix2, 3: apply_fix3}[args.fix](check_only)
    else:
        apply_fix1(check_only)
        apply_fix2(check_only)
        apply_fix3(check_only)

    print(f"\n{'='*55}")
    if check_only:
        print("  [CHECK] No files were modified.")
    else:
        print("  Done. .bak files created for every modified file.")
        print("  Re-run with --check to verify no anchors remain.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()