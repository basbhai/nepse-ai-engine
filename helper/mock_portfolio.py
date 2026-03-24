"""
mock_portfolio.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Test Tool
Purpose : Switch between two pipeline test scenarios without touching
          real data permanently.

  SCENARIO A — FRESH START
    Wipes portfolio table → 0 positions, full capital liquid.
    Expected: Claude produces BUY signals.

  SCENARIO B — CURRENT STATE
    Reads your REAL 6 positions from Neon as-is (does NOT overwrite them).
    Only updates CAPITAL_TOTAL_NPR if you pass it.
    Expected: Claude AVOIDs new buys, capital_allocator advises on holdings.

Usage:
  python mock_portfolio.py status              → show portfolio + settings
  python mock_portfolio.py fresh               → Scenario A (wipe + fresh)
  python mock_portfolio.py fresh 150000        → Scenario A with custom capital
  python mock_portfolio.py current             → Scenario B (read real Neon data)
  python mock_portfolio.py current 150000      → Scenario B + set capital to 150k
  python mock_portfolio.py restore             → restore from saved snapshot
  python mock_portfolio.py reset               → wipe portfolio, restore defaults
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MOCK] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST           = timezone(timedelta(hours=5, minutes=45))
SNAPSHOT_FILE = "mock_portfolio_snapshot.json"


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _set(key: str, value: str):
    from sheets import update_setting
    update_setting(key, value, set_by="mock_portfolio")
    log.info("  SET %s = %s", key, value)


def _get(key: str, default: str = "") -> str:
    from sheets import get_setting
    return get_setting(key, default)


# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT — save / restore
# ══════════════════════════════════════════════════════════════════════════════

def save_snapshot():
    """Save current portfolio + settings to a local JSON file."""
    from sheets import read_tab
    rows = read_tab("portfolio")
    snapshot = {
        "timestamp":         datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S NST"),
        "portfolio":         rows,
        "CAPITAL_TOTAL_NPR": _get("CAPITAL_TOTAL_NPR", "100000"),
        "PAPER_MODE":        _get("PAPER_MODE", "true"),
        "MARKET_STATE":      _get("MARKET_STATE", "SIDEWAYS"),
    }
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info("Snapshot saved -> %s (%d portfolio rows)", SNAPSHOT_FILE, len(rows))
    return snapshot


def restore_snapshot():
    """Restore portfolio + settings from saved snapshot."""
    try:
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except FileNotFoundError:
        print(f"\n  No snapshot found at {SNAPSHOT_FILE}")
        print(f"  Run 'python mock_portfolio.py current' first to create one.\n")
        return

    from sheets import run_raw_sql, upsert_row

    run_raw_sql("DELETE FROM portfolio")
    inserted = 0
    for row in snap.get("portfolio", []):
        clean = {k: v for k, v in row.items()
                 if k not in ("id", "inserted_at") and v is not None and str(v).strip() != ""}
        if clean.get("symbol"):
            upsert_row("portfolio", clean, conflict_columns=["symbol"])
            inserted += 1

    _set("CAPITAL_TOTAL_NPR", snap.get("CAPITAL_TOTAL_NPR", "100000"))
    _set("PAPER_MODE",        snap.get("PAPER_MODE",        "true"))
    _set("MARKET_STATE",      snap.get("MARKET_STATE",      "SIDEWAYS"))

    print(f"\n  Restored snapshot from {snap.get('timestamp')}")
    print(f"  {inserted} portfolio rows restored")
    print(f"  Capital: NPR {float(snap.get('CAPITAL_TOTAL_NPR', 100000)):,.0f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STATUS
# ══════════════════════════════════════════════════════════════════════════════

def show_status():
    from sheets import read_tab

    rows     = read_tab("portfolio")
    open_pos = [r for r in rows if str(r.get("status", "")).upper() == "OPEN"]
    closed   = [r for r in rows if str(r.get("status", "")).upper() not in ("OPEN", "")]

    capital     = float(_get("CAPITAL_TOTAL_NPR", "100000"))
    paper_mode  = _get("PAPER_MODE", "true")
    mkt_state   = _get("MARKET_STATE", "SIDEWAYS")

    total_cost  = sum(float(r.get("total_cost",    0) or 0) for r in open_pos)
    total_value = sum(float(r.get("current_value", 0) or 0) for r in open_pos)
    total_pnl   = total_value - total_cost
    liquid      = max(0.0, capital - total_cost)
    slots       = max(0, 3 - len(open_pos))

    print(f"\n{'='*65}")
    print(f"  CURRENT PORTFOLIO STATE")
    print(f"{'='*65}")
    print(f"  Capital:   NPR {capital:>12,.0f}  | Market: {mkt_state}")
    print(f"  Invested:  NPR {total_cost:>12,.0f}  | Paper:  {paper_mode}")
    print(f"  Liquid:    NPR {liquid:>12,.0f}  | Slots:  {slots}/3 open")
    print(f"  Value:     NPR {total_value:>12,.0f}  | P&L:    NPR {total_pnl:+,.0f}")

    if open_pos:
        print(f"\n  OPEN ({len(open_pos)}):")
        print(f"  {'':2} {'Symbol':<10} {'Shares':>6} {'Entry':>8} {'LTP':>8} {'P&L%':>8}  Sector")
        print(f"  {'─'*65}")
        for r in sorted(open_pos, key=lambda x: float(x.get("pnl_pct", 0) or 0), reverse=True):
            pnl   = float(r.get("pnl_pct", 0) or 0)
            entry = float(r.get("entry_price") or r.get("wacc") or 0)
            ltp   = float(r.get("current_price", 0) or 0)
            icon  = "+" if pnl >= 0 else "-"
            print(
                f"  {icon}  {r.get('symbol','?'):<10}"
                f"{r.get('shares','?'):>6} "
                f"{entry:>8.0f} "
                f"{ltp:>8.0f} "
                f"{pnl:>+7.1f}%  "
                f"{str(r.get('sector', r.get('company', '')))[:18]}"
            )
    else:
        print(f"\n  OPEN: none")

    if closed:
        print(f"\n  CLOSED: {', '.join(r.get('symbol','?') for r in closed)}")

    print(f"\n  PIPELINE:")
    if slots > 0 and liquid > 5000:
        print(f"  -> Claude WILL generate BUY signals (slots={slots}, liquid=NPR {liquid:,.0f})")
    elif slots == 0:
        print(f"  -> Claude WILL NOT buy — portfolio full (0 slots)")
    else:
        print(f"  -> Claude WILL NOT buy — insufficient liquid cash (NPR {liquid:,.0f})")

    if open_pos:
        print(f"  -> capital_allocator WILL advise on {len(open_pos)} positions")
    else:
        print(f"  -> capital_allocator has nothing to advise on")

    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO A — FRESH START
# ══════════════════════════════════════════════════════════════════════════════

def setup_fresh(capital: float = 100_000):
    print(f"\n{'='*55}")
    print(f"  SCENARIO A — FRESH START")
    print(f"{'='*55}")

    print(f"  Saving snapshot of current state...")
    snap = save_snapshot()
    print(f"  Saved {len(snap['portfolio'])} rows -> {SNAPSHOT_FILE}")

    from sheets import run_raw_sql
    run_raw_sql("DELETE FROM portfolio")

    _set("CAPITAL_TOTAL_NPR", str(capital))
    _set("PAPER_MODE",        "true")
    _set("MARKET_STATE",      "CAUTIOUS_BULL")

    print(f"\n  Portfolio:    EMPTY")
    print(f"  Capital:      NPR {capital:,.0f}  (fully liquid)")
    print(f"  Slots:        3/3 open")
    print(f"  Paper mode:   ON")
    print(f"  Market state: CAUTIOUS_BULL  (tech threshold=58)")

    print(f"\n  TEST COMMANDS:")
    print(f"    python gemini_filter.py")
    print(f"    python claude_analyst.py")
    print(f"    python workflows/capital_allocator.py")

    print(f"\n  RESTORE AFTER:")
    print(f"    python mock_portfolio.py restore")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO B — CURRENT STATE (reads Neon as-is, does NOT overwrite)
# ══════════════════════════════════════════════════════════════════════════════

def setup_current(capital: float = None):
    print(f"\n{'='*55}")
    print(f"  SCENARIO B — CURRENT STATE")
    print(f"{'='*55}")

    from sheets import read_tab
    rows     = read_tab("portfolio")
    open_pos = [r for r in rows if str(r.get("status", "")).upper() == "OPEN"]

    if not open_pos:
        print(f"\n  No OPEN positions in Neon portfolio table.")
        print(f"  Options:")
        print(f"    python modules/meroshare.py   (sync real holdings)")
        print(f"    python mock_portfolio.py fresh (test BUY flow instead)\n")
        return

    # Save snapshot before doing anything
    print(f"  Saving snapshot...")
    save_snapshot()
    print(f"  Saved -> {SNAPSHOT_FILE}")

    # Only update capital if explicitly passed
    if capital is not None:
        _set("CAPITAL_TOTAL_NPR", str(capital))
    else:
        # Sanity check: capital must be >= invested
        current_cap = float(_get("CAPITAL_TOTAL_NPR", "100000"))
        total_cost  = sum(float(r.get("total_cost", 0) or 0) for r in open_pos)
        if current_cap < total_cost:
            suggested = round(total_cost + 5_000, -3)   # round to nearest 1k
            _set("CAPITAL_TOTAL_NPR", str(suggested))
            print(f"  Capital ({current_cap:,.0f}) < invested ({total_cost:,.0f}) — auto-fixed to NPR {suggested:,.0f}")

    _set("PAPER_MODE",   "true")
    _set("MARKET_STATE", "CAUTIOUS_BULL")

    # Summary
    total_cost  = sum(float(r.get("total_cost",    0) or 0) for r in open_pos)
    total_value = sum(float(r.get("current_value", 0) or 0) for r in open_pos)
    final_cap   = float(_get("CAPITAL_TOTAL_NPR", "100000"))
    liquid      = max(0.0, final_cap - total_cost)
    slots       = max(0, 3 - len(open_pos))

    print(f"\n  POSITIONS IN NEON ({len(open_pos)}):")
    print(f"  {'':2} {'Symbol':<10} {'Shares':>6} {'Entry':>8} {'LTP':>8} {'P&L%':>8}")
    print(f"  {'─'*50}")
    for r in sorted(open_pos, key=lambda x: float(x.get("pnl_pct", 0) or 0), reverse=True):
        pnl   = float(r.get("pnl_pct", 0) or 0)
        entry = float(r.get("entry_price") or r.get("wacc") or 0)
        ltp   = float(r.get("current_price", 0) or 0)
        icon  = "+" if pnl >= 0 else "-"
        print(
            f"  {icon}  {r.get('symbol','?'):<10}"
            f"{r.get('shares','?'):>6} "
            f"{entry:>8.0f} "
            f"{ltp:>8.0f} "
            f"{pnl:>+7.1f}%"
        )

    print(f"\n  Capital:  NPR {final_cap:,.0f}")
    print(f"  Invested: NPR {total_cost:,.0f}")
    print(f"  Liquid:   NPR {liquid:,.0f}  (slots={slots})")

    if liquid < 5000:
        print(f"  -> Claude will AVOID new buys (no cash)")
    else:
        print(f"  -> Claude MAY buy if a slot is open")

    print(f"\n  TEST COMMANDS:")
    print(f"    python workflows/capital_allocator.py  (primary test for this scenario)")
    print(f"    python claude_analyst.py               (expect AVOID)")

    print(f"\n  RESTORE AFTER:")
    print(f"    python mock_portfolio.py restore")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
# RESET
# ══════════════════════════════════════════════════════════════════════════════

def reset():
    print(f"\n  Saving snapshot first...")
    save_snapshot()

    from sheets import run_raw_sql
    run_raw_sql("DELETE FROM portfolio")

    _set("CAPITAL_TOTAL_NPR", "100000")
    _set("PAPER_MODE",        "true")
    _set("MARKET_STATE",      "SIDEWAYS")

    print(f"  Portfolio wiped, defaults restored.")
    print(f"  Run 'python mock_portfolio.py restore' to get back.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd  = args[0].lower() if args else "status"

    if cmd == "fresh":
        capital = float(args[1]) if len(args) > 1 else 100_000
        setup_fresh(capital)

    elif cmd == "current":
        capital = float(args[1]) if len(args) > 1 else None
        setup_current(capital)

    elif cmd == "restore":
        restore_snapshot()

    elif cmd == "reset":
        confirm = input("  Wipe portfolio and reset all settings? (yes/no): ")
        if confirm.strip().lower() == "yes":
            reset()
        else:
            print("  Cancelled\n")

    else:
        show_status()
