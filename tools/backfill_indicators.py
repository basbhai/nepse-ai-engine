"""
tools/backfill_indicators.py
─────────────────────────────────────────────────────────────────────────────
Recomputes indicators for specific broken dates where RSI was wrong due to the
floorsheet date normalization bug.

Broken dates: 2026-04-27, 2026-04-28, 2026-04-29, 2026-04-30, 2026-05-04,
              2026-05-06, 2026-05-07, 2026-05-08, 2026-05-15

Usage:
  c                   → all 9 broken dates
  python tools/backfill_indicators.py --date 2026-05-08  → single date
  python tools/backfill_indicators.py --verify-only      → print RSI distribution only
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import os
from collections import defaultdict
from dataclasses import dataclass

# Allow imports from project root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheets import run_raw_sql, write_indicators_batch
from modules.indicators import compute_indicators, HistoryCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BACKFILL] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

BROKEN_DATES = [
    "2026-04-27",
    "2026-04-28",
    "2026-04-29",
    "2026-04-30",
    "2026-05-04",
    "2026-05-06",
    "2026-05-07",
    "2026-05-08",
    "2026-05-15",
]

PERIODS = 250


# ── Minimal price row compatible with compute_indicators() ────────────────────

@dataclass
class _PriceRow:
    symbol:    str
    ltp:       float = 0.0
    close:     float = 0.0
    high:      float = 0.0
    low:       float = 0.0
    volume:    float = 0.0
    open_price: float = 0.0
    prev_close: float = 0.0


# ── Date-capped history loader ────────────────────────────────────────────────

def _load_history_capped(target_date: str, periods: int = PERIODS) -> dict[str, dict]:
    """
    Load the most recent `periods` distinct trading days from price_history
    where date <= target_date. Returns same structure as load_history_all_symbols().
    """
    date_rows = run_raw_sql(
        """
        SELECT DISTINCT date FROM price_history
        WHERE date <= %s
        ORDER BY date DESC
        LIMIT %s
        """,
        (target_date, periods),
    )

    if not date_rows:
        log.warning("No price history found up to %s", target_date)
        return {}

    dates = sorted(r["date"] for r in date_rows)
    log.info("  History: %d trading days (%s → %s)", len(dates), dates[0], dates[-1])

    rows = run_raw_sql(
        """
        SELECT date, symbol, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date = ANY(%s)
        ORDER BY symbol, date ASC
        """,
        (dates,),
    )

    if not rows:
        return {}

    def _f(val) -> float:
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    symbol_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        symbol_rows[row["symbol"]].append(row)

    result = {}
    for symbol, sym_rows in symbol_rows.items():
        result[symbol] = {
            "dates":   [r["date"]               for r in sym_rows],
            "closes":  [_f(r["close"] or r["ltp"]) for r in sym_rows],
            "highs":   [_f(r["high"])            for r in sym_rows],
            "lows":    [_f(r["low"])             for r in sym_rows],
            "volumes": [_f(r["volume"])          for r in sym_rows],
        }

    return result


def _build_cache(hist_data: dict[str, dict]) -> HistoryCache:
    """Build a HistoryCache from pre-loaded history dict."""
    cache = HistoryCache()
    all_dates: set[str] = set()

    for symbol, hist in hist_data.items():
        sym = symbol.upper()
        cache.closes[sym]  = hist.get("closes",  [])
        cache.highs[sym]   = hist.get("highs",   [])
        cache.lows[sym]    = hist.get("lows",    [])
        cache.volumes[sym] = hist.get("volumes", [])
        all_dates.update(hist.get("dates", []))

    cache.dates = sorted(all_dates)
    return cache


def _load_market_data(target_date: str) -> dict[str, _PriceRow]:
    """
    Load closes/highs/lows/volumes for every symbol on target_date
    directly from price_history (not live ATrad prices).
    """
    rows = run_raw_sql(
        """
        SELECT symbol, close, ltp, high, low, volume
        FROM price_history
        WHERE date = %s
        """,
        (target_date,),
    )

    def _f(val) -> float:
        try:
            return float(val) if val else 0.0
        except (ValueError, TypeError):
            return 0.0

    market_data: dict[str, _PriceRow] = {}
    for row in rows:
        sym   = row["symbol"].upper()
        close = _f(row["close"]) or _f(row["ltp"])
        market_data[sym] = _PriceRow(
            symbol=sym,
            ltp=close,
            close=close,
            high=_f(row["high"])   or close,
            low=_f(row["low"])     or close,
            volume=_f(row["volume"]),
        )

    return market_data


# ── RSI distribution helper ───────────────────────────────────────────────────

def _rsi_distribution(target_date: str) -> dict:
    rows = run_raw_sql(
        "SELECT rsi_14 FROM indicators WHERE date = %s",
        (target_date,),
    )

    total   = len(rows)
    under10 = 0
    mid     = 0
    over70  = 0
    missing = 0

    for r in rows:
        val = r.get("rsi_14") or ""
        if not val or val == "":
            missing += 1
            continue
        try:
            rsi = float(val)
            if rsi < 10:
                under10 += 1
            elif rsi > 70:
                over70 += 1
            else:
                mid += 1
        except (ValueError, TypeError):
            missing += 1

    return {
        "date":     target_date,
        "total":    total,
        "under10":  under10,
        "mid":      mid,
        "over70":   over70,
        "missing":  missing,
    }


def print_rsi_distribution(target_date: str) -> None:
    d = _rsi_distribution(target_date)
    broken = "  *** BROKEN ***" if d["total"] > 0 and d["under10"] / d["total"] > 0.10 else ""
    print(
        f"  {target_date}  total={d['total']:>4}  "
        f"rsi<10={d['under10']:>4}  "
        f"10-70={d['mid']:>4}  "
        f"rsi>70={d['over70']:>4}  "
        f"missing={d['missing']:>4}"
        f"{broken}"
    )


# ── Per-date backfill ─────────────────────────────────────────────────────────

def backfill_date(target_date: str) -> int:
    """
    Recompute and overwrite indicators for target_date.
    Returns number of rows written.
    """
    print(f"\n{'-'*60}")
    print(f"  Backfilling {target_date} ...")

    # 1. Load date-capped price history
    hist_data = _load_history_capped(target_date, periods=PERIODS)
    if not hist_data:
        log.error("  No history data found up to %s — skipping", target_date)
        return 0

    # 2. Build HistoryCache
    cache = _build_cache(hist_data)
    log.info("  Cache: %d symbols | %d dates | up to %s",
             len(cache.closes), len(cache.dates),
             cache.dates[-1] if cache.dates else "—")

    # 3. Load market_data from price_history for exactly target_date
    market_data = _load_market_data(target_date)
    if not market_data:
        log.error("  No price_history rows for %s — skipping", target_date)
        return 0
    log.info("  market_data: %d symbols on %s", len(market_data), target_date)

    # 4. Compute indicators for all symbols
    results = []
    skipped = 0

    for symbol, price_row in market_data.items():
        if not str(symbol).replace("-", "").replace("_", "").isalpha():
            skipped += 1
            continue
        try:
            result = compute_indicators(symbol, price_row, cache, date=target_date)
            results.append(result.to_dict())
        except Exception as exc:
            log.warning("  %s: compute failed — %s", symbol, exc)
            skipped += 1

    log.info("  Computed: %d | skipped: %d", len(results), skipped)

    if not results:
        log.error("  No results to write — skipping DB write")
        return 0

    # 5. Upsert into indicators table (overwrites broken rows)
    written = write_indicators_batch(results)
    log.info("  Written: %d rows to indicators table", written)

    # 6. Verify RSI distribution
    print(f"  After backfill:")
    print_rsi_distribution(target_date)

    return written


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    # ── verify-only: print distribution for all 9 dates ──────────────────────
    if "--verify-only" in args:
        print("\n" + "=" * 70)
        print("  RSI DISTRIBUTION - broken dates (current state in DB)")
        print("=" * 70)
        print(f"  {'Date':<12}  {'Total':>5}  {'rsi<10':>6}  {'10-70':>5}  {'rsi>70':>6}  {'missing':>7}")
        print("  " + "-" * 60)
        for d in BROKEN_DATES:
            print_rsi_distribution(d)
        print()
        return

    # ── single date ───────────────────────────────────────────────────────────
    if "--date" in args:
        idx = args.index("--date")
        if idx + 1 >= len(args):
            print("ERROR: --date requires a YYYY-MM-DD argument")
            sys.exit(1)
        target = args[idx + 1]
        if target not in BROKEN_DATES:
            print(f"WARNING: {target} is not in the known broken dates list")
            print(f"Known: {BROKEN_DATES}")

        print(f"\nBefore:")
        print_rsi_distribution(target)
        backfill_date(target)
        return

    # ── full run: all 9 broken dates ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BACKFILL — recomputing indicators for 9 broken dates")
    print("=" * 70)

    print("\nBEFORE (current broken state):")
    print(f"  {'Date':<12}  {'Total':>5}  {'rsi<10':>6}  {'10-70':>5}  {'rsi>70':>6}  {'missing':>7}")
    print("  " + "-" * 60)
    for d in BROKEN_DATES:
        print_rsi_distribution(d)

    total_written = 0
    for d in BROKEN_DATES:
        written = backfill_date(d)
        total_written += written

    print("\n" + "=" * 70)
    print(f"  DONE — {total_written} total rows written")
    print("=" * 70)

    print("\nAFTER (post-fix state):")
    print(f"  {'Date':<12}  {'Total':>5}  {'rsi<10':>6}  {'10-70':>5}  {'rsi>70':>6}  {'missing':>7}")
    print("  " + "-" * 60)
    for d in BROKEN_DATES:
        print_rsi_distribution(d)
    print()


if __name__ == "__main__":
    main()
