"""
find_big_movers.py
────────────────────────────────────────────────────────────────────────────
Find all NEPSE stocks that gained ≥50% in any rolling 30/60/90 day window
between 2023-07-01 and 2026-05-27 (NST).

For each qualifying move records:
  - symbol, window size, move start date, move end date
  - price at start, price at end, gain %
  - peak gain within window

Output:
  stat_method/output/big_movers_YYYY-MM-DD.csv

Usage:
    cd ~/nepse-engine
    python stat_method/find_big_movers.py
    python stat_method/find_big_movers.py --threshold 30   # lower threshold
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import date
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MOVERS] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR   = Path(__file__).parent / "output"
NST       = ZoneInfo("Asia/Kathmandu")
FROM_DATE = "2023-07-01"
TO_DATE   = "2026-05-27"
WINDOWS   = [30, 60, 90]       # calendar days
THRESHOLD = 50.0               # % gain to qualify


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_prices(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading price_history %s → %s ...", from_date, to_date)
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT symbol,
                   date::date AS date,
                   COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
            FROM price_history
            WHERE date >= %s
              AND date <= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (from_date, to_date))
        rows = cur.fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        log.error("No price data returned")
        return df

    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    log.info("  Loaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIND MOVES
# ══════════════════════════════════════════════════════════════════════════════

def find_moves(df: pd.DataFrame, windows: list[int],
               threshold: float) -> list[dict]:
    """
    For each symbol, roll each window size and find periods where
    close[end] / close[start] >= 1 + threshold/100.

    Only records the FIRST qualifying instance per symbol per window
    (the earliest start date) to avoid duplicate overlapping moves.
    Also records the BEST (highest gain) instance.
    """
    results = []
    symbols = df["symbol"].unique()
    log.info("Scanning %d symbols across windows %s ...", len(symbols), windows)

    for sym in symbols:
        sym_df = df[df["symbol"] == sym].sort_values("date").reset_index(drop=True)

        if len(sym_df) < 10:
            continue

        dates  = sym_df["date"].values
        closes = sym_df["close"].values

        for window in windows:
            window_td = pd.Timedelta(days=window)

            # For each start index, find the end index ~window days later
            found_first = False
            best = None

            for i in range(len(sym_df)):
                start_date  = dates[i]
                end_date    = start_date + window_td
                start_price = closes[i]

                # Find rows within window
                mask = (dates > start_date) & (dates <= end_date)
                if not mask.any():
                    continue

                window_closes = closes[mask]
                window_dates  = dates[mask]

                max_close = window_closes.max()
                max_idx   = window_closes.argmax()
                max_date  = window_dates[max_idx]

                gain = (max_close - start_price) / start_price * 100

                if gain < threshold:
                    continue

                record = {
                    "symbol":       sym,
                    "window_days":  window,
                    "move_start":   pd.Timestamp(start_date).date(),
                    "move_end":     pd.Timestamp(max_date).date(),
                    "price_start":  round(float(start_price), 2),
                    "price_end":    round(float(max_close), 2),
                    "gain_pct":     round(gain, 1),
                    "calendar_days": (pd.Timestamp(max_date) - pd.Timestamp(start_date)).days,
                }

                # Track first occurrence
                if not found_first:
                    results.append({**record, "instance": "first"})
                    found_first = True

                # Track best occurrence
                if best is None or gain > best["gain_pct"]:
                    best = record

            # Add best if different from first
            if best is not None:
                # Check if we already added this as first
                existing_first = [r for r in results
                                  if r["symbol"] == sym
                                  and r["window_days"] == window
                                  and r["instance"] == "first"]
                if existing_first and existing_first[0]["move_start"] != best["move_start"]:
                    results.append({**best, "instance": "best"})
                elif existing_first:
                    # Mark first as also best if same
                    existing_first[0]["instance"] = "first+best"

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], threshold: float) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        log.info("No moves found above %.0f%%", threshold)
        return

    log.info("\n%s", "=" * 70)
    log.info("BIG MOVERS — ≥%.0f%% gain — %s → %s", threshold, FROM_DATE, TO_DATE)
    log.info("=" * 70)

    for window in WINDOWS:
        wdf = df[df["window_days"] == window]
        # One row per symbol (best gain)
        best = wdf.loc[wdf.groupby("symbol")["gain_pct"].idxmax()] if not wdf.empty else wdf

        log.info("\n── %d-day window: %d qualifying symbols ──", window, best["symbol"].nunique())
        if best.empty:
            continue

        best_sorted = best.sort_values("gain_pct", ascending=False)
        log.info("  %-10s %8s %10s %10s %10s %12s",
                 "Symbol", "Window", "Start", "End", "Gain%", "Start→End")
        for _, r in best_sorted.head(20).iterrows():
            log.info("  %-10s %6dd  %10s %10s  %+7.1f%%  %s→%s",
                     r["symbol"], r["window_days"],
                     r["move_start"], r["move_end"],
                     r["gain_pct"],
                     f"{r['price_start']:.1f}", f"{r['price_end']:.1f}")
        if len(best_sorted) > 20:
            log.info("  ... and %d more", len(best_sorted) - 20)

    # Overall unique symbols
    all_syms = df["symbol"].unique()
    log.info("\n── TOTAL unique symbols with ≥%.0f%% move in any window: %d ──",
             threshold, len(all_syms))
    log.info("   %s", ", ".join(sorted(all_syms)))


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(results: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    today    = date.today().strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"big_movers_{today}.csv"

    if not results:
        log.warning("No results to save")
        return

    keys = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    log.info("\nSaved: %s (%d rows)", out_path, len(results))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Min gain %% to qualify (default {THRESHOLD})")
    parser.add_argument("--from-date", default=FROM_DATE)
    parser.add_argument("--to-date",   default=TO_DATE)
    args = parser.parse_args()

    df = load_prices(args.from_date, args.to_date)
    if df.empty:
        return

    results = find_moves(df, WINDOWS, args.threshold)
    log.info("Found %d qualifying move instances", len(results))

    print_summary(results, args.threshold)
    save_csv(results)


if __name__ == "__main__":
    main()