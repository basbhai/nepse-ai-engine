"""
momentum_screener.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Momentum Screener
Finds all (symbol, date) pairs where close >= 15% above close 15 trading
days earlier, from July 2025 onward.

Output: stat_method/output/momentum_hits.json

Usage:
    cd ~/nepse-engine
    python stat_method/momentum_screener.py
    python stat_method/momentum_screener.py --min-gain 20   # change threshold
    python stat_method/momentum_screener.py --dry-run       # print stats only
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SCREENER] %(message)s",
)
log = logging.getLogger(__name__)

START_DATE  = "2025-07-01"
MIN_GAIN    = 15.0          # % threshold
LOOKBACK    = 15            # trading days


def load_price_history() -> dict[str, list[tuple]]:
    """
    Load price_history from July 2025 onward.
    Returns dict: symbol → sorted list of (date, close, volume, turnover, conf_score, vwap, week52_high, week52_low)
    Only rows with valid numeric close are included.
    """
    log.info("Loading price_history from %s onward...", START_DATE)
    with _db() as cur:
        cur.execute("""
            SELECT
                symbol,
                date,
                close,
                volume,
                turnover,
                conf_score,
                vwap,
                week52_high,
                week52_low
            FROM price_history
            WHERE date >= %s
              AND close IS NOT NULL
              AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (START_DATE,))
        rows = cur.fetchall()

    log.info("Loaded %d raw rows", len(rows))

    by_symbol: dict[str, list] = defaultdict(list)
    for r in rows:
        sym = str(r["symbol"]).upper().strip()
        try:
            close = float(r["close"])
            vol   = float(r["volume"]) if r["volume"] and r["volume"] != "" else 0.0
            turn  = float(r["turnover"]) if r["turnover"] and r["turnover"] != "" else 0.0
            conf  = float(r["conf_score"]) if r["conf_score"] and r["conf_score"] != "" else None
            vwap  = float(r["vwap"]) if r["vwap"] and r["vwap"] != "" else None
            w52h  = float(r["week52_high"]) if r["week52_high"] and r["week52_high"] != "" else None
            w52l  = float(r["week52_low"]) if r["week52_low"] and r["week52_low"] != "" else None
        except (ValueError, TypeError):
            continue
        by_symbol[sym].append((r["date"], close, vol, turn, conf, vwap, w52h, w52l))

    log.info("Loaded %d symbols", len(by_symbol))
    return dict(by_symbol)


def screen(price_data: dict, min_gain: float = MIN_GAIN) -> list[dict]:
    """
    For each symbol, scan every date where the 15-day return >= min_gain%.
    Returns list of hit dicts sorted by date.
    """
    hits = []

    for symbol, rows in price_data.items():
        # rows already sorted by date ASC
        n = len(rows)
        if n < LOOKBACK + 1:
            continue

        for i in range(LOOKBACK, n):
            date_today,  close_today,  vol_today,  turn_today,  conf_today,  vwap_today,  w52h, w52l  = rows[i]
            date_15d,    close_15d,    vol_15d,    _,           _,           _,           _,    _      = rows[i - LOOKBACK]

            if close_15d <= 0:
                continue

            gain_pct = (close_today - close_15d) / close_15d * 100

            if gain_pct < min_gain:
                continue

            vol_ratio = round(vol_today / vol_15d, 2) if vol_15d > 0 else None

            # 52-week position: where in the range is today's close
            pct_52w_range = None
            if w52h and w52l and (w52h - w52l) > 0:
                pct_52w_range = round((close_today - w52l) / (w52h - w52l) * 100, 1)

            hits.append({
                "symbol":          symbol,
                "trigger_date":    date_today,
                "date_15d_ago":    date_15d,
                "close":           round(close_today, 2),
                "close_15d_ago":   round(close_15d, 2),
                "gain_15d_pct":    round(gain_pct, 2),
                "volume":          round(vol_today, 0),
                "vol_ratio":       vol_ratio,
                "turnover":        round(turn_today, 0),
                "conf_score":      conf_today,
                "vwap":            vwap_today,
                "week52_high":     w52h,
                "week52_low":      w52l,
                "pct_52w_range":   pct_52w_range,
            })

    hits.sort(key=lambda x: (x["trigger_date"], x["symbol"]))
    return hits


def print_summary(hits: list[dict]) -> None:
    from collections import Counter
    log.info("=" * 60)
    log.info("MOMENTUM SCREENER RESULTS")
    log.info("=" * 60)
    log.info("Total hits (≥%.0f%% in 15 days): %d", MIN_GAIN, len(hits))

    # Unique symbols
    symbols = set(h["symbol"] for h in hits)
    log.info("Unique symbols: %d", len(symbols))

    # Hits per month
    by_month = Counter(h["trigger_date"][:7] for h in hits)
    log.info("\nHits by month:")
    for month in sorted(by_month):
        log.info("  %s: %d hits", month, by_month[month])

    # Gain distribution
    gains = [h["gain_15d_pct"] for h in hits]
    if gains:
        log.info("\nGain distribution:")
        log.info("  15-20%%: %d", sum(1 for g in gains if 15 <= g < 20))
        log.info("  20-30%%: %d", sum(1 for g in gains if 20 <= g < 30))
        log.info("  30-50%%: %d", sum(1 for g in gains if 30 <= g < 50))
        log.info("  50%%+  : %d", sum(1 for g in gains if g >= 50))

    # Top symbols by hit count
    sym_count = Counter(h["symbol"] for h in hits)
    log.info("\nTop 10 most frequent symbols:")
    for sym, cnt in sym_count.most_common(10):
        log.info("  %s: %d hits", sym, cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gain", type=float, default=MIN_GAIN)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    price_data = load_price_history()
    hits       = screen(price_data, min_gain=args.min_gain)
    print_summary(hits)

    if args.dry_run:
        log.info("Dry run — not writing output")
        return

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "momentum_hits.json"

    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "start_date":  START_DATE,
                "min_gain_pct": args.min_gain,
                "lookback_days": LOOKBACK,
                "total_hits":  len(hits),
                "unique_symbols": len(set(h["symbol"] for h in hits)),
            },
            "hits": hits,
        }, f, indent=2)

    log.info("Written to %s", out_path)


if __name__ == "__main__":
    main()
