"""
accumulation_detector.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Pre-Signal Accumulation Detector

Hypothesis: Smart money accumulates quietly before a visible price spike.
If we can detect that accumulation phase early, we can enter before the
crowd and capture more of the move with better risk/reward.

What this script does:
  1. For every (symbol, date) from July 2023 onward, scores a rolling
     15-day floorsheet window on accumulation signals
  2. Flags dates where accumulation score exceeds threshold AND price
     is still calm (not yet spiking)
  3. Measures forward outcomes: D+5, D+10, D+20, D+30 price return
  4. Tracks whether a momentum trigger (≥15% in 15 days) fired within
     30 trading days of detection
  5. Tests two calm-price thresholds: 5% and 8%

Output: stat_method/output/accumulation_signals.json

Usage:
    cd ~/nepse-engine
    python stat_method/accumulation_detector.py
    python stat_method/accumulation_detector.py --dry-run
    python stat_method/accumulation_detector.py --from-date 2025-01-01
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ACCUM] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE        = "2023-07-01"
WINDOW            = 15        # rolling floorsheet window (trading days)
CALM_THRESHOLDS   = [0.05, 0.08]   # test both: 5% and 8% max price gain
MIN_ACC_SCORE     = 3         # minimum score to flag as accumulation
MOMENTUM_TRIGGER  = 0.15      # ≥15% in 15 days = momentum screen fired
FORWARD_WINDOWS   = [5, 10, 20, 30]

# Market-wide baselines from the data (use as comparison reference)
BASELINE_BP   = 0.385   # avg buyer_pressure market-wide
BASELINE_BC   = 0.615   # avg broker_concentration market-wide


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_floorsheet() -> dict[str, list[dict]]:
    """
    Load all floorsheet_signals from START_DATE.
    Returns: symbol → list of {date, buyer_pressure, broker_concentration,
                                institutional_flag, total_volume} sorted ASC
    """
    log.info("Loading floorsheet_signals from %s...", START_DATE)
    with _db() as cur:
        cur.execute("""
            SELECT
                symbol, date,
                buyer_pressure::float        AS bp,
                broker_concentration::float  AS bc,
                institutional_flag,
                total_volume::float          AS vol,
                large_order_pct::float       AS lop
            FROM floorsheet_signals
            WHERE date >= %s
              AND buyer_pressure IS NOT NULL AND buyer_pressure != ''
              AND broker_concentration IS NOT NULL AND broker_concentration != ''
            ORDER BY symbol, date ASC
        """, (START_DATE,))
        rows = cur.fetchall()

    by_symbol: dict[str, list] = defaultdict(list)
    for r in rows:
        sym = str(r["symbol"]).upper().strip()
        by_symbol[sym].append({
            "date": r["date"],
            "bp":   round(r["bp"], 4),
            "bc":   round(r["bc"], 4),
            "inst": str(r["institutional_flag"]).lower() == "true",
            "vol":  r["vol"] or 0.0,
            "lop":  r["lop"] or 0.0,
        })

    log.info("Loaded floorsheet for %d symbols", len(by_symbol))
    return dict(by_symbol)


def load_price_history() -> dict[str, list[dict]]:
    """
    Load price_history from START_DATE.
    Returns: symbol → sorted list of {date, close, volume}
    """
    log.info("Loading price_history from %s...", START_DATE)
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date, close::float AS close, volume::float AS vol
            FROM price_history
            WHERE date >= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (START_DATE,))
        rows = cur.fetchall()

    by_symbol: dict[str, list] = defaultdict(list)
    for r in rows:
        sym = str(r["symbol"]).upper().strip()
        try:
            by_symbol[sym].append({
                "date":  r["date"],
                "close": float(r["close"]),
                "vol":   float(r["vol"]) if r["vol"] else 0.0,
            })
        except (ValueError, TypeError):
            continue

    log.info("Price history loaded for %d symbols", len(by_symbol))
    return dict(by_symbol)


def load_sectors() -> dict[str, str]:
    with _db() as cur:
        cur.execute("SELECT symbol, sectorname FROM share_sectors")
        rows = cur.fetchall()
    return {str(r["symbol"]).upper(): r["sectorname"] for r in rows}


# ══════════════════════════════════════════════════════════════════════════════
# ACCUMULATION SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_window(window: list[dict]) -> dict:
    """
    Score a 15-day floorsheet window for accumulation signals.
    Returns a dict with individual signal components and total score.

    Scoring logic (each component 0 or 1, total max = 6):

    1. bp_rising      — buyer_pressure has upward slope over window
    2. bp_above_base  — avg buyer_pressure in window > market baseline (0.385)
    3. inst_days      — ≥3 institutional_flag=true days in window (out of 15)
    4. bc_tightening  — broker_concentration has downward slope (fewer brokers
                        absorbing more = smart money concentrating)
                        NOTE: lower bc = broader distribution, higher bc =
                        more concentrated. Tightening = bc rising, meaning
                        a smaller group handling more volume.
    5. bc_below_base  — avg bc in window < market baseline — broad participation
                        still (early accumulation, not yet operator-controlled)
    6. vol_quiet      — volume in second half of window not yet exploding
                        (ratio of last-5-day avg vol to first-5-day avg vol < 2x)
    """
    if len(window) < 5:
        return {"score": 0}

    bps  = [d["bp"]   for d in window]
    bcs  = [d["bc"]   for d in window]
    insts = [d["inst"] for d in window]
    vols  = [d["vol"]  for d in window]

    n = len(bps)

    # 1. buyer_pressure slope (linear regression sign)
    bp_slope = _slope(bps)
    bp_rising = 1 if bp_slope > 0.001 else 0

    # 2. avg buyer_pressure above baseline
    avg_bp = sum(bps) / n
    bp_above_base = 1 if avg_bp > BASELINE_BP else 0

    # 3. institutional days
    inst_count = sum(1 for x in insts if x)
    inst_score = 1 if inst_count >= 3 else 0

    # 4. broker concentration tightening (slope rising = concentrating)
    bc_slope = _slope(bcs)
    bc_tightening = 1 if bc_slope > 0.001 else 0

    # 5. avg bc below baseline (still broad, not yet operator-dominated)
    avg_bc = sum(bcs) / n
    bc_below_base = 1 if avg_bc < BASELINE_BC else 0

    # 6. volume still quiet (no explosion yet)
    if len(vols) >= 10 and vols[0] > 0:
        early_vol = sum(vols[:5]) / 5
        late_vol  = sum(vols[-5:]) / 5
        vol_quiet = 1 if (early_vol == 0 or late_vol / early_vol < 2.0) else 0
    else:
        vol_quiet = 1  # insufficient data, assume quiet

    total = bp_rising + bp_above_base + inst_score + bc_tightening + bc_below_base + vol_quiet

    return {
        "score":          total,
        "bp_rising":      bp_rising,
        "bp_above_base":  bp_above_base,
        "inst_days":      inst_count,
        "inst_score":     inst_score,
        "bc_tightening":  bc_tightening,
        "bc_below_base":  bc_below_base,
        "vol_quiet":      vol_quiet,
        "avg_bp":         round(avg_bp, 4),
        "avg_bc":         round(avg_bc, 4),
        "bp_slope":       round(bp_slope, 6),
        "bc_slope":       round(bc_slope, 6),
    }


def _slope(values: list[float]) -> float:
    """Linear regression slope (least squares)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_m = (n - 1) / 2
    y_m = sum(values) / n
    num = sum((i - x_m) * (values[i] - y_m) for i in range(n))
    den = sum((i - x_m) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PRICE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_price_index(price_rows: list[dict]) -> dict[str, int]:
    """Date → index mapping for fast lookup."""
    return {r["date"]: i for i, r in enumerate(price_rows)}


def price_calm(price_rows: list[dict], price_idx: dict,
               detection_date: str, threshold: float) -> bool:
    """
    Returns True if price gain over last 15 trading days is below threshold.
    This ensures we're in the pre-spike phase.
    """
    try:
        i = price_idx[detection_date]
    except KeyError:
        i = max((j for j, r in enumerate(price_rows) if r["date"] <= detection_date),
                default=-1)
    if i < 15:
        return False
    close_now  = price_rows[i]["close"]
    close_15d  = price_rows[i - 15]["close"]
    if close_15d <= 0:
        return False
    gain = (close_now - close_15d) / close_15d
    return gain < threshold


def forward_returns(price_rows: list[dict], price_idx: dict,
                    detection_date: str) -> dict:
    """
    Compute forward returns and check if momentum trigger fires within 30 days.
    """
    try:
        i0 = price_idx[detection_date]
    except KeyError:
        i0 = max((j for j, r in enumerate(price_rows) if r["date"] <= detection_date),
                 default=-1)
    if i0 < 0:
        return {}

    base = price_rows[i0]["close"]
    result = {}

    # Forward returns at each window
    for w in FORWARD_WINDOWS:
        fi = i0 + w
        if fi < len(price_rows):
            ret = (price_rows[fi]["close"] - base) / base * 100
            result[f"fwd_{w}d_pct"]  = round(ret, 2)
            result[f"fwd_{w}d_date"] = price_rows[fi]["date"]
        else:
            result[f"fwd_{w}d_pct"]  = None
            result[f"fwd_{w}d_date"] = None

    # Did momentum screen fire within 30 days?
    # Check every (i, i-15) pair from i0+1 to i0+30
    triggered_day  = None
    trigger_gain   = None
    max_gain_30d   = None
    max_gain_date  = None

    gains_30d = []
    for offset in range(1, 31):
        fi = i0 + offset
        if fi >= len(price_rows):
            break
        fwd_close = price_rows[fi]["close"]
        gains_30d.append((price_rows[fi]["date"],
                          round((fwd_close - base) / base * 100, 2)))

        # Check if momentum screen fires: fwd_close vs close 15 days before fi
        fi_15 = fi - 15
        if fi_15 >= 0:
            close_15_before = price_rows[fi_15]["close"]
            if close_15_before > 0:
                gain_15d = (fwd_close - close_15_before) / close_15_before
                if gain_15d >= MOMENTUM_TRIGGER and triggered_day is None:
                    triggered_day = offset
                    trigger_gain  = round(gain_15d * 100, 2)

    if gains_30d:
        best = max(gains_30d, key=lambda x: x[1])
        max_gain_30d  = best[1]
        max_gain_date = best[0]

    result["momentum_triggered"]      = triggered_day is not None
    result["momentum_trigger_day"]    = triggered_day
    result["momentum_trigger_gain"]   = trigger_gain
    result["max_gain_30d_pct"]        = max_gain_30d
    result["max_gain_30d_date"]       = max_gain_date

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def detect(floorsheet: dict, prices: dict, sectors: dict,
           calm_threshold: float, from_date: str = None) -> list[dict]:
    """
    Scan all symbols for accumulation signals.
    Returns list of detection events with scores and forward outcomes.
    """
    signals = []
    symbols = set(floorsheet.keys()) & set(prices.keys())
    log.info("Scanning %d symbols (calm threshold: %.0f%%)...",
             len(symbols), calm_threshold * 100)

    for sym_i, symbol in enumerate(sorted(symbols)):
        if (sym_i + 1) % 50 == 0:
            log.info("  %d / %d symbols...", sym_i + 1, len(symbols))

        fs_rows    = floorsheet[symbol]   # sorted by date
        price_rows = prices[symbol]
        price_idx  = build_price_index(price_rows)
        sector     = sectors.get(symbol, "Unknown")

        n = len(fs_rows)
        if n < WINDOW + 1:
            continue

        # Track last detection date per symbol to avoid overlapping signals
        # (minimum 10 trading days between detections)
        last_detection_i = -999

        for i in range(WINDOW, n):
            # Apply from_date filter
            if from_date and fs_rows[i]["date"] < from_date:
                continue

            # Avoid overlapping detections
            if i - last_detection_i < 10:
                continue

            window = fs_rows[i - WINDOW:i]
            detection_date = fs_rows[i]["date"]

            # Check price is still calm
            if not price_calm(price_rows, price_idx, detection_date, calm_threshold):
                continue

            # Score the accumulation window
            acc = score_window(window)
            if acc["score"] < MIN_ACC_SCORE:
                continue

            # Compute forward returns from detection date
            fwd = forward_returns(price_rows, price_idx, detection_date)
            if not fwd:
                continue

            # Get price context at detection
            try:
                pi = price_idx.get(detection_date)
                if pi is None:
                    pi = max((j for j, r in enumerate(price_rows)
                              if r["date"] <= detection_date), default=-1)
                close_now = price_rows[pi]["close"] if pi >= 0 else None
                # 15-day price gain at detection
                gain_15d_at_detection = None
                if pi is not None and pi >= 15:
                    c15 = price_rows[pi - 15]["close"]
                    if c15 > 0:
                        gain_15d_at_detection = round(
                            (price_rows[pi]["close"] - c15) / c15 * 100, 2)
            except Exception:
                close_now = None
                gain_15d_at_detection = None

            signals.append({
                "symbol":          symbol,
                "sector":          sector,
                "detection_date":  detection_date,
                "calm_threshold":  calm_threshold,
                "close_at_detection": round(close_now, 2) if close_now else None,
                "gain_15d_at_detection": gain_15d_at_detection,
                # Accumulation score
                **{k: v for k, v in acc.items()},
                # Forward outcomes
                **fwd,
            })

            last_detection_i = i

    signals.sort(key=lambda x: (x["detection_date"], x["symbol"]))
    return signals


def summarise(signals: list[dict], label: str) -> dict:
    """Build summary statistics for a set of signals."""
    if not signals:
        return {"n": 0}

    n = len(signals)

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 2) if v else None

    def _median(vals):
        v = sorted(x for x in vals if x is not None)
        if not v:
            return None
        mid = len(v) // 2
        return round((v[mid - 1] + v[mid]) / 2 if len(v) % 2 == 0 else v[mid], 2)

    def _pct_positive(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None
        return round(sum(1 for x in v if x > 0) / len(v) * 100, 1)

    fwd5  = [s.get("fwd_5d_pct")  for s in signals]
    fwd10 = [s.get("fwd_10d_pct") for s in signals]
    fwd20 = [s.get("fwd_20d_pct") for s in signals]
    fwd30 = [s.get("fwd_30d_pct") for s in signals]
    max30 = [s.get("max_gain_30d_pct") for s in signals]

    triggered = [s for s in signals if s.get("momentum_triggered")]
    trigger_days = [s["momentum_trigger_day"] for s in triggered
                    if s.get("momentum_trigger_day")]

    # Score distribution
    score_dist = {}
    for s in signals:
        sc = str(s.get("score", 0))
        score_dist[sc] = score_dist.get(sc, 0) + 1

    # Sector breakdown
    sector_counts = {}
    for s in signals:
        sec = s.get("sector", "Unknown")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # High score subset (score >= 5)
    high_score = [s for s in signals if s.get("score", 0) >= 5]
    hs_fwd20   = [s.get("fwd_20d_pct") for s in high_score]

    return {
        "label":           label,
        "n":               n,
        "unique_symbols":  len(set(s["symbol"] for s in signals)),
        "date_range": {
            "first": min(s["detection_date"] for s in signals),
            "last":  max(s["detection_date"] for s in signals),
        },
        "momentum_triggered": {
            "count":   len(triggered),
            "pct":     round(len(triggered) / n * 100, 1),
            "avg_trigger_day": _avg(trigger_days),
            "median_trigger_day": _median(trigger_days),
        },
        "forward_returns": {
            "fwd_5d":  {"avg": _avg(fwd5),  "median": _median(fwd5),  "pct_positive": _pct_positive(fwd5),  "n": sum(1 for x in fwd5  if x is not None)},
            "fwd_10d": {"avg": _avg(fwd10), "median": _median(fwd10), "pct_positive": _pct_positive(fwd10), "n": sum(1 for x in fwd10 if x is not None)},
            "fwd_20d": {"avg": _avg(fwd20), "median": _median(fwd20), "pct_positive": _pct_positive(fwd20), "n": sum(1 for x in fwd20 if x is not None)},
            "fwd_30d": {"avg": _avg(fwd30), "median": _median(fwd30), "pct_positive": _pct_positive(fwd30), "n": sum(1 for x in fwd30 if x is not None)},
            "max_30d": {"avg": _avg(max30), "median": _median(max30), "n": sum(1 for x in max30 if x is not None)},
        },
        "high_score_subset": {
            "n":      len(high_score),
            "fwd_20d_avg":    _avg(hs_fwd20),
            "fwd_20d_median": _median(hs_fwd20),
            "fwd_20d_pct_positive": _pct_positive(hs_fwd20),
        },
        "score_distribution": score_dist,
        "sector_breakdown":   dict(sorted(sector_counts.items(),
                                          key=lambda x: -x[1])[:10]),
        "signal_quality": {
            "avg_inst_days":  _avg([s.get("inst_days") for s in signals]),
            "avg_bp":         _avg([s.get("avg_bp")    for s in signals]),
            "avg_bc":         _avg([s.get("avg_bc")    for s in signals]),
            "pct_bp_rising":  round(sum(1 for s in signals if s.get("bp_rising")) / n * 100, 1),
            "pct_bc_tighten": round(sum(1 for s in signals if s.get("bc_tightening")) / n * 100, 1),
            "pct_vol_quiet":  round(sum(1 for s in signals if s.get("vol_quiet")) / n * 100, 1),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--from-date",  default=None,
                        help="Only detect from this date (YYYY-MM-DD)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Single calm threshold to test (0.05 or 0.08)")
    args = parser.parse_args()

    floorsheet = load_floorsheet()
    prices     = load_price_history()
    sectors    = load_sectors()

    thresholds = [args.threshold] if args.threshold else CALM_THRESHOLDS

    all_results = {}
    for threshold in thresholds:
        label = f"calm_{int(threshold*100)}pct"
        log.info("=" * 60)
        log.info("Running detection with calm threshold: %.0f%%", threshold * 100)

        signals = detect(floorsheet, prices, sectors,
                         calm_threshold=threshold,
                         from_date=args.from_date)

        summary = summarise(signals, label)
        all_results[label] = {
            "summary":  summary,
            "signals":  signals if not args.dry_run else signals[:20],
        }

        log.info("Detected %d accumulation signals", summary["n"])
        log.info("  Unique symbols:      %d",   summary["unique_symbols"])
        log.info("  Momentum triggered:  %d (%.1f%%)",
                 summary["momentum_triggered"]["count"],
                 summary["momentum_triggered"]["pct"])
        log.info("  Fwd 20d avg:         %s%%", summary["forward_returns"]["fwd_20d"]["avg"])
        log.info("  Fwd 20d median:      %s%%", summary["forward_returns"]["fwd_20d"]["median"])
        log.info("  Fwd 20d pct pos:     %s%%", summary["forward_returns"]["fwd_20d"]["pct_positive"])
        log.info("  High-score (≥5) n:   %d",   summary["high_score_subset"]["n"])
        log.info("  High-score fwd20 pos: %s%%", summary["high_score_subset"]["fwd_20d_pct_positive"])

    if args.dry_run:
        log.info("Dry run — showing sample signals only, not writing full output")
        print(json.dumps(
            {k: {"summary": v["summary"], "sample": v["signals"][:5]}
             for k, v in all_results.items()},
            indent=2
        ))
        return

    out_dir  = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "accumulation_signals.json"

    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "start_date":        START_DATE,
                "window":            WINDOW,
                "min_score":         MIN_ACC_SCORE,
                "momentum_trigger":  MOMENTUM_TRIGGER,
                "forward_windows":   FORWARD_WINDOWS,
                "baseline_bp":       BASELINE_BP,
                "baseline_bc":       BASELINE_BC,
            },
            "results": all_results,
        }, f, indent=2)

    log.info("Written to %s", out_path)

    # Print comparison table
    print("\n" + "=" * 65)
    print("THRESHOLD COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<35} {'5% calm':>12} {'8% calm':>12}")
    print("-" * 65)
    for metric, keys in [
        ("Total signals",          ["n"]),
        ("Unique symbols",         ["unique_symbols"]),
        ("Momentum triggered %",   ["momentum_triggered", "pct"]),
        ("Avg trigger day",        ["momentum_triggered", "avg_trigger_day"]),
        ("Fwd 20d avg %",          ["forward_returns", "fwd_20d", "avg"]),
        ("Fwd 20d median %",       ["forward_returns", "fwd_20d", "median"]),
        ("Fwd 20d % positive",     ["forward_returns", "fwd_20d", "pct_positive"]),
        ("Fwd 30d avg %",          ["forward_returns", "fwd_30d", "avg"]),
        ("Max 30d avg %",          ["forward_returns", "max_30d", "avg"]),
        ("High-score n",           ["high_score_subset", "n"]),
        ("High-score fwd20 pos%",  ["high_score_subset", "fwd_20d_pct_positive"]),
    ]:
        vals = []
        for label in ["calm_5pct", "calm_8pct"]:
            if label not in all_results:
                vals.append("N/A")
                continue
            d = all_results[label]["summary"]
            for k in keys:
                d = d.get(k, {}) if isinstance(d, dict) else d
            vals.append(str(d))
        print(f"{metric:<35} {vals[0]:>12} {vals[1]:>12}")
    print("=" * 65)


if __name__ == "__main__":
    main()
