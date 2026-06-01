# -*- coding: utf-8 -*-
"""
spring_loading_precision.py
─────────────────────────────────────────────────────────────────────────────
Computes the TRUE precision and recall of the spring loading pattern by
evaluating ALL symbol × window combinations, not just the 50%+ winners.

This solves the base rate problem:
  - True Positive  (TP): spring loading present → stock gained 50%+
  - False Positive (FP): spring loading present → stock did NOT gain 50%+
  - False Negative (FN): spring loading absent  → stock DID gain 50%+
  - True Negative  (TN): spring loading absent  → stock did NOT gain 50%+

Precision = TP / (TP + FP)  ← "of all spring loading signals, how many paid off?"
Recall    = TP / (TP + FN)  ← "of all 50%+ runs, what % had spring loading?"

Spring loading criteria (same as screener):
  1. RSI 28–42
  2. MACD histogram negative but improving 2+ consecutive days
  3. OBV 5-day slope positive
  4. ADX < 25
  5. BB width < 90% of 10-day average
  6. Volume < 85% of 20-day average

Evaluated on FIRST 5 DAYS of each window (pre-entry signal).

Run: python -m analysis.spring_loading_precision
     python -m analysis.spring_loading_precision --threshold 4  (use 4/6 instead of 6/6)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import csv
import os
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta

from config import NST
from sheets import run_raw_sql, get_setting

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE        = "2021-01-01"
WINDOW_SIZE       = 60
MIN_GAIN_PCT      = 50.0
MIN_HISTORY_DAYS  = 60
IPO_EXCLUDE_DAYS  = 60
SIGNAL_DAYS       = 5      # evaluate spring loading on first N days of window

# Spring loading thresholds
RSI_MIN           = 28.0
RSI_MAX           = 42.0
ADX_MAX           = 25.0
MACD_IMPROVE_DAYS = 2
OBV_SLOPE_DAYS    = 5
BB_SQUEEZE_RATIO  = 0.90
VOLUME_RATIO_MAX  = 0.85

# Score threshold for "signal present" — read from CLI or default 5
args = sys.argv[1:]
SCORE_THRESHOLD = 5  # 5/6 or 6/6 for "signal present"
for a in args:
    if a.startswith("--threshold"):
        try:
            SCORE_THRESHOLD = int(a.split("=")[1]) if "=" in a else int(args[args.index(a)+1])
        except Exception:
            pass

OUTPUT_DIR = os.path.expanduser("~/nepse-engine/output/spring_precision")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def _load_all_price_history() -> dict[str, pd.DataFrame]:
    """Load all price history since START_DATE. Returns {symbol: DataFrame}."""
    log.info("Loading price_history since %s...", START_DATE)
    rows = run_raw_sql(
        """
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s
        ORDER BY symbol, date ASC
        """,
        (START_DATE,),
    )
    log.info("Loaded %d rows", len(rows))

    def _f(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    sym_data: dict[str, list] = {}
    for r in rows:
        sym   = str(r["symbol"]).upper()
        close = _f(r["close"]) or _f(r["ltp"])
        if close is None:
            continue
        sym_data.setdefault(sym, []).append({
            "date":   r["date"],
            "open":   _f(r["open"]) or close,
            "high":   _f(r["high"]) or close,
            "low":    _f(r["low"])  or close,
            "close":  close,
            "volume": _f(r["volume"]) or 0.0,
        })

    result = {}
    for sym, records in sym_data.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_HISTORY_DAYS:
            result[sym] = df

    log.info("Symbols with %d+ days history: %d", MIN_HISTORY_DAYS, len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SPRING LOADING SCORE ON FIRST N DAYS
# ══════════════════════════════════════════════════════════════════════════════

def _spring_score_on_slice(df: pd.DataFrame) -> int:
    """
    Compute spring loading score (0-6) on a DataFrame slice.
    df should be the full history UP TO the end of the signal window
    so indicators are computed correctly.
    Returns only the score for the last row (the signal day).
    """
    score = 0
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # 1. RSI
        rsi = ta.rsi(close, length=14)
        if rsi is not None and len(rsi.dropna()) > 0:
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            if rsi_val and RSI_MIN <= rsi_val <= RSI_MAX:
                score += 1

        # 2. MACD histogram negative but improving
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            hist = macd.get("MACDh_12_26_9")
            if hist is not None and len(hist.dropna()) >= MACD_IMPROVE_DAYS + 1:
                recent = hist.dropna().iloc[-(MACD_IMPROVE_DAYS + 1):]
                if len(recent) >= MACD_IMPROVE_DAYS + 1:
                    is_negative  = float(recent.iloc[-1]) < 0
                    is_improving = all(
                        recent.iloc[i] > recent.iloc[i - 1]
                        for i in range(1, len(recent))
                    )
                    if is_negative and is_improving:
                        score += 1

        # 3. OBV 5-day slope positive
        obv = ta.obv(close, volume)
        if obv is not None and len(obv.dropna()) >= OBV_SLOPE_DAYS + 1:
            obv_clean = obv.dropna()
            obv_start = float(obv_clean.iloc[-OBV_SLOPE_DAYS - 1])
            obv_end   = float(obv_clean.iloc[-1])
            slope     = (obv_end - obv_start) / max(abs(obv_start), 1)
            if slope > 0:
                score += 1

        # 4. ADX < 25
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = adx_df.get("ADX_14")
            if adx_col is not None and len(adx_col.dropna()) > 0:
                adx_val = float(adx_col.iloc[-1]) if not pd.isna(adx_col.iloc[-1]) else None
                if adx_val and adx_val < ADX_MAX:
                    score += 1

        # 5. BB width contracting
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            bb_width = bb.get("BBB_20_2.0")
            if bb_width is not None and len(bb_width.dropna()) >= 10:
                curr = float(bb_width.iloc[-1])
                avg  = float(bb_width.iloc[-10:].mean())
                if curr < avg * BB_SQUEEZE_RATIO:
                    score += 1

        # 6. Volume quiet
        if len(volume) >= 21:
            curr_vol = float(volume.iloc[-1])
            avg_vol  = float(volume.iloc[-21:-1].mean())
            ratio    = curr_vol / avg_vol if avg_vol > 0 else 1.0
            if ratio < VOLUME_RATIO_MAX:
                score += 1

    except Exception as exc:
        log.debug("_spring_score_on_slice failed: %s", exc)

    return score


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_precision_study() -> dict:
    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    log.info("=" * 60)
    log.info("spring_loading_precision — threshold=%d/6", SCORE_THRESHOLD)

    sym_data = _load_all_price_history()
    if not sym_data:
        log.error("No price data")
        return {}

    # Global sorted trading dates
    all_dates = sorted(set(
        d for df in sym_data.values() for d in df["date"].tolist()
    ))
    log.info("Total trading days: %d", len(all_dates))

    # Build windows
    windows = []
    i, w = 0, 1
    while i + WINDOW_SIZE - 1 < len(all_dates):
        windows.append({
            "num":   w,
            "start": all_dates[i],
            "end":   all_dates[i + WINDOW_SIZE - 1],
            "signal_end": all_dates[min(i + SIGNAL_DAYS - 1, len(all_dates) - 1)],
        })
        i += WINDOW_SIZE
        w += 1
    log.info("Total windows: %d", len(windows))

    # First dates for IPO exclusion
    sym_first = {sym: df["date"].min() for sym, df in sym_data.items()}

    # Counters
    TP = 0  # signal present, gained 50%+
    FP = 0  # signal present, did NOT gain 50%+
    FN = 0  # signal absent,  DID gain 50%+
    TN = 0  # signal absent,  did NOT gain 50%+

    all_results = []

    log.info("Evaluating all symbol × window combinations...")
    for win in windows:
        w_num       = win["num"]
        w_start     = win["start"]
        w_end       = win["end"]
        signal_end  = win["signal_end"]

        for sym, df in sym_data.items():
            # IPO exclusion
            first_date = sym_first[sym]
            dates_before = [d for d in all_dates if d <= w_start]
            cutoff = dates_before[-IPO_EXCLUDE_DAYS] if len(dates_before) >= IPO_EXCLUDE_DAYS else all_dates[0]
            if first_date >= cutoff:
                continue

            # Get full window slice
            win_df = df[(df["date"] >= w_start) & (df["date"] <= w_end)]
            if len(win_df) < 10:
                continue

            first_close = float(win_df.iloc[0]["close"])
            last_close  = float(win_df.iloc[-1]["close"])
            if first_close <= 0:
                continue

            gain_pct = (last_close - first_close) / first_close * 100
            gained   = gain_pct >= MIN_GAIN_PCT

            # Compute spring loading score on first SIGNAL_DAYS of window
            # Use full history up to signal_end for accurate indicators
            history_for_signal = df[df["date"] <= signal_end]
            if len(history_for_signal) < 30:
                continue

            score          = _spring_score_on_slice(history_for_signal)
            signal_present = score >= SCORE_THRESHOLD

            # Confusion matrix
            if signal_present and gained:
                TP += 1
                outcome = "TP"
            elif signal_present and not gained:
                FP += 1
                outcome = "FP"
            elif not signal_present and gained:
                FN += 1
                outcome = "FN"
            else:
                TN += 1
                outcome = "TN"

            all_results.append({
                "symbol":         sym,
                "window":         w_num,
                "start":          w_start.strftime("%Y-%m-%d"),
                "end":            w_end.strftime("%Y-%m-%d"),
                "gain_pct":       round(gain_pct, 2),
                "gained_50":      gained,
                "spring_score":   score,
                "signal_present": signal_present,
                "outcome":        outcome,
            })

        if w_num % 3 == 0:
            log.info(
                "Window %d/%d | TP=%d FP=%d FN=%d TN=%d",
                w_num, len(windows), TP, FP, FN, TN,
            )

    # ── Compute metrics ───────────────────────────────────────────────────────
    total    = TP + FP + FN + TN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (TP + TN) / total if total > 0 else 0
    base_rate = (TP + FN) / total if total > 0 else 0  # % of all windows that gained 50%+

    metrics = {
        "threshold":        SCORE_THRESHOLD,
        "total_evaluated":  total,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision":        round(precision * 100, 1),
        "recall":           round(recall    * 100, 1),
        "f1_score":         round(f1        * 100, 1),
        "accuracy":         round(accuracy  * 100, 1),
        "base_rate_pct":    round(base_rate * 100, 2),
        "signal_rate_pct":  round((TP + FP) / total * 100, 2) if total > 0 else 0,
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SPRING LOADING PRECISION STUDY")
    print(f"Signal threshold: {SCORE_THRESHOLD}/6 criteria")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"  True  Positives (TP): {TP:>6}  (signal → gained 50%+)")
    print(f"  False Positives (FP): {FP:>6}  (signal → did NOT gain 50%+)")
    print(f"  False Negatives (FN): {FN:>6}  (no signal → DID gain 50%+)")
    print(f"  True  Negatives (TN): {TN:>6}  (no signal → no 50%+ gain)")
    print(f"  Total evaluated:      {total:>6}")
    print(f"\nMetrics:")
    print(f"  Precision:   {precision*100:.1f}%  (of signals fired, this % led to 50%+ gain)")
    print(f"  Recall:      {recall*100:.1f}%  (of 50%+ runs, this % had the signal)")
    print(f"  F1 Score:    {f1*100:.1f}%")
    print(f"  Accuracy:    {accuracy*100:.1f}%")
    print(f"\nBase rates:")
    print(f"  % of all windows that gained 50%+: {base_rate*100:.2f}%")
    print(f"  % of all windows with signal:       {metrics['signal_rate_pct']:.2f}%")
    print(f"\nInterpretation:")
    if precision >= 0.5:
        print(f"  ✅ Precision {precision*100:.1f}% — signal is USEFUL (beats random)")
    else:
        print(f"  ⚠️  Precision {precision*100:.1f}% — signal fires too often on non-starters")
    lift = precision / base_rate if base_rate > 0 else 0
    print(f"  Lift vs base rate: {lift:.1f}x (signal is {lift:.1f}x better than random pick)")

    # ── Export CSVs ───────────────────────────────────────────────────────────
    # Full results
    results_path = os.path.join(OUTPUT_DIR, "precision_all_results.csv")
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    log.info("Full results saved to %s", results_path)

    # Metrics summary
    metrics_path = os.path.join(OUTPUT_DIR, "precision_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    log.info("Metrics saved to %s", metrics_path)

    # FP analysis — what kinds of stocks fire false signals?
    fp_df = pd.DataFrame([r for r in all_results if r["outcome"] == "FP"])
    if not fp_df.empty:
        fp_path = os.path.join(OUTPUT_DIR, "false_positives.csv")
        fp_df.to_csv(fp_path, index=False)
        log.info("False positives saved to %s (%d rows)", fp_path, len(fp_df))

    # Score distribution
    score_df = pd.DataFrame(all_results)
    if not score_df.empty:
        dist = score_df.groupby("spring_score").agg(
            total=("outcome", "count"),
            gained_50=("gained_50", "sum"),
        ).reset_index()
        dist["precision_at_score"] = (dist["gained_50"] / dist["total"] * 100).round(1)
        print(f"\nPrecision by score level:")
        print(dist.to_string(index=False))
        dist_path = os.path.join(OUTPUT_DIR, "score_distribution.csv")
        dist.to_csv(dist_path, index=False)

    print(f"\nOutput directory: {OUTPUT_DIR}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SPRING_PRECISION] %(levelname)s: %(message)s",
    )
    try:
        from log_config import attach_file_handler
        attach_file_handler(__name__)
    except Exception:
        pass

    run_precision_study()