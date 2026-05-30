"""
ml_predictor.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — ML Predictor

Scores all current NEPSE symbols using the trained XGBoost model.
Uses the most recent 30-day window of floorsheet + price data.
Outputs ranked candidates above the optimal probability threshold.

Output: stat_method/output/ml_predictions_YYYY-MM-DD.csv

Usage:
    cd ~/nepse-engine
    python stat_method/ml_predictor.py
    python stat_method/ml_predictor.py --top 20        # show top 20 only
    python stat_method/ml_predictor.py --threshold 0.6 # override threshold
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("Missing xgboost. Run: pip install xgboost --break-system-packages")
    sys.exit(1)

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PREDICT] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "output"

# Import feature computation functions from ml_feature_builder
sys.path.insert(0, str(Path(__file__).parent))
from ml_feature_builder import (
    load_price_history, load_floorsheet_signals,
    load_raw_floorsheet_for_symbol, load_nepse_regime, load_sectors,
    compute_broker_features, compute_price_features,
    WINDOW_DAYS, MIN_FS_DAYS,
)


def load_model_artifacts() -> dict:
    path = OUT_DIR / "ml_model.json"
    if not path.exists():
        log.error("ml_model.json not found. Run ml_trainer.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_xgb_model():
    path = OUT_DIR / "ml_model.ubj"
    if not path.exists():
        log.error("ml_model.ubj not found. Run ml_trainer.py first.")
        sys.exit(1)
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def build_current_features(
    symbols: list[str],
    price_data: dict,
    fs_sig_data: dict,
    regime: pd.Series,
    sectors: dict,
    sector_map: dict,
) -> pd.DataFrame:
    """
    Build features using the most recent 30-day window for each symbol.
    """
    rows = []

    for symbol in sorted(symbols):
        price_df = price_data.get(symbol)
        fs_sig   = fs_sig_data.get(symbol)

        if price_df is None or fs_sig is None:
            continue

        raw_fs = load_raw_floorsheet_for_symbol(symbol)
        if raw_fs.empty:
            continue
        if len(price_df) < WINDOW_DAYS + 5:
            continue

        # Use last WINDOW_DAYS trading days as window
        win_end     = len(price_df) - 1
        win_end_date= price_df["date"].iloc[win_end]
        win_start_date = price_df["date"].iloc[max(0, win_end - WINDOW_DAYS)]

        fs_win = fs_sig[
            (fs_sig["date"] >= win_start_date) &
            (fs_sig["date"] <= win_end_date)
        ]
        if len(fs_win) < MIN_FS_DAYS:
            continue

        raw_win   = raw_fs[
            (raw_fs["date"] >= win_start_date) &
            (raw_fs["date"] <= win_end_date)
        ]
        price_win = price_df.iloc[max(0, win_end - WINDOW_DAYS):win_end + 1]

        broker_feats = compute_broker_features(raw_win, fs_win, price_win)
        price_feats  = compute_price_features(price_df, win_end)

        regime_val = regime.asof(win_end_date) if not regime.empty else np.nan

        sector   = sectors.get(symbol, "Unknown")
        sec_enc  = sector_map.get(sector, -1)

        close_now = price_df["close"].iloc[win_end]

        rows.append({
            "symbol":          symbol,
            "window_end_date": win_end_date,
            "sector":          sector,
            "sector_enc":      sec_enc,
            "market_regime":   int(regime_val) if not pd.isna(regime_val) else -1,
            "close":           round(close_now, 2),
            **broker_feats,
            **price_feats,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",       type=int,   default=30)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override model threshold")
    args = parser.parse_args()

    # Load model artifacts
    artifacts   = load_model_artifacts()
    model       = load_xgb_model()
    feat_cols   = artifacts["feature_cols"]
    medians     = pd.Series(artifacts["medians"])
    threshold   = args.threshold or artifacts["threshold"]
    sector_map  = {}

    # Load sector map from meta if available
    meta_path = OUT_DIR / "ml_features_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            sector_map = meta.get("sector_map", {})

    log.info("Model loaded | Threshold: %.2f | Mean CV precision: %.1f%%",
             threshold, artifacts.get("mean_opt_precision", 0) * 100)

    # Load current data
    price_data  = load_price_history()
    fs_sig_data = load_floorsheet_signals()
    regime      = load_nepse_regime()
    sectors     = load_sectors()

    symbols = sorted(
        set(price_data.keys()) &
        set(fs_sig_data.keys())
    )
    log.info("Symbols with all data: %d", len(symbols))

    # Build current features
    log.info("Building current features...")
    feat_df = build_current_features(
        symbols, price_data, fs_sig_data,
        regime, sectors, sector_map,
    )

    if feat_df.empty:
        log.error("No features built — check data availability")
        return

    log.info("Features built for %d symbols", len(feat_df))

    # Prepare feature matrix
    X = feat_df[[c for c in feat_cols if c in feat_df.columns]].copy()
    # Add missing cols as NaN
    for c in feat_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feat_cols]
    X = X.fillna(medians.reindex(X.columns))

    # Predict
    probas = model.predict_proba(X)[:, 1]
    feat_df["probability"] = probas
    feat_df["signal"]      = (probas >= threshold).astype(int)

    # Sort by probability
    results = feat_df.sort_values("probability", ascending=False)

    # All signals above threshold
    signals = results[results["signal"] == 1].copy()

    log.info("\n=== PREDICTION RESULTS ===")
    log.info("Total symbols scored: %d", len(results))
    log.info("Signals above threshold (%.2f): %d", threshold, len(signals))
    log.info("Market regime today: %s",
             "BULL" if results["market_regime"].iloc[0] == 1 else "BEAR"
             if len(results) > 0 else "UNKNOWN")

    display_cols = [
        "symbol", "sector", "close", "probability",
        "absorption_on_down_days", "consistent_buyer_count",
        "bp_slope", "bc_slope", "institutional_day_pct",
        "rsi_14", "ema_trend", "net_60d_return",
        "market_regime", "window_end_date",
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    if len(signals) > 0:
        log.info("\nTop signals:")
        top = signals.head(args.top)[display_cols]
        print(top.to_string(index=False))
    else:
        log.info("No signals above threshold today.")
        log.info("\nTop 10 by probability (below threshold):")
        top = results.head(10)[display_cols]
        print(top.to_string(index=False))

    # Save predictions
    today    = date.today().strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"ml_predictions_{today}.csv"
    results[display_cols + ["signal"]].to_csv(out_path, index=False)
    log.info("\nPredictions saved to %s", out_path)

    # Print regime warning
    if len(results) > 0 and results["market_regime"].iloc[0] == 0:
        log.warning(
            "⚠️  BEAR REGIME: NEPSE below 60d MA. "
            "Model performance is lower in bear markets. "
            "Consider raising threshold or reducing position size."
        )

    # Summary stats on signals
    if len(signals) > 0:
        log.info("\nSignal breakdown:")
        log.info("  By sector:")
        sec_counts = signals["sector"].value_counts()
        for sec, cnt in sec_counts.items():
            log.info("    %-30s %d", sec, cnt)

        log.info("\n  Avg absorption on down days: %.2f",
                 signals["absorption_on_down_days"].mean()
                 if "absorption_on_down_days" in signals else np.nan)
        log.info("  Avg consistent buyers:       %.1f",
                 signals["consistent_buyer_count"].mean()
                 if "consistent_buyer_count" in signals else np.nan)
        log.info("  Avg bp_slope:                %.5f",
                 signals["bp_slope"].mean()
                 if "bp_slope" in signals else np.nan)
        log.info("  Avg RSI:                     %.1f",
                 signals["rsi_14"].mean()
                 if "rsi_14" in signals else np.nan)


if __name__ == "__main__":
    main()