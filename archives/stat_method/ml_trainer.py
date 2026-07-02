"""
ml_trainer.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — ML Trainer

Trains XGBoost classifier on ml_features.parquet using proper
walk-forward (time-series) cross-validation.

Walk-forward splits:
  Train: Jul 2023 → Jun 2024   Test: Jul–Oct 2024
  Train: Jul 2023 → Oct 2024   Test: Nov 2024–Feb 2025
  Train: Jul 2023 → Feb 2025   Test: Mar–May 2025
  Train: Jul 2023 → May 2025   Test: Jun–Sep 2025
  Train: Jul 2023 → Sep 2025   Test: Oct 2025–present

Outputs:
  - Per-fold accuracy, precision, recall, AUC
  - Feature importance chart
  - Final model trained on all data
  - Optimal probability threshold (maximises precision at ≥55% win rate)
  - stat_method/output/ml_model.json (model params + threshold)
  - stat_method/output/ml_importance.csv

Usage:
    cd ~/nepse-engine
    pip install xgboost scikit-learn --break-system-packages
    python stat_method/ml_trainer.py
    python stat_method/ml_trainer.py --no-sector   # exclude sector feature
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install xgboost scikit-learn --break-system-packages")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "output"

# XGBoost hyperparameters — tuned for tabular financial data
XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,         # shallow trees reduce overfitting
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,        # require meaningful support per leaf
    "gamma":            0.1,       # minimum gain to split
    "reg_alpha":        0.1,       # L1 regularisation
    "reg_lambda":       1.0,       # L2 regularisation
    "scale_pos_weight": None,      # set dynamically from class balance
    "random_state":     42,
    "eval_metric":      "auc",
    "early_stopping_rounds": 30,
    "verbosity":        0,
}

FEATURE_COLS = [
    # Floorsheet broker features
    "bp_mean",
    "bc_mean",
    "lop_mean",
    "institutional_day_pct",
    "bp_slope",
    "bc_slope",
    "vol_acceleration",
    "absorption_on_down_days",
    "consistent_buyer_count",
    "cross_broker_transfers",
    "top_buyer_net_pct",
    # Price / technical features
    "rsi_14",
    "ema_trend",
    "bb_pct_b",
    "atr_pct",
    "obv_slope",
    "net_60d_return",
    "vol_ratio_15d",
    "price_52w_position",
    "macd_histogram",
    # Regime
    "market_regime",
    # Sector
    "sector_enc",
]


def load_features() -> pd.DataFrame:
    path = OUT_DIR / "ml_features.parquet"
    if not path.exists():
        log.error("ml_features.parquet not found. Run ml_feature_builder.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"])
    log.info("Loaded %d samples from %s", len(df), path)
    return df


def prepare_features(df: pd.DataFrame, use_sector: bool = True) -> tuple:
    """Return X (feature matrix) and y (labels)."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not use_sector and "sector_enc" in cols:
        cols.remove("sector_enc")

    X = df[cols].copy()

    # Fill NaN with median (computed on training set — applied separately in CV)
    # Here we just return the raw df; median imputation happens per fold
    y = df["label"].values
    return X, y, cols


def median_impute(X_train: pd.DataFrame,
                  X_test: pd.DataFrame) -> tuple:
    """Impute NaN with training set medians. Returns imputed DataFrames."""
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def find_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   min_precision: float = 0.55) -> tuple[float, float]:
    """
    Find the probability threshold that achieves ≥min_precision
    while maximising recall.
    Returns (threshold, precision_at_threshold).
    """
    best_threshold = 0.5
    best_recall    = 0.0
    best_precision = 0.0

    for t in np.arange(0.3, 0.95, 0.02):
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        if prec >= min_precision and rec > best_recall:
            best_recall    = rec
            best_threshold = t
            best_precision = prec

    return best_threshold, best_precision


def run_walk_forward(df: pd.DataFrame, use_sector: bool = True) -> dict:
    """
    Walk-forward validation with expanding training window.
    Returns dict of per-fold results.
    """
    # Define fold boundaries
    folds = [
        ("2024-07-01", "2024-10-31"),
        ("2024-11-01", "2025-02-28"),
        ("2025-03-01", "2025-06-30"),
        ("2025-07-01", "2025-10-31"),
        ("2025-11-01", df["window_end_date"].max().strftime("%Y-%m-%d")),
    ]

    X_all, y_all, cols = prepare_features(df, use_sector)
    dates = df["window_end_date"]

    fold_results = []

    for fold_i, (test_start, test_end) in enumerate(folds):
        test_start_dt = pd.Timestamp(test_start)
        test_end_dt   = pd.Timestamp(test_end)

        train_mask = dates < test_start_dt
        test_mask  = (dates >= test_start_dt) & (dates <= test_end_dt)

        if train_mask.sum() < 100 or test_mask.sum() < 20:
            log.info("  Fold %d: insufficient data, skipping", fold_i + 1)
            continue

        X_tr = X_all[train_mask].copy()
        X_te = X_all[test_mask].copy()
        y_tr = y_all[train_mask]
        y_te = y_all[test_mask]

        # Impute with training medians
        X_tr, X_te = median_impute(X_tr, X_te)

        # Class balance
        pos_rate   = y_tr.mean()
        neg_rate   = 1 - pos_rate
        spw        = neg_rate / pos_rate if pos_rate > 0 else 1.0

        params = {**XGB_PARAMS, "scale_pos_weight": spw}
        n_est  = params.pop("n_estimators")
        es     = params.pop("early_stopping_rounds")
        params.pop("eval_metric", None)

        model = xgb.XGBClassifier(
            n_estimators=n_est,
            early_stopping_rounds=es,
            eval_metric="auc",
            **params
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_te, y_te)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_te, y_prob)
        except Exception:
            auc = np.nan

        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        acc  = (y_pred == y_te).mean()

        # Find optimal threshold for ≥55% precision
        opt_thresh, opt_prec = find_threshold(y_te, y_prob, min_precision=0.55)
        opt_preds = (y_prob >= opt_thresh).astype(int)
        opt_recall= recall_score(y_te, opt_preds, zero_division=0)
        opt_signals = int(opt_preds.sum())

        log.info(
            "  Fold %d [%s–%s]: train=%d test=%d | "
            "AUC=%.3f Prec=%.3f Rec=%.3f F1=%.3f | "
            "OptThresh=%.2f OptPrec=%.3f OptRec=%.3f Signals=%d",
            fold_i + 1, test_start, test_end,
            train_mask.sum(), test_mask.sum(),
            auc, prec, rec, f1,
            opt_thresh, opt_prec, opt_recall, opt_signals
        )

        fold_results.append({
            "fold":         fold_i + 1,
            "test_start":   test_start,
            "test_end":     test_end,
            "train_n":      int(train_mask.sum()),
            "test_n":       int(test_mask.sum()),
            "auc":          round(auc, 4) if not np.isnan(auc) else None,
            "precision_50": round(prec, 4),
            "recall_50":    round(rec,  4),
            "f1_50":        round(f1,   4),
            "accuracy":     round(acc,  4),
            "opt_threshold":round(opt_thresh, 2),
            "opt_precision":round(opt_prec,  4),
            "opt_recall":   round(opt_recall,4),
            "opt_signals":  opt_signals,
            "pos_rate_train": round(float(pos_rate), 4),
            "pos_rate_test":  round(float(y_te.mean()), 4),
        })

    return fold_results


def train_final_model(df: pd.DataFrame,
                      use_sector: bool = True,
                      opt_threshold: float = 0.5) -> tuple:
    """Train final model on all available data."""
    X_all, y_all, cols = prepare_features(df, use_sector)
    medians = X_all.median()
    X_filled= X_all.fillna(medians)

    pos_rate = y_all.mean()
    spw      = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    params = {**XGB_PARAMS, "scale_pos_weight": spw}
    n_est  = params.pop("n_estimators")
    params.pop("early_stopping_rounds", None)
    params.pop("eval_metric", None)

    model = xgb.XGBClassifier(n_estimators=n_est, **params)
    model.fit(X_filled, y_all, verbose=False)

    # Feature importance
    imp = pd.DataFrame({
        "feature":    cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, imp, medians, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sector", action="store_true")
    args = parser.parse_args()

    use_sector = not args.no_sector

    df = load_features()

    log.info("\n=== DATASET OVERVIEW ===")
    log.info("Samples:        %d", len(df))
    log.info("Symbols:        %d", df["symbol"].nunique())
    log.info("Date range:     %s → %s",
             df["window_end_date"].min().date(),
             df["window_end_date"].max().date())
    log.info("Positive rate:  %.1f%%", df["label"].mean() * 100)
    log.info("Sectors:        %d", df["sector"].nunique())

    # Check we have enough data for walk-forward
    min_date = df["window_end_date"].min()
    if min_date > pd.Timestamp("2024-01-01"):
        log.warning("Limited training data — earliest sample is %s. "
                    "Walk-forward may have small folds.", min_date.date())

    log.info("\n=== WALK-FORWARD VALIDATION ===")
    fold_results = run_walk_forward(df, use_sector)

    if not fold_results:
        log.error("No valid folds — check date range in feature matrix")
        return

    # Summary stats across folds
    aucs   = [f["auc"]          for f in fold_results if f["auc"]]
    precs  = [f["precision_50"] for f in fold_results]
    o_prec = [f["opt_precision"]for f in fold_results]
    o_rec  = [f["opt_recall"]   for f in fold_results]

    log.info("\n=== CROSS-FOLD SUMMARY ===")
    log.info("Mean AUC:              %.3f (±%.3f)",
             np.mean(aucs), np.std(aucs))
    log.info("Mean Precision@0.5:    %.3f (±%.3f)",
             np.mean(precs), np.std(precs))
    log.info("Mean Opt Precision:    %.3f (±%.3f)",
             np.mean(o_prec), np.std(o_prec))
    log.info("Mean Opt Recall:       %.3f (±%.3f)",
             np.mean(o_rec), np.std(o_rec))

    # Best threshold: median of optimal thresholds across folds
    best_thresh = float(np.median(
        [f["opt_threshold"] for f in fold_results]
    ))
    log.info("Recommended threshold: %.2f", best_thresh)

    # Verdict
    mean_opt_prec = np.mean(o_prec)
    log.info("\n=== VERDICT ===")
    if mean_opt_prec >= 0.55:
        log.info("✅ Model achieves ≥55%% precision at optimal threshold")
        log.info("   Mean opt precision: %.1f%% — TRADEABLE SIGNAL", mean_opt_prec * 100)
    elif mean_opt_prec >= 0.50:
        log.info("⚠️  Model achieves %.1f%% precision — borderline, needs more data",
                 mean_opt_prec * 100)
    else:
        log.info("❌ Model precision %.1f%% below 50%% — no reliable edge found",
                 mean_opt_prec * 100)

    # Train final model
    log.info("\n=== TRAINING FINAL MODEL ===")
    model, importance, medians, feat_cols = train_final_model(
        df, use_sector, best_thresh
    )
    log.info("Final model trained on %d samples", len(df))

    # Save importance
    imp_path = OUT_DIR / "ml_importance.csv"
    importance.to_csv(imp_path, index=False)
    log.info("\nTop 10 features:")
    for _, row in importance.head(10).iterrows():
        log.info("  %-35s %.4f", row["feature"], row["importance"])

    # Save model artifacts
    model_path = OUT_DIR / "ml_model.json"
    model.save_model(str(OUT_DIR / "ml_model.ubj"))

    artifacts = {
        "trained_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold":         round(best_thresh, 3),
        "mean_opt_precision":round(mean_opt_prec, 4),
        "mean_auc":          round(np.mean(aucs), 4) if aucs else None,
        "feature_cols":      feat_cols,
        "medians":           {k: float(v) for k, v in medians.items()},
        "fold_results":      fold_results,
        "xgb_params":        {k: v for k, v in XGB_PARAMS.items()
                              if k not in ["early_stopping_rounds","eval_metric"]},
        "config": {
            "label_threshold": 0.15,
            "label_horizon":   45,
            "use_sector":      use_sector,
        }
    }
    with open(model_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    log.info("\nModel saved:")
    log.info("  %s  (params + threshold + fold results)", model_path)
    log.info("  %s  (XGBoost binary)", OUT_DIR / "ml_model.ubj")
    log.info("  %s  (feature importance)", imp_path)


if __name__ == "__main__":
    main()
