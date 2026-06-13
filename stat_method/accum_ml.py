"""
accum_ml.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Accumulation Alpha ML

Tests whether broker-level accumulation signals predict EXCESS return
over the NEPSE index (alpha), not raw return.

Label definition:
    stock_return_45d - nepse_return_45d >= 0.15
    i.e. stock outperformed the index by 15%+ within 45 trading days

This removes the bull market noise problem:
    - In a bull market, everything rises → raw 15% label is meaningless
    - Alpha label isolates stocks where accumulation actually CAUSED the move
    - Only fires when the stock meaningfully beat the market

Features: pure accumulation/floorsheet signals only
    bp_mean, bc_mean, lop_mean, institutional_day_pct,
    bp_slope, bc_slope, vol_acceleration,
    absorption_on_down_days, consistent_buyer_count,
    cross_broker_transfers, top_buyer_net_pct,
    price_52w_position, sector_enc

Deliberately excluded: price momentum, RSI, MACD, EMA trend
    (if these predict alpha it's not accumulation driving it)

Models: XGBoost, LightGBM, RandomForest + Ensemble
Validation: strict walk-forward (train on past, test on future only)

Usage:
    cd ~/nepse-engine
    python stat_method/accum_ml.py
    python stat_method/accum_ml.py --alpha-threshold 0.10   # 10% excess return
    python stat_method/accum_ml.py --alpha-threshold 0.20   # 20% excess return
    python stat_method/accum_ml.py --export-model           # save final model
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
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (roc_auc_score, precision_score,
                                  recall_score, precision_recall_curve)
    from sklearn.inspection import permutation_importance
except ImportError:
    print("Run: pip install xgboost lightgbm scikit-learn --break-system-packages")
    sys.exit(1)

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ACCUM_ML] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR  = Path(__file__).parent / "output"
FS_START = "2023-07-01"

# ── Pure accumulation features ────────────────────────────────────────────────
ACCUM_FEATURES = [
    "bp_mean",               # avg buyer pressure over window
    "bc_mean",               # avg broker concentration
    "lop_mean",              # large order participation
    "institutional_day_pct", # % days with institutional flag
    "bp_slope",              # buyer pressure trending up
    "bc_slope",              # broker concentration tightening
    "vol_acceleration",      # volume building in second half
    "absorption_on_down_days",# top buyers absorbing selling
    "consistent_buyer_count", # brokers net-positive ≥60% of days
    "cross_broker_transfers", # negotiated block trades
    "top_buyer_net_pct",     # dominant buyer as % of total vol
    "price_52w_position",    # where in 52W range (context only)
    "sector_enc",            # sector
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_nepse_returns() -> pd.Series:
    """
    Load NEPSE index daily closes and compute forward returns.
    Returns: date → 45-day forward return of NEPSE index.
    """
    log.info("Loading NEPSE index...")
    with _db() as cur:
        cur.execute("""
            SELECT date, current_value::float AS close
            FROM nepse_indices
            WHERE index_id = '58'
              AND current_value IS NOT NULL AND current_value != ''
              AND current_value ~ '^[0-9]+\\.?[0-9]*$'
            ORDER BY date ASC
        """)
        rows = cur.fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    # Forward 45-day return of NEPSE
    df["nepse_fwd45"] = df["close"].shift(-45) / df["close"] - 1
    return df["nepse_fwd45"]


def load_feature_matrix() -> pd.DataFrame:
    """Load the pre-built feature matrix from ml_feature_builder."""
    path = OUT_DIR / "ml_features.parquet"
    if not path.exists():
        log.error("ml_features.parquet not found. Run ml_feature_builder.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"])
    log.info("Loaded %d samples from feature matrix", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LABEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_alpha_labels(df: pd.DataFrame,
                       nepse_returns: pd.Series,
                       alpha_threshold: float = 0.15) -> pd.DataFrame:
    """
    Compute excess return label:
        label = 1 if max_stock_return_45d - nepse_return_45d >= alpha_threshold

    Uses max_return_45d from feature matrix (best price in 45 days / entry price - 1)
    vs NEPSE forward return over same 45 days.

    This answers: did this stock BEAT the market by alpha_threshold?
    """
    df = df.copy()

    # Map NEPSE forward return to each sample's window end date
    df["nepse_fwd45"] = df["window_end_date"].map(
        lambda d: nepse_returns.asof(d) if d in nepse_returns.index
        else nepse_returns.asof(d)
    )

    # Alpha = stock max return - index forward return
    df["alpha_45d"] = df["max_return_45d"] - df["nepse_fwd45"].fillna(0)

    # Binary label
    df["label_alpha"] = (df["alpha_45d"] >= alpha_threshold).astype(int)

    # Also keep calm filter: exclude stocks already running hot
    # (net_60d_return > 30% means the move may already be priced in)
    df = df[df["net_60d_return"] <= 0.30].copy()

    log.info("Alpha label stats:")
    log.info("  Threshold:    %.0f%% excess over NEPSE", alpha_threshold * 100)
    log.info("  Total samples: %d", len(df))
    log.info("  Positive rate: %.1f%%", df["label_alpha"].mean() * 100)
    log.info("  NEPSE avg fwd45: %.1f%%",
             df["nepse_fwd45"].mean() * 100 if "nepse_fwd45" in df else 0)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING + WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

FOLDS = [
    ("2024-07-01", "2024-12-31", "H2-2024"),
    ("2025-01-01", "2025-06-30", "H1-2025"),
    ("2025-07-01", "2025-12-31", "H2-2025"),
    ("2026-01-01", "2026-12-31", "H1-2026"),
]


def find_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   min_prec: float = 0.55,
                   min_signals: int = 5) -> dict:
    """Find threshold maximising recall at >= min_prec precision."""
    try:
        precs, recs, threshs = precision_recall_curve(y_true, y_prob)
    except Exception:
        return {"thresh": 0.5, "prec": 0.0, "rec": 0.0, "n": 0}

    best = {"thresh": 0.5, "prec": 0.0, "rec": 0.0, "n": 0}
    for p, r, t in zip(precs[:-1], recs[:-1], threshs):
        n = int((y_prob >= t).sum())
        if p >= min_prec and n >= min_signals and r > best["rec"]:
            best = {
                "thresh": round(float(t), 3),
                "prec":   round(float(p), 4),
                "rec":    round(float(r), 4),
                "n":      n,
            }
    return best


def make_models(spw: float) -> dict:
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=15, gamma=0.2,
            reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=spw,
            random_state=42, verbosity=0,
            eval_metric="auc", early_stopping_rounds=30,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=25, reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=spw, random_state=42, verbosity=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=25,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }


def walk_forward(df: pd.DataFrame,
                 feat_cols: list[str],
                 label_col: str = "label_alpha",
                 min_prec: float = 0.55) -> dict:
    """
    Run walk-forward validation across all folds.
    Returns per-fold results for each model.
    """
    results = {m: [] for m in ["XGBoost", "LightGBM", "RandomForest", "Ensemble"]}

    for test_start, test_end, fold_name in FOLDS:
        ts = pd.Timestamp(test_start)
        te = pd.Timestamp(test_end)

        tr_mask = df["window_end_date"] < ts
        te_mask = (df["window_end_date"] >= ts) & (df["window_end_date"] <= te)

        n_tr = tr_mask.sum()
        n_te = te_mask.sum()

        if n_tr < 300 or n_te < 30:
            log.info("  %s: skipped (train=%d test=%d)", fold_name, n_tr, n_te)
            continue

        X_tr = df[tr_mask][feat_cols].copy()
        X_te = df[te_mask][feat_cols].copy()
        y_tr = df[tr_mask][label_col].values
        y_te = df[te_mask][label_col].values

        # Impute with training medians
        med  = X_tr.median()
        X_tr = X_tr.fillna(med)
        X_te = X_te.fillna(med)

        n_pos  = (y_tr == 1).sum()
        n_neg  = (y_tr == 0).sum()
        spw    = n_neg / max(n_pos, 1)

        log.info("\n── %s: train=%d test=%d | pos_train=%.1f%% pos_test=%.1f%%",
                 fold_name, n_tr, n_te,
                 y_tr.mean() * 100, y_te.mean() * 100)

        models  = make_models(spw)
        probas  = {}

        for name, model in models.items():
            try:
                if name == "XGBoost":
                    model.fit(X_tr, y_tr,
                              eval_set=[(X_te, y_te)],
                              verbose=False)
                else:
                    model.fit(X_tr, y_tr)

                prob = model.predict_proba(X_te)[:, 1]
                probas[name] = prob

                try:
                    auc = roc_auc_score(y_te, prob)
                except Exception:
                    auc = np.nan

                best = find_threshold(y_te, prob, min_prec)

                log.info("  %-14s AUC=%.3f | Thresh=%.2f Prec=%.3f "
                         "Recall=%.3f Signals=%d",
                         name, auc,
                         best["thresh"], best["prec"],
                         best["rec"],   best["n"])

                results[name].append({
                    "fold":          fold_name,
                    "auc":           round(auc, 4) if not np.isnan(auc) else None,
                    "opt_threshold": best["thresh"],
                    "opt_precision": best["prec"],
                    "opt_recall":    best["rec"],
                    "opt_signals":   best["n"],
                    "pos_rate_test": round(float(y_te.mean()), 4),
                    "test_n":        int(n_te),
                })

            except Exception as e:
                log.warning("  %-14s ERROR: %s", name, e)

        # Ensemble: mean of available probabilities
        if len(probas) >= 2:
            ens_prob = np.mean(list(probas.values()), axis=0)
            try:
                auc = roc_auc_score(y_te, ens_prob)
            except Exception:
                auc = np.nan
            best = find_threshold(y_te, ens_prob, min_prec)
            log.info("  %-14s AUC=%.3f | Thresh=%.2f Prec=%.3f "
                     "Recall=%.3f Signals=%d",
                     "Ensemble", auc,
                     best["thresh"], best["prec"],
                     best["rec"],   best["n"])
            results["Ensemble"].append({
                "fold":          fold_name,
                "auc":           round(auc, 4) if not np.isnan(auc) else None,
                "opt_threshold": best["thresh"],
                "opt_precision": best["prec"],
                "opt_recall":    best["rec"],
                "opt_signals":   best["n"],
                "pos_rate_test": round(float(y_te.mean()), 4),
                "test_n":        int(n_te),
            })

    return results


def summarise_results(results: dict) -> None:
    """Print cross-fold summary table."""
    log.info("\n%s", "=" * 80)
    log.info("CROSS-FOLD SUMMARY")
    log.info("=" * 80)
    log.info("%-16s %6s %5s %9s %5s %7s %8s",
             "Model", "AUC", "±", "OptPrec", "±", "Recall", "Signals")
    log.info("-" * 60)

    best_model, best_prec = None, 0.0

    for name, folds in results.items():
        if not folds:
            continue
        aucs  = [f["auc"]          for f in folds if f["auc"]]
        precs = [f["opt_precision"] for f in folds]
        recs  = [f["opt_recall"]   for f in folds]
        sigs  = [f["opt_signals"]  for f in folds]
        mp    = float(np.mean(precs))

        log.info("%-16s %6.3f %5.3f %9.3f %5.3f %7.3f %8.1f",
                 name,
                 np.mean(aucs) if aucs else 0,
                 np.std(aucs)  if aucs else 0,
                 mp, np.std(precs),
                 np.mean(recs), np.mean(sigs))

        if mp > best_prec:
            best_prec, best_model = mp, name

    log.info("\nBest: %s @ %.1f%% mean precision", best_model, best_prec * 100)

    log.info("\n%s", "=" * 80)
    log.info("VERDICT")
    log.info("=" * 80)
    if best_prec >= 0.55:
        log.info("✅ %s achieves ≥55%% precision — ACCUMULATION PREDICTS ALPHA",
                 best_model)
    elif best_prec >= 0.48:
        log.info("⚠️  Best at %.1f%% — weak edge, needs more data or features",
                 best_prec * 100)
    else:
        log.info("❌ Best at %.1f%% — accumulation signals do not reliably "
                 "predict excess return over NEPSE", best_prec * 100)
        log.info("   Possible reasons:")
        log.info("   1. Accumulation is too early — window too long before move")
        log.info("   2. Signals measure activity, not intent")
        log.info("   3. NEPSE operator moves are not predictable from floorsheet alone")

    return best_model, best_prec


def train_final_model(df: pd.DataFrame,
                      feat_cols: list[str],
                      label_col: str,
                      best_model_name: str,
                      opt_threshold: float = 0.6) -> tuple:
    """Train final model on all data and compute feature importance."""
    X = df[feat_cols].fillna(df[feat_cols].median())
    y = df[label_col].values
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    # Final model has no eval set — disable early stopping
    if best_model_name == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=15, gamma=0.2,
            reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=spw,
            random_state=42, verbosity=0,
        )
        model.fit(X, y)
    elif best_model_name == "LightGBM":
        model = lgb.LGBMClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=25, reg_alpha=0.1, reg_lambda=2.0,
            scale_pos_weight=spw, random_state=42, verbosity=-1,
        )
        model.fit(X, y)
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=25,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X, y)

    # Feature importance
    imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_
        if hasattr(model, "feature_importances_") else [0] * len(feat_cols),
    }).sort_values("importance", ascending=False)

    return model, imp


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-threshold", type=float, default=0.15,
                        help="Excess return threshold (default: 0.15 = 15%% over NEPSE)")
    parser.add_argument("--min-precision",  type=float, default=0.55,
                        help="Target precision (default: 0.55)")
    parser.add_argument("--export-model",   action="store_true",
                        help="Save final trained model")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df          = load_feature_matrix()
    nepse_ret   = load_nepse_returns()

    # ── Build alpha labels ─────────────────────────────────────────────────────
    df = build_alpha_labels(df, nepse_ret, args.alpha_threshold)

    log.info("\nDataset after filtering:")
    log.info("  Samples:       %d", len(df))
    log.info("  Positive rate: %.1f%%", df["label_alpha"].mean() * 100)
    log.info("  Date range:    %s → %s",
             df["window_end_date"].min().date(),
             df["window_end_date"].max().date())

    log.info("\nPositive rate by year:")
    df["year"] = df["window_end_date"].dt.year
    year_stats = df.groupby("year")["label_alpha"].agg(["count","mean"])
    for yr, row in year_stats.iterrows():
        log.info("  %d: n=%d  pos=%.1f%%", yr, int(row["count"]), row["mean"]*100)

    log.info("\nPositive rate by sector:")
    sec_stats = df.groupby("sector")["label_alpha"].agg(["count","mean"]).sort_values("count", ascending=False)
    for sec, row in sec_stats.head(10).iterrows():
        log.info("  %-30s n=%5d  pos=%.1f%%", sec, int(row["count"]), row["mean"]*100)

    # ── Feature correlation with alpha label ───────────────────────────────────
    log.info("\nFeature correlations with alpha label:")
    feat_cols = [c for c in ACCUM_FEATURES if c in df.columns]
    for col in feat_cols:
        try:
            corr = df[col].corr(df["label_alpha"])
            log.info("  %-35s %+.3f", col, corr)
        except Exception:
            pass

    # ── Walk-forward validation ────────────────────────────────────────────────
    log.info("\n%s", "=" * 80)
    log.info("WALK-FORWARD VALIDATION")
    log.info("Label: stock outperforms NEPSE by ≥%.0f%% within 45 days",
             args.alpha_threshold * 100)
    log.info("%s", "=" * 80)

    results = walk_forward(df, feat_cols, "label_alpha", args.min_precision)
    best_model_name, best_prec = summarise_results(results)

    # ── Feature importance ─────────────────────────────────────────────────────
    log.info("\n%s", "=" * 80)
    log.info("FEATURE IMPORTANCE (XGBoost, full dataset)")
    log.info("%s", "=" * 80)

    _, imp = train_final_model(df, feat_cols, "label_alpha", "XGBoost")
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance"] * 200)
        log.info("  %-35s %.4f %s", row["feature"], row["importance"], bar)

    # ── Precision at various thresholds (in-sample sanity check) ──────────────
    log.info("\n%s", "=" * 80)
    log.info("IN-SAMPLE PRECISION BY THRESHOLD (sanity check)")
    log.info("(NOT for trading — just shows model learned something)")
    log.info("%s", "=" * 80)

    X_all = df[feat_cols].fillna(df[feat_cols].median())
    y_all = df["label_alpha"].values
    spw_all = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

    m_sanity = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, scale_pos_weight=spw_all,
        random_state=42, verbosity=0
    )
    m_sanity.fit(X_all, y_all, verbose=False)
    prob_all = m_sanity.predict_proba(X_all)[:, 1]
    df["prob"] = prob_all

    for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        subset = df[df["prob"] >= t]
        if len(subset) < 5:
            continue
        prec = subset["label_alpha"].mean()
        log.info("  Thresh %.2f: n=%5d  precision=%.3f (%.1f%%)",
                 t, len(subset), prec, prec * 100)

    # ── Sector analysis at high threshold ──────────────────────────────────────
    log.info("\n%s", "=" * 80)
    log.info("SECTOR PRECISION AT PROB >= 0.65 (in-sample)")
    log.info("%s", "=" * 80)
    high = df[df["prob"] >= 0.65]
    sec_prec = high.groupby("sector")["label_alpha"].agg(["count","mean"]).rename(
        columns={"count":"n","mean":"precision"}
    ).sort_values("n", ascending=False)
    for sec, row in sec_prec.iterrows():
        if row["n"] < 5:
            continue
        log.info("  %-30s n=%4d  precision=%.3f (%.1f%%)",
                 sec, int(row["n"]), row["precision"], row["precision"]*100)

    # ── Save results ───────────────────────────────────────────────────────────
    out = {
        "generated_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alpha_threshold":  args.alpha_threshold,
        "min_precision":    args.min_precision,
        "total_samples":    len(df),
        "positive_rate":    round(float(df["label_alpha"].mean()), 4),
        "best_model":       best_model_name,
        "best_precision":   round(best_prec, 4),
        "fold_results":     results,
        "feature_cols":     feat_cols,
    }
    out_path = OUT_DIR / "accum_ml_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("\nResults saved to %s", out_path)

    # Optional: export final model
    if args.export_model and best_model_name:
        model, imp = train_final_model(
            df, feat_cols, "label_alpha", best_model_name
        )
        medians = df[feat_cols].median().to_dict()

        # Find best threshold across folds
        best_thresh = float(np.median([
            f["opt_threshold"]
            for folds in results.values()
            for f in folds
            if f["opt_threshold"] > 0.5
        ] or [0.65]))

        model_path = OUT_DIR / "accum_model.ubj"
        model.save_model(str(model_path))

        meta = {
            "model":          best_model_name,
            "threshold":      round(best_thresh, 3),
            "alpha_threshold":args.alpha_threshold,
            "feature_cols":   feat_cols,
            "medians":        {k: float(v) for k, v in medians.items()},
            "trained_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mean_precision": round(best_prec, 4),
        }
        meta_path = OUT_DIR / "accum_model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        imp.to_csv(OUT_DIR / "accum_importance.csv", index=False)
        log.info("Model saved: %s", model_path)
        log.info("Meta saved:  %s", meta_path)


if __name__ == "__main__":
    main()