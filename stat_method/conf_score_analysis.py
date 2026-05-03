# -*- coding: utf-8 -*-
"""
conf_score_analysis.py — NEPSE AI Engine (Enhanced)
====================================================
Runs 6 statistical methods on conf_score vs price returns
for ALL available symbols, fully segmented by sector.

Enhancements (v2):
  - Stationarity check (ADF) before Granger; differencing if needed.
  - Benjamini-Hochberg multiple‑testing correction for regressions.
  - Adaptive threshold sweep (5th–95th percentile).
  - Newey‑West HAC standard errors in per‑symbol regressions.
  - Improved missing‑data handling in volume independence.
  - Richer report with executive summary & warnings.

Methods:
  1. Quintile Sort          — monotonic return pattern per bucket
  2. Information Coefficient — Spearman rank correlation (IC/ICIR)
  3. Granger Causality      — does conf_score lead price or follow it
  4. Panel Regression       — marginal effect with volume control
  5. Threshold Detection    — where is the real breakpoint
  6. Volume Independence    — confirm conf_score is not volume proxy

Output:
  conf_score_results/
    sector_quintile_returns.csv
    ic_timeseries.csv
    sector_ic_summary.csv
    granger_per_symbol.csv
    sector_granger_summary.csv
    per_symbol_regression.csv
    sector_regression_summary.csv
    sector_threshold.csv
    volume_per_symbol.csv
    sector_volume_summary.csv
    report.txt
"""

import os
import sys
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_URL       = os.getenv("DATABASE_URL")
OUTPUT_DIR   = "conf_score_results"
START_DATE   = "2021-01-01"
MIN_ROWS     = 60
FORWARD_DAYS = [1, 3, 5, 10]
QUINTILES    = 5
GRANGER_MAXLAG = 3
HAC_LAGS     = 4               # Newey-West lags for OLS

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        log.error("psycopg2 missing. Run: pip install psycopg2-binary --break-system-packages")
        sys.exit(1)

    query = f"""
        SELECT
            p.date,
            p.symbol,
            COALESCE(s.sectorname, 'Unknown') AS sector,
            NULLIF(p.conf_score, '')::float    AS conf_score,
            NULLIF(p.close, '')::float         AS close,
            NULLIF(p.prev_close, '')::float    AS prev_close,
            NULLIF(p.volume, '')::float        AS volume,
            NULLIF(p.turnover, '')::float      AS turnover
        FROM price_history p
        LEFT JOIN share_sectors s ON s.symbol = p.symbol
        WHERE p.date >= '{START_DATE}'
          AND NULLIF(p.conf_score, '') IS NOT NULL
          AND p.conf_score != '0'
          AND NULLIF(p.close, '') IS NOT NULL
          AND NULLIF(p.prev_close, '') IS NOT NULL
          AND NULLIF(p.prev_close, '')::float > 0
        ORDER BY p.symbol, p.date
    """

    log.info("Loading all symbols from DB (from %s)...", START_DATE)
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["conf_score", "close", "prev_close"])
    df = df[(df["prev_close"] > 0) & (df["close"] > 0)]

    # Exclude bonds/debentures
    import re
    is_equity = ~df["symbol"].apply(
        lambda s: bool(re.search(r'\d{2,}', str(s)[3:])) or
                  any(str(s).upper().endswith(p) for p in
                      ["SY2", "SY3", "EF", "MF1", "MF2", "MF3",
                       "STF", "LTF", "LICF", "HLICF"])
    )
    before = df["symbol"].nunique()
    df = df[is_equity]
    after = df["symbol"].nunique()
    log.info("Filtered bonds/debentures: %d → %d equity symbols", before, after)

    # Same-day return
    df["day_return"] = (df["close"] - df["prev_close"]) / df["prev_close"] * 100

    # Forward returns
    log.info("Computing forward returns...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for fwd in FORWARD_DAYS:
        df[f"fwd_{fwd}d"] = (
            df.groupby("symbol")["close"]
            .transform(lambda x: (x.shift(-fwd) / x - 1) * 100)
        )

    # Drop symbols with too few rows
    counts = df.groupby("symbol").size()
    df = df[df["symbol"].isin(counts[counts >= MIN_ROWS].index)]

    log.info("Final: %d rows | %d symbols | %d sectors",
             len(df), df["symbol"].nunique(), df["sector"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — QUINTILE SORT (unchanged except for adaptive later)
# ─────────────────────────────────────────────────────────────────────────────

def quintile_sort(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Method 1: Quintile Sort...")
    results = []

    def run_q(label, sub):
        try:
            sub = sub.copy()
            sub["q"] = pd.qcut(sub["conf_score"], QUINTILES,
                                labels=False, duplicates="drop") + 1
        except Exception:
            return
        for fwd in FORWARD_DAYS:
            col = f"fwd_{fwd}d"
            s = sub.dropna(subset=[col])
            for q in sorted(s["q"].dropna().unique()):
                qd = s[s["q"] == q]
                ret = qd[col]
                if len(ret) < 5:
                    continue
                results.append({
                    "group": label, "quintile": int(q),
                    "forward_days": fwd, "n": len(ret),
                    "avg_conf": round(qd["conf_score"].mean(), 2),
                    "avg_return": round(ret.mean(), 4),
                    "median_return": round(ret.median(), 4),
                    "win_rate_pct": round((ret > 0).mean() * 100, 2),
                    "std_return": round(ret.std(), 4),
                })

    run_q("ALL_MARKET", df)
    for sector, sdf in df.groupby("sector"):
        if sdf["symbol"].nunique() >= 2:
            run_q(sector, sdf)

    out = pd.DataFrame(results)
    if not out.empty:
        q1 = out[out["quintile"] == 1][["group","forward_days","avg_return"]].rename(
            columns={"avg_return":"q1_ret"})
        q5 = out[out["quintile"] == 5][["group","forward_days","avg_return"]].rename(
            columns={"avg_return":"q5_ret"})
        sp = q1.merge(q5, on=["group","forward_days"], how="inner")
        sp["q5_q1_spread"] = round(sp["q5_ret"] - sp["q1_ret"], 4)
        out = out.merge(sp[["group","forward_days","q5_q1_spread"]],
                        on=["group","forward_days"], how="left")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — INFORMATION COEFFICIENT (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def information_coefficient(df: pd.DataFrame) -> tuple:
    log.info("Method 2: Information Coefficient...")
    ic_rows = []

    def run_ic(label, sub):
        for fwd in FORWARD_DAYS:
            col = f"fwd_{fwd}d"
            s = sub.dropna(subset=[col, "conf_score"])
            for date, ddf in s.groupby("date"):
                if len(ddf) < 5:
                    continue
                try:
                    rho, pval = spearmanr(ddf["conf_score"], ddf[col])
                    ic_rows.append({
                        "group": label, "date": date,
                        "forward_days": fwd,
                        "ic": round(rho, 4),
                        "pval": round(pval, 4),
                        "n_stocks": len(ddf),
                    })
                except Exception:
                    pass

    run_ic("ALL_MARKET", df)
    for sector, sdf in df.groupby("sector"):
        if sdf["symbol"].nunique() >= 3:
            run_ic(sector, sdf)

    ic_ts = pd.DataFrame(ic_rows)
    summary_rows = []

    for (group, fwd), gdf in ic_ts.groupby(["group","forward_days"]):
        ic_vals = gdf["ic"].dropna()
        if len(ic_vals) < 10:
            continue
        mean_ic = ic_vals.mean()
        std_ic  = ic_vals.std()
        icir    = mean_ic / std_ic if std_ic > 0 else 0
        t_stat  = mean_ic / (std_ic / np.sqrt(len(ic_vals))) if std_ic > 0 else 0
        summary_rows.append({
            "group": group, "forward_days": fwd,
            "mean_ic": round(mean_ic, 4),
            "std_ic":  round(std_ic, 4),
            "icir":    round(icir, 4),
            "t_stat":  round(t_stat, 4),
            "hit_rate_pct": round((ic_vals > 0).mean() * 100, 2),
            "n_days":  len(ic_vals),
            "signal_strength": (
                "STRONG"   if abs(mean_ic) > 0.05 and abs(icir) > 0.5
                else "MODERATE" if abs(mean_ic) > 0.02
                else "WEAK"
            ),
        })

    return ic_ts, pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — GRANGER CAUSALITY (with stationarity checks)
# ─────────────────────────────────────────────────────────────────────────────

def check_stationarity(series, series_name, symbol):
    """Return (is_stationary, differenced_series) after ADF test."""
    clean = series.dropna()
    if len(clean) < 30:
        return False, None
    try:
        p = adfuller(clean, autolag="AIC")[1]
    except:
        return False, None
    if p <= 0.05:
        return True, clean
    else:
        # Try differencing
        diff = clean.diff().dropna()
        if len(diff) < 30:
            return False, None
        try:
            p_diff = adfuller(diff, autolag="AIC")[1]
        except:
            return False, None
        if p_diff <= 0.05:
            log.debug("  %s: %s non-stationary -> differenced", symbol, series_name)
            return True, diff
        return False, None

def granger_causality(df: pd.DataFrame) -> tuple:
    log.info("Method 3: Granger Causality (per symbol, with stationarity)...")

    per_symbol = []
    symbols = df["symbol"].unique()

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            log.info("  Granger: %d / %d", i, len(symbols))
        sdf = df[df["symbol"] == symbol].sort_values("date").copy()
        sdf = sdf.dropna(subset=["conf_score","day_return"])
        if len(sdf) < MIN_ROWS:
            continue
        sector = sdf["sector"].iloc[0]

        # Check stationarity
        stat_conf, conf_series = check_stationarity(sdf["conf_score"], "conf_score", symbol)
        stat_ret, ret_series   = check_stationarity(sdf["day_return"], "day_return", symbol)

        if not stat_conf or not stat_ret:
            continue   # skip if not stationary even after differencing

        # Align the transformed series
        # For Granger we need a common DataFrame with these two columns
        if conf_series.index.equals(ret_series.index):
            gc_df = pd.DataFrame({"conf_score": conf_series, "day_return": ret_series})
        else:
            # Merge on date index if they differ due to differencing shift
            merged = pd.concat([conf_series.rename("conf_score"),
                                ret_series.rename("day_return")], axis=1).dropna()
            if len(merged) < 30:
                continue
            gc_df = merged

        def gc(y_col, x_col):
            try:
                data = gc_df[[y_col, x_col]].dropna()
                if len(data) < 30:
                    return None, None
                res = grangercausalitytests(data, maxlag=GRANGER_MAXLAG, verbose=False)
                pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, GRANGER_MAXLAG+1)]
                bp = min(pvals)
                bl = pvals.index(bp) + 1
                return round(bp, 4), bl
            except Exception:
                return None, None

        p_fwd, l_fwd = gc("day_return",  "conf_score")
        p_rev, l_rev = gc("conf_score",  "day_return")

        sl = p_fwd is not None and p_fwd < 0.05
        pl = p_rev is not None and p_rev < 0.05

        direction = (
            "SCORE_LEADS_PRICE"  if sl and not pl else
            "PRICE_LEADS_SCORE"  if pl and not sl else
            "BIDIRECTIONAL"      if sl and pl else
            "NO_CAUSALITY"
        )

        per_symbol.append({
            "symbol": symbol, "sector": sector, "n": len(sdf),
            "score_to_price_pval": p_fwd, "score_to_price_lag": l_fwd,
            "price_to_score_pval": p_rev, "price_to_score_lag": l_rev,
            "direction": direction,
            "conf_differenced": not stat_conf or (conf_series is not sdf["conf_score"]),
            "ret_differenced": not stat_ret or (ret_series is not sdf["day_return"]),
        })

    per_sym_df = pd.DataFrame(per_symbol)
    sector_rows = []

    if not per_sym_df.empty:
        for sector, sdf in per_sym_df.groupby("sector"):
            c = sdf["direction"].value_counts().to_dict()
            n = len(sdf)
            psl = c.get("SCORE_LEADS_PRICE", 0)
            ppl = c.get("PRICE_LEADS_SCORE", 0)
            sector_rows.append({
                "sector": sector, "n_symbols": n,
                "score_leads_price": psl,
                "price_leads_score": ppl,
                "bidirectional": c.get("BIDIRECTIONAL", 0),
                "no_causality":  c.get("NO_CAUSALITY",  0),
                "pct_score_leads": round(psl / n * 100, 1) if n else 0,
                "pct_price_leads": round(ppl / n * 100, 1) if n else 0,
                "verdict": "LEADING" if psl > ppl else "LAGGING",
            })

    return per_sym_df, pd.DataFrame(sector_rows)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4 — PER-SYMBOL REGRESSION (with Newey‑West & multiple testing)
# ─────────────────────────────────────────────────────────────────────────────

def benjamini_hochberg(pvals: np.ndarray, alpha=0.05) -> np.ndarray:
    """Return array of booleans (True if significant after BH correction)."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    ranks = np.argsort(np.argsort(pvals)) + 1
    thresholds = (ranks / n) * alpha
    return pvals <= thresholds

def per_symbol_regression(df: pd.DataFrame) -> tuple:
    log.info("Method 4: Per-Symbol Regression (Newey‑West & adj. p-values)...")
    from scipy.stats import t as t_dist

    per_symbol = []
    symbols = df["symbol"].unique()

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            log.info("  Regression: %d / %d", i, len(symbols))
        sdf = df[df["symbol"] == symbol].sort_values("date").copy()
        sector = sdf["sector"].iloc[0]

        for fwd in [1, 5]:
            col = f"fwd_{fwd}d"
            sub = sdf.dropna(subset=[col,"conf_score","volume"]).copy()
            sub = sub[sub["volume"] > 0]
            if len(sub) < MIN_ROWS:
                continue

            sub["log_vol"] = np.log(sub["volume"] + 1)
            X = sm.add_constant(sub[["conf_score", "log_vol"]].values)
            y = sub[col].values
            n = len(y)

            try:
                # Use OLS with HAC standard errors (Newey-West)
                model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
                beta_conf = model.params[1]
                se_conf = model.bse[1]
                t_stat = model.tvalues[1]
                pval = model.pvalues[1]
                r2 = model.rsquared

                corr_ret = spearmanr(sub["conf_score"], sub[col])[0]
                corr_vol = spearmanr(sub["conf_score"], sub["volume"])[0]

                per_symbol.append({
                    "symbol": symbol, "sector": sector,
                    "forward_days": fwd, "n": n,
                    "beta_conf": round(float(beta_conf), 6),
                    "se_conf": round(float(se_conf), 6),
                    "t_stat": round(float(t_stat), 3),
                    "pval": round(float(pval), 6),
                    "r2": round(float(r2), 6),
                    "significant_p05": bool(pval < 0.05),   # raw
                    "corr_conf_return": round(corr_ret, 4),
                    "corr_conf_volume": round(corr_vol, 4),
                    "avg_conf": round(sub["conf_score"].mean(), 2),
                    "avg_return": round(sub[col].mean(), 4),
                })
            except Exception as e:
                log.debug("Regression failed for %s (%d): %s", symbol, fwd, str(e))
                pass

    per_sym_df = pd.DataFrame(per_symbol)

    # Apply BH correction per forward horizon
    if not per_sym_df.empty:
        for fwd in per_sym_df["forward_days"].unique():
            mask = per_sym_df["forward_days"] == fwd
            pvals = per_sym_df.loc[mask, "pval"].values
            adj_sig = benjamini_hochberg(pvals, alpha=0.05)
            per_sym_df.loc[mask, "pval_adj_significant"] = adj_sig
            # Also store adjusted p-values? We can compute them, but simple flag suffices.
            # For completeness, we could compute FDR q-values, but for brevity we'll just mark.
            per_sym_df.loc[mask, "significant_adj"] = adj_sig
    else:
        per_sym_df["significant_adj"] = False

    # Sector summary using adjusted significance
    sector_rows = []
    if not per_sym_df.empty:
        for (sector, fwd), sdf in per_sym_df.groupby(["sector","forward_days"]):
            sig = sdf[sdf["significant_adj"] == True]
            sector_rows.append({
                "sector": sector, "forward_days": fwd,
                "n_symbols": len(sdf),
                "n_significant": len(sig),
                "pct_significant": round(len(sig)/len(sdf)*100, 1) if len(sdf) else 0,
                "avg_beta_conf": round(sdf["beta_conf"].mean(), 6),
                "avg_corr_conf_return": round(sdf["corr_conf_return"].mean(), 4),
                "avg_corr_conf_volume": round(sdf["corr_conf_volume"].mean(), 4),
                "avg_r2": round(sdf["r2"].mean(), 6),
            })

    return per_sym_df, pd.DataFrame(sector_rows)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 5 — ADAPTIVE THRESHOLD DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def threshold_detection(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Method 5: Adaptive Threshold Detection...")
    results = []

    groups = {"ALL_MARKET": df}
    for sector, sdf in df.groupby("sector"):
        if len(sdf) >= MIN_ROWS * 5:
            groups[sector] = sdf

    for group_name, gdf in groups.items():
        for fwd in FORWARD_DAYS:
            col = f"fwd_{fwd}d"
            sub = gdf.dropna(subset=[col,"conf_score"])
            if len(sub) < 100:
                continue

            # Adaptive thresholds: 5th to 95th percentile, step 2
            low = np.percentile(sub["conf_score"], 5)
            high = np.percentile(sub["conf_score"], 95)
            if high - low < 10:
                continue
            thresholds = np.arange(low, high, 2)

            best = {"threshold": None, "spread": -999}

            for thresh in thresholds:
                above = sub[sub["conf_score"] >= thresh][col]
                below = sub[sub["conf_score"] <  thresh][col]
                if len(above) < 30 or len(below) < 30:
                    continue
                spread = above.mean() - below.mean()
                try:
                    _, pval = stats.ttest_ind(above, below, equal_var=False)
                except:
                    pval = 1.0
                if spread > best["spread"]:
                    best = {
                        "threshold": thresh, "spread": spread,
                        "above": round(above.mean(), 4),
                        "below": round(below.mean(), 4),
                        "pval":  round(pval, 4),
                        "n_above": len(above),
                        "n_below": len(below),
                    }

            if best["threshold"] is not None:
                results.append({
                    "group": group_name, "forward_days": fwd,
                    "best_threshold": best["threshold"],
                    "spread_pct": round(best["spread"], 4),
                    "above_return": best["above"],
                    "below_return": best["below"],
                    "ttest_pval": best["pval"],
                    "significant": bool(best["pval"] and best["pval"] < 0.05),
                    "n_above": best["n_above"],
                    "n_below": best["n_below"],
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 6 — VOLUME INDEPENDENCE (improved gap handling)
# ─────────────────────────────────────────────────────────────────────────────

def volume_independence(df: pd.DataFrame) -> tuple:
    log.info("Method 6: Volume Independence (improved gap handling)...")
    per_symbol = []
    symbols = df["symbol"].unique()

    for i, symbol in enumerate(symbols):
        if i % 50 == 0:
            log.info("  Volume: %d / %d", i, len(symbols))
        sdf = df[df["symbol"] == symbol].sort_values("date").copy()
        sdf = sdf.dropna(subset=["conf_score","volume","turnover"])
        sdf = sdf[sdf["volume"] > 0]
        if len(sdf) < MIN_ROWS:
            continue
        sector = sdf["sector"].iloc[0]

        corr_vol = spearmanr(sdf["conf_score"], sdf["volume"])[0]
        corr_to  = spearmanr(sdf["conf_score"], sdf["turnover"])[0]

        # Compute lagged returns using forward-fill to handle missing days
        sdf["day_return_ffill"] = sdf["day_return"].fillna(method="ffill")
        sdf["lag1"] = sdf["day_return_ffill"].shift(1)
        sdf["lag3"] = sdf["day_return_ffill"].rolling(3, min_periods=1).mean().shift(1)
        sub_lag = sdf.dropna(subset=["lag1","lag3"])

        corr_m1 = spearmanr(sub_lag["conf_score"], sub_lag["lag1"])[0] if len(sub_lag) >= 20 else None
        corr_m3 = spearmanr(sub_lag["conf_score"], sub_lag["lag3"])[0] if len(sub_lag) >= 20 else None

        per_symbol.append({
            "symbol": symbol, "sector": sector, "n": len(sdf),
            "corr_conf_volume":   round(corr_vol, 4),
            "corr_conf_turnover": round(corr_to, 4),
            "corr_conf_lag1_return": round(corr_m1, 4) if corr_m1 else None,
            "corr_conf_lag3_return": round(corr_m3, 4) if corr_m3 else None,
            "volume_proxy_risk": (
                "HIGH"   if abs(corr_vol) > 0.5 else
                "MEDIUM" if abs(corr_vol) > 0.3 else "LOW"
            ),
            "independent_signal": bool(abs(corr_vol) < 0.3 and abs(corr_to) < 0.3),
        })

    per_sym_df = pd.DataFrame(per_symbol)
    sector_rows = []

    if not per_sym_df.empty:
        for sector, sdf in per_sym_df.groupby("sector"):
            sector_rows.append({
                "sector": sector,
                "n_symbols": len(sdf),
                "avg_corr_volume":   round(sdf["corr_conf_volume"].mean(), 4),
                "avg_corr_turnover": round(sdf["corr_conf_turnover"].mean(), 4),
                "avg_corr_lag1_ret": round(sdf["corr_conf_lag1_return"].dropna().mean(), 4),
                "pct_high_vol_risk": round((sdf["volume_proxy_risk"]=="HIGH").mean()*100, 1),
                "pct_independent":   round(sdf["independent_signal"].mean()*100, 1),
            })

    return per_sym_df, pd.DataFrame(sector_rows)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER (enhanced)
# ─────────────────────────────────────────────────────────────────────────────

def write_report(quintile_df, ic_summary, granger_sector,
                 reg_sector, threshold_df, vol_sector, df) -> str:
    sep = "=" * 70
    lines = [sep,
             "CONF_SCORE ANALYSIS — NEPSE AI ENGINE (v2)",
             f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"Data from : {START_DATE}",
             f"Symbols   : {df['symbol'].nunique()}",
             f"Sectors   : {df['sector'].nunique()}",
             f"Total rows: {len(df):,}",
             sep]

    # Executive Summary
    lines.append("\n── EXECUTIVE SUMMARY ──")
    if not ic_summary.empty:
        mkt1 = ic_summary[(ic_summary["group"]=="ALL_MARKET") & (ic_summary["forward_days"]==1)]
        if not mkt1.empty:
            ic_v = mkt1.iloc[0]["mean_ic"]
            lines.append(f"Market-wide 1-day IC: {ic_v:.4f}")
            if abs(ic_v) > 0.05:
                lines.append("  → conf_score is a STRONG signal.")
            elif abs(ic_v) > 0.02:
                lines.append("  → conf_score is a MODERATE signal.")
            else:
                lines.append("  → conf_score is a WEAK signal; use cautiously.")

    if not reg_sector.empty:
        d1 = reg_sector[reg_sector["forward_days"]==1]
        if not d1.empty:
            best_sec = d1.loc[d1["avg_corr_conf_return"].idxmax()]
            lines.append(f"Strongest sector by avg correlation: {best_sec['sector']} "
                         f"({best_sec['avg_corr_conf_return']:.4f})")

    # Overlapping window note
    lines.append("\nNote: Forward returns (3/5/10‑day) use overlapping windows. "
                 "Standard errors are corrected via Newey‑West in regressions.")

    # IC
    lines.append("\n── IC / ICIR ──")
    if not ic_summary.empty:
        mkt = ic_summary[ic_summary["group"]=="ALL_MARKET"].sort_values("forward_days")
        lines.append("  Market-wide:")
        for _, r in mkt.iterrows():
            lines.append(f"    {r['forward_days']:>2}d | IC={r['mean_ic']:>7.4f} | "
                         f"ICIR={r['icir']:>6.3f} | t={r['t_stat']:>6.2f} | "
                         f"Hit={r['hit_rate_pct']:>5.1f}% | {r['signal_strength']}")
        lines.append("\n  Per sector (1d):")
        sec = ic_summary[(ic_summary["group"]!="ALL_MARKET") &
                         (ic_summary["forward_days"]==1)].sort_values("mean_ic", ascending=False)
        for _, r in sec.iterrows():
            lines.append(f"    {r['group']:<32} IC={r['mean_ic']:>7.4f} | "
                         f"ICIR={r['icir']:>6.3f} | {r['signal_strength']}")

    # Granger
    lines.append("\n── GRANGER CAUSALITY ──")
    if not granger_sector.empty:
        for _, r in granger_sector.sort_values("pct_score_leads", ascending=False).iterrows():
            lines.append(f"  {r['sector']:<32} leads={r['pct_score_leads']:>5.1f}% | "
                         f"lags={r['pct_price_leads']:>5.1f}% | {r['verdict']}")

    # Regression (using adjusted significance)
    lines.append("\n── REGRESSION BY SECTOR (1d, adj. p-values) ──")
    if not reg_sector.empty:
        d1 = reg_sector[reg_sector["forward_days"]==1].sort_values(
            "avg_corr_conf_return", ascending=False)
        for _, r in d1.iterrows():
            lines.append(f"  {r['sector']:<32} sig={r['pct_significant']:>5.1f}% | "
                         f"corr={r['avg_corr_conf_return']:>7.4f} | "
                         f"beta={r['avg_beta_conf']:>10.6f}")

    # Threshold
    lines.append("\n── THRESHOLD DETECTION (1d) ──")
    if not threshold_df.empty:
        d1 = threshold_df[threshold_df["forward_days"]==1].sort_values(
            "spread_pct", ascending=False)
        for _, r in d1.iterrows():
            sig = "✓" if r["significant"] else "✗"
            lines.append(f"  {r['group']:<32} thresh={r['best_threshold']:>4.1f} | "
                         f"spread={r['spread_pct']:>6.4f}% | "
                         f"above={r['above_return']:>7.4f}% | "
                         f"below={r['below_return']:>7.4f}% | {sig}")

    # Quintile
    lines.append("\n── QUINTILE RETURNS — MARKET-WIDE (1d) ──")
    if not quintile_df.empty:
        mq = quintile_df[(quintile_df["group"]=="ALL_MARKET") &
                         (quintile_df["forward_days"]==1)].sort_values("quintile")
        for _, r in mq.iterrows():
            lines.append(f"  Q{int(r['quintile'])} conf≈{r['avg_conf']:>5.1f} | "
                         f"return={r['avg_return']:>7.4f}% | "
                         f"win={r['win_rate_pct']:>5.1f}%")

    # Volume
    lines.append("\n── VOLUME INDEPENDENCE BY SECTOR ──")
    if not vol_sector.empty:
        for _, r in vol_sector.sort_values("avg_corr_volume").iterrows():
            risk_warn = "⚠️ HIGH VOL PROXY" if r["pct_high_vol_risk"] > 30 else ""
            lines.append(f"  {r['sector']:<32} corr_vol={r['avg_corr_volume']:>7.4f} | "
                         f"corr_mom={r['avg_corr_lag1_ret']:>7.4f} | "
                         f"indep={r['pct_independent']:>5.1f}% {risk_warn}")

    # Conclusion
    lines += ["\n"+sep, "CONCLUSION & RECOMMENDATIONS", sep]
    if not ic_summary.empty:
        m = ic_summary[(ic_summary["group"]=="ALL_MARKET") &
                       (ic_summary["forward_days"]==1)]
        if not m.empty:
            ic_v = m.iloc[0]["mean_ic"]
            ic_i = m.iloc[0]["icir"]
            q = ("STRONG"   if abs(ic_v)>0.05 and abs(ic_i)>0.5 else
                 "MODERATE" if abs(ic_v)>0.02 else "WEAK")
            lines.append(f"Signal quality: {q}")

    if not threshold_df.empty:
        mt = threshold_df[(threshold_df["group"]=="ALL_MARKET") &
                          (threshold_df["forward_days"]==1) &
                          (threshold_df["significant"]==True)]
        if not mt.empty:
            t = mt.iloc[0]["best_threshold"]
            lines.append(f"Best threshold : conf_score >= {t}")
            lines.append(f"  Above {t}   → ADD_TO_REASONING positive weight")
            lines.append(f"  Below {t-10} → REDUCE_CONFIDENCE_BY_15")
            lines.append("  (Consider piecewise non‑linear response across the score range)")

    if not granger_sector.empty:
        leads = (granger_sector["verdict"]=="LEADING").sum()
        lines.append(f"Leading indicator in {leads}/{len(granger_sector)} sectors")

    if not vol_sector.empty:
        risky = vol_sector[vol_sector["pct_high_vol_risk"] > 30]
        if len(risky):
            lines.append(f"WARNING: {len(risky)} sector(s) with >30% high volume-proxy risk: "
                         f"{', '.join(risky['sector'].tolist())}")
            lines.append("Interpret conf_score cautiously there; add volume as a control.")

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("Output directory: %s/", OUTPUT_DIR)

    df = load_data()

    quintile_df               = quintile_sort(df)
    ic_ts, ic_summary         = information_coefficient(df)
    granger_sym, granger_sec  = granger_causality(df)
    reg_sym, reg_sec          = per_symbol_regression(df)
    threshold_df              = threshold_detection(df)
    vol_sym, vol_sec          = volume_independence(df)

    files = {
        "sector_quintile_returns.csv":    quintile_df,
        "ic_timeseries.csv":              ic_ts,
        "sector_ic_summary.csv":          ic_summary,
        "granger_per_symbol.csv":         granger_sym,
        "sector_granger_summary.csv":     granger_sec,
        "per_symbol_regression.csv":      reg_sym,
        "sector_regression_summary.csv":  reg_sec,
        "sector_threshold.csv":           threshold_df,
        "volume_per_symbol.csv":          vol_sym,
        "sector_volume_summary.csv":      vol_sec,
    }

    for fname, frame in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        frame.to_csv(path, index=False)
        log.info("Saved %s (%d rows)", fname, len(frame))

    report = write_report(quintile_df, ic_summary, granger_sec,
                          reg_sec, threshold_df, vol_sec, df)
    rpath = os.path.join(OUTPUT_DIR, "report.txt")
    with open(rpath, "w") as f:
        f.write(report)

    log.info("Report: %s", rpath)
    print("\n" + report)
    log.info("Done. Results in %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()