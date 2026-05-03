# -*- coding: utf-8 -*-
"""
conf_score_analysis.py — NEPSE AI Engine
=========================================
Runs 6 statistical methods on conf_score vs price returns
for each symbol and sector.

Methods:
  1. Quintile Sort          — monotonic return pattern per bucket
  2. Information Coefficient — Spearman rank correlation (IC/ICIR)
  3. Granger Causality      — does conf_score lead price or follow it
  4. Panel Regression       — marginal effect with volume control
  5. Threshold Detection    — where is the real breakpoint
  6. Volume Independence    — confirm conf_score ≠ volume proxy

Output:
  conf_score_results/
    per_symbol_summary.csv     — all metrics per symbol
    sector_summary.csv         — aggregated per sector
    quintile_returns.csv       — bucket return profiles
    ic_timeseries.csv          — daily IC values
    granger_results.csv        — causality test results
    threshold_results.csv      — detected thresholds per sector
    report.txt                 — plain text summary

Run:
  cd ~/nepse-engine
  python conf_score_analysis.py

Requirements:
  pip install pandas numpy scipy statsmodels psycopg2-binary --break-system-packages
"""

import os
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_URL = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
OUTPUT_DIR = "conf_score_results"
START_DATE = "2022-01-01"
MIN_ROWS_PER_SYMBOL = 100
FORWARD_DAYS = [1, 3, 5, 10]
QUINTILES = 5
GRANGER_LAGS = [1, 2, 3, 5]

# Symbols to analyse — top liquid stocks per sector
SYMBOLS = [
    # Commercial Banks
    "KBL", "PRVU", "GBIME", "NABIL", "HBL", "EBL", "NMB",
    # Development Banks
    "JBBL", "LBBL", "SHINE", "MLBL", "MNBBL", "SADBL", "GBBL",
    # Finance
    "MFIL", "NFS", "RLFL", "SFCL", "PROFL", "PFL", "GUFL",
    # Hotels
    "SHL", "CGH", "TRH", "OHL",
    # Hydro Power
    "NGPL", "HDHPC", "API", "NHPC", "AKJCL", "RIDI", "AKPL", "AHPC", "UPPER",
    # Investment
    "NIFRA", "HIDCL", "NRN", "HATHY", "CHDC",
    # Life Insurance
    "NLIC", "SJLIC", "ALICL", "NLICL", "HLI", "RNLI",
    # Manufacturing
    "SHIVM", "GCIL", "HDL", "SARBTM",
    # Microfinance
    "FMDBL", "CBBL", "SKBBL", "RSDC", "NUBL",
    # Mutual Fund
    "PRSF", "NSIF2", "GIBF1", "NICBF", "NMB50",
    # Non Life Insurance
    "IGI", "SGIC", "HEI", "NLG", "NICL", "SPIL",
    # Others
    "NTC", "NRIC", "NRM",
    # Tradings
    "STC",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load price_history + sector from DB."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        log.error("psycopg2 not installed. Run: pip install psycopg2-binary --break-system-packages")
        raise

    sym_list = "', '".join(SYMBOLS)
    query = f"""
        SELECT
            p.date,
            p.symbol,
            s.sectorname as sector,
            NULLIF(p.conf_score, '')::float    as conf_score,
            NULLIF(p.close, '')::float          as close,
            NULLIF(p.prev_close, '')::float     as prev_close,
            NULLIF(p.volume, '')::float         as volume,
            NULLIF(p.turnover, '')::float       as turnover
        FROM price_history p
        LEFT JOIN share_sectors s ON s.symbol = p.symbol
        WHERE p.symbol IN ('{sym_list}')
          AND p.date >= '{START_DATE}'
          AND NULLIF(p.conf_score, '') IS NOT NULL
          AND p.conf_score != '0'
          AND NULLIF(p.close, '') IS NOT NULL
          AND NULLIF(p.prev_close, '') IS NOT NULL
          AND p.prev_close != '0'
        ORDER BY p.symbol, p.date
    """

    log.info("Loading data from DB...")
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
    df = df[df["prev_close"] > 0]

    # Compute day return
    df["day_return"] = (df["close"] - df["prev_close"]) / df["prev_close"] * 100

    # Compute forward returns per symbol
    log.info("Computing forward returns...")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for fwd in FORWARD_DAYS:
        df[f"fwd_{fwd}d"] = (
            df.groupby("symbol")["close"]
            .transform(lambda x: x.shift(-fwd) / x - 1) * 100
        )

    log.info("Loaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — QUINTILE SORT
# ─────────────────────────────────────────────────────────────────────────────

def quintile_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort all stock-days into quintiles by conf_score.
    Compute mean forward returns per quintile per sector.
    """
    log.info("Method 1: Quintile Sort...")
    results = []

    # Market-wide quintiles
    df["quintile"] = pd.qcut(df["conf_score"], QUINTILES, labels=False) + 1

    for fwd in FORWARD_DAYS:
        col = f"fwd_{fwd}d"
        sub = df.dropna(subset=[col])
        for q in range(1, QUINTILES + 1):
            mask = sub["quintile"] == q
            qdata = sub[mask][col]
            results.append({
                "sector": "ALL",
                "quintile": q,
                "forward_days": fwd,
                "n": len(qdata),
                "avg_return": round(qdata.mean(), 4),
                "median_return": round(qdata.median(), 4),
                "win_rate": round((qdata > 0).mean() * 100, 2),
                "avg_conf_score": round(sub[mask]["conf_score"].mean(), 2),
            })

    # Per sector quintiles
    for sector, sdf in df.groupby("sector"):
        if len(sdf) < MIN_ROWS_PER_SYMBOL * 3:
            continue
        try:
            sdf = sdf.copy()
            sdf["sq"] = pd.qcut(sdf["conf_score"], QUINTILES, labels=False) + 1
        except Exception:
            continue
        for fwd in FORWARD_DAYS:
            col = f"fwd_{fwd}d"
            sub = sdf.dropna(subset=[col])
            for q in range(1, QUINTILES + 1):
                mask = sub["sq"] == q
                qdata = sub[mask][col]
                if len(qdata) < 10:
                    continue
                results.append({
                    "sector": sector,
                    "quintile": q,
                    "forward_days": fwd,
                    "n": len(qdata),
                    "avg_return": round(qdata.mean(), 4),
                    "median_return": round(qdata.median(), 4),
                    "win_rate": round((qdata > 0).mean() * 100, 2),
                    "avg_conf_score": round(sub[mask]["conf_score"].mean(), 2),
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — INFORMATION COEFFICIENT
# ─────────────────────────────────────────────────────────────────────────────

def information_coefficient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily cross-sectional Spearman correlation between conf_score and fwd returns.
    Returns IC timeseries and per-symbol IC summary.
    """
    log.info("Method 2: Information Coefficient...")
    ic_rows = []

    for fwd in FORWARD_DAYS:
        col = f"fwd_{fwd}d"
        sub = df.dropna(subset=[col])
        for date, ddf in sub.groupby("date"):
            if len(ddf) < 5:
                continue
            rho, pval = spearmanr(ddf["conf_score"], ddf[col])
            ic_rows.append({
                "date": date,
                "forward_days": fwd,
                "ic": round(rho, 4),
                "pval": round(pval, 4),
                "n_stocks": len(ddf),
            })

    ic_ts = pd.DataFrame(ic_rows)

    # Summary per forward horizon
    summary_rows = []
    for fwd in FORWARD_DAYS:
        sub = ic_ts[ic_ts["forward_days"] == fwd]["ic"].dropna()
        if len(sub) < 10:
            continue
        mean_ic = sub.mean()
        std_ic = sub.std()
        icir = mean_ic / std_ic if std_ic > 0 else 0
        t_stat = mean_ic / (std_ic / np.sqrt(len(sub))) if std_ic > 0 else 0
        summary_rows.append({
            "forward_days": fwd,
            "mean_ic": round(mean_ic, 4),
            "std_ic": round(std_ic, 4),
            "icir": round(icir, 4),
            "t_stat": round(t_stat, 4),
            "hit_rate": round((sub > 0).mean() * 100, 2),
            "n_days": len(sub),
            "signal_strength": (
                "STRONG" if abs(mean_ic) > 0.05 and abs(icir) > 0.5
                else "MODERATE" if abs(mean_ic) > 0.02
                else "WEAK"
            ),
        })

    return ic_ts, pd.DataFrame(summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — GRANGER CAUSALITY
# ─────────────────────────────────────────────────────────────────────────────

def granger_causality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test whether conf_score Granger-causes price returns, or the reverse.
    Uses statsmodels grangercausalitytests.
    """
    log.info("Method 3: Granger Causality...")
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        log.warning("statsmodels not installed — skipping Granger test")
        return pd.DataFrame()

    results = []
    for symbol, sdf in df.groupby("symbol"):
        sdf = sdf.sort_values("date").dropna(subset=["conf_score", "day_return"])
        if len(sdf) < MIN_ROWS_PER_SYMBOL:
            continue

        # Test 1: conf_score → day_return (does score predict future return?)
        try:
            data_fwd = sdf[["day_return", "conf_score"]].dropna()
            gc_fwd = grangercausalitytests(data_fwd, maxlag=3, verbose=False)
            # Get best p-value across lags
            pvals_fwd = [gc_fwd[lag][0]["ssr_ftest"][1] for lag in range(1, 4)]
            best_p_fwd = min(pvals_fwd)
            best_lag_fwd = pvals_fwd.index(best_p_fwd) + 1
        except Exception:
            best_p_fwd = None
            best_lag_fwd = None

        # Test 2: day_return → conf_score (does price drive score?)
        try:
            data_rev = sdf[["conf_score", "day_return"]].dropna()
            gc_rev = grangercausalitytests(data_rev, maxlag=3, verbose=False)
            pvals_rev = [gc_rev[lag][0]["ssr_ftest"][1] for lag in range(1, 4)]
            best_p_rev = min(pvals_rev)
            best_lag_rev = pvals_rev.index(best_p_rev) + 1
        except Exception:
            best_p_rev = None
            best_lag_rev = None

        # Interpret
        score_leads = best_p_fwd is not None and best_p_fwd < 0.05
        price_leads = best_p_rev is not None and best_p_rev < 0.05

        if score_leads and not price_leads:
            direction = "SCORE_LEADS_PRICE"
        elif price_leads and not score_leads:
            direction = "PRICE_LEADS_SCORE"
        elif score_leads and price_leads:
            direction = "BIDIRECTIONAL"
        else:
            direction = "NO_CAUSALITY"

        results.append({
            "symbol": symbol,
            "sector": sdf["sector"].iloc[0],
            "n": len(sdf),
            "score_to_price_pval": round(best_p_fwd, 4) if best_p_fwd is not None else None,
            "score_to_price_lag": best_lag_fwd,
            "price_to_score_pval": round(best_p_rev, 4) if best_p_rev is not None else None,
            "price_to_score_lag": best_lag_rev,
            "direction": direction,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4 — PANEL REGRESSION PER SYMBOL
# ─────────────────────────────────────────────────────────────────────────────

def per_symbol_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS regression: fwd_return ~ conf_score + log(volume)
    Per symbol, per forward horizon.
    Gives beta, t-stat, R².
    """
    log.info("Method 4: Per-Symbol Regression...")
    from scipy.stats import t as t_dist

    results = []
    for symbol, sdf in df.groupby("symbol"):
        sdf = sdf.sort_values("date").copy()
        sector = sdf["sector"].iloc[0]
        n_total = len(sdf)

        for fwd in [1, 5]:
            col = f"fwd_{fwd}d"
            sub = sdf.dropna(subset=[col, "conf_score", "volume"]).copy()
            sub = sub[sub["volume"] > 0]
            if len(sub) < MIN_ROWS_PER_SYMBOL:
                continue

            sub["log_vol"] = np.log(sub["volume"])
            X = np.column_stack([
                np.ones(len(sub)),
                sub["conf_score"].values,
                sub["log_vol"].values,
            ])
            y = sub[col].values

            try:
                # OLS via numpy
                beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                # Standard errors
                n, k = X.shape
                sigma2 = ss_res / (n - k)
                try:
                    cov = sigma2 * np.linalg.inv(X.T @ X)
                    se = np.sqrt(np.diag(cov))
                    t_stat = beta[1] / se[1]
                    pval = 2 * t_dist.sf(abs(t_stat), df=n - k)
                except Exception:
                    t_stat = None
                    pval = None

                results.append({
                    "symbol": symbol,
                    "sector": sector,
                    "forward_days": fwd,
                    "n": n_total,
                    "obs_used": len(sub),
                    "beta_conf": round(float(beta[1]), 6),
                    "t_stat": round(float(t_stat), 4) if t_stat is not None else None,
                    "pval": round(float(pval), 4) if pval is not None else None,
                    "r2": round(float(r2), 6),
                    "significant": pval is not None and pval < 0.05,
                    "corr_conf_return": round(
                        spearmanr(sub["conf_score"], sub[col])[0], 4
                    ),
                    "corr_conf_volume": round(
                        spearmanr(sub["conf_score"], sub["volume"])[0], 4
                    ),
                    "avg_conf": round(sub["conf_score"].mean(), 2),
                    "avg_return": round(sub[col].mean(), 4),
                })
            except Exception as e:
                log.debug("Regression failed for %s fwd=%d: %s", symbol, fwd, e)

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 5 — THRESHOLD DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def threshold_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search for the conf_score threshold that maximises the
    return difference between above-threshold and below-threshold stocks.
    Tests every 2.5 point increment from 30 to 65.
    """
    log.info("Method 5: Threshold Detection...")
    results = []
    thresholds = np.arange(30, 66, 2.5)

    groups = {"ALL": df}
    for sector, sdf in df.groupby("sector"):
        if len(sdf) >= MIN_ROWS_PER_SYMBOL * 3:
            groups[sector] = sdf

    for group_name, gdf in groups.items():
        for fwd in [1, 5]:
            col = f"fwd_{fwd}d"
            sub = gdf.dropna(subset=[col])
            if len(sub) < 50:
                continue

            best_threshold = None
            best_spread = -999
            best_above_ret = None
            best_below_ret = None
            best_pval = None
            best_n_above = None
            best_n_below = None

            for thresh in thresholds:
                above = sub[sub["conf_score"] >= thresh][col]
                below = sub[sub["conf_score"] < thresh][col]

                if len(above) < 20 or len(below) < 20:
                    continue

                spread = above.mean() - below.mean()

                # t-test for significance
                try:
                    _, pval = stats.ttest_ind(above, below, equal_var=False)
                except Exception:
                    pval = 1.0

                if spread > best_spread:
                    best_spread = spread
                    best_threshold = thresh
                    best_above_ret = round(above.mean(), 4)
                    best_below_ret = round(below.mean(), 4)
                    best_pval = round(pval, 4)
                    best_n_above = len(above)
                    best_n_below = len(below)

            if best_threshold is not None:
                results.append({
                    "group": group_name,
                    "forward_days": fwd,
                    "best_threshold": best_threshold,
                    "spread_pct": round(best_spread, 4),
                    "above_threshold_avg_return": best_above_ret,
                    "below_threshold_avg_return": best_below_ret,
                    "ttest_pval": best_pval,
                    "significant": best_pval < 0.05,
                    "n_above": best_n_above,
                    "n_below": best_n_below,
                })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 6 — VOLUME INDEPENDENCE PER SYMBOL
# ─────────────────────────────────────────────────────────────────────────────

def volume_independence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol correlation between conf_score and volume/turnover.
    Confirms conf_score is not just a volume proxy.
    """
    log.info("Method 6: Volume Independence...")
    results = []

    for symbol, sdf in df.groupby("symbol"):
        sdf = sdf.dropna(subset=["conf_score", "volume", "turnover"])
        sdf = sdf[sdf["volume"] > 0]
        if len(sdf) < MIN_ROWS_PER_SYMBOL:
            continue

        corr_vol, pval_vol = spearmanr(sdf["conf_score"], sdf["volume"])
        corr_to, pval_to = spearmanr(sdf["conf_score"], sdf["turnover"])

        # Also correlate with momentum (lagged return)
        sdf = sdf.sort_values("date")
        sdf["lag1_return"] = sdf["day_return"].shift(1)
        sub_lag = sdf.dropna(subset=["lag1_return"])
        corr_mom = spearmanr(sub_lag["conf_score"], sub_lag["lag1_return"])[0] if len(sub_lag) >= 20 else None

        results.append({
            "symbol": symbol,
            "sector": sdf["sector"].iloc[0],
            "n": len(sdf),
            "corr_conf_volume": round(corr_vol, 4),
            "corr_conf_turnover": round(corr_to, 4),
            "corr_conf_momentum": round(corr_mom, 4) if corr_mom is not None else None,
            "volume_proxy_risk": (
                "HIGH" if abs(corr_vol) > 0.5
                else "MEDIUM" if abs(corr_vol) > 0.3
                else "LOW"
            ),
            "independent_signal": abs(corr_vol) < 0.3 and abs(corr_to) < 0.3,
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    quintile_df: pd.DataFrame,
    ic_summary: pd.DataFrame,
    ic_ts: pd.DataFrame,
    granger_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CONF_SCORE STATISTICAL ANALYSIS — NEPSE AI ENGINE")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # IC Summary
    lines.append("\n── METHOD 2: INFORMATION COEFFICIENT ──")
    if not ic_summary.empty:
        for _, row in ic_summary.iterrows():
            lines.append(
                f"  {row['forward_days']:>2}d fwd | IC={row['mean_ic']:>7.4f} | "
                f"ICIR={row['icir']:>6.4f} | t={row['t_stat']:>6.2f} | "
                f"Hit={row['hit_rate']:>5.1f}% | {row['signal_strength']}"
            )

    # Granger summary
    lines.append("\n── METHOD 3: GRANGER CAUSALITY ──")
    if not granger_df.empty:
        counts = granger_df["direction"].value_counts()
        total = len(granger_df)
        for direction, count in counts.items():
            lines.append(f"  {direction:<25} {count:>3} / {total} symbols ({count/total*100:.1f}%)")

        leads = granger_df[granger_df["direction"] == "SCORE_LEADS_PRICE"]
        follows = granger_df[granger_df["direction"] == "PRICE_LEADS_SCORE"]
        lines.append(f"\n  Score leads price (useful signal): {len(leads)} symbols")
        lines.append(f"  Price leads score (lagging indicator): {len(follows)} symbols")

    # Regression summary
    lines.append("\n── METHOD 4: REGRESSION (1-day forward) ──")
    if not regression_df.empty:
        d1 = regression_df[regression_df["forward_days"] == 1]
        sig = d1[d1["significant"] == True]
        lines.append(f"  Significant beta (p<0.05): {len(sig)} / {len(d1)} symbols")
        if len(d1) > 0:
            lines.append(f"  Avg beta_conf: {d1['beta_conf'].mean():.6f}")
            lines.append(f"  Avg Spearman corr: {d1['corr_conf_return'].mean():.4f}")

    # Threshold summary
    lines.append("\n── METHOD 5: THRESHOLD DETECTION ──")
    if not threshold_df.empty:
        market = threshold_df[
            (threshold_df["group"] == "ALL") &
            (threshold_df["forward_days"] == 1)
        ]
        if not market.empty:
            row = market.iloc[0]
            lines.append(f"  Market-wide best threshold (1d): {row['best_threshold']}")
            lines.append(f"  Above threshold avg return:  {row['above_threshold_avg_return']:>7.4f}%")
            lines.append(f"  Below threshold avg return:  {row['below_threshold_avg_return']:>7.4f}%")
            lines.append(f"  Spread: {row['spread_pct']:.4f}% | p={row['ttest_pval']:.4f} | sig={row['significant']}")

        lines.append("\n  Per-sector thresholds (1d):")
        sector_thresh = threshold_df[
            (threshold_df["group"] != "ALL") &
            (threshold_df["forward_days"] == 1)
        ].sort_values("best_threshold")
        for _, row in sector_thresh.iterrows():
            lines.append(
                f"    {row['group']:<30} threshold={row['best_threshold']:>4.1f} | "
                f"spread={row['spread_pct']:>6.4f}% | sig={row['significant']}"
            )

    # Volume independence
    lines.append("\n── METHOD 6: VOLUME INDEPENDENCE ──")
    if not volume_df.empty:
        risk_counts = volume_df["volume_proxy_risk"].value_counts()
        for risk, count in risk_counts.items():
            lines.append(f"  {risk:<8} volume proxy risk: {count} symbols")
        independent = volume_df["independent_signal"].sum()
        lines.append(f"  Genuinely independent signal: {independent} / {len(volume_df)} symbols")
        lines.append(f"  Avg corr(conf, volume):   {volume_df['corr_conf_volume'].mean():.4f}")
        lines.append(f"  Avg corr(conf, momentum): {volume_df['corr_conf_momentum'].dropna().mean():.4f}")

    # Quintile summary
    lines.append("\n── METHOD 1: QUINTILE SORT (Market-wide, 1d forward) ──")
    if not quintile_df.empty:
        market_q = quintile_df[
            (quintile_df["sector"] == "ALL") &
            (quintile_df["forward_days"] == 1)
        ].sort_values("quintile")
        for _, row in market_q.iterrows():
            lines.append(
                f"  Q{int(row['quintile'])} (avg score={row['avg_conf_score']:>5.1f}) | "
                f"avg_return={row['avg_return']:>7.4f}% | "
                f"win_rate={row['win_rate']:>5.1f}% | n={row['n']}"
            )

    lines.append("\n" + "=" * 70)
    lines.append("CONCLUSION")
    lines.append("=" * 70)

    # Auto-conclusion
    if not ic_summary.empty:
        ic_1d = ic_summary[ic_summary["forward_days"] == 1]
        if not ic_1d.empty:
            ic_val = ic_1d.iloc[0]["mean_ic"]
            icir_val = ic_1d.iloc[0]["icir"]
            if abs(ic_val) > 0.05 and abs(icir_val) > 0.5:
                lines.append("Signal quality: STRONG — conf_score is actionable")
            elif abs(ic_val) > 0.02:
                lines.append("Signal quality: MODERATE — use as ADD_TO_REASONING")
            else:
                lines.append("Signal quality: WEAK — noise, do not gate on this")

    if not threshold_df.empty:
        market_1d = threshold_df[
            (threshold_df["group"] == "ALL") &
            (threshold_df["forward_days"] == 1) &
            (threshold_df["significant"] == True)
        ]
        if not market_1d.empty:
            thresh = market_1d.iloc[0]["best_threshold"]
            lines.append(f"Recommended threshold: conf_score >= {thresh}")
            lines.append(f"  Above {thresh}: ADD confidence")
            lines.append(f"  Below {thresh - 10}: REDUCE confidence")
        else:
            lines.append("No statistically significant threshold found — treat as continuous signal")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("Starting conf_score analysis — output dir: %s", OUTPUT_DIR)

    # Load
    df = load_data()

    # Run all methods
    quintile_df = quintile_sort(df)
    ic_ts, ic_summary = information_coefficient(df)
    granger_df = granger_causality(df)
    regression_df = per_symbol_regression(df)
    threshold_df = threshold_detection(df)
    volume_df = volume_independence(df)

    # Save CSVs
    quintile_df.to_csv(f"{OUTPUT_DIR}/quintile_returns.csv", index=False)
    ic_ts.to_csv(f"{OUTPUT_DIR}/ic_timeseries.csv", index=False)
    ic_summary.to_csv(f"{OUTPUT_DIR}/ic_summary.csv", index=False)
    granger_df.to_csv(f"{OUTPUT_DIR}/granger_results.csv", index=False)
    regression_df.to_csv(f"{OUTPUT_DIR}/per_symbol_regression.csv", index=False)
    threshold_df.to_csv(f"{OUTPUT_DIR}/threshold_results.csv", index=False)
    volume_df.to_csv(f"{OUTPUT_DIR}/volume_independence.csv", index=False)

    log.info("CSVs saved to %s/", OUTPUT_DIR)

    # Write report
    report = write_report(
        quintile_df, ic_summary, ic_ts,
        granger_df, regression_df,
        threshold_df, volume_df,
    )
    report_path = f"{OUTPUT_DIR}/report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    log.info("Report saved to %s", report_path)
    print("\n" + report)


if __name__ == "__main__":
    main()