"""
analysis/fundamental_study.py

Full statistical study of fundamental variables vs NEPSE stock returns.
Matches quarterly fundamentals to price_history using Nepali calendar.

Statistical tools:
  - Spearman rank correlation (primary — non-parametric)
  - Pearson correlation (with winsorization)
  - Lag analysis (0, 1, 2, 3 quarters)
  - Rolling correlation (4-quarter window, per-symbol then aggregate)
  - Beta vs NEPSE index (business month-end returns)
  - OLS regression (multivariate with sector dummies)
  - VIF (multicollinearity)
  - Bonferroni + Benjamini-Hochberg corrections
  - Sector-wise breakdown (fixed sector mapping)

Outputs (all in analysis/output/):
  - fundamental_study_results.csv   : all correlations, all lags
  - fundamental_validated.csv       : only Bonferroni & FDR-significant signals
  - fundamental_beta.csv            : per-symbol beta vs NEPSE
  - fundamental_sector.csv          : sector-wise correlation breakdown
  - fundamental_summary.txt         : human-readable summary report

Usage:
    python -m analysis.fundamental_study
    python -m analysis.fundamental_study --sector "Hydro Power"
    python -m analysis.fundamental_study --symbol NABIL
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. OLS and VIF will be skipped.")

from nepali_datetime import date as NepaliDate
HAS_NEPALI = True
from db.connection import _db

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Output directory
OUTPUT_DIR = Path("analysis/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fundamental columns to analyse (skip metadata fields)
FUNDAMENTAL_COLS = [
    "eps", "net_worth", "roe", "roa", "paidup_capital", "reserve",
    "total_assets", "total_liabilities", "deposit", "loan",
    "net_interest_income", "operating_profit", "net_profit",
    "npl", "capital_fund_to_rwa", "cost_of_fund", "base_rate",
    "interest_spread", "cd_ratio", "dps", "promoter_shares",
    "public_shares", "core_capital", "gram_value", "prev_quarter_profit",
    "growth_rate", "close", "discount_rate", "pe_ratio", "peg_value",
]

# Forward return windows (in calendar months)
FORWARD_WINDOWS = {
    "ret_1m":  1,
    "ret_3m":  3,
    "ret_6m":  6,
}

# Lag in quarters to test
LAGS = [0, 1, 2, 3]

# Winsorization limits for Pearson/OLS
WINSORIZE_LIMITS = (0.01, 0.99)

# Price matching tolerance (days) after quarter end
PRICE_TOLERANCE_DAYS = 30


# ─────────────────────────────────────────────
# DATE UTILITIES (PRECISE NEPALI QUARTER ENDS)
# ─────────────────────────────────────────────

def quarter_end_ad(fiscal_year: str, quarter: str) -> date | None:
    """
    Convert Nepali fiscal year + quarter to exact Gregorian quarter-end date.
    Fiscal year starts Shrawan 1 (BS month 4, day 1).
    Quarter end dates (last day of the quarter):
      Q1 → Ashoj end   (month 6, last day)
      Q2 → Poush end   (month 9, last day)
      Q3 → Chaitra end (month 12, last day)
      Q4 → Ashadh end  (month 3 of next BS year, last day)

    Returns None if parsing fails.
    """
    try:
        parts = fiscal_year.strip().split("/")
        if len(parts) != 2:
            return None

        # '080/081' → year_start=2080, year_end=2081
        p0 = int(parts[0])
        p1 = int(parts[1])
        year_start = (2000 + p0) if p0 < 100 else p0
        # '073/074' → p1=74, '074/75' → p1=75 (short form)
        # In all cases year_end = year_start + 1
        year_end = year_start + 1

        q = quarter.strip().lower()

        # Nepali month lengths
        def last_day_of_month(year, month):
            # Try days 32 down to 28 until NepaliDate accepts it
            for d in range(32, 27, -1):
                try:
                    NepaliDate(year, month, d)
                    return d
                except Exception:
                    continue
            return 30  # fallback
        if q == "q1":
            # Ashoj (month 6) of the fiscal year (year_end, because fiscal year ends in Ashadh of year_end)
            nd = NepaliDate(year_end, 6, last_day_of_month(year_end, 6))
        elif q == "q2":
            # Poush (month 9) of year_end
            nd = NepaliDate(year_end, 9, last_day_of_month(year_end, 9))
        elif q == "q3":
            # Chaitra (month 12) of year_end
            nd = NepaliDate(year_end, 12, last_day_of_month(year_end, 12))
        elif q == "q4":
            # Ashadh (month 3) of year_end+1? Actually Q4 ends Ashadh of the fiscal year end.
            # Fiscal year ends Ashadh of year_end. Example: FY 080/81 ends 2081-03-31 (Ashadh 31)
            nd = NepaliDate(year_end, 3, last_day_of_month(year_end, 3))
        else:
            return None

        return nd.to_datetime_date()

    except Exception:
        return None


# ─────────────────────────────────────────────
# DATA LOADERS (WITHOUT SURVIVORSHIP BIAS)
# ─────────────────────────────────────────────

def load_fundamentals(sector_filter: str = None, symbol_filter: str = None) -> pd.DataFrame:
    """Load all fundamentals from DB (including delisted companies)."""
    log.info("Loading fundamentals (all companies, no survivorship filter)...")
    sql = "SELECT * FROM fundamentals ORDER BY symbol, fiscal_year, quarter"
    with _db() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    if df.empty:
        log.error("No fundamentals data found. Run fundamentals_scraper first.")
        sys.exit(1)

    if symbol_filter:
        df = df[df["symbol"] == symbol_filter.upper()]

    # Sector filter will be applied after loading sector map
    log.info("Loaded %d fundamental rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def load_price_history() -> pd.DataFrame:
    """Load full price_history as DataFrame."""
    log.info("Loading price_history...")
    sql = """
        SELECT symbol, date, open, high, low, close, volume
        FROM price_history
        ORDER BY symbol, date
    """
    with _db() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open"]  = pd.to_numeric(df["open"],  errors="coerce")
    df["high"]  = pd.to_numeric(df["high"],  errors="coerce")
    df["low"]   = pd.to_numeric(df["low"],   errors="coerce")
    df["volume"]= pd.to_numeric(df["volume"],errors="coerce")
    df = df.sort_values(["symbol", "date"])
    # Remove zero/negative closes to avoid division errors
    df = df[df["close"] > 0]
    log.info("Loaded %d price rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def load_nepse_index() -> pd.DataFrame:
    """Load NEPSE index from nepse_indices for beta calculation."""
    log.info("Loading nepse_indices...")
    sql = """
        SELECT date, current_value as nepse_close
        FROM nepse_indices
        WHERE index_id='58'
        ORDER BY date
    """
    try:
        with _db() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Empty")
        df["date"] = pd.to_datetime(df["date"])
        df["nepse_close"] = pd.to_numeric(df["nepse_close"], errors="coerce")
        df = df.dropna(subset=["nepse_close"])
        log.info("Loaded %d NEPSE index rows", len(df))
        return df
    except Exception as e:
        log.warning("Could not load nepse_indices (%s) — beta calc will be skipped", e)
        return pd.DataFrame()


def load_sector_map() -> pd.DataFrame:
    """Load symbol → sector name mapping as DataFrame."""
    try:
        with _db() as cur:
            cur.execute('SELECT "symbol", "sectorname" FROM share_sectors')
            rows = cur.fetchall()
        df = pd.DataFrame(rows)
        if df.empty:
            log.warning("No sector mapping found.")
        return df
    except Exception as e:
        log.warning("Could not load sector map: %s", e)
        return pd.DataFrame()


# ─────────────────────────────────────────────
# PRICE AGGREGATION (CALENDAR MONTH RETURNS)
# ─────────────────────────────────────────────

def compute_forward_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized forward return computation using calendar month offsets.
    For each symbol+date, compute forward returns over 1, 3, 6 calendar months.
    Returns DataFrame with columns: symbol, date, ret_1m, ret_3m, ret_6m
    """
    log.info("Computing forward returns (calendar months)...")

    df = price_df[["symbol", "date", "close"]].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Use month-end prices to align with quarter ends
    # First, resample to month-end for each symbol
    monthly = []
    for sym, grp in df.groupby("symbol"):
        grp = grp.set_index("date")
        # Get last available price each calendar month
        month_end = grp.resample("ME").last().dropna()
        month_end = month_end.reset_index()
        month_end["symbol"] = sym
        monthly.append(month_end)
    monthly_df = pd.concat(monthly, ignore_index=True)
    monthly_df = monthly_df.sort_values(["symbol", "date"])

    # Compute forward returns using calendar month offsets
    for label, months in FORWARD_WINDOWS.items():
        # Shift within symbol group by months
        future_prices = monthly_df.groupby("symbol")["close"].shift(-months)
        monthly_df[label] = (future_prices - monthly_df["close"]) / monthly_df["close"] * 100.0

    # Merge back to daily price DataFrame (keep only original dates with forward returns attached)
    # But we need returns from the exact date (not just month-end). So we keep monthly_df as the base for merging.
    # However, quarter-end matching uses daily dates. We'll align later using asof merge.
    # For now, return monthly_df which has month-end dates.
    log.info("Forward returns computed on month-end dates for %d rows", len(monthly_df))
    return monthly_df


# ─────────────────────────────────────────────
# QUARTERLY RETURN COMPUTATION (IMPROVED MATCHING)
# ─────────────────────────────────────────────

def build_merged_dataset(fund_df: pd.DataFrame, monthly_ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fundamental row, find the quarter-end date and attach
    forward returns from that date using asof merge (increased tolerance).
    """
    log.info("Building merged dataset (fundamental + price)...")

    # Add quarter_end_date to fundamentals
    fund_df = fund_df.copy()
    fund_df["quarter_end_date"] = fund_df.apply(
        lambda r: quarter_end_ad(str(r["fiscal_year"]), str(r["quarter"])),
        axis=1
    )

    # Drop rows with unparseable dates
    invalid = fund_df["quarter_end_date"].isna().sum()
    if invalid:
        log.warning("Dropped %d rows with unparseable fiscal_year/quarter", invalid)
    fund_df = fund_df.dropna(subset=["quarter_end_date"])
    fund_df["quarter_end_date"] = pd.to_datetime(fund_df["quarter_end_date"])

    # Prepare monthly returns for asof merge
    monthly_ret_df = monthly_ret_df.sort_values("date").copy()

    merged_rows = []
    for _, row in fund_df.iterrows():
        symbol = row["symbol"]
        qend = row["quarter_end_date"]

        sym_ret = monthly_ret_df[monthly_ret_df["symbol"] == symbol].sort_values("date")
        if sym_ret.empty:
            continue

        # asof merge: find nearest monthly return date >= qend
        idx = sym_ret["date"].searchsorted(qend)
        if idx >= len(sym_ret):
            continue
        nearest_date = sym_ret.iloc[idx]["date"]
        if (nearest_date - qend).days > PRICE_TOLERANCE_DAYS:
            continue  # too far

        matched = sym_ret[sym_ret["date"] == nearest_date].iloc[0]
        merged = row.to_dict()
        merged["price_date"] = nearest_date
        merged["ret_1m"] = matched["ret_1m"]
        merged["ret_3m"] = matched["ret_3m"]
        merged["ret_6m"] = matched["ret_6m"]
        merged_rows.append(merged)

    df = pd.DataFrame(merged_rows)
    log.info("Merged dataset: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


# ─────────────────────────────────────────────
# WINSORIZATION UTILITY
# ─────────────────────────────────────────────
def winsorize_series(s: pd.Series, limits=(0.01, 0.99)) -> pd.Series:
    """Winsorize a series to given percentile limits."""
    s = pd.to_numeric(s, errors="coerce")
    low = s.quantile(limits[0])
    high = s.quantile(limits[1])
    return s.clip(low, high)

# ─────────────────────────────────────────────
# STATISTICAL ANALYSIS (ENHANCED)
# ─────────────────────────────────────────────

def run_correlation_analysis(df: pd.DataFrame, lag: int = 0, winsorize: bool = True) -> pd.DataFrame:
    """
    For each fundamental column, compute Spearman and Pearson correlation
    with forward returns (1m, 3m, 6m) at given lag.
    """
    results = []
    return_cols = list(FORWARD_WINDOWS.keys())

    # Apply lag: group by symbol, shift fundamentals forward by lag quarters
    if lag > 0:
        df = df.copy().sort_values(["symbol", "quarter_end_date"])
        shifted_parts = []
        for sym, grp in df.groupby("symbol"):
            grp = grp.copy()
            for col in FUNDAMENTAL_COLS:
                if col in grp.columns:
                    grp[col] = grp[col].shift(lag)
            shifted_parts.append(grp)
        df = pd.concat(shifted_parts, ignore_index=True)

    for fcol in FUNDAMENTAL_COLS:
        if fcol not in df.columns:
            continue

        for rcol in return_cols:
            sub = df[[fcol, rcol]].dropna()
            n = len(sub)
            if n < 10:
                continue

            x = sub[fcol].values
            y = sub[rcol].values

            sub = sub.copy()
            sub[fcol] = pd.to_numeric(sub[fcol], errors="coerce")
            sub[rcol] = pd.to_numeric(sub[rcol], errors="coerce")
            sub = sub.dropna()
            # Winsorize for Pearson (optional)
            if winsorize:
                x_w = winsorize_series(pd.Series(x), WINSORIZE_LIMITS).values
                y_w = winsorize_series(pd.Series(y), WINSORIZE_LIMITS).values
            else:
                x_w, y_w = x, y

            # Spearman (no winsorization needed)
            try:
                sp_rho, sp_p = spearmanr(x, y)
            except Exception:
                sp_rho, sp_p = np.nan, np.nan

            # Pearson
            try:
                pe_r, pe_p = pearsonr(x_w, y_w)
            except Exception:
                pe_r, pe_p = np.nan, np.nan

            results.append({
                "fundamental":    fcol,
                "return_window":  rcol,
                "lag_quarters":   lag,
                "n":              n,
                "spearman_rho":   round(sp_rho, 4) if not np.isnan(sp_rho) else None,
                "spearman_p":     round(sp_p,  6) if not np.isnan(sp_p)  else None,
                "pearson_r":      round(pe_r,  4) if not np.isnan(pe_r)  else None,
                "pearson_p":      round(pe_p,  6) if not np.isnan(pe_p)  else None,
            })

    return pd.DataFrame(results)


def apply_multiple_testing_corrections(results_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Bonferroni and Benjamini-Hochberg FDR corrections."""
    df = results_df.copy()
    # Use Spearman p-values for correction (primary metric)
    mask = df["spearman_p"].notna()
    pvals = df.loc[mask, "spearman_p"].values
    n_tests = len(pvals)

    if n_tests == 0:
        return df

    # Bonferroni
    bonf_thresh = 0.05 / n_tests
    df["bonferroni_threshold"] = bonf_thresh
    df["significant_bonferroni"] = df["spearman_p"] < bonf_thresh

    # Benjamini-Hochberg
    from statsmodels.stats.multitest import multipletests
    if n_tests > 0:
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        # Create mapping
        df.loc[mask, "fdr_q"] = pvals_corrected
        df.loc[mask, "significant_fdr"] = reject
    else:
        df["fdr_q"] = np.nan
        df["significant_fdr"] = False

    log.info("Bonferroni threshold: %.8f (n_tests=%d)", bonf_thresh, n_tests)
    n_sig_bonf = df["significant_bonferroni"].sum()
    n_sig_fdr = df["significant_fdr"].sum()
    log.info("Signals passing Bonferroni: %d, passing FDR: %d", n_sig_bonf, n_sig_fdr)

    # For validated output, use FDR (more powerful)
    df["significant_validated"] = df["significant_fdr"]
    return df


def run_rolling_correlation_per_symbol(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Rolling Spearman correlation over N-quarter window PER SYMBOL, then aggregate median.
    Returns DataFrame with median rho and p-value over symbols.
    """
    log.info("Computing rolling correlations (window=%d quarters) per symbol...", window)
    results = []

    df = df.sort_values(["symbol", "quarter_end_date"])

    for fcol in FUNDAMENTAL_COLS:
        if fcol not in df.columns:
            continue

        symbol_results = []
        for symbol, grp in df.groupby("symbol"):
            grp = grp.sort_values("quarter_end_date")
            sub = grp[["quarter_end_date", fcol, "ret_3m"]].dropna()
            if len(sub) < window:
                continue

            rolling_rhos = []
            rolling_pvals = []
            for i in range(window - 1, len(sub)):
                window_data = sub.iloc[i - window + 1 : i + 1]
                if len(window_data) < 5:
                    continue
                try:
                    rho, p = spearmanr(window_data[fcol], window_data["ret_3m"])
                    rolling_rhos.append(rho)
                    rolling_pvals.append(p)
                except Exception:
                    continue

            if rolling_rhos:
                symbol_results.append({
                    "fundamental": fcol,
                    "median_rho": np.median(rolling_rhos),
                    "median_p": np.median(rolling_pvals),
                    "n_windows": len(rolling_rhos),
                })

        if symbol_results:
            # Aggregate across symbols: median of medians
            median_rho = np.median([r["median_rho"] for r in symbol_results])
            median_p = np.median([r["median_p"] for r in symbol_results])
            results.append({
                "fundamental": fcol,
                "median_rolling_rho": round(median_rho, 4),
                "median_rolling_p": round(median_p, 6),
                "n_symbols": len(symbol_results),
            })

    return pd.DataFrame(results)


def run_beta_analysis(price_df: pd.DataFrame, nepse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-symbol beta vs NEPSE index using business month-end returns.
    """
    if nepse_df.empty:
        log.warning("No NEPSE index data — skipping beta calculation")
        return pd.DataFrame()

    log.info("Computing beta vs NEPSE using business month-end returns...")

    # Prepare NEPSE monthly returns using business month end
    nepse_df = nepse_df.sort_values("date").copy()
    nepse_df = nepse_df.set_index("date")
    # Resample to business month end (last trading day)
    nepse_monthly = nepse_df["nepse_close"].resample("BME").last().pct_change().dropna()

    results = []

    for symbol, grp in price_df.groupby("symbol"):
        grp = grp.sort_values("date").set_index("date")
        # Business month end for stock
        stock_monthly = grp["close"].resample("BME").last().pct_change().dropna()

        # Align with NEPSE
        aligned = pd.concat([stock_monthly, nepse_monthly], axis=1, join="inner")
        aligned.columns = ["stock", "market"]
        aligned = aligned.dropna()

        if len(aligned) < 12:
            continue

        try:
            cov_matrix = np.cov(aligned["stock"], aligned["market"])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            corr, p = pearsonr(aligned["stock"], aligned["market"])

            results.append({
                "symbol":        symbol,
                "beta":          round(beta, 4),
                "market_corr":   round(corr, 4),
                "market_corr_p": round(p,   6),
                "n_months":      len(aligned),
            })
        except Exception:
            continue

    df = pd.DataFrame(results)
    log.info("Beta computed for %d symbols", len(df))
    return df


def run_sector_breakdown(df: pd.DataFrame, sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Spearman correlation per sector for top fundamental signals.
    Focus on ret_3m, lag=0.
    """
    log.info("Running sector-wise correlation breakdown...")

    # Merge sector info
    df = df.merge(sector_df, on="symbol", how="left")
    df["sector"] = df["sectorname"].fillna("Unknown")

    results = []

    for sector, sdf in df.groupby("sector"):
        if len(sdf) < 20:
            continue

        for fcol in FUNDAMENTAL_COLS:
            if fcol not in sdf.columns:
                continue

            sub = sdf[[fcol, "ret_3m"]].dropna()
            if len(sub) < 10:
                continue

            try:
                rho, p = spearmanr(sub[fcol], sub["ret_3m"])
                results.append({
                    "sector":       sector,
                    "fundamental":  fcol,
                    "spearman_rho": round(rho, 4),
                    "spearman_p":   round(p,   6),
                    "n":            len(sub),
                    "significant":  p < 0.05,
                })
            except Exception:
                continue

    return pd.DataFrame(results)


def run_ols_regression(df: pd.DataFrame, validated_cols: list, sector_df: pd.DataFrame) -> dict:
    """
    OLS regression: validated fundamentals + sector dummies → ret_3m.
    Returns dict with summary text, VIF, and R².
    """
    if not HAS_STATSMODELS:
        return {}

    if not validated_cols:
        log.warning("No validated columns for OLS regression")
        return {}

    log.info("Running OLS regression with %d predictors + sector dummies...", len(validated_cols))

    # Merge sector
    df = df.merge(sector_df, on="symbol", how="left")
    df["sector"] = df["sectorname"].fillna("Unknown")

    available = [c for c in validated_cols if c in df.columns]
    sub = df[available + ["ret_3m", "sector"]].dropna()

    if len(sub) < 30:
        log.warning("Not enough data for OLS (n=%d)", len(sub))
        return {}

    # Winsorize fundamentals and returns
    for col in available:
        sub[col] = winsorize_series(sub[col], WINSORIZE_LIMITS)
    sub["ret_3m"] = winsorize_series(sub["ret_3m"], WINSORIZE_LIMITS)

    # Create formula with sector dummies
    formula = "ret_3m ~ " + " + ".join(available) + " + C(sector)"
    try:
        model = ols(formula, data=sub).fit()
        summary_text = model.summary().as_text()
    except Exception as e:
        log.warning("OLS failed: %s", e)
        return {}

    # VIF (only on continuous predictors, excluding dummies)
    X = sub[available]
    X_const = sm.add_constant(X)
    vif_data = {}
    try:
        for i, col in enumerate(available):
            vif_data[col] = round(variance_inflation_factor(X_const.values, i + 1), 2)
    except Exception:
        pass

    return {
        "summary": summary_text,
        "vif":     vif_data,
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "n": len(sub),
    }


# ─────────────────────────────────────────────
# SUMMARY REPORT (ENHANCED)
# ─────────────────────────────────────────────

def build_summary_report(
    all_corr:      pd.DataFrame,
    validated:     pd.DataFrame,
    rolling:       pd.DataFrame,
    beta:          pd.DataFrame,
    sector:        pd.DataFrame,
    ols_result:    dict,
    bonferroni_threshold: float,
) -> str:
    lines = []

    def h(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def sub(title):
        lines.append("")
        lines.append(f"--- {title} ---")

    lines.append("NEPSE FUNDAMENTAL STUDY — STATISTICAL RESULTS (ENHANCED VERSION)")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total correlation tests run: {len(all_corr)}")
    lines.append(f"Bonferroni threshold: {bonferroni_threshold:.8f}")
    lines.append("NOTE: Lag=0 results are for research only. Live trading should use lag≥1.")

    # ── Validated signals (FDR)
    h("1. VALIDATED SIGNALS (pass Benjamini-Hochberg FDR, q<0.05)")
    if validated.empty:
        lines.append("  No signals passed FDR correction.")
        lines.append("  This may indicate fundamentals have weak predictive power,")
        lines.append("  OR insufficient data. Check nominal p<0.05 results below.")
    else:
        top = validated.sort_values("spearman_rho", key=abs, ascending=False)
        lines.append(f"  {len(validated)} signals validated.\n")
        lines.append(f"  {'Fundamental':<25} {'Return':<10} {'Lag':<6} {'Rho':>8} {'q-value':>12} {'n':>6}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*8} {'-'*12} {'-'*6}")
        for _, r in top.iterrows():
            lines.append(
                f"  {r['fundamental']:<25} {r['return_window']:<10} "
                f"{'Q'+str(int(r['lag_quarters'])):<6} "
                f"{r['spearman_rho']:>8.4f} {r['fdr_q']:>12.6f} {int(r['n']):>6}"
            )

    # ── Nominal signals (p<0.05, not FDR)
    h("2. NOMINAL SIGNALS (p<0.05, NOT FDR — treat as exploratory)")
    nominal = all_corr[
        (all_corr["spearman_p"] < 0.05) &
        (all_corr["significant_fdr"] == False)
    ].sort_values("spearman_rho", key=abs, ascending=False)

    if nominal.empty:
        lines.append("  None.")
    else:
        lines.append(f"  {len(nominal)} nominal signals (use with caution).\n")
        lines.append(f"  {'Fundamental':<25} {'Return':<10} {'Lag':<6} {'Rho':>8} {'p-value':>12}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*8} {'-'*12}")
        for _, r in nominal.head(20).iterrows():
            lines.append(
                f"  {r['fundamental']:<25} {r['return_window']:<10} "
                f"{'Q'+str(int(r['lag_quarters'])):<6} "
                f"{r['spearman_rho']:>8.4f} {r['spearman_p']:>12.6f}"
            )

    # ── Best lag per fundamental
    h("3. BEST LAG PER FUNDAMENTAL (strongest Spearman rho across all lags)")
    sub("Based on ret_3m")
    lag_df = all_corr[all_corr["return_window"] == "ret_3m"].copy()
    if not lag_df.empty:
        best_lag = (
            lag_df.sort_values("spearman_rho", key=abs, ascending=False)
            .groupby("fundamental")
            .first()
            .reset_index()
            [["fundamental", "lag_quarters", "spearman_rho", "spearman_p", "n"]]
            .sort_values("spearman_rho", key=abs, ascending=False)
        )
        lines.append(f"  {'Fundamental':<25} {'Best Lag':<10} {'Rho':>8} {'p-value':>12} {'n':>6}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")
        for _, r in best_lag.iterrows():
            sig_marker = " *" if r["spearman_p"] < 0.05 else ""
            lines.append(
                f"  {r['fundamental']:<25} Q{int(r['lag_quarters']):<9} "
                f"{r['spearman_rho']:>8.4f} {r['spearman_p']:>12.6f} "
                f"{int(r['n']):>6}{sig_marker}"
            )
        lines.append("  (* = p<0.05 nominal)")

    # ── Beta summary
    h("4. BETA VS NEPSE INDEX (business month-end returns)")
    if beta.empty:
        lines.append("  Beta calculation skipped (no nepse_indices data).")
    else:
        lines.append(f"  Symbols analysed: {len(beta)}")
        lines.append(f"  Median beta:      {beta['beta'].median():.4f}")
        lines.append(f"  Mean beta:        {beta['beta'].mean():.4f}")
        lines.append(f"  Beta > 1 (aggressive): {(beta['beta'] > 1).sum()} symbols")
        lines.append(f"  Beta < 1 (defensive):  {(beta['beta'] < 1).sum()} symbols")
        lines.append(f"  Beta < 0 (inverse):    {(beta['beta'] < 0).sum()} symbols")
        sub("Top 10 highest beta (most volatile vs NEPSE)")
        top10 = beta.nlargest(10, "beta")
        for _, r in top10.iterrows():
            lines.append(f"    {r['symbol']:<12} beta={r['beta']:>7.4f}  market_corr={r['market_corr']:>7.4f}")
        sub("Top 10 lowest beta (most defensive)")
        bot10 = beta.nsmallest(10, "beta")
        for _, r in bot10.iterrows():
            lines.append(f"    {r['symbol']:<12} beta={r['beta']:>7.4f}  market_corr={r['market_corr']:>7.4f}")

    # ── Sector breakdown
    h("5. SECTOR-WISE CORRELATION (ret_3m, lag=0)")
    if sector.empty:
        lines.append("  No sector data available.")
    else:
        sig_sector = sector[sector["significant"] == True].sort_values(
            "spearman_rho", key=abs, ascending=False
        )
        if sig_sector.empty:
            lines.append("  No sector-level signals at p<0.05.")
        else:
            lines.append(f"  {len(sig_sector)} sector-level significant signals.\n")
            lines.append(f"  {'Sector':<25} {'Fundamental':<25} {'Rho':>8} {'p':>10} {'n':>6}")
            lines.append(f"  {'-'*25} {'-'*25} {'-'*8} {'-'*10} {'-'*6}")
            for _, r in sig_sector.head(30).iterrows():
                lines.append(
                    f"  {r['sector']:<25} {r['fundamental']:<25} "
                    f"{r['spearman_rho']:>8.4f} {r['spearman_p']:>10.6f} {int(r['n']):>6}"
                )

    # ── OLS
    h("6. OLS REGRESSION (validated fundamentals + sector dummies → ret_3m)")
    if not ols_result:
        lines.append("  OLS skipped (no validated signals or statsmodels missing).")
    else:
        lines.append(f"  R²:     {ols_result['r_squared']}")
        lines.append(f"  Adj R²: {ols_result['adj_r_squared']}")
        lines.append(f"  N:      {ols_result['n']}")
        if ols_result.get("vif"):
            sub("VIF Scores (>5 = multicollinearity concern, >10 = severe)")
            for col, vif_val in sorted(ols_result["vif"].items(), key=lambda x: -x[1]):
                flag = " ⚠ HIGH" if vif_val > 5 else ""
                lines.append(f"    {col:<25} VIF={vif_val:.2f}{flag}")
        lines.append("")
        lines.append("  Full OLS Summary:")
        lines.append(ols_result["summary"])

    # ── Decision guide
    h("7. SYSTEM INTEGRATION RECOMMENDATIONS")
    lines.append("""
  RULE: Only include fundamentals that pass FDR (q<0.05) in filter_engine scoring.
  Nominal signals (p<0.05 only) can be tracked in learning_hub but NOT used as hard gates.

  PROCESS:
  1. Take validated signals from Section 1 (FDR).
  2. Check sector breakdown (Section 5) — does the signal hold in your
     target sectors (Hydro, Dev Banks, Finance, Life Insurance)?
  3. Check VIF (Section 6) — if two validated signals have VIF>5,
     they are redundant. Keep the one with stronger Spearman rho.
  4. Use lag=1 or higher for live trading (fundamentals are released weeks after quarter end).
  5. Re-run this study every quarter as new fundamental data arrives.

  NEPSE-SPECIFIC CAVEATS:
  - Fiscal year starts Shrawan 1 (mid-July). Quarter ends: Ashoj (Oct), Poush (Jan), Chaitra (Apr), Ashadh (Jul).
  - Q4 reports often delayed 1-2 months. Use lag=1 results for live trading.
  - Banking sector: NPL, CD ratio, capital_fund_to_rwa are key.
  - Small sample sizes per sector — treat sector results as directional.
  - Corporate actions (bonus, rights) not adjusted in price_history — may affect returns.
""")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(sector_filter: str = None, symbol_filter: str = None):
    log.info("=== NEPSE Fundamental Study started (enhanced version) ===")

    # Load data
    fund_df  = load_fundamentals(sector_filter, symbol_filter)
    price_df = load_price_history()
    nepse_df = load_nepse_index()
    sector_df = load_sector_map()

    # Compute forward returns (calendar months)
    monthly_ret_df = compute_forward_returns(price_df)

    # Build merged dataset
    merged = build_merged_dataset(fund_df, monthly_ret_df)

    if merged.empty:
        log.error("Merged dataset is empty. Check that price_history and fundamentals overlap.")
        sys.exit(1)

    log.info("Merged dataset ready: %d rows", len(merged))

    # ── Correlation analysis across all lags
    all_corr_parts = []
    for lag in LAGS:
        log.info("Running correlation analysis at lag=%d quarters...", lag)
        corr_df = run_correlation_analysis(merged, lag=lag, winsorize=True)
        all_corr_parts.append(corr_df)

    all_corr = pd.concat(all_corr_parts, ignore_index=True)

    # Apply multiple testing corrections
    all_corr = apply_multiple_testing_corrections(all_corr)
    bonferroni_threshold = all_corr["bonferroni_threshold"].iloc[0] if not all_corr.empty else 0.05

    validated = all_corr[all_corr["significant_validated"] == True].copy()
    validated_cols = validated["fundamental"].unique().tolist()

    # ── Rolling correlation (per symbol aggregate)
    rolling = run_rolling_correlation_per_symbol(merged)

    # ── Beta
    beta = run_beta_analysis(price_df, nepse_df)

    # ── Sector breakdown (using fixed sector map)
    sector_corr = run_sector_breakdown(merged, sector_df)

    # ── OLS regression (with sector dummies)
    ols_result = run_ols_regression(merged, validated_cols, sector_df)

    # ── Save outputs
    log.info("Saving outputs to %s ...", OUTPUT_DIR)

    all_corr.to_csv(OUTPUT_DIR / "fundamental_study_results.csv", index=False)
    log.info("Saved: fundamental_study_results.csv")

    if not validated.empty:
        validated.to_csv(OUTPUT_DIR / "fundamental_validated.csv", index=False)
        log.info("Saved: fundamental_validated.csv")

    if not beta.empty:
        beta.to_csv(OUTPUT_DIR / "fundamental_beta.csv", index=False)
        log.info("Saved: fundamental_beta.csv")

    if not sector_corr.empty:
        sector_corr.to_csv(OUTPUT_DIR / "fundamental_sector.csv", index=False)
        log.info("Saved: fundamental_sector.csv")

    if not rolling.empty:
        rolling.to_csv(OUTPUT_DIR / "fundamental_rolling.csv", index=False)
        log.info("Saved: fundamental_rolling.csv")

    # ── Summary report
    summary = build_summary_report(
        all_corr, validated, rolling, beta, sector_corr,
        ols_result, bonferroni_threshold
    )

    summary_path = OUTPUT_DIR / "fundamental_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    log.info("Saved: fundamental_summary.txt")

    # Print summary to console
    print("\n" + summary)

    log.info("=== Fundamental study complete ===")
    log.info("Outputs in: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    parser = argparse.ArgumentParser(description="NEPSE Fundamental Study (Enhanced)")
    parser.add_argument("--sector", type=str, default=None,
                        help="Filter to specific sector name (after loading sector map)")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Filter to specific symbol (e.g. NABIL)")
    args = parser.parse_args()
    run(sector_filter=args.sector, symbol_filter=args.symbol)