"""
stat_method/macro_correlation.py  (v3)
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Statistical Analysis Phase 1

What changed in v3:
  - Uses ALL non-null NRB macro variables
  - COALESCE(interest_rate, policy_rate) handles both old/new field names
  - Bonferroni n_tests computed from actual valid pairs, not hardcoded
  - Correct CLI: runs as module from project root
  - CSV saves to project root

Usage (from project root):
  python -m stat_method.macro_correlation              # full run + DB write
  python -m stat_method.macro_correlation --dry-run   # no DB write
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import re
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

# Load .env from project root (parent of stat_method/)
load_dotenv(Path(__file__).parent.parent / ".env")

# Allow imports from project root (sheets, db, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")

NST     = timezone(timedelta(hours=5, minutes=45))
DRY_RUN = "--dry-run" in sys.argv

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# All variables we want to test. Any with insufficient non-null data
# are automatically skipped (< 8 valid pairs per lag).
# ══════════════════════════════════════════════════════════════════════════════

MACRO_VARS = {
    "bank_rate":             "bank_rate",
    "cpi_inflation":             "CPI Inflation (%)",
    "credit_growth_rate":        "Credit Growth Rate (%)",
    "remittance_yoy_change_pct": "Remittance YoY Change (%)",
    "fx_reserve_months":         "FX Reserve (months import)",
    "bop_overall_balance_usd_m": "BOP Overall Balance (USD M)",
    "bop_current_account_usd_m": "BOP Current Account (USD M)",
}

LAGS      = [0, 1, 2, 3]
ALPHA_RAW = 0.05

NEPSE_INDEX_ID = "58"   # NEPSE composite in nepse_indices table


def effect_size_label(rho: float) -> str:
    r = abs(rho)
    if r < 0.10: return "negligible"
    if r < 0.30: return "small"
    if r < 0.50: return "medium"
    return "large"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FY PERIOD → CALENDAR MONTH
# Nepal FY starts Shrawan ≈ mid-July:
#   month1=Aug, month2=Sep, ..., month5=Dec,
#   month6=Jan(+1yr), ..., month12=Jul(+1yr)
# ══════════════════════════════════════════════════════════════════════════════

def fy_period_to_calendar(period: str):
    """
    'fy2023/24-month1'   → (2023, 8)
    'fy2023/24-month06'  → (2024, 1)   zero-padded OK
    'fy2025/26-month04'  → (2025, 11)
    annual rows          → None (skipped)
    """
    if "annual" in period.lower():
        return None

    m = re.match(r"fy(\d{4})/\d{2,4}-month0*(\d+)", period)
    if not m:
        return None

    fy_start = int(m.group(1))
    fy_month = int(m.group(2))

    cal_month_raw = 7 + fy_month
    if cal_month_raw <= 12:
        return (fy_start, cal_month_raw)
    else:
        return (fy_start + 1, cal_month_raw - 12)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_nepse_monthly() -> pd.DataFrame:
    """
    Pull NEPSE composite daily closes from nepse_indices.
    Average per calendar month → one monthly row.
    """
    from db.connection import _db

    print("Loading NEPSE daily index from nepse_indices...")

    with _db() as cur:
        cur.execute("""
            SELECT
                date,
                current_value::float AS close_value
            FROM nepse_indices
            WHERE index_id = %s
              AND current_value IS NOT NULL
              AND current_value != ''
            ORDER BY date ASC
        """, (NEPSE_INDEX_ID,))
        rows = cur.fetchall()

    if not rows:
        print("ERROR: No data in nepse_indices for index_id=58.")
        print("       Run sharehub_index_scraper.py first.")
        sys.exit(1)

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby(["year", "month"])["close_value"]
        .mean()
        .reset_index()
        .rename(columns={"close_value": "nepse_avg"})
    )
    monthly["nepse_avg"] = monthly["nepse_avg"].round(2)

    print(f"  Daily rows:       {len(df)}")
    print(f"  Monthly averages: {len(monthly)} months")
    print(f"  Date range:       {df['date'].min().date()} → {df['date'].max().date()}")
    print()
    return monthly

def load_nrb_monthly() -> pd.DataFrame:
    from db.connection import _db

    print("Loading NRB macro data from nrb_monthly...")

    with _db() as cur:
        cur.execute("""
            SELECT
                period,
                bank_rate,                 
                cpi_inflation,
                credit_growth_rate,
                remittance_yoy_change_pct,
                fx_reserve_months,
                bop_overall_balance_usd_m,
                bop_current_account_usd_m
            FROM nrb_monthly
            WHERE is_annual = 'false'
              AND period NOT LIKE '%%annual%%'
            ORDER BY period
        """)
        rows = cur.fetchall()

    if not rows:
        print("ERROR: No monthly data in nrb_monthly. Run seed_nrb.py first.")
        sys.exit(1)

    # Convert to DataFrame with correct column names
    df = pd.DataFrame(rows)
    df.columns = ["period", "bank_rate", "cpi_inflation", "credit_growth_rate",
                  "remittance_yoy_change_pct", "fx_reserve_months",
                  "bop_overall_balance_usd_m", "bop_current_account_usd_m"]

    # Convert numeric columns using pandas (coerce errors to NaN)
    numeric_cols = ["bank_rate", "cpi_inflation", "credit_growth_rate",
                    "remittance_yoy_change_pct", "fx_reserve_months",
                    "bop_overall_balance_usd_m", "bop_current_account_usd_m"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Map fiscal period to calendar month
    records = []
    for _, row in df.iterrows():
        cal = fy_period_to_calendar(row["period"])
        if cal is None:
            continue
        cal_year, cal_month = cal
        record = {"period": row["period"], "year": cal_year, "month": cal_month}
        for col in MACRO_VARS:
            record[col] = row.get(col)
        records.append(record)

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(["year", "month"]).reset_index(drop=True)

    print(f"  NRB monthly rows: {len(result_df)}")
    print(f"  Period range:     {result_df['period'].iloc[0]} → {result_df['period'].iloc[-1]}")
    print(f"  Non-null counts per variable:")
    for col, label in MACRO_VARS.items():
        n_valid = result_df[col].notna().sum() if col in result_df.columns else 0
        bar = "█" * int(n_valid / len(result_df) * 20)
        print(f"    {label:<44} {n_valid:>3}/{len(result_df)}  {bar}")
    print()
    return result_df

def build_merged_df() -> pd.DataFrame:
    """Merge NRB macro with NEPSE monthly averages on (year, month)."""
    nrb   = load_nrb_monthly()
    nepse = load_nepse_monthly()

    merged = nrb.merge(nepse, on=["year", "month"], how="inner")
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)
    merged["date"] = pd.to_datetime(
        merged["year"].astype(str) + "-"
        + merged["month"].astype(str).str.zfill(2) + "-01"
    )

    print(f"Merged rows (NRB ∩ NEPSE monthly): {len(merged)}")
    if len(merged) == 0:
        print("ERROR: No overlapping months. Run sharehub_index_scraper.py to add NEPSE history.")
        sys.exit(1)

    print(f"Date range: {merged['date'].min().strftime('%Y-%m')} → {merged['date'].max().strftime('%Y-%m')}")
    print()
    return merged


def add_nepse_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute month-over-month and 3-month forward returns. Use returns not levels."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["nepse_return_1m"] = df["nepse_avg"].pct_change(1) * 100
    df["nepse_return_3m"] = df["nepse_avg"].pct_change(3).shift(-3) * 100
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SPEARMAN CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def run_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman ρ for each macro variable × lag vs NEPSE 1m return.
    Skips variable/lag pairs with < 8 valid observations.
    Bonferroni denominator = actual valid test count.
    """
    target    = df["nepse_return_1m"].copy()
    target_3m = df["nepse_return_3m"].copy()

    # Count valid tests first for correct Bonferroni denominator
    valid_test_count = sum(
        1
        for col in MACRO_VARS
        if col in df.columns
        for lag in LAGS
        if pd.concat([df[col].shift(lag), target], axis=1).dropna().__len__() >= 8
    )
    n_tests         = max(valid_test_count, 1)
    alpha_corrected = ALPHA_RAW / n_tests
    print(f"  Valid tests (n≥8): {n_tests}  →  Bonferroni p < {alpha_corrected:.4f}")

    results = []
    for col, col_label in MACRO_VARS.items():
        if col not in df.columns:
            continue

        for lag in LAGS:
            macro_lagged = df[col].shift(lag)

            valid = pd.concat([macro_lagged, target], axis=1).dropna()
            n = len(valid)
            if n < 8:
                continue

            rho, p_val  = stats.spearmanr(valid.iloc[:, 0].values, valid.iloc[:, 1].values)
            p_corrected = min(p_val * n_tests, 1.0)
            significant = p_corrected < ALPHA_RAW

            valid_3m     = pd.concat([macro_lagged, target_3m], axis=1).dropna()
            n_3m         = len(valid_3m)
            rho_3m, p_3m = (np.nan, np.nan)
            if n_3m >= 8:
                rho_3m, p_3m = stats.spearmanr(
                    valid_3m.iloc[:, 0].values, valid_3m.iloc[:, 1].values
                )

            results.append({
                "variable":        col,
                "label":           col_label,
                "lag_months":      lag,
                "spearman_rho":    round(rho, 4),
                "p_value":         round(p_val, 6),
                "p_corrected":     round(p_corrected, 6),
                "significant":     significant,
                "effect_size":     effect_size_label(rho),
                "n_pairs":         n,
                "rho_3m_forward":  round(rho_3m, 4) if not np.isnan(rho_3m) else None,
                "p_3m_forward":    round(p_3m, 6)   if not np.isnan(p_3m)   else None,
                "n_pairs_3m":      n_3m,
                "n_total_tests":   n_tests,
                "alpha_corrected": round(alpha_corrected, 6),
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_results(results_df: pd.DataFrame):
    sep = "=" * 74

    print(f"\n{sep}")
    print(f"  SPEARMAN CORRELATION RESULTS")
    print(sep)

    # Best lag per variable
    print(f"\n  ── BEST LAG PER VARIABLE ──────────────────────────────────────────")
    print(f"  {'Variable':<44} {'Lag':>4}  {'ρ':>7}  {'p_raw':>8}  {'Effect':>10}  {'n':>4}")
    print(f"  {'─'*74}")

    for col, col_label in MACRO_VARS.items():
        var_df = results_df[results_df["variable"] == col]
        if var_df.empty:
            print(f"       {col_label:<43} (skipped — < 8 valid pairs)")
            continue
        best = var_df.loc[var_df["spearman_rho"].abs().idxmax()]
        sig  = "✅" if best["significant"] else ("⚠️ " if best["p_value"] < 0.05 else "")
        d    = "↑" if best["spearman_rho"] > 0 else "↓"
        print(
            f"  {d} {col_label:<43}"
            f"{int(best['lag_months']):>3}m  "
            f"{best['spearman_rho']:>+7.3f}  "
            f"{best['p_value']:>8.4f}  "
            f"{best['effect_size']:>10}  "
            f"{int(best['n_pairs']):>4}  {sig}"
        )

    # Marginal results
    print(f"\n  ── MARGINAL RESULTS (uncorrected p < 0.10) ────────────────────────")
    marginal = results_df[results_df["p_value"] < 0.10].sort_values(
        "spearman_rho", key=abs, ascending=False
    )
    if not marginal.empty:
        for _, row in marginal.iterrows():
            d = "↑" if row["spearman_rho"] > 0 else "↓"
            print(
                f"  {d} {row['label']:<43} lag {int(row['lag_months'])}m  "
                f"ρ={row['spearman_rho']:+.3f}  p={row['p_value']:.4f}  "
                f"{row['effect_size']}  n={int(row['n_pairs'])}"
            )
    else:
        print("  None (all p > 0.10)")

    # System implications
    print(f"\n  ── WHAT TO DO WITH THESE RESULTS ──────────────────────────────────")

    medium_plus = results_df[
        results_df["effect_size"].isin(["medium", "large"])
    ].drop_duplicates("variable")

    if not medium_plus.empty:
        print("  Keep in nepal_score with calibrated weights:")
        for _, row in medium_plus.iterrows():
            best = results_df[results_df["variable"] == row["variable"]]
            best = best.loc[best["spearman_rho"].abs().idxmax()]
            d = "↑ positive" if best["spearman_rho"] > 0 else "↓ INVERSE (rising = bad for NEPSE)"
            print(f"    ✅ {row['label']:<44} ρ={best['spearman_rho']:+.3f}  lag {int(best['lag_months'])}m  {d}")
    else:
        print("  No medium/large effects — results need more data before calibration.")

    negligible_vars = results_df[
        results_df["effect_size"] == "negligible"
    ].drop_duplicates("variable")
    if not negligible_vars.empty:
        print("\n  Consider removing from scoring (negligible effect):")
        for _, row in negligible_vars.iterrows():
            best = results_df[results_df["variable"] == row["variable"]]
            best = best.loc[best["spearman_rho"].abs().idxmax()]
            print(f"    ❌ {row['label']:<44} ρ={best['spearman_rho']:+.3f}")

    n_max = int(results_df["n_pairs"].max()) if not results_df.empty else 0
    ac    = results_df["alpha_corrected"].iloc[0] if not results_df.empty else 0
    print(f"\n  n = {n_max} monthly observations | Bonferroni p < {ac:.4f}")
    print(f"  Note: with n<30, treat medium effects as signals, not confirmed findings.")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def write_results_to_db(results_df: pd.DataFrame):
    from db.connection import _db
    import psycopg2.extras

    run_date = datetime.now(tz=NST).strftime("%Y-%m-%d")
    rows = []

    for _, row in results_df.iterrows():
        rows.append((
            row["variable"],
            row["label"],
            int(row["lag_months"]),
            float(row["spearman_rho"])   if not pd.isna(row["spearman_rho"])   else None,
            float(row["p_value"])        if not pd.isna(row["p_value"])        else None,
            float(row["p_corrected"])    if not pd.isna(row["p_corrected"])    else None,
            bool(row["significant"]),
            row["effect_size"],
            int(row["n_pairs"]),
            float(row["rho_3m_forward"]) if row["rho_3m_forward"] is not None else None,
            float(row["p_3m_forward"])   if row["p_3m_forward"]   is not None else None,
            int(row["n_pairs_3m"]),
            int(row["n_total_tests"]),
            float(row["alpha_corrected"]),
            run_date,
            f"v3: nepse_indices daily avg/month, n={int(row['n_pairs'])}",
        ))

    with _db() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO macro_stat_results (
                variable, variable_label, lag_months, spearman_rho,
                p_value, p_corrected, significant, effect_size,
                n_pairs, rho_3m_forward, p_3m_forward, n_pairs_3m,
                n_total_tests, alpha_corrected, run_date, notes
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (variable, lag_months, run_date)
            DO UPDATE SET
                spearman_rho   = EXCLUDED.spearman_rho,
                p_value        = EXCLUDED.p_value,
                p_corrected    = EXCLUDED.p_corrected,
                significant    = EXCLUDED.significant,
                effect_size    = EXCLUDED.effect_size,
                n_pairs        = EXCLUDED.n_pairs,
                rho_3m_forward = EXCLUDED.rho_3m_forward,
                notes          = EXCLUDED.notes,
                inserted_at    = NOW()
            """,
            rows,
        )

    print(f"  {len(rows)} results written to macro_stat_results")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 74)
    print("  NEPSE MACRO CORRELATION ANALYSIS  v3")
    print("  Source: nepse_indices (daily avg → monthly) + nrb_monthly")
    print("=" * 74 + "\n")

    df = build_merged_df()
    df = add_nepse_returns(df)

    valid_returns = df["nepse_return_1m"].dropna()
    print(f"  NEPSE monthly returns:")
    print(f"    n     = {len(valid_returns)}")
    print(f"    mean  = {valid_returns.mean():.2f}%")
    print(f"    std   = {valid_returns.std():.2f}%")
    print(f"    range = {valid_returns.min():.2f}% → {valid_returns.max():.2f}%")
    print()

    results_df = run_correlation(df)

    if results_df.empty:
        print("No results — not enough overlapping data.")
        print("Run sharehub_index_scraper.py to backfill NEPSE history.")
        sys.exit(1)

    print_results(results_df)

    csv_path = Path(__file__).parent.parent / "nepse_macro_results_v3.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    if not DRY_RUN:
        write_results_to_db(results_df)
    else:
        print("  [DRY RUN] — DB write skipped")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()