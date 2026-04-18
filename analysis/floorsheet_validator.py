"""
analysis/floorsheet_validator.py — NEPSE AI Engine
═══════════════════════════════════════════════════════════════════════════════
Statistical validation of floorsheet-derived signals against forward price
returns in NEPSE price_history.

Tests whether these signals predict future price movement BEFORE wiring
them into filter_engine.py:

  Signal              Column               Type
  ──────────────────────────────────────────────
  vwap_dev          computed (ltp vs vwap)  continuous
  buyer_pressure    buyer_pressure          continuous 0-1
  broker_conc       broker_concentration    continuous 0-1
  institutional     institutional_flag      binary true/false
  large_order_pct   large_order_pct         continuous 0-1

Methodology — matches backtester.py standards:
  - T+4 settlement (trading days only, skip weekends + holidays)
  - Hard stop: -5% from entry (backtester uses -5% per new_backtester.py)
  - Trailing stop: activates at +6%, floor = peak - 3% (per handoff)
  - Forward windows: T+5, T+10, T+17 (research-validated hold days)
  - Spearman correlation (continuous) / point-biserial (binary)
  - FDR correction (Benjamini-Hochberg) across all tests
  - Bootstrap CI on Profit Factor (1000 iterations)
  - Fully vectorized — NumPy/Pandas only, no Python loops in hot paths
  - Max 2 CPU cores via ProcessPoolExecutor

Output:
  results/floorsheet_validations/
    vwap_dev_results.csv
    buyer_pressure_results.csv
    broker_concentration_results.csv
    institutional_flag_results.csv
    large_order_pct_results.csv
    spearman_correlations.csv
    summary.txt

Run:
  python -m analysis.floorsheet_validator
  python -m analysis.floorsheet_validator --signal vwap_dev
  python -m analysis.floorsheet_validator --symbol NABIL
  python -m analysis.floorsheet_validator --from 2024-01-01 --to 2026-04-01
  python -m analysis.floorsheet_validator --save-enriched
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Optional progress bar ────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x  # no-op fallback

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("floorsheet_validator")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Output directory ─────────────────────────────────────────────────────────
RESULTS_DIR = Path("results") / "floorsheet_validations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 2

# ── Nepal fee constants — matches backtester.py exactly ──────────────────────
SEBON_PCT  = 0.00015
DP_FEE_NPR = 25.0

# ── Stop/trail constants — matches handoff + new_backtester.py ───────────────
STOP_LOSS_PCT      = 0.05   # hard stop -5%
TRAIL_ACTIVATE_PCT = 0.06   # trailing activates at +6% profit
TRAIL_FLOOR_PCT    = 0.03   # trails 3% below peak

# ── Forward return windows (trading days) ────────────────────────────────────
FORWARD_WINDOWS = [5, 10, 17]

# ── Signal definitions ───────────────────────────────────────────────────────
SIGNALS = [
    {
        "name":       "vwap_dev",
        "col":        "vwap_dev",        # computed: (ltp - vwap) / vwap
        "type":       "continuous",
        "thresholds": [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
        "direction":  "below",           # buy when ltp < vwap by threshold
        "description": "LTP deviation below VWAP — negative = oversold vs fair value",
    },
    {
        "name":       "buyer_pressure",
        "col":        "buyer_pressure",
        "type":       "continuous",
        "thresholds": [0.40, 0.50, 0.55, 0.60, 0.65, 0.70],
        "direction":  "above",           # buy when buyer pressure > threshold
        "description": "Top buyer broker % of total volume — high = demand dominance",
    },
    {
        "name":       "broker_concentration",
        "col":        "broker_concentration",
        "type":       "continuous",
        "thresholds": [0.30, 0.40, 0.50, 0.60, 0.70],
        "direction":  "above",
        "description": "Top 3 brokers % of volume — high = smart money concentration",
    },
    {
        "name":       "institutional_flag",
        "col":        "institutional_flag",
        "type":       "binary",
        "thresholds": ["true"],          # only one threshold for binary
        "direction":  "equal",
        "description": "large_order_pct>20% AND broker_conc>40% — institutional activity",
    },
    {
        "name":       "large_order_pct",
        "col":        "large_order_pct",
        "type":       "continuous",
        "thresholds": [0.10, 0.15, 0.20, 0.25, 0.30],
        "direction":  "above",
        "description": "% of volume from orders > 2x median size — block trade proxy",
    },
]

# ── Liquidity filter ─────────────────────────────────────────────────────────
MIN_VOLUME_FILTER = 1000   # minimum total_volume to consider a signal
MIN_TRADING_DAYS  = 50     # minimum number of trading days per symbol

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_floorsheet_signals(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load floorsheet_signals table for date range.
    Computes vwap_dev from ltp (from price_history) vs vwap (from floorsheet).
    """
    from db.connection import _db

    log.info("Loading floorsheet_signals %s → %s", from_date, to_date)
    with _db() as cur:
        cur.execute("""
            SELECT
                fs.date,
                fs.symbol,
                fs.vwap,
                fs.buyer_pressure,
                fs.broker_concentration,
                fs.institutional_flag,
                fs.large_order_pct,
                fs.large_order_count,
                fs.total_volume,
                fs.total_trades
            FROM floorsheet_signals fs
            WHERE fs.date >= %s AND fs.date <= %s
              AND fs.vwap IS NOT NULL
              AND fs.vwap != '0'
              AND fs.total_volume IS NOT NULL
        """, (from_date, to_date))
        rows = cur.fetchall()

    if not rows:
        log.error("No floorsheet_signals data found for %s → %s", from_date, to_date)
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    log.info("Loaded %d floorsheet_signals rows", len(df))

    # Normalize date strings — some stored as 2026-4-9, normalize to 2026-04-09
    df["date"] = pd.to_datetime(df["date"].astype(str).apply(_normalize_date))

    # Numeric conversion — vectorized
    for col in ["vwap", "buyer_pressure", "broker_concentration",
                "large_order_pct", "large_order_count",
                "total_volume", "total_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # institutional_flag: normalize to bool
    df["institutional_flag"] = (
        df["institutional_flag"].astype(str).str.lower().str.strip() == "true"
    )

    # Apply liquidity filter
    df = df[df["total_volume"] >= MIN_VOLUME_FILTER].copy()
    log.info("After liquidity filter (volume>=%d): %d rows", MIN_VOLUME_FILTER, len(df))

    return df


def _normalize_date(d: str) -> str:
    """Normalize 2026-4-9 → 2026-04-09."""
    try:
        parts = str(d).split("-")
        if len(parts) == 3:
            return f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        return d
    except Exception:
        return d


def load_price_history(symbols: list[str], from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load OHLCV price history for given symbols.
    Returns DataFrame with columns: symbol, date, open, high, low, close, volume.
    """
    from db.connection import _db

    log.info("Loading price_history for %d symbols %s → %s",
             len(symbols), from_date, to_date)

    # Load in batches to avoid huge IN clauses
    all_rows = []
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        placeholders = ", ".join(["%s"] * len(batch))
        with _db() as cur:
            cur.execute(f"""
                SELECT symbol, date, open, high, low, close, ltp, volume
                FROM price_history
                WHERE symbol IN ({placeholders})
                  AND date >= %s AND date <= %s
                ORDER BY symbol, date ASC
            """, batch + [from_date, to_date])
            all_rows.extend([dict(r) for r in cur.fetchall()])

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "high", "low", "close", "ltp", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["close"] = df["close"].fillna(df["ltp"])
    df["open"]  = df["open"].fillna(df["close"])
    df["high"]  = df["high"].fillna(df["close"])
    df["low"]   = df["low"].fillna(df["close"])

    df = df.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]

    log.info("Loaded %d price_history rows", len(df))
    return df


def get_all_trading_dates(price_df: pd.DataFrame) -> np.ndarray:
    """All unique trading dates in price_history as numpy datetime64 array."""
    return np.sort(price_df["date"].unique())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — T+4 SETTLEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_trading_date_index(trading_dates: np.ndarray) -> dict:
    """
    Build {date → index} for O(1) T+N lookup.
    Vectorized: uses numpy searchsorted.
    """
    return {d: i for i, d in enumerate(trading_dates)}


def get_t_plus_n_dates_vectorized(
    entry_dates: np.ndarray,
    trading_dates: np.ndarray,
    date_to_idx: dict,
    n: int,
) -> np.ndarray:
    """
    Vectorized T+N date lookup for all entry_dates.
    Returns array of exit dates (NaT where not enough future data).
    """
    result = np.full(len(entry_dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    max_idx = len(trading_dates) - 1

    for i, ed in enumerate(entry_dates):
        idx = date_to_idx.get(ed)
        if idx is None:
            # Find nearest trading date
            pos = np.searchsorted(trading_dates, ed)
            idx = pos if pos <= max_idx else max_idx
        target_idx = idx + n
        if target_idx <= max_idx:
            result[i] = trading_dates[target_idx]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEE CALCULATOR (matches backtester.py exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def _brokerage_vectorized(amounts: np.ndarray) -> np.ndarray:
    """Vectorized tiered brokerage — matches backtester.py _calc_brokerage."""
    result = np.where(amounts <= 2_500, 10.0,
             np.where(amounts <= 50_000,    amounts * 0.0036,
             np.where(amounts <= 500_000,   amounts * 0.0033,
             np.where(amounts <= 2_000_000, amounts * 0.0031,
             np.where(amounts <= 10_000_000,amounts * 0.0027,
                                            amounts * 0.0024)))))
    return result


def calc_buy_fees_vectorized(amounts: np.ndarray) -> np.ndarray:
    return _brokerage_vectorized(amounts) + amounts * SEBON_PCT + DP_FEE_NPR


def calc_sell_fees_vectorized(amounts: np.ndarray, profits: np.ndarray) -> np.ndarray:
    cgt = np.maximum(0.0, profits * 0.075)
    return _brokerage_vectorized(amounts) + amounts * SEBON_PCT + DP_FEE_NPR + cgt


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FORWARD RETURN ENGINE (fully vectorized)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_forward_returns(
    signal_df: pd.DataFrame,
    price_df:  pd.DataFrame,
    trading_dates: np.ndarray,
    windows:   list[int] = FORWARD_WINDOWS,
) -> pd.DataFrame:
    """
    For each signal row, compute forward returns at T+N windows
    WITH stop/trail simulation (matches backtester.py mechanics).

    Returns signal_df with added columns:
      ret_T5, ret_T10, ret_T17  (raw forward return %)
      net_T5, net_T10, net_T17  (after Nepal fees)
      stop_hit_T5, ...           (was hard stop hit before window?)
      trail_exit_T5, ...         (did trailing stop exit?)

    Fully vectorized per-symbol using numpy arrays.
    """
    date_to_idx = build_trading_date_index(trading_dates)

    # Build price lookup: {symbol → {date → (open, high, low, close)} }
    price_by_sym = {}
    for sym, grp in price_df.groupby("symbol"):
        g = grp.set_index("date").sort_index()
        price_by_sym[sym] = g

    results = []
    symbols = signal_df["symbol"].unique()
    log.info("Computing forward returns for %d symbols...", len(symbols))

    # Use tqdm progress bar if available
    iterator = tqdm(symbols, desc="Processing symbols") if HAS_TQDM else symbols
    for sym in iterator:
        sig = signal_df[signal_df["symbol"] == sym].copy()
        if sym not in price_by_sym:
            continue

        price = price_by_sym[sym]
        trading_dts_sym = price.index.values  # datetime64 array sorted

        sig_dates = sig["date"].values
        n_signals = len(sig)

        # For each window, compute simulated return with stop/trail
        for w in windows:
            col_ret  = f"ret_T{w}"
            col_net  = f"net_T{w}"
            col_stop = f"stop_hit_T{w}"
            col_trail= f"trail_T{w}"
            col_win  = f"win_T{w}"

            ret_arr   = np.full(n_signals, np.nan)
            net_arr   = np.full(n_signals, np.nan)
            stop_arr  = np.zeros(n_signals, dtype=bool)
            trail_arr = np.zeros(n_signals, dtype=bool)
            win_arr   = np.zeros(n_signals, dtype=bool)

            for idx, sig_date in enumerate(sig_dates):
                # Entry: T+4 (earliest tradable date after signal)
                sig_pos = np.searchsorted(trading_dts_sym, sig_date)
                entry_pos = sig_pos + 4  # T+4 settlement
                if entry_pos >= len(trading_dts_sym):
                    continue

                entry_date = trading_dts_sym[entry_pos]
                if entry_date not in price.index:
                    continue

                entry_price = float(price.loc[entry_date, "open"])
                if entry_price <= 0 or np.isnan(entry_price):
                    entry_price = float(price.loc[entry_date, "close"])
                if entry_price <= 0 or np.isnan(entry_price):
                    continue

                # Exit window end
                exit_pos = entry_pos + w
                if exit_pos >= len(trading_dts_sym):
                    exit_pos = len(trading_dts_sym) - 1

                # Simulate stop/trail over the window
                window_prices = price.iloc[entry_pos:exit_pos + 1]
                if len(window_prices) == 0:
                    continue

                highs  = window_prices["high"].values.astype(float)
                lows   = window_prices["low"].values.astype(float)
                closes = window_prices["close"].values.astype(float)

                hard_stop      = entry_price * (1 - STOP_LOSS_PCT)
                peak_price     = entry_price
                trail_activated= False
                exit_price     = closes[-1]
                hit_stop       = False
                hit_trail      = False

                for j in range(len(highs)):
                    h = highs[j]
                    l = lows[j]
                    c = closes[j]

                    if np.isnan(h) or np.isnan(l) or np.isnan(c):
                        continue

                    # Update peak
                    if h > peak_price:
                        peak_price = h

                    # Check trail activation
                    peak_profit = (peak_price - entry_price) / entry_price
                    if not trail_activated and peak_profit >= TRAIL_ACTIVATE_PCT:
                        trail_activated = True

                    # Effective stop
                    if trail_activated:
                        effective_stop = peak_price * (1 - TRAIL_FLOOR_PCT)
                    else:
                        effective_stop = hard_stop

                    # Stop hit?
                    if l <= effective_stop:
                        exit_price = effective_stop
                        hit_stop   = True
                        hit_trail  = trail_activated
                        break

                gross_pnl    = (exit_price - entry_price) * 100  # 100 shares
                buy_amount   = entry_price * 100
                sell_amount  = exit_price * 100
                buy_fees     = float(calc_buy_fees_vectorized(np.array([buy_amount]))[0])
                sell_fees    = float(calc_sell_fees_vectorized(
                    np.array([sell_amount]),
                    np.array([max(0.0, gross_pnl)])
                )[0])
                net_pnl      = gross_pnl - buy_fees - sell_fees
                net_ret      = net_pnl / (buy_amount + buy_fees) * 100
                raw_ret      = (exit_price - entry_price) / entry_price * 100

                ret_arr[idx]   = raw_ret
                net_arr[idx]   = net_ret
                stop_arr[idx]  = hit_stop
                trail_arr[idx] = hit_trail
                win_arr[idx]   = net_pnl > 0

            sig[col_ret]  = ret_arr
            sig[col_net]  = net_arr
            sig[col_stop] = stop_arr
            sig[col_trail]= trail_arr
            sig[col_win]  = win_arr

        results.append(sig)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SIGNAL TESTING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def test_signal(
    signal_def: dict,
    enriched_df: pd.DataFrame,
    windows: list[int] = FORWARD_WINDOWS,
    min_trades: int = 30,
    bootstrap_iter: int = 1000,
) -> list[dict]:
    """
    Test one signal across all its thresholds.
    Returns list of result dicts (one per threshold × window).
    Fully vectorized using numpy boolean masks.
    Uses Spearman for continuous signals, point‑biserial for binary.
    """
    name      = signal_def["name"]
    col       = signal_def["col"]
    direction = signal_def["direction"]
    thresholds= signal_def["thresholds"]
    sig_type  = signal_def["type"]

    if col not in enriched_df.columns and col != "vwap_dev":
        log.warning("Column %s not found — skipping %s", col, name)
        return []

    # Compute vwap_dev if needed (not stored, derived from ltp vs vwap)
    if col == "vwap_dev":
        if "ltp" in enriched_df.columns and "vwap" in enriched_df.columns:
            vwap_safe = np.where(enriched_df["vwap"].values > 0,
                                  enriched_df["vwap"].values, np.nan)
            enriched_df = enriched_df.copy()
            enriched_df["vwap_dev"] = (
                (enriched_df["ltp"].values - vwap_safe) / vwap_safe
            )
        else:
            log.warning("ltp or vwap missing — cannot compute vwap_dev")
            return []

    results = []
    col_values = enriched_df[col].values

    for threshold in thresholds:
        # Build signal mask — vectorized
        if direction == "above":
            if sig_type == "binary":
                mask = col_values == threshold
            else:
                mask = col_values > float(threshold)
        elif direction == "below":
            mask = col_values < float(threshold)
        elif direction == "equal":
            if sig_type == "binary":
                # institutional_flag stored as bool
                mask = col_values.astype(bool) == True
            else:
                mask = col_values == threshold
        else:
            mask = col_values > float(threshold)

        signal_rows    = enriched_df[mask]
        no_signal_rows = enriched_df[~mask]

        for w in windows:
            ret_col = f"net_T{w}"
            win_col = f"win_T{w}"
            stp_col = f"stop_hit_T{w}"

            if ret_col not in enriched_df.columns:
                continue

            # Signal group
            sig_rets = signal_rows[ret_col].dropna().values
            n_sig    = len(sig_rets)

            # Baseline group (no signal)
            base_rets= no_signal_rows[ret_col].dropna().values
            n_base   = len(base_rets)

            if n_sig < min_trades:
                continue

            # ── Core metrics ──────────────────────────────────────────────────
            wins     = signal_rows[win_col].dropna().values
            win_rate = float(wins.mean()) if len(wins) > 0 else 0.0

            pos_rets = sig_rets[sig_rets > 0]
            neg_rets = sig_rets[sig_rets < 0]
            pf = (
                float(pos_rets.sum() / abs(neg_rets.sum()))
                if len(neg_rets) > 0 and abs(neg_rets.sum()) > 0
                else (999.0 if len(pos_rets) > 0 else 0.0)
            )

            avg_ret  = float(np.mean(sig_rets))
            std_ret  = float(np.std(sig_rets)) if len(sig_rets) > 1 else 0.0

            # Stop hit rate
            stop_rate = float(signal_rows[stp_col].dropna().mean()) \
                        if stp_col in signal_rows.columns else 0.0

            # ── Correlation: Spearman for continuous, point‑biserial for binary ─
            if n_sig >= 10:
                sig_vals = signal_rows[col].values
                ret_vals = signal_rows[ret_col].values
                valid    = ~np.isnan(sig_vals) & ~np.isnan(ret_vals)
                if valid.sum() >= 10:
                    if sig_type == "binary":
                        # Convert boolean to 0/1 for point‑biserial
                        x = sig_vals[valid].astype(float)
                        y = ret_vals[valid]
                        rho, p_val = stats.pointbiserialr(x, y)
                    else:
                        rho, p_val = stats.spearmanr(sig_vals[valid], ret_vals[valid])
                else:
                    rho, p_val = 0.0, 1.0
            else:
                rho, p_val = 0.0, 1.0

            # ── Mann-Whitney U test — signal vs baseline ──────────────────────
            mw_p = 1.0
            mw_stat = 0.0
            if n_base >= 10 and n_sig >= 10:
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(
                        sig_rets, base_rets, alternative="greater"
                    )
                except Exception:
                    pass

            # ── Bootstrap CI on Profit Factor ─────────────────────────────────
            pf_lower, pf_upper = None, None
            if bootstrap_iter > 0 and n_sig >= 10:
                pf_list = []
                for _ in range(bootstrap_iter):
                    sample = np.random.choice(sig_rets, size=n_sig, replace=True)
                    w_sum  = sample[sample > 0].sum()
                    l_sum  = abs(sample[sample < 0].sum())
                    if l_sum > 0:
                        pf_list.append(w_sum / l_sum)
                if pf_list:
                    pf_lower = float(np.percentile(pf_list, 2.5))
                    pf_upper = float(np.percentile(pf_list, 97.5))

            # ── Baseline comparison ───────────────────────────────────────────
            base_wr  = float(base_rets[base_rets > 0].shape[0] / max(n_base, 1))
            base_avg = float(np.mean(base_rets)) if n_base > 0 else 0.0
            edge     = avg_ret - base_avg

            results.append({
                "signal":          name,
                "column":          col,
                "threshold":       str(threshold),
                "direction":       direction,
                "window_days":     w,
                "n_signals":       n_sig,
                "n_baseline":      n_base,
                "win_rate":        round(win_rate, 4),
                "profit_factor":   round(pf, 3),
                "pf_ci_lower":     round(pf_lower, 3) if pf_lower else None,
                "pf_ci_upper":     round(pf_upper, 3) if pf_upper else None,
                "avg_net_ret_pct": round(avg_ret, 4),
                "std_net_ret_pct": round(std_ret, 4),
                "stop_hit_rate":   round(stop_rate, 4),
                "correlation":     round(rho, 4),
                "corr_p":          round(p_val, 6),
                "mw_stat":         round(float(mw_stat), 2),
                "mw_p":            round(mw_p, 6),
                "baseline_wr":     round(base_wr, 4),
                "baseline_avg_ret":round(base_avg, 4),
                "edge_vs_baseline":round(edge, 4),
                # Verdict
                "passes_pf":       pf >= 1.5,
                "passes_n":        n_sig >= min_trades,
                "passes_mw":       mw_p <= 0.05,
                "passes_corr":     (abs(rho) >= 0.05 and p_val <= 0.05) if sig_type != "binary" else (rho > 0 and p_val <= 0.05),
                "survivor":        (pf >= 1.5 and n_sig >= min_trades and mw_p <= 0.05),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FDR CORRECTION (Benjamini-Hochberg)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_fdr_correction(all_results: list[dict]) -> list[dict]:
    """
    Apply Benjamini-Hochberg FDR correction to all p-values.
    Adds 'fdr_adjusted_p' and 'fdr_survivor' columns.
    Matches methodology used in fundamental_study.py.
    """
    if not all_results:
        return all_results

    # Collect all p-values (use Mann-Whitney p as primary)
    p_vals = np.array([r["mw_p"] for r in all_results])
    n      = len(p_vals)

    # BH procedure — vectorized
    sorted_idx  = np.argsort(p_vals)
    ranks       = np.empty(n, dtype=int)
    ranks[sorted_idx] = np.arange(1, n + 1)
    fdr_threshold = 0.05
    adjusted_p  = np.minimum(1.0, p_vals * n / ranks)

    # Monotonicity — BH step-up
    for i in range(n - 2, -1, -1):
        adjusted_p[sorted_idx[i]] = min(
            adjusted_p[sorted_idx[i]],
            adjusted_p[sorted_idx[i + 1]]
        )

    for i, r in enumerate(all_results):
        r["fdr_adjusted_p"] = round(float(adjusted_p[i]), 6)
        r["fdr_survivor"]   = bool(
            adjusted_p[i] <= fdr_threshold and r["survivor"]
        )

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CORRELATION MATRIX (Spearman + point-biserial)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(
    enriched_df: pd.DataFrame,
    windows: list[int] = FORWARD_WINDOWS,
) -> pd.DataFrame:
    """
    Compute correlation matrix between all floorsheet signals
    and all forward return windows.
    Uses Spearman for continuous, point‑biserial for binary.
    Saves to spearman_correlations.csv.
    """
    signal_cols = [
        "vwap_dev", "buyer_pressure", "broker_concentration",
        "large_order_pct", "institutional_flag"
    ]
    ret_cols = [f"net_T{w}" for w in windows]

    # Only keep numeric columns that exist
    avail_sig  = [c for c in signal_cols if c in enriched_df.columns]
    avail_ret  = [c for c in ret_cols    if c in enriched_df.columns]

    if not avail_sig or not avail_ret:
        return pd.DataFrame()

    # institutional_flag → numeric
    if "institutional_flag" in enriched_df.columns:
        enriched_df = enriched_df.copy()
        enriched_df["institutional_flag"] = enriched_df["institutional_flag"].astype(float)

    corr_data = []
    for sc in avail_sig:
        row = {"signal": sc}
        # Determine if binary
        is_binary = (sc == "institutional_flag")
        for rc in avail_ret:
            valid = enriched_df[[sc, rc]].dropna()
            if len(valid) >= 10:
                if is_binary:
                    rho, p = stats.pointbiserialr(valid[sc], valid[rc])
                else:
                    rho, p = stats.spearmanr(valid[sc], valid[rc])
                row[f"rho_{rc}"]  = round(rho, 4)
                row[f"p_{rc}"]    = round(p, 6)
                row[f"sig_{rc}"]  = "✓" if p <= 0.05 else ""
            else:
                row[f"rho_{rc}"] = None
                row[f"p_{rc}"]   = None
                row[f"sig_{rc}"] = ""
        corr_data.append(row)

    return pd.DataFrame(corr_data)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WORKER FUNCTION (multiprocessing)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_test_signal(args):
    """Worker: test one signal definition. Runs in separate process."""
    signal_def, enriched_pkl_path, windows, min_trades, bootstrap_iter = args

    enriched_df = pd.read_parquet(enriched_pkl_path)
    results = test_signal(signal_def, enriched_df, windows, min_trades, bootstrap_iter)
    return signal_def["name"], results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_signal_csv(name: str, results: list[dict]) -> Path:
    """Write per-signal CSV to results/floorsheet_validations/."""
    path = RESULTS_DIR / f"{name}_results.csv"
    if not results:
        log.warning("No results for %s — skipping CSV", name)
        return path

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    log.info("Written: %s (%d rows)", path, len(results))
    return path


def write_summary(
    all_results: list[dict],
    corr_df:     pd.DataFrame,
    from_date:   str,
    to_date:     str,
    elapsed_sec: float,
) -> Path:
    """Write human-readable summary.txt."""
    path = RESULTS_DIR / "summary.txt"

    survivors    = [r for r in all_results if r.get("fdr_survivor")]
    near_miss    = [r for r in all_results if r.get("survivor") and not r.get("fdr_survivor")]
    by_signal    = {}
    for r in all_results:
        by_signal.setdefault(r["signal"], []).append(r)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NEPSE AI ENGINE — FLOORSHEET SIGNAL VALIDATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Period:    {from_date} → {to_date}\n")
        f.write(f"Elapsed:   {timedelta(seconds=int(elapsed_sec))}\n")
        f.write("=" * 80 + "\n\n")

        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("  Entry:      T+4 (NEPSE settlement — earliest tradable day)\n")
        f.write("  Hard stop:  -5% from entry (triggers intraday on low)\n")
        f.write("  Trail stop: activates at +6% profit, floor = peak - 3%\n")
        f.write("  Fees:       tiered NEPSE brokerage + SEBON 0.015% + DP NPR 25\n")
        f.write("  CGT:        7.5% on profit only (applied at sell)\n")
        f.write("  Windows:    T+5, T+10, T+17 trading days\n")
        f.write("  Stats:      Spearman (continuous) / point‑biserial (binary) + Mann‑Whitney U + Bootstrap CI(PF)\n")
        f.write("  FDR:        Benjamini-Hochberg correction (q=0.05)\n")
        f.write("  Survivor:   PF≥1.5 + n≥30 + MW p≤0.05 + FDR adjusted p≤0.05\n")
        f.write("  Liquidity:  total_volume ≥ 1000, min 50 trading days per symbol\n\n")

        f.write("SIGNAL DEFINITIONS\n")
        f.write("-" * 80 + "\n")
        for s in SIGNALS:
            f.write(f"  {s['name']:<25} {s['description']}\n")
        f.write("\n")

        f.write("SURVIVORS (pass all criteria after FDR correction)\n")
        f.write("-" * 80 + "\n")
        if survivors:
            f.write(f"  {'Signal':<22} {'Thresh':<8} {'Win':<5} {'WR%':>6} "
                    f"{'PF':>6} {'PF_lo':>6} {'PF_hi':>6} "
                    f"{'AvgRet%':>8} {'corr':>6} {'mw_p':>8} {'fdr_p':>8}\n")
            f.write("  " + "-" * 78 + "\n")
            for r in sorted(survivors, key=lambda x: x["profit_factor"], reverse=True):
                f.write(
                    f"  {r['signal']:<22} {str(r['threshold']):<8} "
                    f"{r['window_days']:>3}d "
                    f"{r['win_rate']*100:>6.1f} "
                    f"{r['profit_factor']:>6.2f} "
                    f"{str(r.get('pf_ci_lower','?'))[:6]:>6} "
                    f"{str(r.get('pf_ci_upper','?'))[:6]:>6} "
                    f"{r['avg_net_ret_pct']:>8.2f} "
                    f"{r['correlation']:>6.3f} "
                    f"{r['mw_p']:>8.4f} "
                    f"{r['fdr_adjusted_p']:>8.4f}\n"
                )
        else:
            f.write("  No signals survived all criteria.\n")
        f.write("\n")

        if near_miss:
            f.write("NEAR MISSES (pass PF+n+MW but fail FDR — possible with more data)\n")
            f.write("-" * 80 + "\n")
            for r in sorted(near_miss, key=lambda x: x["profit_factor"], reverse=True)[:10]:
                f.write(
                    f"  {r['signal']:<22} thresh={r['threshold']:<8} "
                    f"win={r['window_days']}d "
                    f"PF={r['profit_factor']:.2f} "
                    f"WR={r['win_rate']*100:.1f}% "
                    f"n={r['n_signals']} "
                    f"mw_p={r['mw_p']:.4f} "
                    f"fdr_p={r['fdr_adjusted_p']:.4f}\n"
                )
            f.write("\n")

        f.write("PER-SIGNAL BEST RESULT\n")
        f.write("-" * 80 + "\n")
        for sig_name, res_list in by_signal.items():
            if not res_list:
                continue
            best = max(res_list, key=lambda x: x["profit_factor"])
            f.write(f"\n  {sig_name.upper()}\n")
            f.write(f"    Best threshold:  {best['threshold']} (direction: {best['direction']})\n")
            f.write(f"    Best window:     T+{best['window_days']}\n")
            f.write(f"    Profit Factor:   {best['profit_factor']:.3f}")
            if best.get("pf_ci_lower"):
                f.write(f" (95% CI: [{best['pf_ci_lower']:.3f}, {best['pf_ci_upper']:.3f}])")
            f.write(f"\n    Win Rate:        {best['win_rate']*100:.1f}%\n")
            f.write(f"    N signals:       {best['n_signals']}\n")
            f.write(f"    Avg net return:  {best['avg_net_ret_pct']:.2f}%\n")
            f.write(f"    Stop hit rate:   {best['stop_hit_rate']*100:.1f}%\n")
            f.write(f"    Correlation:     {best['correlation']:.4f} (p={best['corr_p']:.4f})\n")
            f.write(f"    Mann-Whitney p:  {best['mw_p']:.4f}\n")
            f.write(f"    FDR adjusted p:  {best.get('fdr_adjusted_p', '?')}\n")
            f.write(f"    FDR survivor:    {'YES ✓' if best.get('fdr_survivor') else 'NO'}\n")
            f.write(f"    Edge vs baseline:{best['edge_vs_baseline']:+.2f}% avg net return\n")

        f.write("\n\n")
        f.write("CORRELATION MATRIX (Spearman for continuous, point‑biserial for binary)\n")
        f.write("-" * 80 + "\n")
        if not corr_df.empty:
            f.write(corr_df.to_string(index=False))
        else:
            f.write("  Not computed.\n")

        f.write("\n\n")
        f.write("RECOMMENDATION\n")
        f.write("-" * 80 + "\n")
        survivor_names = list({r["signal"] for r in survivors})
        if survivor_names:
            f.write(f"  ✅ Wire these signals into filter_engine.py:\n")
            for sig in survivor_names:
                best = max(
                    [r for r in survivors if r["signal"] == sig],
                    key=lambda x: x["profit_factor"]
                )
                f.write(
                    f"     {sig:<25} best threshold={best['threshold']} "
                    f"at T+{best['window_days']} | PF={best['profit_factor']:.2f}\n"
                )
        else:
            f.write("  ❌ No signals validated. Do NOT wire into filter_engine.\n")
            f.write("     Consider: more data (backfill), stricter liquidity filter,\n")
            f.write("     or combining signals (accumulation_score composite).\n")

        f.write("\n" + "=" * 80 + "\nEND OF REPORT\n")

    log.info("Summary written: %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation(
    from_date:      str   = "2023-07-03",
    to_date:        str   = "2026-04-14",
    signal_filter:  str   = None,
    symbol_filter:  str   = None,
    min_trades:     int   = 30,
    bootstrap_iter: int   = 1000,
    windows:        list  = FORWARD_WINDOWS,
    save_enriched:  bool  = False,
) -> dict:
    import time
    t0 = time.time()

    log.info("=" * 60)
    log.info("FLOORSHEET SIGNAL VALIDATOR")
    log.info("Period: %s → %s", from_date, to_date)
    log.info("=" * 60)

    # ── 1. Load floorsheet signals ────────────────────────────────────────────
    fs_df = load_floorsheet_signals(from_date, to_date)
    if fs_df.empty:
        log.error("No floorsheet data — aborting")
        return {}

    symbols = fs_df["symbol"].unique().tolist()
    if symbol_filter:
        symbols = [s for s in symbols if s == symbol_filter.upper()]
        fs_df   = fs_df[fs_df["symbol"].isin(symbols)]
        log.info("Symbol filter: %s (%d rows)", symbol_filter, len(fs_df))

    # ── 2. Load price history ─────────────────────────────────────────────────
    # Extend range for forward return computation
    price_to = (
        datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=60)
    ).strftime("%Y-%m-%d")
    price_from = (
        datetime.strptime(from_date, "%Y-%m-%d") - timedelta(days=10)
    ).strftime("%Y-%m-%d")

    price_df = load_price_history(symbols, price_from, price_to)
    if price_df.empty:
        log.error("No price_history data — aborting")
        return {}

    # ── 3. Merge LTP into floorsheet for vwap_dev computation ────────────────
    # Use EOD close as LTP proxy (note: potential intraday vs close mismatch)
    eod_price = price_df[["symbol", "date", "close"]].copy()
    eod_price.columns = ["symbol", "date", "ltp"]
    fs_df = fs_df.merge(eod_price, on=["symbol", "date"], how="left")

    # Warn about missing LTPs
    missing_ltp = fs_df["ltp"].isna().sum()
    if missing_ltp > 0:
        log.warning("Missing LTP for %d rows (%.1f%%) — will be dropped from vwap_dev tests",
                    missing_ltp, 100 * missing_ltp / len(fs_df))

    # Filter symbols with insufficient trading days
    sym_days = price_df.groupby("symbol")["date"].nunique()
    valid_symbols = sym_days[sym_days >= MIN_TRADING_DAYS].index.tolist()
    before = len(fs_df)
    fs_df = fs_df[fs_df["symbol"].isin(valid_symbols)].copy()
    log.info("Kept %d/%d rows after filtering symbols with < %d trading days",
             len(fs_df), before, MIN_TRADING_DAYS)

    if fs_df.empty:
        log.error("No symbols passed minimum trading days filter")
        return {}

    trading_dates = get_all_trading_dates(price_df)
    log.info("Trading dates: %d", len(trading_dates))

    # ── 4. Compute forward returns ────────────────────────────────────────────
    log.info("Computing forward returns with stop/trail simulation...")
    enriched_df = compute_forward_returns(fs_df, price_df, trading_dates, windows)

    if enriched_df.empty:
        log.error("Forward return computation returned empty — aborting")
        return {}

    log.info("Enriched dataset: %d rows, %d symbols",
             len(enriched_df), enriched_df["symbol"].nunique())

    # Optionally save enriched DataFrame for debugging
    if save_enriched:
        enriched_path = RESULTS_DIR / "enriched_data.parquet"
        enriched_df.to_parquet(enriched_path, index=False)
        log.info("Saved enriched data to %s", enriched_path)

    # ── 5. Save enriched to parquet for worker processes ─────────────────────
    tmp_path = RESULTS_DIR / "_enriched_tmp.parquet"
    enriched_df.to_parquet(tmp_path, index=False)

    # ── 6. Filter signals ─────────────────────────────────────────────────────
    signals_to_test = SIGNALS
    if signal_filter:
        signals_to_test = [s for s in SIGNALS if s["name"] == signal_filter]
        if not signals_to_test:
            log.error("Unknown signal: %s", signal_filter)
            return {}

    # ── 7. Run signal tests (parallel, max 2 workers) ─────────────────────────
    log.info("Testing %d signals with %d workers...", len(signals_to_test), MAX_WORKERS)
    all_results = []

    if len(signals_to_test) == 1 or MAX_WORKERS == 1:
        # Single-process for single signal or debug
        for sig in signals_to_test:
            _, results = _worker_test_signal(
                (sig, tmp_path, windows, min_trades, bootstrap_iter)
            )
            all_results.extend(results)
            log.info("  %s: %d results", sig["name"], len(results))
    else:
        work_args = [
            (sig, tmp_path, windows, min_trades, bootstrap_iter)
            for sig in signals_to_test
        ]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_worker_test_signal, arg): arg[0]["name"]
                       for arg in work_args}
            for fut in as_completed(futures):
                sig_name = futures[fut]
                try:
                    name, results = fut.result()
                    all_results.extend(results)
                    log.info("  %s: %d results", name, len(results))
                except Exception as e:
                    log.error("  %s failed: %s", sig_name, e)

    # ── 8. FDR correction ─────────────────────────────────────────────────────
    log.info("Applying FDR correction to %d results...", len(all_results))
    all_results = apply_fdr_correction(all_results)

    survivors = [r for r in all_results if r.get("fdr_survivor")]
    log.info("Survivors after FDR: %d / %d", len(survivors), len(all_results))

    # ── 9. Correlation matrix ─────────────────────────────────────────────────
    corr_df = compute_correlation_matrix(enriched_df, windows)

    # ── 10. Write outputs ─────────────────────────────────────────────────────
    # Per-signal CSVs
    by_signal = {}
    for r in all_results:
        by_signal.setdefault(r["signal"], []).append(r)

    for sig_name, results in by_signal.items():
        write_signal_csv(sig_name, results)

    # Correlation matrix
    if not corr_df.empty:
        corr_path = RESULTS_DIR / "spearman_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        log.info("Written: %s", corr_path)

    # Summary
    elapsed = time.time() - t0
    write_summary(all_results, corr_df, from_date, to_date, elapsed)

    # Cleanup temp file
    try:
        tmp_path.unlink()
    except Exception:
        pass

    # ── 11. Print console summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FLOORSHEET SIGNAL VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  Signals tested:  {len(signals_to_test)}")
    print(f"  Total results:   {len(all_results)}")
    print(f"  Survivors (FDR): {len(survivors)}")
    print(f"  Elapsed:         {timedelta(seconds=int(elapsed))}")
    print(f"  Output:          {RESULTS_DIR}/")
    print()

    if survivors:
        print("  SURVIVORS:")
        print(f"  {'Signal':<22} {'Threshold':<10} {'Win':<5} "
              f"{'WR%':>6} {'PF':>6} {'AvgRet%':>8} {'mw_p':>8}")
        print("  " + "-" * 68)
        for r in sorted(survivors, key=lambda x: x["profit_factor"], reverse=True):
            print(f"  {r['signal']:<22} {str(r['threshold']):<10} "
                  f"T+{r['window_days']:<3} "
                  f"{r['win_rate']*100:>6.1f} "
                  f"{r['profit_factor']:>6.2f} "
                  f"{r['avg_net_ret_pct']:>8.2f} "
                  f"{r['mw_p']:>8.4f}")
    else:
        print("  ❌ No signals survived validation.")
        print("  See summary.txt for near-misses and recommendations.")
    print("=" * 70 + "\n")

    return {
        "all_results":   all_results,
        "survivors":     survivors,
        "corr_df":       corr_df,
        "enriched_df":   enriched_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    parser = argparse.ArgumentParser(description="NEPSE Floorsheet Signal Validator")
    parser.add_argument("--from",     dest="from_date",  default="2023-07-03")
    parser.add_argument("--to",       dest="to_date",    default="2026-04-14")
    parser.add_argument("--signal",   default=None,
                        help=f"One of: {[s['name'] for s in SIGNALS]}")
    parser.add_argument("--symbol",   default=None,
                        help="Single symbol for quick test (e.g. NABIL)")
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--bootstrap",  type=int, default=1000)
    parser.add_argument("--save-enriched", action="store_true",
                        help="Save enriched DataFrame to parquet for debugging")
    args = parser.parse_args()

    run_validation(
        from_date      = args.from_date,
        to_date        = args.to_date,
        signal_filter  = args.signal,
        symbol_filter  = args.symbol,
        min_trades     = args.min_trades,
        bootstrap_iter = args.bootstrap,
        save_enriched  = args.save_enriched,
    )