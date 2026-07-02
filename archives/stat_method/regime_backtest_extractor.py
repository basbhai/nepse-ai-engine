"""
stat_method/regime_backtest_extractor.py
──────────────────────────────────────────────────────────────────────────────
Standalone backtest data extractor.

Extracts winner/loser events with technical indicators and broker signals
for every (symbol, date) in price_history, labelled by market regime.

READ-ONLY — no writes to any production table.

Usage:
    python stat_method/regime_backtest_extractor.py
    python stat_method/regime_backtest_extractor.py --dry-run   # first 20 symbols only
"""

import sys
import os
import time
import argparse
import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# ── project root on path so we can import sheets ─────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sheets import run_raw_sql  # noqa: E402  (Neon DB)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = [
    ("2019-07-15", "2020-02-19", "BEAR_DECAY"),
    ("2020-02-20", "2020-04-15", "CRISIS_COVID"),
    ("2020-04-16", "2021-08-31", "FULL_BULL_COVID"),
    ("2021-09-01", "2023-06-30", "BEAR_NRB_TIGHTENING"),
    ("2023-07-01", "2024-06-30", "CAUTIOUS_BULL_RECOVERY"),
    ("2024-07-01", "2024-09-30", "FULL_BULL_SPIKE"),
    ("2024-10-01", "2026-06-26", "SIDEWAYS"),
]

BROKER_CUTOFF    = "2023-07-03"
FORWARD_DAYS     = 20
MIN_HISTORY_ROWS = 30
WIN_THRESHOLD    = 10.0   # % forward return
LOSS_THRESHOLD   = 0.0    # strictly below 0 → loser

LOCAL_PG_DSN = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"

OUTPUT_DIR  = Path(__file__).resolve().parents[1] / "results" / "data"
OUT_ALL     = OUTPUT_DIR / "winner_loser_events.csv"
OUT_BROKER  = OUTPUT_DIR / "winner_loser_events_broker.csv"

VALIDATED_BROKERS = {
    "broker_38": ("buyer_broker_id", "38"),   # Dipshikha
    "broker_56": ("buyer_broker_id", "56"),   # Sri Hari
    "broker_49": ("buyer_broker_id", "49"),   # Online
    "broker_48": ("buyer_broker_id", "48"),   # Trishakti
    # broker_58 (Naasa): counterintuitively bullish as SELLER — check seller_broker_id
    "broker_58": ("seller_broker_id", "58"),
}

# ─────────────────────────────────────────────────────────────────────────────
# REGIME LOOKUP  (precomputed for speed)
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_PARSED = [
    (date.fromisoformat(s), date.fromisoformat(e), label)
    for s, e, label in REGIMES
]


def assign_regime(d: date) -> str:
    for start, end, label in _REGIME_PARSED:
        if start <= d <= end:
            return label
    return "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load price_history from Neon
# ─────────────────────────────────────────────────────────────────────────────

def load_price_history() -> pd.DataFrame:
    log.info("Loading price_history from Neon …")
    rows = run_raw_sql(
        "SELECT date, symbol, open, high, low, close, volume FROM price_history ORDER BY symbol, date"
    )
    if not rows:
        raise RuntimeError("price_history returned no rows — check Neon connection.")

    df = pd.DataFrame(rows)

    def _to_float(col: pd.Series) -> pd.Series:
        return pd.to_numeric(col.replace("", np.nan), errors="coerce")

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = _to_float(df[c])

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["close"].notna() & (df["close"] > 0)].copy()
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info("price_history loaded: %d rows, %d symbols",
             len(df), df["symbol"].nunique())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Forward returns + winner/loser labels
# ─────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward_return_pct, exit_price_d20, is_winner, is_loser per group."""
    log.info("Computing forward 20-day returns …")

    results = []
    for symbol, grp in df.groupby("symbol", sort=False):
        grp = grp.sort_values("date").copy()
        closes = grp["close"].values
        n = len(closes)

        fwd_ret = np.full(n, np.nan)
        exit_px = np.full(n, np.nan)
        if n > FORWARD_DAYS:
            fwd_ret[: n - FORWARD_DAYS] = (
                (closes[FORWARD_DAYS:] - closes[: n - FORWARD_DAYS])
                / closes[: n - FORWARD_DAYS]
                * 100
            )
            exit_px[: n - FORWARD_DAYS] = closes[FORWARD_DAYS:]

        grp["forward_return_pct"] = fwd_ret
        grp["exit_price_d20"] = exit_px
        results.append(grp)

    out = pd.concat(results, ignore_index=True)
    out = out[out["forward_return_pct"].notna()].copy()

    # keep only definitive winners / losers — drop ambiguous (0 ≤ return < 10)
    out["is_winner"] = out["forward_return_pct"] >= WIN_THRESHOLD
    out["is_loser"]  = out["forward_return_pct"] < LOSS_THRESHOLD
    out = out[out["is_winner"] | out["is_loser"]].copy()
    out.reset_index(drop=True, inplace=True)

    log.info("Events after filtering: %d  (winners=%d, losers=%d)",
             len(out), out["is_winner"].sum(), out["is_loser"].sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Overlap detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap(events: pd.DataFrame) -> pd.DataFrame:
    """
    For winner events (independently for loser events):
    - is_first_trigger: no prior winner within last 20 calendar days
    - overlap_count:    count of other events within ±20 days
    """
    log.info("Computing overlap flags …")

    is_first_list   = np.zeros(len(events), dtype=bool)
    overlap_counts  = np.zeros(len(events), dtype=int)

    for label_col in ("is_winner", "is_loser"):
        subset = events[events[label_col]].copy()
        if subset.empty:
            continue

        for symbol, grp in subset.groupby("symbol", sort=False):
            grp = grp.sort_values("date")
            dates = [d for d in grp["date"]]
            idxs  = list(grp.index)
            n     = len(dates)

            for i in range(n):
                d_i = dates[i]

                # is_first_trigger: no earlier event within last 20 calendar days
                first = True
                for j in range(i - 1, -1, -1):
                    if (d_i - dates[j]).days > FORWARD_DAYS:
                        break
                    first = False
                    break
                if first:
                    is_first_list[idxs[i]] = True

                # overlap_count: events within ±20 calendar days (excluding self)
                count = 0
                for j in range(n):
                    if j == i:
                        continue
                    if abs((d_i - dates[j]).days) <= FORWARD_DAYS:
                        count += 1
                overlap_counts[idxs[i]] = count

    events = events.copy()
    events["is_first_trigger"] = is_first_list
    events["overlap_count"]    = overlap_counts
    return events


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Indicator computation (pure numpy, no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Vectorised EMA — returns array same length as arr."""
    k = 2.0 / (period + 1)
    out = np.full_like(arr, np.nan, dtype=float)
    # seed on first valid value
    start = 0
    while start < len(arr) and np.isnan(arr[start]):
        start += 1
    if start >= len(arr):
        return out
    out[start] = arr[start]
    for i in range(start + 1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing — used for RSI and ATR averages."""
    k = 1.0 / period
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    first = valid[0]
    # seed = simple mean of first `period` valid values
    out[first + period - 1] = np.nanmean(arr[first: first + period])
    for i in range(first + period, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]
        else:
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def compute_indicators_for_symbol(
    closes: np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    volumes: np.ndarray,
) -> dict:
    """
    Compute all indicators on the full history array.
    Returns a dict of 1-D arrays (same length as closes).
    All operations are strictly causal — index i uses only [0..i].
    """
    n = len(closes)

    # ── RSI(14) via Wilder smoothing ────────────────────────────────────────
    delta  = np.diff(closes, prepend=np.nan)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder_smooth(gains,  14)
    avg_loss = _wilder_smooth(losses, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs   = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi  = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))

    # ── EMAs ────────────────────────────────────────────────────────────────
    ema20  = _ema(closes, 20)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200)
    with np.errstate(invalid="ignore"):
        price_vs_ema200 = np.where(
            ema200 > 0, (closes - ema200) / ema200 * 100, np.nan
        )

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12       = _ema(closes, 12)
    ema26       = _ema(closes, 26)
    macd_line   = ema12 - ema26
    macd_signal = _ema(macd_line, 9)
    macd_hist   = macd_line - macd_signal

    # ── Bollinger Bands (20, 2σ) ─────────────────────────────────────────────
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_mid   = np.full(n, np.nan)
    for i in range(19, n):
        w     = closes[i - 19: i + 1]
        mu    = np.mean(w)
        sigma = np.std(w, ddof=0)
        bb_mid[i]   = mu
        bb_upper[i] = mu + 2 * sigma
        bb_lower[i] = mu - 2 * sigma

    with np.errstate(invalid="ignore", divide="ignore"):
        band_range = bb_upper - bb_lower
        bb_pct_b   = np.where(band_range > 0,
                              (closes - bb_lower) / band_range, np.nan)
        bb_bw      = np.where(bb_mid > 0, band_range / bb_mid * 100, np.nan)

    # ── OBV slope (5-day linear regression, normalised) ──────────────────────
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    obv_slope = np.full(n, np.nan)
    for i in range(4, n):
        w   = obv[i - 4: i + 1]
        mu  = np.mean(w)
        if mu == 0:
            continue
        x   = np.arange(5, dtype=float)
        xm  = x - x.mean()
        wm  = w - mu
        denom = np.dot(xm, xm)
        if denom > 0:
            obv_slope[i] = np.dot(xm, wm) / denom / (abs(mu) + 1e-10)

    # ── ATR(14) via Wilder smoothing ─────────────────────────────────────────
    prev_close = np.roll(closes, 1)
    prev_close[0] = np.nan
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close))
    )
    atr14 = _wilder_smooth(tr, 14)

    # ── Volume ratio 20d ────────────────────────────────────────────────────
    vol_ratio = np.full(n, np.nan)
    for i in range(19, n):
        mean20 = np.mean(volumes[i - 19: i + 1])
        if mean20 > 0:
            vol_ratio[i] = volumes[i] / mean20

    return {
        "rsi14":              rsi,
        "ema20":              ema20,
        "ema50":              ema50,
        "ema200":             ema200,
        "price_vs_ema200_pct": price_vs_ema200,
        "macd_line":          macd_line,
        "macd_signal":        macd_signal,
        "macd_hist":          macd_hist,
        "bb_pct_b":           bb_pct_b,
        "bb_bandwidth":       bb_bw,
        "obv_slope_5d":       obv_slope,
        "atr14":              atr14,
        "volume_ratio_20d":   vol_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6a — Load broker_flow (bulk, Neon)
# ─────────────────────────────────────────────────────────────────────────────

def load_broker_flow() -> dict:
    """
    Returns dict keyed by (symbol, date_str) → {net_flow_1d, acc_qty_1d, ...}
    """
    log.info("Loading broker_flow from Neon (>= %s) …", BROKER_CUTOFF)
    rows = run_raw_sql(
        """
        SELECT date, symbol,
               net_flow_1d, acc_qty_1d, acc_top_broker_pct_1d,
               acc_top_broker_1d, flow_bias_1d
        FROM broker_flow
        WHERE date >= %s
        """,
        (BROKER_CUTOFF,),
    )

    def _f(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    lookup = {}
    for r in rows:
        key = (r["symbol"], str(r["date"]))
        lookup[key] = {
            "net_flow_1d":           _f(r["net_flow_1d"]),
            "acc_qty_1d":            _f(r["acc_qty_1d"]),
            "acc_top_broker_pct_1d": _f(r["acc_top_broker_pct_1d"]),
            "acc_top_broker_1d":     r["acc_top_broker_1d"],
            "flow_bias_1d":          r["flow_bias_1d"],
        }

    log.info("broker_flow lookup: %d entries", len(lookup))
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6b — Load floorsheet broker presence (LOCAL postgres, bulk)
# ─────────────────────────────────────────────────────────────────────────────

def load_floorsheet_broker_presence() -> dict:
    """
    Bulk-loads floorsheet for validated brokers (>= 2023-07-03).
    Returns dict keyed by (symbol, date_str) → set of (role, broker_id) tuples.

    broker_58 (Naasa): bullish as SELLER — presence checked via seller_broker_id.
    """
    log.info("Loading floorsheet broker presence from local postgres …")
    try:
        conn = psycopg2.connect(LOCAL_PG_DSN)
        cur  = conn.cursor(cursor_factory=RealDictCursor)

        # Single bulk query — only fetch the columns/brokers we need
        cur.execute(
            """
            SELECT date, symbol, buyer_broker_id, seller_broker_id
            FROM floorsheet
            WHERE date >= %s
              AND (
                  buyer_broker_id  IN ('38', '56', '49', '48')
               OR seller_broker_id IN ('58')
              )
            """,
            (BROKER_CUTOFF,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        log.error("Floorsheet bulk load failed: %s", exc)
        return {}

    # Build (symbol, date_str) → set of (role, broker_id)
    presence: dict = {}
    for r in rows:
        key = (r["symbol"], str(r["date"]))
        if key not in presence:
            presence[key] = set()
        if r["buyer_broker_id"] in ("38", "56", "49", "48"):
            presence[key].add(("buyer_broker_id", r["buyer_broker_id"]))
        if r["seller_broker_id"] == "58":
            presence[key].add(("seller_broker_id", "58"))

    log.info("Floorsheet presence loaded: %d (symbol, date) keys", len(presence))
    return presence


def _broker_presence_window(
    symbol: str,
    entry_date: date,
    trading_dates_for_symbol: list,  # sorted list of date objects for symbol
    floorsheet_presence: dict,
) -> dict:
    """
    For a given symbol + entry_date, find the 10 most recent trading dates
    in floorsheet that are <= entry_date, then check each validated broker.
    Returns dict of {broker_key: True/False}.
    """
    # trading_dates_for_symbol is sorted ascending
    # find the 10 most recent dates <= entry_date
    window_dates = [d for d in trading_dates_for_symbol if d <= entry_date]
    window_dates = window_dates[-10:]  # last 10 trading days

    result = {}
    for broker_key, (role, broker_id) in VALIDATED_BROKERS.items():
        found = False
        for wd in window_dates:
            key = (symbol, str(wd))
            if key in floorsheet_presence:
                if (role, broker_id) in floorsheet_presence[key]:
                    found = True
                    break
        result[broker_key + "_present_10d"] = found
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load price history ───────────────────────────────────────────
    price_df = load_price_history()

    if dry_run:
        symbols_all = sorted(price_df["symbol"].unique())[:20]
        price_df = price_df[price_df["symbol"].isin(symbols_all)].copy()
        log.info("DRY-RUN: restricted to %d symbols", len(symbols_all))

    # ── Step 2: Forward returns + labels ────────────────────────────────────
    events = compute_forward_returns(price_df)

    # ── Step 3: Overlap detection ────────────────────────────────────────────
    events = compute_overlap(events)

    # ── Step 4: Regime labels ────────────────────────────────────────────────
    log.info("Assigning regime labels …")
    events["regime"] = events["date"].apply(assign_regime)

    # ── Step 5: Indicators ───────────────────────────────────────────────────
    log.info("Computing indicators per symbol …")

    INDICATOR_COLS = [
        "rsi14", "ema20", "ema50", "ema200", "price_vs_ema200_pct",
        "macd_line", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "obv_slope_5d", "atr14", "volume_ratio_20d",
    ]
    for col in INDICATOR_COLS:
        events[col] = np.nan

    skipped_insufficient = []
    processed_event_count = 0

    # Pre-index: for each symbol, build a dict date → row_position in price_df
    sym_price_map = {}
    for sym, grp in price_df.groupby("symbol", sort=False):
        grp_sorted = grp.sort_values("date")
        sym_price_map[sym] = grp_sorted

    event_symbols = events["symbol"].unique()

    for sym in event_symbols:
        if sym not in sym_price_map:
            continue
        sym_hist = sym_price_map[sym]
        sym_dates  = sym_hist["date"].values          # numpy array of date objects
        sym_closes  = sym_hist["close"].values.astype(float)
        sym_highs   = sym_hist["high"].fillna(sym_hist["close"]).values.astype(float)
        sym_lows    = sym_hist["low"].fillna(sym_hist["close"]).values.astype(float)
        sym_vols    = sym_hist["volume"].fillna(0).values.astype(float)

        try:
            ind_arrays = compute_indicators_for_symbol(
                sym_closes, sym_highs, sym_lows, sym_vols
            )
        except Exception as exc:
            log.warning("Indicator compute failed for %s: %s", sym, exc)
            continue

        # For each event of this symbol, look up the indicator at entry date
        sym_events = events[events["symbol"] == sym]
        for idx, row in sym_events.iterrows():
            entry_date = row["date"]

            # find position in sym_hist at entry_date
            pos_arr = np.where(sym_dates == entry_date)[0]
            if len(pos_arr) == 0:
                continue
            pos = pos_arr[0]

            if pos < MIN_HISTORY_ROWS - 1:
                skipped_insufficient.append((sym, str(entry_date)))
                for col in INDICATOR_COLS:
                    events.at[idx, col] = np.nan
                continue

            for col in INDICATOR_COLS:
                events.at[idx, col] = ind_arrays[col][pos]

            processed_event_count += 1
            if processed_event_count % 10_000 == 0:
                elapsed = time.time() - t0
                log.info("  … %d events processed  (%.0fs elapsed)",
                         processed_event_count, elapsed)

    log.info("Indicators done. Skipped %d events (insufficient history).",
             len(skipped_insufficient))

    # ── Step 6: Broker data attachment ───────────────────────────────────────
    broker_cutoff_date = date.fromisoformat(BROKER_CUTOFF)

    broker_flow_lookup    = load_broker_flow()
    floorsheet_presence   = load_floorsheet_broker_presence()

    # Build per-symbol sorted list of floorsheet trading dates for 10-day window
    log.info("Building per-symbol floorsheet trading date index …")
    fs_sym_dates: dict = {}
    for (sym, date_str), _ in floorsheet_presence.items():
        d = date.fromisoformat(date_str)
        if sym not in fs_sym_dates:
            fs_sym_dates[sym] = set()
        fs_sym_dates[sym].add(d)
    for sym in fs_sym_dates:
        fs_sym_dates[sym] = sorted(fs_sym_dates[sym])

    BROKER_FLOW_COLS = [
        "net_flow_1d", "acc_qty_1d", "acc_top_broker_pct_1d",
        "acc_top_broker_1d", "flow_bias_1d",
    ]
    BROKER_PRESENCE_COLS = [
        "broker_38_present_10d", "broker_56_present_10d",
        "broker_49_present_10d", "broker_48_present_10d",
        "broker_58_present_10d",
    ]

    for col in BROKER_FLOW_COLS + BROKER_PRESENCE_COLS:
        events[col] = None

    log.info("Attaching broker data to events >= %s …", BROKER_CUTOFF)
    broker_mask = events["date"] >= broker_cutoff_date
    broker_events = events[broker_mask]
    attach_count  = 0

    for idx, row in broker_events.iterrows():
        sym        = row["symbol"]
        entry_date = row["date"]
        date_str   = str(entry_date)

        # 6a: broker_flow
        bf_key = (sym, date_str)
        bf     = broker_flow_lookup.get(bf_key)
        if bf:
            for col in BROKER_FLOW_COLS:
                events.at[idx, col] = bf.get(col)

        # 6b: floorsheet broker presence (10 prior trading days)
        sym_td = fs_sym_dates.get(sym, [])
        presence = _broker_presence_window(sym, entry_date, sym_td, floorsheet_presence)
        for col in BROKER_PRESENCE_COLS:
            events.at[idx, col] = presence.get(col, None)

        attach_count += 1
        if attach_count % 10_000 == 0:
            log.info("  … broker attachment: %d events", attach_count)

    log.info("Broker attachment complete for %d events.", attach_count)

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    log.info("Writing output files …")

    # Round floats to 4dp
    float_cols = INDICATOR_COLS + ["forward_return_pct", "entry_price",
                                   "exit_price_d20", "net_flow_1d",
                                   "acc_qty_1d", "acc_top_broker_pct_1d"]
    for col in float_cols:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce").round(4)

    # Rename close → entry_price for clarity in output
    events = events.rename(columns={"close": "entry_price"})

    ALL_COLS = [
        "symbol", "date", "regime", "entry_price", "exit_price_d20",
        "forward_return_pct", "is_winner", "is_loser",
        "is_first_trigger", "overlap_count",
        "rsi14", "ema20", "ema50", "ema200", "price_vs_ema200_pct",
        "macd_line", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_bandwidth", "obv_slope_5d", "atr14", "volume_ratio_20d",
    ]
    events_all = events[ALL_COLS]
    events_all.to_csv(OUT_ALL, index=False)
    log.info("Written: %s  (%d rows)", OUT_ALL, len(events_all))

    BROKER_COLS = ALL_COLS + BROKER_FLOW_COLS + BROKER_PRESENCE_COLS
    events_broker = events[events["date"] >= broker_cutoff_date][BROKER_COLS]
    events_broker.to_csv(OUT_BROKER, index=False)
    log.info("Written: %s  (%d rows)", OUT_BROKER, len(events_broker))

    # ── Step 8: Summary stats ─────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "═" * 60)
    print("BACKTEST EXTRACTOR — SUMMARY")
    print("═" * 60)
    print(f"Total events processed   : {len(events_all):>10,}")
    print(f"  Winner events          : {events_all['is_winner'].sum():>10,}")
    print(f"  Loser events           : {events_all['is_loser'].sum():>10,}")
    print()
    print("Events per regime:")
    for regime, cnt in events_all.groupby("regime").size().sort_values(ascending=False).items():
        print(f"  {regime:<35} {cnt:>8,}")
    print()
    print(f"Broker file rows (>=2023): {len(events_broker):>10,}")
    print(f"Events skipped (insuff.)  : {len(skipped_insufficient):>10,}")
    if skipped_insufficient[:5]:
        print("  (first 5):", skipped_insufficient[:5])
    print(f"Time taken               : {elapsed:>9.1f}s")
    print("═" * 60)

    if dry_run:
        print("\nDRY-RUN — sample rows (events_all):")
        print(events_all.head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract winner/loser backtest events with indicators + broker signals."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 20 symbols and print 5 sample rows.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
