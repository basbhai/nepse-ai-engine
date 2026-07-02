"""
ml_feature_builder.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — ML Feature Builder

Builds a labeled feature matrix for ML training.

For every (symbol, window_end_date) pair:
  - 30-day rolling window of raw floorsheet → broker-level features
  - Price indicators at window end → technical features
  - Market regime at window end
  - Sector
  - Label: did price rise ≥15% within 45 trading days after window end?

Feature categories (Option B — broker-level):
  Floorsheet broker features (from raw floorsheet):
    - absorption_on_down_days    : top3 net buyers' buy % on down-close days
    - consistent_buyer_count     : brokers net-positive ≥60% of days
    - bp_slope                   : buyer_pressure linear slope over window
    - bc_slope                   : broker_concentration slope (tightening)
    - cross_broker_transfers     : matched buy/sell pairs >5k units same day
    - top_buyer_net_pct          : top buyer net / total volume
    - institutional_day_pct      : % days with institutional_flag=true
    - vol_acceleration           : last-10d avg vol / first-10d avg vol
    - bp_mean, bc_mean           : avg buyer_pressure, broker_concentration
    - large_order_pct_mean       : avg large_order_pct over window

  Price / technical features (from price_history):
    - rsi_14                     : RSI at window end
    - ema_trend                  : close vs EMA20/50/200
    - bb_pct_b                   : Bollinger %B at window end
    - atr_pct                    : ATR as % of close
    - obv_divergence             : OBV slope vs price slope (last 20d)
    - net_60d_return             : price change over prior 60 trading days
    - vol_ratio_15d              : today volume vs 15d ago volume
    - price_52w_position         : % of 52W range
    - macd_histogram             : MACD histogram value

  Regime / sector:
    - market_regime              : 1=bull (NEPSE > 60d MA), 0=bear
    - sector_encoded             : integer-encoded sector

  Label:
    - label                      : 1 if price rose ≥15% within 45 trading
                                   days after window_end_date, else 0

Output: stat_method/output/ml_features.parquet
        stat_method/output/ml_features_meta.json

Usage:
    cd ~/nepse-engine
    python stat_method/ml_feature_builder.py
    python stat_method/ml_feature_builder.py --from-date 2024-01-01
    python stat_method/ml_feature_builder.py --dry-run   # 50 symbols only
"""

import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FEAT] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
FS_START        = "2023-07-01"   # floorsheet available from
WINDOW_DAYS     = 30             # rolling floorsheet window (trading days)
LABEL_HORIZON   = 45             # trading days forward for label
LABEL_THRESHOLD = 0.15           # ≥15% gain = positive label
STEP_DAYS       = 5              # generate a sample every N trading days per symbol
MIN_FS_DAYS     = 15             # min floorsheet days in window to include sample
ASSUMED_SHARES  = 10_000_000     # for float % calc


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_price_history() -> dict[str, pd.DataFrame]:
    """Load full OHLCV for all symbols. Returns symbol → DataFrame."""
    log.info("Loading price_history...")
    with _db() as cur:
        cur.execute("""
            SELECT
                symbol, date,
                CASE WHEN open   IS NOT NULL AND open   ~ '^[0-9]+\\.?[0-9]*$' THEN open::float   ELSE NULL END AS open,
                CASE WHEN high   IS NOT NULL AND high   ~ '^[0-9]+\\.?[0-9]*$' THEN high::float   ELSE NULL END AS high,
                CASE WHEN low    IS NOT NULL AND low    ~ '^[0-9]+\\.?[0-9]*$' THEN low::float    ELSE NULL END AS low,
                COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close,
                CASE WHEN volume IS NOT NULL AND volume ~ '^[0-9]+\\.?[0-9]*$' THEN volume::float ELSE 0    END AS volume
            FROM price_history
            WHERE close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """)
        rows = cur.fetchall()

    log.info("  %d price rows loaded", len(rows))
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper()

    result = {}
    for sym, grp in df.groupby("symbol"):
        g = grp.sort_values("date").reset_index(drop=True)
        g["open"]   = g["open"].fillna(g["close"])
        g["high"]   = g["high"].fillna(g["close"])
        g["low"]    = g["low"].fillna(g["close"])
        g["volume"] = g["volume"].fillna(0)
        result[sym] = g
    log.info("  %d symbols loaded", len(result))
    return result


def load_floorsheet_signals() -> dict[str, pd.DataFrame]:
    """Load precomputed floorsheet_signals. Returns symbol → DataFrame."""
    log.info("Loading floorsheet_signals...")
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date,
                   buyer_pressure::float    AS bp,
                   seller_pressure::float   AS sp,
                   broker_concentration::float AS bc,
                   institutional_flag,
                   large_order_pct::float   AS lop,
                   total_volume::float      AS fs_vol,
                   total_trades::float      AS fs_trades
            FROM floorsheet_signals
            WHERE date >= %s
              AND buyer_pressure IS NOT NULL AND buyer_pressure != ''
              AND broker_concentration IS NOT NULL AND broker_concentration != ''
        """, (FS_START,))
        rows = cur.fetchall()

    log.info("  %d floorsheet_signals rows loaded", len(rows))
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper()
    df["inst"] = df["institutional_flag"].astype(str).str.lower() == "true"

    result = {}
    for sym, grp in df.groupby("symbol"):
        result[sym] = grp.sort_values("date").reset_index(drop=True)
    log.info("  %d symbols in floorsheet_signals", len(result))
    return result


def load_raw_floorsheet() -> dict[str, pd.DataFrame]:
    """
    DEPRECATED — do not call. Returns empty dict.
    Raw floorsheet is now loaded per-symbol on demand via
    load_raw_floorsheet_for_symbol() to avoid OOM.
    """
    log.info("Raw floorsheet: using per-symbol loading (24M rows — never load all at once)")
    return {}


def load_raw_floorsheet_for_symbol(symbol: str) -> pd.DataFrame:
    """
    Load raw floorsheet for a single symbol using local DB connection.
    Falls back to _db() if local unavailable.
    """
    return _load_raw_fs_batch([symbol]).get(symbol, pd.DataFrame())


def _local_db():
    """
    Direct psycopg2 connection to local PostgreSQL.
    Much faster than Neon for bulk reads — no network round-trip.
    """
    import psycopg2
    import psycopg2.extras
    import os

    # Try local first, fall back to DATABASE_URL
    local_url = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    db_url    = os.environ.get("DATABASE_URL", local_url)
    # Prefer local if DATABASE_URL points to Neon
    if "neon" in db_url or "neon.tech" in db_url:
        db_url = local_url

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def _load_raw_fs_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load raw floorsheet for a batch of symbols in ONE query.
    Returns symbol → DataFrame.
    Batch size of 20 symbols = one query per 20 symbols instead of one per symbol.
    """
    if not symbols:
        return {}

    import psycopg2.extras

    try:
        conn = _local_db()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        placeholders = ",".join(["%s"] * len(symbols))
        cur.execute(f"""
            SELECT symbol, date,
                   buyer_broker_id, seller_broker_id,
                   quantity::float AS qty,
                   amount::float   AS amount
            FROM floorsheet
            WHERE symbol IN ({placeholders})
              AND date >= %s
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
              AND quantity::float > 0
            ORDER BY symbol, date ASC
        """, symbols + [FS_START])
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        log.warning("Local DB batch load failed (%s), falling back to Neon per-symbol", e)
        # Fallback: use _db() per symbol
        result = {}
        for sym in symbols:
            with _db() as cur2:
                cur2.execute("""
                    SELECT date, buyer_broker_id, seller_broker_id,
                           quantity::float AS qty, amount::float AS amount
                    FROM floorsheet
                    WHERE symbol = %s AND date >= %s
                      AND quantity IS NOT NULL AND quantity != ''
                      AND quantity ~ '^[0-9]+\\.?[0-9]*$'
                      AND quantity::float > 0
                    ORDER BY date ASC
                """, (sym, FS_START))
                sym_rows = cur2.fetchall()
            if sym_rows:
                df = pd.DataFrame([dict(r) for r in sym_rows])
                df["date"] = pd.to_datetime(df["date"])
                df["buyer_broker_id"]  = df["buyer_broker_id"].astype(str).str.strip()
                df["seller_broker_id"] = df["seller_broker_id"].astype(str).str.strip()
                result[sym] = df.sort_values("date").reset_index(drop=True)
        return result

    if not rows:
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"]   = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper()
    df["buyer_broker_id"]  = df["buyer_broker_id"].astype(str).str.strip()
    df["seller_broker_id"] = df["seller_broker_id"].astype(str).str.strip()

    result = {}
    for sym, grp in df.groupby("symbol"):
        result[sym] = grp.drop(columns="symbol").sort_values("date").reset_index(drop=True)
    return result


def load_nepse_regime() -> pd.Series:
    """Returns date → regime (1=bull, 0=bear) based on close vs 60d MA."""
    log.info("Loading NEPSE regime...")
    with _db() as cur:
        cur.execute("""
            SELECT date, current_value::float AS close
            FROM nepse_indices WHERE index_id = '58'
              AND current_value IS NOT NULL AND current_value != ''
              AND current_value ~ '^[0-9]+\\.?[0-9]*$'
            ORDER BY date ASC
        """)
        rows = cur.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["ma60"] = df["close"].rolling(60).mean()
    df["regime"] = (df["close"] > df["ma60"]).astype(int)
    return df["regime"]


def load_sectors() -> dict[str, str]:
    with _db() as cur:
        cur.execute("SELECT symbol, sectorname FROM share_sectors")
        rows = cur.fetchall()
    return {str(r["symbol"]).upper(): r["sectorname"] for r in rows}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR MATH
# ══════════════════════════════════════════════════════════════════════════════

def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = values - values.mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return np.nan
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains[:period].mean()
    avg_l  = losses[:period].mean()
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 2)


def calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
    if len(closes) < period:
        return np.array([])
    k = 2 / (period + 1)
    emas = [closes[:period].mean()]
    for c in closes[period:]:
        emas.append(c * k + emas[-1] * (1 - k))
    return np.array(emas)


def calc_bb_pct_b(closes: np.ndarray, period: int = 20) -> float:
    if len(closes) < period:
        return np.nan
    w   = closes[-period:]
    mid = w.mean()
    std = w.std()
    if std == 0:
        return 0.5
    upper = mid + 2 * std
    lower = mid - 2 * std
    return float((closes[-1] - lower) / (upper - lower))


def calc_atr(highs, lows, closes, period=14) -> float:
    if len(closes) < period + 1:
        return np.nan
    trs = [max(highs[i] - lows[i],
               abs(highs[i] - closes[i-1]),
               abs(lows[i]  - closes[i-1]))
           for i in range(1, len(closes))]
    atr = np.mean(trs[:period])
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 4)


def calc_macd_hist(closes: np.ndarray) -> float:
    e12 = calc_ema(closes, 12)
    e26 = calc_ema(closes, 26)
    if len(e12) == 0 or len(e26) == 0:
        return np.nan
    diff_len = len(e12) - len(e26)
    e12a = e12[diff_len:] if diff_len >= 0 else e12
    e26a = e26[-diff_len:] if diff_len < 0 else e26
    macd = e12a - e26a
    if len(macd) < 9:
        return np.nan
    signal = calc_ema(macd, 9)
    if len(signal) == 0:
        return np.nan
    return float(macd[-1] - signal[-1])


def calc_obv_slope(closes: np.ndarray, volumes: np.ndarray,
                   window: int = 20) -> float:
    if len(closes) < window + 1:
        return 0.0
    c = closes[-(window+1):]
    v = volumes[-(window+1):]
    obv = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            obv.append(obv[-1] + v[i])
        elif c[i] < c[i-1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    return _slope(np.array(obv))


# ══════════════════════════════════════════════════════════════════════════════
# BROKER-LEVEL FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_broker_features(
    raw_fs_window: pd.DataFrame,
    fs_sig_window: pd.DataFrame,
    price_window:  pd.DataFrame,
) -> dict:
    """
    Compute all broker-level and floorsheet features for a 30-day window.
    raw_fs_window : raw floorsheet rows in window
    fs_sig_window : floorsheet_signals rows in window
    price_window  : price_history rows in window
    """
    feats = {}

    # ── From floorsheet_signals ───────────────────────────────────────────────
    if len(fs_sig_window) > 0:
        feats["bp_mean"]          = fs_sig_window["bp"].mean()
        feats["bc_mean"]          = fs_sig_window["bc"].mean()
        feats["lop_mean"]         = fs_sig_window["lop"].mean()
        feats["institutional_day_pct"] = fs_sig_window["inst"].mean()
        feats["bp_slope"]         = _slope(fs_sig_window["bp"].values)
        feats["bc_slope"]         = _slope(fs_sig_window["bc"].values)
        # Volume acceleration: last half vs first half
        n = len(fs_sig_window)
        if n >= 10:
            half = n // 2
            early = fs_sig_window["fs_vol"].iloc[:half].mean()
            late  = fs_sig_window["fs_vol"].iloc[half:].mean()
            feats["vol_acceleration"] = late / early if early > 0 else 1.0
        else:
            feats["vol_acceleration"] = 1.0
    else:
        for k in ["bp_mean","bc_mean","lop_mean","institutional_day_pct",
                  "bp_slope","bc_slope","vol_acceleration"]:
            feats[k] = np.nan

    # ── From raw floorsheet ───────────────────────────────────────────────────
    if len(raw_fs_window) == 0 or len(price_window) == 0:
        for k in ["absorption_on_down_days","consistent_buyer_count",
                  "cross_broker_transfers","top_buyer_net_pct"]:
            feats[k] = np.nan
        return feats

    # Daily net per broker
    daily_buy  = raw_fs_window.groupby(["date","buyer_broker_id"])["qty"].sum()
    daily_sell = raw_fs_window.groupby(["date","seller_broker_id"])["qty"].sum()

    # Broker net over full window
    broker_buy  = raw_fs_window.groupby("buyer_broker_id")["qty"].sum()
    broker_sell = raw_fs_window.groupby("seller_broker_id")["qty"].sum()
    broker_net  = broker_buy.subtract(broker_sell, fill_value=0)
    total_vol   = raw_fs_window["qty"].sum()

    # Exclude near-zero net brokers (market makers)
    broker_total = broker_buy.add(broker_sell, fill_value=0)
    net_ratio    = broker_net.abs() / (broker_total + 1)
    active_net   = broker_net[net_ratio > 0.05]

    top3_buyers  = active_net.nlargest(3).index.tolist() if len(active_net) >= 3 \
                   else active_net.nlargest(len(active_net)).index.tolist()

    # Absorption on down days
    price_window = price_window.copy()
    price_window["prev_close"] = price_window["close"].shift(1)
    price_window["down_day"]   = price_window["close"] < price_window["prev_close"]
    down_dates = set(price_window[price_window["down_day"]]["date"])

    down_fs = raw_fs_window[raw_fs_window["date"].isin(down_dates)]
    if len(down_fs) > 0 and len(top3_buyers) > 0:
        top3_buy_on_down = down_fs[down_fs["buyer_broker_id"].isin(top3_buyers)]["qty"].sum()
        total_buy_on_down= down_fs["qty"].sum()
        feats["absorption_on_down_days"] = top3_buy_on_down / total_buy_on_down \
                                           if total_buy_on_down > 0 else 0.0
    else:
        feats["absorption_on_down_days"] = 0.0

    # Consistent buyer count (net positive ≥60% of days they traded)
    consistent = 0
    for bid in active_net[active_net > 0].index:
        grp = daily_buy.xs(bid, level="buyer_broker_id") if bid in daily_buy.index.get_level_values(1) \
              else pd.Series(dtype=float)
        # Count days net positive
        buy_daily  = daily_buy.xs(bid, level="buyer_broker_id") \
                     if bid in daily_buy.index.get_level_values(1) else pd.Series(dtype=float)
        sell_daily = daily_sell.xs(bid, level="seller_broker_id") \
                     if bid in daily_sell.index.get_level_values(1) else pd.Series(dtype=float)
        net_daily  = buy_daily.subtract(sell_daily, fill_value=0)
        if len(net_daily) >= 5 and (net_daily > 0).mean() >= 0.60:
            consistent += 1
    feats["consistent_buyer_count"] = consistent

    # Cross-broker transfers (matched pairs same day, >5k units, within 10%)
    d_buy_large  = raw_fs_window[raw_fs_window["qty"] > 5000].groupby(
        ["date","buyer_broker_id"])["qty"].sum().reset_index()
    d_sell_large = raw_fs_window[raw_fs_window["qty"] > 5000].groupby(
        ["date","seller_broker_id"])["qty"].sum().reset_index()
    transfers = 0
    if len(d_buy_large) > 0 and len(d_sell_large) > 0:
        mg = d_buy_large.merge(d_sell_large, on="date",
                               suffixes=("_b","_s"))
        mg = mg[mg["buyer_broker_id"] != mg["seller_broker_id"]]
        if len(mg) > 0:
            mg["ratio"] = mg[["qty_b","qty_s"]].min(axis=1) / \
                          mg[["qty_b","qty_s"]].max(axis=1)
            transfers = int((mg["ratio"] >= 0.90).sum())
    feats["cross_broker_transfers"] = min(transfers, 20)  # cap outliers

    # Top buyer net as % of total volume
    top_net = float(active_net.nlargest(1).values[0]) if len(active_net) > 0 else 0.0
    feats["top_buyer_net_pct"] = top_net / total_vol if total_vol > 0 else 0.0

    return feats


def compute_price_features(price_df: pd.DataFrame, window_end_idx: int) -> dict:
    """Compute technical features up to window_end_idx (inclusive)."""
    feats = {}
    if window_end_idx < 20:
        return {k: np.nan for k in [
            "rsi_14","ema_trend","bb_pct_b","atr_pct","obv_slope",
            "net_60d_return","vol_ratio_15d","price_52w_position","macd_histogram"
        ]}

    closes  = price_df["close"].values[:window_end_idx + 1]
    highs   = price_df["high"].values[:window_end_idx + 1]
    lows    = price_df["low"].values[:window_end_idx + 1]
    volumes = price_df["volume"].values[:window_end_idx + 1]
    close_now = closes[-1]

    feats["rsi_14"]   = calc_rsi(closes)
    feats["bb_pct_b"] = calc_bb_pct_b(closes)
    feats["macd_histogram"] = calc_macd_hist(closes)

    atr = calc_atr(highs, lows, closes)
    feats["atr_pct"] = atr / close_now if (atr and close_now > 0) else np.nan

    # EMA trend: 0=below_all, 1=mixed, 2=above_all
    e20  = calc_ema(closes, 20)
    e50  = calc_ema(closes, 50)
    e200 = calc_ema(closes, 200)
    if len(e20) and len(e50) and len(e200):
        if close_now > e20[-1] > e50[-1] > e200[-1]:
            feats["ema_trend"] = 2
        elif close_now < e20[-1] < e50[-1] < e200[-1]:
            feats["ema_trend"] = 0
        else:
            feats["ema_trend"] = 1
    else:
        feats["ema_trend"] = np.nan

    # OBV slope
    feats["obv_slope"] = calc_obv_slope(closes, volumes)

    # Net 60d return
    if window_end_idx >= 60:
        c60 = closes[-61]
        feats["net_60d_return"] = (close_now - c60) / c60 if c60 > 0 else np.nan
    else:
        feats["net_60d_return"] = np.nan

    # Volume ratio 15d
    if window_end_idx >= 15:
        feats["vol_ratio_15d"] = volumes[-1] / volumes[-16] \
                                  if volumes[-16] > 0 else np.nan
    else:
        feats["vol_ratio_15d"] = np.nan

    # 52W position
    if window_end_idx >= 240:
        w52_hi = highs[-240:].max()
        w52_lo = lows[-240:].min()
    else:
        w52_hi = highs.max()
        w52_lo = lows.min()
    rng = w52_hi - w52_lo
    feats["price_52w_position"] = (close_now - w52_lo) / rng if rng > 0 else 0.5

    return feats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

SECTOR_MAP = {}

def build_features(
    symbols: list[str],
    price_data: dict,
    fs_sig_data: dict,
    regime: pd.Series,
    sectors: dict,
    from_date: str = None,
) -> pd.DataFrame:

    global SECTOR_MAP
    all_sectors = sorted(set(sectors.values()))
    SECTOR_MAP  = {s: i for i, s in enumerate(all_sectors)}

    all_rows   = []
    n_sym      = len(symbols)
    BATCH_SIZE = 20   # load floorsheet for 20 symbols per DB query

    # Pre-load floorsheet in batches
    sorted_syms = sorted(symbols)
    raw_fs_cache: dict[str, pd.DataFrame] = {}

    log.info("Pre-loading raw floorsheet in batches of %d...", BATCH_SIZE)
    for batch_start in range(0, len(sorted_syms), BATCH_SIZE):
        batch = sorted_syms[batch_start:batch_start + BATCH_SIZE]
        batch_data = _load_raw_fs_batch(batch)
        raw_fs_cache.update(batch_data)
        if (batch_start // BATCH_SIZE + 1) % 5 == 0:
            log.info("  Floorsheet batch %d / %d loaded",
                     batch_start // BATCH_SIZE + 1,
                     (len(sorted_syms) + BATCH_SIZE - 1) // BATCH_SIZE)
    log.info("Floorsheet pre-loaded for %d symbols", len(raw_fs_cache))

    for sym_i, symbol in enumerate(sorted_syms):
        if (sym_i + 1) % 50 == 0:
            log.info("  %d / %d symbols...", sym_i + 1, n_sym)

        price_df = price_data.get(symbol)
        fs_sig   = fs_sig_data.get(symbol)
        raw_fs   = raw_fs_cache.get(symbol)

        if price_df is None or len(price_df) < WINDOW_DAYS + LABEL_HORIZON + 5:
            continue
        if fs_sig is None:
            continue
        if raw_fs is None or raw_fs.empty:
            continue

        price_df = price_df.reset_index(drop=True)
        sector   = sectors.get(symbol, "Unknown")
        sec_enc  = SECTOR_MAP.get(sector, -1)

        # Iterate over window end positions (every STEP_DAYS trading days)
        # Start after WINDOW_DAYS, end before LABEL_HORIZON from last date
        start_idx = WINDOW_DAYS
        end_idx   = len(price_df) - LABEL_HORIZON - 1

        if from_date:
            from_dt = pd.Timestamp(from_date)
            # Find first index where date >= from_date
            start_idx = max(start_idx,
                int((price_df["date"] >= from_dt).idxmax()))

        for win_end in range(start_idx, end_idx + 1, STEP_DAYS):
            window_end_date = price_df["date"].iloc[win_end]

            # ── Filter: only windows where FS data is available ───────────────
            win_start_date = price_df["date"].iloc[win_end - WINDOW_DAYS]
            fs_win = fs_sig[
                (fs_sig["date"] >= win_start_date) &
                (fs_sig["date"] <= window_end_date)
            ]
            if len(fs_win) < MIN_FS_DAYS:
                continue

            raw_win = raw_fs[
                (raw_fs["date"] >= win_start_date) &
                (raw_fs["date"] <= window_end_date)
            ]
            price_win = price_df.iloc[max(0, win_end - WINDOW_DAYS):win_end + 1]

            # ── Broker features ───────────────────────────────────────────────
            broker_feats = compute_broker_features(raw_win, fs_win, price_win)

            # ── Price features ────────────────────────────────────────────────
            price_feats = compute_price_features(price_df, win_end)

            # ── Regime ────────────────────────────────────────────────────────
            regime_val = regime.get(window_end_date, np.nan)
            if pd.isna(regime_val):
                # Find nearest
                try:
                    regime_val = regime.asof(window_end_date)
                except Exception:
                    regime_val = np.nan

            # ── Label: ≥15% rise within next 45 trading days ─────────────────
            future_closes = price_df["close"].iloc[win_end + 1:win_end + LABEL_HORIZON + 1]
            base_close    = price_df["close"].iloc[win_end]
            if len(future_closes) < LABEL_HORIZON or base_close <= 0:
                continue

            max_future = future_closes.max()
            label      = int((max_future - base_close) / base_close >= LABEL_THRESHOLD)

            # Also compute actual max forward return for analysis
            max_return = (max_future - base_close) / base_close

            row = {
                "symbol":          symbol,
                "window_end_date": window_end_date,
                "sector":          sector,
                "sector_enc":      sec_enc,
                "market_regime":   int(regime_val) if not pd.isna(regime_val) else -1,
                "label":           label,
                "max_return_45d":  round(max_return, 4),
                "base_close":      round(base_close, 2),
                **broker_feats,
                **price_feats,
            }
            all_rows.append(row)

    log.info("Total samples generated: %d", len(all_rows))
    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Process first 50 symbols only")
    args = parser.parse_args()

    # Load all data
    price_data  = load_price_history()
    fs_sig_data = load_floorsheet_signals()
    regime      = load_nepse_regime()
    sectors     = load_sectors()

    symbols = sorted(
        set(price_data.keys()) &
        set(fs_sig_data.keys())
    )
    log.info("Symbols with all data sources: %d", len(symbols))

    if args.dry_run:
        symbols = symbols[:50]
        log.info("Dry run: processing %d symbols", len(symbols))

    # Build features
    log.info("Building feature matrix...")
    feat_df = build_features(
        symbols, price_data, fs_sig_data,
        regime, sectors,
        from_date=args.from_date,
    )

    if feat_df.empty:
        log.error("No features generated — check data availability")
        return

    # Summary
    log.info("\n=== FEATURE MATRIX SUMMARY ===")
    log.info("Total samples:    %d", len(feat_df))
    log.info("Unique symbols:   %d", feat_df["symbol"].nunique())
    log.info("Date range:       %s → %s",
             feat_df["window_end_date"].min().date(),
             feat_df["window_end_date"].max().date())
    log.info("Label=1 (positive): %d (%.1f%%)",
             feat_df["label"].sum(),
             feat_df["label"].mean() * 100)
    log.info("Label=0 (negative): %d (%.1f%%)",
             (feat_df["label"] == 0).sum(),
             (feat_df["label"] == 0).mean() * 100)
    log.info("Features:         %d",
             len([c for c in feat_df.columns
                  if c not in ["symbol","window_end_date","sector",
                               "label","max_return_45d","base_close"]]))

    log.info("\nFeature null rates:")
    feature_cols = [c for c in feat_df.columns
                    if c not in ["symbol","window_end_date","sector",
                                 "sector_enc","label","max_return_45d",
                                 "base_close","market_regime"]]
    for col in feature_cols:
        null_pct = feat_df[col].isna().mean() * 100
        if null_pct > 5:
            log.info("  %-35s %.1f%% null", col, null_pct)

    # Save
    out_dir  = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ml_features.parquet"
    feat_df.to_parquet(out_path, index=False)
    log.info("\nSaved to %s (%.1f MB)",
             out_path, out_path.stat().st_size / 1024 / 1024)

    # Meta
    meta = {
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples":   len(feat_df),
        "unique_symbols":  int(feat_df["symbol"].nunique()),
        "label_positive":  int(feat_df["label"].sum()),
        "label_negative":  int((feat_df["label"] == 0).sum()),
        "positive_rate":   round(feat_df["label"].mean(), 4),
        "date_range":      [str(feat_df["window_end_date"].min().date()),
                            str(feat_df["window_end_date"].max().date())],
        "feature_cols":    feature_cols,
        "sector_map":      SECTOR_MAP,
        "config": {
            "window_days":     WINDOW_DAYS,
            "label_horizon":   LABEL_HORIZON,
            "label_threshold": LABEL_THRESHOLD,
            "step_days":       STEP_DAYS,
        }
    }
    meta_path = out_dir / "ml_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()