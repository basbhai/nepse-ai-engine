#!/usr/bin/env python3
"""
accumulation_scanner.py
────────────────────────────────────────────────────────────────────────────
Pure accumulation detection using broker‑level floorsheet data.

For each symbol, computes an accumulation score (0‑1) over the most recent
WINDOW_DAYS trading window. The score is built from 7 sub‑signals:
  1. Top buyer net % of total volume
  2. Absorption on down days (top‑3 buyers’ share of down‑day volume)
  3. Number of consistent net‑buying brokers (≥60% net‑positive days)
  4. Institutional participation (fraction of days flagged institutional)
  5. Volume acceleration (last half vs first half of window)
  6. Buyer pressure slope (rising net buy ratio)
  7. Price restraint (return not too negative and not already explosive)

Output: sorted table of symbols with accumulation score, plus CSV/Parquet.

Usage:
    python stat_method/accumulation_scanner.py
    python stat_method/accumulation_scanner.py --window 30 --lookback 2025-06-01
    python stat_method/accumulation_scanner.py --top 20 --threshold 0.65
    python stat_method/accumulation_scanner.py --dry-run
"""

import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ACCUM] %(message)s",
)
log = logging.getLogger("accum_scanner")

# ── Config defaults ──────────────────────────────────────────────────────────
DEFAULT_WINDOW  = 45            # trading days
FS_START        = "2023-07-01"  # floorsheet data available from
MIN_FS_DAYS     = 20            # minimum floorsheet days in window to score
BATCH_SIZE      = 20            # symbols per DB batch load
ASSUMED_SHARES  = 10_000_000    # (not used but kept for compatibility)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE LOADERS (same as ml_feature_builder)
# ══════════════════════════════════════════════════════════════════════════════

def load_price_history() -> Dict[str, pd.DataFrame]:
    """Returns symbol → DataFrame with columns [date, open, high, low, close, volume]."""
    log.info("Loading price_history...")
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date,
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
        g["open"]   = g["open"].ffill().fillna(g["close"])
        g["high"]   = g["high"].ffill().fillna(g["close"])
        g["low"]    = g["low"].ffill().fillna(g["close"])
        g["volume"] = g["volume"].fillna(0)
        result[sym] = g
    log.info("  %d symbols loaded", len(result))
    return result


def load_floorsheet_signals() -> Dict[str, pd.DataFrame]:
    log.info("Loading floorsheet_signals...")
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date,
                   buyer_pressure::float    AS bp,
                   broker_concentration::float AS bc,
                   institutional_flag,
                   large_order_pct::float   AS lop,
                   total_volume::float      AS fs_vol
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


def load_raw_floorsheet_batch(symbols: list) -> Dict[str, pd.DataFrame]:
    """Load raw floorsheet for a batch of symbols (local DB, fast)."""
    if not symbols:
        return {}
    import psycopg2.extras
    try:
        conn = _local_db()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        placeholders = ",".join(["%s"] * len(symbols))
        cur.execute(f"""
            SELECT symbol, date,
                   buyer_broker_id, seller_broker_id,
                   quantity::float AS qty
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
        log.warning("Batch load failed: %s, falling back to per‑symbol Neon", e)
        result = {}
        for sym in symbols:
            with _db() as cur2:
                cur2.execute("""
                    SELECT date, buyer_broker_id, seller_broker_id,
                           quantity::float AS qty
                    FROM floorsheet
                    WHERE symbol = %s AND date >= %s
                      AND quantity IS NOT NULL AND quantity != ''
                      AND quantity ~ '^[0-9]+\\.?[0-9]*$'
                      AND quantity::float > 0
                    ORDER BY date ASC
                """, (sym, FS_START))
                rows2 = cur2.fetchall()
            if rows2:
                df_sym = pd.DataFrame([dict(r) for r in rows2])
                df_sym["date"] = pd.to_datetime(df_sym["date"])
                df_sym["buyer_broker_id"]  = df_sym["buyer_broker_id"].astype(str).str.strip()
                df_sym["seller_broker_id"] = df_sym["seller_broker_id"].astype(str).str.strip()
                result[sym] = df_sym.sort_values("date").reset_index(drop=True)
        return result

    if not rows:
        return {}
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].str.upper()
    df["buyer_broker_id"]  = df["buyer_broker_id"].astype(str).str.strip()
    df["seller_broker_id"] = df["seller_broker_id"].astype(str).str.strip()
    result = {}
    for sym, grp in df.groupby("symbol"):
        result[sym] = grp.drop(columns="symbol").sort_values("date").reset_index(drop=True)
    return result


def _local_db():
    import psycopg2
    import os
    local_url = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    db_url = os.environ.get("DATABASE_URL", local_url)
    if "neon" in db_url or "neon.tech" in db_url:
        db_url = local_url
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# ACCUMULATION SCORE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float) - np.arange(n, dtype=float).mean()
    y = values - values.mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def accumulation_score(
    raw_fs_window: pd.DataFrame,
    fs_sig_window: pd.DataFrame,
    price_window: pd.DataFrame,
) -> float:
    """
    Compute accumulation score (0‑1) for a trading window.
    Pure broker‑driven features.
    """
    # No data – zero score
    if raw_fs_window.empty or len(price_window) < 2:
        return 0.0

    total_vol = raw_fs_window["qty"].sum()
    if total_vol == 0:
        return 0.0

    # 1. Top buyer net %
    buy_grp  = raw_fs_window.groupby("buyer_broker_id")["qty"].sum()
    sell_grp = raw_fs_window.groupby("seller_broker_id")["qty"].sum()
    net = buy_grp.subtract(sell_grp, fill_value=0)
    top_net = float(net.nlargest(1).values[0]) if len(net) > 0 else 0.0
    top_net_pct = top_net / total_vol
    score_top = np.clip(top_net_pct / 0.15, 0, 1)   # 15% = excellent

    # 2. Absorption on down days
    price_window = price_window.copy()
    price_window["prev_close"] = price_window["close"].shift(1)
    price_window["down"] = price_window["close"] < price_window["prev_close"]
    down_dates = set(price_window[price_window["down"]]["date"])
    down_fs = raw_fs_window[raw_fs_window["date"].isin(down_dates)]
    if not down_fs.empty:
        # top3 net buyers (exclude near‑zero nets)
        total_act = buy_grp.add(sell_grp, fill_value=0)
        net_ratio = net.abs() / (total_act + 1)
        active_net = net[net_ratio > 0.05]
        top3 = active_net.nlargest(3).index.tolist()
        top3_buy = down_fs[down_fs["buyer_broker_id"].isin(top3)]["qty"].sum()
        total_buy_down = down_fs["qty"].sum()
        absorption = top3_buy / total_buy_down if total_buy_down > 0 else 0.0
    else:
        absorption = 0.0
    score_abs = np.clip(absorption / 0.5, 0, 1)   # 50% absorption strong

    # 3. Consistent buyer count
    daily_buy  = raw_fs_window.groupby(["date","buyer_broker_id"])["qty"].sum()
    daily_sell = raw_fs_window.groupby(["date","seller_broker_id"])["qty"].sum()
    consist_cnt = 0
    for bid in active_net[active_net > 0].index:
        try:
            b = daily_buy.xs(bid, level="buyer_broker_id")
        except KeyError:
            b = pd.Series(dtype=float)
        try:
            s = daily_sell.xs(bid, level="seller_broker_id")
        except KeyError:
            s = pd.Series(dtype=float)
        net_day = b.subtract(s, fill_value=0)
        if len(net_day) >= 5 and (net_day > 0).mean() >= 0.6:
            consist_cnt += 1
    mapping = {0:0.0, 1:0.4, 2:0.7}
    score_consist = mapping.get(min(consist_cnt, 3), 1.0)

    # 4. Institutional participation
    inst_pct = fs_sig_window["inst"].mean() if not fs_sig_window.empty else 0.0
    score_inst = np.clip(inst_pct / 0.4, 0, 1)

    # 5. Volume acceleration
    n = len(fs_sig_window)
    if n >= 10:
        half = n // 2
        early = fs_sig_window["fs_vol"].iloc[:half].mean()
        late  = fs_sig_window["fs_vol"].iloc[half:].mean()
        acc = late / early if early > 0 else 1.0
    else:
        acc = 1.0
    score_vol = np.clip((acc - 1.0) / 1.0, 0, 1)   # 2x = 1.0

    # 6. Buyer pressure slope
    if len(fs_sig_window) > 10 and "bp" in fs_sig_window.columns:
        bp_vals = fs_sig_window["bp"].values.astype(float)
        slope = _slope(bp_vals)
        # a positive slope of ~0.03 per day is strong
        score_bp = np.clip((slope + 0.01) / 0.03, 0, 1)
    else:
        score_bp = 0.0

    # 7. Price restraint
    ret = (price_window["close"].iloc[-1] - price_window["close"].iloc[0]) / price_window["close"].iloc[0]
    if -0.05 <= ret <= 0.15:
        score_price = 1.0
    elif ret < -0.05:
        score_price = max(0, 1 + ret / 0.2)
    else:  # ret > 0.15
        score_price = max(0, 1 - (ret - 0.15) / 0.3)

    # ── Weighted total ──
    weights = {
        "top_net": 0.20, "absorb": 0.25, "consist": 0.15,
        "inst": 0.10, "vol_acc": 0.10, "bp_slope": 0.15, "price": 0.05
    }
    total = (
        score_top * weights["top_net"] +
        score_abs * weights["absorb"] +
        score_consist * weights["consist"] +
        score_inst * weights["inst"] +
        score_vol * weights["vol_acc"] +
        score_bp * weights["bp_slope"] +
        score_price * weights["price"]
    )
    return round(float(total), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_date_with_data(symbols, price_data, window_days):
    """Find the latest trading date for which we can compute a window for all symbols."""
    common_dates = None
    for sym in symbols:
        dates = set(price_data[sym]["date"])
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates & dates
    if not common_dates:
        return None
    return max(common_dates)


def run_scan(args):
    # 1. Load all data
    price_data = load_price_history()
    fs_sig_data = load_floorsheet_signals()
    symbols = sorted(set(price_data.keys()) & set(fs_sig_data.keys()))
    log.info("Symbols with both price & FS signals: %d", len(symbols))

    if args.dry_run:
        symbols = symbols[:50]
        log.info("Dry‑run: processing %d symbols", len(symbols))

    # 2. Determine lookback date
    if args.lookback:
        lookback_date = pd.Timestamp(args.lookback)
    else:
        lookback_date = find_latest_date_with_data(symbols, price_data, args.window)
        if lookback_date is None:
            log.error("No common dates found across symbols.")
            sys.exit(1)
    log.info("Using window end date: %s", lookback_date.date())

    # 3. Pre‑load raw floorsheet for all symbols in batches
    sorted_syms = sorted(symbols)
    raw_fs_cache = {}
    log.info("Pre‑loading raw floorsheet batches...")
    for bstart in range(0, len(sorted_syms), BATCH_SIZE):
        batch = sorted_syms[bstart:bstart+BATCH_SIZE]
        batch_data = load_raw_floorsheet_batch(batch)
        raw_fs_cache.update(batch_data)
    log.info("Raw floorsheet loaded for %d symbols", len(raw_fs_cache))

    # 4. Scan each symbol
    results = []
    for sym in sorted_syms:
        price_df = price_data.get(sym)
        fs_sig_df = fs_sig_data.get(sym)
        raw_fs = raw_fs_cache.get(sym)

        if price_df is None or fs_sig_df is None or raw_fs is None or raw_fs.empty:
            continue

        # Find the index of the lookback date in price data
        price_df = price_df.reset_index(drop=True)
        mask = price_df["date"] <= lookback_date
        if not mask.any():
            continue
        win_end_idx = mask[::-1].idxmax()   # last True index

        # Need at least window_days of data before this date
        if win_end_idx < args.window - 1:
            continue

        win_start_idx = win_end_idx - args.window + 1
        window_start_date = price_df["date"].iloc[win_start_idx]
        window_end_date = price_df["date"].iloc[win_end_idx]

        # Filter floorsheet signals and raw FS for the window
        fs_win = fs_sig_df[(fs_sig_df["date"] >= window_start_date) & (fs_sig_df["date"] <= window_end_date)]
        if len(fs_win) < MIN_FS_DAYS:
            continue

        raw_win = raw_fs[(raw_fs["date"] >= window_start_date) & (raw_fs["date"] <= window_end_date)]
        if raw_win.empty:
            continue

        price_win = price_df.iloc[win_start_idx:win_end_idx + 1]

        score = accumulation_score(raw_win, fs_win, price_win)
        results.append({
            "symbol": sym,
            "window_end": window_end_date,
            "accumulation_score": score,
            "window_days": args.window,
        })

    if not results:
        log.error("No stocks could be scored with available data.")
        sys.exit(1)

    # 5. Build DataFrame and filter/sort
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("accumulation_score", ascending=False)

    if args.threshold is not None:
        result_df = result_df[result_df["accumulation_score"] >= args.threshold]

    if args.top_n:
        result_df = result_df.head(args.top_n)

    # 6. Output
    log.info("\n=== ACCUMULATION SCORE RESULTS ===")
    log.info("Window end date: %s", lookback_date.date())
    log.info("Scored symbols: %d", len(result_df))
    log.info("Top 10:\n%s", result_df.head(10).to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            result_df.to_csv(out_path, index=False)
        else:
            result_df.to_parquet(out_path, index=False)
        log.info("Saved results to %s", out_path)

    return result_df


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pure accumulation scanner using broker‑level floorsheet data"
    )
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Trading days in the window (default: 45)")
    parser.add_argument("--lookback", type=str, default=None,
                        help="Window end date YYYY-MM-DD (default: latest available)")
    parser.add_argument("--top", dest="top_n", type=int, default=None,
                        help="Show only top N stocks")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum accumulation score (0‑1) to include")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to .csv or .parquet file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 50 symbols only")

    args = parser.parse_args()

    if args.window < 10:
        log.error("Window too short – at least 10 trading days required.")
        sys.exit(1)

    run_scan(args)


if __name__ == "__main__":
    main()