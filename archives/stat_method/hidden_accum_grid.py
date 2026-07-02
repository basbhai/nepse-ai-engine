"""
hidden_accum_grid.py
────────────────────────────────────────────────────────────────────────────
Grid search over all threshold combinations for hidden accumulation signal.

Tests every combination of:
  - vol_ratio:     0.30, 0.40, 0.50, 0.60
  - price_range:   0.03, 0.05, 0.08, 0.10
  - min_streak:    5, 7, 10, 14

For each combination × broker × forward window (30/60/90d):
  - hit rate (% positive return)
  - avg return
  - median return
  - signal count

Output:
  stat_method/output/hidden_grid_YYYY-MM-DD.csv  — all combinations
  stat_method/output/hidden_grid_top_YYYY-MM-DD.csv — top combos by hit rate

Usage:
    cd ~/nepse-engine
    python stat_method/hidden_accum_grid.py
"""

import sys
import csv
import logging
from pathlib import Path
from datetime import date
from itertools import product

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GRID] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR   = Path(__file__).parent / "output"
FROM_DATE = "2023-07-01"
TO_DATE   = "2026-05-27"

BROKERS = {
    "38": "Dipshikha Dhitopatra",
    "56": "Sri Hari Securities",
    "49": "Online Securities",
    "48": "Trishakti Securities",
    "58": "Naasa Securities",
}

FORWARD_WINDOWS = [30, 60, 90]

# ── Grid parameters ───────────────────────────────────────────────────────────
VOL_RATIOS    = [0.30, 0.40, 0.50, 0.60]
PRICE_RANGES  = [0.03, 0.05, 0.08, 0.10]
MIN_STREAKS   = [5, 7, 10, 14]
# Total combinations: 4 × 4 × 4 = 64


# ══════════════════════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════════════════════

def _local_db():
    import psycopg2, os
    local = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    url   = os.environ.get("DATABASE_URL", local)
    if "neon" in url:
        url = local
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_volume(from_date: str, to_date: str) -> pd.DataFrame:
    """Daily total volume per symbol — all brokers aggregated."""
    log.info("Loading daily volume (all brokers)...")
    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT date::date AS date, symbol,
               SUM(quantity::float) AS total_volume
        FROM floorsheet
        WHERE date >= %s AND date <= %s
          AND quantity IS NOT NULL AND quantity != ''
          AND quantity ~ '^[0-9]+\\.?[0-9]*$'
        GROUP BY date::date, symbol
        ORDER BY symbol, date
    """, (from_date, to_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]         = pd.to_datetime(df["date"])
    df["total_volume"] = df["total_volume"].astype(float)
    log.info("  Loaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol","date"]).reset_index(drop=True)


def load_broker_daily(from_date: str, to_date: str) -> pd.DataFrame:
    broker_ids = list(BROKERS.keys())
    log.info("Loading 5-broker daily data...")
    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    ph   = ",".join(["%s"] * len(broker_ids))
    cur.execute(f"""
        WITH buys AS (
            SELECT date::date AS date, symbol,
                   buyer_broker_id AS broker_id,
                   SUM(quantity::float) AS buy_units
            FROM floorsheet
            WHERE date >= %s AND date <= %s
              AND buyer_broker_id IN ({ph})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, buyer_broker_id
        ),
        sells AS (
            SELECT date::date AS date, symbol,
                   seller_broker_id AS broker_id,
                   SUM(quantity::float) AS sell_units
            FROM floorsheet
            WHERE date >= %s AND date <= %s
              AND seller_broker_id IN ({ph})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, seller_broker_id
        )
        SELECT
            COALESCE(b.date,    s.date)        AS date,
            COALESCE(b.symbol,  s.symbol)      AS symbol,
            COALESCE(b.broker_id, s.broker_id) AS broker_id,
            COALESCE(b.buy_units,  0)          AS buy_units,
            COALESCE(s.sell_units, 0)          AS sell_units
        FROM buys b
        FULL OUTER JOIN sells s
            ON s.date=b.date AND s.symbol=b.symbol AND s.broker_id=b.broker_id
        ORDER BY date, symbol, broker_id
    """, [from_date, to_date] + broker_ids + [from_date, to_date] + broker_ids)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]      = pd.to_datetime(df["date"])
    df["net_units"] = df["buy_units"] - df["sell_units"]
    df["broker_id"] = df["broker_id"].astype(str).str.strip()
    log.info("  Loaded %d broker rows", len(df))
    return df.sort_values(["symbol","broker_id","date"]).reset_index(drop=True)


def load_prices(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading price_history...")
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date::date AS date,
                   COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
            FROM price_history
            WHERE date >= %s AND date <= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date
        """, (from_date, to_date))
        rows = cur.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    log.info("  Loaded %d price rows for %d symbols",
             len(df), df["symbol"].nunique())
    return df.sort_values(["symbol","date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE PER-SYMBOL ARRAYS (fast lookup)
# ══════════════════════════════════════════════════════════════════════════════

def build_symbol_data(price_df: pd.DataFrame,
                      vol_df: pd.DataFrame) -> dict:
    """
    Build per-symbol dict with:
      dates, closes, vol_dates, volumes, vol_60d_avg (rolling)
    """
    log.info("Building symbol lookup arrays...")
    idx = {}

    price_grouped = {sym: grp.sort_values("date")
                     for sym, grp in price_df.groupby("symbol")}
    vol_grouped   = {sym: grp.sort_values("date")
                     for sym, grp in vol_df.groupby("symbol")}

    for sym in price_grouped:
        pg = price_grouped[sym]
        vg = vol_grouped.get(sym, pd.DataFrame())

        p_dates  = pg["date"].values
        p_closes = pg["close"].values

        if vg.empty:
            idx[sym] = {
                "p_dates":  p_dates,
                "p_closes": p_closes,
                "v_dates":  np.array([]),
                "volumes":  np.array([]),
                "vol_60avg": np.array([]),
            }
            continue

        v_dates  = vg["date"].values
        volumes  = vg["total_volume"].values
        # Rolling 60-day avg volume
        vol_series  = pd.Series(volumes)
        vol_60avg   = vol_series.rolling(60, min_periods=5).mean().values

        idx[sym] = {
            "p_dates":   p_dates,
            "p_closes":  p_closes,
            "v_dates":   v_dates,
            "volumes":   volumes,
            "vol_60avg": vol_60avg,
        }

    log.info("  Built arrays for %d symbols", len(idx))
    return idx


def get_fwd_return(sym_data: dict, signal_ts: pd.Timestamp,
                   fwd_days: int) -> float | None:
    dates  = sym_data["p_dates"]
    closes = sym_data["p_closes"]
    sig    = signal_ts.to_datetime64()

    mask_entry = dates >= sig
    if not mask_entry.any():
        return None
    entry = closes[mask_entry][0]
    edate = dates[mask_entry][0]

    exit_target = edate + np.timedelta64(fwd_days, 'D')
    mask_exit   = dates >= exit_target
    if not mask_exit.any():
        return None
    return round((closes[mask_exit][0] - entry) / entry * 100, 2)


def get_vol_ratio(sym_data: dict,
                  signal_ts: pd.Timestamp) -> float | None:
    """Today's volume / 60d avg volume."""
    v_dates  = sym_data["v_dates"]
    volumes  = sym_data["volumes"]
    vol_60avg = sym_data["vol_60avg"]

    if len(v_dates) == 0:
        return None
    sig = signal_ts.to_datetime64()
    mask = v_dates == sig
    if not mask.any():
        return None
    i = np.where(mask)[0][0]
    avg = vol_60avg[i]
    if avg is None or np.isnan(avg) or avg == 0:
        return None
    return float(volumes[i]) / float(avg)


def get_price_range(sym_data: dict,
                    signal_ts: pd.Timestamp,
                    lookback: int = 20) -> float | None:
    """(max - min) / min over last `lookback` trading days."""
    dates  = sym_data["p_dates"]
    closes = sym_data["p_closes"]
    sig    = signal_ts.to_datetime64()

    mask   = dates < sig
    recent = closes[mask][-lookback:]
    if len(recent) < 5:
        return None
    mn = recent.min()
    if mn == 0:
        return None
    return float((recent.max() - mn) / mn)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD RAW SIGNAL TABLE (once, then filter by thresholds)
# ══════════════════════════════════════════════════════════════════════════════

def build_raw_signals(broker_df: pd.DataFrame,
                      sym_data: dict) -> pd.DataFrame:
    """
    For every broker × symbol × day where a net buy streak starts at day N,
    record: vol_ratio, price_range, streak_length, forward returns.
    Do this ONCE across the loosest thresholds, then filter in grid search.
    """
    log.info("Building raw signal table (loosest thresholds)...")
    records = []

    MAX_STREAK = max(MIN_STREAKS)  # 14 — only emit when streak >= 5
    MIN_STREAK_EMIT = min(MIN_STREAKS)  # 5

    symbols_done = 0
    total_syms   = broker_df["symbol"].nunique()

    for (broker_id, symbol), grp in broker_df.groupby(["broker_id","symbol"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        broker_name = BROKERS.get(str(broker_id), str(broker_id))

        if symbol not in sym_data:
            continue

        sd = sym_data[symbol]
        b_dates = grp["date"].values
        b_net   = grp["net_units"].values

        streak = 0
        for i in range(len(grp)):
            if b_net[i] > 0:
                streak += 1
            else:
                streak = 0
                continue

            # Emit at every streak length from MIN to MAX
            if streak < MIN_STREAK_EMIT:
                continue
            if streak > MAX_STREAK:
                continue  # beyond our grid — skip

            signal_ts  = pd.Timestamp(b_dates[i])
            vol_ratio  = get_vol_ratio(sd, signal_ts)
            price_rng  = get_price_range(sd, signal_ts)

            if vol_ratio is None or price_rng is None:
                continue

            row = {
                "broker_id":     broker_id,
                "broker_name":   broker_name,
                "symbol":        symbol,
                "signal_date":   signal_ts.date(),
                "streak_days":   streak,
                "vol_ratio":     round(vol_ratio, 4),
                "price_range":   round(price_rng, 4),
            }
            for fwd in FORWARD_WINDOWS:
                ret = get_fwd_return(sd, signal_ts, fwd)
                row[f"ret_{fwd}d"] = ret

            records.append(row)

        symbols_done += 1
        if symbols_done % 100 == 0:
            log.info("  %d / %d symbols processed", symbols_done, total_syms)

    df = pd.DataFrame(records)
    log.info("Raw signal rows: %d", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def grid_search(raw: pd.DataFrame) -> list[dict]:
    """
    Filter raw signals by each threshold combination and compute stats.
    Returns one row per combo × broker × forward window.
    """
    log.info("Running grid search over %d combinations...",
             len(VOL_RATIOS) * len(PRICE_RANGES) * len(MIN_STREAKS))
    results = []

    total_combos = len(VOL_RATIOS) * len(PRICE_RANGES) * len(MIN_STREAKS)
    done = 0

    for vol_thresh, price_thresh, streak_thresh in product(
            VOL_RATIOS, PRICE_RANGES, MIN_STREAKS):

        # Filter
        subset = raw[
            (raw["vol_ratio"]   <= vol_thresh) &
            (raw["price_range"] <= price_thresh) &
            (raw["streak_days"] >= streak_thresh)
        ]

        # Remove duplicate signals: same symbol×broker×signal_date
        # (streak 5,6,7,8 all fire — keep the first one per streak start)
        # We do this by keeping the minimum streak per group
        if not subset.empty:
            subset = subset.sort_values("streak_days")
            subset = subset.drop_duplicates(
                subset=["broker_id","symbol","signal_date"], keep="first"
            )

        n_signals = len(subset)
        n_symbols = subset["symbol"].nunique() if n_signals > 0 else 0

        # Per broker stats
        for broker_id, broker_name in BROKERS.items():
            bsub = subset[subset["broker_id"].astype(str) == broker_id]
            n_b  = len(bsub)

            for fwd in FORWARD_WINDOWS:
                col  = f"ret_{fwd}d"
                valid = bsub[bsub[col].notna()]
                n_v   = len(valid)

                if n_v < 5:
                    hit_rate = None
                    avg_ret  = None
                    med_ret  = None
                    pct_20   = None
                    pct_50   = None
                else:
                    rets     = valid[col]
                    hit_rate = round((rets > 0).mean() * 100, 1)
                    avg_ret  = round(rets.mean(), 2)
                    med_ret  = round(rets.median(), 2)
                    pct_20   = round((rets > 20).mean() * 100, 1)
                    pct_50   = round((rets > 50).mean() * 100, 1)

                results.append({
                    "vol_thresh":    vol_thresh,
                    "price_thresh":  price_thresh,
                    "streak_thresh": streak_thresh,
                    "broker_id":     broker_id,
                    "broker_name":   broker_name,
                    "fwd_days":      fwd,
                    "n_signals":     n_b,
                    "n_valid":       n_v,
                    "hit_rate":      hit_rate,
                    "avg_return":    avg_ret,
                    "median_return": med_ret,
                    "pct_above_20":  pct_20,
                    "pct_above_50":  pct_50,
                })

        done += 1
        if done % 10 == 0:
            log.info("  %d / %d combos done", done, total_combos)

    log.info("Grid search complete: %d result rows", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRINT TOP RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def print_top_results(results: list[dict]) -> list[dict]:
    df = pd.DataFrame(results)

    # Require minimum signal count to avoid overfitting on tiny samples
    MIN_SIGNALS = 30
    df_valid = df[df["n_valid"] >= MIN_SIGNALS].copy()

    print("\n" + "="*80)
    print("HIDDEN ACCUMULATION GRID SEARCH RESULTS")
    print(f"Minimum {MIN_SIGNALS} signals required")
    print("="*80)

    # Top combos by hit rate for each forward window
    for fwd in FORWARD_WINDOWS:
        wdf = df_valid[df_valid["fwd_days"] == fwd]
        if wdf.empty:
            continue

        top = wdf.nlargest(15, "hit_rate")
        print(f"\n── TOP 15 by hit rate — {fwd}d forward ──")
        print(f"  {'Broker':<25} {'VolMax':>7} {'PriceMax':>9} {'Streak':>7} "
              f"{'HitRate':>8} {'AvgRet':>7} {'MedRet':>7} "
              f"{'%>20':>6} {'%>50':>6} {'N':>5}")
        print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*7} "
              f"{'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5}")

        for _, r in top.iterrows():
            print(f"  {r['broker_name']:<25} "
                  f"{r['vol_thresh']:>7.0%} "
                  f"{r['price_thresh']:>9.0%} "
                  f"{int(r['streak_thresh']):>7} "
                  f"{r['hit_rate']:>7.1f}% "
                  f"{r['avg_return']:>+7.1f}% "
                  f"{r['median_return']:>+7.1f}% "
                  f"{r['pct_above_20']:>6.1f}% "
                  f"{r['pct_above_50']:>6.1f}% "
                  f"{int(r['n_valid']):>5}")

    # Best combo overall: highest hit rate at 90d with n >= 50
    best = df_valid[
        (df_valid["fwd_days"] == 90) &
        (df_valid["n_valid"] >= 50)
    ].nlargest(5, "hit_rate")

    print(f"\n── BEST 5 OVERALL (90d, n≥50) ──")
    for _, r in best.iterrows():
        print(f"  {r['broker_name']:<25} "
              f"vol<{r['vol_thresh']:.0%} "
              f"range<{r['price_thresh']:.0%} "
              f"streak≥{int(r['streak_thresh'])}d  "
              f"→ hit={r['hit_rate']:.1f}%  "
              f"avg={r['avg_return']:+.1f}%  "
              f"median={r['median_return']:+.1f}%  "
              f"n={int(r['n_valid'])}")

    # Extract top rows for saving
    top_rows = df_valid[df_valid["fwd_days"]==90].nlargest(50, "hit_rate")
    return top_rows.to_dict("records")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(data: list[dict], name: str) -> None:
    if not data:
        return
    OUT_DIR.mkdir(exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")
    path  = OUT_DIR / f"{name}_{today}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    log.info("Saved: %s (%d rows)", path, len(data))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Load all data
    price_df  = load_prices(FROM_DATE, TO_DATE)
    vol_df    = load_daily_volume(FROM_DATE, TO_DATE)
    broker_df = load_broker_daily(FROM_DATE, TO_DATE)

    if price_df.empty or vol_df.empty or broker_df.empty:
        log.error("Missing data — aborting")
        return

    # Build fast lookup arrays
    sym_data = build_symbol_data(price_df, vol_df)

    # Build raw signal table once (loosest thresholds)
    raw = build_raw_signals(broker_df, sym_data)
    if raw.empty:
        log.error("No raw signals found")
        return

    # Save raw signals
    save_csv(raw.to_dict("records"), "hidden_grid_raw")

    # Grid search
    results = grid_search(raw)
    save_csv(results, "hidden_grid_all")

    # Print and save top results
    top = print_top_results(results)
    save_csv(top, "hidden_grid_top")

    log.info("Done.")


if __name__ == "__main__":
    main()