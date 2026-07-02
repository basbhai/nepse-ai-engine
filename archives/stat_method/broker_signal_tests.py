"""
broker_signal_tests.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Three broker signal tests inspired by US quant methods

TEST 1: Order Flow Imbalance (OFI)
  For each stock × day, compute OFI = (buy - sell) / (buy + sell)
  Check if sustained positive OFI predicts price rises in 30/60/90 days
  Uses FULL floorsheet (all brokers) — pre-aggregated in SQL

TEST 2: VPIN (Broker Toxicity / Informed Flow)
  For each of the 5 brokers, compute their "correct side %" —
  how often were they net buying before up moves, net selling before down moves?
  Compare to 50% random baseline. Above 55% = potentially informed.

TEST 3: Hidden Accumulation
  Find stocks where simultaneously:
    - Volume is LOW vs own 60d average (quiet)
    - Price is FLAT (calm, <5% range in 20 days)
    - One of the 5 brokers is net buying on most days
  All three = hidden accumulation pattern

Output:
  stat_method/output/ofi_results_YYYY-MM-DD.csv
  stat_method/output/vpin_results_YYYY-MM-DD.csv
  stat_method/output/hidden_accum_YYYY-MM-DD.csv
  stat_method/output/signal_tests_report_YYYY-MM-DD.txt

Usage:
    cd ~/nepse-engine
    python stat_method/broker_signal_tests.py
    python stat_method/broker_signal_tests.py --test ofi    # run only OFI
    python stat_method/broker_signal_tests.py --test vpin   # run only VPIN
    python stat_method/broker_signal_tests.py --test hidden # run only hidden accum
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIGNALS] %(message)s",
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

FORWARD_WINDOWS = [30, 60, 90]   # days to measure price change after signal


# ══════════════════════════════════════════════════════════════════════════════
# DB CONNECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _local_db():
    import psycopg2
    import os
    local = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    url   = os.environ.get("DATABASE_URL", local)
    if "neon" in url:
        url = local
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def _neon_db():
    from db.connection import _db
    return _db


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_daily_ofi(from_date: str, to_date: str) -> pd.DataFrame:
    """
    Load daily aggregated buy/sell volume per symbol across ALL brokers.
    Pre-aggregated in SQL — single query, fast.
    """
    log.info("Loading daily OFI (all brokers) %s → %s ...", from_date, to_date)
    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT
            date::date                  AS date,
            symbol,
            SUM(quantity::float)        AS total_volume,
            SUM(CASE WHEN buyer_broker_id IS NOT NULL
                     THEN quantity::float ELSE 0 END) AS buy_volume,
            SUM(CASE WHEN seller_broker_id IS NOT NULL
                     THEN quantity::float ELSE 0 END) AS sell_volume,
            COUNT(DISTINCT buyer_broker_id)  AS buyer_count,
            COUNT(DISTINCT seller_broker_id) AS seller_count
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
    df["buy_volume"]   = df["buy_volume"].astype(float)
    df["sell_volume"]  = df["sell_volume"].astype(float)

    # OFI = (buy - sell) / (buy + sell), range -1 to +1
    total = df["buy_volume"] + df["sell_volume"]
    df["ofi"] = np.where(
        total > 0,
        (df["buy_volume"] - df["sell_volume"]) / total,
        0.0
    )

    log.info("  Loaded %d daily OFI rows for %d symbols",
             len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_broker_daily(from_date: str, to_date: str) -> pd.DataFrame:
    """Load daily net position for the 5 brokers from local postgres."""
    broker_ids = list(BROKERS.keys())
    log.info("Loading 5-broker daily data %s → %s ...", from_date, to_date)
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
    return df.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


def load_prices(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading price_history %s → %s ...", from_date, to_date)
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
            ORDER BY symbol, date ASC
        """, (from_date, to_date))
        rows = cur.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    log.info("  Loaded %d price rows for %d symbols",
             len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE LOOKUP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_price_index(price_df: pd.DataFrame) -> dict:
    """Build symbol → sorted (dates, closes) arrays for fast lookup."""
    idx = {}
    for sym, grp in price_df.groupby("symbol"):
        grp = grp.sort_values("date")
        idx[sym] = (grp["date"].values, grp["close"].values)
    return idx


def get_fwd_return(price_idx: dict, symbol: str,
                   from_date: pd.Timestamp, days: int) -> float | None:
    if symbol not in price_idx:
        return None
    dates, closes = price_idx[symbol]
    # entry price: first available on or after from_date
    mask_entry = dates >= from_date.to_datetime64()
    if not mask_entry.any():
        return None
    entry_price = closes[mask_entry][0]
    entry_date  = dates[mask_entry][0]

    # exit price: first available on or after entry + days
    exit_target = entry_date + np.timedelta64(days, 'D')
    mask_exit   = dates >= exit_target
    if not mask_exit.any():
        return None
    exit_price = closes[mask_exit][0]
    return round((exit_price - entry_price) / entry_price * 100, 2)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — ORDER FLOW IMBALANCE (OFI)
# ══════════════════════════════════════════════════════════════════════════════

def test_ofi(ofi_df: pd.DataFrame, price_df: pd.DataFrame) -> list[dict]:
    """
    Find periods where OFI is consistently positive for 5/10/20 days
    and measure forward returns.

    Signal: rolling N-day average OFI > threshold
    """
    log.info("Running TEST 1: Order Flow Imbalance...")
    price_idx = build_price_index(price_df)
    results   = []

    OFI_WINDOWS    = [5, 10, 20]   # rolling average windows
    OFI_THRESHOLDS = [0.05, 0.10, 0.20]  # OFI must exceed this

    symbols = ofi_df["symbol"].unique()
    log.info("  Scanning %d symbols...", len(symbols))

    for sym in symbols:
        sym_df = ofi_df[ofi_df["symbol"] == sym].sort_values("date").copy()
        if len(sym_df) < 25:
            continue

        dates = sym_df["date"].values
        ofi   = sym_df["ofi"].values
        vol   = sym_df["total_volume"].values

        for win in OFI_WINDOWS:
            # Rolling mean OFI
            roll_ofi = pd.Series(ofi).rolling(win).mean().values

            for thresh in OFI_THRESHOLDS:
                # Find signal days: rolling OFI crossed above threshold
                for i in range(win, len(sym_df) - 1):
                    # Signal: today crosses above threshold, yesterday was below
                    if roll_ofi[i] >= thresh and roll_ofi[i-1] < thresh:
                        signal_date = pd.Timestamp(dates[i])
                        avg_vol     = float(np.mean(vol[max(0,i-60):i])) if i >= 5 else 0

                        row = {
                            "symbol":       sym,
                            "signal_date":  signal_date.date(),
                            "ofi_window":   win,
                            "ofi_threshold": thresh,
                            "ofi_value":    round(float(roll_ofi[i]), 4),
                            "avg_volume_60d": round(avg_vol, 0),
                        }
                        for fwd in FORWARD_WINDOWS:
                            ret = get_fwd_return(price_idx, sym, signal_date, fwd)
                            row[f"return_{fwd}d"] = ret
                            row[f"positive_{fwd}d"] = ret > 0 if ret is not None else None

                        results.append(row)

    log.info("  OFI signals found: %d", len(results))
    return results


def summarize_ofi(results: list[dict]) -> None:
    if not results:
        log.warning("No OFI results to summarize")
        return
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("TEST 1 — ORDER FLOW IMBALANCE SUMMARY")
    print("="*70)
    print(f"Total signal instances: {len(df)}")

    for win in [5, 10, 20]:
        for thresh in [0.05, 0.10, 0.20]:
            sub = df[(df["ofi_window"]==win) & (df["ofi_threshold"]==thresh)]
            if len(sub) < 10:
                continue
            print(f"\n  OFI window={win}d threshold={thresh:.0%} — {len(sub)} signals")
            for fwd in FORWARD_WINDOWS:
                col = f"return_{fwd}d"
                pos = f"positive_{fwd}d"
                valid = sub[sub[col].notna()]
                if len(valid) == 0:
                    continue
                hit_rate = valid[pos].mean() * 100
                avg_ret  = valid[col].mean()
                print(f"    {fwd}d forward: hit_rate={hit_rate:.1f}%  avg_return={avg_ret:+.1f}%  n={len(valid)}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — VPIN (BROKER TOXICITY)
# ══════════════════════════════════════════════════════════════════════════════

def test_vpin(broker_df: pd.DataFrame, price_df: pd.DataFrame) -> list[dict]:
    """
    For each broker × symbol, compute correct-side % —
    how often the broker was on the right side of the subsequent price move.

    Net buyer → price should rise (correct side = positive return)
    Net seller → price should fall (correct side = negative return)

    Compare to 50% random baseline.
    """
    log.info("Running TEST 2: VPIN / Broker Toxicity...")
    price_idx = build_price_index(price_df)
    results   = []

    for (broker_id, symbol), grp in broker_df.groupby(["broker_id", "symbol"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        broker_name = BROKERS.get(str(broker_id), str(broker_id))

        if len(grp) < 10:
            continue

        correct_30  = []
        correct_60  = []
        correct_90  = []

        for _, row in grp.iterrows():
            net   = float(row["net_units"])
            if net == 0:
                continue

            direction = 1 if net > 0 else -1  # 1=buying, -1=selling

            for fwd, lst in [(30, correct_30), (60, correct_60), (90, correct_90)]:
                ret = get_fwd_return(price_idx, symbol,
                                     pd.Timestamp(row["date"]), fwd)
                if ret is None:
                    continue
                # Correct if: buying before price rise OR selling before price fall
                correct = (direction == 1 and ret > 0) or \
                          (direction == -1 and ret < 0)
                lst.append(correct)

        if not correct_30:
            continue

        def safe_mean(lst):
            return round(np.mean(lst) * 100, 1) if lst else None

        results.append({
            "broker_id":         broker_id,
            "broker_name":       broker_name,
            "symbol":            symbol,
            "total_trading_days": len(grp[grp["net_units"] != 0]),
            "correct_pct_30d":   safe_mean(correct_30),
            "correct_pct_60d":   safe_mean(correct_60),
            "correct_pct_90d":   safe_mean(correct_90),
            "n_30d":             len(correct_30),
            "n_60d":             len(correct_60),
            "n_90d":             len(correct_90),
        })

    log.info("  VPIN results: %d broker×symbol pairs", len(results))
    return results


def summarize_vpin(results: list[dict]) -> None:
    if not results:
        log.warning("No VPIN results")
        return
    df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("TEST 2 — VPIN / BROKER TOXICITY SUMMARY")
    print("="*70)
    print("Random baseline = 50%. Above 55% = potentially informed.\n")

    for broker_id, broker_name in BROKERS.items():
        bdf = df[df["broker_id"].astype(str) == broker_id]
        if bdf.empty:
            continue

        # Overall correct %
        for fwd, col, ncol in [(30,"correct_pct_30d","n_30d"),
                                (60,"correct_pct_60d","n_60d"),
                                (90,"correct_pct_90d","n_90d")]:
            valid = bdf[bdf[col].notna()]
            if valid.empty:
                continue
            # Weighted average by n
            weighted = np.average(valid[col], weights=valid[ncol])
            above_55 = (valid[col] >= 55).mean() * 100
            above_60 = (valid[col] >= 60).mean() * 100

            if fwd == 30:
                print(f"  {broker_name}")
                print(f"    {'Window':<8} {'Weighted Correct%':>18} {'% pairs >55%':>13} {'% pairs >60%':>13} {'pairs':>6}")
            print(f"    {fwd}d{'':<5} {weighted:>17.1f}%  {above_55:>12.1f}%  {above_60:>12.1f}%  {len(valid):>5}")

        # Top 5 stocks where this broker is most "informed"
        top = bdf.nlargest(5, "correct_pct_60d")[
            ["symbol","correct_pct_30d","correct_pct_60d","correct_pct_90d","n_60d"]
        ]
        print(f"    Top stocks (by 60d correct%):")
        print("    " + top.to_string(index=False))
        print()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — HIDDEN ACCUMULATION
# ══════════════════════════════════════════════════════════════════════════════

def test_hidden_accumulation(broker_df: pd.DataFrame,
                              ofi_df: pd.DataFrame,
                              price_df: pd.DataFrame) -> list[dict]:
    """
    Find stocks where all three conditions hold simultaneously:
      1. Volume LOW — today's volume < 50% of 60d average volume
      2. Price FLAT — close range over last 20 days < 8%
      3. Broker NET BUYING — consecutive net buy days >= 5

    Then measure what price did 30/60/90 days later.
    """
    log.info("Running TEST 3: Hidden Accumulation...")
    price_idx = build_price_index(price_df)
    results   = []

    MIN_STREAK   = 5     # consecutive net buy days
    VOL_RATIO    = 0.60  # volume < 60% of 60d avg
    PRICE_RANGE  = 0.08  # price range < 8% over 20 days

    symbols = broker_df["symbol"].unique()
    log.info("  Scanning %d symbols...", len(symbols))

    for sym in symbols:
        # Price data for this symbol
        sym_price = price_df[price_df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(sym_price) < 25:
            continue
        price_dates  = sym_price["date"].values
        price_closes = sym_price["close"].values

        # Volume data for this symbol
        sym_vol = ofi_df[ofi_df["symbol"] == sym].sort_values("date").set_index("date")

        # Broker data for this symbol
        sym_broker = broker_df[broker_df["symbol"] == sym]

        for broker_id, broker_name in BROKERS.items():
            b_data = sym_broker[sym_broker["broker_id"] == broker_id].sort_values("date").reset_index(drop=True)
            if len(b_data) < MIN_STREAK:
                continue

            b_dates = b_data["date"].values
            b_net   = b_data["net_units"].values

            # Walk through and find consecutive buy streaks
            streak = 0
            for i in range(len(b_data)):
                if b_net[i] > 0:
                    streak += 1
                else:
                    streak = 0
                    continue

                if streak < MIN_STREAK:
                    continue

                # Signal day: streak just hit MIN_STREAK
                signal_date = pd.Timestamp(b_dates[i])

                # Condition 1: Volume LOW
                if signal_date not in sym_vol.index:
                    continue
                today_vol  = float(sym_vol.loc[signal_date, "total_volume"])
                # 60d avg volume
                vol_window = sym_vol[sym_vol.index < signal_date].tail(60)
                if len(vol_window) < 10:
                    continue
                avg_vol_60d = float(vol_window["total_volume"].mean())
                if avg_vol_60d == 0:
                    continue
                vol_ratio = today_vol / avg_vol_60d
                if vol_ratio >= VOL_RATIO:
                    continue  # volume not low enough

                # Condition 2: Price FLAT
                price_mask = price_dates < signal_date.to_datetime64()
                recent_prices = price_closes[price_mask][-20:]
                if len(recent_prices) < 10:
                    continue
                price_range = (recent_prices.max() - recent_prices.min()) / recent_prices.min()
                if price_range >= PRICE_RANGE:
                    continue  # price not flat enough

                # All three conditions met — record signal
                entry_price = float(recent_prices[-1]) if len(recent_prices) > 0 else None
                row = {
                    "symbol":        sym,
                    "broker_id":     broker_id,
                    "broker_name":   broker_name,
                    "signal_date":   signal_date.date(),
                    "streak_days":   streak,
                    "vol_ratio":     round(vol_ratio, 3),
                    "price_range_pct": round(price_range * 100, 2),
                    "entry_price":   entry_price,
                    "avg_vol_60d":   round(avg_vol_60d, 0),
                    "today_vol":     round(today_vol, 0),
                }
                for fwd in FORWARD_WINDOWS:
                    ret = get_fwd_return(price_idx, sym, signal_date, fwd)
                    row[f"return_{fwd}d"]  = ret
                    row[f"positive_{fwd}d"] = ret > 0 if ret is not None else None

                results.append(row)

    log.info("  Hidden accumulation signals found: %d", len(results))
    return results


def summarize_hidden(results: list[dict]) -> None:
    if not results:
        log.warning("No hidden accumulation signals found")
        return
    df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("TEST 3 — HIDDEN ACCUMULATION SUMMARY")
    print("="*70)
    print(f"Total signal instances: {len(df)} across {df['symbol'].nunique()} symbols\n")

    # Per broker
    for broker_id, broker_name in BROKERS.items():
        bdf = df[df["broker_id"].astype(str) == broker_id]
        if bdf.empty:
            continue
        print(f"  {broker_name} — {len(bdf)} signals")
        for fwd in FORWARD_WINDOWS:
            col = f"return_{fwd}d"
            pos = f"positive_{fwd}d"
            valid = bdf[bdf[col].notna()]
            if valid.empty:
                continue
            hit  = valid[pos].mean() * 100
            avg  = valid[col].mean()
            med  = valid[col].median()
            print(f"    {fwd}d: hit={hit:.1f}%  avg={avg:+.1f}%  median={med:+.1f}%  n={len(valid)}")
        print()

    # Top signals — positive return across all windows
    print("  Top hidden accumulation detections (positive all 3 windows):")
    top = df[
        (df["return_30d"] > 0) &
        (df["return_60d"] > 0) &
        (df["return_90d"] > 0)
    ].sort_values("return_90d", ascending=False)
    if not top.empty:
        print(top[["symbol","broker_name","signal_date","streak_days",
                   "vol_ratio","price_range_pct",
                   "return_30d","return_60d","return_90d"]].head(20).to_string(index=False))
    else:
        print("  None found with positive returns across all 3 windows.")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["ofi","vpin","hidden","all"],
                        default="all", help="Which test to run")
    parser.add_argument("--from-date", default=FROM_DATE)
    parser.add_argument("--to-date",   default=TO_DATE)
    args = parser.parse_args()

    run_ofi    = args.test in ("ofi",    "all")
    run_vpin   = args.test in ("vpin",   "all")
    run_hidden = args.test in ("hidden", "all")

    # Load price always
    price_df = load_prices(args.from_date, args.to_date)
    if price_df.empty:
        log.error("No price data")
        return

    # Load OFI if needed
    ofi_df = pd.DataFrame()
    if run_ofi or run_hidden:
        ofi_df = load_daily_ofi(args.from_date, args.to_date)
        if ofi_df.empty:
            log.error("No OFI data")
            return

    # Load broker data if needed
    broker_df = pd.DataFrame()
    if run_vpin or run_hidden:
        broker_df = load_broker_daily(args.from_date, args.to_date)
        if broker_df.empty:
            log.error("No broker data")
            return

    # ── Test 1: OFI ──────────────────────────────────────────────────────────
    if run_ofi:
        ofi_results = test_ofi(ofi_df, price_df)
        summarize_ofi(ofi_results)
        save_csv(ofi_results, "ofi_results")

    # ── Test 2: VPIN ─────────────────────────────────────────────────────────
    if run_vpin:
        vpin_results = test_vpin(broker_df, price_df)
        summarize_vpin(vpin_results)
        save_csv(vpin_results, "vpin_results")

    # ── Test 3: Hidden Accumulation ──────────────────────────────────────────
    if run_hidden:
        hidden_results = test_hidden_accumulation(broker_df, ofi_df, price_df)
        summarize_hidden(hidden_results)
        save_csv(hidden_results, "hidden_accum")

    log.info("All tests complete.")


if __name__ == "__main__":
    main()