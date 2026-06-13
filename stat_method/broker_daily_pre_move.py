"""
broker_daily_pre_move.py
────────────────────────────────────────────────────────────────────────────
For every qualified big move (≥75% gain, IPO filtered), outputs the raw
daily net buy/sell for each of the 5 brokers in the lookback window
BEFORE the move started.

One row per: symbol × window × broker × day

Output:
  stat_method/output/broker_daily_pre_move_YYYY-MM-DD.csv

Usage:
    cd ~/nepse-engine
    python stat_method/broker_daily_pre_move.py
    python stat_method/broker_daily_pre_move.py --threshold 50
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
    format="%(asctime)s [DAILY] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR   = Path(__file__).parent / "output"
FROM_DATE = "2023-07-01"
TO_DATE   = "2026-05-27"
WINDOWS   = [30, 60, 90]
THRESHOLD = 75.0

MIN_PRIOR_DAYS = {30: 20, 60: 40, 90: 60}

BROKERS = {
    "38": "Dipshikha Dhitopatra",
    "56": "Sri Hari Securities",
    "49": "Online Securities",
    "48": "Trishakti Securities",
    "58": "Naasa Securities",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_prices(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading price_history %s → %s ...", from_date, to_date)
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT symbol,
                   date::date AS date,
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
    log.info("  Loaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_broker_daily(from_date: str, to_date: str) -> pd.DataFrame:
    broker_ids = list(BROKERS.keys())
    log.info("Loading floorsheet for 5 brokers %s → %s ...", from_date, to_date)

    import psycopg2
    import psycopg2.extras
    import os
    local = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    url   = os.environ.get("DATABASE_URL", local)
    if "neon" in url:
        url = local
    conn = psycopg2.connect(url)
    conn.autocommit = True
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    placeholders = ",".join(["%s"] * len(broker_ids))

    cur.execute(f"""
        WITH buys AS (
            SELECT date::date AS date, symbol,
                   buyer_broker_id  AS broker_id,
                   buyer_broker     AS broker_name,
                   SUM(quantity::float) AS buy_units
            FROM floorsheet
            WHERE date >= %s AND date <= %s
              AND buyer_broker_id IN ({placeholders})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, buyer_broker_id, buyer_broker
        ),
        sells AS (
            SELECT date::date AS date, symbol,
                   seller_broker_id AS broker_id,
                   SUM(quantity::float) AS sell_units
            FROM floorsheet
            WHERE date >= %s AND date <= %s
              AND seller_broker_id IN ({placeholders})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, seller_broker_id
        )
        SELECT
            COALESCE(b.date,    s.date)         AS date,
            COALESCE(b.symbol,  s.symbol)       AS symbol,
            COALESCE(b.broker_id, s.broker_id)  AS broker_id,
            b.broker_name,
            COALESCE(b.buy_units,  0)           AS buy_units,
            COALESCE(s.sell_units, 0)           AS sell_units
        FROM buys b
        FULL OUTER JOIN sells s
            ON  s.date      = b.date
            AND s.symbol    = b.symbol
            AND s.broker_id = b.broker_id
        ORDER BY date, symbol, broker_id
    """, [from_date, to_date] + broker_ids + [from_date, to_date] + broker_ids)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    log.info("  Loaded %d broker rows", len(rows))
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]      = pd.to_datetime(df["date"])
    df["net_units"] = df["buy_units"] - df["sell_units"]
    df["broker_id"] = df["broker_id"].astype(str).str.strip()
    return df.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIND QUALIFIED MOVERS (same IPO filter as before)
# ══════════════════════════════════════════════════════════════════════════════

def find_qualified_movers(price_df: pd.DataFrame,
                          threshold: float) -> list[dict]:
    results = []
    symbols = price_df["symbol"].unique()
    log.info("Scanning %d symbols for ≥%.0f%% moves...", len(symbols), threshold)

    for sym in symbols:
        sym_df = price_df[price_df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(sym_df) < 10:
            continue

        dates  = sym_df["date"].values
        closes = sym_df["close"].values

        for window in WINDOWS:
            window_td = pd.Timedelta(days=window)
            min_days  = MIN_PRIOR_DAYS[window]
            best      = None

            for i in range(len(sym_df)):
                start_date  = dates[i]
                start_price = closes[i]

                # IPO filter
                lookback_start = start_date - pd.Timedelta(days=90)
                prior_days = np.sum((dates >= lookback_start) & (dates < start_date))
                if prior_days < min_days:
                    continue

                end_date = start_date + window_td
                mask = (dates > start_date) & (dates <= end_date)
                if not mask.any():
                    continue

                window_closes = closes[mask]
                window_dates  = dates[mask]
                max_close     = window_closes.max()
                max_idx       = window_closes.argmax()
                max_date      = window_dates[max_idx]
                gain          = (max_close - start_price) / start_price * 100

                if gain < threshold:
                    continue

                record = {
                    "symbol":      sym,
                    "window_days": window,
                    "move_start":  pd.Timestamp(start_date).date(),
                    "move_end":    pd.Timestamp(max_date).date(),
                    "price_start": round(float(start_price), 2),
                    "price_end":   round(float(max_close), 2),
                    "gain_pct":    round(gain, 1),
                    "prior_days":  int(prior_days),
                }

                if best is None or gain > best["gain_pct"]:
                    best = record

            if best:
                results.append(best)

    log.info("Qualified moves: %d across %d symbols",
             len(results), len({m["symbol"] for m in results}))
    for w in WINDOWS:
        wm = [m for m in results if m["window_days"] == w]
        log.info("  %dd: %d moves", w, len(wm))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DUMP DAILY BROKER ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════

def dump_daily_activity(movers: list[dict],
                        broker_df: pd.DataFrame,
                        price_df: pd.DataFrame) -> list[dict]:
    """
    For each move, output one row per broker per trading day
    in the lookback window before move_start.
    """
    records = []
    total   = len(movers)
    log.info("Dumping daily activity for %d moves...", total)

    for i, move in enumerate(movers):
        sym        = move["symbol"]
        move_start = pd.Timestamp(move["move_start"])
        window     = move["window_days"]
        gain_pct   = move["gain_pct"]
        lookback_start = move_start - pd.Timedelta(days=window)

        # Get all trading days in the lookback window from price_history
        sym_price = price_df[
            (price_df["symbol"] == sym) &
            (price_df["date"] >= lookback_start) &
            (price_df["date"] < move_start)
        ].sort_values("date")

        if sym_price.empty:
            continue

        # All trading dates in window
        trading_dates = sym_price["date"].tolist()
        # Day number: 1 = first day of lookback, N = last day before move
        n_days = len(trading_dates)

        # Broker data for this symbol in lookback window
        sym_broker = broker_df[
            (broker_df["symbol"] == sym) &
            (broker_df["date"] >= lookback_start) &
            (broker_df["date"] < move_start)
        ]

        # For each broker, fill in daily activity
        for broker_id, broker_name in BROKERS.items():
            b_data = sym_broker[sym_broker["broker_id"] == broker_id].set_index("date")

            # Running cumulative net for this broker
            cumulative_net = 0.0

            for day_num, dt in enumerate(trading_dates, 1):
                close = float(sym_price[sym_price["date"] == dt]["close"].iloc[0])

                if dt in b_data.index:
                    row      = b_data.loc[dt]
                    buy_u    = float(row["buy_units"])
                    sell_u   = float(row["sell_units"])
                    net_u    = float(row["net_units"])
                    active   = True
                else:
                    buy_u  = 0.0
                    sell_u = 0.0
                    net_u  = 0.0
                    active = False

                cumulative_net += net_u
                days_to_move = n_days - day_num  # days remaining until move_start

                records.append({
                    "symbol":          sym,
                    "window_days":     window,
                    "move_start":      move["move_start"],
                    "move_end":        move["move_end"],
                    "gain_pct":        gain_pct,
                    "broker_id":       broker_id,
                    "broker_name":     broker_name,
                    "date":            dt.date(),
                    "day_num":         day_num,        # 1 = first day of lookback
                    "days_to_move":    days_to_move,   # 0 = last day before move
                    "close":           round(close, 2),
                    "buy_units":       round(buy_u, 0),
                    "sell_units":      round(sell_u, 0),
                    "net_units":       round(net_u, 0),
                    "cumulative_net":  round(cumulative_net, 0),
                    "active":          active,
                })

        if (i + 1) % 50 == 0:
            log.info("  %d / %d moves done", i + 1, total)

    log.info("Total daily rows: %d", len(records))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(records: list[dict], name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    today    = date.today().strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"{name}_{today}.csv"
    if not records:
        log.warning("No records to save for %s", name)
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    log.info("Saved: %s (%d rows)", out_path, len(records))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--from-date", default=FROM_DATE)
    parser.add_argument("--to-date",   default=TO_DATE)
    args = parser.parse_args()

    price_df  = load_prices(args.from_date, args.to_date)
    if price_df.empty:
        return

    movers = find_qualified_movers(price_df, args.threshold)
    if not movers:
        log.error("No qualified moves found")
        return

    broker_df = load_broker_daily(args.from_date, args.to_date)
    if broker_df.empty:
        log.error("No broker data")
        return

    records = dump_daily_activity(movers, broker_df, price_df)
    save_csv(records, "broker_daily_pre_move")

    # Also save the qualified movers list
    save_csv(movers, "qualified_movers_75pct")

    log.info("Done.")


if __name__ == "__main__":
    main()