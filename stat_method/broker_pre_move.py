"""
broker_pre_move.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Broker Pre-Move Activity Analysis

Step 1: Find all stocks with ≥50% gain in any 30/60/90 day rolling window
        since 2023-07-01, excluding IPO/fresh listings using per-window
        minimum prior trading day requirement:
          30d move → ≥20 trading days of history before move_start
          60d move → ≥40 trading days of history before move_start
          90d move → ≥60 trading days of history before move_start

Step 2: For each qualified move, check daily net activity of 5 smart brokers
        in the lookback window BEFORE the move started (not during).
        Lookback = same as window size (30/60/90 days before move_start).

Output:
  stat_method/output/qualified_movers_YYYY-MM-DD.csv   — filtered move list
  stat_method/output/broker_pre_move_YYYY-MM-DD.csv    — broker activity per move
  stat_method/output/broker_pre_move_summary_YYYY-MM-DD.csv — hit rate summary

Usage:
    cd ~/nepse-engine
    python stat_method/broker_pre_move.py
    python stat_method/broker_pre_move.py --threshold 30   # lower gain threshold
    python stat_method/broker_pre_move.py --min-days 10    # looser IPO filter
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
    format="%(asctime)s [PRE-MOVE] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR   = Path(__file__).parent / "output"
FROM_DATE = "2023-07-01"
TO_DATE   = "2026-05-27"
WINDOWS   = [30, 60, 90]
THRESHOLD = 50.0   # % gain to qualify as a big move

# Minimum prior trading days required per window (IPO filter)
MIN_PRIOR_DAYS = {30: 20, 60: 40, 90: 60}

BROKERS = {
    "38": "Dipshikha Dhitopatra",
    "56": "Sri Hari Securities",
    "49": "Online Securities",
    "48": "Trishakti Securities",
    "58": "Naasa Securities",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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
            WHERE date >= %s
              AND date <= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (from_date, to_date))
        rows = cur.fetchall()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        log.error("No price data returned")
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    log.info("  Loaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_broker_daily(from_date: str, to_date: str) -> pd.DataFrame:
    """Load daily net position for all 5 brokers from local postgres."""
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
# STEP 1 — FIND QUALIFIED BIG MOVERS (with IPO filter)
# ══════════════════════════════════════════════════════════════════════════════

def find_qualified_movers(price_df: pd.DataFrame,
                          threshold: float,
                          min_prior_days: dict) -> list[dict]:
    """
    Find all (symbol, window) pairs where price gained >= threshold%
    AND the symbol had sufficient prior trading history (IPO filter).
    Records only the BEST (highest gain) qualifying move per symbol×window.
    """
    results  = []
    symbols  = price_df["symbol"].unique()
    excluded = 0
    log.info("Scanning %d symbols for ≥%.0f%% moves...", len(symbols), threshold)

    for sym in symbols:
        sym_df = price_df[price_df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        if len(sym_df) < 10:
            continue

        dates  = sym_df["date"].values
        closes = sym_df["close"].values

        for window in WINDOWS:
            window_td  = pd.Timedelta(days=window)
            min_days   = min_prior_days[window]
            best       = None

            for i in range(len(sym_df)):
                start_date  = dates[i]
                start_price = closes[i]

                # ── IPO filter ────────────────────────────────────────────
                # Count trading days in the [start_date - 90d, start_date) range
                lookback_start = start_date - pd.Timedelta(days=90)
                prior_days = np.sum((dates >= lookback_start) & (dates < start_date))
                if prior_days < min_days:
                    excluded += 1
                    continue

                # ── Find max price within window ──────────────────────────
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
                    "symbol":       sym,
                    "window_days":  window,
                    "move_start":   pd.Timestamp(start_date).date(),
                    "move_end":     pd.Timestamp(max_date).date(),
                    "price_start":  round(float(start_price), 2),
                    "price_end":    round(float(max_close), 2),
                    "gain_pct":     round(gain, 1),
                    "prior_days":   int(prior_days),
                }

                if best is None or gain > best["gain_pct"]:
                    best = record

            if best:
                results.append(best)

    log.info("Qualified moves: %d | IPO-excluded instances: %d", len(results), excluded)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BROKER PRE-MOVE ACTIVITY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_broker_pre_move(movers: list[dict],
                            broker_df: pd.DataFrame) -> list[dict]:
    """
    For each qualified move, check each broker's activity in the
    lookback window (same size as move window) BEFORE move_start.

    Returns one row per move × broker.
    """
    records = []
    total   = len(movers)
    log.info("Analyzing broker pre-move activity for %d moves...", total)

    for i, move in enumerate(movers):
        sym        = move["symbol"]
        move_start = pd.Timestamp(move["move_start"])
        window     = move["window_days"]
        lookback_start = move_start - pd.Timedelta(days=window)

        # Broker data for this symbol in the lookback window
        sym_broker = broker_df[
            (broker_df["symbol"] == sym) &
            (broker_df["date"] >= lookback_start) &
            (broker_df["date"] < move_start)
        ]

        for broker_id, broker_name in BROKERS.items():
            b_data = sym_broker[sym_broker["broker_id"] == broker_id].sort_values("date")

            if b_data.empty:
                records.append({
                    "symbol":           sym,
                    "window_days":      window,
                    "move_start":       move["move_start"],
                    "move_end":         move["move_end"],
                    "gain_pct":         move["gain_pct"],
                    "broker_id":        broker_id,
                    "broker_name":      broker_name,
                    "active_days":      0,
                    "net_buy_days":     0,
                    "net_sell_days":    0,
                    "total_net_units":  0,
                    "total_buy_units":  0,
                    "total_sell_units": 0,
                    "max_single_day":   0,
                    "consecutive_buy_streak": 0,
                    "was_active":       False,
                    "was_net_buyer":    False,
                    "was_net_seller":   False,
                    "dominant":         False,
                    "first_active_date": None,
                    "days_before_move": None,
                })
                continue

            buy_units  = b_data["buy_units"].sum()
            sell_units = b_data["sell_units"].sum()
            net_units  = b_data["net_units"].sum()
            active_days    = len(b_data)
            net_buy_days   = int((b_data["net_units"] > 0).sum())
            net_sell_days  = int((b_data["net_units"] < 0).sum())
            max_single_day = float(b_data["net_units"].abs().max())

            # Longest consecutive buy streak in lookback
            streak = 0
            best_streak = 0
            for v in b_data["net_units"].values:
                if v > 0:
                    streak += 1
                    best_streak = max(best_streak, streak)
                else:
                    streak = 0

            # Baseline: broker's avg daily net in this stock BEFORE the lookback window
            pre_lookback = broker_df[
                (broker_df["symbol"] == sym) &
                (broker_df["broker_id"] == broker_id) &
                (broker_df["date"] < lookback_start)
            ]
            baseline_avg = float(pre_lookback["net_units"].mean()) if not pre_lookback.empty else 0.0
            current_avg  = float(b_data["net_units"].mean())
            dominant     = (baseline_avg == 0 and net_units > 0) or \
                           (baseline_avg > 0 and current_avg > baseline_avg * 2) or \
                           (baseline_avg < 0 and current_avg > 0)

            # First active date and days before move
            first_date = b_data["date"].min()
            days_before = (move_start - first_date).days

            records.append({
                "symbol":           sym,
                "window_days":      window,
                "move_start":       move["move_start"],
                "move_end":         move["move_end"],
                "gain_pct":         move["gain_pct"],
                "broker_id":        broker_id,
                "broker_name":      broker_name,
                "active_days":      active_days,
                "net_buy_days":     net_buy_days,
                "net_sell_days":    net_sell_days,
                "total_net_units":  round(float(net_units), 0),
                "total_buy_units":  round(float(buy_units), 0),
                "total_sell_units": round(float(sell_units), 0),
                "max_single_day":   round(max_single_day, 0),
                "consecutive_buy_streak": best_streak,
                "was_active":       True,
                "was_net_buyer":    net_units > 0,
                "was_net_seller":   net_units < 0,
                "dominant":         dominant,
                "first_active_date": first_date.date(),
                "days_before_move": days_before,
                "baseline_avg":     round(baseline_avg, 1),
                "current_avg":      round(current_avg, 1),
            })

        if (i + 1) % 50 == 0:
            log.info("  %d / %d moves processed", i + 1, total)

    return records


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(records: list[dict], movers: list[dict]) -> list[dict]:
    """
    For each broker × window: what % of big moves had that broker
    active/net-buying/dominant before the move?
    """
    df       = pd.DataFrame(records)
    summary  = []

    for window in WINDOWS:
        wdf        = df[df["window_days"] == window]
        total_moves = len(movers_by_window(movers, window))

        for broker_id, broker_name in BROKERS.items():
            bdf = wdf[wdf["broker_id"] == broker_id]

            active_count   = int(bdf["was_active"].sum())
            buyer_count    = int(bdf["was_net_buyer"].sum())
            dominant_count = int(bdf["dominant"].sum())

            summary.append({
                "window_days":       window,
                "broker_id":         broker_id,
                "broker_name":       broker_name,
                "total_moves":       total_moves,
                "active_before":     active_count,
                "active_pct":        round(active_count / total_moves * 100, 1) if total_moves else 0,
                "net_buyer_before":  buyer_count,
                "net_buyer_pct":     round(buyer_count / total_moves * 100, 1) if total_moves else 0,
                "dominant_before":   dominant_count,
                "dominant_pct":      round(dominant_count / total_moves * 100, 1) if total_moves else 0,
                "avg_streak":        round(float(bdf[bdf["was_net_buyer"]]["consecutive_buy_streak"].mean()), 1)
                                     if buyer_count > 0 else 0,
                "avg_net_units":     round(float(bdf[bdf["was_net_buyer"]]["total_net_units"].mean()), 0)
                                     if buyer_count > 0 else 0,
            })

    return summary


def movers_by_window(movers: list[dict], window: int) -> list[dict]:
    return [m for m in movers if m["window_days"] == window]


# ══════════════════════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(movers: list[dict], summary: list[dict],
                 records: list[dict]) -> None:
    sep  = "=" * 80
    sep2 = "-" * 80

    print(sep)
    print("BROKER PRE-MOVE ACTIVITY REPORT")
    print(f"Generated: {date.today()}")
    print(f"Gain threshold: ≥{THRESHOLD:.0f}% | IPO filter: per-window min prior days")
    print(sep)

    for window in WINDOWS:
        wm = movers_by_window(movers, window)
        print(f"\n── {window}-DAY WINDOW — {len(wm)} qualified moves ──")
        print(f"  {'Broker':<25} {'Active%':>8} {'NetBuy%':>8} {'Dominant%':>10} {'AvgStreak':>10} {'AvgUnits':>10}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        wsummary = [s for s in summary if s["window_days"] == window]
        for s in sorted(wsummary, key=lambda x: -x["dominant_pct"]):
            print(f"  {s['broker_name']:<25} {s['active_pct']:>7.1f}% "
                  f"{s['net_buyer_pct']:>7.1f}% "
                  f"{s['dominant_pct']:>9.1f}% "
                  f"{s['avg_streak']:>10.1f} "
                  f"{s['avg_net_units']:>10,.0f}")

    # Top pre-move signals — stocks where dominant broker was active
    print(f"\n{sep}")
    print("TOP PRE-MOVE DETECTIONS — Dominant broker active before ≥200% move")
    print(sep2)
    df = pd.DataFrame(records)
    top = df[
        (df["dominant"] == True) &
        (df["was_net_buyer"] == True) &
        (df["gain_pct"] >= 200)
    ].sort_values("gain_pct", ascending=False)

    if top.empty:
        print("  None found.")
    else:
        print(f"  {'Symbol':<10} {'Broker':<25} {'Win':>4} {'Gain%':>7} "
              f"{'Streak':>7} {'NetUnits':>10} {'DaysBefore':>11}")
        print(f"  {'-'*10} {'-'*25} {'-'*4} {'-'*7} {'-'*7} {'-'*10} {'-'*11}")
        for _, r in top.head(40).iterrows():
            print(f"  {r['symbol']:<10} {r['broker_name']:<25} "
                  f"{r['window_days']:>3}d "
                  f"{r['gain_pct']:>6.1f}% "
                  f"{r['consecutive_buy_streak']:>7} "
                  f"{r['total_net_units']:>10,.0f} "
                  f"{str(r['days_before_move']):>11}")

    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(movers: list[dict],
                 records: list[dict],
                 summary: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    def _save(data, name):
        if not data:
            return
        path = OUT_DIR / f"{name}_{today}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        log.info("Saved: %s (%d rows)", path, len(data))

    _save(movers,   "qualified_movers")
    _save(records,  "broker_pre_move")
    _save(summary,  "broker_pre_move_summary")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Min gain %% (default {THRESHOLD})")
    parser.add_argument("--from-date", default=FROM_DATE)
    parser.add_argument("--to-date",   default=TO_DATE)
    args = parser.parse_args()

    # Load price data
    price_df = load_prices(args.from_date, args.to_date)
    if price_df.empty:
        return

    # Step 1: find qualified movers (IPO filter applied)
    movers = find_qualified_movers(price_df, args.threshold, MIN_PRIOR_DAYS)
    if not movers:
        log.error("No qualified moves found")
        return

    log.info("Qualified moves: %d across %d symbols",
             len(movers),
             len({m["symbol"] for m in movers}))
    for w in WINDOWS:
        wm = movers_by_window(movers, w)
        log.info("  %dd window: %d moves", w, len(wm))

    # Load broker floorsheet data
    broker_df = load_broker_daily(args.from_date, args.to_date)
    if broker_df.empty:
        log.error("No broker data loaded")
        return

    # Step 2: broker pre-move activity
    records = analyze_broker_pre_move(movers, broker_df)

    # Step 3: summary
    summary = build_summary(records, movers)

    # Print
    print_report(movers, summary, records)

    # Save
    save_outputs(movers, records, summary)


if __name__ == "__main__":
    main()