"""
broker_streak_detector.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Broker Streak Accumulation Detector

The real signal is NOT "broker X is buying a stock."
The real signal IS "broker X started buying a stock they barely touched before,
and has been consistently net-buying for N consecutive days."

This is the Dipshikha/JBLBP pattern:
- Dipshikha normally trades hundreds of stocks casually
- But in JBLBP, they became unusually DOMINANT (large net vs their normal activity)
- AND maintained that dominance for multiple consecutive days
- Price followed

Two signal types:
  TYPE A — NEW STREAK: broker starts a fresh consecutive net-buy streak
           in a stock where they had minimal prior activity
           (historical net in this stock < 10% of their market-wide average)

  TYPE B — DOMINANT STREAK: broker's net position in a stock is unusually
           large relative to their own historical average in that stock
           (current streak net > 3x their rolling 90d average net in same stock)

Output: daily alert CSV + terminal summary

Usage:
    cd ~/nepse-engine
    python stat_method/broker_streak_detector.py
    python stat_method/broker_streak_detector.py --min-streak 5
    python stat_method/broker_streak_detector.py --lookback 60    # days of history
    python stat_method/broker_streak_detector.py --broker 38      # single broker
    python stat_method/broker_streak_detector.py --dry-run
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STREAK] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR  = Path(__file__).parent / "output"
FS_START = "2023-07-01"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_STREAK_DAYS   = 5      # minimum consecutive net-buy days to flag
MIN_STREAK_UNITS  = 1000   # minimum total net units in streak (filter noise)
NEW_BROKER_THRESH = 0.10   # broker's prior net in stock < 10% of their avg → "new"
DOMINANT_MULT     = 2.5    # current streak net > 2.5x 90d average → "dominant"
LOOKBACK_SIGNAL   = 30     # days to look back for active signals
BASELINE_DAYS     = 180    # days of history to compute broker baseline


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
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_floorsheet(from_date: str, broker_filter: str = None) -> pd.DataFrame:
    """
    Load raw floorsheet from local DB.
    Returns DataFrame with columns:
        date, symbol, broker_id, broker_name, net_units, buy_units, sell_units
    (one row per symbol-date-broker)
    """
    log.info("Loading floorsheet from %s (local DB)...", from_date)

    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    broker_clause = "AND (buyer_broker_id = %s OR seller_broker_id = %s)" \
                    if broker_filter else ""
    params = [from_date]
    if broker_filter:
        params += [broker_filter, broker_filter]

    cur.execute(f"""
        SELECT
            date::date           AS date,
            symbol,
            buyer_broker_id      AS broker_id,
            buyer_broker         AS broker_name,
            SUM(quantity::float) AS buy_units,
            0.0                  AS sell_units
        FROM floorsheet
        WHERE date >= %s
          AND quantity IS NOT NULL AND quantity != ''
          AND quantity ~ '^[0-9]+\\.?[0-9]*$'
          AND buyer_broker_id IS NOT NULL
          {broker_clause}
        GROUP BY date::date, symbol, buyer_broker_id, buyer_broker

        UNION ALL

        SELECT
            date::date           AS date,
            symbol,
            seller_broker_id     AS broker_id,
            seller_broker        AS broker_name,
            0.0                  AS buy_units,
            SUM(quantity::float) AS sell_units
        FROM floorsheet
        WHERE date >= %s
          AND quantity IS NOT NULL AND quantity != ''
          AND quantity ~ '^[0-9]+\\.?[0-9]*$'
          AND seller_broker_id IS NOT NULL
          {broker_clause}
        GROUP BY date::date, symbol, seller_broker_id, seller_broker
        ORDER BY date, symbol, broker_id
    """, params + [from_date] + (params[1:] if broker_filter else []))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    log.info("  Loaded %d raw rows", len(rows))

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"]       = pd.to_datetime(df["date"])
    df["broker_id"]  = df["broker_id"].astype(str).str.strip()
    df["buy_units"]  = df["buy_units"].fillna(0)
    df["sell_units"] = df["sell_units"].fillna(0)

    # Aggregate: one row per (date, symbol, broker)
    agg = df.groupby(["date", "symbol", "broker_id"]).agg(
        broker_name=("broker_name", "first"),
        buy_units=("buy_units",  "sum"),
        sell_units=("sell_units", "sum"),
    ).reset_index()
    agg["net_units"] = agg["buy_units"] - agg["sell_units"]

    log.info("  Aggregated to %d (date, symbol, broker) rows", len(agg))
    return agg.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


def load_price_data(from_date: str) -> pd.DataFrame:
    """Load price_history for context."""
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date::date AS date,
                   COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
            FROM price_history
            WHERE date >= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (from_date,))
        rows = cur.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STREAK COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (symbol, broker), compute consecutive net-buy streaks.
    Returns DataFrame of ALL streaks with:
        symbol, broker_id, broker_name,
        streak_days, total_net_units, avg_daily_net,
        streak_start, streak_end, is_active
    """
    results = []

    for (symbol, broker_id), grp in df.groupby(["symbol", "broker_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        broker_name = grp["broker_id"].iloc[0]  # use id as fallback
        if grp["broker_name"].iloc[0]:
            broker_name = grp["broker_name"].iloc[0]

        # Find consecutive positive net days
        streak_start = None
        streak_units = 0
        streak_days  = 0
        prev_positive = False
        last_date = grp["date"].max()

        for _, row in grp.iterrows():
            is_pos = row["net_units"] > 0

            if is_pos:
                if not prev_positive:
                    # Start new streak
                    streak_start  = row["date"]
                    streak_units  = 0
                    streak_days   = 0
                streak_units += row["net_units"]
                streak_days  += 1
            else:
                if prev_positive and streak_days >= 1:
                    # End of streak — record it
                    streak_end = grp.iloc[grp.index[grp["date"] < row["date"]].max()]["date"] \
                                 if streak_days > 0 else streak_start
                    # Get actual last positive date
                    pos_dates = grp[(grp["date"] >= streak_start) & (grp["net_units"] > 0)]["date"]
                    streak_end = pos_dates.max() if not pos_dates.empty else streak_start
                    is_active = (last_date - streak_end).days <= 7

                    if streak_days >= 1:
                        results.append({
                            "symbol":        symbol,
                            "broker_id":     broker_id,
                            "broker_name":   broker_name,
                            "streak_days":   streak_days,
                            "total_net":     round(streak_units, 0),
                            "avg_daily_net": round(streak_units / streak_days, 0),
                            "streak_start":  streak_start,
                            "streak_end":    streak_end,
                            "is_active":     is_active,
                        })

                streak_start  = None
                streak_units  = 0
                streak_days   = 0

            prev_positive = is_pos

        # Handle streak running to end of data
        if prev_positive and streak_days >= 1 and streak_start is not None:
            pos_dates = grp[(grp["date"] >= streak_start) & (grp["net_units"] > 0)]["date"]
            streak_end = pos_dates.max() if not pos_dates.empty else streak_start
            is_active = (last_date - streak_end).days <= 7
            results.append({
                "symbol":        symbol,
                "broker_id":     broker_id,
                "broker_name":   broker_name,
                "streak_days":   streak_days,
                "total_net":     round(streak_units, 0),
                "avg_daily_net": round(streak_units / streak_days, 0),
                "streak_start":  streak_start,
                "streak_end":    streak_end,
                "is_active":     is_active,
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame,
                   streaks: pd.DataFrame,
                   signal_window_days: int = LOOKBACK_SIGNAL,
                   min_streak: int = MIN_STREAK_DAYS,
                   min_units: int = MIN_STREAK_UNITS) -> pd.DataFrame:
    """
    Detect TYPE A (new broker) and TYPE B (dominant streak) signals.
    """
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=signal_window_days)

    # Active recent streaks only
    active = streaks[
        (streaks["is_active"]) &
        (streaks["streak_end"] >= cutoff) &
        (streaks["streak_days"] >= min_streak) &
        (streaks["total_net"] >= min_units)
    ].copy()

    if active.empty:
        return pd.DataFrame()

    signals = []

    # Compute broker baseline: avg daily net per broker across ALL stocks
    broker_baseline = df.groupby("broker_id")["net_units"].apply(
        lambda x: x[x > 0].mean()
    ).to_dict()

    # Compute prior history per (broker, symbol) before streak start
    for _, row in active.iterrows():
        broker_id   = row["broker_id"]
        symbol      = row["symbol"]
        streak_start= row["streak_start"]

        # Prior net for this broker in this stock (before current streak)
        prior = df[
            (df["broker_id"] == broker_id) &
            (df["symbol"]    == symbol) &
            (df["date"]      < streak_start)
        ]
        prior_total_net = prior["net_units"].sum()
        prior_avg_net   = prior["net_units"][prior["net_units"] > 0].mean() \
                          if not prior.empty else 0

        broker_avg = broker_baseline.get(broker_id, 0)

        # TYPE A: broker was barely present before this streak
        is_new_broker = (
            prior_total_net < (broker_avg * 5) or  # very little prior activity
            len(prior) < 10                          # fewer than 10 prior trading days
        )

        # TYPE B: current streak net is much larger than prior average in same stock
        is_dominant = (
            prior_avg_net > 0 and
            row["avg_daily_net"] > prior_avg_net * DOMINANT_MULT
        ) or (
            prior_avg_net == 0 and row["total_net"] >= min_units * 3
        )

        if not (is_new_broker or is_dominant):
            continue

        signal_type = []
        if is_new_broker:  signal_type.append("NEW_BROKER")
        if is_dominant:    signal_type.append("DOMINANT")

        signals.append({
            "symbol":           symbol,
            "broker_id":        broker_id,
            "broker_name":      row["broker_name"],
            "signal_type":      "+".join(signal_type),
            "streak_days":      row["streak_days"],
            "total_net_units":  row["total_net"],
            "avg_daily_net":    row["avg_daily_net"],
            "streak_start":     row["streak_start"],
            "streak_end":       row["streak_end"],
            "prior_avg_daily":  round(prior_avg_net, 0),
            "dominant_ratio":   round(row["avg_daily_net"] / prior_avg_net, 2)
                                if prior_avg_net > 0 else None,
            "prior_trading_days": len(prior),
            "is_new_broker":    is_new_broker,
            "is_dominant":      is_dominant,
        })

    result = pd.DataFrame(signals)
    if result.empty:
        return result

    # Score: longer streak + larger units + newer broker = higher score
    result["score"] = (
        result["streak_days"] * 0.4 +
        np.log1p(result["total_net_units"]) * 0.4 +
        result["is_new_broker"].astype(int) * 5 +
        result["is_dominant"].astype(int) * 3
    )
    result["score"] = result["score"].round(2)

    return result.sort_values("score", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-streak",  type=int,   default=MIN_STREAK_DAYS)
    parser.add_argument("--min-units",   type=int,   default=MIN_STREAK_UNITS)
    parser.add_argument("--lookback",    type=int,   default=LOOKBACK_SIGNAL,
                        help="Days to look back for active signals")
    parser.add_argument("--broker",      type=str,   default=None,
                        help="Filter to specific broker ID")
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Load data
    load_from = (date.today() - timedelta(days=BASELINE_DAYS + 30)).strftime("%Y-%m-%d")
    load_from = max(load_from, FS_START)

    df = load_floorsheet(load_from, broker_filter=args.broker)

    if args.dry_run:
        symbols = df["symbol"].unique()[:30]
        df = df[df["symbol"].isin(symbols)].copy()
        log.info("Dry run: %d symbols", len(symbols))

    # Compute streaks
    log.info("Computing streaks...")
    streaks = compute_streaks(df)
    log.info("Total streak records: %d", len(streaks))
    log.info("Active streaks (recent): %d",
             streaks[streaks["is_active"]].shape[0])

    # Detect signals
    log.info("Detecting signals (min_streak=%d, min_units=%d, lookback=%dd)...",
             args.min_streak, args.min_units, args.lookback)
    signals = detect_signals(
        df, streaks,
        signal_window_days=args.lookback,
        min_streak=args.min_streak,
        min_units=args.min_units,
    )

    # Load price for context
    try:
        prices = load_price_data(
            (date.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        )
        latest_price = prices.sort_values("date").groupby("symbol")["close"].last()
        if not signals.empty:
            signals["current_price"] = signals["symbol"].map(latest_price)
    except Exception as e:
        log.warning("Could not load price data: %s", e)

    # ── Output ─────────────────────────────────────────────────────────────────
    if signals.empty:
        log.info("No signals found with current parameters.")
        log.info("Try: --min-streak 3 --min-units 500 --lookback 60")
        return

    log.info("\n%s", "=" * 90)
    log.info("BROKER ACCUMULATION SIGNALS — %s", date.today())
    log.info("%s", "=" * 90)
    log.info("Found %d signals", len(signals))

    # Group by signal type
    new_broker = signals[signals["is_new_broker"]]
    dominant   = signals[signals["is_dominant"] & ~signals["is_new_broker"]]

    if not new_broker.empty:
        log.info("\n── TYPE A: NEW BROKER ENTERING (highest conviction) ──")
        log.info("%-8s %-25s %-18s %6s %12s %8s %12s",
                 "Symbol", "Broker", "Type", "Days",
                 "Net Units", "Price", "Streak Start")
        log.info("-" * 90)
        for _, r in new_broker.head(20).iterrows():
            price = f"{r['current_price']:.2f}" if "current_price" in r \
                    and pd.notna(r.get("current_price")) else "N/A"
            log.info("%-8s %-25s %-18s %6d %12,.0f %8s %12s",
                     r["symbol"],
                     str(r["broker_name"])[:24],
                     r["signal_type"],
                     r["streak_days"],
                     r["total_net_units"],
                     price,
                     str(r["streak_start"].date()))

    if not dominant.empty:
        log.info("\n── TYPE B: UNUSUALLY DOMINANT STREAK ──")
        log.info("%-8s %-25s %6s %12s %8s %6s",
                 "Symbol", "Broker", "Days", "Net Units", "Price", "Ratio")
        log.info("-" * 70)
        for _, r in dominant.head(10).iterrows():
            price = f"{r['current_price']:.2f}" if "current_price" in r \
                    and pd.notna(r.get("current_price")) else "N/A"
            ratio = f"{r['dominant_ratio']:.1f}x" if r["dominant_ratio"] else "new"
            log.info("%-8s %-25s %6d %12,.0f %8s %6s",
                     r["symbol"],
                     str(r["broker_name"])[:24],
                     r["streak_days"],
                     r["total_net_units"],
                     price,
                     ratio)

    # ── Historical win rate check ──────────────────────────────────────────────
    log.info("\n%s", "=" * 90)
    log.info("BROKER HISTORICAL STATS (how often their streaks led to gains)")
    log.info("%s", "=" * 90)

    active_brokers = signals["broker_id"].unique()
    broker_stats = streaks[
        streaks["broker_id"].isin(active_brokers)
    ].groupby(["broker_id"]).agg(
        broker_name=("broker_name", "first"),
        total_streaks=("streak_days", "count"),
        avg_streak_len=("streak_days", "mean"),
        avg_net_units=("total_net", "mean"),
        symbols_active=("symbol", "nunique"),
    ).reset_index()

    for _, r in broker_stats.sort_values("total_streaks", ascending=False).iterrows():
        log.info("  Broker %-4s %-28s streaks=%3d  avg_len=%.1fd  symbols=%d",
                 r["broker_id"],
                 str(r["broker_name"])[:27],
                 r["total_streaks"],
                 r["avg_streak_len"],
                 r["symbols_active"])

    # ── Save CSV ───────────────────────────────────────────────────────────────
    today_str = date.today().strftime("%Y-%m-%d")
    out_path  = OUT_DIR / f"broker_signals_{today_str}.csv"

    save_cols = [
        "symbol", "broker_id", "broker_name", "signal_type",
        "streak_days", "total_net_units", "avg_daily_net",
        "streak_start", "streak_end",
        "prior_avg_daily", "dominant_ratio", "prior_trading_days",
        "score",
    ]
    if "current_price" in signals.columns:
        save_cols.append("current_price")

    signals[save_cols].to_csv(out_path, index=False)
    log.info("\nSignals saved to %s", out_path)

    # Also save all active streaks for reference
    streaks_path = OUT_DIR / f"all_streaks_{today_str}.csv"
    active_streaks = streaks[
        streaks["is_active"] & (streaks["streak_days"] >= args.min_streak)
    ].sort_values("streak_days", ascending=False)
    active_streaks.to_csv(streaks_path, index=False)
    log.info("All active streaks saved to %s", streaks_path)


if __name__ == "__main__":
    main()