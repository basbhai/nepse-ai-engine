"""
broker_backtest.py
────────────────────────────────────────────────────────────────────────────
Hit-rate backtest for 5 smart brokers on NEPSE.

For every broker × symbol pair, finds ALL historical streaks (not just current)
and measures what price did 30/60/90 days after signal fired.

BUY streaks  (all 5 brokers) → expect price UP
SELL streaks (all 5 brokers) → expect price DOWN (except Naasa sell = UP?)

Output:
  - broker_backtest_summary.csv   — per broker hit rates
  - broker_backtest_signals.csv   — every signal instance with outcomes
  - broker_backtest_report.txt    — human-readable summary

Usage:
    cd ~/nepse-engine
    python stat_method/broker_backtest.py
    python stat_method/broker_backtest.py --min-streak 3   # looser threshold
    python stat_method/broker_backtest.py --min-streak 7   # stricter
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
    format="%(asctime)s [BACKTEST] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "output"

# ── Brokers ───────────────────────────────────────────────────────────────────
BROKERS = {
    "38": "Dipshikha Dhitopatra",
    "56": "Sri Hari Securities",
    "49": "Online Securities",
    "48": "Trishakti Securities",
    "58": "Naasa Securities",
}

# ── Config ────────────────────────────────────────────────────────────────────
MIN_STREAK     = 5      # consecutive days to count as a signal
MIN_NET_UNITS  = 1000   # minimum net units in streak
FORWARD_DAYS   = [30, 60, 90]
HIT_THRESHOLD  = 0.10   # 10% move = hit
FS_START       = "2023-07-01"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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


def load_broker_daily(from_date: str) -> pd.DataFrame:
    """Load daily net position for all 5 brokers from local postgres."""
    broker_ids = list(BROKERS.keys())
    log.info("Loading floorsheet for %d brokers from %s...", len(broker_ids), from_date)

    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    placeholders = ",".join(["%s"] * len(broker_ids))

    cur.execute(f"""
        WITH buys AS (
            SELECT date::date AS date, symbol,
                   buyer_broker_id  AS broker_id,
                   buyer_broker     AS broker_name,
                   SUM(quantity::float) AS buy_units
            FROM floorsheet
            WHERE date >= %s
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
            WHERE date >= %s
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
    """, [from_date] + broker_ids + [from_date] + broker_ids)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    log.info("  Loaded %d rows", len(rows))
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df

    df["date"]       = pd.to_datetime(df["date"])
    df["net_units"]  = df["buy_units"] - df["sell_units"]
    df["broker_id"]  = df["broker_id"].astype(str).str.strip()
    df["buy_units"]  = df["buy_units"].fillna(0)
    df["sell_units"] = df["sell_units"].fillna(0)
    return df.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


def load_price_history(from_date: str) -> pd.DataFrame:
    """Load full price history from Neon."""
    log.info("Loading price history from %s...", from_date)
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
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    log.info("  Loaded %d price rows for %d symbols",
             len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STREAK FINDER — finds ALL streaks, not just current
# ══════════════════════════════════════════════════════════════════════════════

def find_all_streaks(series: pd.Series, dates: pd.Series,
                     direction: str, min_streak: int,
                     min_units: float) -> list[dict]:
    """
    Find all completed and ongoing streaks in a net_units series.
    direction: 'buy' (positive runs) or 'sell' (negative runs)
    Returns list of dicts with streak_start, streak_end, streak_days, total_units.
    Only counts the FIRST day of each streak as the signal date (no overlap).
    """
    vals  = series.values
    dts   = dates.values
    n     = len(vals)
    streaks = []

    i = 0
    while i < n:
        v = vals[i]
        is_signal = (v > 0) if direction == "buy" else (v < 0)

        if not is_signal:
            i += 1
            continue

        # Start of a streak — walk forward
        j = i
        total = 0.0
        while j < n:
            vj = vals[j]
            still_going = (vj > 0) if direction == "buy" else (vj < 0)
            if still_going:
                total += abs(vj)
                j += 1
            else:
                break

        length = j - i
        if length >= min_streak and total >= min_units:
            streaks.append({
                "signal_date":  pd.Timestamp(dts[i]).date(),
                "streak_end":   pd.Timestamp(dts[j-1]).date(),
                "streak_days":  length,
                "total_units":  round(total, 0),
                "avg_daily":    round(total / length, 0),
            })

        i = j  # jump past this streak entirely (no overlap)

    return streaks


# ══════════════════════════════════════════════════════════════════════════════
# PRICE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def get_price_on_or_after(price_df: pd.DataFrame, symbol: str,
                          target_date) -> float | None:
    """Get closest closing price on or after target_date for symbol."""
    sym = price_df[price_df["symbol"] == symbol]
    td  = pd.Timestamp(target_date)
    fut = sym[sym["date"] >= td]
    if fut.empty:
        return None
    return float(fut.iloc[0]["close"])


def get_price_on_or_before(price_df: pd.DataFrame, symbol: str,
                           target_date) -> float | None:
    """Get closest closing price on or before target_date for symbol."""
    sym = price_df[price_df["symbol"] == symbol]
    td  = pd.Timestamp(target_date)
    past = sym[sym["date"] <= td]
    if past.empty:
        return None
    return float(past.iloc[-1]["close"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, price_df: pd.DataFrame,
                 min_streak: int, min_units: int) -> list[dict]:
    """
    For every broker × symbol × direction, find all streaks and measure outcomes.
    """
    records = []
    today   = date.today()
    cutoff  = today - timedelta(days=90)  # need 90d of future price to evaluate

    total_pairs = df.groupby(["broker_id", "symbol"]).ngroups
    log.info("Scanning %d broker×symbol pairs...", total_pairs)

    done = 0
    for (broker_id, symbol), grp in df.groupby(["broker_id", "symbol"]):
        grp = grp.sort_values("date").reset_index(drop=True)
        broker_name = BROKERS.get(broker_id, broker_id)

        for direction in ["buy", "sell"]:
            streaks = find_all_streaks(
                grp["net_units"], grp["date"],
                direction, min_streak, min_units
            )

            for streak in streaks:
                sig_date  = streak["signal_date"]
                entry_price = get_price_on_or_after(price_df, symbol, sig_date)

                if entry_price is None:
                    continue

                row = {
                    "broker_id":    broker_id,
                    "broker_name":  broker_name,
                    "symbol":       symbol,
                    "direction":    direction,
                    "signal_date":  sig_date,
                    "streak_days":  streak["streak_days"],
                    "total_units":  streak["total_units"],
                    "avg_daily":    streak["avg_daily"],
                    "entry_price":  entry_price,
                }

                # Forward price at 30/60/90 days
                for fwd in FORWARD_DAYS:
                    fwd_date  = sig_date + timedelta(days=fwd)
                    fwd_price = get_price_on_or_after(price_df, symbol, fwd_date)

                    if fwd_price is None:
                        row[f"price_{fwd}d"]  = None
                        row[f"return_{fwd}d"] = None
                        row[f"hit_{fwd}d"]    = None
                    else:
                        ret = (fwd_price - entry_price) / entry_price
                        row[f"price_{fwd}d"]  = round(fwd_price, 2)
                        row[f"return_{fwd}d"] = round(ret * 100, 2)

                        # Hit = direction-adjusted
                        if direction == "buy":
                            row[f"hit_{fwd}d"] = ret >= HIT_THRESHOLD
                        else:
                            # sell signal → price should DROP
                            # exception: Naasa sell → price should rise
                            if broker_id == "58":
                                row[f"hit_{fwd}d"] = ret >= HIT_THRESHOLD
                            else:
                                row[f"hit_{fwd}d"] = ret <= -HIT_THRESHOLD

                # Mark if signal is too recent to evaluate
                row["evaluable"] = sig_date <= cutoff

                records.append(row)

        done += 1
        if done % 500 == 0:
            log.info("  %d / %d pairs done", done, total_pairs)

    log.info("Total signal instances found: %d", len(records))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(records: list[dict]) -> list[dict]:
    """Aggregate hit rates per broker × direction × window."""
    df = pd.DataFrame(records)
    df = df[df["evaluable"] == True]  # only signals with enough forward data

    summary = []
    for (broker_id, broker_name, direction), grp in df.groupby(
            ["broker_id", "broker_name", "direction"]):

        row = {
            "broker_id":    broker_id,
            "broker_name":  broker_name,
            "direction":    direction,
            "total_signals": len(grp),
            "symbols_covered": grp["symbol"].nunique(),
        }

        for fwd in FORWARD_DAYS:
            col_ret = f"return_{fwd}d"
            col_hit = f"hit_{fwd}d"

            valid = grp[grp[col_ret].notna()]
            if len(valid) == 0:
                row[f"hit_rate_{fwd}d"]   = None
                row[f"avg_return_{fwd}d"] = None
                row[f"med_return_{fwd}d"] = None
                continue

            hits = valid[col_hit].sum()
            row[f"hit_rate_{fwd}d"]   = round(hits / len(valid) * 100, 1)
            row[f"avg_return_{fwd}d"] = round(valid[col_ret].mean(), 1)
            row[f"med_return_{fwd}d"] = round(valid[col_ret].median(), 1)

        summary.append(row)

    return sorted(summary, key=lambda x: (x["broker_name"], x["direction"]))


def print_report(summary: list[dict], records: list[dict]) -> str:
    """Print and return human-readable report."""
    lines = []
    sep  = "=" * 80
    sep2 = "-" * 80

    lines.append(sep)
    lines.append("SMART BROKER BACKTEST REPORT")
    lines.append(f"Generated: {date.today()}")
    lines.append(f"Signal threshold: ≥{MIN_STREAK} consecutive days, ≥{MIN_NET_UNITS} net units")
    lines.append(f"Hit threshold: ±{HIT_THRESHOLD*100:.0f}% move")
    lines.append(f"Forward windows: {FORWARD_DAYS} days")
    lines.append(sep)

    total_signals = len([r for r in records if r["evaluable"]])
    lines.append(f"\nTotal evaluable signal instances: {total_signals}")
    lines.append("")

    for row in summary:
        direction_label = "BUY streak" if row["direction"] == "buy" else "SELL streak"
        expected = "price UP" if (
            row["direction"] == "buy" or row["broker_id"] == "58"
        ) else "price DOWN"

        lines.append(sep2)
        lines.append(f"{row['broker_name']} ({row['broker_id']}) — {direction_label}")
        lines.append(f"  Signals: {row['total_signals']}  |  Symbols: {row['symbols_covered']}  |  Expected: {expected}")
        lines.append("")
        lines.append(f"  {'Window':<10} {'Hit Rate':>10} {'Avg Return':>12} {'Median Return':>14}")
        lines.append(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*14}")

        for fwd in FORWARD_DAYS:
            hr  = row.get(f"hit_rate_{fwd}d")
            avg = row.get(f"avg_return_{fwd}d")
            med = row.get(f"med_return_{fwd}d")
            hr_str  = f"{hr:.1f}%"  if hr  is not None else "N/A"
            avg_str = f"{avg:+.1f}%" if avg is not None else "N/A"
            med_str = f"{med:+.1f}%" if med is not None else "N/A"
            lines.append(f"  {fwd}d{'':<7} {hr_str:>10} {avg_str:>12} {med_str:>14}")

        lines.append("")

    lines.append(sep)
    lines.append("VERDICT SUMMARY")
    lines.append(sep2)

    for row in summary:
        hr_60 = row.get("hit_rate_60d")
        if hr_60 is None:
            continue
        direction_label = "BUY" if row["direction"] == "buy" else "SELL"
        if hr_60 >= 60:
            verdict = "✅ STRONG SIGNAL"
        elif hr_60 >= 45:
            verdict = "🟡 WEAK SIGNAL"
        else:
            verdict = "❌ NO EDGE"

        lines.append(
            f"  {row['broker_name']:<25} {direction_label:<5} "
            f"60d hit={hr_60:.1f}%  {verdict}"
        )

    report = "\n".join(lines)
    print(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(records: list[dict], summary: list[dict], report: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    today = date.today().strftime("%Y-%m-%d")

    # All signal instances
    signals_path = OUT_DIR / f"broker_backtest_signals_{today}.csv"
    if records:
        keys = list(records[0].keys())
        with open(signals_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        log.info("Signals saved: %s", signals_path)

    # Summary
    summary_path = OUT_DIR / f"broker_backtest_summary_{today}.csv"
    if summary:
        keys = list(summary[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)
        log.info("Summary saved: %s", summary_path)

    # Report
    report_path = OUT_DIR / f"broker_backtest_report_{today}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("Report saved: %s", report_path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-streak", type=int, default=MIN_STREAK)
    parser.add_argument("--min-units",  type=int, default=MIN_NET_UNITS)
    args = parser.parse_args()

    # Load data
    df       = load_broker_daily(FS_START)
    price_df = load_price_history(FS_START)

    if df.empty:
        log.error("No floorsheet data — aborting")
        return
    if price_df.empty:
        log.error("No price data — aborting")
        return

    log.info("Floorsheet: %d rows | %d symbols | %s → %s",
             len(df), df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date())

    # Run backtest
    records = run_backtest(df, price_df, args.min_streak, args.min_units)

    if not records:
        log.error("No signals found")
        return

    # Summarize
    summary = build_summary(records)
    report  = print_report(summary, records)

    # Save
    save_outputs(records, summary, report)


if __name__ == "__main__":
    main()