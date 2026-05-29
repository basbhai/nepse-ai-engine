"""
broker_accumulation_report.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Broker Accumulation Report

Step 1: Find all symbols that gained ≥15% within any 45-day window
        from a given start date.
Step 2: For each such symbol, pull raw floorsheet and summarise
        per-broker: total units bought, total units sold, net position,
        total turnover bought, total turnover sold, number of trades.
Step 3: Write one CSV per symbol + a combined summary CSV.

Output: stat_method/output/broker_reports/
        ├── SYMBOL1_broker_summary.csv
        ├── SYMBOL2_broker_summary.csv
        └── _all_symbols_summary.csv

Usage:
    cd ~/nepse-engine

    # Default: July 2025 onward, ≥15% in 45 days
    python stat_method/broker_accumulation_report.py

    # Custom date range and threshold
    python stat_method/broker_accumulation_report.py \\
        --from-date 2024-01-01 --to-date 2026-05-27 --min-gain 20

    # Single symbol
    python stat_method/broker_accumulation_report.py --symbol AKJCL
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BROKER] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_FROM   = "2025-07-01"
DEFAULT_TO     = "2026-05-27"
MIN_GAIN_PCT   = 15.0
LOOKBACK_DAYS  = 45      # calendar-day window to check for ≥15% gain


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Find qualifying symbols
# ══════════════════════════════════════════════════════════════════════════════

def find_qualifying_symbols(from_date: str, to_date: str,
                             min_gain: float, symbol_filter: str = None) -> list[dict]:
    """
    Find all symbols where close gained ≥min_gain% within any 45-trading-day
    window between from_date and to_date.

    Returns list of:
        {symbol, first_date, first_close, peak_date, peak_close, gain_pct,
         window_start, window_end}
    """
    log.info("Finding symbols with ≥%.0f%% gain in %d days (%s → %s)...",
             min_gain, LOOKBACK_DAYS, from_date, to_date)

    sym_filter_clause = "AND symbol = %s" if symbol_filter else ""
    params = [from_date, to_date]
    if symbol_filter:
        params.append(symbol_filter.upper())

    with _db() as cur:
        cur.execute(f"""
            WITH ranked AS (
                SELECT
                    symbol,
                    date,
                    close::float AS c,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
                FROM price_history
                WHERE date >= %s
                  AND date <= %s
                  AND close IS NOT NULL AND close != ''
                  AND close ~ '^[0-9]+\\.?[0-9]*$'
                  AND close::float > 0
                  {sym_filter_clause}
            ),
            pairs AS (
                SELECT
                    r1.symbol,
                    r1.date  AS start_date,
                    r1.c     AS start_close,
                    r2.date  AS end_date,
                    r2.c     AS end_close,
                    ROUND(((r2.c - r1.c) / r1.c * 100)::numeric, 2) AS gain_pct
                FROM ranked r1
                JOIN ranked r2
                  ON r2.symbol = r1.symbol
                  AND r2.rn > r1.rn
                  AND r2.rn <= r1.rn + {LOOKBACK_DAYS}
                WHERE r2.c >= r1.c * (1 + %s / 100.0)
            ),
            best AS (
                SELECT DISTINCT ON (symbol)
                    symbol, start_date, start_close,
                    end_date, end_close, gain_pct
                FROM pairs
                ORDER BY symbol, gain_pct DESC
            )
            SELECT * FROM best ORDER BY gain_pct DESC
        """, params + [min_gain])
        rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "symbol":       str(r["symbol"]).upper(),
            "window_start": r["start_date"],
            "window_end":   r["end_date"],
            "start_close":  round(float(r["start_close"]), 2),
            "peak_close":   round(float(r["end_close"]), 2),
            "gain_pct":     round(float(r["gain_pct"]), 2),
        })

    log.info("Found %d qualifying symbols", len(results))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Pull raw floorsheet and summarise per broker
# ══════════════════════════════════════════════════════════════════════════════

def load_floorsheet_for_symbol(symbol: str,
                                from_date: str, to_date: str) -> list[dict]:
    """Load raw floorsheet rows for a symbol between dates."""
    with _db() as cur:
        cur.execute("""
            SELECT
                date,
                buyer_broker_id,
                buyer_broker,
                seller_broker_id,
                seller_broker,
                quantity::float  AS qty,
                rate::float      AS rate,
                amount::float    AS amount
            FROM floorsheet
            WHERE symbol = %s
              AND date >= %s
              AND date <= %s
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            ORDER BY date ASC
        """, (symbol, from_date, to_date))
        return cur.fetchall()


def summarise_by_broker(rows: list[dict]) -> dict[str, dict]:
    """
    Aggregate floorsheet rows per broker.
    Each broker appears as both buyer and seller across trades.

    Returns: broker_id → {
        broker_id, broker_name,
        buy_units, buy_turnover, buy_trades,
        sell_units, sell_turnover, sell_trades,
        net_units (buy - sell),
        net_turnover (buy - sell),
        avg_buy_rate, avg_sell_rate,
        first_date, last_date
    }
    """
    brokers: dict[str, dict] = defaultdict(lambda: {
        "broker_id":     "",
        "broker_name":   "",
        "buy_units":     0.0,
        "buy_turnover":  0.0,
        "buy_trades":    0,
        "sell_units":    0.0,
        "sell_turnover": 0.0,
        "sell_trades":   0,
        "first_date":    None,
        "last_date":     None,
    })

    for r in rows:
        try:
            qty    = float(r["qty"])    if r["qty"]    else 0.0
            amount = float(r["amount"]) if r["amount"] else qty * float(r["rate"] or 0)
            date   = r["date"]
        except (ValueError, TypeError):
            continue

        # Buyer side
        bid = str(r["buyer_broker_id"] or "").strip()
        if bid and bid not in ("", "None", "0"):
            b = brokers[bid]
            b["broker_id"]    = bid
            b["broker_name"]  = str(r["buyer_broker"] or bid).strip()
            b["buy_units"]    += qty
            b["buy_turnover"] += amount
            b["buy_trades"]   += 1
            if b["first_date"] is None or date < b["first_date"]:
                b["first_date"] = date
            if b["last_date"] is None or date > b["last_date"]:
                b["last_date"] = date

        # Seller side
        sid = str(r["seller_broker_id"] or "").strip()
        if sid and sid not in ("", "None", "0"):
            s = brokers[sid]
            s["broker_id"]     = sid
            s["broker_name"]   = str(r["seller_broker"] or sid).strip()
            s["sell_units"]    += qty
            s["sell_turnover"] += amount
            s["sell_trades"]   += 1
            if s["first_date"] is None or date < s["first_date"]:
                s["first_date"] = date
            if s["last_date"] is None or date > s["last_date"]:
                s["last_date"] = date

    # Compute derived fields
    result = {}
    for bid, b in brokers.items():
        net_units    = b["buy_units"]    - b["sell_units"]
        net_turnover = b["buy_turnover"] - b["sell_turnover"]
        avg_buy_rate  = (b["buy_turnover"]  / b["buy_units"])  if b["buy_units"]  > 0 else 0.0
        avg_sell_rate = (b["sell_turnover"] / b["sell_units"]) if b["sell_units"] > 0 else 0.0
        result[bid] = {
            "broker_id":       b["broker_id"],
            "broker_name":     b["broker_name"],
            "buy_units":       round(b["buy_units"],    0),
            "buy_turnover":    round(b["buy_turnover"], 2),
            "buy_trades":      b["buy_trades"],
            "sell_units":      round(b["sell_units"],   0),
            "sell_turnover":   round(b["sell_turnover"],2),
            "sell_trades":     b["sell_trades"],
            "net_units":       round(net_units,    0),
            "net_turnover":    round(net_turnover, 2),
            "avg_buy_rate":    round(avg_buy_rate,  2),
            "avg_sell_rate":   round(avg_sell_rate, 2),
            "first_date":      b["first_date"],
            "last_date":       b["last_date"],
            "total_trades":    b["buy_trades"] + b["sell_trades"],
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Write CSVs
# ══════════════════════════════════════════════════════════════════════════════

BROKER_COLS = [
    "broker_id", "broker_name",
    "buy_units", "buy_turnover", "buy_trades", "avg_buy_rate",
    "sell_units", "sell_turnover", "sell_trades", "avg_sell_rate",
    "net_units", "net_turnover",
    "total_trades", "first_date", "last_date",
]

SUMMARY_COLS = [
    "symbol", "window_start", "window_end",
    "start_close", "peak_close", "gain_pct",
    "total_floorsheet_rows",
    "unique_brokers",
    "top_net_buyer_broker", "top_net_buyer_units",
    "top_net_seller_broker", "top_net_seller_units",
    "top_volume_broker", "top_volume_units",
]


def write_symbol_csv(symbol: str, broker_data: dict,
                     out_dir: Path, symbol_meta: dict) -> None:
    """Write per-broker CSV for one symbol, sorted by net_units DESC."""
    rows = sorted(broker_data.values(),
                  key=lambda x: x["net_units"], reverse=True)

    path = out_dir / f"{symbol}_broker_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BROKER_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in BROKER_COLS})

    log.info("  Written %s (%d brokers)", path.name, len(rows))


def write_combined_csv(all_summaries: list[dict], out_dir: Path) -> None:
    """Write the combined summary row for all qualifying symbols."""
    path = out_dir / "_all_symbols_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        for row in all_summaries:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_COLS})
    log.info("Combined summary written: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date",  default=DEFAULT_FROM)
    parser.add_argument("--to-date",    default=DEFAULT_TO)
    parser.add_argument("--min-gain",   type=float, default=MIN_GAIN_PCT)
    parser.add_argument("--symbol",     default=None,
                        help="Run for a single symbol only")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "output" / "broker_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — find qualifying symbols
    qualifiers = find_qualifying_symbols(
        from_date=args.from_date,
        to_date=args.to_date,
        min_gain=args.min_gain,
        symbol_filter=args.symbol,
    )

    if not qualifiers:
        log.info("No qualifying symbols found.")
        return

    log.info("\nQualifying symbols:")
    for q in qualifiers[:20]:
        log.info("  %-10s  +%.1f%%  (%s → %s)  %.2f → %.2f",
                 q["symbol"], q["gain_pct"],
                 q["window_start"], q["window_end"],
                 q["start_close"], q["peak_close"])
    if len(qualifiers) > 20:
        log.info("  ... and %d more", len(qualifiers) - 20)

    # Step 2 + 3 — per symbol
    all_summaries = []

    for q in qualifiers:
        symbol = q["symbol"]
        log.info("\nProcessing %s (+%.1f%%)...", symbol, q["gain_pct"])

        # Pull floorsheet for the gain window + 15 days before
        # (to capture pre-move accumulation)
        fs_rows = load_floorsheet_for_symbol(
            symbol=symbol,
            from_date=q["window_start"],
            to_date=q["window_end"],
        )

        if not fs_rows:
            log.info("  No floorsheet data for %s in this window", symbol)
            all_summaries.append({
                **{k: q[k] for k in ["symbol","window_start","window_end",
                                      "start_close","peak_close","gain_pct"]},
                "total_floorsheet_rows": 0,
                "unique_brokers": 0,
                "top_net_buyer_broker": "",
                "top_net_buyer_units": "",
                "top_net_seller_broker": "",
                "top_net_seller_units": "",
                "top_volume_broker": "",
                "top_volume_units": "",
            })
            continue

        broker_data = summarise_by_broker(fs_rows)

        # Write per-symbol CSV
        write_symbol_csv(symbol, broker_data, out_dir, q)

        # Build summary row
        sorted_by_net  = sorted(broker_data.values(),
                                key=lambda x: x["net_units"], reverse=True)
        sorted_by_vol  = sorted(broker_data.values(),
                                key=lambda x: x["buy_units"] + x["sell_units"],
                                reverse=True)

        top_buyer  = sorted_by_net[0]  if sorted_by_net else {}
        top_seller = sorted_by_net[-1] if len(sorted_by_net) > 1 else {}
        top_vol    = sorted_by_vol[0]  if sorted_by_vol else {}

        all_summaries.append({
            "symbol":               symbol,
            "window_start":         q["window_start"],
            "window_end":           q["window_end"],
            "start_close":          q["start_close"],
            "peak_close":           q["peak_close"],
            "gain_pct":             q["gain_pct"],
            "total_floorsheet_rows":len(fs_rows),
            "unique_brokers":       len(broker_data),
            "top_net_buyer_broker": top_buyer.get("broker_name", ""),
            "top_net_buyer_units":  top_buyer.get("net_units", ""),
            "top_net_seller_broker":top_seller.get("broker_name", ""),
            "top_net_seller_units": top_seller.get("net_units", ""),
            "top_volume_broker":    top_vol.get("broker_name", ""),
            "top_volume_units":     top_vol.get(
                "buy_units", 0) + top_vol.get("sell_units", 0),
        })

    # Write combined summary
    write_combined_csv(all_summaries, out_dir)

    log.info("\nDone. Output in: %s", out_dir)
    log.info("Files written:")
    log.info("  _all_symbols_summary.csv — one row per qualifying symbol")
    log.info("  SYMBOL_broker_summary.csv — per-broker detail for each symbol")


if __name__ == "__main__":
    main()
