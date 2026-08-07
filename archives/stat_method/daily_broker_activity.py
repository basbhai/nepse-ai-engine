"""
daily_broker_activity.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Daily Broker Activity Export

Exports one CSV with:
  Symbol, Date, Broker_Code, Broker_Name,
  Buy_Volume, Buy_Turnover, Buy_Trades,
  Sell_Volume, Sell_Turnover, Sell_Trades,
  Net_Volume,
  Open, High, Low, Close

Coverage: last 1 year (rolling from today)
One row per (symbol, date, broker).

Output: stat_method/output/daily_broker_activity.csv

Usage:
    cd ~/nepse-engine
    python stat_method/daily_broker_activity.py

    # Custom date range
    python stat_method/daily_broker_activity.py --from-date 2025-01-01 --to-date 2026-05-27

    # Single symbol
    python stat_method/daily_broker_activity.py --symbol NICA
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BROKER_DAILY] %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_COLS = [
    "Symbol", "Date", "Broker_Code", "Broker_Name",
    "Buy_Volume", "Buy_Turnover", "Buy_Trades",
    "Sell_Volume", "Sell_Turnover", "Sell_Trades",
    "Net_Volume",
    "Open", "High", "Low", "Close",
]


def get_date_range(from_date: str = None, to_date: str = None):
    today    = date.today()
    one_year = today - timedelta(days=365)
    f = from_date or one_year.strftime("%Y-%m-%d")
    t = to_date   or today.strftime("%Y-%m-%d")
    return f, t


def load_price_ohlc(from_date: str, to_date: str,
                    symbol_filter: str = None) -> dict:
    """
    Load OHLC from price_history.
    Returns: (symbol, date) → {open, high, low, close}
    """
    log.info("Loading OHLC data...")
    sym_clause = "AND symbol = %s" if symbol_filter else ""
    params = [from_date, to_date]
    if symbol_filter:
        params.append(symbol_filter.upper())

    with _db() as cur:
        cur.execute(f"""
            SELECT
                symbol, date,
                open::float  AS open,
                high::float  AS high,
                low::float   AS low,
                COALESCE(NULLIF(close,''), ltp)::float AS close
            FROM price_history
            WHERE date >= %s AND date <= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
              {sym_clause}
        """, params)
        rows = cur.fetchall()

    ohlc = {}
    for r in rows:
        key = (str(r["symbol"]).upper(), r["date"])
        ohlc[key] = {
            "open":  round(float(r["open"]  or 0), 2),
            "high":  round(float(r["high"]  or 0), 2),
            "low":   round(float(r["low"]   or 0), 2),
            "close": round(float(r["close"] or 0), 2),
        }

    log.info("OHLC loaded: %d (symbol, date) pairs", len(ohlc))
    return ohlc


def stream_floorsheet(from_date: str, to_date: str,
                      symbol_filter: str = None):
    """
    Stream raw floorsheet rows in batches.
    Yields dicts: symbol, date, buyer_broker_id, buyer_broker,
                  seller_broker_id, seller_broker, qty, amount
    """
    sym_clause = "AND symbol = %s" if symbol_filter else ""
    params = [from_date, to_date]
    if symbol_filter:
        params.append(symbol_filter.upper())

    log.info("Loading floorsheet (%s → %s)...", from_date, to_date)

    with _db() as cur:
        cur.execute(f"""
            SELECT
                symbol,
                date,
                buyer_broker_id,
                buyer_broker,
                seller_broker_id,
                seller_broker,
                quantity::float AS qty,
                amount::float   AS amount
            FROM floorsheet
            WHERE date >= %s AND date <= %s
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
              {sym_clause}
            ORDER BY symbol, date, buyer_broker_id
        """, params)

        batch_size = 50000
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for r in rows:
                yield r


def aggregate_broker_day(from_date: str, to_date: str,
                          symbol_filter: str = None) -> dict:
    """
    Aggregate floorsheet into:
    (symbol, date, broker_id) → {
        broker_name, buy_vol, buy_turn, buy_trades,
        sell_vol, sell_turn, sell_trades
    }
    """
    # key: (symbol, date, broker_id)
    agg: dict = defaultdict(lambda: {
        "broker_name": "",
        "buy_vol":    0.0,
        "buy_turn":   0.0,
        "buy_trades": 0,
        "sell_vol":   0.0,
        "sell_turn":  0.0,
        "sell_trades":0,
    })

    processed = 0
    for r in stream_floorsheet(from_date, to_date, symbol_filter):
        sym  = str(r["symbol"]).upper().strip()
        dt   = r["date"]
        qty  = float(r["qty"]    or 0)
        amt  = float(r["amount"] or 0)

        # Buyer side
        bid  = str(r["buyer_broker_id"] or "").strip()
        bname= str(r["buyer_broker"]     or bid).strip()
        if bid and bid not in ("", "None", "0", "nan"):
            k = (sym, dt, bid)
            agg[k]["broker_name"]  = bname
            agg[k]["buy_vol"]    += qty
            agg[k]["buy_turn"]   += amt
            agg[k]["buy_trades"] += 1

        # Seller side
        sid  = str(r["seller_broker_id"] or "").strip()
        sname= str(r["seller_broker"]     or sid).strip()
        if sid and sid not in ("", "None", "0", "nan"):
            k = (sym, dt, sid)
            if not agg[k]["broker_name"]:
                agg[k]["broker_name"] = sname
            agg[k]["sell_vol"]    += qty
            agg[k]["sell_turn"]   += amt
            agg[k]["sell_trades"] += 1

        processed += 1
        if processed % 500000 == 0:
            log.info("  Processed %d floorsheet rows...", processed)

    log.info("Total floorsheet rows processed: %d", processed)
    log.info("Unique (symbol, date, broker) keys: %d", len(agg))
    return dict(agg)


def write_csv(agg: dict, ohlc: dict, out_path: Path) -> int:
    """Write the final CSV. Returns row count written."""
    log.info("Writing CSV to %s...", out_path)

    # Sort by symbol, date, broker_code
    keys_sorted = sorted(agg.keys(), key=lambda k: (k[0], k[1], k[2]))

    rows_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLS)

        for (symbol, dt, broker_id) in keys_sorted:
            b = agg[(symbol, dt, broker_id)]

            price = ohlc.get((symbol, dt), {})
            open_  = price.get("open",  "")
            high_  = price.get("high",  "")
            low_   = price.get("low",   "")
            close_ = price.get("close", "")

            net_vol = round(b["buy_vol"] - b["sell_vol"], 0)

            writer.writerow([
                symbol,
                dt,
                broker_id,
                b["broker_name"],
                int(b["buy_vol"]),
                round(b["buy_turn"],  2),
                b["buy_trades"],
                int(b["sell_vol"]),
                round(b["sell_turn"], 2),
                b["sell_trades"],
                int(net_vol),
                open_,
                high_,
                low_,
                close_,
            ])
            rows_written += 1

    log.info("CSV written: %d rows", rows_written)
    return rows_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default=None,
                        help="Start date YYYY-MM-DD (default: 1 year ago)")
    parser.add_argument("--to-date",   default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--symbol",    default=None,
                        help="Filter to single symbol e.g. NICA")
    args = parser.parse_args()

    from_date, to_date = get_date_range(args.from_date, args.to_date)
    log.info("Date range: %s → %s", from_date, to_date)

    # Output path
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    if args.symbol:
        fname = f"daily_broker_{args.symbol.upper()}_{from_date}_{to_date}.csv"
    else:
        fname = f"daily_broker_activity_{from_date}_{to_date}.csv"

    out_path = out_dir / fname

    # Load OHLC
    ohlc = load_price_ohlc(from_date, to_date, args.symbol)

    # Aggregate floorsheet
    agg = aggregate_broker_day(from_date, to_date, args.symbol)

    if not agg:
        log.error("No floorsheet data found for this range.")
        sys.exit(1)

    # Write CSV
    n = write_csv(agg, ohlc, out_path)

    log.info("Done. %d rows written to:", n)
    log.info("  %s", out_path)

    # Quick preview
    log.info("\nFirst 3 rows preview:")
    with open(out_path, "r") as f:
        for i, line in enumerate(f):
            if i > 3:
                break
            print(line.strip())


if __name__ == "__main__":
    main()