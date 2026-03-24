"""
history_bootstrap.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine – Historical Price Updater (Direct Web Scrape)
─────────────────────────────────────────────────────────────────────────────

What this script does NOW:
  • Scrapes today's share price from ShareSansar and upserts directly into Neon.
  • Provides utilities to read historical data for indicators (HistoryCache).
  • Shows status of loaded data in the price_history table.

CSV file loading has been REMOVED. All data comes from live web scraping.
─────────────────────────────────────────────────────────────────────────────

HOW TO USE:
  # Daily update (after market close)
  python history_bootstrap.py --scrape

  # Check what data is already in Neon
  python history_bootstrap.py --status

  # Dry run – parse but do not insert
  python history_bootstrap.py --scrape --dry-run
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import lxml.html as html
import pandas as pd  # still used? Only if we keep the DataFrame approach – we'll avoid it.

from sheets import run_raw_sql

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOOTSTRAP] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST = timezone(timedelta(hours=5, minutes=45))

# ──── CSV‑related constants (FILENAME_PATTERN, COLUMN_MAP) have been removed ────


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NUMBER PARSING (kept for web data)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(val) -> Optional[float]:
    """
    Convert a string like "1,000.00" or "-" to float or None.
    Used for cleaning scraped text.
    """
    if val is None:
        return None
    s = str(val).strip().replace(",", "")
    if s in ("", "-", "N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATABASE OPERATIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_table() -> bool:
    """Ensure price_history table exists (same as before)."""
    try:
        run_raw_sql("""
            CREATE TABLE IF NOT EXISTS price_history (
                id           SERIAL PRIMARY KEY,
                date         TEXT NOT NULL,
                symbol       TEXT NOT NULL,
                open         TEXT,
                high         TEXT,
                low          TEXT,
                close        TEXT,
                ltp          TEXT,
                volume       TEXT,
                turnover     TEXT,
                vwap         TEXT,
                prev_close   TEXT,
                transactions TEXT,
                conf_score   TEXT,
                avg_120d     TEXT,
                avg_180d     TEXT,
                week52_high  TEXT,
                week52_low   TEXT,
                source       TEXT DEFAULT 'omitnomis_csv',
                inserted_at  TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_price_history_date_symbol
                ON price_history (date, symbol);
            CREATE INDEX IF NOT EXISTS ix_price_history_symbol
                ON price_history (symbol);
            CREATE INDEX IF NOT EXISTS ix_price_history_date
                ON price_history (date);
        """)
        log.info("price_history table ready")
        return True
    except Exception as exc:
        log.error("Failed to create price_history table: %s", exc)
        return False


def _date_already_loaded(date_str: str) -> bool:
    """Check if any rows exist for the given date."""
    try:
        rows = run_raw_sql(
            "SELECT COUNT(*) as cnt FROM price_history WHERE date = %s",
            (date_str,)
        )
        count = int(rows[0]["cnt"]) if rows else 0
        return count > 0
    except Exception:
        return False


def _bulk_insert(rows: list[dict]) -> int:
    """
    Bulk insert rows into price_history.
    Uses ON CONFLICT DO NOTHING – safe to run daily.
    Returns number of rows inserted.
    """
    if not rows:
        return 0

    try:
        from db.connection import _db
        import psycopg2.extras

        columns = [
            "date", "symbol", "open", "high", "low", "close", "ltp",
            "volume", "turnover", "vwap", "prev_close", "transactions",
            "conf_score", "avg_120d", "avg_180d", "week52_high", "week52_low",
            "source",
        ]

        col_sql = ", ".join(columns)
        val_sql = ", ".join(["%s"] * len(columns))

        values = [
            tuple(str(row.get(c, "")) for c in columns)
            for row in rows
        ]

        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO price_history ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT (date, symbol) DO NOTHING
                """,
                values,
                page_size=200,
            )
            # rowcount unreliable with DO NOTHING, so we return len(rows)
        return len(rows)

    except Exception as exc:
        log.error("Bulk insert failed: %s", exc)
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DIRECT WEB SCRAPE (replaces CSV bootstrap)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_and_upsert(dry_run: bool = False) -> dict:
    """
    Scrape today's share prices from ShareSansar and upsert into Neon.
    Returns a summary dict with counts.
    """
    url = 'https://www.sharesansar.com/today-share-price'
    log.info("Starting direct scrape from: %s", url)

    try:
        # 1. Fetch the webpage
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        tree = html.fromstring(response.content)

        # 2. Locate the main table – use the same XPath as your snippet
        rows = tree.xpath('//table//tr')
        if not rows:
            return {"error": "No data table found on ShareSansar"}

        # 3. Extract headers (first row)
        header_cells = rows[0].xpath('.//th//text()')
        header_cells = [cell.strip() for cell in header_cells if cell.strip()]
        if not header_cells:
            # fallback: try <td> in header row (unlikely)
            header_cells = rows[0].xpath('.//td//text()')
            header_cells = [cell.strip() for cell in header_cells if cell.strip()]

        log.info(f"Detected columns: {header_cells}")

        # 4. Determine target date (today in Nepal time)
        date_str = datetime.now(NST).strftime('%Y-%m-%d')

        # 5. Optional: skip if already loaded
        if not dry_run and _date_already_loaded(date_str):
            log.info("Data for %s already exists in Neon. Skipping.", date_str)
            return {"files_loaded": 0, "rows_inserted": 0}

        # 6. Ensure table exists (if not dry run)
        if not dry_run:
            if not _ensure_table():
                return {"error": "Failed to create price_history table"}

        # 7. Parse data rows – mimic your snippet but build dicts directly
        db_rows = []
        for row in rows[1:]:
            cells = row.xpath('.//td//text()')
            cells = [cell.strip() for cell in cells if cell.strip()]
            if not cells:
                continue

            # If number of cells doesn't match headers, skip (malformed row)
            if len(cells) != len(header_cells):
                log.debug(f"Skipping row with {len(cells)} cells, expected {len(header_cells)}")
                continue

            # Build dict using headers as keys
            row_dict = dict(zip(header_cells, cells))
            symbol = row_dict.get("Symbol", "").strip().upper()
            if not symbol:
                continue

            # Map scraped fields to database columns
            # (adjust these keys based on actual ShareSansar column names)
            db_rows.append({
                "date":         date_str,
                "symbol":       symbol,
                "open":         str(_parse_number(row_dict.get("Open")) or ""),
                "high":         str(_parse_number(row_dict.get("High")) or ""),
                "low":          str(_parse_number(row_dict.get("Low")) or ""),
                "close":        str(_parse_number(row_dict.get("Close")) or ""),
                "ltp":          str(_parse_number(row_dict.get("LTP")) or ""),
                "volume":       str(_parse_number(row_dict.get("Vol")) or ""),
                "turnover":     str(_parse_number(row_dict.get("Turnover")) or ""),
                "vwap":         str(_parse_number(row_dict.get("VWAP")) or ""),
                "prev_close":   str(_parse_number(row_dict.get("Prev. Close")) or ""),
                "transactions": str(_parse_number(row_dict.get("Trans.")) or ""),
                "conf_score":   str(_parse_number(row_dict.get("Conf.")) or ""),
                "avg_120d":     str(_parse_number(row_dict.get("120 Days")) or ""),
                "avg_180d":     str(_parse_number(row_dict.get("180 Days")) or ""),
                "week52_high":  str(_parse_number(row_dict.get("52 Weeks High")) or ""),
                "week52_low":   str(_parse_number(row_dict.get("52 Weeks Low")) or ""),
                "source":       "direct_web_scrape",
            })

        log.info(f"Parsed {len(db_rows)} symbols for {date_str}")

        if dry_run:
            log.info(f"[DRY RUN] Would insert {len(db_rows)} rows for {date_str}")
            return {
                "files_found": 0,
                "files_loaded": 1,
                "rows_inserted": len(db_rows),
                "dates_loaded": [date_str]
            }

        # 8. Upsert to Neon
        inserted = _bulk_insert(db_rows)
        log.info(f"Upserted {inserted} symbols into Neon for {date_str}")

        return {
            "files_found": 0,
            "files_loaded": 1,
            "rows_inserted": inserted,
            "dates_loaded": [date_str]
        }

    except Exception as e:
        log.error("Scrape/upsert failed: %s", e)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HISTORY CACHE READERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_history_for_symbol(symbol: str, periods: int = 250) -> dict:
    """Load historical OHLCV for one symbol (same as before)."""
    try:
        rows = run_raw_sql(
            """
            SELECT date, open, high, low, close, ltp, volume
            FROM price_history
            WHERE symbol = %s
            ORDER BY date ASC
            LIMIT %s
            """,
            (symbol.upper(), periods)
        )

        if not rows:
            return _empty_history()

        def _f(val) -> float:
            try:
                return float(val) if val else 0.0
            except (ValueError, TypeError):
                return 0.0

        return {
            "dates":   [r["date"]   for r in rows],
            "opens":   [_f(r["open"])   for r in rows],
            "highs":   [_f(r["high"])   for r in rows],
            "lows":    [_f(r["low"])    for r in rows],
            "closes":  [_f(r["close"] or r["ltp"]) for r in rows],
            "volumes": [_f(r["volume"]) for r in rows],
        }

    except Exception as exc:
        log.warning("load_history_for_symbol(%s) failed: %s", symbol, exc)
        return _empty_history()


def load_history_all_symbols(periods: int = 250) -> dict[str, dict]:
    """Load history for all symbols in one query (same as before)."""
    log.info("Loading price history for all symbols (last %d days)...", periods)

    try:
        date_rows = run_raw_sql(
            """
            SELECT DISTINCT date FROM price_history
            ORDER BY date DESC
            LIMIT %s
            """,
            (periods,)
        )

        if not date_rows:
            log.warning("No price history found in Neon — run scrape first")
            return {}

        dates = sorted([r["date"] for r in date_rows])
        log.info("History covers %d trading days: %s to %s",
                 len(dates), dates[0], dates[-1])

        date_list = tuple(dates)
        rows = run_raw_sql(
            """
            SELECT date, symbol, open, high, low, close, ltp, volume
            FROM price_history
            WHERE date = ANY(%s)
            ORDER BY symbol, date ASC
            """,
            (list(date_list),)
        )

        if not rows:
            return {}

        from collections import defaultdict
        symbol_data: dict[str, list] = defaultdict(list)
        for row in rows:
            symbol_data[row["symbol"]].append(row)

        def _f(val) -> float:
            try:
                return float(val) if val else 0.0
            except (ValueError, TypeError):
                return 0.0

        result = {}
        for symbol, sym_rows in symbol_data.items():
            result[symbol] = {
                "dates":   [r["date"]   for r in sym_rows],
                "opens":   [_f(r["open"])   for r in sym_rows],
                "highs":   [_f(r["high"])   for r in sym_rows],
                "lows":    [_f(r["low"])    for r in sym_rows],
                "closes":  [_f(r["close"] or r["ltp"]) for r in sym_rows],
                "volumes": [_f(r["volume"]) for r in sym_rows],
            }

        log.info("Loaded history for %d symbols", len(result))
        return result

    except Exception as exc:
        log.error("load_history_all_symbols failed: %s", exc)
        return {}


def _empty_history() -> dict:
    return {
        "dates":   [],
        "opens":   [],
        "highs":   [],
        "lows":    [],
        "closes":  [],
        "volumes": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STATUS CHECK (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def print_status() -> None:
    """Print current status of price_history table in Neon."""
    try:
        rows = run_raw_sql("""
            SELECT
                COUNT(DISTINCT date)   AS trading_days,
                COUNT(DISTINCT symbol) AS symbols,
                COUNT(*)               AS total_rows,
                MIN(date)              AS earliest,
                MAX(date)              AS latest
            FROM price_history
        """)

        if not rows or not rows[0]["trading_days"]:
            print("\n  price_history table is EMPTY")
            print("  Run: python history_bootstrap.py --scrape\n")
            return

        r = rows[0]
        print(f"\n{'='*50}")
        print(f"  PRICE HISTORY STATUS")
        print(f"{'='*50}")
        print(f"  Trading days loaded:  {r['trading_days']}")
        print(f"  Symbols covered:      {r['symbols']}")
        print(f"  Total rows:           {r['total_rows']:,}")
        print(f"  Date range:           {r['earliest']}  to  {r['latest']}")
        print(f"{'='*50}")

        days = int(r["trading_days"])
        print(f"\n  Indicator readiness:")
        print(f"  RSI 14:      {'✅ Ready' if days >= 14  else f'❌ Need {14-days} more days'}")
        print(f"  EMA 20:      {'✅ Ready' if days >= 20  else f'❌ Need {20-days} more days'}")
        print(f"  MACD:        {'✅ Ready' if days >= 26  else f'❌ Need {26-days} more days'}")
        print(f"  EMA 50:      {'✅ Ready' if days >= 50  else f'❌ Need {50-days} more days'}")
        print(f"  EMA 200:     {'✅ Ready' if days >= 200 else f'⚠️  Need {200-days} more days'}")
        print(f"  Candles:     {'✅ Ready' if days >= 5   else f'❌ Need {5-days} more days'}")
        print()

    except Exception as exc:
        print(f"\n  ❌ Cannot check status: {exc}")
        print("  Is price_history table created? Run: python -m db.migrations\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.history_bootstrap --scrape
#   python -m modules.history_bootstrap --scrape --dry-run
#   python -m modules.history_bootstrap --status
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    # ── Status check ──────────────────────────────────────────────────────────
    if "--status" in args:
        print_status()
        sys.exit(0)

    # ── Scrape mode ──────────────────────────────────────────────────────────
    if "--scrape" in args:
        dry_run = "--dry-run" in args
        log.info("=" * 60)
        log.info("DIRECT SCRAPE TO NEON starting")
        if dry_run:
            log.info("DRY RUN mode – no data will be written")
        log.info("=" * 60)

        stats = scrape_and_upsert(dry_run=dry_run)

        if "error" in stats:
            print(f"\n❌ Error: {stats['error']}\n")
            sys.exit(1)

        # Print summary
        print(f"\n{'='*50}")
        print(f"  SCRAPE COMPLETE")
        print(f"{'='*50}")
        print(f"  Files found:    {stats['files_found']}")
        print(f"  Files loaded:   {stats['files_loaded']}")
        print(f"  Rows inserted:  {stats['rows_inserted']:,}")
        if stats["dates_loaded"]:
            print(f"  Date range:     {min(stats['dates_loaded'])}  to  {max(stats['dates_loaded'])}")
        print(f"{'='*50}\n")

        if not dry_run:
            print_status()

        sys.exit(0)

    # ── No valid command ─────────────────────────────────────────────────────
    print("\nUsage:")
    print("  python history_bootstrap.py --scrape          # Scrape today's data and upsert")
    print("  python history_bootstrap.py --scrape --dry-run # Parse only, no DB write")
    print("  python history_bootstrap.py --status           # Show loaded data summary\n")
    sys.exit(1)