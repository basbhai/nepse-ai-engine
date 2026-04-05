"""
history_bootstrap.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine – Historical Price Updater (Direct Web Scrape)
─────────────────────────────────────────────────────────────────────────────

What this script does:
  • TODAY MODE  – Scrapes today's share price from ShareSansar (lxml, same
                  as before) and upserts directly into Neon.
  • HISTORY MODE – Fetches any past date range via ShareSansar AJAX endpoint
                   (POST /ajaxtodayshareprice) and bulk-inserts into Neon.
                   Used for the 2019-2026 data collection needed by the
                   dividend-pattern backtester and backtester.py.
  • Provides utilities to read historical data for indicators (HistoryCache).
  • Shows status of loaded data in the price_history table.

─────────────────────────────────────────────────────────────────────────────
HOW TO USE:

  # Daily update (after market close – existing behaviour)
  python history_bootstrap.py --scrape

  # Dry-run – parse but do not insert
  python history_bootstrap.py --scrape --dry-run

  # Historical backfill for a date range
  python history_bootstrap.py --history --from 2019-01-01 --to 2019-12-31

  # Historical backfill with dry run
  python history_bootstrap.py --history --from 2019-01-01 --to 2019-01-31 --dry-run

  # Skip dates already in DB (default) – add --force to re-fetch anyway
  python history_bootstrap.py --history --from 2020-01-01 --to 2020-12-31 --force

  # Check what data is already in Neon
  python history_bootstrap.py --status
─────────────────────────────────────────────────────────────────────────────
AJAX ENDPOINT NOTES:
  URL  : POST https://www.sharesansar.com/ajaxtodayshareprice
  Form : _token=<csrf>, sector=all_sec, date=YYYY-MM-DD
  Auth : Requires session cookie + CSRF token from GET of today-share-price
  HTML columns in response (from observed response):
    SN, Symbol, LTP, Change%, Day High, Day Low, Prev. Close, Volume,
    Turnover, Trans., Open, Week High, Week Low, VWAP, Avg (120), Avg (180),
    % Change (Week), % Change (Month), % Change (Year)
  Note: column names differ slightly from the live table – we map both.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
import lxml.html as html

from sheets import run_raw_sql
from config import NST

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

BASE_URL         = "https://www.sharesansar.com"
TODAY_URL        = f"{BASE_URL}/today-share-price"
AJAX_URL         = f"{BASE_URL}/ajaxtodayshareprice"
REQUEST_DELAY_S  = 2      # polite delay between AJAX calls (seconds)
MAX_RETRIES      = 3      # retry on network error


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NUMBER PARSING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(val) -> Optional[float]:
    """Convert "1,000.00" or "-" to float or None."""
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
# SECTION 2 — DATABASE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_table() -> bool:
    """Ensure price_history table exists."""
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


def _get_loaded_dates(from_str: str, to_str: str) -> set:
    """Return set of dates already in DB within the given range."""
    try:
        rows = run_raw_sql(
            "SELECT DISTINCT date FROM price_history WHERE date >= %s AND date <= %s",
            (from_str, to_str)
        )
        return {r["date"] for r in rows}
    except Exception:
        return set()


def _bulk_insert(rows: list[dict]) -> int:
    """
    Bulk insert rows into price_history.
    Uses ON CONFLICT DO NOTHING – safe to re-run.
    Returns number of rows passed (rowcount unreliable with DO NOTHING).
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
        return len(rows)

    except Exception as exc:
        log.error("Bulk insert failed: %s", exc)
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROW MAPPING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ShareSansar uses slightly different column names between the live table
# (lxml scrape) and the AJAX response (BeautifulSoup scrape).
# This unified mapper handles both.

_FIELD_ALIASES = {
    # Live table header → DB column
    "Symbol":        "symbol",
    "LTP":           "ltp",
    "Open":          "open",
    "High":          "high",
    "Day High":      "high",
    "Low":           "low",
    "Day Low":       "low",
    "Close":         "close",
    "Vol":           "volume",
    "Volume":        "volume",
    "Turnover":      "turnover",
    "VWAP":          "vwap",
    "Prev. Close":   "prev_close",
    "Trans.":        "transactions",
    "Conf.":         "conf_score",
    "120 Days":      "avg_120d",
    "Avg (120)":     "avg_120d",
    "180 Days":      "avg_180d",
    "Avg (180)":     "avg_180d",
    "52 Weeks High": "week52_high",
    "Week High":     "week52_high",
    "52 Weeks Low":  "week52_low",
    "Week Low":      "week52_low",
}


def _map_row(raw: dict, date_str: str, source: str) -> Optional[dict]:
    """
    Convert a raw header→value dict (from either scrape path) into
    a DB-ready dict. Returns None if no valid symbol found.
    """
    # Normalise keys via alias map
    normalised = {}
    for k, v in raw.items():
        db_col = _FIELD_ALIASES.get(k.strip())
        if db_col:
            normalised[db_col] = v

    symbol = normalised.get("symbol", "").strip().upper()
    if not symbol:
        return None

    return {
        "date":         date_str,
        "symbol":       symbol,
        "open":         str(_parse_number(normalised.get("open"))    or ""),
        "high":         str(_parse_number(normalised.get("high"))    or ""),
        "low":          str(_parse_number(normalised.get("low"))     or ""),
        "close":        str(_parse_number(normalised.get("close"))   or ""),
        "ltp":          str(_parse_number(normalised.get("ltp"))     or ""),
        "volume":       str(_parse_number(normalised.get("volume"))  or ""),
        "turnover":     str(_parse_number(normalised.get("turnover")) or ""),
        "vwap":         str(_parse_number(normalised.get("vwap"))    or ""),
        "prev_close":   str(_parse_number(normalised.get("prev_close")) or ""),
        "transactions": str(_parse_number(normalised.get("transactions")) or ""),
        "conf_score":   str(_parse_number(normalised.get("conf_score")) or ""),
        "avg_120d":     str(_parse_number(normalised.get("avg_120d")) or ""),
        "avg_180d":     str(_parse_number(normalised.get("avg_180d")) or ""),
        "week52_high":  str(_parse_number(normalised.get("week52_high")) or ""),
        "week52_low":   str(_parse_number(normalised.get("week52_low")) or ""),
        "source":       source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TODAY SCRAPE (existing path, uses lxml)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_and_upsert(dry_run: bool = False) -> dict:
    """
    Scrape today's share prices from ShareSansar (lxml) and upsert into Neon.
    """
    log.info("Starting direct scrape from: %s", TODAY_URL)

    try:
        response = requests.get(TODAY_URL, timeout=15)
        response.raise_for_status()
        tree = html.fromstring(response.content)

        rows = tree.xpath('//table//tr')
        if not rows:
            return {"error": "No data table found on ShareSansar"}

        header_cells = rows[0].xpath('.//th//text()')
        header_cells = [cell.strip() for cell in header_cells if cell.strip()]
        if not header_cells:
            header_cells = rows[0].xpath('.//td//text()')
            header_cells = [cell.strip() for cell in header_cells if cell.strip()]

        log.info("Detected columns: %s", header_cells)

        date_str = datetime.now(NST).strftime('%Y-%m-%d')

        if not dry_run and _date_already_loaded(date_str):
            log.info("Data for %s already exists in Neon. Skipping.", date_str)
            return {"files_loaded": 0, "rows_inserted": 0, "dates_loaded": []}

        if not dry_run:
            if not _ensure_table():
                return {"error": "Failed to create price_history table"}

        db_rows = []
        for row in rows[1:]:
            cells = row.xpath('.//td//text()')
            cells = [cell.strip() for cell in cells if cell.strip()]
            if not cells or len(cells) != len(header_cells):
                continue

            raw = dict(zip(header_cells, cells))
            mapped = _map_row(raw, date_str, "direct_web_scrape")
            if mapped:
                db_rows.append(mapped)

        log.info("Parsed %d symbols for %s", len(db_rows), date_str)

        if dry_run:
            log.info("[DRY RUN] Would insert %d rows for %s", len(db_rows), date_str)
            return {
                "files_found": 0, "files_loaded": 1,
                "rows_inserted": len(db_rows), "dates_loaded": [date_str]
            }

        inserted = _bulk_insert(db_rows)
        log.info("Upserted %d symbols into Neon for %s", inserted, date_str)

        return {
            "files_found": 0, "files_loaded": 1,
            "rows_inserted": inserted, "dates_loaded": [date_str]
        }

    except Exception as e:
        log.error("Scrape/upsert failed: %s", e)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HISTORICAL BACKFILL (new – AJAX path)
# ══════════════════════════════════════════════════════════════════════════════

def _get_csrf_token(session: requests.Session) -> Optional[str]:
    """
    GET the today-share-price page and extract the CSRF token.
    The session object is updated with the session cookie automatically.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(TODAY_URL, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            token_input = soup.find("input", {"name": "_token"})
            if token_input:
                return token_input["value"]
            log.warning("CSRF token not found in page (attempt %d)", attempt)
        except Exception as exc:
            log.warning("CSRF fetch attempt %d failed: %s", attempt, exc)
        time.sleep(REQUEST_DELAY_S)
    return None


def _fetch_ajax_for_date(
    session: requests.Session,
    token: str,
    date_str: str
) -> Optional[list[dict]]:
    """
    POST to the AJAX endpoint for a specific date.
    Returns a list of raw header→value dicts, or None on failure.
    The response is HTML containing a <table>.
    """
    payload = {
        "_token": token,
        "sector":  "all_sec",
        "date":    date_str,
    }
    headers = {
        "User-Agent":       "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer":          TODAY_URL,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(AJAX_URL, data=payload, headers=headers, timeout=20)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                # Empty response = weekend / holiday / future date
                log.debug("No table for %s (holiday or no trading)", date_str)
                return []

            # Extract headers
            header_row = table.find("tr")
            if not header_row:
                return []
            headers_list = [
                th.get_text(strip=True)
                for th in header_row.find_all(["th", "td"])
            ]

            rows_out = []
            for tr in table.find_all("tr")[1:]:
                cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                if not cells:
                    continue
                # Pad or trim to match header length
                if len(cells) < len(headers_list):
                    cells += [""] * (len(headers_list) - len(cells))
                raw = dict(zip(headers_list, cells))
                rows_out.append(raw)

            return rows_out

        except Exception as exc:
            log.warning("AJAX fetch attempt %d for %s failed: %s",
                        attempt, date_str, exc)
            time.sleep(REQUEST_DELAY_S * attempt)

    log.error("All %d retries failed for %s", MAX_RETRIES, date_str)
    return None


def _date_range(from_str: str, to_str: str) -> list[str]:
    """Generate list of YYYY-MM-DD strings between from_str and to_str inclusive."""
    try:
        start = date.fromisoformat(from_str)
        end   = date.fromisoformat(to_str)
    except ValueError as exc:
        raise ValueError(f"Invalid date format: {exc}. Use YYYY-MM-DD") from exc

    if start > end:
        raise ValueError(f"--from {from_str} must be <= --to {to_str}")

    result = []
    current = start
    while current <= end:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def backfill_history(
    from_str: str,
    to_str: str,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Fetch historical share prices via the ShareSansar AJAX endpoint for every
    calendar date in [from_str, to_str] and insert into price_history.

    Weekends and holidays return an empty table – those dates are silently
    skipped (no row inserted, no error).

    Args:
        from_str : Start date, inclusive, YYYY-MM-DD
        to_str   : End date, inclusive, YYYY-MM-DD
        dry_run  : Parse but do not write to DB
        force    : Re-fetch even if date already in DB
    """
    dates = _date_range(from_str, to_str)
    log.info("Historical backfill: %s → %s (%d calendar days)",
             from_str, to_str, len(dates))

    if not dry_run:
        if not _ensure_table():
            return {"error": "Failed to create price_history table"}

    # Determine which dates to skip
    if not force and not dry_run:
        already_loaded = _get_loaded_dates(from_str, to_str)
        skip_count = len([d for d in dates if d in already_loaded])
        if skip_count:
            log.info("Skipping %d dates already in DB (use --force to re-fetch)",
                     skip_count)
    else:
        already_loaded = set()

    session = requests.Session()
    token   = _get_csrf_token(session)
    if not token:
        return {"error": "Could not obtain CSRF token from ShareSansar"}

    log.info("CSRF token acquired. Starting date loop...")

    total_inserted   = 0
    trading_days     = 0
    skipped_existing = 0
    skipped_empty    = 0
    errors           = 0
    dates_loaded     = []

    for i, date_str in enumerate(dates, 1):
        # Skip already-loaded dates
        if not force and date_str in already_loaded:
            skipped_existing += 1
            continue

        log.info("[%d/%d] Fetching %s ...", i, len(dates), date_str)

        raw_rows = _fetch_ajax_for_date(session, token, date_str)

        # Refresh CSRF token every 50 requests to avoid expiry
        if i % 50 == 0:
            new_token = _get_csrf_token(session)
            if new_token:
                token = new_token
                log.info("CSRF token refreshed at request %d", i)
            else:
                log.warning("CSRF refresh failed at request %d — continuing with old token", i)

        if raw_rows is None:
            log.error("Failed to fetch %s after %d retries", date_str, MAX_RETRIES)
            errors += 1
            continue

        if not raw_rows:
            # Holiday / weekend / market closed
            skipped_empty += 1
            log.debug("No data for %s (holiday/weekend)", date_str)
            time.sleep(REQUEST_DELAY_S)
            continue

        # Map rows
        db_rows = []
        for raw in raw_rows:
            mapped = _map_row(raw, date_str, "sharesansar_ajax")
            if mapped:
                db_rows.append(mapped)

        if not db_rows:
            log.warning("No valid rows parsed for %s", date_str)
            skipped_empty += 1
            time.sleep(REQUEST_DELAY_S)
            continue

        trading_days += 1
        log.info("  %s: %d symbols parsed", date_str, len(db_rows))

        if dry_run:
            log.info("  [DRY RUN] Would insert %d rows", len(db_rows))
            total_inserted += len(db_rows)
            dates_loaded.append(date_str)
        else:
            inserted = _bulk_insert(db_rows)
            total_inserted += inserted
            dates_loaded.append(date_str)
            log.info("  Inserted %d rows", inserted)

        # Polite delay between requests
        time.sleep(REQUEST_DELAY_S)

    summary = {
        "from":             from_str,
        "to":               to_str,
        "calendar_days":    len(dates),
        "trading_days":     trading_days,
        "skipped_existing": skipped_existing,
        "skipped_empty":    skipped_empty,
        "errors":           errors,
        "rows_inserted":    total_inserted,
        "dates_loaded":     dates_loaded,
    }

    log.info(
        "Backfill complete: %d trading days, %d rows inserted, "
        "%d skipped (exists), %d skipped (empty/holiday), %d errors",
        trading_days, total_inserted, skipped_existing, skipped_empty, errors
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HISTORY CACHE READERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_history_for_symbol(symbol: str, periods: int = 250) -> dict:
    """Load historical OHLCV for one symbol."""
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
    """Load history for all symbols in one query."""
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
# SECTION 7 — STATUS CHECK
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

        # Source breakdown
        src_rows = run_raw_sql("""
            SELECT source, COUNT(*) as cnt
            FROM price_history
            GROUP BY source ORDER BY cnt DESC
        """)
        if src_rows:
            print(f"  Source breakdown:")
            for sr in src_rows:
                print(f"    {sr['source']:<30} {sr['cnt']:>10,} rows")
        print()

    except Exception as exc:
        print(f"\n  ❌ Cannot check status: {exc}")
        print("  Is price_history table created? Run: python -m db.migrations\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.history_bootstrap --scrape
#   python -m modules.history_bootstrap --scrape --dry-run
#   python -m modules.history_bootstrap --status
#   python -m modules.history_bootstrap --history --from 2019-01-01 --to 2019-12-31
#   python -m modules.history_bootstrap --history --from 2020-01-01 --to 2020-12-31 --dry-run
#   python -m modules.history_bootstrap --history --from 2021-01-01 --to 2021-12-31 --force
# ══════════════════════════════════════════════════════════════════════════════

def _get_arg(args: list, flag: str) -> Optional[str]:
    """Return value after a named flag, e.g. --from 2019-01-01."""
    try:
        idx = args.index(flag)
        return args[idx + 1]
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    args = sys.argv[1:]

    # ── Status check ──────────────────────────────────────────────────────────
    if "--status" in args:
        print_status()
        sys.exit(0)

    # ── Daily scrape (today) ──────────────────────────────────────────────────
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

        print(f"\n{'='*50}")
        print(f"  SCRAPE COMPLETE")
        print(f"{'='*50}")
        print(f"  Files found:    {stats.get('files_found', 0)}")
        print(f"  Files loaded:   {stats.get('files_loaded', 0)}")
        print(f"  Rows inserted:  {stats.get('rows_inserted', 0):,}")
        if stats.get("dates_loaded"):
            print(f"  Date range:     {min(stats['dates_loaded'])}  to  {max(stats['dates_loaded'])}")
        print(f"{'='*50}\n")

        if not dry_run:
            print_status()

        sys.exit(0)

    # ── Historical backfill ───────────────────────────────────────────────────
    if "--history" in args:
        from_date = _get_arg(args, "--from")
        to_date   = _get_arg(args, "--to")
        dry_run   = "--dry-run" in args
        force     = "--force" in args

        if not from_date or not to_date:
            print("\n❌ --history requires --from YYYY-MM-DD --to YYYY-MM-DD\n")
            print("  Example: python history_bootstrap.py --history --from 2019-01-01 --to 2019-12-31\n")
            sys.exit(1)

        log.info("=" * 60)
        log.info("HISTORICAL BACKFILL starting: %s → %s", from_date, to_date)
        if dry_run:
            log.info("DRY RUN mode – no data will be written")
        if force:
            log.info("FORCE mode – re-fetching dates already in DB")
        log.info("=" * 60)

        try:
            stats = backfill_history(from_date, to_date, dry_run=dry_run, force=force)
        except ValueError as exc:
            print(f"\n❌ {exc}\n")
            sys.exit(1)

        if "error" in stats:
            print(f"\n❌ Error: {stats['error']}\n")
            sys.exit(1)

        print(f"\n{'='*55}")
        print(f"  HISTORICAL BACKFILL COMPLETE")
        print(f"{'='*55}")
        print(f"  Date range:          {stats['from']}  to  {stats['to']}")
        print(f"  Calendar days:       {stats['calendar_days']}")
        print(f"  Trading days found:  {stats['trading_days']}")
        print(f"  Rows inserted:       {stats['rows_inserted']:,}")
        print(f"  Skipped (existing):  {stats['skipped_existing']}")
        print(f"  Skipped (holiday):   {stats['skipped_empty']}")
        print(f"  Errors:              {stats['errors']}")
        print(f"{'='*55}\n")

        if not dry_run:
            print_status()

        sys.exit(0)

    # ── No valid command ──────────────────────────────────────────────────────
    print("\nUsage:")
    print("  python -m modules.history_bootstrap --scrape")
    print("  python -m modules.history_bootstrap --scrape --dry-run")
    print("  python -m modules.history_bootstrap --status")
    print("  python -m modules.history_bootstrap --history --from 2019-01-01 --to 2019-12-31")
    print("  python -m modules.history_bootstrap --history --from 2019-01-01 --to 2019-12-31 --dry-run")
    print("  python -m modules.history_bootstrap --history --from 2019-01-01 --to 2019-12-31 --force")
    print()
    sys.exit(1)
