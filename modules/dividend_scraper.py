"""
dividend_scraper.py — modules/dividend_scraper.py
Scrapes sharesansar.com/proposed-dividend → dividend_announcements table.
Runs daily at 10:30 AM NST via morning_brief.yml.

Usage:
    python modules/dividend_scraper.py           # current + previous FY (default)
    python modules/dividend_scraper.py --all     # all years (first-time backfill)
    python modules/dividend_scraper.py --dry-run # print only, no DB write
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone, timedelta

import requests
from bs4 import BeautifulSoup

try:
    from sheets import upsert_row
except ImportError:
    from sheets import upsert_row  # noqa

log = logging.getLogger("dividend_scraper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Year map: ShareSansar internal year_id → fiscal year label
# ─────────────────────────────────────────────────────────────────────────────
YEAR_TO_FY: dict[int, str] = {
    31: "2081/2082",
    30: "2080/2081",
    29: "2079/2080",
    28: "2078/2079",
    27: "2077/2078",
    26: "2076/2077",
    24: "2075/2076",
    5:  "2074/2075",
    4:  "2073/2074",
    3:  "2072/2073",
    2:  "2071/2072",
    1:  "2070/2071",
    16: "2069/2070",
    15: "2068/2069",
    14: "2067/2068",
    13: "2066/2067",
    12: "2065/2066",
    11: "2064/2065",
    23: "2063/2064",
    17: "2062/2063",
    18: "2061/2062",
    19: "2060/2061",
    20: "2059/2060",
    21: "2058/2059",
    22: "2057/2058",
}

# Two most recent FY year_ids for daily default run
RECENT_YEAR_IDS = [31, 30]

BASE_URL = "https://www.sharesansar.com/proposed-dividend"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.sharesansar.com/proposed-dividend"
}

NST = timezone(timedelta(hours=5, minutes=45))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(html: str | None) -> str:
    """Strip HTML tags from a cell value."""
    if not html or not isinstance(html, str):
        return ""
    return BeautifulSoup(html, "html.parser").get_text(strip=True)


def _clean_date(raw: str | None) -> str:
    """Return YYYY-MM-DD or empty string."""
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.replace("[Closed]", "").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw[:10] if len(raw) >= 10 else ""


def _safe_float(val) -> str:
    """Return float string or empty."""
    try:
        if val is None or val == "" or val != val:  # NaN check
            return ""
        return str(float(val))
    except (TypeError, ValueError):
        return ""


def _dividend_type(bonus: str, cash: str) -> str:
    b = float(bonus) if bonus else 0.0
    c = float(cash) if cash else 0.0
    if b > 0 and c > 0:
        return "BOTH"
    if b > 0:
        return "BONUS"
    return "CASH"


def _build_params(year_id: int, start: int = 0, length: int = 100) -> dict:
    return {
        "draw": "1",
        "columns[0][data]": "DT_Row_Index","columns[0][name]": "","columns[0][searchable]": "false","columns[0][orderable]": "false","columns[0][search][value]": "","columns[0][search][regex]": "false",
        "columns[1][data]": "symbol","columns[1][name]": "tbl_company_list.symbol","columns[1][searchable]": "true","columns[1][orderable]": "true","columns[1][search][value]": "","columns[1][search][regex]": "false",
        "columns[2][data]": "companyname","columns[2][name]": "tbl_company_list.companyname","columns[2][searchable]": "true","columns[2][orderable]": "true","columns[2][search][value]": "","columns[2][search][regex]": "false",
        "columns[3][data]": "bonus_share","columns[3][name]": "","columns[3][searchable]": "true","columns[3][orderable]": "true","columns[3][search][value]": "","columns[3][search][regex]": "false",
        "columns[4][data]": "cash_dividend","columns[4][name]": "","columns[4][searchable]": "true","columns[4][orderable]": "true","columns[4][search][value]": "","columns[4][search][regex]": "false",
        "columns[5][data]": "total_dividend","columns[5][name]": "","columns[5][searchable]": "true","columns[5][orderable]": "true","columns[5][search][value]": "","columns[5][search][regex]": "false",
        "columns[6][data]": "announcement_date","columns[6][name]": "","columns[6][searchable]": "true","columns[6][orderable]": "true","columns[6][search][value]": "","columns[6][search][regex]": "false",
        "columns[7][data]": "bookclose_date","columns[7][name]": "","columns[7][searchable]": "true","columns[7][orderable]": "true","columns[7][search][value]": "","columns[7][search][regex]": "false",
        "columns[8][data]": "distribution_date","columns[8][name]": "","columns[8][searchable]": "true","columns[8][orderable]": "true","columns[8][search][value]": "","columns[8][search][regex]": "false",
        "columns[9][data]": "bonus_listing_date","columns[9][name]": "","columns[9][searchable]": "true","columns[9][orderable]": "true","columns[9][search][value]": "","columns[9][search][regex]": "false",
        "columns[10][data]": "year","columns[10][name]": "tbl_macro_year.year","columns[10][searchable]": "true","columns[10][orderable]": "true","columns[10][search][value]": "","columns[10][search][regex]": "false",
        "order[0][column]": "6","order[0][dir]": "desc",
        "start": str(start),"length": str(length),
        "search[value]": "","search[regex]": "false",
        "type": "YEARWISE","year": str(year_id),"sector": "0"
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_year(session: requests.Session, year_id: int) -> list[dict]:
    """Fetch ALL records for one year_id (handles pagination)."""
    fiscal_year = YEAR_TO_FY.get(year_id, "")
    records: list[dict] = []
    start = 0
    page_size = 50

    while True:
        params = _build_params(year_id, start=start, length=page_size)
        try:
            r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            
            data = r.json()
        except Exception as e:
            log.error("Fetch failed year_id=%s start=%s: %s", year_id, start, e)
            break

        page = data.get("data", [])
        records.extend(page)
        log.info("  year_id=%s FY=%s start=%s fetched=%s", year_id, fiscal_year, start, len(page))

        total = int(data.get("recordsTotal", 0))
        start += page_size
        if start >= total or not page:
            break
        time.sleep(0.5)  # polite delay

    return records


def _parse_record(raw: dict, fiscal_year: str) -> dict | None:
    """Map one raw API record → dividend_announcements row dict."""
    symbol = _extract_text(raw.get("symbol", ""))
    if not symbol:
        return None

    announcement_date = _clean_date(raw.get("announcement_date", ""))
    if not announcement_date:
        return None

    bonus  = _safe_float(raw.get("bonus_share"))
    cash   = _safe_float(raw.get("cash_dividend"))
    total  = _safe_float(raw.get("total_dividend"))

    return {
        "symbol":              symbol,
        "company":             _extract_text(raw.get("companyname", "")),
        "sector":              "",          # not provided by API — enriched later
        "announcement_date":   announcement_date,
        "fiscal_year":         fiscal_year,
        "dividend_type":       _dividend_type(bonus, cash),
        "cash_dividend_pct":   cash,
        "bonus_share_pct":     bonus,
        "total_dividend_pct":  total,
        "book_close_date":     _clean_date(raw.get("bookclose_date", "")),
        "direction":           "",          # computed by dividend_study.py
        "prev_dividend_pct":   "",          # computed by dividend_study.py
        "source":              "sharesansar",
        "scraped_at":          datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run(year_ids: list[int], dry_run: bool = False) -> int:
    """
    Scrape dividend data for given year_ids.
    Returns total rows upserted.
    """
    session = requests.Session()
    # Warm up cookie
    try:
        session.get("https://www.sharesansar.com/proposed-dividend", headers=HEADERS)
    except Exception as e:
        log.warning("Cookie warmup failed (continuing): %s", e)

    total_upserted = 0
    total_skipped  = 0

    for year_id in year_ids:
        fiscal_year = YEAR_TO_FY.get(year_id, str(year_id))
        log.info("Scraping FY %s (year_id=%s) ...", fiscal_year, year_id)

        raw_records = _fetch_year(session, year_id)
        log.info("  Raw records fetched: %s", len(raw_records))

        for raw in raw_records:
            row = _parse_record(raw, fiscal_year)
            if row is None:
                total_skipped += 1
                continue

            if dry_run:
                log.info("  [DRY-RUN] %s %s bonus=%.2f cash=%.2f total=%.2f",
                         row["symbol"], row["announcement_date"],
                         float(row["bonus_share_pct"] or 0),
                         float(row["cash_dividend_pct"] or 0),
                         float(row["total_dividend_pct"] or 0))
            else:
                try:
                    upsert_row(
                        "dividend_announcements",
                        row,
                        conflict_columns=["symbol", "announcement_date"],
                    )
                    total_upserted += 1
                except Exception as e:
                    log.error("  Upsert failed %s %s: %s",
                              row["symbol"], row["announcement_date"], e)
                    total_skipped += 1

        time.sleep(1)  # polite between years

    log.info("Done. upserted=%s skipped=%s dry_run=%s",
             total_upserted, total_skipped, dry_run)
    return total_upserted


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# python -m modules.dividend_scraper --all --dry-run   # verify first
# python -m modules.dividend_scraper --all             # then write
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape NEPSE dividend announcements")
    parser.add_argument(
        "--all", action="store_true",
        help="Scrape all historical years (backfill). Default: current + previous FY only."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print records without writing to DB."
    )
    args = parser.parse_args()

    year_ids = list(YEAR_TO_FY.keys()) if args.all else RECENT_YEAR_IDS

    log.info("dividend_scraper starting | years=%s dry_run=%s", len(year_ids), args.dry_run)
    count = run(year_ids=year_ids, dry_run=args.dry_run)
    log.info("dividend_scraper complete | rows=%s", count)
    sys.exit(0 if count >= 0 else 1)


if __name__ == "__main__":
    main()