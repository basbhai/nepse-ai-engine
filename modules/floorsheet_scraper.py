"""
modules/floorsheet_scraper.py — NEPSE Floorsheet Data Collection
Sources:
  - Merolagani  : 2019-present, HTML POST, 500 rows/page, ~16 pages/day
  - Sharehubnepal: 2023 July-present, JSON API, max 100 rows/page

Rate limiting:
  - Merolagani  : 2s delay between pages — never parallel
  - Sharehubnepal: 1s delay between pages
  - One day at a time — never bulk parallel scraping

Architecture rules:
  - from sheets import write_row, run_raw_sql
  - Vectorized signal computation in floorsheet_signals.py
  - Polite scraping — never hammer either site
  - Periodic save + progress reporting
  - Resumes from last scraped date automatically
"""

import os
import re
import time
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sheets import write_row, run_raw_sql
from config import NST

load_dotenv()

log = logging.getLogger("floorsheet_scraper")

# ─── Constants ────────────────────────────────────────────────────────────────

ML_BASE        = "https://www.merolagani.com/Floorsheet.aspx"
SH_BASE        = "https://sharehubnepal.com/live/api/v2/floorsheet"
ML_PAGE_SIZE   = 500
SH_PAGE_SIZE   = 100
ML_DELAY_SEC   = 2.0   # delay between Merolagani pages
SH_DELAY_SEC   = 1.0   # delay between Sharehubnepal pages
SH_START_DATE  = date(2023, 7, 2)  # Sharehubnepal data starts here

ML_HEADERS = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"     : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer"    : ML_BASE,
}

SH_HEADERS = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"     : "application/json",
}


# ─── DB Helpers ───────────────────────────────────────────────────────────────

def _get_last_scraped_date(source: str) -> date:
    """Get the last date successfully scraped for a given source."""
    try:
        rows = run_raw_sql("""
            SELECT MAX(date) as last_date
            FROM floorsheet
            WHERE source = %s
        """, [source])
        if rows and rows[0]["last_date"]:
            return datetime.strptime(str(rows[0]["last_date"]), "%Y-%m-%d").date()
    except Exception:
        pass
    return None


def _date_already_scraped(target_date: date, source: str) -> bool:
    """Check if a date already has data for this source."""
    try:
        rows = run_raw_sql("""
            SELECT COUNT(*) as cnt
            FROM floorsheet
            WHERE date = %s AND source = %s
        """, [target_date.strftime("%Y-%m-%d"), source])
        return rows and int(rows[0]["cnt"]) > 0
    except Exception:
        return False


def _write_rows(records: list[dict]) -> int:
    """Bulk write floorsheet rows. Returns count written."""
    written = 0
    for rec in records:
        try:
            write_row("floorsheet", rec)
            written += 1
        except Exception as e:
            log.debug(f"floorsheet write skip: {e}")
    return written


# ─── Merolagani Scraper ───────────────────────────────────────────────────────

class MerolaganiScraper:
    """
    Scrapes floorsheet from Merolagani via ASP.NET WebForms POST.
    Gets VIEWSTATE token first, then paginates through all pages.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(ML_HEADERS)
        self._viewstate          = None
        self._viewstate_gen      = None
        self._event_validation   = None

    def _get_tokens(self) -> bool:
        """Fetch initial page to get VIEWSTATE tokens."""
        try:
            r = self.session.get(ML_BASE, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")

            vs   = soup.find("input", {"id": "__VIEWSTATE"})
            vsg  = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})
            ev   = soup.find("input", {"id": "__EVENTVALIDATION"})

            if not vs:
                log.error("Merolagani: VIEWSTATE not found")
                return False

            self._viewstate        = vs.get("value", "")
            self._viewstate_gen    = vsg.get("value", "") if vsg else ""
            self._event_validation = ev.get("value", "")  if ev  else ""
            return True

        except Exception as e:
            log.error(f"Merolagani get_tokens: {e}")
            return False

    def _parse_total_pages(self, soup: BeautifulSoup) -> int:
        """Extract total page count from 'Showing X - Y of Z records' text."""
        try:
            text = soup.get_text()
            match = re.search(r"Total pages:\s*(\d+)", text)
            if match:
                return int(match.group(1))
            match = re.search(r"of\s+([\d,]+)\s+records", text)
            if match:
                total = int(match.group(1).replace(",", ""))
                return max(1, (total + ML_PAGE_SIZE - 1) // ML_PAGE_SIZE)
        except Exception:
            pass
        return 1

    def _parse_table(self, soup: BeautifulSoup, target_date: date) -> list[dict]:
        """Parse floorsheet HTML table into list of dicts."""
        records = []
        table = soup.find("table", {"id": lambda x: x and "gvData" in str(x)})
        if not table:
            # Try generic table find
            tables = soup.find_all("table")
            table  = tables[-1] if tables else None
        if not table:
            return records

        rows = table.find_all("tr")
        date_str = target_date.strftime("%Y-%m-%d")

        for row in rows[1:]:  # skip header
            cols = row.find_all("td")
            if len(cols) < 7:
                continue
            try:
                records.append({
                    "date"            : date_str,
                    "symbol"          : cols[1].get_text(strip=True),
                    "contract_id"     : cols[0].get_text(strip=True),
                    "buyer_broker_id" : cols[2].get_text(strip=True),
                    "seller_broker_id": cols[3].get_text(strip=True),
                    "buyer_broker"    : "",
                    "seller_broker"   : "",
                    "quantity"        : cols[4].get_text(strip=True).replace(",", ""),
                    "rate"            : cols[5].get_text(strip=True).replace(",", ""),
                    "amount"          : cols[6].get_text(strip=True).replace(",", ""),
                    "trade_time"      : None,
                    "source"          : "merolagani",
                })
            except Exception:
                continue

        return records

    def scrape_date(self, target_date: date) -> int:
        """
        Scrape all pages for a single date from Merolagani.
        Returns total rows written.
        """
        date_str = target_date.strftime("%m/%d/%Y")
        log.info(f"Merolagani scraping: {target_date}")

        if not self._get_tokens():
            return 0

        total_written = 0
        page          = 0
        total_pages   = 1  # will update after first page

        while page < total_pages:
            try:
                payload = {
                    "__EVENTTARGET"    : "",
                    "__EVENTARGUMENT"  : "",
                    "__VIEWSTATE"      : self._viewstate,
                    "__VIEWSTATEGENERATOR": self._viewstate_gen,
                    "__EVENTVALIDATION": self._event_validation,
                    "ctl00$ContentPlaceHolder1$ASCompanyFilter$hdnAutoSuggest": "0",
                    "ctl00$ContentPlaceHolder1$ASCompanyFilter$txtAutoSuggest": "",
                    "ctl00$ContentPlaceHolder1$txtBuyerBrokerCodeFilter"      : "",
                    "ctl00$ContentPlaceHolder1$txtSellerBrokerCodeFilter"     : "",
                    "ctl00$ContentPlaceHolder1$txtFloorsheetDateFilter"       : date_str,
                    "ctl00$ContentPlaceHolder1$PagerControl1$hdnPCID"         : "PC1",
                    "ctl00$ContentPlaceHolder1$PagerControl1$hdnCurrentPage"  : "0",
                    "ctl00$ContentPlaceHolder1$PagerControl2$hdnPCID"         : "PC2",
                    "ctl00$ContentPlaceHolder1$PagerControl2$hdnCurrentPage"  : str(page),
                    "ctl00$ContentPlaceHolder1$PagerControl2$btnPaging"       : "",
                }

                r    = self.session.post(ML_BASE, data=payload, timeout=30)
                soup = BeautifulSoup(r.text, "html.parser")

                # Update tokens for next page
                vs  = soup.find("input", {"id": "__VIEWSTATE"})
                ev  = soup.find("input", {"id": "__EVENTVALIDATION"})
                if vs:
                    self._viewstate        = vs.get("value", "")
                if ev:
                    self._event_validation = ev.get("value", "")

                # Get total pages on first fetch
                if page == 0:
                    total_pages = self._parse_total_pages(soup)
                    log.info(f"Merolagani {target_date}: {total_pages} pages")

                records = self._parse_table(soup, target_date)
                written = _write_rows(records)
                total_written += written

                log.info(f"  Page {page+1}/{total_pages} → {written} rows | total: {total_written}")
                page += 1

                # Polite delay
                time.sleep(ML_DELAY_SEC)

            except Exception as e:
                log.error(f"Merolagani page {page} error: {e}")
                time.sleep(5)
                break

        log.info(f"Merolagani {target_date} complete: {total_written} rows")
        return total_written


# ─── Sharehubnepal Scraper ────────────────────────────────────────────────────

class SharehubScraper:
    """
    Scrapes floorsheet from Sharehubnepal JSON API.
    Cleaner than Merolagani — includes trade timestamps.
    Available from 2023-07-02.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(SH_HEADERS)

    def scrape_date(self, target_date: date) -> int:
        """
        Scrape all pages for a single date from Sharehubnepal.
        Returns total rows written.
        """
        date_str = f"{target_date.year}-{target_date.month}-{target_date.day}"
        log.info(f"Sharehub scraping: {target_date}")

        total_written = 0
        page          = 1
        has_next      = True

        while has_next:
            try:
                r = self.session.get(
                    SH_BASE,
                    params={
                        "Size" : SH_PAGE_SIZE,
                        "date" : date_str,
                        "Page" : page,
                    },
                    timeout=30,
                )

                if r.status_code != 200:
                    log.error(f"Sharehub HTTP {r.status_code} on page {page}")
                    break

                data    = r.json()
                payload = data.get("data", {})
                trades  = payload.get("floorSheets", []) or payload.get("floorsheets", [])

                if not trades:
                    # Try alternate key
                    for key in payload:
                        if isinstance(payload[key], list) and len(payload[key]) > 0:
                            trades = payload[key]
                            break

                if not trades:
                    log.warning(f"Sharehub {target_date} page {page}: no trades found")
                    break

                records = []
                for t in trades:
                    try:
                        trade_time = None
                        raw_time   = t.get("tradeTime", "")
                        if raw_time:
                            # Format: 2026-04-08T14:59:59.958014Z
                            trade_time = raw_time[11:19]  # extract HH:MM:SS

                        records.append({
                            "date"            : target_date.strftime("%Y-%m-%d"),
                            "symbol"          : t.get("symbol", ""),
                            "contract_id"     : str(t.get("contractId", "")),
                            "buyer_broker_id" : str(t.get("buyerMemberId", "")),
                            "seller_broker_id": str(t.get("sellerMemberId", "")),
                            "buyer_broker"    : t.get("buyerBrokerName", ""),
                            "seller_broker"   : t.get("sellerBrokerName", ""),
                            "quantity"        : str(t.get("contractQuantity", 0)),
                            "rate"            : str(t.get("contractRate", 0)),
                            "amount"          : str(t.get("contractAmount", 0)),
                            "trade_time"      : trade_time,
                            "source"          : "sharehubnepal",
                        })
                    except Exception:
                        continue

                written = _write_rows(records)
                total_written += written

                has_next = payload.get("hasNext", False)
                log.info(f"  Page {page} → {written} rows | total: {total_written} | hasNext: {has_next}")

                page += 1
                time.sleep(SH_DELAY_SEC)

            except Exception as e:
                log.error(f"Sharehub page {page} error: {e}")
                time.sleep(5)
                break

        log.info(f"Sharehub {target_date} complete: {total_written} rows")
        return total_written


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def _get_trading_days(start: date, end: date) -> list[date]:
    """Get list of Mon-Fri dates between start and end (no holiday check for backfill)."""
    days   = pd.date_range(start=start, end=end, freq="B")  # business days
    return [d.date() for d in days]


def run_backfill(
    start_date : date = date(2019, 7, 15),
    end_date   : date = None,
    source     : str  = "both",
    batch_size : int  = 50,
) -> None:
    """
    Backfill historical floorsheet data.
    Runs overnight — polite rate limiting enforced.

    Args:
        start_date : earliest date to scrape
        end_date   : latest date (defaults to yesterday)
        source     : 'merolagani' | 'sharehubnepal' | 'both'
        batch_size : save progress every N days
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    ml_scraper = MerolaganiScraper() if source in ("merolagani", "both") else None
    sh_scraper = SharehubScraper()   if source in ("sharehubnepal", "both") else None

    trading_days = _get_trading_days(start_date, end_date)
    total_days   = len(trading_days)

    log.info(f"Backfill: {start_date} → {end_date} | {total_days} trading days | source={source}")

    total_rows = 0

    for i, target_date in enumerate(trading_days):
        date_str = target_date.strftime("%Y-%m-%d")

        # Decide which scraper to use for this date
        use_sh = sh_scraper and target_date >= SH_START_DATE
        use_ml = ml_scraper and not use_sh  # prefer Sharehubnepal for 2023+

        if source == "both":
            # Use Sharehubnepal for 2023+, Merolagani for pre-2023
            use_sh = target_date >= SH_START_DATE
            use_ml = not use_sh

        # Skip already scraped dates
        src_name = "sharehubnepal" if use_sh else "merolagani"
        if _date_already_scraped(target_date, src_name):
            log.info(f"[{i+1}/{total_days}] {date_str} already scraped ({src_name}) — skip")
            continue

        # Scrape
        log.info(f"[{i+1}/{total_days}] Scraping {date_str}")
        rows = 0

        try:
            if use_sh:
                rows = sh_scraper.scrape_date(target_date)
            elif use_ml:
                rows = ml_scraper.scrape_date(target_date)
        except Exception as e:
            log.error(f"Scrape failed {date_str}: {e}")
            time.sleep(10)
            continue

        total_rows += rows

        # Progress report every batch_size days
        if (i + 1) % batch_size == 0:
            pct = (i + 1) / total_days * 100
            log.info(f"Progress: {i+1}/{total_days} days ({pct:.1f}%) | {total_rows:,} rows total")

        # Extra delay between days
        time.sleep(2)

    log.info(f"Backfill complete: {total_rows:,} total rows written")


def run_daily(target_date: date = None) -> int:
    """
    Scrape yesterday's floorsheet. Called by morning_workflow at 10:30 AM.
    Uses Sharehubnepal if date >= 2023-07-02, else Merolagani.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    # Skip weekends
    if target_date.weekday() >= 5:
        log.info(f"run_daily: {target_date} is weekend — skip")
        return 0

    src_name = "sharehubnepal" if target_date >= SH_START_DATE else "merolagani"

    if _date_already_scraped(target_date, src_name):
        log.info(f"run_daily: {target_date} already scraped — skip")
        return 0

    if target_date >= SH_START_DATE:
        scraper = SharehubScraper()
        return scraper.scrape_date(target_date)
    else:
        scraper = MerolaganiScraper()
        return scraper.scrape_date(target_date)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="NEPSE Floorsheet Scraper")
    parser.add_argument("--mode",  choices=["backfill", "daily"], default="daily")
    parser.add_argument("--start", default="2019-07-15")
    parser.add_argument("--end",   default=None)
    parser.add_argument("--source", choices=["merolagani","sharehubnepal","both"], default="both")
    args = parser.parse_args()

    if args.mode == "backfill":
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end else None
        run_backfill(start_date=start, end_date=end, source=args.source)
    else:
        rows = run_daily()
        print(f"Daily scrape: {rows} rows written")
