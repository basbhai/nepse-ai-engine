"""
modules/floorsheet_scraper.py — NEPSE Floorsheet Data Collection
Sources:
  - Merolagani  : 2019-present, HTML POST, 500 rows/page, ~16 pages/day
  - Sharehubnepal: 2023 July-present, JSON API, max 100 rows/page

Rate limiting:
  - Merolagani  : 2s delay between pages — never parallel
  - Sharehubnepal: 1s delay between pages
  - One day at a time — never bulk parallel scraping
"""

import os
import re
import sys
import time
import logging
import requests
import pandas as pd
from datetime import date, datetime, timedelta
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
ML_DELAY_SEC   = 2.0
SH_DELAY_SEC   = 1.0
SH_START_DATE  = date(2023, 7, 2)

ML_HEADERS = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"     : "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer"    : ML_BASE,
}
SH_HEADERS = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept"     : "application/json",
}


# ─── Progress Bar ─────────────────────────────────────────────────────────────

def _progress(current: int, total: int, current_date: str, rows: int, status: str = ""):
    """Print inline progress bar."""
    pct      = current / total if total > 0 else 0
    filled   = int(40 * pct)
    bar      = "█" * filled + "░" * (40 - filled)
    elapsed  = f"{rows:,} rows"
    sys.stdout.write(
        f"\r[{bar}] {current}/{total} ({pct*100:.1f}%)  "
        f"Date: {current_date}  "
        f"Total: {elapsed}  "
        f"{status}          "
    )
    sys.stdout.flush()


# ─── DB Helpers ───────────────────────────────────────────────────────────────

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


def _mark_no_data(target_date: date, source: str) -> None:
    """Write a marker row for holiday/non-trading days so they are skipped next run."""
    try:
        write_row("floorsheet", {
            "date"    : target_date.strftime("%Y-%m-%d"),
            "symbol"  : "_NODATA_",
            "quantity": "0",
            "rate"    : "0",
            "amount"  : "0",
            "source"  : source,
        })
    except Exception:
        pass


def _write_rows(records: list) -> int:
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

    def __init__(self):
        self.session             = requests.Session()
        self.session.headers.update(ML_HEADERS)
        self._viewstate          = None
        self._viewstate_gen      = None
        self._event_validation   = None

    def _get_tokens(self) -> bool:
        try:
            r    = self.session.get(ML_BASE, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            vs   = soup.find("input", {"id": "__VIEWSTATE"})
            vsg  = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})
            ev   = soup.find("input", {"id": "__EVENTVALIDATION"})
            if not vs:
                log.error("Merolagani: VIEWSTATE not found")
                return False
            self._viewstate        = vs.get("value", "")
            self._viewstate_gen    = vsg.get("value", "") if vsg else ""
            self._event_validation = ev.get("value", "") if ev  else ""
            return True
        except Exception as e:
            log.error(f"Merolagani get_tokens: {e}")
            return False

    def _parse_total_pages(self, soup: BeautifulSoup) -> int:
        try:
            text  = soup.get_text()
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

    def _parse_table(self, soup: BeautifulSoup, target_date: date) -> list:
        records  = []
        table    = soup.find("table", {"id": lambda x: x and "gvData" in str(x)})
        if not table:
            tables = soup.find_all("table")
            table  = tables[-1] if tables else None
        if not table:
            return records

        date_str = target_date.strftime("%Y-%m-%d")
        for row in table.find_all("tr")[1:]:
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
        date_str = target_date.strftime("%m/%d/%Y")

        if not self._get_tokens():
            return 0

        total_written = 0
        page          = 0
        total_pages   = 1

        while page < total_pages:
            try:
                payload = {
                    "__EVENTTARGET"   : "",
                    "__EVENTARGUMENT" : "",
                    "__VIEWSTATE"     : self._viewstate,
                    "__VIEWSTATEGENERATOR"                                         : self._viewstate_gen,
                    "__EVENTVALIDATION"                                            : self._event_validation,
                    "ctl00$ContentPlaceHolder1$ASCompanyFilter$hdnAutoSuggest"     : "0",
                    "ctl00$ContentPlaceHolder1$ASCompanyFilter$txtAutoSuggest"     : "",
                    "ctl00$ContentPlaceHolder1$txtBuyerBrokerCodeFilter"           : "",
                    "ctl00$ContentPlaceHolder1$txtSellerBrokerCodeFilter"          : "",
                    "ctl00$ContentPlaceHolder1$txtFloorsheetDateFilter"            : date_str,
                    "ctl00$ContentPlaceHolder1$PagerControl1$hdnPCID"             : "PC1",
                    "ctl00$ContentPlaceHolder1$PagerControl1$hdnCurrentPage"      : "0",
                    "ctl00$ContentPlaceHolder1$PagerControl2$hdnPCID"             : "PC2",
                    "ctl00$ContentPlaceHolder1$PagerControl2$hdnCurrentPage"      : str(page),
                    "ctl00$ContentPlaceHolder1$PagerControl2$btnPaging"           : "",
                }

                r    = self.session.post(ML_BASE, data=payload, timeout=30)
                soup = BeautifulSoup(r.text, "html.parser")

                vs = soup.find("input", {"id": "__VIEWSTATE"})
                ev = soup.find("input", {"id": "__EVENTVALIDATION"})
                if vs:
                    self._viewstate        = vs.get("value", "")
                if ev:
                    self._event_validation = ev.get("value", "")

                if page == 0:
                    total_pages = self._parse_total_pages(soup)

                records = self._parse_table(soup, target_date)
                if not records and page == 0:
                    _mark_no_data(target_date, "merolagani")
                    return 0

                written        = _write_rows(records)
                total_written += written
                page          += 1
                time.sleep(ML_DELAY_SEC)

            except Exception as e:
                log.error(f"Merolagani page {page} error: {e}")
                time.sleep(5)
                break

        return total_written


# ─── Sharehubnepal Scraper ────────────────────────────────────────────────────

class SharehubScraper:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(SH_HEADERS)

    def scrape_date(self, target_date: date) -> int:
        date_str      = f"{target_date.year}-{target_date.month}-{target_date.day}"
        total_written = 0
        page          = 1
        has_next      = True

        while has_next:
            try:
                r = self.session.get(
                    SH_BASE,
                    params={"Size": SH_PAGE_SIZE, "date": date_str, "Page": page},
                    timeout=30,
                )

                # 500 = holiday or non-trading day — mark and skip
                if r.status_code == 500:
                    log.debug(f"Sharehub {target_date}: holiday/non-trading day")
                    if page == 1:
                        _mark_no_data(target_date, "sharehubnepal")
                    break

                if r.status_code != 200:
                    log.error(f"Sharehub HTTP {r.status_code} on {target_date} page {page}")
                    break

                data    = r.json()
                payload = data.get("data", {})

                # Find trade list — try multiple key names
                trades = (payload.get("floorSheets")
                       or payload.get("floorsheets")
                       or payload.get("trades")
                       or [])

                if not trades and isinstance(payload, dict):
                    for key, val in payload.items():
                        if isinstance(val, list) and len(val) > 0:
                            trades = val
                            break

                if not trades:
                    if page == 1:
                        _mark_no_data(target_date, "sharehubnepal")
                    break

                records = []
                for t in trades:
                    try:
                        raw_time   = t.get("tradeTime", "")
                        trade_time = raw_time[11:19] if raw_time else None
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

                written        = _write_rows(records)
                total_written += written
                has_next       = payload.get("hasNext", False)
                page          += 1
                time.sleep(SH_DELAY_SEC)

            except Exception as e:
                log.error(f"Sharehub {target_date} page {page} error: {e}")
                time.sleep(5)
                break

        return total_written


# ─── Trading Days ─────────────────────────────────────────────────────────────

def _get_trading_days(start: date, end: date) -> list:
    days = pd.date_range(start=start, end=end, freq="B")
    return [d.date() for d in days]


# ─── Backfill ─────────────────────────────────────────────────────────────────

def run_backfill(
    start_date : date = date(2019, 7, 15),
    end_date   : date = None,
    source     : str  = "both",
) -> None:
    """
    Backfill historical floorsheet data with progress bar.
    Auto-resumes from last scraped date.
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    ml_scraper = MerolaganiScraper() if source in ("merolagani", "both") else None
    sh_scraper = SharehubScraper()   if source in ("sharehubnepal", "both") else None

    trading_days = _get_trading_days(start_date, end_date)
    total_days   = len(trading_days)
    total_rows   = 0
    done_days    = 0

    print(f"\nBackfill: {start_date} → {end_date}")
    print(f"Trading days: {total_days} | Source: {source}\n")

    start_time = time.time()

    for i, target_date in enumerate(trading_days):
        date_str = target_date.strftime("%Y-%m-%d")

        # Decide source for this date
        if source == "both":
            use_sh = target_date >= SH_START_DATE
            use_ml = not use_sh
        else:
            use_sh = source == "sharehubnepal"
            use_ml = source == "merolagani"

        src_name = "sharehubnepal" if use_sh else "merolagani"

        # Skip already scraped
        if _date_already_scraped(target_date, src_name):
            done_days += 1
            _progress(i + 1, total_days, date_str, total_rows, "SKIP")
            continue

        # Scrape
        _progress(i + 1, total_days, date_str, total_rows, "...")

        rows = 0
        try:
            if use_sh and sh_scraper:
                rows = sh_scraper.scrape_date(target_date)
            elif use_ml and ml_scraper:
                rows = ml_scraper.scrape_date(target_date)
        except Exception as e:
            log.error(f"\nScrape failed {date_str}: {e}")
            time.sleep(10)
            continue

        total_rows += rows
        done_days  += 1

        status = f"+{rows}" if rows > 0 else "HOLIDAY"
        _progress(i + 1, total_days, date_str, total_rows, status)

        # ETA every 50 days
        if (i + 1) % 50 == 0:
            elapsed  = time.time() - start_time
            rate     = (i + 1) / elapsed if elapsed > 0 else 1
            remaining = (total_days - i - 1) / rate
            eta_min  = remaining / 60
            print(f"\n  ETA: {eta_min:.0f} min remaining | {total_rows:,} rows so far")

        time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\n\nBackfill complete!")
    print(f"  Days scraped : {done_days}/{total_days}")
    print(f"  Total rows   : {total_rows:,}")
    print(f"  Time elapsed : {elapsed/60:.1f} minutes")


# ─── Daily ────────────────────────────────────────────────────────────────────

def run_daily(target_date: date = None) -> int:
    """
    Scrape yesterday's floorsheet.
    Called by morning_workflow at 10:30 AM.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    if target_date.weekday() >= 5:
        log.info(f"run_daily: {target_date} is weekend — skip")
        return 0

    src_name = "sharehubnepal" if target_date >= SH_START_DATE else "merolagani"

    if _date_already_scraped(target_date, src_name):
        log.info(f"run_daily: {target_date} already scraped — skip")
        return 0

    print(f"Scraping floorsheet for {target_date}...")

    if target_date >= SH_START_DATE:
        scraper = SharehubScraper()
    else:
        scraper = MerolaganiScraper()

    rows = scraper.scrape_date(target_date)
    print(f"Done: {rows:,} rows written")
    return rows


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level  = logging.WARNING,  # suppress info during backfill — progress bar handles output
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    parser = argparse.ArgumentParser(description="NEPSE Floorsheet Scraper")
    parser.add_argument("--mode",   choices=["backfill", "daily"], default="daily")
    parser.add_argument("--start",  default="2023-07-02")
    parser.add_argument("--end",    default=None)
    parser.add_argument("--source", choices=["merolagani", "sharehubnepal", "both"], default="both")
    args = parser.parse_args()

    if args.mode == "backfill":
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end else None
        run_backfill(start_date=start, end_date=end, source=args.source)
    else:
        rows = run_daily()
        print(f"Daily scrape: {rows:,} rows written")