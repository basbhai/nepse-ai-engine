import logging
import re
import sys
import time
from datetime import datetime, date, timedelta, timezone
from typing import Optional

# Assuming these are in your local sheets.py
from sheets import (
    write_index_batch,
    get_latest_index_date,
    get_index_coverage,
)

log = logging.getLogger(__name__)
NST = timezone(timedelta(hours=5, minutes=45))
PAGE_URL = "https://www.nepalstock.com/indices"

INDEX_NAMES: dict[int, str] = {
    58: "NEPSE", 51: "Banking", 52: "Hotels And Tourism", 53: "Others",
    54: "Hydropower", 55: "Development Bank", 56: "Manufacturing And Processing",
    57: "Sensitive", 59: "Non-Life Insurance", 60: "Finance", 61: "Trading",
    62: "Float", 63: "Sensitive Float", 64: "Microfinance", 65: "Life Insurance",
    66: "Mutual Fund", 67: "Investment",
}

ALL_INDEX_IDS = list(INDEX_NAMES.keys())
DEFAULT_FROM  = date(2023, 7, 15)
BATCH_DELAY   = 2.0
PAGE_DELAY    = 1.5

# =============================================================================
# PARSING HELPERS
# =============================================================================

def _clean_number(text: str) -> Optional[float]:
    if not text: return None
    s = re.sub(r'[,\s]', '', text.strip())
    if s in ('', '-', 'N/A'): return None
    try: return float(s)
    except ValueError: return None

def _parse_change_cell(cell) -> Optional[float]:
    span = cell.find("span", class_=lambda c: c and ("positive" in c or "negative" in c or "changeup" in c or "changedown" in c))
    if not span: return 0.0
    classes = span.get("class", [])
    is_negative = "negative" in classes or "changedown" in classes
    raw = re.sub(r'[(),\s%a-zA-Z]', '', span.get_text())
    try:
        val = float(raw)
        return -abs(val) if is_negative else abs(val)
    except ValueError: return None

def _parse_date(text: str) -> Optional[str]:
    if not text: return None
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try: return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError: pass
    return None

def parse_table(html: str, index_id: int) -> list[dict]:
    from bs4 import BeautifulSoup
    name = INDEX_NAMES.get(index_id, f"Index_{index_id}")
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    if not tbody: return []

    parsed = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 10: continue
        date_str = _parse_date(cells[1].get_text(strip=True))
        if not date_str: continue
        close = _clean_number(cells[2].get_text(strip=True))
        if close is None: continue

        parsed.append({
            "date": date_str,
            "index_id": str(index_id),
            "index_name": name,
            "current_value": str(close),
            "high": str(_clean_number(cells[3].get_text(strip=True)) or ""),
            "low": str(_clean_number(cells[4].get_text(strip=True)) or ""),
            "change_abs": str(_parse_change_cell(cells[5]) or ""),
            "change_pct": str(_parse_change_cell(cells[6]) or ""),
            "week52_high": str(_clean_number(cells[7].get_text(strip=True)) or ""),
            "week52_low": str(_clean_number(cells[8].get_text(strip=True)) or ""),
            "turnover": str(_clean_number(cells[9].get_text(strip=True)) or ""),
            "volume": str(_clean_number(cells[10].get_text(strip=True)) or ""),
            "transactions": str(_clean_number(cells[11].get_text(strip=True)) or ""),
            "source": "nepalstock",
        })
    return parsed

# =============================================================================
# PLAYWRIGHT SCRAPER
# =============================================================================

def _total_pages(page) -> int:
    try:
        text = page.locator("li.small-screen").text_content(timeout=3000)
        parts = text.strip().split("/")
        if len(parts) == 2: return int(parts[1].strip())
    except: pass
    return 1

def scrape_index(index_id: int, from_date: str = None, headless: bool = True) -> list[dict]:
    from playwright.sync_api import sync_playwright
    name = INDEX_NAMES.get(index_id, f"Index_{index_id}")
    all_rows = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        ctx = browser.new_context(viewport={'width': 1920, 'height': 1080}, ignore_https_errors=True)
        page = ctx.new_page()

        try:
            log.info("[%s] Navigating...", name)
            page.goto(PAGE_URL, wait_until="networkidle", timeout=40000)
            
            # Select Index and Rows
            page.locator("div.box__filter--field select").first.select_option(str(index_id))
            page.locator("div.table__perpage select").select_option("500")
            
            # Filter and Wait
            page.click("button.box__filter--search")
            page.wait_for_load_state("networkidle")
            time.sleep(PAGE_DELAY)

            total = _total_pages(page)
            log.info("[%s] %d pages to scrape", name, total)

            stop = False
            for pg in range(1, total + 1):
                page.wait_for_selector("table tbody tr", timeout=15000)
                # FIX: Using .inner_html() instead of .outer_html()
                html = page.locator("div.table-responsive").inner_html()
                rows = parse_table(html, index_id)
                
                if from_date:
                    original_len = len(rows)
                    rows = [r for r in rows if r["date"] >= from_date]
                    if len(rows) < original_len: stop = True

                all_rows.extend(rows)
                if stop or pg == total: break

                page.locator("li.pagination-next a").click()
                page.wait_for_load_state("networkidle")
                time.sleep(PAGE_DELAY)

        except Exception as e:
            log.error("[%s] Scrape error: %s", name, e)
        finally:
            browser.close()
    return all_rows

# =============================================================================
# RUNNER LOGIC
# =============================================================================

def run_backfill(index_ids=None, from_date=None, dry_run=False, incremental=True, headless=True):
    if index_ids is None: index_ids = ALL_INDEX_IDS
    if from_date is None: from_date = DEFAULT_FROM
    from_str = from_date.strftime("%Y-%m-%d")
    today = datetime.now(tz=NST).date().strftime("%Y-%m-%d")

    summary = {}
    for idx_id in index_ids:
        name = INDEX_NAMES.get(idx_id, f"Index_{idx_id}")
        eff_from = from_str
        
        if incremental and not dry_run:
            latest = get_latest_index_date(idx_id)
            if latest and latest >= today:
                log.info("[%s] Up to date (%s)", name, latest)
                continue
            if latest and latest > from_str:
                eff_from = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        rows = scrape_index(idx_id, from_date=eff_from, headless=headless)
        if rows:
            if not dry_run:
                written = write_index_batch(rows)
                log.info("[%s] Written %d rows", name, written)
            else:
                log.info("[%s] Dry run: %d rows", name, len(rows))
        time.sleep(BATCH_DELAY)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-browser", action="store_true")
    args = parser.parse_args()
    run_backfill(headless=not args.show_browser)