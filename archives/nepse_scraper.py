"""
NEPSE Share Price Scraper
=========================
3-step flow:
  1. GET  /                      -> seed XSRF-TOKEN + session cookies
  2. POST /home-indices          -> refresh session
  3. POST /ajaxtodayshareprice   -> fetch share price HTML table

KEY FIX: requests automatically URL-decodes cookie values when storing them,
but the browser sends the RAW URL-encoded cookie value as X-CSRF-Token header.
We intercept the raw Set-Cookie response header before it gets decoded.

Usage:
  pip install requests beautifulsoup4 pandas
  python nepse_scraper.py
  python nepse_scraper.py --date 2026-03-12
  python nepse_scraper.py --date 2026-03-12 --sector banking
  python nepse_scraper.py --date 2026-03-12 --output csv
"""

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd


BASE_URL     = "https://www.sharesansar.com"
HOME_PAGE    = f"{BASE_URL}/"
HOME_INDICES = f"{BASE_URL}/home-indices"
AJAX_PRICE   = f"{BASE_URL}/ajaxtodayshareprice"

VALID_SECTORS = [
    "all_sec", "banking", "development_bank", "finance", "microfinance",
    "life_insurance", "non_life_insurance", "hydropower", "manufacturing", "others",
]

BROWSER_HEADERS = {
    "User-Agent":         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/146.0.0.0 Safari/537.36",
    "Accept-Language":    "en-GB,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding":    "gzip, deflate, br",
    "sec-ch-ua":          '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
}


# -- Raw XSRF token extraction ------------------------------------------------

def get_raw_xsrf(response: requests.Response) -> str:
    """
    Extract the XSRF-TOKEN value directly from the raw Set-Cookie header
    BEFORE requests URL-decodes it.

    Why: requests stores cookies URL-decoded internally (e.g. %3D -> =),
    but Laravel's VerifyCsrfToken expects the X-CSRF-Token header to match
    the raw URL-encoded cookie value exactly (as the browser sends it).

    response.headers may contain multiple Set-Cookie entries; in urllib3
    they are joined with ', ' but we can also use response.raw.headers
    which preserves duplicates.
    """
    # Try raw headers first (preserves multiple Set-Cookie lines)
    raw_headers = response.raw.headers.getlist("Set-Cookie") if hasattr(response.raw.headers, "getlist") else []

    if not raw_headers:
        # Fallback: parse the combined Set-Cookie string
        combined = response.headers.get("Set-Cookie", "")
        raw_headers = [combined]

    for cookie_str in raw_headers:
        m = re.search(r"XSRF-TOKEN=([^;]+)", cookie_str)
        if m:
            token = m.group(1).strip()
            return token

    return ""


# -- Step 1: GET homepage ------------------------------------------------------

def init_session() -> tuple[requests.Session, str]:
    """GET / to seed initial cookies. Returns (session, raw_xsrf_token)."""
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    print("[1/3] GET homepage to seed cookies...")
    resp = session.get(
        HOME_PAGE,
        headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
        timeout=20,
    )
    resp.raise_for_status()

    xsrf = get_raw_xsrf(resp)
    if not xsrf:
        raise RuntimeError("XSRF-TOKEN not found in GET / response headers.")

    print(f"    Cookie set.  Raw XSRF-TOKEN: {xsrf[:60]}...")
    return session, xsrf


# -- Step 2: POST /home-indices -----------------------------------------------

def refresh_session(session: requests.Session, xsrf: str) -> str:
    """POST /home-indices (empty body) to refresh session. Returns updated xsrf."""
    print("[2/3] POST /home-indices to refresh session...")
    resp = session.post(
        HOME_INDICES,
        data="",
        headers={
            "Accept":           "*/*",
            "Content-Type":     "application/x-www-form-urlencoded; charset=UTF-8",
            "Content-Length":   "0",
            "Origin":           BASE_URL,
            "Referer":          f"{BASE_URL}/",
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token":     xsrf,
            "sec-fetch-dest":   "empty",
            "sec-fetch-mode":   "cors",
            "sec-fetch-site":   "same-origin",
        },
        timeout=20,
    )
    resp.raise_for_status()

    new_xsrf = get_raw_xsrf(resp)
    if new_xsrf:
        xsrf = new_xsrf

    print(f"    Session refreshed.  Raw XSRF-TOKEN: {xsrf[:60]}...")
    return xsrf


# -- Step 3: POST /ajaxtodayshareprice ----------------------------------------

def fetch_share_prices(
    session: requests.Session,
    xsrf: str,
    trade_date: str,
    sector: str = "all_sec",
) -> str:
    print(f"[3/3] POST /ajaxtodayshareprice  date={trade_date}  sector={sector}...")

    resp = session.post(
        AJAX_PRICE,
        data={
            "_token": xsrf,
            "sector": sector,
            "date":   trade_date,
        },
        headers={
            "Accept":           "*/*",
            "Content-Type":     "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin":           BASE_URL,
            "Referer":          f"{BASE_URL}/today-share-price",
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token":     xsrf,
            "sec-fetch-dest":   "empty",
            "sec-fetch-mode":   "cors",
            "sec-fetch-site":   "same-origin",
            "priority":         "u=1, i",
        },
        timeout=30,
    )
    resp.raise_for_status()
    print(f"    OK  status={resp.status_code}  size={len(resp.text):,} chars")
    return resp.text


# -- Parse HTML table ---------------------------------------------------------

def parse_html(html: str) -> tuple[str, list[dict]]:
    soup = BeautifulSoup(html, "html.parser")

    as_of_el = soup.find(class_="text-org")
    as_of = as_of_el.get_text(strip=True) if as_of_el else "unknown"

    rows = soup.select("tbody tr")
    if not rows:
        print("[!] No rows found -- market may be closed on that date.")
        return as_of, []

    def to_float(text):
        try:    return float(text.strip().replace(",", ""))
        except: return None

    def to_int(text):
        try:    return int(float(text.strip().replace(",", "")))
        except: return None

    records = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 18:
            continue
        link   = cells[1].find("a")
        symbol = link.get_text(strip=True) if link else cells[1].get_text(strip=True)
        records.append({
            "sno":           to_int  (cells[0].get_text()),
            "symbol":        symbol,
            "company":       link.get("title", "") if link else "",
            "url":           link["href"] if link else "",
            "confidence":    to_float(cells[2].get_text()),
            "open":          to_float(cells[3].get_text()),
            "high":          to_float(cells[4].get_text()),
            "low":           to_float(cells[5].get_text()),
            "close":         to_float(cells[6].get_text()),
            "ltp":           to_float(cells[7].get_text()),
            "close_ltp":     to_float(cells[8].get_text()),
            "close_ltp_pct": to_float(cells[9].get_text()),
            "vwap":          to_float(cells[10].get_text()),
            "volume":        to_int  (cells[11].get_text()),
            "prev_close":    to_float(cells[12].get_text()),
            "turnover":      to_float(cells[13].get_text()),
            "transactions":  to_int  (cells[14].get_text()),
            "diff":          to_float(cells[15].get_text()),
            "range":         to_float(cells[16].get_text()),
            "diff_pct":      to_float(cells[17].get_text()),
            "range_pct":     to_float(cells[18].get_text()) if len(cells) > 18 else None,
            "vwap_pct":      to_float(cells[19].get_text()) if len(cells) > 19 else None,
            "week52_high":   to_float(cells[20].get_text()) if len(cells) > 20 else None,
            "week52_low":    to_float(cells[21].get_text()) if len(cells) > 21 else None,
        })

    print(f"    Parsed {len(records)} rows  |  as of: {as_of}")
    return as_of, records


# -- Save outputs -------------------------------------------------------------

def save_outputs(records: list[dict], trade_date: str, sector: str, fmt: str):
    if not records:
        print("[!] Nothing to save.")
        return

    base = f"nepse_{trade_date.replace('-', '')}_{sector}"

    if fmt in ("csv", "both"):
        p = Path(f"{base}.csv")
        pd.DataFrame(records).to_csv(p, index=False)
        print(f"[OK] Saved -> {p}")

    if fmt in ("json", "both"):
        p = Path(f"{base}.json")
        p.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] Saved -> {p}")

    gainers = sum(1 for r in records if (r["diff_pct"] or 0) > 0)
    losers  = sum(1 for r in records if (r["diff_pct"] or 0) < 0)
    flat    = len(records) - gainers - losers
    print(f"\n  Total: {len(records)}  | Gainers: {gainers}  | Losers: {losers}  | Flat: {flat}")


# -- CLI ----------------------------------------------------------------------

def main():
    today     = date.today()
    days_back = 1 if today.weekday() not in (5, 6) else (today.weekday() - 4)

    p = argparse.ArgumentParser(description="NEPSE share price scraper")
    p.add_argument("--date",   default=(today - timedelta(days=days_back)).isoformat())
    p.add_argument("--sector", default="all_sec", choices=VALID_SECTORS)
    p.add_argument("--output", default="both",    choices=["csv", "json", "both"])
    args = p.parse_args()

    print(f"\n{'='*55}")
    print(f"  NEPSE Scraper  |  date={args.date}  sector={args.sector}")
    print(f"{'='*55}\n")

    try:
        session, xsrf  = init_session()
        xsrf           = refresh_session(session, xsrf)
        html           = fetch_share_prices(session, xsrf, args.date, args.sector)
        _, records     = parse_html(html)
        save_outputs(records, args.date, args.sector, args.output)
    except requests.HTTPError as e:
        print(f"\n[ERROR] HTTP {e.response.status_code}: {e}", file=sys.stderr)
        print(f"        Body: {e.response.text[:500]}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()