"""
modules/fundamentals_scraper.py

Scrapes all historical quarterly fundamentals from Chukul API.
Stores raw data into fundamentals table via Prisma schema.
Rate limited to 30 req/min with 1-2s random delay.
Run once per quarter via GitHub Actions.

Usage:
    python -m modules.fundamentals_scraper
    python -m modules.fundamentals_scraper --dry-run
"""

import time
import random
import logging
import argparse
import requests
from datetime import datetime, timezone
from db.connection import _db

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

BASE_URL = "https://chukul.com/api/stock/"
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0
MAX_PER_MINUTE = 30

# Fields to extract from each quarter row — stored raw, no cleaning
FIELDS = [
    "quarter", "fiscal_year", "eps", "net_worth", "roe", "roa",
    "paidup_capital", "reserve", "total_assets", "total_liabilities",
    "deposit", "loan", "net_interest_income", "operating_profit",
    "net_profit", "npl", "capital_fund_to_rwa", "cost_of_fund",
    "base_rate", "interest_spread", "cd_ratio", "stock_id", "dps",
    "symbol", "sector_id", "promoter_shares", "public_shares",
    "share_registar", "is_delisted", "is_merged", "core_capital",
    "gram_value", "prev_quarter_profit", "growth_rate", "close",
    "discount_rate", "pe_ratio", "peg_value",
]


def _fetch_json(url: str) -> dict | list | None:
    """Fetch JSON from URL. Returns None on any error — silent skip."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _get_all_stocks() -> list[dict]:
    """Fetch master stock list. Skip debentures and mutual funds by name."""
    data = _fetch_json(BASE_URL)
    if not data:
        log.error("Failed to fetch stock list from Chukul API")
        return []

    filtered = []
    for s in data:
        name = s.get("name", "").lower()
        if "debenture" in name or "mutual" in name:
            continue
        filtered.append(s)

    log.info("Total symbols to process: %d", len(filtered))
    return filtered


def _upsert_row(row: dict, dry_run: bool) -> bool:
    """
    Upsert one fundamental row into DB.
    Conflict on (symbol, fiscal_year, quarter) → update all fields.
    Returns True on success.
    """
    if dry_run:
        log.info("[DRY RUN] Would upsert: %s %s %s",
                 row.get("symbol"), row.get("fiscal_year"), row.get("quarter"))
        return True

    sql = """
        INSERT INTO fundamentals (
            symbol, stock_id, fiscal_year, quarter,
            eps, net_worth, roe, roa, paidup_capital, reserve,
            total_assets, total_liabilities, deposit, loan,
            net_interest_income, operating_profit, net_profit,
            npl, capital_fund_to_rwa, cost_of_fund, base_rate,
            interest_spread, cd_ratio, dps, sector_id,
            promoter_shares, public_shares, share_registar,
            is_delisted, is_merged, core_capital, gram_value,
            prev_quarter_profit, growth_rate, close, discount_rate,
            pe_ratio, peg_value, scraped_at
        ) VALUES (
            %(symbol)s, %(stock_id)s, %(fiscal_year)s, %(quarter)s,
            %(eps)s, %(net_worth)s, %(roe)s, %(roa)s, %(paidup_capital)s,
            %(reserve)s, %(total_assets)s, %(total_liabilities)s,
            %(deposit)s, %(loan)s, %(net_interest_income)s,
            %(operating_profit)s, %(net_profit)s, %(npl)s,
            %(capital_fund_to_rwa)s, %(cost_of_fund)s, %(base_rate)s,
            %(interest_spread)s, %(cd_ratio)s, %(dps)s, %(sector_id)s,
            %(promoter_shares)s, %(public_shares)s, %(share_registar)s,
            %(is_delisted)s, %(is_merged)s, %(core_capital)s,
            %(gram_value)s, %(prev_quarter_profit)s, %(growth_rate)s,
            %(close)s, %(discount_rate)s, %(pe_ratio)s, %(peg_value)s,
            %(scraped_at)s
        )
        ON CONFLICT (symbol, fiscal_year, quarter)
        DO UPDATE SET
            stock_id            = EXCLUDED.stock_id,
            eps                 = EXCLUDED.eps,
            net_worth           = EXCLUDED.net_worth,
            roe                 = EXCLUDED.roe,
            roa                 = EXCLUDED.roa,
            paidup_capital      = EXCLUDED.paidup_capital,
            reserve             = EXCLUDED.reserve,
            total_assets        = EXCLUDED.total_assets,
            total_liabilities   = EXCLUDED.total_liabilities,
            deposit             = EXCLUDED.deposit,
            loan                = EXCLUDED.loan,
            net_interest_income = EXCLUDED.net_interest_income,
            operating_profit    = EXCLUDED.operating_profit,
            net_profit          = EXCLUDED.net_profit,
            npl                 = EXCLUDED.npl,
            capital_fund_to_rwa = EXCLUDED.capital_fund_to_rwa,
            cost_of_fund        = EXCLUDED.cost_of_fund,
            base_rate           = EXCLUDED.base_rate,
            interest_spread     = EXCLUDED.interest_spread,
            cd_ratio            = EXCLUDED.cd_ratio,
            dps                 = EXCLUDED.dps,
            sector_id           = EXCLUDED.sector_id,
            promoter_shares     = EXCLUDED.promoter_shares,
            public_shares       = EXCLUDED.public_shares,
            share_registar      = EXCLUDED.share_registar,
            is_delisted         = EXCLUDED.is_delisted,
            is_merged           = EXCLUDED.is_merged,
            core_capital        = EXCLUDED.core_capital,
            gram_value          = EXCLUDED.gram_value,
            prev_quarter_profit = EXCLUDED.prev_quarter_profit,
            growth_rate         = EXCLUDED.growth_rate,
            close               = EXCLUDED.close,
            discount_rate       = EXCLUDED.discount_rate,
            pe_ratio            = EXCLUDED.pe_ratio,
            peg_value           = EXCLUDED.peg_value,
            scraped_at          = EXCLUDED.scraped_at
    """
    try:
        with _db() as cur:
            cur.execute(sql, row)
        return True
    except Exception as e:
        log.debug("DB upsert failed for %s %s %s: %s",
                  row.get("symbol"), row.get("fiscal_year"),
                  row.get("quarter"), e)
        return False


def _process_symbol(stock: dict, dry_run: bool) -> tuple[int, int]:
    """
    Fetch all quarterly reports for one symbol, upsert each row.
    Returns (rows_upserted, rows_failed).
    """
    stock_id = stock.get("id")
    symbol = stock.get("symbol", "").upper()

    if not stock_id or not symbol:
        return 0, 0

    url = f"{BASE_URL}{stock_id}/report/"
    reports = _fetch_json(url)

    if not reports:
        log.debug("No report data for %s — skipping", symbol)
        return 0, 0

    if not isinstance(reports, list):
        log.debug("Unexpected report format for %s — skipping", symbol)
        return 0, 0

    upserted = 0
    failed = 0
    now = datetime.now(timezone.utc)

    for entry in reports:
        fiscal_year = entry.get("fiscal_year")
        quarter = entry.get("quarter")

        if not fiscal_year or not quarter:
            failed += 1
            continue

        row = {
            "symbol":               symbol,
            "stock_id":             entry.get("stock_id") or stock_id,
            "fiscal_year":          str(fiscal_year),
            "quarter":              str(quarter).lower(),
            "eps":                  entry.get("eps"),
            "net_worth":            entry.get("net_worth"),
            "roe":                  entry.get("roe"),
            "roa":                  entry.get("roa"),
            "paidup_capital":       entry.get("paidup_capital"),
            "reserve":              entry.get("reserve"),
            "total_assets":         entry.get("total_assets"),
            "total_liabilities":    entry.get("total_liabilities"),
            "deposit":              entry.get("deposit"),
            "loan":                 entry.get("loan"),
            "net_interest_income":  entry.get("net_interest_income"),
            "operating_profit":     entry.get("operating_profit"),
            "net_profit":           entry.get("net_profit"),
            "npl":                  entry.get("npl"),
            "capital_fund_to_rwa":  entry.get("capital_fund_to_rwa"),
            "cost_of_fund":         entry.get("cost_of_fund"),
            "base_rate":            entry.get("base_rate"),
            "interest_spread":      entry.get("interest_spread"),
            "cd_ratio":             entry.get("cd_ratio"),
            "dps":                  entry.get("dps"),
            "sector_id":            entry.get("sector_id"),
            "promoter_shares":      entry.get("promoter_shares"),
            "public_shares":        entry.get("public_shares"),
            "share_registar":       entry.get("share_registar"),
            "is_delisted":          entry.get("is_delisted"),
            "is_merged":            entry.get("is_merged"),
            "core_capital":         entry.get("core_capital"),
            "gram_value":           entry.get("gram_value"),
            "prev_quarter_profit":  entry.get("prev_quarter_profit"),
            "growth_rate":          entry.get("growth_rate"),
            "close":                entry.get("close"),
            "discount_rate":        entry.get("discount_rate"),
            "pe_ratio":             entry.get("pe_ratio"),
            "peg_value":            entry.get("peg_value"),
            "scraped_at":           now,
        }

        if _upsert_row(row, dry_run):
            upserted += 1
        else:
            failed += 1

    return upserted, failed


def run(dry_run: bool = False):
    """
    Main runner. Processes all symbols with rate limiting.
    Logs summary at end.
    """
    log.info("=== Fundamentals scraper started (dry_run=%s) ===", dry_run)
    stocks = _get_all_stocks()

    if not stocks:
        log.error("No stocks to process. Exiting.")
        return

    total_upserted = 0
    total_failed = 0
    total_skipped = 0

    req_count = 0
    window_start = time.monotonic()

    for i, stock in enumerate(stocks):
        symbol = stock.get("symbol", "?").upper()

        # Rate limit: max 30 requests per minute
        req_count += 1
        if req_count >= MAX_PER_MINUTE:
            elapsed = time.monotonic() - window_start
            if elapsed < 60.0:
                sleep_for = 60.0 - elapsed + 0.5
                log.info("Rate limit reached — sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)
            req_count = 0
            window_start = time.monotonic()

        # Random delay between requests
        delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
        time.sleep(delay)

        upserted, failed = _process_symbol(stock, dry_run)

        if upserted == 0 and failed == 0:
            total_skipped += 1
            log.debug("Skipped %s (no data)", symbol)
        else:
            total_upserted += upserted
            total_failed += failed
            log.info("[%d/%d] %s — upserted: %d, failed: %d",
                     i + 1, len(stocks), symbol, upserted, failed)

    log.info("=== Scrape complete ===")
    log.info("Total rows upserted : %d", total_upserted)
    log.info("Total rows failed   : %d", total_failed)
    log.info("Total symbols skipped (no data): %d", total_skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEPSE Fundamentals Scraper")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute everything, write nothing")
    args = parser.parse_args()
    run(dry_run=args.dry_run)