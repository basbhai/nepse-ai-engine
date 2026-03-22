"""
sharehub_index_scraper.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Historical Index Backfill
Source : https://sharehubnepal.com/data/api/v1/index/date-wise-analysis?date=YYYY-MM-DD
Purpose: Fetch all NEPSE indices per trading day and upsert into nepse_indices table.
         Backfills from 2020-08-01 to today by default.

API returns all indices in one call per date — very efficient.
No login, no captcha, no Playwright needed.

Usage:
    python sharehub_index_scraper.py                    # full backfill 2020-08-01 → today
    python sharehub_index_scraper.py --from 2023-07-15  # custom start date
    python sharehub_index_scraper.py --dry-run          # parse only, no DB write
    python sharehub_index_scraper.py --status           # show what's in DB
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import logging
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SHAREHUB] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST = timezone(timedelta(hours=5, minutes=45))

API_URL      = "https://sharehubnepal.com/data/api/v1/index/date-wise-analysis"
DEFAULT_FROM = date(2023, 10, 23)
REQUEST_DELAY = 0.8   # seconds between requests — polite to server
TIMEOUT       = 15    # seconds per request
MAX_RETRIES   = 3

HEADERS = {
    "accept":          "*/*",
    "accept-language": "en-US,en;q=0.9",
    "referer":         "https://sharehubnepal.com/nepse/indices",
    "sec-ch-ua":       '"Chromium";v="146", "Not-A.Brand";v="24", "Microsoft Edge";v="146"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest":  "empty",
    "sec-fetch-mode":  "cors",
    "sec-fetch-site":  "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36 Edg/146.0.0.0"
    ),
}

# Map ShareHub symbol → index_id used in our nepse_indices table
# Existing IDs from index_scraper.py: 58=NEPSE, 57=Sensitive, 62=Float,
# 63=SensitiveFloat, 51=Banking, 52=Hotels, 53=Others, 54=Hydro,
# 55=DevBank, 56=Manufacturing, 59=NonLifeInsurance, 60=Finance,
# 61=Trading, 64=Microfinance, 65=LifeInsurance, 66=MutualFund, 67=Investment
SYMBOL_TO_ID = {
    "NEPSE":          "58",
    "SENSITIVE":      "57",
    "FLOAT":          "62",
    "SENSITIVE FLOAT":"63",
    "SENSITIVEFLOAT": "63",
    "BANKING":        "51",
    "HOTELS AND TOURISM": "52",
    "HOTELS":         "52",
    "OTHERS":         "53",
    "HYDROPOWER":     "54",
    "HYDRO":          "54",
    "DEVELOPMENT BANK":"55",
    "DEVBANK":        "55",
    "MANUFACTURING AND PROCESSING": "56",
    "MANUFACTURING":  "56",
    "NON LIFE INSURANCE": "59",
    "NON-LIFE INSURANCE": "59",
    "NONLIFE":        "59",
    "FINANCE":        "60",
    "TRADING":        "61",
    "MICROFINANCE":   "64",
    "LIFE INSURANCE": "65",
    "LIFEINSURANCE":  "65",
    "MUTUAL FUND":    "66",
    "MUTUALFUND":     "66",
    "INVESTMENT":     "67",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRADING DAY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def get_trading_days(from_date: date, to_date: date) -> list[date]:
    """
    Generate all potential trading days (Sun–Thu) between two dates.
    We don't filter holidays here — the API will return empty for holidays.
    """
    days = []
    current = from_date
    while current <= to_date:
        # weekday(): Mon=0 ... Sun=6
        # Trading days: Sun=6, Mon=0, Tue=1, Wed=2, Thu=3
        if current.weekday() in (6, 0, 1, 2, 3):
            days.append(current)
        current += timedelta(days=1)
    return days


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FETCH ONE DATE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_date(target_date: date) -> list[dict]:
    """
    Fetch all index values for one trading date from ShareHub API.

    Returns list of parsed row dicts ready for DB insert.
    Returns empty list if market was closed that day or API error.
    """
    date_str = target_date.strftime("%Y-%m-%d")

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                API_URL,
                params={"date": date_str},
                headers=HEADERS,
                timeout=TIMEOUT,
            )

            if resp.status_code != 200:
                log.warning("HTTP %d for %s (attempt %d)", resp.status_code, date_str, attempt + 1)
                time.sleep(2)
                continue

            data = resp.json()

            if not data.get("success"):
                log.debug("API returned success=false for %s — market closed", date_str)
                return []

            content = data.get("data", {}).get("content", [])
            if not content:
                log.debug("No content for %s — market closed or holiday", date_str)
                return []

            rows = []
            for item in content:
                # Only process actual index entries
                if not item.get("isIndex", False):
                    continue

                symbol = item.get("symbol", "").upper().strip()
                name   = item.get("name", "").strip()

                # Resolve index_id
                index_id = SYMBOL_TO_ID.get(symbol)
                if not index_id:
                    # Try matching by name
                    name_upper = name.upper()
                    index_id = SYMBOL_TO_ID.get(name_upper)
                if not index_id:
                    log.debug("Unknown index symbol '%s' on %s — skipping", symbol, date_str)
                    continue

                rows.append({
                    "date":          date_str,
                    "index_id":      index_id,
                    "index_name":    name,
                    "current_value": str(item.get("close", "")),
                    "high":          str(item.get("high", "")),
                    "low":           str(item.get("low", "")),
                    "change_abs":    str(item.get("change", "")),
                    "change_pct":    str(item.get("changePercent", "")),
                    "turnover":      str(item.get("turnover", "")),
                    "volume":        str(item.get("volume", "")),
                    "transactions":  str(item.get("transactions", "")),
                    "source":        "sharehubnepal",
                })

            return rows

        except requests.exceptions.Timeout:
            log.warning("Timeout for %s (attempt %d)", date_str, attempt + 1)
            time.sleep(3)
        except Exception as exc:
            log.warning("Error fetching %s (attempt %d): %s", date_str, attempt + 1, exc)
            time.sleep(2)

    return []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATABASE WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_to_db(rows: list[dict]) -> int:
    """
    Bulk upsert rows into nepse_indices table.
    ON CONFLICT (date, index_id) DO UPDATE — safe to re-run.
    Returns number of rows written.
    """
    if not rows:
        return 0

    from db.connection import _db
    import psycopg2.extras

    columns = [
        "date", "index_id", "index_name", "current_value",
        "high", "low", "change_abs", "change_pct",
        "turnover", "volume", "transactions", "source",
    ]

    col_sql  = ", ".join(f'"{c}"' for c in columns)
    val_sql  = ", ".join(["%s"] * len(columns))
    upd_sql  = ", ".join(
        f'"{c}" = EXCLUDED."{c}"'
        for c in columns if c not in ("date", "index_id")
    )

    values = [tuple(r.get(c, "") for c in columns) for r in rows]

    try:
        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO nepse_indices ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT (date, index_id)
                DO UPDATE SET {upd_sql}
                """,
                values,
                page_size=500,
            )
        return len(rows)
    except Exception as exc:
        log.error("DB write failed: %s", exc)
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ALREADY LOADED CHECK
# ══════════════════════════════════════════════════════════════════════════════

def get_loaded_dates() -> set[str]:
    """
    Return set of dates already in nepse_indices for NEPSE composite (index_id=58).
    Used to skip already-loaded dates for incremental runs.
    """
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute(
                "SELECT DISTINCT date FROM nepse_indices WHERE index_id = %s",
                ("58",)
            )
            return {row["date"] for row in cur.fetchall()}
    except Exception as exc:
        log.warning("Could not read loaded dates: %s", exc)
        return set()


def print_status():
    """Print current coverage of nepse_indices table."""
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("""
                SELECT
                    index_id,
                    index_name,
                    COUNT(*)    AS trading_days,
                    MIN(date)   AS earliest,
                    MAX(date)   AS latest
                FROM nepse_indices
                GROUP BY index_id, index_name
                ORDER BY index_id::int
            """)
            rows = cur.fetchall()

        if not rows:
            print("\n  nepse_indices table is EMPTY. Run the scraper.\n")
            return

        print(f"\n{'='*65}")
        print(f"  NEPSE INDICES TABLE STATUS")
        print(f"{'='*65}")
        print(f"  {'ID':<5} {'Index':<30} {'Days':>6}  {'Earliest':<12}  {'Latest'}")
        print(f"  {'─'*60}")
        for r in rows:
            print(f"  {r['index_id']:<5} {(r['index_name'] or ''):<30} "
                  f"{r['trading_days']:>6}  {r['earliest']:<12}  {r['latest']}")
        print(f"{'='*65}\n")
    except Exception as exc:
        print(f"\n  Error reading status: {exc}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(from_date: date, dry_run: bool = False):
    today    = date.today()
    all_days = get_trading_days(from_date, today)

    print(f"\n{'='*65}")
    print(f"  SHAREHUB NEPSE INDEX BACKFILL")
    print(f"{'='*65}")
    print(f"  From:          {from_date}")
    print(f"  To:            {today}")
    print(f"  Trading days:  {len(all_days)}")
    print(f"  Dry run:       {dry_run}")
    print(f"{'='*65}\n")

    # Skip already-loaded dates (incremental mode)
    if not dry_run:
        loaded = get_loaded_dates()
        days_to_fetch = [d for d in all_days if d.strftime("%Y-%m-%d") not in loaded]
        print(f"  Already loaded: {len(loaded)} dates")
        print(f"  To fetch:       {len(days_to_fetch)} dates\n")
    else:
        days_to_fetch = all_days[:5]   # dry run: only test first 5 dates
        print(f"  [DRY RUN] Testing first 5 dates only\n")

    if not days_to_fetch:
        print("  ✅ All dates already loaded — nothing to do.\n")
        return

    # Fetch and write
    total_rows   = 0
    ok_dates     = 0
    empty_dates  = 0
    failed_dates = 0
    batch        = []
    BATCH_SIZE   = 50   # write to DB every 50 dates

    for i, day in enumerate(days_to_fetch, 1):
        rows = fetch_date(day)

        if rows:
            ok_dates  += 1
            total_rows += len(rows)
            if dry_run:
                print(f"  [DRY RUN] {day}  →  {len(rows)} indices")
                for r in rows[:3]:   # show first 3
                    print(f"    {r['index_id']:>3} {r['index_name']:<25} close={r['current_value']}")
            else:
                batch.extend(rows)
        else:
            empty_dates += 1
            log.debug("No data for %s (holiday/closed)", day)

        # Progress log every 20 dates
        if i % 20 == 0:
            pct = i / len(days_to_fetch) * 100
            log.info("Progress: %d/%d (%.0f%%) | ok=%d empty=%d rows=%d",
                     i, len(days_to_fetch), pct, ok_dates, empty_dates, total_rows)

        # Batch write to DB
        if not dry_run and len(batch) >= BATCH_SIZE * 17:  # ~17 indices per date
            written = write_to_db(batch)
            log.info("Wrote batch: %d rows to DB", written)
            batch = []

        time.sleep(REQUEST_DELAY)

    # Write remaining batch
    if not dry_run and batch:
        written = write_to_db(batch)
        log.info("Wrote final batch: %d rows to DB", written)

    # Summary
    print(f"\n{'='*65}")
    print(f"  DONE")
    print(f"{'='*65}")
    print(f"  Dates fetched:    {ok_dates}")
    print(f"  Dates empty:      {empty_dates} (holidays/closed)")
    print(f"  Total rows:       {total_rows}")
    if not dry_run:
        print(f"  Written to DB:    ✅")
    print(f"{'='*65}\n")

    if not dry_run:
        print_status()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        print_status()
        sys.exit(0)

    dry_run   = "--dry-run" in args
    from_date = DEFAULT_FROM

    if "--from" in args:
        idx = args.index("--from")
        try:
            from_date = date.fromisoformat(args[idx + 1])
        except (IndexError, ValueError):
            print("ERROR: --from requires a date in YYYY-MM-DD format")
            sys.exit(1)

    run(from_date=from_date, dry_run=dry_run)