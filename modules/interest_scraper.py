"""
interest_scraper.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Fetch current Fixed Deposit rates from bankbyaj.com API.
          Store in fd_rates table in Neon.
          Feed two downstream consumers:

          1. nepal_pulse.py / capital_allocator.py — fd_rate_level scoring
             Research basis (macro papers synthesis):
               FD rate < 7%   → stocks clearly better → +2 to nepal_score
               FD rate 7-8%   → stocks slightly better → +1
               FD rate 8-9%   → roughly equal after fees → 0
               FD rate 9-10%  → FD becoming attractive → -1
               FD rate > 10%  → retail money leaves NEPSE → -2

          2. capital_allocator.py — FD recommendation engine
             When market is BEAR/SIDEWAYS, Claude recommends specific
             bank + tenure based on current best rates in this table.

Source   : https://bankbyaj.com (Nepal FD rate aggregator)
API      : https://admin.bankbyaj.com/api/v1/category/fd-individual
Schedule : Monthly via GitHub Actions (rates change slowly)
           Can also run manually anytime

─────────────────────────────────────────────────────────────────────────────

SOP:
  python interest_scraper.py              → fetch + store + print summary
  python interest_scraper.py --dry-run    → fetch + print, no DB write
  python interest_scraper.py --status     → show latest rates from DB
  python interest_scraper.py --best       → print best rates by tenure
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FD_SCRAPER] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

API_URL = "https://admin.bankbyaj.com/api/v1/category/fd-individual?"

HEADERS = {
    "Accept":             "application/json, text/plain, */*",
    "Accept-Encoding":    "gzip, deflate, br, zstd",
    "Accept-Language":    "en-GB,en-US;q=0.9,en;q=0.8",
    "Connection":         "keep-alive",
    "Host":               "admin.bankbyaj.com",
    "Origin":             "https://bankbyaj.com",
    "Referer":            "https://bankbyaj.com/",
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-site",
    "User-Agent":         (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua":          '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# ── Institute type classification ─────────────────────────────────────────────
# Used to categorize bank type for capital_allocator recommendations
INSTITUTE_TYPES = {
    "bank":     ["Bank", "Banque", "BANK"],
    "finance":  ["Finance", "Fin.", "FINANCE", "Financial"],
    "dev_bank": ["Development", "Dev.", "Dev Bank"],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENSURE TABLE EXISTS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_fd_rates_table() -> bool:
    """
    Create fd_rates table if it doesn't exist.
    Also creates fd_rate_summary for the monthly aggregated view.
    """
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fd_rates (
                    id               SERIAL PRIMARY KEY,
                    fetch_date       TEXT NOT NULL,
                    institute_code   TEXT,
                    institute_name   TEXT,
                    product_name     TEXT,
                    interest_rate    TEXT,
                    interest_pct     TEXT,
                    tenure_label     TEXT,
                    tenure_months    TEXT,
                    tenure_category  TEXT,
                    minimum_balance  TEXT,
                    institute_type   TEXT,
                    source           TEXT DEFAULT 'bankbyaj',
                    inserted_at      TIMESTAMPTZ DEFAULT NOW(),
                    CONSTRAINT ux_fd_rates_date_inst_product
                        UNIQUE (fetch_date, institute_code, product_name)
                );
                CREATE INDEX IF NOT EXISTS ix_fd_rates_fetch_date
                    ON fd_rates (fetch_date);
                CREATE INDEX IF NOT EXISTS ix_fd_rates_tenure_category
                    ON fd_rates (tenure_category);
                CREATE INDEX IF NOT EXISTS ix_fd_rates_interest_pct
                    ON fd_rates (interest_pct);

                CREATE TABLE IF NOT EXISTS fd_rate_summary (
                    id               SERIAL PRIMARY KEY,
                    fetch_date       TEXT NOT NULL UNIQUE,
                    avg_rate_pct     TEXT,
                    max_rate_pct     TEXT,
                    min_rate_pct     TEXT,
                    best_bank_name   TEXT,
                    best_bank_rate   TEXT,
                    best_tenure      TEXT,
                    rate_vs_prev_pct TEXT,
                    rate_direction   TEXT,
                    fd_score_signal  TEXT,
                    total_products   TEXT,
                    inserted_at      TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        log.info("fd_rates and fd_rate_summary tables ready")
        return True
    except Exception as exc:
        log.error("Failed to create fd_rates table: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FETCH FROM API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fd_rates() -> list[dict]:
    """
    Fetch all FD products from bankbyaj API.
    Returns list of raw product dicts.
    """
    log.info("Fetching FD rates from bankbyaj.com...")
    try:
        session  = requests.Session()
        response = session.get(API_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data     = response.json()
        products = data.get("products", [])
        log.info("Fetched %d FD products", len(products))
        return products
    except requests.exceptions.Timeout:
        log.error("bankbyaj API timed out")
        return []
    except Exception as exc:
        log.error("Failed to fetch FD rates: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PARSE AND CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def _parse_interest_rate(raw: str) -> Optional[float]:
    """
    Extract float from strings like '7.00%', '7.5 %', '8%'.
    Returns None if unparseable.
    """
    if not raw:
        return None
    cleaned = str(raw).replace("%", "").replace(" ", "").strip()
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _parse_tenure(max_tenure: str, product_name: str) -> tuple[Optional[int], str, str]:
    """
    Parse tenure from strings like:
      '2 years', '1 year', '6 months', '3 months', '90 days'
    Returns (tenure_months: int, tenure_label: str, tenure_category: str)

    Categories:
      SHORT   = 1-3 months
      MEDIUM  = 4-6 months
      LONG    = 7-12 months
      VLONG   = 13+ months
    """
    raw = (max_tenure or product_name or "").lower().strip()

    months = None

    # Try to extract from string
    if "year" in raw:
        m = re.search(r'(\d+\.?\d*)\s*year', raw)
        if m:
            months = int(float(m.group(1)) * 12)
    elif "month" in raw:
        m = re.search(r'(\d+)\s*month', raw)
        if m:
            months = int(m.group(1))
    elif "day" in raw:
        m = re.search(r'(\d+)\s*day', raw)
        if m:
            months = max(1, int(m.group(1)) // 30)

    # Also check product name for clues
    if months is None:
        for keyword, val in [
            ("2y", 24), ("1y", 12), ("6m", 6), ("3m", 3),
            ("2 year", 24), ("1 year", 12), ("above", 24),
        ]:
            if keyword in raw:
                months = val
                break

    if months is None:
        months = 12   # default assumption

    # Category
    if months <= 3:
        category = "SHORT"
    elif months <= 6:
        category = "MEDIUM"
    elif months <= 12:
        category = "LONG"
    else:
        category = "VLONG"

    label = max_tenure or f"{months} months"

    return months, label, category


def _classify_institute(name: str) -> str:
    """Classify institute as bank, finance, dev_bank, or other."""
    n = name or ""
    for itype, keywords in INSTITUTE_TYPES.items():
        if any(kw.lower() in n.lower() for kw in keywords):
            return itype
    return "other"


def parse_products(raw_products: list[dict], fetch_date: str) -> list[dict]:
    """
    Clean and enrich raw API products into DB-ready rows.
    """
    rows = []
    seen = set()   # dedup within same fetch

    for p in raw_products:
        name      = str(p.get("name", "")).strip()
        inst_code = str(p.get("institute_code", "")).strip()
        inst_name = str(p.get("institute_name", "")).strip()
        interest  = str(p.get("interest", "")).strip()
        min_bal   = str(p.get("minimum_balance", "0")).strip()
        max_ten   = str(p.get("max_tenure", "")).strip()

        if not inst_code or not interest:
            continue

        # Dedup key
        dedup_key = f"{fetch_date}_{inst_code}_{name}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        interest_pct              = _parse_interest_rate(interest)
        tenure_months, ten_label, ten_cat = _parse_tenure(max_ten, name)
        inst_type                 = _classify_institute(inst_name)

        rows.append({
            "fetch_date":      fetch_date,
            "institute_code":  inst_code,
            "institute_name":  inst_name,
            "product_name":    name,
            "interest_rate":   interest,
            "interest_pct":    str(interest_pct) if interest_pct is not None else "",
            "tenure_label":    ten_label,
            "tenure_months":   str(tenure_months),
            "tenure_category": ten_cat,
            "minimum_balance": min_bal,
            "institute_type":  inst_type,
            "source":          "bankbyaj",
        })

    log.info("Parsed %d valid FD products", len(rows))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPUTE FD SCORE SIGNAL
# Evidence: macro papers synthesis
#   FD rate < 7%   → stocks clearly better → signal STRONG_BUY_STOCKS
#   FD rate 7-8%   → stocks slightly better → signal BUY_STOCKS
#   FD rate 8-9%   → roughly equal → signal NEUTRAL
#   FD rate 9-10%  → FD becoming attractive → signal PREFER_FD
#   FD rate > 10%  → retail money exits NEPSE → signal STRONG_FD
# ══════════════════════════════════════════════════════════════════════════════

def _fd_score_signal(avg_rate: float) -> str:
    """
    Translate average FD rate into a signal label for capital_allocator.
    Based on research paper finding: FD > 9-10% → retail exits stocks.
    """
    if avg_rate < 7.0:
        return "STRONG_BUY_STOCKS"
    elif avg_rate < 8.0:
        return "BUY_STOCKS"
    elif avg_rate < 9.0:
        return "NEUTRAL"
    elif avg_rate < 10.0:
        return "PREFER_FD"
    else:
        return "STRONG_FD"


def _fd_nepal_score(avg_rate: float) -> int:
    """
    Translate FD rate into nepal_score contribution (-2 to +2).
    Used by nepal_pulse.py fd_rate_level scoring.
    """
    if avg_rate < 7.0:
        return +2
    elif avg_rate < 8.0:
        return +1
    elif avg_rate < 9.0:
        return  0
    elif avg_rate < 10.0:
        return -1
    else:
        return -2


def compute_summary(rows: list[dict], fetch_date: str, prev_avg: Optional[float]) -> dict:
    """
    Compute summary statistics from parsed rows.
    Includes MoM rate direction — key for capital_allocator decision.
    """
    rates = [
        float(r["interest_pct"])
        for r in rows
        if r.get("interest_pct") and r["interest_pct"] != ""
    ]

    if not rates:
        return {}

    avg_rate = round(sum(rates) / len(rates), 2)
    max_rate = round(max(rates), 2)
    min_rate = round(min(rates), 2)

    # Best bank — highest rate, prefer longer tenure, prefer commercial bank
    best_products = sorted(
        [r for r in rows if r.get("interest_pct")],
        key=lambda r: (
            float(r["interest_pct"] or 0),
            int(r["tenure_months"] or 0),
            1 if r["institute_type"] == "bank" else 0,
        ),
        reverse=True,
    )
    best = best_products[0] if best_products else {}

    # Rate direction vs previous month
    rate_vs_prev = None
    rate_direction = "UNCHANGED"
    if prev_avg is not None:
        rate_vs_prev  = round(avg_rate - prev_avg, 2)
        if rate_vs_prev > 0.1:
            rate_direction = "RISING"    # FD rates rising = money leaving stocks
        elif rate_vs_prev < -0.1:
            rate_direction = "FALLING"   # FD rates falling = money returning to stocks
        else:
            rate_direction = "UNCHANGED"

    signal = _fd_score_signal(avg_rate)

    log.info(
        "FD Summary: avg=%.2f%% max=%.2f%% min=%.2f%% direction=%s signal=%s",
        avg_rate, max_rate, min_rate, rate_direction, signal,
    )

    return {
        "fetch_date":      fetch_date,
        "avg_rate_pct":    str(avg_rate),
        "max_rate_pct":    str(max_rate),
        "min_rate_pct":    str(min_rate),
        "best_bank_name":  best.get("institute_name", ""),
        "best_bank_rate":  best.get("interest_pct", ""),
        "best_tenure":     best.get("tenure_label", ""),
        "rate_vs_prev_pct": str(rate_vs_prev) if rate_vs_prev is not None else "",
        "rate_direction":  rate_direction,
        "fd_score_signal": signal,
        "total_products":  str(len(rows)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATABASE WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_fd_rates(rows: list[dict]) -> int:
    """Bulk upsert FD rate rows into fd_rates table."""
    if not rows:
        return 0
    try:
        import psycopg2.extras
        from db.connection import _db

        columns = [
            "fetch_date", "institute_code", "institute_name", "product_name",
            "interest_rate", "interest_pct", "tenure_label", "tenure_months",
            "tenure_category", "minimum_balance", "institute_type", "source",
        ]
        col_sql = ", ".join(f'"{c}"' for c in columns)
        val_sql = ", ".join(["%s"] * len(columns))
        upd_sql = ", ".join(
            f'"{c}" = EXCLUDED."{c}"'
            for c in columns if c not in ("fetch_date", "institute_code", "product_name")
        )
        values = [tuple(r.get(c, "") for c in columns) for r in rows]

        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO fd_rates ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT (fetch_date, institute_code, product_name)
                DO UPDATE SET {upd_sql}
                """,
                values,
                page_size=200,
            )
        log.info("Wrote %d FD rate rows to Neon", len(rows))
        return len(rows)
    except Exception as exc:
        log.error("write_fd_rates failed: %s", exc)
        return 0


def write_summary(summary: dict) -> bool:
    """Write monthly summary to fd_rate_summary table."""
    if not summary:
        return False
    try:
        from db.connection import _db
        columns = list(summary.keys())
        col_sql = ", ".join(f'"{c}"' for c in columns)
        val_sql = ", ".join(["%s"] * len(columns))
        upd_sql = ", ".join(
            f'"{c}" = EXCLUDED."{c}"'
            for c in columns if c != "fetch_date"
        )
        values = tuple(str(summary[c]) for c in columns)
        with _db() as cur:
            cur.execute(
                f"""
                INSERT INTO fd_rate_summary ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT (fetch_date)
                DO UPDATE SET {upd_sql}
                """,
                values,
            )
        log.info("FD summary written for %s", summary.get("fetch_date"))
        return True
    except Exception as exc:
        log.error("write_summary failed: %s", exc)
        return False


def update_settings(avg_rate: float, signal: str, nepal_score_contribution: int) -> None:
    """
    Update settings table so other modules can read current FD rate instantly
    without querying fd_rates table.
    Also updates FD_RATE_PCT used by capital_allocator.py.
    """
    try:
        from sheets import update_setting
        update_setting("FD_RATE_PCT",            str(avg_rate),              set_by="interest_scraper")
        update_setting("FD_SCORE_SIGNAL",         signal,                     set_by="interest_scraper")
        update_setting("FD_NEPAL_SCORE",          str(nepal_score_contribution), set_by="interest_scraper")
        update_setting("FD_RATE_LAST_UPDATED",    datetime.now(tz=NST).strftime("%Y-%m-%d"), set_by="interest_scraper")
        log.info(
            "Settings updated: FD_RATE_PCT=%.2f%% FD_SCORE_SIGNAL=%s FD_NEPAL_SCORE=%+d",
            avg_rate, signal, nepal_score_contribution,
        )
    except Exception as exc:
        log.warning("Could not update settings: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GET PREVIOUS MONTH AVG (for direction calculation)
# ══════════════════════════════════════════════════════════════════════════════

def get_prev_avg_rate() -> Optional[float]:
    """Get previous month's average FD rate from fd_rate_summary."""
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("""
                SELECT avg_rate_pct
                FROM fd_rate_summary
                ORDER BY fetch_date DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            if row and row["avg_rate_pct"]:
                return float(row["avg_rate_pct"])
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC API (for capital_allocator and nepal_pulse)
# ══════════════════════════════════════════════════════════════════════════════

def get_best_fd_options(
    tenure_category: str = None,
    institute_type:  str = None,
    top_n:           int = 5,
) -> list[dict]:
    """
    Get best FD options from latest fetch, optionally filtered.
    Used by capital_allocator.py when recommending FD.

    Args:
        tenure_category: SHORT / MEDIUM / LONG / VLONG or None for all
        institute_type:  bank / finance / dev_bank or None for all
        top_n:           how many to return

    Returns:
        List of dicts with institute_name, interest_pct, tenure_label,
        institute_type, product_name — sorted by interest_pct desc

    Example:
        # Get top 5 commercial bank FDs for 1 year
        options = get_best_fd_options(tenure_category="LONG", institute_type="bank")
    """
    try:
        from db.connection import _db
        with _db() as cur:
            # Get latest fetch date
            cur.execute("SELECT MAX(fetch_date) AS latest FROM fd_rates")
            row = cur.fetchone()
            if not row or not row["latest"]:
                return []
            latest_date = row["latest"]

            # Build query
            conditions = ["fetch_date = %s", "interest_pct != ''", "interest_pct IS NOT NULL"]
            params     = [latest_date]

            if tenure_category:
                conditions.append("tenure_category = %s")
                params.append(tenure_category)
            if institute_type:
                conditions.append("institute_type = %s")
                params.append(institute_type)

            where = " AND ".join(conditions)
            cur.execute(
                f"""
                SELECT institute_name, institute_code, institute_type,
                       product_name, interest_pct, tenure_label,
                       tenure_months, minimum_balance
                FROM fd_rates
                WHERE {where}
                ORDER BY interest_pct::float DESC
                LIMIT %s
                """,
                params + [top_n],
            )
            return [dict(r) for r in cur.fetchall()]

    except Exception as exc:
        log.error("get_best_fd_options failed: %s", exc)
        return []


def get_current_avg_rate() -> Optional[float]:
    """
    Get current average FD rate. Used by nepal_pulse.py and capital_allocator.py.
    Reads from settings table (fast) rather than querying fd_rates.
    """
    try:
        from sheets import get_setting
        val = get_setting("FD_RATE_PCT", "")
        return float(val) if val else None
    except Exception:
        return None


def get_fd_score_signal() -> str:
    """Get current FD score signal. Used by nepal_pulse.py."""
    try:
        from sheets import get_setting
        return get_setting("FD_SCORE_SIGNAL", "NEUTRAL")
    except Exception:
        return "NEUTRAL"


def get_fd_nepal_score() -> int:
    """
    Get current FD rate contribution to nepal_score (-2 to +2).
    Called by nepal_pulse.py _compute_nepal_score().
    """
    try:
        from sheets import get_setting
        val = get_setting("FD_NEPAL_SCORE", "0")
        return int(val) if val else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — STATUS / DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_status():
    """Show current FD rate coverage and latest summary."""
    try:
        from db.connection import _db
        with _db() as cur:
            # Summary table
            cur.execute("""
                SELECT fetch_date, avg_rate_pct, max_rate_pct, min_rate_pct,
                       best_bank_name, best_bank_rate, best_tenure,
                       rate_direction, fd_score_signal, total_products
                FROM fd_rate_summary
                ORDER BY fetch_date DESC
                LIMIT 6
            """)
            summaries = [dict(r) for r in cur.fetchall()]

        if not summaries:
            print("\n  fd_rate_summary is EMPTY. Run: python interest_scraper.py\n")
            return

        print(f"\n{'='*65}")
        print(f"  FD RATE HISTORY")
        print(f"{'='*65}")
        print(f"  {'Date':<12} {'Avg%':>6} {'Max%':>6} {'Min%':>6} "
              f"{'Direction':<12} {'Signal':<20} {'Products'}")
        print(f"  {'─'*62}")
        for s in summaries:
            print(
                f"  {s['fetch_date']:<12} "
                f"{float(s['avg_rate_pct'] or 0):>6.2f} "
                f"{float(s['max_rate_pct'] or 0):>6.2f} "
                f"{float(s['min_rate_pct'] or 0):>6.2f} "
                f"{(s['rate_direction'] or '')::<12} "
                f"{(s['fd_score_signal'] or '')::<20} "
                f"{s['total_products']}"
            )

        latest = summaries[0]
        print(f"\n  Best option: {latest['best_bank_name']} @ "
              f"{latest['best_bank_rate']}% for {latest['best_tenure']}")
        print(f"{'='*65}\n")

    except Exception as exc:
        print(f"\n  Error reading fd_rate_summary: {exc}\n")


def print_best_rates():
    """Print best FD rates by tenure category."""
    categories = [("SHORT", "1-3 months"), ("MEDIUM", "4-6 months"),
                  ("LONG",  "7-12 months"), ("VLONG", "13+ months")]

    print(f"\n{'='*65}")
    print(f"  BEST FD RATES BY TENURE")
    print(f"{'='*65}")

    for cat, label in categories:
        options = get_best_fd_options(tenure_category=cat, top_n=3)
        if not options:
            continue
        print(f"\n  {label} ({cat}):")
        print(f"  {'Institution':<30} {'Rate%':>6}  {'Tenure':<15} {'Type'}")
        print(f"  {'─'*62}")
        for o in options:
            print(
                f"  {o['institute_name']:<30} "
                f"{float(o['interest_pct'] or 0):>6.2f}%  "
                f"{o['tenure_label']:<15} "
                f"{o['institute_type']}"
            )

    # Nepal score implication
    avg = get_current_avg_rate()
    if avg:
        score = _fd_nepal_score(avg)
        signal = _fd_score_signal(avg)
        print(f"\n  Current avg rate: {avg:.2f}%")
        print(f"  Nepal score contribution: {score:+d}")
        print(f"  Signal: {signal}")
        print(f"  Research basis: FD > 9% = retail money exits NEPSE (macro papers)")

    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> bool:
    """
    Full FD rate fetch and store pipeline.

    Flow:
      1. Ensure tables exist
      2. Fetch products from bankbyaj API
      3. Parse and clean
      4. Get previous avg rate for direction calculation
      5. Compute summary + FD score signal
      6. Write to Neon
      7. Update settings table (FD_RATE_PCT, FD_SCORE_SIGNAL)

    Returns True on success.
    """
    nst_now    = datetime.now(tz=NST)
    fetch_date = nst_now.strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("INTEREST SCRAPER starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    if not dry_run:
        if not ensure_fd_rates_table():
            return False

    # Fetch
    raw_products = fetch_fd_rates()
    if not raw_products:
        log.error("No products fetched — check API or internet connection")
        return False

    # Parse
    rows = parse_products(raw_products, fetch_date)
    if not rows:
        log.error("No valid rows after parsing")
        return False

    # Previous avg for direction
    prev_avg = get_prev_avg_rate() if not dry_run else None

    # Summary
    summary = compute_summary(rows, fetch_date, prev_avg)

    if dry_run:
        log.info("[DRY RUN] Would write %d rows to fd_rates", len(rows))
        log.info("[DRY RUN] Summary: %s", summary)
        return True

    # Write
    written = write_fd_rates(rows)
    if not written:
        log.error("Failed to write FD rates to Neon")
        return False

    write_summary(summary)

    # Update settings for fast access by other modules
    if summary.get("avg_rate_pct"):
        avg_rate  = float(summary["avg_rate_pct"])
        signal    = summary.get("fd_score_signal", "NEUTRAL")
        np_score  = _fd_nepal_score(avg_rate)
        update_settings(avg_rate, signal, np_score)

    log.info("✅ FD rates updated successfully")
    log.info("   Products stored : %d", written)
    log.info("   Avg rate        : %s%%", summary.get("avg_rate_pct"))
    log.info("   Max rate        : %s%%", summary.get("max_rate_pct"))
    log.info("   Best option     : %s @ %s%% for %s",
             summary.get("best_bank_name"),
             summary.get("best_bank_rate"),
             summary.get("best_tenure"))
    log.info("   Rate direction  : %s", summary.get("rate_direction"))
    log.info("   FD signal       : %s", summary.get("fd_score_signal"))

    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python interest_scraper.py              → fetch + store
#   python interest_scraper.py --dry-run    → fetch + print, no DB write
#   python interest_scraper.py --status     → show history from DB
#   python interest_scraper.py --best       → show best rates by tenure
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args     = sys.argv[1:]
    dry_run  = "--dry-run" in args

    if "--status" in args:
        print_status()
        sys.exit(0)

    if "--best" in args:
        print_best_rates()
        sys.exit(0)

    success = run(dry_run=dry_run)

    if success and not dry_run:
        print_best_rates()

    sys.exit(0 if success else 1)
