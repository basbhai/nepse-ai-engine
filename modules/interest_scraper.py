"""
interest_scraper.py — NEPSE AI Engine
Fetches FD rates from bankbyaj.com API.
Runs monthly. Feeds nepal_score and capital_allocator.

KEY FIX (April 2026):
    avg_rate_pct in fd_rate_summary = full average across all 224 products
                                      (includes savings, call deposits etc.)
    benchmark_rate                  = average of 1Y+ FDs only (LONG + VLONG tenure)
    FD_RATE_PCT, FD_SCORE_SIGNAL, FD_NEPAL_SCORE in settings
                                    = driven by benchmark_rate NOT avg_rate_pct

Rationale: retail investors compare NEPSE returns against 1-year FD rates,
not savings account rates. Using the full average (4.08%) gives a
misleadingly bullish signal. The benchmark rate (avg of 1Y+ FDs) is
the correct comparison point for "should I be in stocks or FDs?"

Research basis: FD > 9-10% → retail money exits NEPSE (confirmed S5/S6/S7)
"""

import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional

import requests

from config import NST

log = logging.getLogger(__name__)

API_URL = "https://admin.bankbyaj.com/api/v1/category/fd-individual?"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

INSTITUTE_TYPES = {
    "bank":     ["Bank", "BNL", "Rastriya", "Nabil", "Standard", "NIC", "Everest",
                 "Himalayan", "Siddhartha", "Laxmi", "Citizens", "Sunrise", "NMB",
                 "Prabhu", "Sanima", "Kumari", "Global IME", "Machhapuchchhre",
                 "Nepal Investment", "Agriculture", "Nepal Bank", "Mega", "Century",
                 "Prime", "Civil", "Janata"],
    "finance":  ["Finance", "Finans"],
    "dev_bank": ["Development Bank", "Dev Bank", "Bikash Bank"],
}

# Minimum tenure months to count as a "benchmark" FD
# (products a retail investor would realistically compare against stocks)
BENCHMARK_MIN_MONTHS = 12


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fd_rates() -> list[dict]:
    """Fetch all FD products from bankbyaj API. Returns list of raw product dicts."""
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
# SECTION 2 — PARSE AND CLEAN
# ══════════════════════════════════════════════════════════════════════════════

def _parse_interest_rate(raw: str) -> Optional[float]:
    if not raw:
        return None
    cleaned = str(raw).replace("%", "").replace(" ", "").strip()
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _parse_tenure(max_tenure: str, product_name: str) -> tuple[Optional[int], str, str]:
    """
    Parse tenure → (tenure_months, tenure_label, tenure_category).
    Categories: SHORT (1-3m), MEDIUM (4-6m), LONG (7-12m), VLONG (13m+)
    """
    raw = (max_tenure or product_name or "").lower().strip()
    months = None

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

    if months is None:
        for keyword, val in [
            ("2y", 24), ("1y", 12), ("6m", 6), ("3m", 3),
            ("2 year", 24), ("1 year", 12), ("above", 24),
        ]:
            if keyword in raw:
                months = val
                break

    if months is None:
        months = 12

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
    n = name or ""
    for itype, keywords in INSTITUTE_TYPES.items():
        if any(kw.lower() in n.lower() for kw in keywords):
            return itype
    return "other"


def parse_products(raw_products: list[dict], fetch_date: str) -> list[dict]:
    """Clean and enrich raw API products into DB-ready rows."""
    rows = []
    seen = set()

    for p in raw_products:
        name      = str(p.get("name", "")).strip()
        inst_code = str(p.get("institute_code", "")).strip()
        inst_name = str(p.get("institute_name", "")).strip()
        interest  = str(p.get("interest", "")).strip()
        min_bal   = str(p.get("minimum_balance", "0")).strip()
        max_ten   = str(p.get("max_tenure", "")).strip()

        if not inst_code or not interest:
            continue

        dedup_key = f"{fetch_date}_{inst_code}_{name}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        interest_pct             = _parse_interest_rate(interest)
        tenure_months, ten_label, ten_cat = _parse_tenure(max_ten, name)
        inst_type                = _classify_institute(inst_name)

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
# SECTION 3 — SIGNAL FUNCTIONS (unchanged — inputs just come from benchmark now)
# ══════════════════════════════════════════════════════════════════════════════

def _fd_score_signal(rate: float) -> str:
    """
    Translate FD rate into signal label.
    Input: benchmark_rate (avg of 1Y+ FDs), NOT full avg.
    Research: FD > 9-10% → retail exits stocks (S5/S6/S7 confirmed).
    """
    if rate < 7.0:
        return "STRONG_BUY_STOCKS"
    elif rate < 8.0:
        return "BUY_STOCKS"
    elif rate < 9.0:
        return "NEUTRAL"
    elif rate < 10.0:
        return "PREFER_FD"
    else:
        return "STRONG_FD"


def _fd_nepal_score(rate: float) -> int:
    """
    Translate FD rate into nepal_score contribution (-2 to +2).
    Input: benchmark_rate (avg of 1Y+ FDs), NOT full avg.
    """
    if rate < 7.0:
        return +2
    elif rate < 8.0:
        return +1
    elif rate < 9.0:
        return  0
    elif rate < 10.0:
        return -1
    else:
        return -2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPUTE SUMMARY
# KEY FIX: benchmark_rate = avg of 1Y+ products only
#          FD_RATE_PCT and signal driven by benchmark_rate
#          avg_rate_pct stored as-is for historical record
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(rows: list[dict], fetch_date: str, prev_avg: Optional[float]) -> dict:
    """
    Compute FD rate summary.

    Two averages computed:
      avg_rate_pct   — full average across all products (stored in DB, not used for signal)
      benchmark_rate — average of 1Y+ (LONG + VLONG) products only
                       → this drives FD_RATE_PCT, FD_SCORE_SIGNAL, FD_NEPAL_SCORE

    Why: retail investors compare NEPSE against 1-year FD rates.
    Savings account rates (2-3%) drag the full average down to ~4%
    and produce a misleadingly bullish signal.
    """
    # ── Full average (all products) ───────────────────────────────────────────
    all_rates = [
        float(r["interest_pct"])
        for r in rows
        if r.get("interest_pct") and r["interest_pct"] != ""
    ]
    if not all_rates:
        return {}

    avg_rate = round(sum(all_rates) / len(all_rates), 2)
    max_rate = round(max(all_rates), 2)
    min_rate = round(min(all_rates), 2)

    # ── Benchmark rate (1Y+ FDs only — what retail compares against stocks) ───
    benchmark_rows = [
        r for r in rows
        if r.get("interest_pct") and r["interest_pct"] != ""
        and r.get("tenure_months") and r["tenure_months"] != ""
        and int(float(r["tenure_months"])) >= BENCHMARK_MIN_MONTHS
    ]
    benchmark_rates = [float(r["interest_pct"]) for r in benchmark_rows]

    if benchmark_rates:
        benchmark_rate     = round(sum(benchmark_rates) / len(benchmark_rates), 2)
        benchmark_products = len(benchmark_rates)
    else:
        # Fallback to full avg if no 1Y+ products found (shouldn't happen)
        log.warning("No 1Y+ FD products found — falling back to full average for signal")
        benchmark_rate     = avg_rate
        benchmark_products = len(all_rates)

    log.info(
        "FD rates: all_avg=%.2f%% benchmark_avg(1Y+)=%.2f%% "
        "(%d products) max=%.2f%% min=%.2f%%",
        avg_rate, benchmark_rate, benchmark_products, max_rate, min_rate,
    )

    # ── Best bank (highest rate, prefer longer tenure, prefer bank) ───────────
    best_products = sorted(
        [r for r in rows if r.get("interest_pct")],
        key=lambda r: (
            float(r["interest_pct"] or 0),
            int(float(r["tenure_months"] or 0)),
            1 if r["institute_type"] == "bank" else 0,
        ),
        reverse=True,
    )
    best = best_products[0] if best_products else {}

    # ── Rate direction vs previous month (compare benchmark to benchmark) ─────
    rate_vs_prev  = None
    rate_direction = "UNCHANGED"
    if prev_avg is not None:
        # prev_avg from fd_rate_summary is the stored avg_rate_pct (full avg)
        # Direction is still useful directionally even if scale differs
        rate_vs_prev = round(avg_rate - prev_avg, 2)
        if rate_vs_prev > 0.1:
            rate_direction = "RISING"
        elif rate_vs_prev < -0.1:
            rate_direction = "FALLING"
        else:
            rate_direction = "UNCHANGED"

    # ── Signal — from benchmark rate only ────────────────────────────────────
    signal = _fd_score_signal(benchmark_rate)

    log.info(
        "FD Signal: benchmark=%.2f%% → %s (was: full_avg=%.2f%%)",
        benchmark_rate, signal, avg_rate,
    )

    return {
        "fetch_date":         fetch_date,
        "avg_rate_pct":       str(avg_rate),          # full avg — stored for history
        "max_rate_pct":       str(max_rate),
        "min_rate_pct":       str(min_rate),
        "benchmark_rate_pct": str(benchmark_rate),    # 1Y+ avg — drives the signal
        "benchmark_products": str(benchmark_products),
        "best_bank_name":     best.get("institute_name", ""),
        "best_bank_rate":     best.get("interest_pct", ""),
        "best_tenure":        best.get("tenure_label", ""),
        "rate_vs_prev_pct":   str(rate_vs_prev) if rate_vs_prev is not None else "",
        "rate_direction":     rate_direction,
        "fd_score_signal":    signal,
        "total_products":     str(len(rows)),
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


def update_settings(benchmark_rate: float, signal: str,
                    nepal_score_contribution: int) -> None:
    """
    Update settings table.
    FIX: FD_RATE_PCT now stores benchmark_rate (1Y+ avg), not full avg.
    This is what nepal_pulse.py and capital_allocator.py read.
    """
    try:
        from sheets import update_setting
        update_setting("FD_RATE_PCT",         str(benchmark_rate),           set_by="interest_scraper")
        update_setting("FD_SCORE_SIGNAL",     signal,                        set_by="interest_scraper")
        update_setting("FD_NEPAL_SCORE",      str(nepal_score_contribution), set_by="interest_scraper")
        update_setting("FD_RATE_LAST_UPDATED",
                       datetime.now(tz=NST).strftime("%Y-%m-%d"),            set_by="interest_scraper")
        log.info(
            "Settings updated: FD_RATE_PCT=%.2f%% (1Y+ benchmark) "
            "FD_SCORE_SIGNAL=%s FD_NEPAL_SCORE=%+d",
            benchmark_rate, signal, nepal_score_contribution,
        )
    except Exception as exc:
        log.warning("Could not update settings: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GET PREVIOUS MONTH AVG
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
# SECTION 7 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_best_fd_options(
    tenure_category: str = None,
    institute_type:  str = None,
    top_n:           int = 5,
) -> list[dict]:
    """
    Get best FD options from latest fetch.
    Used by capital_allocator.py when recommending FD.
    """
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("SELECT MAX(fetch_date) AS latest FROM fd_rates")
            row = cur.fetchone()
            if not row or not row["latest"]:
                return []
            latest_date = row["latest"]

            conditions = ["fetch_date = %s", "interest_pct != ''",
                          "interest_pct IS NOT NULL"]
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
    Get current FD benchmark rate (1Y+ avg) from settings.
    Fast — reads settings table, no fd_rates query.
    """
    try:
        from sheets import get_setting
        val = get_setting("FD_RATE_PCT", "")
        return float(val) if val else None
    except Exception:
        return None


def get_fd_score_signal() -> str:
    try:
        from sheets import get_setting
        return get_setting("FD_SCORE_SIGNAL", "NEUTRAL")
    except Exception:
        return "NEUTRAL"


def get_fd_nepal_score() -> int:
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

        print(f"\n{'='*70}")
        print(f"  FD RATE HISTORY")
        print(f"{'='*70}")
        print(f"  {'Date':<12} {'AllAvg%':>8} {'Max%':>6} {'Min%':>6} "
              f"{'Direction':<12} {'Signal':<22} {'Products'}")
        print(f"  {'─'*68}")
        for s in summaries:
            print(
                f"  {s['fetch_date']:<12} "
                f"{float(s['avg_rate_pct'] or 0):>8.2f} "
                f"{float(s['max_rate_pct'] or 0):>6.2f} "
                f"{float(s['min_rate_pct'] or 0):>6.2f} "
                f"{(s['rate_direction'] or '')::<12} "
                f"{(s['fd_score_signal'] or '')::<22} "
                f"{s['total_products']}"
            )

        latest = summaries[0]
        bm     = get_current_avg_rate()
        print(f"\n  Best option    : {latest['best_bank_name']} @ "
              f"{latest['best_bank_rate']}% for {latest['best_tenure']}")
        if bm:
            sig   = _fd_score_signal(bm)
            score = _fd_nepal_score(bm)
            print(f"  Benchmark rate : {bm:.2f}% (1Y+ FD avg) → "
                  f"{sig} | nepal_score {score:+d}")
        print(f"{'='*70}\n")

    except Exception as exc:
        print(f"\n  Error reading fd_rate_summary: {exc}\n")


def print_best_rates():
    """Print best FD rates by tenure category."""
    categories = [
        ("SHORT",  "1-3 months"),
        ("MEDIUM", "4-6 months"),
        ("LONG",   "7-12 months"),
        ("VLONG",  "13+ months"),
    ]

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

    bm = get_current_avg_rate()
    if bm:
        score  = _fd_nepal_score(bm)
        signal = _fd_score_signal(bm)
        print(f"\n  Benchmark rate (1Y+ avg): {bm:.2f}%")
        print(f"  Nepal score contribution: {score:+d}")
        print(f"  Signal: {signal}")
        print(f"  Research: FD > 9% = retail money exits NEPSE")
    print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TABLE ENSURE (migrations safety net)
# ══════════════════════════════════════════════════════════════════════════════

def ensure_fd_rates_table() -> bool:
    """Verify fd_rates and fd_rate_summary tables exist."""
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("SELECT 1 FROM fd_rates LIMIT 1")
            cur.execute("SELECT 1 FROM fd_rate_summary LIMIT 1")
        return True
    except Exception as exc:
        log.error("fd_rates table missing: %s — run db.migrations first", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> bool:
    """
    Full FD rate fetch and store pipeline.

    Flow:
      1. Ensure tables exist
      2. Fetch products from bankbyaj API
      3. Parse and clean
      4. Get previous avg rate for direction calculation
      5. Compute summary — benchmark_rate = avg of 1Y+ products
      6. Write to Neon
      7. Update settings: FD_RATE_PCT = benchmark_rate (NOT full avg)
    """
    nst_now    = datetime.now(tz=NST)
    fetch_date = nst_now.strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("INTEREST SCRAPER starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    if not dry_run:
        if not ensure_fd_rates_table():
            return False

    raw_products = fetch_fd_rates()
    if not raw_products:
        log.error("No products fetched — check API or internet connection")
        return False

    rows = parse_products(raw_products, fetch_date)
    if not rows:
        log.error("No valid rows after parsing")
        return False

    prev_avg = get_prev_avg_rate() if not dry_run else None
    summary  = compute_summary(rows, fetch_date, prev_avg)

    if dry_run:
        bm = float(summary.get("benchmark_rate_pct", 0))
        log.info("[DRY RUN] Would write %d rows to fd_rates", len(rows))
        log.info("[DRY RUN] Full avg: %s%% | Benchmark (1Y+): %.2f%% | Signal: %s",
                 summary.get("avg_rate_pct"), bm, summary.get("fd_score_signal"))
        return True

    written = write_fd_rates(rows)
    if not written:
        log.error("Failed to write FD rates to Neon")
        return False

    write_summary(summary)

    # KEY FIX: use benchmark_rate for settings, not avg_rate
    benchmark_rate = float(summary.get("benchmark_rate_pct",
                                       summary.get("avg_rate_pct", 8.5)))
    signal         = summary.get("fd_score_signal", "NEUTRAL")
    np_score       = _fd_nepal_score(benchmark_rate)
    update_settings(benchmark_rate, signal, np_score)

    log.info("✅ FD rates updated successfully")
    log.info("   Products stored   : %d", written)
    log.info("   Full avg rate     : %s%% (all products, stored in DB)",
             summary.get("avg_rate_pct"))
    log.info("   Benchmark rate    : %.2f%% (1Y+ FDs — drives signal)",
             benchmark_rate)
    log.info("   Benchmark products: %s", summary.get("benchmark_products"))
    log.info("   Max rate          : %s%%", summary.get("max_rate_pct"))
    log.info("   Best option       : %s @ %s%% for %s",
             summary.get("best_bank_name"),
             summary.get("best_bank_rate"),
             summary.get("best_tenure"))
    log.info("   Rate direction    : %s", summary.get("rate_direction"))
    log.info("   FD signal         : %s (from benchmark)", signal)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.interest_scraper              → fetch + store
#   python -m modules.interest_scraper --dry-run    → fetch + print, no DB write
#   python -m modules.interest_scraper --status     → show history from DB
#   python -m modules.interest_scraper --best       → show best rates by tenure
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args    = sys.argv[1:]
    dry_run = "--dry-run" in args

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