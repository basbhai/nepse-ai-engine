"""
nrb_loader.py
Load NRB monthly JSON data into nrb_monthly table.
Usage:
    python nrb_loader.py --file nrb_data.json
    python nrb_loader.py --status
"""

import json
import re
import sys
import logging
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sheets import upsert_row, run_raw_sql

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NRB_LOADER] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


def _extract_nepse_index(text: str) -> float:
    """Extract NEPSE value from string like 'NEPSE stood at 2097.1'"""
    if not text:
        return None
    match = re.search(r'[\d,]+\.?\d*', str(text).replace(',', ''))
    return float(match.group()) if match else None


def _extract_liquidity(text: str) -> float:
    """Extract billion amount from '61.22 billion injected'"""
    if not text:
        return None
    match = re.search(r'[\d.]+', str(text))
    return float(match.group()) if match else None


def _parse_period(period: str) -> tuple:
    """
    Parse 'fy2023/24-month3' → ('FY2023/24', 3, False)
    Parse 'fy2022/23-annual' → ('FY2022/23', None, True)
    """
    period = period.lower()
    fy_match = re.search(r'fy(\d{4}/\d{2,4})', period)
    fy = f"FY{fy_match.group(1)}" if fy_match else ""

    if 'annual' in period:
        return fy, None, True

    month_match = re.search(r'month(\d+)', period)
    month_num = int(month_match.group(1)) if month_match else None
    return fy, month_num, False


def load_nrb_json(filepath: str) -> dict:
    """Load and parse NRB JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def import_to_db(data: dict, dry_run: bool = False) -> int:
    """
    Import all periods from NRB JSON into nrb_monthly table.
    Returns number of records inserted/updated.
    """
    inserted = 0

    for period_key, values in data.items():
        if not values:
            continue

        fy, month_num, is_annual = _parse_period(period_key)

        row = {
            "period":                   period_key,
            "fiscal_year":              fy,
            "month_number":             str(month_num) if month_num else None,
            "is_annual":                str(is_annual).lower(),
            "nepse_index":              str(_extract_nepse_index(
                                            values.get("bop_impact_on_nepse", "")
                                        ) or ""),
            "policy_rate":              str(values.get("policy_rate") or ""),
            "bank_rate":                str(values.get("bank_rate") or ""),
            "crr_percentage":           str(values.get("crr_percentage") or ""),
            "slr_percentage":           str(values.get("slr_percentage") or ""),
            "cpi_inflation":            str(values.get("cpi_inflation") or ""),
            "credit_growth_rate":       str(values.get("credit_growth_rate") or ""),
            "liquidity_injected_billion": str(_extract_liquidity(
                                            values.get("liquidity_status", "")
                                        ) or ""),
            "remittance_yoy_change_pct": str(values.get("remittance_yoy_change_percent") or ""),
            "fx_reserve_months":        str(values.get("fx_reserve_months") or ""),
            "bop_overall_balance_usd_m": str(values.get("bop_overall_balance_usd_million") or ""),
            "bop_current_account_usd_m": str(values.get("bop_current_account_usd_million") or ""),
            "bop_status":               str(values.get("bop_status") or ""),
            "bop_trend":                str(values.get("bop_trend") or ""),
            "overall_sentiment":        str(values.get("overall_sentiment") or ""),
            "forward_guidance":         str(values.get("forward_guidance") or ""),
            "key_risks":                str(values.get("key_risks") or ""),
        }

        # Remove empty strings → None for numeric fields
        numeric_fields = [
            "nepse_index", "policy_rate", "bank_rate", "crr_percentage",
            "slr_percentage", "cpi_inflation", "credit_growth_rate",
            "liquidity_injected_billion", "remittance_yoy_change_pct",
            "fx_reserve_months", "bop_overall_balance_usd_m",
            "bop_current_account_usd_m", "month_number"
        ]
        for field in numeric_fields:
            if row.get(field) == "" or row.get(field) == "None":
                row[field] = None

        if dry_run:
            log.info(
                "[DRY RUN] %s → FY=%s month=%s nepse=%s cpi=%s remit=%s",
                period_key, fy, month_num,
                row["nepse_index"], row["cpi_inflation"],
                row["remittance_yoy_change_pct"]
            )
            inserted += 1
            continue

        ok = upsert_row("nrb_monthly", row, conflict_columns=["period"])
        if ok:
            inserted += 1
            log.info(
                "Upserted: %s | NEPSE=%s | CPI=%s | Remit=%s%%",
                period_key, row["nepse_index"],
                row["cpi_inflation"], row["remittance_yoy_change_pct"]
            )
        else:
            log.warning("Failed to upsert: %s", period_key)

    return inserted


def print_status():
    """Show what's loaded in nrb_monthly table."""
    rows = run_raw_sql("""
        SELECT
            COUNT(*)                    AS total_periods,
            COUNT(DISTINCT fiscal_year) AS fiscal_years,
            MIN(period)                 AS earliest,
            MAX(period)                 AS latest,
            COUNT(nepse_index)          AS has_nepse_index,
            COUNT(cpi_inflation)        AS has_cpi,
            COUNT(remittance_yoy_change_pct) AS has_remittance
        FROM nrb_monthly
        WHERE is_annual = false
    """)
    if rows:
        r = rows[0]
        print(f"\n{'='*55}")
        print(f"  NRB MONTHLY DATA STATUS")
        print(f"{'='*55}")
        print(f"  Total periods:    {r['total_periods']}")
        print(f"  Fiscal years:     {r['fiscal_years']}")
        print(f"  Date range:       {r['earliest']} → {r['latest']}")
        print(f"  Has NEPSE index:  {r['has_nepse_index']}")
        print(f"  Has CPI:          {r['has_cpi']}")
        print(f"  Has Remittance:   {r['has_remittance']}")
        print(f"{'='*55}\n")

        # Show correlation readiness
        print(f"  Correlation readiness (need min 20 data points):")
        print(f"  CPI vs NEPSE:        {'✅' if int(r['has_cpi'] or 0) >= 20 else '⚠️ '} {r['has_cpi']} points")
        print(f"  Remittance vs NEPSE: {'✅' if int(r['has_remittance'] or 0) >= 20 else '⚠️ '} {r['has_remittance']} points")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        print_status()
        sys.exit(0)

    if "--file" in args:
        idx = args.index("--file")
        filepath = args[idx + 1]
        dry_run = "--dry-run" in args

        log.info("Loading NRB data from: %s", filepath)
        data = load_nrb_json(filepath)
        count = import_to_db(data, dry_run=dry_run)

        print(f"\n  {'[DRY RUN] ' if dry_run else ''}Processed {count} periods")
        if not dry_run:
            print_status()
        sys.exit(0)

    print("\nUsage:")
    print("  python nrb_loader.py --file nrb_data.json")
    print("  python nrb_loader.py --file nrb_data.json --dry-run")
    print("  python nrb_loader.py --status\n")
    sys.exit(1)