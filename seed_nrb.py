"""
seed_nrb.py — One-time NRB historical data loader
─────────────────────────────────────────────────
Loads NRB data from nrb_data.json into the nrb_monthly table in Neon.
Run once: python seed_nrb.py

After this, history_bootstrap.py / NotebookLM handles future months.
"""

import json
import re
import logging
from sheets import upsert_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Load data from JSON file ────────────────────────────────────────────────

try:
    with open("nrb_data.json", "r", encoding="utf-8") as f:
        NRB_DATA = json.load(f)
    log.info("Successfully loaded nrb_data.json")
except FileNotFoundError:
    log.error("nrb_data.json not found in current directory")
    exit(1)
except json.JSONDecodeError:
    log.error("Invalid JSON in nrb_data.json")
    exit(1)

# ── Parse period key → fiscal_year + month_number + is_annual ────────────────

def parse_period(period: str) -> dict:
    """
    'fy2025/26-month3'  → fiscal_year='FY2025/26', month_number=3,  is_annual=False
    'fy2024/25-annual'  → fiscal_year='FY2024/25', month_number=None, is_annual=True
    """
    m = re.match(r'fy(\d{4}/\d{2})-(month(\d+)|annual)', period)
    if not m:
        return {}
    fy        = f"FY{m.group(1)}"
    is_annual = m.group(2) == "annual"
    month_num = int(m.group(3)) if m.group(3) else None
    return {"fiscal_year": fy, "month_number": month_num, "is_annual": is_annual}

# ── Extract NEPSE index from bop_impact string ────────────────────────────────

def extract_nepse(impact_str) -> float | None:
    if not impact_str:
        return None
    m = re.search(r'[\d,]+\.?\d*', impact_str)
    return float(m.group().replace(',', '')) if m else None

# ── Main seed ─────────────────────────────────────────────────────────────────

def seed():
    ok, skipped = 0, 0

    for period, raw in NRB_DATA.items():
        meta = parse_period(period)
        if not meta:
            log.warning("Could not parse period key: %s", period)
            skipped += 1
            continue

        nepse_index = extract_nepse(raw.get("bop_impact_on_nepse"))

        row = {
            "period":                     period,
            "fiscal_year":                meta["fiscal_year"],
            "month_number":               str(meta["month_number"]) if meta["month_number"] else None,
            "is_annual":                  "true" if meta["is_annual"] else "false",
            "nepse_index":                str(nepse_index) if nepse_index is not None else None,
            "policy_rate":                str(raw["policy_rate"]) if raw.get("policy_rate") is not None else None,
            "bank_rate":                  str(raw["interest_rate"]) if raw.get("interest_rate") is not None else None,
            "crr_percentage":             str(raw["crr_percentage"]) if raw.get("crr_percentage") is not None else None,
            "slr_percentage":             str(raw["slr_percentage"]) if raw.get("slr_percentage") is not None else None,
            "cpi_inflation":              str(raw["cpi_inflation"]) if raw.get("cpi_inflation") is not None else None,
            "credit_growth_rate":         str(raw["credit_growth_rate"]) if raw.get("credit_growth_rate") is not None else None,
            "liquidity_injected_billion": raw.get("liquidity_status"),   # keep as string
            "remittance_yoy_change_pct":  str(raw["remittance_yoy_change_percent"]) if raw.get("remittance_yoy_change_percent") is not None else None,
            "fx_reserve_months":          str(raw["fx_reserve_months"]) if raw.get("fx_reserve_months") is not None else None,
            "bop_overall_balance_usd_m":  str(raw["bop_overall_balance_usd_million"]) if raw.get("bop_overall_balance_usd_million") is not None else None,
            "bop_current_account_usd_m":  str(raw["bop_current_account_usd_million"]) if raw.get("bop_current_account_usd_million") is not None else None,
            "bop_status":                 raw.get("bop_status"),
            "bop_trend":                  raw.get("bop_trend"),
            "overall_sentiment":          raw.get("overall_sentiment"),
            "forward_guidance":           raw.get("forward_guidance"),
            "key_risks":                  raw.get("key_risks"),
        }

        # Remove None values — avoid overwriting existing data with nulls
        row = {k: v for k, v in row.items() if v is not None}

        success = upsert_row("nrb_monthly", row, conflict_columns=["period"])
        if success:
            ok += 1
            log.info("  ✅ %s", period)
        else:
            skipped += 1
            log.warning("  ❌ %s — upsert failed", period)

    print(f"\n  Seeded: {ok}  |  Failed/Skipped: {skipped}\n")


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    seed()