"""
web_ui/nrb_api.py
=================
NEPSE AI Engine — NRB Monthly Data Entry API

Tiny FastAPI app that receives parsed NRB JSON from the web UI
and upserts it into the nrb_monthly table.

Run:
    uvicorn web_ui.nrb_api:app --port 8765 --reload

Architecture rules:
    from sheets import upsert_row  ← all DB writes
    Never raw psycopg2 here
"""

import logging
import sys
import os
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup so we can import from project root ────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sheets import upsert_row, read_tab

logging.basicConfig(level=logging.INFO, format="%(asctime)s [NRB_API] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")

app = FastAPI(title="NEPSE AI — NRB Entry API", version="1.0.0")

# Allow calls from file:// and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class NRBPayload(BaseModel):
    # Period metadata
    period:                                str
    fiscal_year:                           str
    month_number:                          str
    is_annual:                             str = "false"

    # Monetary Policy
    policy_rate:                           Optional[str] = None
    bank_rate:                             Optional[str] = None
    crr_percentage:                        Optional[str] = None
    slr_percentage:                        Optional[str] = None

    # Interest Rates
    lending_rate_pct:                      Optional[str] = None
    deposit_rate_pct:                      Optional[str] = None
    interbank_rate_pct:                    Optional[str] = None
    tbill_91d_rate_pct:                    Optional[str] = None

    # Inflation & Credit
    cpi_inflation:                         Optional[str] = None
    credit_growth_rate:                    Optional[str] = None
    private_sector_credit_growth_yoy_pct:  Optional[str] = None
    m2_growth_yoy_pct:                     Optional[str] = None
    deposit_growth_yoy_pct:                Optional[str] = None
    npl_ratio_pct:                         Optional[str] = None

    # Liquidity
    liquidity_injected_billion:            Optional[str] = None
    liquidity_status:                      Optional[str] = None

    # Remittance
    remittance_yoy_change_pct:             Optional[str] = None
    remittance_total_billion_npr:          Optional[str] = None

    # Foreign Exchange
    fx_reserve_months:                     Optional[str] = None
    usd_npr_rate:                          Optional[str] = None

    # Balance of Payments
    bop_overall_balance_usd_m:             Optional[str] = None
    bop_current_account_usd_m:             Optional[str] = None
    bop_capital_account_usd_m:             Optional[str] = None
    bop_trade_deficit_usd_m:               Optional[str] = None
    bop_status:                            Optional[str] = None
    bop_trend:                             Optional[str] = None
    bop_impact_on_nepse:                   Optional[str] = None

    # Capital Market
    nepse_index_value:                     Optional[str] = None
    nepse_index:                           Optional[str] = None
    market_cap_billion_npr:                Optional[str] = None

    # Macro Sentiment
    overall_sentiment:                     Optional[str] = None
    forward_guidance:                      Optional[str] = None
    key_risks:                             Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_str(val) -> Optional[str]:
    if val is None:
        return None
    return str(val)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "nrb_api", "time": datetime.now(NST).isoformat()}


@app.get("/nrb/periods")
def get_periods():
    """Return all existing periods so UI can warn about duplicates."""
    try:
        rows = read_tab("nrb_monthly") or []
        # Sort by fiscal_year desc, month_number desc in Python
        rows.sort(key=lambda r: (r.get("fiscal_year",""), int(r.get("month_number") or 0)), reverse=True)
        return {
            "periods": [
                {
                    "period":       r.get("period"),
                    "fiscal_year":  r.get("fiscal_year"),
                    "month_number": r.get("month_number"),
                    "inserted_at":  str(r.get("inserted_at", "")),
                }
                for r in rows
            ]
        }
    except Exception as e:
        log.error("get_periods failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nrb/upsert")
def upsert_nrb(payload: NRBPayload):
    """Upsert one NRB monthly record. Conflict key: period."""
    try:
        row = {
            "period":                               payload.period,
            "fiscal_year":                          payload.fiscal_year,
            "month_number":                         payload.month_number,
            "is_annual":                            payload.is_annual,
            "policy_rate":                          _to_str(payload.policy_rate),
            "bank_rate":                            _to_str(payload.bank_rate),
            "crr_percentage":                       _to_str(payload.crr_percentage),
            "slr_percentage":                       _to_str(payload.slr_percentage),
            "lending_rate_pct":                     _to_str(payload.lending_rate_pct),
            "deposit_rate_pct":                     _to_str(payload.deposit_rate_pct),
            "interbank_rate_pct":                   _to_str(payload.interbank_rate_pct),
            "tbill_91d_rate_pct":                   _to_str(payload.tbill_91d_rate_pct),
            "cpi_inflation":                        _to_str(payload.cpi_inflation),
            "credit_growth_rate":                   _to_str(payload.credit_growth_rate),
            "private_sector_credit_growth_yoy_pct": _to_str(payload.private_sector_credit_growth_yoy_pct),
            "m2_growth_yoy_pct":                    _to_str(payload.m2_growth_yoy_pct),
            "deposit_growth_yoy_pct":               _to_str(payload.deposit_growth_yoy_pct),
            "npl_ratio_pct":                        _to_str(payload.npl_ratio_pct),
            "liquidity_injected_billion":           _to_str(payload.liquidity_injected_billion),
            "liquidity_status":                     _to_str(payload.liquidity_status),
            "remittance_yoy_change_pct":            _to_str(payload.remittance_yoy_change_pct),
            "remittance_total_billion_npr":         _to_str(payload.remittance_total_billion_npr),
            "fx_reserve_months":                    _to_str(payload.fx_reserve_months),
            "usd_npr_rate":                         _to_str(payload.usd_npr_rate),
            "bop_overall_balance_usd_m":            _to_str(payload.bop_overall_balance_usd_m),
            "bop_current_account_usd_m":            _to_str(payload.bop_current_account_usd_m),
            "bop_capital_account_usd_m":            _to_str(payload.bop_capital_account_usd_m),
            "bop_trade_deficit_usd_m":              _to_str(payload.bop_trade_deficit_usd_m),
            "bop_status":                           _to_str(payload.bop_status),
            "bop_trend":                            _to_str(payload.bop_trend),
            "bop_impact_on_nepse":                  _to_str(payload.bop_impact_on_nepse),
            "nepse_index_value":                    _to_str(payload.nepse_index_value),
            "nepse_index":                          _to_str(payload.nepse_index or payload.nepse_index_value),
            "market_cap_billion_npr":               _to_str(payload.market_cap_billion_npr),
            "overall_sentiment":                    _to_str(payload.overall_sentiment),
            "forward_guidance":                     _to_str(payload.forward_guidance),
            "key_risks":                            _to_str(payload.key_risks),
        }

        # Remove None values — don't overwrite existing data with nulls
        row = {k: v for k, v in row.items() if v is not None}

        ok = upsert_row("nrb_monthly", row, conflict_columns=["period"])

        if ok:
            log.info("Upserted NRB record: %s", payload.period)
            return {
                "status":        "success",
                "period":        payload.period,
                "message":       f"NRB data for {payload.period} saved successfully.",
                "fields_written": len(row),
            }
        else:
            raise HTTPException(status_code=500, detail="upsert_row returned False")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("upsert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))