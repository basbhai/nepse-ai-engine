"""
meroshare.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Fetch live portfolio combining two working APIs:
          1. Meroshare waccReport  → symbols, shares, WACC, total cost
          2. TMS clientPortfolio   → live prices, daily change

Auth:
  - Meroshare: POST to auth URL → token in response header
  - TMS: Gemini captcha login (tms_scraper.get_session)

SOP:
  python meroshare.py         → full sync, write to Neon
  python meroshare.py status  → read from Neon only (no API call)

CREDENTIALS NEEDED IN .env:
  MEROSHARE_USERNAME=
  MEROSHARE_PASSWORD=
  MEROSHARE_DP_ID=
  MEROSHARE_DEMAT=1301180000232764
  TMS_CLIENT_ID=2181770
  TMS_REQUEST_OWNER=109268
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
load_dotenv()

from sheets import read_tab, upsert_row
from modules.tms_scraper import get_session, BASE_HEADERS, TMS_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MEROSHARE] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST           = timezone(timedelta(hours=5, minutes=45))
DEMAT         = os.getenv("MEROSHARE_DEMAT", "1301180000232764")
TMS_CLIENT_ID = os.getenv("TMS_CLIENT_ID", "2181770")
REQUEST_OWNER = os.getenv("TMS_REQUEST_OWNER", "109268")
MEMBER_CODE   = "49"

AUTH_URL      = "https://webbackend.cdsc.com.np/api/meroShare/auth/"
WACC_URL      = "https://webbackend.cdsc.com.np/api/myPurchase/waccReport/"
PORTFOLIO_URL = f"{TMS_BASE}/tmsapi/rtApi/ws/clientPortfolio/{TMS_CLIENT_ID}"

MEROSHARE_HEADERS = {
    "Accept":        "application/json, text/plain, */*",
    "Content-Type":  "application/json",
    "Origin":        "https://meroshare.cdsc.com.np",
    "Referer":       "https://meroshare.cdsc.com.np/",
    "Connection":    "keep-alive",
    "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Holding:
    symbol:        str
    company:       str   = ""
    shares:        int   = 0
    wacc:          float = 0.0   # from waccReport
    total_cost:    float = 0.0   # from waccReport (totalCost)
    current_price: float = 0.0   # lastTradePrice from TMS
    prev_close:    float = 0.0   # closePrice from TMS
    current_value: float = 0.0   # shares × current_price
    pnl_npr:       float = 0.0   # current_value - total_cost
    pnl_pct:       float = 0.0   # (current_price - wacc) / wacc × 100
    day_change_pct:float = 0.0   # perChange from TMS (daily)
    high:          float = 0.0
    low:           float = 0.0
    source:        str   = "tms+meroshare"
    timestamp:     str   = field(default_factory=lambda:
                           datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class PortfolioSummary:
    total_holdings:     int   = 0
    total_cost_npr:     float = 0.0
    total_value_npr:    float = 0.0
    total_pnl_npr:      float = 0.0
    total_pnl_pct:      float = 0.0
    holdings:           list  = field(default_factory=list)
    timestamp:          str   = field(default_factory=lambda:
                                 datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MEROSHARE AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _get_token() -> Optional[str]:
    payload = {
        "clientId": os.getenv("MEROSHARE_DP_ID"),
        "username": os.getenv("MEROSHARE_USERNAME"),
        "password": os.getenv("MEROSHARE_PASSWORD"),
    }
    try:
        r = requests.post(AUTH_URL, json=payload,
                          headers=MEROSHARE_HEADERS, timeout=15)
        if r.status_code != 200:
            log.error("Meroshare login failed: %d — %s", r.status_code, r.text[:200])
            return None
        token = r.headers.get("Authorization")
        if not token:
            log.error("Token not in response headers")
            return None
        log.info("Meroshare login OK")
        return token
    except Exception as e:
        log.error("Meroshare login error: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WACC REPORT (symbols + shares + cost)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_wacc_report(token: str) -> dict[str, dict]:
    """
    Returns dict keyed by symbol:
    { "NABIL": { wacc, shares, total_cost }, ... }
    """
    headers = {**MEROSHARE_HEADERS, "Authorization": token}
    try:
        r = requests.post(WACC_URL, headers=headers,
                          json={"demat": DEMAT}, timeout=15)
        if r.status_code != 200:
            log.error("waccReport failed: %d", r.status_code)
            return {}

        data     = r.json()
        records  = data.get("waccReportResponse", [])
        result   = {}

        for rec in records:
            symbol = rec.get("scrip", "").upper().strip()
            if not symbol:
                continue
            result[symbol] = {
                "wacc":       float(rec.get("averageBuyRate", 0)),
                "shares":     int(rec.get("totalQuantity", 0)),
                "total_cost": float(rec.get("totalCost", 0)),
            }

        log.info("waccReport: %d positions", len(result))
        return result

    except Exception as e:
        log.error("waccReport error: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TMS CLIENT PORTFOLIO (live prices)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_tms_portfolio() -> dict[str, dict]:
    """
    Returns dict keyed by symbol:
    { "NABIL": { current_price, prev_close, day_change_pct, high, low, company }, ... }
    """
    try:
        session, host_sid = get_session()

        headers = {
            **BASE_HEADERS,
            "Referer":         f"{TMS_BASE}/tms/client/dashboard",
            "Membercode":      MEMBER_CODE,
            "Request-Owner":   REQUEST_OWNER,
            "Host-Session-Id": host_sid,
            "X-Xsrf-Token":    session.cookies.get("XSRF-TOKEN", ""),
        }

        r = session.get(PORTFOLIO_URL, headers=headers, timeout=15)
        if r.status_code != 200:
            log.error("TMS clientPortfolio failed: %d", r.status_code)
            return {}

        payload = r.json().get("payload", {})
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        result  = {}

        for rec in (records or []):
            symbol = rec.get("symbol", "").upper().strip()
            if not symbol:
                continue
            result[symbol] = {
                "company":       rec.get("securityName", symbol),
                "current_price": float(rec.get("lastTradePrice", 0)),
                "prev_close":    float(rec.get("closePrice", 0)),
                "day_change_pct":float(rec.get("perChange", 0)),
                "high":          float(rec.get("highPrice", 0)),
                "low":           float(rec.get("lowPrice", 0)),
                "open_price":    float(rec.get("openPrice", 0)),
            }

        log.info("TMS clientPortfolio: %d positions", len(result))
        return result

    except Exception as e:
        log.error("TMS portfolio error: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MERGE + BUILD HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════

def _build_holdings(wacc_data: dict, tms_data: dict) -> list[Holding]:
    """
    Only build holdings for symbols present in BOTH waccReport AND TMS.
    Symbols in waccReport but not TMS = already sold, skip them.
    """
    holdings = []

    for symbol, w in wacc_data.items():
        # ✅ Skip if not in TMS — means already sold
        if symbol not in tms_data:
            log.info("  %-10s — not in TMS (already sold, skipping)", symbol)
            continue

        t             = tms_data[symbol]
        shares        = w["shares"]
        wacc          = w["wacc"]
        total_cost    = w["total_cost"]
        current_price = t.get("current_price", 0.0)
        current_value = shares * current_price if current_price > 0 else 0.0
        pnl_npr       = current_value - total_cost if total_cost > 0 else 0.0
        pnl_pct       = (pnl_npr / total_cost * 100) if total_cost > 0 else 0.0

        h = Holding(
            symbol         = symbol,
            company        = t.get("company", symbol),
            shares         = shares,
            wacc           = round(wacc, 2),
            total_cost     = round(total_cost, 2),
            current_price  = round(current_price, 2),
            prev_close     = round(t.get("prev_close", 0.0), 2),
            current_value  = round(current_value, 2),
            pnl_npr        = round(pnl_npr, 2),
            pnl_pct        = round(pnl_pct, 2),
            day_change_pct = round(t.get("day_change_pct", 0.0), 2),
            high           = round(t.get("high", 0.0), 2),
            low            = round(t.get("low", 0.0), 2),
        )
        holdings.append(h)

        log.info("  %-10s %4d shares  WACC %7.2f  LTP %7.2f  P&L %+.1f%%  Day %+.1f%%",
                 symbol, shares, wacc, current_price, pnl_pct, h.day_change_pct)

    return holdings


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def _write_portfolio(holdings: list[Holding]) -> int:
    written       = 0
    today         = datetime.now(tz=NST).strftime("%Y-%m-%d")
    active_symbols = {h.symbol for h in holdings}

    # Mark symbols no longer in portfolio as CLOSED
    try:
        existing = read_tab("portfolio")
        for row in existing:
            sym = row.get("symbol", "")
            if row.get("status", "").upper() == "OPEN" and sym not in active_symbols:
                upsert_row("portfolio",
                           {**row, "status": "CLOSED"},
                           conflict_columns=["symbol"])
                log.info("  %-10s → marked CLOSED (sold)", sym)
    except Exception as e:
        log.warning("Could not mark closed positions: %s", e)

    # Write active holdings
    for h in holdings:
        row = {**h.to_dict(), "status": "OPEN", "entry_date": today}
        try:
            ok = upsert_row("portfolio", row, conflict_columns=["symbol"])
            if ok:
                written += 1
        except Exception as e:
            log.warning("Failed to write %s: %s", h.symbol, e)

    return written

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def sync() -> Optional[PortfolioSummary]:
    """Full sync: waccReport + TMS → Neon. Called by capital_allocator."""
    log.info("=" * 60)
    log.info("PORTFOLIO SYNC — %s", datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M NST"))
    log.info("=" * 60)

    # Step 1: Meroshare token
    token = _get_token()
    if not token:
        log.error("Cannot proceed without Meroshare token")
        return None

    # Step 2: WACC report — symbols + shares + cost
    wacc_data = _fetch_wacc_report(token)
    if not wacc_data:
        log.error("No WACC data returned")
        return None

    # Step 3: TMS live prices
    tms_data = _fetch_tms_portfolio()
    if not tms_data:
        log.warning("No TMS price data — P&L will be 0")

    # Step 4: Merge
    holdings = _build_holdings(wacc_data, tms_data)
    if not holdings:
        return PortfolioSummary()

    # Step 5: Write to Neon
    written = _write_portfolio(holdings)
    log.info("Written %d positions to Neon", written)

    # Step 6: Summary
    total_cost  = sum(h.total_cost    for h in holdings)
    total_value = sum(h.current_value for h in holdings)
    total_pnl   = total_value - total_cost
    pnl_pct     = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    summary = PortfolioSummary(
        total_holdings  = len(holdings),
        total_cost_npr  = round(total_cost, 2),
        total_value_npr = round(total_value, 2),
        total_pnl_npr   = round(total_pnl, 2),
        total_pnl_pct   = round(pnl_pct, 2),
        holdings        = holdings,
    )

    log.info("✅ Sync complete — %d positions | Value NPR %.0f | P&L %+.1f%%",
            summary.total_holdings, summary.total_value_npr, summary.total_pnl_pct)
    return summary


def get_portfolio_summary() -> Optional[PortfolioSummary]:
    """Read from Neon only — no API call. Called by briefing.py."""
    try:
        rows      = read_tab("portfolio")
        open_rows = [r for r in rows if r.get("status", "").upper() == "OPEN"]
        holdings  = []

        for r in open_rows:
            def sf(k): return float(r.get(k, 0) or 0)
            holdings.append(Holding(
                symbol        = r.get("symbol", ""),
                company       = r.get("company", ""),
                shares        = int(sf("shares")),
                wacc          = sf("wacc"),
                total_cost    = sf("total_cost"),
                current_price = sf("current_price"),
                prev_close    = sf("prev_close"),
                current_value = sf("current_value"),
                pnl_npr       = sf("pnl_npr"),
                pnl_pct       = sf("pnl_pct"),
                day_change_pct= sf("day_change_pct"),
            ))

        total_cost  = sum(h.total_cost    for h in holdings)
        total_value = sum(h.current_value for h in holdings)
        total_pnl   = total_value - total_cost
        pnl_pct     = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        return PortfolioSummary(
            total_holdings  = len(holdings),
            total_cost_npr  = round(total_cost, 2),
            total_value_npr = round(total_value, 2),
            total_pnl_npr   = round(total_pnl, 2),
            total_pnl_pct   = round(pnl_pct, 2),
            holdings        = holdings,
        )
    except Exception as e:
        log.error("get_portfolio_summary failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    summary = get_portfolio_summary() if arg == "status" else sync()

    if summary:
        print(f"\n{'='*60}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"  Positions : {summary.total_holdings}")
        print(f"  Cost      : NPR {summary.total_cost_npr:>12,.2f}")
        print(f"  Value     : NPR {summary.total_value_npr:>12,.2f}")
        print(f"  P&L       : NPR {summary.total_pnl_npr:>+12,.2f}  ({summary.total_pnl_pct:+.1f}%)")
        if summary.holdings:
            print(f"\n  {'Symbol':<10} {'Shares':>6} {'WACC':>8} {'LTP':>8} {'P&L%':>8} {'Day%':>7}")
            print("  " + "-" * 55)
            for h in sorted(summary.holdings, key=lambda x: x.pnl_pct, reverse=True):
                print(f"  {h.symbol:<10} {h.shares:>6} {h.wacc:>8.2f} "
                      f"{h.current_price:>8.2f} {h.pnl_pct:>+7.1f}% {h.day_change_pct:>+6.1f}%")
        print(f"{'='*60}\n")
    else:
        print("\n  No portfolio data\n")
        sys.exit(1)
