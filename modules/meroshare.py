"""
meroshare.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Fetch live portfolio combining two working APIs:
          1. Meroshare waccReport   → cost basis (WACC, total cost) per scrip
          2. Meroshare myPortfolio  → real current DEMAT balance + live price

myPortfolio is CDSC-level and broker-agnostic — it reflects the true DEMAT
balance regardless of which broker executed the trade. ATrad's own
getPortfolio (BS92009) was tried first but only reflects trades executed
through that specific broker/TMS, so it silently misses anything bought
elsewhere — confirmed empty even when 5 real positions existed. Don't use
ATrad getPortfolio as the live-holdings source; myPortfolio is authoritative.

Auth:
  - Meroshare: POST to auth URL → token in response header

SOP:
  python meroshare.py         → full sync, write to Neon
  python meroshare.py status  → read from Neon only (no API call)

CREDENTIALS NEEDED IN .env:
  MEROSHARE_USERNAME=
  MEROSHARE_PASSWORD=
  MEROSHARE_DP_ID=
  MEROSHARE_DEMAT=1301180000232764
  MEROSHARE_CLIENT_CODE=11800
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MEROSHARE] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST           = timezone(timedelta(hours=5, minutes=45))
DEMAT         = os.getenv("MEROSHARE_DEMAT", "1301180000232764")
CLIENT_CODE   = os.getenv("MEROSHARE_CLIENT_CODE", "11800")

AUTH_URL          = "https://webbackend.cdsc.com.np/api/meroShare/auth/"
WACC_URL          = "https://webbackend.cdsc.com.np/api/myPurchase/waccReport/"
WACC_SHARE_URL    = "https://webbackend.cdsc.com.np/api/myPurchase/share/"
WACC_SEARCH_URL   = "https://webbackend.cdsc.com.np/api/myPurchase/search/wacc/"
WACC_UPLOAD_URL   = "https://webbackend.cdsc.com.np/api/myPurchase/upload/"
WACC_VIEW_URL     = "https://webbackend.cdsc.com.np/api/myPurchase/view/"
MY_PORTFOLIO_URL  = "https://webbackend.cdsc.com.np/api/meroShareView/myPortfolio/"

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
# SECTION 2B — AUTO-COMPLETE PENDING WACC
# A scrip only shows up in waccReport once its purchase-rate transactions
# have been confirmed. myPurchase/share/ lists scrips still waiting on that
# confirmation; for each one, search -> upload (confirm at the known rate,
# unchanged) -> view completes it, exactly mirroring the manual flow in the
# Meroshare UI.
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_pending_wacc_scrips(token: str) -> list[str]:
    """Scrips with unconfirmed WACC transactions. Empty list on any failure."""
    headers = {**MEROSHARE_HEADERS, "Authorization": token}
    try:
        r = requests.post(WACC_SHARE_URL, headers=headers,
                          json={"isFilterByAllScript": False}, timeout=15)
        if r.status_code != 200:
            log.warning("myPurchase/share failed: %d", r.status_code)
            return []
        return list(r.json() or [])
    except Exception as e:
        log.warning("myPurchase/share error: %s", e)
        return []


def _confirm_wacc_for_scrip(token: str, scrip: str) -> Optional[dict]:
    """
    Confirm all pending purchase-rate transactions for one scrip at their
    known rate (no rate changes), then return the finalized WACC summary.
    Returns None on any failure — never raises.
    """
    headers = {**MEROSHARE_HEADERS, "Authorization": token}
    try:
        r = requests.post(WACC_SEARCH_URL, headers=headers,
                          json={"demat": DEMAT, "scrip": scrip}, timeout=15)
        if r.status_code != 200:
            log.error("WACC search failed for %s: %d", scrip, r.status_code)
            return None

        pending = r.json().get("waccUpdateResponse", []) or []
        if not pending:
            log.info("  %-10s — no pending WACC transactions", scrip)
            return r.json().get("waccSummaryResponse")

        # Confirm at the rate Meroshare already knows (rate/userPrice
        # untouched) — this only accepts the pending transaction into the
        # WACC calc, it doesn't change the price. isEdit:True is literally
        # the "confirm" flag: the record comes back from search with it
        # False, and flipping it True + adding remarks is what the manual
        # Meroshare UI does when you click through the same confirmation.
        upload_payload = [{**rec, "isEdit": True, "remarks": ""} for rec in pending]

        r2 = requests.post(WACC_UPLOAD_URL, headers=headers, json=upload_payload, timeout=15)
        if r2.status_code != 202:
            log.error("WACC upload failed for %s: %d — %s", scrip, r2.status_code, r2.text[:200])
            return None
        log.info("  %-10s — WACC confirmed (%d transaction(s))", scrip, len(pending))

        r3 = requests.post(WACC_VIEW_URL, headers=headers,
                           json={"demat": DEMAT, "scrip": scrip}, timeout=15)
        if r3.status_code != 200:
            log.error("WACC view failed for %s: %d", scrip, r3.status_code)
            return None
        return r3.json()

    except Exception as e:
        log.error("WACC confirm error for %s: %s", scrip, e)
        return None


def _complete_pending_wacc(token: str) -> dict:
    """
    Auto-complete WACC confirmation for every scrip that's pending it.
    Returns {scrip: finalized_wacc_dict_or_None}. Never raises — a failure
    on one scrip doesn't block the rest or the caller's sync().
    """
    pending_scrips = _fetch_pending_wacc_scrips(token)
    if not pending_scrips:
        log.info("No pending WACC confirmations")
        return {}

    log.info("Pending WACC confirmations: %s", pending_scrips)
    results = {}
    for scrip in pending_scrips:
        results[scrip] = _confirm_wacc_for_scrip(token, scrip)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MEROSHARE MY PORTFOLIO (authoritative shares + live price)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_my_portfolio(token: str) -> dict[str, dict]:
    """
    Real current DEMAT balance via CDSC meroShareView/myPortfolio —
    broker-agnostic (unlike ATrad's getPortfolio, which only reflects trades
    executed through that specific broker/TMS and silently misses anything
    bought elsewhere). This is the authoritative "what do I actually own"
    source. Cost basis (wacc/total_cost) still comes from Meroshare's
    waccReport — myPortfolio has no purchase-price data.

    Returns dict keyed by symbol:
    { "NABIL": { shares, current_price, market_value, company }, ... }
    """
    headers = {**MEROSHARE_HEADERS, "Authorization": token}
    try:
        r = requests.post(MY_PORTFOLIO_URL, headers=headers, json={
            "sortBy":   "script",
            "demat":    [DEMAT],
            "clientCode": CLIENT_CODE,
            "page":     1,
            "size":     200,
            "sortAsc":  True,
        }, timeout=15)
        if r.status_code != 200:
            log.error("myPortfolio failed: %d", r.status_code)
            return {}

        rows   = r.json().get("meroShareMyPortfolio", []) or []
        result = {}
        for row in rows:
            symbol = row.get("script", "").upper().strip()
            if not symbol:
                continue
            result[symbol] = {
                "shares":        int(row.get("currentBalance", 0) or 0),
                "current_price": float(row.get("lastTransactionPrice", 0) or 0),
                "market_value":  float(row.get("valueOfLastTransPrice", 0) or 0),
                "company":       row.get("scriptDesc", symbol),
            }
        log.info("myPortfolio: %d positions", len(result))
        return result
    except Exception as e:
        log.error("myPortfolio error: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MERGE + BUILD HOLDINGS
# ══════════════════════════════════════════════════════════════════════════════

def _build_holdings(wacc_data: dict, my_portfolio_data: dict) -> list[Holding]:
    """
    Only build holdings for symbols present in BOTH waccReport AND the real
    DEMAT balance (myPortfolio). Symbols in waccReport but not in myPortfolio
    = already sold, skip them. Shares and current_price come from myPortfolio
    (CDSC-level, broker-agnostic, live); wacc/total_cost stay Meroshare's
    waccReport (myPortfolio has no purchase-price data).
    """
    holdings = []

    for symbol, w in wacc_data.items():
        # ✅ Skip if not in the real DEMAT balance — means already sold
        if symbol not in my_portfolio_data:
            log.info("  %-10s — not in myPortfolio (already sold, skipping)", symbol)
            continue

        m             = my_portfolio_data[symbol]
        shares        = m["shares"]
        wacc          = w["wacc"]
        total_cost    = w["total_cost"]
        current_price = m.get("current_price", 0.0)
        # myPortfolio already gives valueOfLastTransPrice directly — prefer
        # that over shares*price so rounding always matches what CDSC itself
        # reports. Only fall back to computing it if that field is missing.
        current_value = m.get("market_value") or (shares * current_price if current_price > 0 else 0.0)
        pnl_npr       = current_value - total_cost if total_cost > 0 else 0.0
        pnl_pct       = (pnl_npr / total_cost * 100) if total_cost > 0 else 0.0

        h = Holding(
            symbol         = symbol,
            company        = m.get("company", symbol),
            shares         = shares,
            wacc           = round(wacc, 2),
            total_cost     = round(total_cost, 2),
            current_price  = round(current_price, 2),
            current_value  = round(current_value, 2),
            pnl_npr        = round(pnl_npr, 2),
            pnl_pct        = round(pnl_pct, 2),
            source         = "meroshare_portfolio+wacc",
        )
        holdings.append(h)

        log.info("  %-10s %4d shares  WACC %7.2f  LTP %7.2f  P&L %+.1f%%",
                 symbol, shares, wacc, current_price, pnl_pct)

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
    """Full sync: waccReport + myPortfolio (real DEMAT balance) → Neon. Called by capital_allocator."""
    log.info("=" * 60)
    log.info("PORTFOLIO SYNC — %s", datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M NST"))
    log.info("=" * 60)

    # Step 1: Meroshare token
    token = _get_token()
    if not token:
        log.error("Cannot proceed without Meroshare token")
        return None

    # Step 1b: auto-complete any pending WACC confirmations first — a scrip
    # doesn't appear in waccReport until this is done.
    _complete_pending_wacc(token)

    # Step 2: WACC report — symbols + shares + cost
    wacc_data = _fetch_wacc_report(token)
    if not wacc_data:
        log.error("No WACC data returned")
        return None

    # Step 3: myPortfolio — real DEMAT balance + live price (broker-agnostic)
    my_portfolio_data = _fetch_my_portfolio(token)
    if not my_portfolio_data:
        log.warning("No myPortfolio data — holdings will be empty")

    # Step 4: Merge
    holdings = _build_holdings(wacc_data, my_portfolio_data)
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


def verify_live_holding(symbol: str, min_shares: float = 1) -> bool:
    """
    Confirm `symbol` is actually held (>= min_shares) right now, via a fresh
    myPortfolio call — the real, broker-agnostic DEMAT balance.

    Safety gate for the LIVE execution monitor: a position can look OPEN in
    the app's own `portfolio` table (e.g. from a stale import, or a symbol
    never actually bought through this app's trading engine) without the
    shares actually being there. Never auto-close a live position without
    this confirming it first.

    Fails closed — any error (login, network, missing symbol) returns False,
    never True. Never raises.
    """
    try:
        token = _get_token()
        if not token:
            log.error("verify_live_holding(%s): no Meroshare token", symbol)
            return False
        data = _fetch_my_portfolio(token)
        held = data.get(symbol.upper().strip(), {}).get("shares", 0)
        return held >= min_shares
    except Exception as e:
        log.error("verify_live_holding(%s) error: %s", symbol, e)
        return False


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
