"""
geo_sentiment.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Global market sentiment score for Nepal stock market context.
          Runs every 30 minutes during market hours via GitHub Actions
          trading.yml workflow.

What this does (plain English):
  1. Fetches 5 global market signals via yfinance (free, no API key):
       Crude Oil (WTI)  — energy cost affects Nepal import bill
       VIX              — global fear index affects FII sentiment
       Nifty 50         — India market directly influences NEPSE
       DXY (USD Index)  — strong dollar hurts remittance value
       Gold             — safe haven signal
  2. Computes geo_score (-5 to +5) from those signals
  3. Writes snapshot to geopolitical_data table in Neon

Score meaning:
  +5  Very positive global environment for NEPSE
   0  Neutral
  -5  Very negative — high fear, crude spike, Nifty crash

How it fits the system:
  geo_sentiment.py  → geo_score   (-5 to +5)
  nepal_pulse.py    → nepal_score (-5 to +5)
  combined_geo      = geo_score + nepal_score  (-10 to +10)

Architecture note:
  Runs in the LIGHT trading.yml workflow (every 6 min).
  Fetch is fast (~10-15 seconds total for 5 tickers).
  Does NOT run in morning_brief.yml — indicators/candles run there.

─────────────────────────────────────────────────────────────────────────────

SOP — STANDARD OPERATING PROCEDURE
───────────────────────────────────
WHAT IT DOES:
  Fetches 5 global market prices and produces geo_score (-5 to +5).

INPUTS (all auto-fetched, no manual work):
  Crude Oil:  Yahoo Finance ticker CL=F  (WTI front month)
  VIX:        Yahoo Finance ticker ^VIX
  Nifty 50:   Yahoo Finance ticker ^NSEI
  DXY:        Yahoo Finance ticker DX-Y.NYB
  Gold:       Yahoo Finance ticker GC=F

OUTPUTS:
  Writes one row to geopolitical_data table in Neon:
    crude_price, crude_change_pct, vix, vix_level,
    nifty, nifty_change_pct, dxy, gold_price,
    geo_score, status, key_event, timestamp

HOW TO TEST MANUALLY:
  python geo_sentiment.py        → full run, print summary
  python geo_sentiment.py score  → print latest geo_score only

COMMON ERRORS AND FIXES:
  yfinance timeout    → Retries 2x then uses last known value from Neon.
                        geo_score defaults to 0 if no data at all.
  Market closed       → yfinance returns last close price. Still valid.
  DB write fails      → Logged, script exits with error code 1.
  "No data for ^VIX"  → Yahoo Finance ticker change. Check TICKERS dict.

HOW TO UPDATE TICKERS (if Yahoo changes them):
  Edit TICKERS dict at top of constants section.
  Common Yahoo Finance tickers:
    Crude WTI:  CL=F  or  BZ=F (Brent)
    VIX:        ^VIX
    Nifty:      ^NSEI
    DXY:        DX-Y.NYB  or  DX=F
    Gold:       GC=F

─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import yfinance as yf

from sheets import write_geo_snapshot, get_latest_geo
from calendar_guard import flag_adhoc_closure, today_nst

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GEO] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST = timezone(timedelta(hours=5, minutes=45))

# Yahoo Finance tickers — update here if they change
TICKERS = {
    "crude": "CL=F",       # WTI Crude Oil front month futures
    "vix":   "^VIX",       # CBOE Volatility Index
    "nifty": "^NSEI",      # Nifty 50 India
    "dxy":   "DX-Y.NYB",   # US Dollar Index
    "gold":  "GC=F",       # Gold front month futures
}

FETCH_TIMEOUT  = 15   # seconds per yfinance call
MAX_RETRIES    = 2    # retry attempts if first fetch fails

# ── Scoring thresholds ────────────────────────────────────────────────────────
# Each signal contributes to geo_score.
# Thresholds tuned for Nepal market context.

# Crude Oil — higher price = more expensive imports = negative for Nepal
CRUDE_HIGH     = 85.0    # above this = negative signal
CRUDE_VERY_HIGH= 95.0    # above this = strong negative
CRUDE_LOW      = 65.0    # below this = positive signal

# VIX — higher = more global fear = negative for emerging markets
VIX_LOW        = 15.0    # below = calm markets = positive
VIX_ELEVATED   = 20.0    # above = caution
VIX_HIGH       = 25.0    # above = fear
VIX_VERY_HIGH  = 30.0    # above = panic

# Nifty change % — India market strongly correlated with NEPSE
NIFTY_BULL     = +0.75   # above = positive India day
NIFTY_BEAR     = -0.75   # below = negative India day
NIFTY_STRONG   = +1.5    # above = strong bull India
NIFTY_CRASH    = -1.5    # below = significant India fall

# DXY — stronger dollar = weaker remittance value for Nepal
DXY_STRONG     = 105.0   # above = remittance worth less in NPR
DXY_VERY_STRONG= 108.0   # above = significant NPR pressure

# Gold — rising gold = safe haven demand = risk-off globally
GOLD_SURGE_PCT = +1.5    # daily % rise = risk-off signal


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FETCH MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_ticker(ticker: str) -> Optional[dict]:
    """
    Fetch latest price and change% for one Yahoo Finance ticker.

    Returns:
        {
          "price":      float  — latest price
          "change_pct": float  — % change from previous close
          "prev_close": float  — previous close price
        }
        None on failure — caller handles gracefully.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            data = yf.Ticker(ticker)
            info = data.fast_info

            price      = float(info.last_price or 0)
            prev_close = float(info.previous_close or price)

            if price <= 0:
                # fast_info failed — try history
                hist = data.history(period="2d", timeout=FETCH_TIMEOUT)
                if hist.empty:
                    log.warning("No data for %s (attempt %d)", ticker, attempt + 1)
                    continue
                price      = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price

            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            log.info(
                "%-12s price=%8.2f  chg=%+.2f%%",
                ticker, price, change_pct
            )
            return {
                "price":      round(price, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(prev_close, 2),
            }

        except Exception as exc:
            log.warning(
                "Fetch %s failed (attempt %d): %s",
                ticker, attempt + 1, exc
            )

    log.error("All retries failed for %s", ticker)
    return None


def _fetch_all() -> dict:
    """
    Fetch all 5 global market signals.

    Returns dict with keys: crude, vix, nifty, dxy, gold
    Each value is a price dict or None if fetch failed.
    """
    log.info("Fetching global market data...")
    results = {}
    for name, ticker in TICKERS.items():
        results[name] = _fetch_ticker(ticker)

    # Log summary
    fetched  = sum(1 for v in results.values() if v is not None)
    failed   = sum(1 for v in results.values() if v is None)
    log.info("Fetched %d/5 signals (%d failed)", fetched, failed)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FALLBACK TO LAST KNOWN VALUES
# ──────────────────────────────────────────────────────────────────────────────
# If yfinance fails for some signals, use the last values from Neon.
# System never crashes — neutral defaults if nothing available.
# ══════════════════════════════════════════════════════════════════════════════

def _get_fallback() -> dict:
    """
    Read last known geo values from Neon geopolitical_data table.
    Used when yfinance fails for specific tickers.
    Returns empty dict if no previous data.
    """
    try:
        latest = get_latest_geo()
        if not latest:
            return {}
        return {
            "crude": {
                "price":      float(latest.get("crude_price", 0) or 0),
                "change_pct": float(latest.get("crude_change_pct", 0) or 0),
            },
            "vix":   {"price": float(latest.get("vix", 0) or 0),   "change_pct": 0.0},
            "nifty": {
                "price":      float(latest.get("nifty", 0) or 0),
                "change_pct": float(latest.get("nifty_change_pct", 0) or 0),
            },
            "dxy":   {"price": float(latest.get("dxy", 0) or 0),   "change_pct": 0.0},
            "gold":  {"price": float(latest.get("gold_price", 0) or 0), "change_pct": 0.0},
        }
    except Exception as exc:
        log.warning("Could not read fallback from Neon: %s", exc)
        return {}


def _merge_with_fallback(fetched: dict, fallback: dict) -> dict:
    """
    For any signal that failed to fetch, use the fallback value.
    If no fallback either, use 0 — neutral contribution to score.
    """
    merged = {}
    for key in TICKERS:
        if fetched.get(key) is not None:
            merged[key] = fetched[key]
        elif fallback.get(key):
            merged[key] = fallback[key]
            log.info("Using fallback for %s: %.2f", key, fallback[key].get("price", 0))
        else:
            merged[key] = {"price": 0.0, "change_pct": 0.0, "prev_close": 0.0}
            log.warning("No data for %s — using 0 (neutral)", key)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPUTE GEO SCORE
# ──────────────────────────────────────────────────────────────────────────────
# Each signal contributes -2 to +2.
# Final score clamped to -5 .. +5.
# ══════════════════════════════════════════════════════════════════════════════

def _score_crude(price: float, change_pct: float) -> tuple[int, str]:
    """
    Crude Oil scoring.
    High crude = expensive imports = bad for Nepal.
    """
    if price <= 0:
        return 0, ""

    if price < CRUDE_LOW:
        score = +2
        detail = f"Crude low at ${price:.1f} — cheap imports, positive"
    elif price < CRUDE_HIGH:
        score = +1 if change_pct <= 0 else 0
        detail = f"Crude ${price:.1f} ({change_pct:+.1f}%) — moderate"
    elif price < CRUDE_VERY_HIGH:
        score = -1
        detail = f"Crude elevated at ${price:.1f} — import cost pressure"
    else:
        score = -2
        detail = f"Crude very high at ${price:.1f} — significant cost pressure"

    return score, detail


def _score_vix(vix: float) -> tuple[int, str]:
    """
    VIX scoring.
    High VIX = global fear = FII outflows from emerging markets including Nepal.
    """
    if vix <= 0:
        return 0, ""

    if vix < VIX_LOW:
        return +2, f"VIX {vix:.1f} — calm markets, risk-on globally"
    elif vix < VIX_ELEVATED:
        return +1, f"VIX {vix:.1f} — low volatility, positive"
    elif vix < VIX_HIGH:
        return 0,  f"VIX {vix:.1f} — elevated, neutral"
    elif vix < VIX_VERY_HIGH:
        return -1, f"VIX {vix:.1f} — fear rising, caution"
    else:
        return -2, f"VIX {vix:.1f} — high fear, risk-off globally"


def _score_nifty(change_pct: float) -> tuple[int, str]:
    """
    Nifty 50 change % scoring.
    Nifty and NEPSE are highly correlated — India sentiment drives Nepal.
    """
    if change_pct >= NIFTY_STRONG:
        return +2, f"Nifty +{change_pct:.1f}% — strong India rally, positive for NEPSE"
    elif change_pct >= NIFTY_BULL:
        return +1, f"Nifty +{change_pct:.1f}% — India up, positive signal"
    elif change_pct > NIFTY_BEAR:
        return 0,  f"Nifty {change_pct:+.1f}% — flat India day, neutral"
    elif change_pct > NIFTY_CRASH:
        return -1, f"Nifty {change_pct:.1f}% — India down, negative signal"
    else:
        return -2, f"Nifty {change_pct:.1f}% — India significant fall, negative for NEPSE"


def _score_dxy(price: float) -> tuple[int, str]:
    """
    DXY scoring.
    Strong dollar = remittance worth less in NPR = negative liquidity for Nepal.
    """
    if price <= 0:
        return 0, ""

    if price >= DXY_VERY_STRONG:
        return -2, f"DXY {price:.1f} — very strong dollar, remittance pressure"
    elif price >= DXY_STRONG:
        return -1, f"DXY {price:.1f} — strong dollar, mild NPR pressure"
    else:
        return +1, f"DXY {price:.1f} — dollar moderate, remittance stable"


def _score_gold(change_pct: float) -> tuple[int, str]:
    """
    Gold change % scoring.
    Rising gold = risk-off globally = negative for equity markets.
    """
    if change_pct >= GOLD_SURGE_PCT:
        return -1, f"Gold +{change_pct:.1f}% — safe haven demand, risk-off"
    elif change_pct <= -GOLD_SURGE_PCT:
        return +1, f"Gold {change_pct:.1f}% — risk appetite returning"
    else:
        return 0, f"Gold {change_pct:+.1f}% — stable, neutral"


def compute_geo_score(data: dict) -> tuple[int, str, str]:
    """
    Compute geo_score from all 5 signals.

    Returns:
        (score: int, status: str, key_event: str)

    Score breakdown:
      Crude oil    : -2 to +2
      VIX          : -2 to +2
      Nifty        : -2 to +2
      DXY          : -2 to +1
      Gold         : -1 to +1
      ───────────────────────
      Range        : -5 to +5 (clamped)
    """
    score   = 0
    details = []

    crude = data.get("crude", {})
    vix   = data.get("vix",   {})
    nifty = data.get("nifty", {})
    dxy   = data.get("dxy",   {})
    gold  = data.get("gold",  {})

    s, d = _score_crude(crude.get("price", 0), crude.get("change_pct", 0))
    score += s
    if d: details.append(d)

    s, d = _score_vix(vix.get("price", 0))
    score += s
    if d: details.append(d)

    s, d = _score_nifty(nifty.get("change_pct", 0))
    score += s
    if d: details.append(d)

    s, d = _score_dxy(dxy.get("price", 0))
    score += s
    if d: details.append(d)

    s, d = _score_gold(gold.get("change_pct", 0))
    score += s
    if d: details.append(d)

    # Clamp to -5 .. +5
    score = max(-5, min(5, score))

    # Status label
    if score >= 3:
        status = "BULLISH"
    elif score >= 1:
        status = "POSITIVE"
    elif score >= -1:
        status = "NEUTRAL"
    elif score >= -3:
        status = "BEARISH"
    else:
        status = "CRISIS"

    # Key event — most significant signal
    nifty_chg = nifty.get("change_pct", 0)
    vix_val   = vix.get("price", 0)
    crude_val = crude.get("price", 0)

    if vix_val >= VIX_VERY_HIGH:
        key_event = f"VIX at {vix_val:.1f} — extreme fear, risk-off globally"
    elif abs(nifty_chg) >= abs(NIFTY_STRONG):
        key_event = f"Nifty {nifty_chg:+.1f}% — strong India move"
    elif crude_val >= CRUDE_VERY_HIGH:
        key_event = f"Crude at ${crude_val:.1f} — high energy cost pressure"
    elif details:
        key_event = details[0]
    else:
        key_event = "Global markets — routine conditions"

    log.info(
        "Geo score: %+d | Status: %s | Key event: %s",
        score, status, key_event
    )

    return score, status, key_event


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VIX LEVEL LABEL
# ══════════════════════════════════════════════════════════════════════════════

def _vix_level(vix: float) -> str:
    """Human-readable VIX level label."""
    if vix < VIX_LOW:
        return "CALM"
    elif vix < VIX_ELEVATED:
        return "LOW"
    elif vix < VIX_HIGH:
        return "ELEVATED"
    elif vix < VIX_VERY_HIGH:
        return "HIGH"
    else:
        return "EXTREME"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run() -> bool:
    """
    Main entry point. Called by trading.yml every 6 minutes.

    Flow:
      1. Fetch 5 global signals via yfinance
      2. Merge with fallback values if any failed
      3. Compute geo_score
      4. Write snapshot to geopolitical_data table

    Returns True on success, False on failure.
    """
    nst_now = datetime.now(tz=NST)
    log.info("=" * 60)
    log.info("GEO SENTIMENT starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # ── Fetch ────────────────────────────────────────────────────────────────
    fetched  = _fetch_all()
    fallback = _get_fallback()
    data     = _merge_with_fallback(fetched, fallback)

    # ── Score ────────────────────────────────────────────────────────────────
    geo_score, status, key_event = compute_geo_score(data)

    # ── Build snapshot ────────────────────────────────────────────────────────
    crude = data.get("crude", {})
    vix   = data.get("vix",   {})
    nifty = data.get("nifty", {})
    dxy   = data.get("dxy",   {})
    gold  = data.get("gold",  {})

    snapshot = {
        "date":             nst_now.strftime("%Y-%m-%d"),
        "time":             nst_now.strftime("%H:%M"),
        "crude_price":      str(crude.get("price", "")),
        "crude_change_pct": str(crude.get("change_pct", "")),
        "vix":              str(vix.get("price", "")),
        "vix_level":        _vix_level(vix.get("price", 0)),
        "nifty":            str(nifty.get("price", "")),
        "nifty_change_pct": str(nifty.get("change_pct", "")),
        "dxy":              str(dxy.get("price", "")),
        "gold_price":       str(gold.get("price", "")),
        "geo_score":        str(geo_score),
        "status":           status,
        "key_event":        key_event,
        "timestamp":        nst_now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Write to Neon ─────────────────────────────────────────────────────────
    success = write_geo_snapshot(snapshot)

    if success:
        log.info("✅ Geo snapshot written successfully")
        log.info("   Score:  %+d", geo_score)
        log.info("   Status: %s",  status)
        log.info("   Crude:  $%.1f (%+.1f%%)", crude.get("price", 0), crude.get("change_pct", 0))
        log.info("   VIX:    %.1f (%s)",        vix.get("price", 0),  _vix_level(vix.get("price", 0)))
        log.info("   Nifty:  %+.2f%%",          nifty.get("change_pct", 0))
        log.info("   DXY:    %.1f",             dxy.get("price", 0))
        log.info("   Gold:   $%.1f (%+.1f%%)",  gold.get("price", 0), gold.get("change_pct", 0))
    else:
        log.error("❌ Failed to write geo snapshot to Neon")

    return success


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — called by other modules
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_geo_score() -> int:
    """
    Returns latest geo_score as integer.
    Called by filter_engine.py and gemini_filter.py.
    Returns 0 (neutral) if no data available.
    """
    try:
        latest = get_latest_geo()
        if latest and latest.get("geo_score"):
            return int(latest["geo_score"])
    except Exception as exc:
        log.warning("get_latest_geo_score failed: %s", exc)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python geo_sentiment.py        → full run, write to Neon
#   python geo_sentiment.py score  → print latest score only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GEO] %(levelname)s: %(message)s",
    )

    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "score":
        score = get_latest_geo_score()
        print(f"\n  Latest geo_score: {score:+d}\n")
        sys.exit(0)

    success = run()

    if success:
        latest = get_latest_geo()
        if latest:
            print(f"\n{'='*50}")
            print(f"  GEO SENTIMENT SUMMARY")
            print(f"{'='*50}")
            print(f"  Date:       {latest.get('date')} {latest.get('time')}")
            print(f"  Geo Score:  {int(latest.get('geo_score', 0)):>+3}")
            print(f"  Status:     {latest.get('status')}")
            print(f"  Key Event:  {latest.get('key_event')}")
            print(f"  Crude:      ${latest.get('crude_price')} ({latest.get('crude_change_pct')}%)")
            print(f"  VIX:        {latest.get('vix')} [{latest.get('vix_level')}]")
            print(f"  Nifty:      {latest.get('nifty_change_pct')}%")
            print(f"  DXY:        {latest.get('dxy')}")
            print(f"  Gold:       ${latest.get('gold_price')}")
            print(f"{'='*50}\n")
        sys.exit(0)
    else:
        sys.exit(1)
