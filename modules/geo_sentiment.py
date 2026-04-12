"""
geo_sentiment.py — UPDATED MARCH 28, 2026
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Global market sentiment score for Nepal stock market context.
          Runs every 30 minutes during market hours via GitHub Actions
          trading.yml workflow.

VALIDATION RESULT (2020-2026 historical data):
  ✅ DXY (USD Index)      — KEEP: ρ=-0.1708, p<0.001, lag=7d (meaningful)
  ❌ VIX                  — REMOVE: ρ=-0.0606, p=0.042 (noise)
  ❌ Crude Oil            — REMOVE: ρ=+0.0617, p=0.038 (noise)
  ❌ Nifty 50             — REMOVE: ρ=+0.0693, p=0.021 (noise, surprising)
  ❌ Gold                 — REMOVE: ρ=-0.0428, p=0.151 (noise)

UPDATED LOGIC:
  1. Fetch only DXY (USD Index) from yfinance
  2. Compute geo_score as pure DXY signal with 7-day context
  3. Write snapshot to geopolitical_data table (keep all 5 columns for backward compat)
  4. Save ~150 tokens per call vs previous version

INTERPRETATION:
  DXY strong (>105) → weaker remittance flows to Nepal → NEPSE pressured (7d lag)
  DXY weak (<102)  → stronger remittance flows → NEPSE positive (7d lag)

─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
from datetime import datetime
from typing import Optional

import yfinance as yf

from sheets import write_geo_snapshot, get_latest_geo
from calendar_guard import flag_adhoc_closure, today_nst
from config import NST

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

# ONLY DXY is validated as meaningful signal (ρ=-0.1708, p<0.001, lag=7d)
TICKERS = {
    "dxy": "DX-Y.NYB",   # US Dollar Index — ONLY validated signal
}

FETCH_TIMEOUT  = 15   # seconds per yfinance call
MAX_RETRIES    = 2    # retry attempts if first fetch fails

# ── DXY Scoring thresholds ────────────────────────────────────────────────────
# Stronger dollar (higher DXY) = weaker remittance value = negative for Nepal
# Empirically validated: 7-day lag between DXY change and NEPSE impact
DXY_VERY_STRONG = 108.0  # above = strong signal: capital outflow risk (-2)
DXY_STRONG      = 105.0  # above = moderate signal: outflow pressure (-1)
DXY_WEAK        = 102.0  # below = positive signal: inflow risk (+1)
DXY_VERY_WEAK   = 100.0  # below = strong positive signal: strong inflows (+2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FETCH MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_ticker(ticker: str) -> Optional[dict]:
    """
    Fetch latest price and change% for DXY only.

    Returns:
        {
          "price":      float  — latest DXY price
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
                "DXY price=%8.2f  chg=%+.2f%% (7-day lag to NEPSE)",
                price, change_pct
            )
            return {
                "price":      round(price, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(prev_close, 2),
            }

        except Exception as exc:
            log.warning(
                "Fetch DXY failed (attempt %d): %s",
                attempt + 1, exc
            )

    log.error("All retries failed for DXY")
    return None


def _fetch_all() -> dict:
    """
    Fetch DXY signal only.

    Returns dict with key: dxy
    Value is a price dict or None if fetch failed.
    """
    log.info("Fetching DXY (validated signal only)...")
    results = {}
    for name, ticker in TICKERS.items():
        results[name] = _fetch_ticker(ticker)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FALLBACK TO LAST KNOWN VALUES
# ══════════════════════════════════════════════════════════════════════════════

def _get_fallback() -> dict:
    """
    Read last known DXY value from Neon geopolitical_data table.
    Used when yfinance fails for DXY.
    """
    try:
        latest = get_latest_geo()
        if latest:
            return {
                "dxy": {
                    "price":      float(latest.get("dxy", 0)),
                    "change_pct": 0.0,  # Use last close, not change
                    "prev_close": float(latest.get("dxy", 0)),
                }
            }
    except Exception as exc:
        log.warning("Fallback read failed: %s", exc)

    return {}


def _merge_with_fallback(fetched: dict, fallback: dict) -> dict:
    """
    Merge fetched data with fallback values.
    Prioritize fetched; use fallback only if fetch failed.
    """
    result = {}
    for key in TICKERS.keys():
        if fetched.get(key) is not None:
            result[key] = fetched[key]
        elif fallback.get(key) is not None:
            result[key] = fallback[key]
            log.info("Using fallback for %s", key)
        else:
            result[key] = {"price": 0, "change_pct": 0, "prev_close": 0}

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _score_dxy(price: float) -> tuple[int, str]:
    """
    Score DXY signal.

    DXY is the ONLY validated signal for NEPSE (ρ=-0.1708, 7-day lag).
    Strong dollar → weaker remittance flows → capital outflow pressure.

    Returns:
        (score: int -2 to +2, description: str)
    """
    if price >= DXY_VERY_STRONG:
        return -2, f"DXY {price:.1f} — very strong USD, significant remittance pressure (-2)"
    elif price >= DXY_STRONG:
        return -1, f"DXY {price:.1f} — strong USD, moderate outflow risk (-1)"
    elif price <= DXY_VERY_WEAK:
        return +2, f"DXY {price:.1f} — weak USD, strong inflow potential (+2)"
    elif price <= DXY_WEAK:
        return +1, f"DXY {price:.1f} — weak USD, positive inflow signal (+1)"
    else:
        return 0, f"DXY {price:.1f} — neutral USD range (0)"


def compute_geo_score(data: dict) -> tuple[int, str, str]:
    """
    Compute geo_score from DXY signal only.

    Score breakdown (simplified):
      DXY: -2 to +2
      ──────────────
      Range: -2 to +2 (clamped to -5..+5 for consistency)

    Note: VIX, Crude, Nifty, Gold removed (ρ<0.07, noise).
    """
    score   = 0
    details = []

    dxy = data.get("dxy", {})

    s, d = _score_dxy(dxy.get("price", 0))
    score += s
    if d:
        details.append(d)

    # Clamp to -5 .. +5 (for schema consistency, but actual range is -2..+2)
    score = max(-5, min(5, score))

    # Status label
    if score >= 2:
        status = "BULLISH"
    elif score >= 1:
        status = "POSITIVE"
    elif score >= -1:
        status = "NEUTRAL"
    elif score >= -2:
        status = "BEARISH"
    else:
        status = "CRISIS"

    # Key event
    key_event = details[0] if details else "DXY neutral"

    log.info(
        "Geo score: %+d | Status: %s | Key event: %s | (DXY only, 7-day lag)",
        score, status, key_event
    )

    return score, status, key_event


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run() -> bool:
    """
    Main entry point. Called by trading.yml every 6 minutes.

    Flow:
      1. Fetch DXY via yfinance
      2. Merge with fallback if fetch failed
      3. Compute geo_score
      4. Write snapshot to geopolitical_data table (all columns for backward compat)

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
    dxy = data.get("dxy", {})

    # Keep all columns for backward compatibility with geopolitical_data schema
    # but only DXY has actual data; others are empty/fallback
    snapshot = {
        "date":             nst_now.strftime("%Y-%m-%d"),
        "time":             nst_now.strftime("%H:%M"),
        "crude_price":      "",  # REMOVED (noise)
        "crude_change_pct": "",  # REMOVED (noise)
        "vix":              "",  # REMOVED (noise)
        "vix_level":        "",  # REMOVED (noise)
        "nifty":            "",  # REMOVED (noise)
        "nifty_change_pct": "",  # REMOVED (noise)
        "dxy":              str(dxy.get("price", "")),  # KEPT (validated signal)
        "gold_price":       "",  # REMOVED (noise)
        "geo_score":        str(geo_score),
        "status":           status,
        "key_event":        key_event,
        "timestamp":        nst_now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Write to Neon ─────────────────────────────────────────────────────────
    success = write_geo_snapshot(snapshot)

    if success:
        log.info("✅ Geo snapshot written successfully")
        log.info("   Score:  %+d (DXY only)", geo_score)
        log.info("   Status: %s",  status)
        log.info("   DXY:    %.1f (7-day lag to NEPSE)", dxy.get("price", 0))
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
    
    NOTE: Score now represents DXY signal only (range -2 to +2).
          Interpretation: DXY is primary capital flow indicator for NEPSE.
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
#   python -m modules.geo_sentiment        → full run, write to Neon
#   python -m modules.geo_sentiment score  → print latest score only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GEO] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "score":
        score = get_latest_geo_score()
        print(f"\n  Latest geo_score: {score:+d} (DXY signal only)\n")
        sys.exit(0)

    success = run()

    if success:
        latest = get_latest_geo()
        if latest:
            print(f"\n{'='*50}")
            print(f"  GEO SENTIMENT SUMMARY")
            print(f"  (DXY Only — Validated Signal)")
            print(f"{'='*50}")
            print(f"  Date:       {latest.get('date')} {latest.get('time')}")
            print(f"  Geo Score:  {int(latest.get('geo_score', 0)):>+3}")
            print(f"  Status:     {latest.get('status')}")
            print(f"  Key Event:  {latest.get('key_event')}")
            print(f"  DXY:        {latest.get('dxy')} (7-day lag)")
            print(f"{'='*50}\n")
            print(f"  REMOVED (noise):")
            print(f"    VIX (ρ=-0.0606)")
            print(f"    Crude Oil (ρ=+0.0617)")
            print(f"    Nifty50 (ρ=+0.0693)")
            print(f"    Gold (ρ=-0.0428)")
            print(f"\n  Token savings: ~150 tokens/call vs previous version\n")
        sys.exit(0)
    else:
        sys.exit(1)