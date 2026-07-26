"""
filter_v3.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — v3: Oversold Recovery Detector

WHAT v3 IS
----------
v3 is NOT a re-rank of already-passing candidates (that is v2's job).
v3 is an **oversold recovery detector** targeting one specific failure mode:
symbols blocked by the tech_score gate whose low score comes from a recent
sharp decline (hence oversold RSI, price below lower Bollinger Band, negative
MACD histogram) but where the decline has exhausted itself and reversal is
beginning.

EMPIRICAL BASIS  (v3_feature_analysis table, nepse_test DB, 2026-04-15 to 2026-07-01)
─────────────────────────────────────────────────────────────────────────────
n = 6,045  (953 FALSE_BLOCK / 5,092 CORRECT_BLOCK), TECH_SCORE gate only.

Top discriminating features (Cohen's |d|, descending):

  |d|=0.872  rsi_was_oversold      FB=60.9%  CB=21.5%  (+39.4pp)
  |d|=0.700  pctb_was_lower_touch  FB=66.5%  CB=33.5%  (+33.0pp)
  |d|=0.682  rsi_min               FB=30.4   CB=39.2   (-8.8 pts — FB much deeper)
  |d|=0.599  pctb_min              FB=-0.01  CB=0.18   (FB was below lower band)
  |d|=0.595  rsi_deep_oversold     FB=39.1%  CB=14.0%  (+25.1pp)
  |d|=0.557  rsi_current           FB=37.8   CB=44.4   (FB still recovering)
  |d|=0.502  obv_dominant_rising   FB=12.7%  CB=33.2%  (LOWER — OBV not yet up)
  |d|=0.495  obv_rising_days       FB=0.63   CB=1.44   (LOWER — accumulation not started)
  |d|=0.420  reversal_score        FB=2.27   CB=1.71   (+0.56)
  |d|=0.407  pctb_slope            FB=+0.012 CB=-0.015 (FB recovering from bottom)
  |d|=0.342  hist_improving_neg    FB=46.5%  CB=30.1%  (+16.4pp)
  |d|=0.333  pctb_recovering       FB=32.8%  CB=18.5%  (+14.3pp)
  |d|=0.299  rsi_slope             FB=+0.209 CB=-0.481 (FB rising, CB still falling)

KEY INSIGHT: FALSE_BLOCK stocks are NOT accumulation plays — OBV is actually
LOWER in FALSE_BLOCK than CORRECT_BLOCK. They are OVERSOLD REVERSALS. The signal
is: RSI was deeply oversold (< 35, often < 30), price touched/broke lower Bollinger
Band, AND now RSI and BB%B are starting to tick up. Tech_score is low because the
snapshot sees the aftermath of the decline, not the nascent recovery.

v2 COMPARISON: v2 detects progression/slope across all indicators (OBV, EMA, RSI
slope, etc.). v3 specifically detects oversold reversal using the DEPTH of the
oversold condition + early recovery signs. v2 would score these low because OBV
and EMA are still negative. v3 scores them high precisely because of the oversold
depth — a feature v2 does not use at all.

WEIGHT PROVENANCE
─────────────────────────────────────────────────────────────────────────────
Weights derived from Cohen's d ranking with proportional allocation:

  Component 1 — Oversold Depth & RSI Recovery   40%   (|d| 0.87 + 0.68 + 0.60)
  Component 2 — Bollinger Lower Band Recovery    25%   (|d| 0.70 + 0.60 + 0.33)
  Component 3 — MACD Histogram Recovery         15%   (|d| 0.34)
  Component 4 — Price Structure                 10%   (price_in_range + support)
  Component 5 — Volume / Accumulation Evidence  10%   (rsi_slope + exhaustion signals)

v3 ELIGIBILITY GATE (pre-score check)
─────────────────────────────────────────────────────────────────────────────
Only run v3 on symbols where:
  1. Blocked by TECH_SCORE gate (not structural: HISTORY, MUTUAL_FUND, etc.)
  2. tech_score >= 20 (absolute floor — avoids insolvent stocks)
  3. AT LEAST ONE of:
       - rsi_min_in_window < 35  (was oversold)
       - pctb_min_in_window < 0.1 (was at lower Bollinger Band)
  4. NOT bounce_failed (rising RSI after extreme oversold that immediately
     reversed — dead-cat trap, already detected by _compute_momentum)

v3 THRESHOLD: composite_v3 >= 55 to flag for Gemini review.

VALIDATION NOTE
─────────────────────────────────────────────────────────────────────────────
False positive risk is real: 39% of FALSE_BLOCK had deep oversold; but 14%
of CORRECT_BLOCK did too. Combined with the 5-feature score, expected false
positive rate is lower — but walk-forward validation on 2026-H2 live data
is required before raising capital allocation above minimum.

Use: python filter_v3.py SYMBOL  — standalone check
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Weights sum to 1.00
INDICATOR_WEIGHTS_V3 = {
    "oversold_recovery": 0.40,
    "bb_lower_recovery": 0.25,
    "macd_recovery":     0.15,
    "price_structure":   0.10,
    "volume_exhaustion": 0.10,
}

# Eligibility: minimum tech_score to even attempt v3 rescue
V3_MIN_TECH_SCORE = 20

# Score threshold to pass to Gemini
V3_COMPOSITE_THRESHOLD = 55.0


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT SCORERS  (each returns 0–100)
# ══════════════════════════════════════════════════════════════════════════════

def _score_oversold_recovery(momentum: dict, recent_rows: list[dict]) -> tuple[float, dict]:
    """
    Component 1 — weight 0.40.

    Measures HOW deeply oversold the stock was AND whether recovery has begun.
    Two sub-signals combined:
      A. Oversold depth (|d|=0.87): was RSI ever < 35 in the 7-day window?
      B. RSI current level AND positive slope (|d|=0.56 + 0.30)

    Scoring:
      - RSI min < 30  (deep oversold) + current RSI starting to rise → 90–100
      - RSI min < 35  (oversold)      + RSI rising                  → 70–85
      - RSI min < 40  + RSI rising                                   → 50–65
      - RSI oversold but slope still negative (not yet turning)       → 30–45
      - No oversold condition                                          → 0–25
    """
    rsi_vals = []
    for r in recent_rows:
        v = r.get("rsi_14")
        try:
            f = float(v)
            if f == f and 0 < f <= 100:
                rsi_vals.append(f)
        except (TypeError, ValueError):
            pass

    if not rsi_vals:
        return 25.0, {"reason": "no_rsi_data"}

    rsi_now  = rsi_vals[0]
    rsi_min  = min(rsi_vals)
    rsi_slope = momentum.get("rsi_slope_3d", 0.0)
    bounce_failed = momentum.get("bounce_failed", False)

    # Penalise dead-cat bounces hard
    if bounce_failed:
        return 5.0, {"reason": "bounce_failed", "rsi_min": rsi_min}

    # Depth bucket
    if rsi_min < 30:
        depth_score = 80.0
        depth_label = "deep_oversold_<30"
    elif rsi_min < 35:
        depth_score = 65.0
        depth_label = "oversold_<35"
    elif rsi_min < 40:
        depth_score = 45.0
        depth_label = "mildly_oversold_<40"
    else:
        depth_score = 15.0
        depth_label = "not_oversold"

    # Recovery modifier: is RSI now rising?
    if rsi_slope > 1.5:
        recovery_boost = 20.0
    elif rsi_slope > 0.5:
        recovery_boost = 12.0
    elif rsi_slope > 0.0:
        recovery_boost = 6.0
    elif rsi_slope > -0.5:
        recovery_boost = 0.0
    else:
        recovery_boost = -15.0  # still falling — penalise

    # Current RSI level: still in recovery zone = better
    if rsi_now < 35:
        level_adj = 5.0   # still very oversold but turning = strong signal
    elif rsi_now < 45:
        level_adj = 3.0
    elif rsi_now < 55:
        level_adj = 0.0
    else:
        level_adj = -10.0  # already recovered = less predictive

    score = min(100.0, max(0.0, depth_score + recovery_boost + level_adj))
    return score, {
        "reason":         depth_label,
        "rsi_min":        round(rsi_min, 1),
        "rsi_now":        round(rsi_now, 1),
        "rsi_slope":      round(rsi_slope, 3),
        "recovery_boost": recovery_boost,
    }


def _score_bb_lower_recovery(momentum: dict, recent_rows: list[dict]) -> tuple[float, dict]:
    """
    Component 2 — weight 0.25.

    Measures how far below the lower Bollinger Band the price fell (BB%B minimum)
    and whether BB%B is now recovering.

    Data: pctb_vals[0] = today, pctb_vals[-1] = oldest in window.

    Scoring:
      BB%B was < 0   (below lower band) + slope positive → 85–100
      BB%B was < 0.1 (at lower band)   + recovering      → 65–80
      BB%B was < 0.2 + recovering                        → 45–60
      BB%B was < 0.3 + slope flat/positive               → 30–40
      No lower-band touch                                 → 10–20
    """
    pctb_vals = []
    for r in recent_rows:
        v = r.get("bb_pct_b")
        try:
            f = float(v)
            if f == f:
                pctb_vals.append(f)
        except (TypeError, ValueError):
            pass

    if not pctb_vals:
        return 20.0, {"reason": "no_pctb_data"}

    pctb_now  = pctb_vals[0]
    pctb_min  = min(pctb_vals)
    pctb_slope = momentum.get("bb_pct_b_slope", 0.0)

    # Depth bucket (how far below lower band was price?)
    if pctb_min < 0.0:
        depth_score = 75.0
        depth_label = "below_lower_band"
    elif pctb_min < 0.10:
        depth_score = 60.0
        depth_label = "at_lower_band"
    elif pctb_min < 0.20:
        depth_score = 40.0
        depth_label = "near_lower_band"
    elif pctb_min < 0.30:
        depth_score = 25.0
        depth_label = "lower_third"
    else:
        depth_score = 10.0
        depth_label = "no_lower_touch"

    # Recovery modifier
    if pctb_slope > 0.03:
        rec_boost = 20.0
    elif pctb_slope > 0.01:
        rec_boost = 12.0
    elif pctb_slope > 0.0:
        rec_boost = 5.0
    elif pctb_slope > -0.01:
        rec_boost = 0.0
    else:
        rec_boost = -10.0

    # Current %B: if already above middle of band, signal weakens
    if pctb_now < 0.3:
        level_adj = 5.0
    elif pctb_now < 0.5:
        level_adj = 0.0
    else:
        level_adj = -8.0

    score = min(100.0, max(0.0, depth_score + rec_boost + level_adj))
    return score, {
        "reason":     depth_label,
        "pctb_min":   round(pctb_min, 3),
        "pctb_now":   round(pctb_now, 3),
        "pctb_slope": round(pctb_slope, 4),
    }


def _score_macd_recovery(momentum: dict, recent_rows: list[dict]) -> tuple[float, dict]:
    """
    Component 3 — weight 0.15.

    MACD histogram recovering from negative values.

    Analysis: hist_improving_from_neg: FB=46.5% vs CB=30.1% (+16.4pp).
    Signal is real but smaller effect than RSI/BB, hence 15% weight.

    Scoring:
      Histogram was negative AND now improving AND hist_slope > 0.3  → 85–100
      Histogram negative AND improving (slope > 0)                   → 60–80
      Histogram near zero cross from below                            → 50–65
      Histogram flat/slightly positive (no edge)                     → 30–40
      Histogram positive and sloping down (deteriorating)             → 5–20
    """
    hist_vals = []
    for r in recent_rows:
        v = r.get("macd_histogram")
        try:
            f = float(v)
            if f == f:
                hist_vals.append(f)
        except (TypeError, ValueError):
            pass

    hist_slope = momentum.get("macd_hist_slope", 0.0)

    if not hist_vals:
        return 35.0, {"reason": "no_hist_data"}  # neutral default

    hist_now  = hist_vals[0]
    hist_min  = min(hist_vals)
    was_negative = hist_min < 0

    if not was_negative:
        # Never went negative — signal not relevant for oversold recovery
        return 20.0, {"reason": "hist_never_negative", "hist_now": round(hist_now, 4)}

    # Was negative — now check recovery
    if hist_slope > 0.5:
        slope_score = 90.0
        slope_label = "strong_recovery"
    elif hist_slope > 0.1:
        slope_score = 70.0
        slope_label = "improving"
    elif hist_slope > 0.0:
        slope_score = 55.0
        slope_label = "barely_improving"
    elif hist_slope > -0.1:
        slope_score = 35.0
        slope_label = "flat"
    else:
        slope_score = 15.0
        slope_label = "still_deteriorating"

    # Near zero cross = extra value
    zero_cross_bonus = 0.0
    if -0.3 < hist_now < 0.1 and hist_slope > 0:
        zero_cross_bonus = 10.0

    score = min(100.0, max(0.0, slope_score + zero_cross_bonus))
    return score, {
        "reason":      slope_label,
        "hist_now":    round(hist_now, 4),
        "hist_min":    round(hist_min, 4),
        "hist_slope":  round(hist_slope, 6),
    }


def _score_price_structure(recent_price_rows: list[dict]) -> tuple[float, dict]:
    """
    Component 4 — weight 0.10.

    Price position in recent range + support proximity.

    Analysis: price_in_range_pct: FB=20.7 vs CB=32.6 (-12pp).
    FALSE_BLOCK stocks were in the bottom 20% of their 20-day range.

    Args:
        recent_price_rows: list of dicts with keys: date, price, high, low, vol.
                           Oldest first, most recent last. Up to 20 rows.
    """
    if len(recent_price_rows) < 3:
        return 30.0, {"reason": "insufficient_price_data"}

    prices = [r["price"] for r in recent_price_rows if r.get("price") and r["price"] > 0]
    highs  = [r.get("high", p) for r, p in zip(recent_price_rows, prices) if p > 0]
    lows   = [r.get("low",  p) for r, p in zip(recent_price_rows, prices) if p > 0]

    if not prices:
        return 30.0, {"reason": "no_prices"}

    cur_p  = prices[-1]
    min20  = min(prices)
    max20  = max(prices)
    rng20  = max20 - min20

    # Where in range?
    range_pct = (cur_p - min20) / rng20 * 100 if rng20 > 0 else 50.0

    # Position score: lower = better for oversold reversal
    if range_pct < 10:
        pos_score = 90.0
    elif range_pct < 20:
        pos_score = 75.0
    elif range_pct < 35:
        pos_score = 55.0
    elif range_pct < 50:
        pos_score = 35.0
    else:
        pos_score = 10.0

    # Support proximity: current price within 4% of recent (10-day) low?
    recent_low = min(lows[-10:]) if len(lows) >= 10 else min(lows)
    near_support = cur_p <= recent_low * 1.04

    # 3-day momentum: slight positive is a good sign at the bottom
    mom3 = (prices[-1] - prices[-4]) / prices[-4] * 100 if len(prices) >= 4 else 0.0
    support_bounce = near_support and mom3 > 0

    # Consecutive down days before today
    consec_down = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] < prices[i - 1]:
            consec_down += 1
        else:
            break

    exhaustion_adj = 0.0
    if consec_down >= 5:
        exhaustion_adj = 8.0   # extended sell-off = more exhaustion
    elif consec_down >= 3:
        exhaustion_adj = 4.0
    elif consec_down == 0 and range_pct < 30:
        exhaustion_adj = 5.0   # already bouncing from bottom

    support_adj = 7.0 if support_bounce else (3.0 if near_support else 0.0)

    score = min(100.0, max(0.0, pos_score + exhaustion_adj + support_adj))
    return score, {
        "reason":       "price_structure",
        "range_pct":    round(range_pct, 1),
        "near_support": near_support,
        "support_bounce": support_bounce,
        "consec_down":  consec_down,
        "mom_3d":       round(mom3, 2),
    }


def _score_volume_exhaustion(recent_price_rows: list[dict],
                              momentum: dict) -> tuple[float, dict]:
    """
    Component 5 — weight 0.10.

    OBV/volume signals of selling exhaustion and accumulation start.

    Analysis showed OBV is actually LOWER in FALSE_BLOCK (not yet rising),
    but the key is DECLINING selling volume (exhaustion) rather than
    rising buying volume. Also RSI slope positive is captured here.

    Scoring:
      Selling exhaustion (down-volume drying up) + RSI rising        → 80–100
      Volume spike (accumulation starting to show)                   → 60–80
      OBV-price divergence (OBV flat while price fell)               → 50–65
      Low volume on down days only                                   → 30–45
      No volume signal                                                → 10–25
    """
    if len(recent_price_rows) < 5:
        return 25.0, {"reason": "insufficient_data"}

    vols    = [r.get("vol", 0) for r in recent_price_rows]
    prices  = [r.get("price", 0) for r in recent_price_rows if r.get("price", 0) > 0]

    if not vols or not prices:
        return 25.0, {"reason": "no_volume_data"}

    # Down-day volumes vs up-day volumes (last 10 rows)
    window = list(zip(recent_price_rows[-10:], recent_price_rows[-9:] + [None]))
    down_vols, up_vols = [], []
    for i in range(1, len(recent_price_rows)):
        p_cur  = recent_price_rows[i].get("price", 0)
        p_prev = recent_price_rows[i-1].get("price", 0)
        v      = recent_price_rows[i].get("vol", 0)
        if p_cur < p_prev and v > 0:
            down_vols.append(v)
        elif p_cur > p_prev and v > 0:
            up_vols.append(v)

    def _avg(lst): return sum(lst) / len(lst) if lst else 1.0

    avg_down = _avg(down_vols[-5:]) if down_vols else 1.0
    avg_up   = _avg(up_vols[-5:])   if up_vols   else 1.0
    selling_exhaustion = avg_down < avg_up * 0.7   # down-day vol < 70% of up-day vol

    # Volume ratio: last 3d vs 10d average
    avg_vol10 = _avg([r.get("vol", 0) for r in recent_price_rows[-10:]])
    avg_vol3  = _avg([r.get("vol", 0) for r in recent_price_rows[-3:]])
    vol_ratio = avg_vol3 / avg_vol10 if avg_vol10 > 0 else 1.0
    vol_spike = vol_ratio > 1.5

    # OBV-price divergence heuristic: volume trending up while price down
    # Use vol_ratio > 1 while price_mom_7d < 0
    price_mom7 = (prices[-1] - prices[-8]) / prices[-8] * 100 if len(prices) >= 8 else 0.0
    obv_divergence = vol_ratio > 1.2 and price_mom7 < -2.0

    # RSI slope boost (captured in component 1 but relevant here too)
    rsi_turning = momentum.get("rsi_slope_3d", 0.0) > 0.3

    # Combine signals
    if selling_exhaustion and rsi_turning:
        base = 85.0
        label = "exhaustion+rsi_turning"
    elif vol_spike and obv_divergence:
        base = 75.0
        label = "vol_spike_divergence"
    elif selling_exhaustion:
        base = 60.0
        label = "selling_exhaustion"
    elif vol_spike:
        base = 55.0
        label = "vol_spike"
    elif obv_divergence:
        base = 45.0
        label = "obv_divergence"
    elif rsi_turning:
        base = 40.0
        label = "rsi_turning"
    else:
        base = 20.0
        label = "no_volume_signal"

    score = min(100.0, max(0.0, base))
    return score, {
        "reason":             label,
        "selling_exhaustion": selling_exhaustion,
        "vol_spike":          vol_spike,
        "obv_divergence":     obv_divergence,
        "vol_ratio":          round(vol_ratio, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ELIGIBILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_v3_eligible(
    tech_score:      int,
    recent_rows:     list[dict],
    momentum:        dict,
    gate_category:   str = "TECH_SCORE",
) -> tuple[bool, str]:
    """
    Returns (eligible: bool, reason: str).

    Pre-score gate — only call compute_v3_score() if this returns True.

    Rules:
      1. Gate was TECH_SCORE (not structural)
      2. tech_score >= V3_MIN_TECH_SCORE (not complete junk)
      3. At least one strong oversold signal in recent indicator history
      4. Not bounce_failed (dead-cat trap)
    """
    if gate_category not in ("TECH_SCORE", "RSI_NO_CONFIRM", "CONF_SCORE"):
        return False, f"gate_category={gate_category} not v3-rescueable"

    if tech_score < V3_MIN_TECH_SCORE:
        return False, f"tech_score={tech_score}<{V3_MIN_TECH_SCORE} absolute floor"

    if momentum.get("bounce_failed", False):
        return False, "bounce_failed — dead-cat trap"

    # Check for oversold RSI or lower-band touch in recent history
    rsi_min   = None
    pctb_min  = None
    for r in recent_rows:
        rsi_v = r.get("rsi_14")
        try:
            f = float(rsi_v)
            if f == f and 0 < f <= 100:
                rsi_min = min(rsi_min, f) if rsi_min is not None else f
        except (TypeError, ValueError):
            pass
        pctb_v = r.get("bb_pct_b")
        try:
            f = float(pctb_v)
            if f == f:
                pctb_min = min(pctb_min, f) if pctb_min is not None else f
        except (TypeError, ValueError):
            pass

    rsi_oversold  = rsi_min  is not None and rsi_min  < 35.0
    pctb_lower    = pctb_min is not None and pctb_min < 0.10

    if not rsi_oversold and not pctb_lower:
        return False, (
            f"no_oversold_signal: rsi_min={round(rsi_min,1) if rsi_min is not None else 'N/A'} "
            f"pctb_min={round(pctb_min,3) if pctb_min is not None else 'N/A'} "
            f"(need rsi<35 or pctb<0.10)"
        )

    return True, "eligible"


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════════════════════

def compute_v3_score(
    momentum:          dict,
    recent_rows:       list[dict],
    recent_price_rows: list[dict],
    sector:            str = "others",
) -> tuple[float, str, dict]:
    """
    Compute the v3 oversold-recovery composite score.

    Args:
        momentum:          output of filter_common._compute_momentum()
        recent_rows:       7-day indicator history, newest first
                           (same format as filter_common._load_recent_indicators)
        recent_price_rows: 20-day price history, oldest first
                           list of dicts with keys: date, price, high, low, vol
        sector:            sector string (for future sector-specific adjustments)

    Returns:
        (composite_score_0_100, primary_signal_label, debug_dict)
    """
    s_osr, d_osr = _score_oversold_recovery(momentum, recent_rows)
    s_bb,  d_bb  = _score_bb_lower_recovery(momentum, recent_rows)
    s_mac, d_mac = _score_macd_recovery(momentum, recent_rows)
    s_px,  d_px  = _score_price_structure(recent_price_rows)
    s_vol, d_vol = _score_volume_exhaustion(recent_price_rows, momentum)

    w = INDICATOR_WEIGHTS_V3
    total = (
        s_osr * w["oversold_recovery"] +
        s_bb  * w["bb_lower_recovery"] +
        s_mac * w["macd_recovery"]     +
        s_px  * w["price_structure"]   +
        s_vol * w["volume_exhaustion"]
    )

    component_scores = {
        "oversold_recovery": round(s_osr, 1),
        "bb_lower_recovery": round(s_bb,  1),
        "macd_recovery":     round(s_mac, 1),
        "price_structure":   round(s_px,  1),
        "volume_exhaustion": round(s_vol, 1),
    }

    # Primary signal = highest contributing component
    weighted = {k: component_scores[k] * w[k] for k in component_scores}
    primary  = max(weighted, key=weighted.get)

    label_map = {
        "oversold_recovery": "OVERSOLD_REVERSAL_V3",
        "bb_lower_recovery": "BB_BOUNCE_V3",
        "macd_recovery":     "MACD_RECOVERY_V3",
        "price_structure":   "SUPPORT_BOUNCE_V3",
        "volume_exhaustion": "EXHAUSTION_V3",
    }

    debug = {
        "scores":    component_scores,
        "weighted":  {k: round(v, 2) for k, v in weighted.items()},
        "details":   {
            "oversold_recovery": d_osr,
            "bb_lower_recovery": d_bb,
            "macd_recovery":     d_mac,
            "price_structure":   d_px,
            "volume_exhaustion": d_vol,
        }
    }

    return round(total, 2), label_map[primary], debug


# ══════════════════════════════════════════════════════════════════════════════
# CLI — standalone check
#   python filter_v3.py NABIL
#   python filter_v3.py --sample 10
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [V3] %(message)s")

    print("\n" + "=" * 70)
    print("  filter_v3.py — oversold recovery score standalone check")
    print("=" * 70)

    args = sys.argv[1:]

    try:
        from filter_common import _load_recent_indicators, _compute_momentum
        from datetime import datetime
        from config import NST
    except Exception as e:
        print(f"  Could not import pipeline modules: {e}")
        sys.exit(1)

    date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    if args and not args[0].startswith("--"):
        symbols = [a.upper() for a in args]
    else:
        # Pull today's tech_score-blocked symbols as demo targets
        try:
            from sheets import run_raw_sql
            rows = run_raw_sql(
                """
                SELECT DISTINCT symbol FROM gate_misses
                WHERE date = (SELECT MAX(date) FROM gate_misses)
                  AND gate_category = 'TECH_SCORE'
                LIMIT 15
                """
            )
            symbols = [r["symbol"] for r in (rows or [])]
            if not symbols:
                print("  No recent gate_misses — nothing to check.")
                sys.exit(0)
        except Exception as e:
            print(f"  Could not load symbols: {e}")
            sys.exit(1)

    recent_map = _load_recent_indicators(symbols, date, lookback=7)

    # Also load price history for price_structure and volume scoring
    price_map: dict = {}
    try:
        from sheets import run_raw_sql
        sym_csv = "','".join(symbols)
        ph = run_raw_sql(
            f"""
            SELECT symbol, date,
                   COALESCE(NULLIF(ltp,''), NULLIF(close,'')) as price,
                   high, low, volume as vol
            FROM price_history
            WHERE symbol IN ('{sym_csv}')
              AND date >= (CURRENT_DATE - INTERVAL '25 days')::text
            ORDER BY symbol, date ASC
            """
        )
        by_sym: dict = {}
        for r in (ph or []):
            s = r.get("symbol", "").upper()
            if s not in by_sym:
                by_sym[s] = []
            try:
                pv = float(r.get("price") or 0)
                if pv > 0:
                    by_sym[s].append({
                        "date":  r["date"],
                        "price": pv,
                        "high":  float(r.get("high") or pv),
                        "low":   float(r.get("low")  or pv),
                        "vol":   float(r.get("vol")  or 0),
                    })
            except (TypeError, ValueError):
                pass
        price_map = by_sym
    except Exception as e:
        logger.warning("Could not load price history: %s", e)

    from sheets import run_raw_sql
    ind_rows = run_raw_sql(
        """SELECT DISTINCT ON (symbol) symbol, tech_score
           FROM indicators WHERE date = %s""",
        (date,)
    ) or []
    tech_map = {r["symbol"]: int(float(r.get("tech_score") or 0)) for r in ind_rows}

    print(f"\n  {'Symbol':<10} {'Tech':>5} {'V3':>7} {'Primary':<24} {'Eligible'}")
    print("  " + "─" * 70)

    for sym in symbols:
        recent_rows  = recent_map.get(sym, [])
        price_rows   = price_map.get(sym, [])
        tech         = tech_map.get(sym, 0)
        momentum     = _compute_momentum(recent_rows)

        eligible, elig_reason = check_v3_eligible(
            tech_score=tech, recent_rows=recent_rows,
            momentum=momentum, gate_category="TECH_SCORE"
        )

        if not eligible:
            print(f"  {sym:<10} {tech:>5}   {'N/A':>6}  {'--':<24} NO ({elig_reason})")
            continue

        score, primary, dbg = compute_v3_score(
            momentum=momentum,
            recent_rows=recent_rows,
            recent_price_rows=price_rows,
            sector="others",
        )

        flag = "PASS" if score >= V3_COMPOSITE_THRESHOLD else "below_threshold"
        print(f"  {sym:<10} {tech:>5} {score:>7.1f}  {primary:<24} {flag}")

    print("\n" + "=" * 70 + "\n")
