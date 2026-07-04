"""
filter_v1.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — filter_v1 / filter_v2 split, Phase 1

v1 = SNAPSHOT scoring engine. Every function here reads today's (T0) frozen
indicator state only — no history, no slopes. This is the original,
production-validated engine (Karki et al. 2023 weights).

PURE MOVE from filter_engine.py — zero logic change. Verify with:
    python filter_engine.py --dry-run   (before vs after split, diff top-10)

Do not add slope/trend features here — that's what filter_v2.py is for.
Gates live in filter_common.py and are shared by both engines.
─────────────────────────────────────────────────────────────────────────────
"""

# Indicator weights — Karki et al. 2023
INDICATOR_WEIGHTS = {
    "macd":        0.35,
    "bollinger":   0.25,
    "sma":         0.20,
    "stochastic":  0.15,
    "rsi":         0.05,
}

# Optimal hold days — Karki et al. 2023
OPTIMAL_HOLD_DAYS = {
    "macd":       17,
    "sma":        33,
    "bollinger":  130,
    "stochastic": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR SCORING (snapshot / state-based)
# Evidence: Karki et al. 2023
# ══════════════════════════════════════════════════════════════════════════════

def _score_macd(ind: dict) -> float:
    """
    Weight 0.35 — highest. 23.64% ann. return, PF=2.97.
    Optimal hold: 17 days.
    """
    cross = str(ind.get("macd_cross", "NONE"))
    hist  = float(ind.get("macd_histogram", 0) or 0)

    if cross == "BULLISH":
        return 100.0
    elif cross == "BEARISH":
        return 0.0
    elif hist > 0:
        return min(65.0 + (hist * 10), 80.0)
    else:
        return max(20.0 + (hist * 5), 0.0)


def _score_bollinger(ind: dict, momentum_status: str = "NEUTRAL", momentum_score: float = 50.0) -> float:
    """
    Weight 0.25. PF=12.19 — best quality signal in NEPSE.
    LOWER_TOUCH uses continuous formula driven by momentum_score (0-100 → 10-65 range).
    """
    signal = str(ind.get("bb_signal", "NEUTRAL"))
    pct_b  = float(ind.get("bb_pct_b", 0.5) or 0.5)

    if signal == "LOWER_TOUCH":
        if momentum_status == "FALLING_KNIFE":
            return 15.0
        else:
            return round(10.0 + (momentum_score / 100.0) * 55.0, 1)  # 10–65 range
    elif signal == "SQUEEZE":
        return 75.0
    elif signal == "NEUTRAL":
        if pct_b < 0.3:
            return 55.0
        elif pct_b > 0.7:
            return 30.0
        return 45.0
    elif signal == "EXPANSION":
        return 40.0
    elif signal == "UPPER_TOUCH":
        return 5.0
    return 40.0


def _score_sma(ind: dict) -> float:
    """
    Weight 0.20. 21.33% ann. return, PF=3.86.
    Uses EMA crosses as SMA proxy. Optimal hold: 33 days.
    """
    ema_trend    = str(ind.get("ema_trend", "MIXED"))
    cross_20_50  = str(ind.get("ema_20_50_cross", "NONE"))
    cross_50_200 = str(ind.get("ema_50_200_cross", "NONE"))

    if ema_trend == "ABOVE_ALL" and cross_20_50 == "GOLDEN":
        return 100.0
    elif ema_trend == "ABOVE_ALL":
        return 75.0
    elif cross_20_50 == "GOLDEN":
        return 85.0
    elif cross_50_200 == "GOLDEN":
        return 80.0
    elif ema_trend == "MIXED":
        return 45.0
    elif ema_trend == "BELOW_ALL":
        return 10.0
    elif cross_20_50 == "DEATH" or cross_50_200 == "DEATH":
        return 10.0
    return 40.0


def _score_stochastic_proxy(ind: dict) -> float:
    """
    Weight 0.15. No stochastic in indicators.py — uses OBV trend
    + RSI momentum as proxy. Context only, not standalone trigger.
    """
    obv_trend = str(ind.get("obv_trend", "FLAT"))
    rsi       = float(ind.get("rsi_14", 50) or 50)

    if obv_trend == "RISING":
        if 40 <= rsi <= 60:
            return 80.0   # momentum building in neutral zone — sweet spot
        elif rsi < 40:
            return 70.0   # rising OBV from oversold
        else:
            return 55.0   # rising OBV but RSI elevated
    elif obv_trend == "FLAT":
        return 40.0
    else:
        return 15.0


def _score_rsi(ind: dict) -> float:
    """
    Weight 0.05. CONTEXT ONLY — -4.81% standalone in NEPSE (Karki 2023).
    Only contributes 5% to composite. Never the reason to buy alone.
    Best zone: 30-45 (recovering from oversold).
    """
    rsi = float(ind.get("rsi_14", 50) or 50)

    if rsi > 70:
        return 0.0      # blocked at gate — won't reach here normally
    elif rsi > 60:
        return 35.0
    elif rsi >= 45:
        return 60.0
    elif rsi >= 30:
        return 75.0     # recovering from oversold — best RSI zone
    else:
        return 50.0     # deeply oversold (needs MACD/BB — checked at gate)


def _compute_indicator_score(
    ind: dict,
    sector: str,
    momentum_status: str = "NEUTRAL",
    momentum_score: float = 50.0,
) -> tuple[float, str, int]:
    """
    Weighted composite indicator score (0–100).
    Returns (score, primary_signal, suggested_hold_days).
    """
    scores = {
        "macd":       _score_macd(ind)            * INDICATOR_WEIGHTS["macd"],
        "bollinger":  _score_bollinger(ind, momentum_status, momentum_score) * INDICATOR_WEIGHTS["bollinger"],
        "sma":        _score_sma(ind)              * INDICATOR_WEIGHTS["sma"],
        "stochastic": _score_stochastic_proxy(ind) * INDICATOR_WEIGHTS["stochastic"],
        "rsi":        _score_rsi(ind)              * INDICATOR_WEIGHTS["rsi"],
    }

    # Hydro RSI penalty: RSI -6.49% in Hydro (worst sector, Karki 2023)
    if "hydro" in sector.lower():
        scores["rsi"] *= 0.5

    total   = sum(scores.values())
    primary = max(scores, key=lambda k: scores[k])

    signal_labels = {
        "macd":       "MACD",
        "bollinger":  "BB",
        "sma":        "SMA",
        "stochastic": "OBV_MOMENTUM",
        "rsi":        "RSI",
    }

    return round(total, 2), signal_labels[primary], OPTIMAL_HOLD_DAYS.get(primary, 17)
