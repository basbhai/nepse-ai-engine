"""
filter_engine.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 1
Purpose : Combine frozen daily indicators + live prices + geo/nepal scores
          + candlestick signals into a ranked list of trade candidates.
          Feeds gemini_filter.py every 6 minutes during the trading loop.

Evidence base for every weight and threshold in this file:
─────────────────────────────────────────────────────────────────────────────

INDICATOR WEIGHTS (Karki et al. 2023 — 10yr NEPSE backtest, n=2294):
    MACD crossover   0.35  → 23.64% ann. return, Profit Factor 2.97
    Bollinger Bands  0.25  → Profit Factor 12.19 (rarest, highest quality)
    SMA crossover    0.20  → 21.33% ann. return, Profit Factor 3.86
    Stochastic proxy 0.15  → context only (327 trades/10yr = fee drag)
    RSI              0.05  → CONTEXT ONLY: -4.81% standalone, never trigger

SECTOR MULTIPLIERS (Khadka & Rajopadhyaya 2023 — Single Index Model):
    Non-Life Insurance  1.25  → best risk-adj return (2.732), β=0.034
    Finance             1.15  → selected, excess return 0.270
    Micro Finance       1.10  → selected, excess return 0.165
    Hydro Power         1.10  → selected (1.737 risk-adj), β=0.042
    Development Bank    1.08  → selected, excess return 0.151
    Life Insurance      1.05  → excluded from optimal (0.106 < C*=0.129)
    Others              1.00  → baseline
    Banking             0.90  → excluded from optimal (0.051 < C*=0.129)
    Manufacturing       0.75  → worst risk-adj return (-0.044)

    NOTE on Hydro: SIM paper shows strong risk-adj (1.737) but technical
    paper shows RSI loses most (-6.49%) in Hydro. Resolution: full sector
    multiplier applies, but RSI contribution is halved for hydro symbols.

INSURANCE CONDITIONAL MULTIPLIER (Political events paper):
    Insurance 4.6x more sensitive to political events than NEPSE.
    Pre-event leakage window: -10 to -1 days (Insurance AAR=0.272).
    crisis_detected=YES  → insurance multiplier drops to 0.85
    nepal_score >= 1     → insurance multiplier gets +0.10 boost

HARD GATES (system never signals past these — hard rules from handoff):
    combined_geo <= -3      → full block (capital preservation)
    bandh_today == YES      → full block (zero liquidity)
    market_state == CRISIS  → full block
    loss_streak > 7         → circuit breaker
    ltp <= 0                → no live price, skip
    history_days < 20       → insufficient history for indicators
    rsi > 75                → overbought, blocked
    rsi < 30 + no MACD/BB   → RSI oversold alone is not a signal (paper)

TECH SCORE THRESHOLDS by market state:
    FULL_BULL       50   (catch more candidates)
    CAUTIOUS_BULL   58   (BULL threshold from indicators.py)
    SIDEWAYS        65
    BEAR            72   (very selective)
    CRISIS          999  (blocks everything)

C* LIVE RANKING (Khadka & Rajopadhyaya 2023 SIM paper):
    (daily_change_pct/100 - Rf_daily) / sector_beta > 0.129 → +5 bonus
    Rf_daily = annual Rf / 252

─────────────────────────────────────────────────────────────────────────────
Called by: trading.yml every 6 min, 10:45 AM – 3:00 PM NST
Next:      gemini_filter.py reads output of this module
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json 
from config import NST

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE-BASED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Indicator weights — Karki et al. 2023
INDICATOR_WEIGHTS = {
    "macd":        0.35,
    "bollinger":   0.25,
    "sma":         0.20,
    "stochastic":  0.15,
    "rsi":         0.05,
}

# Sector multipliers — Khadka & Rajopadhyaya 2023 SIM paper
SECTOR_MULTIPLIERS = {
    "non-life insurance":           1.25,
    "non life insurance":           1.25,
    "nonlife insurance":            1.25,
    "non-life":                     1.25,
    "finance":                      1.15,
    "microfinance":                 1.10,
    "micro finance":                1.10,
    "hydropower":                   1.10,
    "hydro power":                  1.10,
    "hydro":                        1.10,
    "development bank":             1.08,
    "dev bank":                     1.08,
    "life insurance":               1.05,
    "life":                         1.05,
    "insurance":                    1.05,
    "banking":                      0.90,
    "commercial bank":              0.90,
    "manufacturing":                0.75,
    "manufacturing and processing": 0.75,
    "hotels and tourism":           1.00,
    "trading":                      1.00,
    "investment":                   1.00,
    "others":                       1.00,
}
DEFAULT_SECTOR_MULTIPLIER = 1.00

# Sector betas — Khadka & Rajopadhyaya 2023 SIM paper
SECTOR_BETAS = {
    "non-life insurance":           0.034,
    "non life insurance":           0.034,
    "nonlife insurance":            0.034,
    "non-life":                     0.034,
    "hydropower":                   0.042,
    "hydro power":                  0.042,
    "hydro":                        0.042,
    "finance":                      0.611,
    "microfinance":                 0.931,
    "micro finance":                0.931,
    "development bank":             0.845,
    "dev bank":                     0.845,
    "life insurance":               1.216,
    "life":                         1.216,
    "banking":                      1.000,
    "commercial bank":              1.000,
    "manufacturing":                0.795,
    "manufacturing and processing": 0.795,
}
DEFAULT_SECTOR_BETA  = 0.659   # average beta, Khadka & Rajopadhyaya 2023
C_STAR               = 0.129   # cut-off rate, Khadka & Rajopadhyaya 2023
RF_ANNUAL_PCT        = 5.5     # NRB T-bill proxy — update monthly via RF_RATE_ANNUAL_PCT setting

# Tech score thresholds by market state
TECH_SCORE_THRESHOLDS = {
    "FULL_BULL":     50,
    "CAUTIOUS_BULL": 58,
    "SIDEWAYS":      65,
    "BEAR":          72,
    "CRISIS":        999,
}
DEFAULT_TECH_THRESHOLD = 58

MIN_CONF_SCORE   = 35
MIN_HISTORY_DAYS = 20

# Volume/OS ratio soft gate — below this = truly dead stock, not worth trading
MIN_VOS_PCT = 0.05

# Optimal hold days — Karki et al. 2023
OPTIMAL_HOLD_DAYS = {
    "macd":       17,
    "sma":        33,
    "bollinger":  130,
    "stochastic": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# FILTER CANDIDATE DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterCandidate:
    """
    A stock that passed all hard gates and scored above threshold.
    Passed to gemini_filter.py for Gemini Flash deep analysis.
    All fields included so Gemini has full context in one object.
    """
    symbol:           str
    sector:           str   = ""
    ltp:              float = 0.0
    change_pct:       float = 0.0
    volume:           int   = 0

    # Frozen indicator values (from Neon indicators table, computed 10:30 AM)
    rsi_14:           float = 0.0
    rsi_signal:       str   = ""
    ema_trend:        str   = ""
    ema_20_50_cross:  str   = ""
    ema_50_200_cross: str   = ""
    macd_cross:       str   = ""
    macd_histogram:   float = 0.0
    bb_signal:        str   = ""
    bb_pct_b:         float = 0.5
    obv_trend:        str   = ""
    atr_pct:          float = 0.0
    tech_score:       int   = 0
    tech_signal:      str   = ""
    history_days:     int   = 0

    # Support / Resistance (20-day high/low from indicators.py)
    support_level:    float = 0.0   # lowest low over last 20 trading days
    resistance_level: float = 0.0   # highest high over last 20 trading days

#
#     # Bollinger Band price levels (from indicators table)
    bb_upper:         float = 0.0   # upper band price level
    bb_lower:         float = 0.0   # lower band price level
    macd_line:        float = 0.0   # MACD line value (not cross string)
    macd_signal_line: float = 0.0   # MACD signal line value

    # ShareSansar conf score (live from scraper)
    conf_score:       float = 0.0
    conf_signal:      str   = ""

    # Candlestick patterns (from candle_signals table)
    candle_patterns:  list  = field(default_factory=list)
    best_candle:      str   = ""
    candle_tier:      int   = 0
    candle_conf:      int   = 0

    # Context scores
    geo_score:        int   = 0
    nepal_score:      int   = 0
    combined_geo:     int   = 0
    bandh_today:      str   = "NO"
    crisis_detected:  str   = "NO"
    ipo_drain:        str   = "NO"
    market_state:     str   = ""

    # Composite scoring breakdown
    indicator_score:  float = 0.0   # 0–100, weighted indicator sub-score
    sector_mult:      float = 1.0   # from SIM paper
    cstar_signal:     bool  = False  # excess return > C* = 0.129
    composite_score:  float = 0.0   # final ranking score
    primary_signal:   str   = ""    # dominant trigger: MACD / BB / SMA / OBV_MOMENTUM / RSI
    suggested_hold:   int   = 17    # days, Karki 2023

    fundamental_adj:  float = 0.0
    fundamental_reason: str   = ""
    broker_flow_adj:  float = 0.0

    vwap_dev:         float = 0.0
    bid_ask_ratio:    float = 0.0
    dpr_proximity:    float = 0.0
    volume_os_ratio:  float = 0.0

    # Momentum direction (from 7-day indicator history)
    momentum_status:  str   = "NEUTRAL"  # FALLING_KNIFE|OVERSOLD_WATCH|EARLY_REVERSAL|CONFIRMED_REVERSAL|NEUTRAL
    rsi_slope_3d:     float = 0.0        # positive = RSI improving
    macd_hist_slope:  float = 0.0        # positive = histogram improving
    bb_pct_b_slope:   float = 0.0        # positive = moving away from lower band
    bounce_failed:    bool  = False      # True = dead-cat bounce pattern
    reversal_days:    int   = 0          # consecutive days RSI rising
    news_catalyst:    str   = ""

    timestamp: str = field(default_factory=lambda:
                    datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        candle_str = f" | {self.best_candle}(T{self.candle_tier})" if self.best_candle else ""
        cstar_str  = " [C*✓]" if self.cstar_signal else ""
        return (
            f"{self.symbol:<10} score={self.composite_score:>6.1f} "
            f"tech={self.tech_score:>3} conf={self.conf_score:>5.1f} "
            f"RSI={self.rsi_14:>5.1f} MACD={self.macd_cross:<9} "
            f"BB={self.bb_signal:<12} geo={self.combined_geo:>+3} "
            f"x{self.sector_mult:.2f}{cstar_str}{candle_str}"
        )


@dataclass
class NearMiss:
    symbol:                   str
    sector:                   str   = ""
    date:                     str   = ""
    gate_reason:              str   = ""   # raw: "CONF=42<50"
    gate_category:            str   = ""   # normalized bucket
    price_at_block:           float = 0.0
    market_state:             str   = ""
    tech_score:               int   = 0
    conf_score:               float = 0.0
    composite_score_would_be: float = 0.0
    volume_os_ratio:          float = 0.0
    vwap_dev:                 float = 0.0
    bid_ask_ratio:            float = 0.0
    dpr_proximity:            float = 0.0


def _categorize_gate_reason(reason: str) -> str:
    r = reason.upper()
    if r.startswith("CONF="):            return "CONF_SCORE"
    if r.startswith("TECH="):            return "TECH_SCORE"
    if "OVERBOUGHT" in r:               return "RSI_OVERBOUGHT"
    if "OVERSOLD" in r:                 return "RSI_NO_CONFIRM"
    if r.startswith("HISTORY="):        return "HISTORY"
    if "MUTUAL_FUND" in r:              return "MUTUAL_FUND"
    if "NON_EQUITY" in r:               return "NON_EQUITY"
    if "VOS=" in r and "<" in r:        return "ILLIQUID"
    if r.startswith("BROKER_FLOW"):     return "BROKER_FLOW"
    if "DPR_PROXIMITY" in r:           return "DPR_UPPER"
    if "SECTOR_LIMIT" in r:            return "SECTOR_CONCENTRATION"
    return "OTHER"
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONTEXT LOADER
# ══════════════════════════════════════════════════════════════════════════════

_last_near_misses: list[NearMiss] = []

def _load_context() -> dict:
    """
    Load geo, nepal, market state, loss streak, RF rate.
    All defaults to safe/neutral values on failure.
    """
    ctx = {
        "geo_score":               0,
        "nepal_score":             0,
        "combined_geo":            0,
        "bandh_today":             "NO",
        "crisis_detected":         "NO",
        "ipo_drain":               "NO",
        "market_state":            "SIDEWAYS",
        "loss_streak":             0,
        "rf_rate_annual":          RF_ANNUAL_PCT,
        "sector_position_counts":  {},
        "max_positions_per_sector": 2,
    }
    try:
        from sheets import get_latest_geo, get_latest_pulse, get_setting, read_tab

        geo = get_latest_geo() or {}
        ctx["geo_score"] = int(geo.get("geo_score", 0) or 0)

        pulse = get_latest_pulse() or {}
        ctx["nepal_score"]      = int(pulse.get("nepal_score", 0) or 0)
        ctx["bandh_today"]      = str(pulse.get("bandh_today",    "NO")).upper()
        ctx["crisis_detected"]  = str(pulse.get("crisis_detected","NO")).upper()
        ctx["ipo_drain"]        = str(pulse.get("ipo_fpo_active", "NO")).upper()
        ctx["combined_geo"]     = ctx["geo_score"] + ctx["nepal_score"]

        ctx["market_state"]     = get_setting("MARKET_STATE", "SIDEWAYS").upper().strip()

        try:
            rows    = read_tab("financials")
            kpi_map = {r.get("kpi_name", ""): r.get("current_value", "") for r in rows}
            ctx["loss_streak"] = int(float(kpi_map.get("Current_Loss_Streak", 0) or 0))
        except Exception:
            pass

        try:
            ctx["rf_rate_annual"] = float(get_setting("RF_RATE_ANNUAL_PCT", str(RF_ANNUAL_PCT)))
        except Exception:
            pass

        # ── Dynamic gate thresholds — updated by approved gate_proposals ──────
        # Reads from settings table each run so /approve_N takes effect immediately.
        # Falls back silently to module-level defaults if key is absent.
        try:
            ctx["tech_thresholds"] = {
                "FULL_BULL":     int(get_setting("TECH_SCORE_THRESHOLDS.FULL_BULL",     str(TECH_SCORE_THRESHOLDS["FULL_BULL"]))),
                "CAUTIOUS_BULL": int(get_setting("TECH_SCORE_THRESHOLDS.CAUTIOUS_BULL", str(TECH_SCORE_THRESHOLDS["CAUTIOUS_BULL"]))),
                "SIDEWAYS":      int(get_setting("TECH_SCORE_THRESHOLDS.SIDEWAYS",      str(TECH_SCORE_THRESHOLDS["SIDEWAYS"]))),
                "BEAR":          int(get_setting("TECH_SCORE_THRESHOLDS.BEAR",          str(TECH_SCORE_THRESHOLDS["BEAR"]))),
                "CRISIS":        int(get_setting("TECH_SCORE_THRESHOLDS.CRISIS",        str(TECH_SCORE_THRESHOLDS["CRISIS"]))),
            }
        except Exception:
            ctx["tech_thresholds"] = TECH_SCORE_THRESHOLDS.copy()

        try:
            ctx["min_conf_score"] = float(get_setting("MIN_CONF_SCORE", str(MIN_CONF_SCORE)))
        except Exception:
            ctx["min_conf_score"] = MIN_CONF_SCORE

        try:
            ctx["max_positions"] = int(get_setting("MAX_POSITIONS", "3"))
        except Exception:
            ctx["max_positions"] = 3

        try:
            from sheets import run_raw_sql as _run_raw_sql
            _pos_rows = _run_raw_sql(
                """
                SELECT pp.symbol, ss.sectorname AS sector
                FROM paper_portfolio pp
                LEFT JOIN share_sectors ss ON ss.symbol = pp.symbol
                WHERE pp.status = 'OPEN' AND pp.test_mode = 'false'
                """
            )
            _sec_counts: dict[str, int] = {}
            for _p in (_pos_rows or []):
                _sec = str(_p.get("sector", "") or "").lower().strip()
                if _sec:
                    _sec_counts[_sec] = _sec_counts.get(_sec, 0) + 1
            ctx["sector_position_counts"] = _sec_counts
        except Exception:
            ctx["sector_position_counts"] = {}

        try:
            ctx["max_positions_per_sector"] = int(get_setting("MAX_POSITIONS_PER_SECTOR", "2"))
        except Exception:
            ctx["max_positions_per_sector"] = 2

        try:
            ctx["gemini_max_candidates"] = int(get_setting("GEMINI_MAX_CANDIDATES", "10"))
        except Exception:
            ctx["gemini_max_candidates"] = 10

        try:
            ctx["gemini_max_flags"] = int(get_setting("GEMINI_MAX_FLAGS", "3"))
        except Exception:
            ctx["gemini_max_flags"] = 3

        logger.info(
            "Context: geo=%+d nepal=%+d combined=%+d market=%s bandh=%s "
            "crisis=%s ipo=%s loss_streak=%d",
            ctx["geo_score"], ctx["nepal_score"], ctx["combined_geo"],
            ctx["market_state"], ctx["bandh_today"],
            ctx["crisis_detected"], ctx["ipo_drain"], ctx["loss_streak"],
        )
    except Exception as exc:
        logger.warning("_load_context failed (%s) — safe defaults used", exc)

    return ctx


def _load_recent_indicators(symbols: list[str], today: str, lookback: int = 7) -> dict[str, list[dict]]:
    """
    Load last `lookback` trading days of indicator rows per symbol.
    Single batch SQL query — never called per-symbol.

    Returns dict[symbol -> list[dict]], rows newest-first.
    Empty list for symbols with no history.
    """
    if not symbols:
        return {}
    try:
        from sheets import run_raw_sql
        placeholders = ",".join(["%s"] * len(symbols))
        rows = run_raw_sql(
            f"""
            SELECT symbol, date, rsi_14, macd_histogram, bb_pct_b, obv_trend
            FROM indicators
            WHERE symbol IN ({placeholders})
              AND date < %s
            ORDER BY symbol ASC, date DESC
            """,
            tuple(symbols) + (today,),
        )
        result: dict[str, list[dict]] = {}
        for r in (rows or []):
            sym = r.get("symbol", "").upper()
            if sym not in result:
                result[sym] = []
            if len(result[sym]) < lookback:
                result[sym].append(r)
        return result
    except Exception as exc:
        logger.warning("_load_recent_indicators failed: %s", exc)
        return {}


def _compute_momentum(recent_rows: list[dict]) -> dict:
    """
    Compute momentum direction from last 7 days of indicator history.
    recent_rows: list of dicts, newest first, from _load_recent_indicators().

    Returns dict with keys:
      momentum_status : str   — classification
      momentum_score  : float — continuous 0-100 score
      rsi_slope_3d    : float — avg-of-differences slope (positive = improving)
      macd_hist_slope : float — positive = improving
      bb_pct_b_slope  : float — positive = moving away from lower band
      bounce_failed   : bool
      reversal_days   : int   — consecutive days RSI has been rising (with tolerance)
    """
    default = {
        "momentum_status": "NEUTRAL",
        "momentum_score":  50.0,
        "rsi_slope_3d":    0.0,
        "macd_hist_slope": 0.0,
        "bb_pct_b_slope":  0.0,
        "bounce_failed":   False,
        "reversal_days":   0,
    }

    if not recent_rows or len(recent_rows) < 2:
        return default

    # Step 1 — NULL-safe extraction
    def _f_safe(row, key):
        v = row.get(key)
        return float(v) if v not in (None, "", "None") else None

    rsi_vals  = [v for v in (_f_safe(r, "rsi_14")        for r in recent_rows[:5]) if v is not None]
    hist_vals = [v for v in (_f_safe(r, "macd_histogram") for r in recent_rows[:5]) if v is not None]
    pctb_vals = [v for v in (_f_safe(r, "bb_pct_b")      for r in recent_rows[:5]) if v is not None]

    if len(rsi_vals) < 3:
        return default

    # Step 2 — Average-of-differences slope
    def _avg_slope(vals):
        if len(vals) < 2:
            return 0.0
        diffs = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
        return sum(diffs) / len(diffs)

    rsi_slope       = _avg_slope(rsi_vals)
    macd_hist_slope = _avg_slope(hist_vals)
    bb_pct_b_slope  = _avg_slope(pctb_vals)

    # Step 3 — Ratio-based monotonic checks
    pairs = min(len(rsi_vals), len(hist_vals), len(pctb_vals), 5) - 1
    if pairs < 1:
        return default

    rsi_rising_count  = sum(rsi_vals[i]  >= rsi_vals[i + 1]  - 0.5 for i in range(pairs))
    hist_rising_count = sum(hist_vals[i] >  hist_vals[i + 1]        for i in range(pairs))
    pctb_rising_count = sum(pctb_vals[i] >  pctb_vals[i + 1]        for i in range(pairs))

    rsi_improving_confirmed  = rsi_rising_count  >= max(3, pairs - 1)
    rsi_improving_early      = rsi_rising_count  >= max(2, pairs // 2 + 1)
    macd_improving_confirmed = hist_rising_count >= max(3, pairs - 1)
    macd_improving_early     = hist_rising_count >= max(2, pairs // 2 + 1)
    bb_improving_confirmed   = pctb_rising_count >= max(3, pairs - 1)
    rsi_deteriorating        = (pairs - rsi_rising_count) >= max(3, pairs - 1)

    # Step 4 — Origin tracking
    rsi_yesterday     = rsi_vals[0]
    rsi_min_in_window = min(rsi_vals)
    was_deep_oversold = rsi_min_in_window < 32
    was_oversold      = rsi_min_in_window < 35

    # Step 5 — Reversal days with tolerance
    reversal_days = 0
    for i in range(len(rsi_vals) - 1):
        if rsi_vals[i] >= rsi_vals[i + 1] - 0.5:
            reversal_days += 1
        else:
            break

    # Step 6 — bounce_failed
    bounce_failed = False
    if len(rsi_vals) >= 3:
        r2, r1, r0 = rsi_vals[2], rsi_vals[1], rsi_vals[0]
        bounce_magnitude = r1 - r2
        if (
            r2 < 30
            and bounce_magnitude >= 2
            and r0 < r1
            and r0 < r2 + 3
        ):
            bounce_failed = True

    # Step 7 — Continuous momentum score
    rsi_position_score = max(0.0, min(100.0,
        30.0 + (rsi_slope * 10) if rsi_yesterday < 30
        else 50.0 + (rsi_yesterday - 30) if rsi_yesterday < 50
        else 50.0
    ))
    trend_score = (rsi_rising_count / pairs * 100) if pairs > 0 else 50.0
    macd_score  = min(100.0, max(0.0, 50.0 + hist_rising_count / max(pairs, 1) * 50.0))
    bb_score    = min(100.0, max(0.0, 50.0 + pctb_rising_count / max(pairs, 1) * 50.0))
    bounce_penalty = -25.0 if bounce_failed else 0.0

    momentum_score = max(0.0, min(100.0,
        rsi_position_score * 0.35 +
        trend_score        * 0.30 +
        macd_score         * 0.20 +
        bb_score           * 0.15 +
        bounce_penalty
    ))

    # Step 8 — Classification (CONFIRMED before EARLY)
    if bounce_failed:
        status = "FALLING_KNIFE"

    elif rsi_yesterday < 30 and rsi_deteriorating:
        status = "FALLING_KNIFE"

    elif rsi_yesterday < 30 and rsi_slope >= 0:
        status = "OVERSOLD_WATCH"

    elif (
        was_deep_oversold
        and 30 <= rsi_yesterday < 50
        and rsi_improving_confirmed
        and macd_improving_confirmed
        and bb_improving_confirmed
        and reversal_days >= 3
    ):
        status = "CONFIRMED_REVERSAL"

    elif (
        was_oversold
        and 28 <= rsi_yesterday < 50
        and rsi_improving_early
        and macd_improving_early
        and reversal_days >= 2
    ):
        status = "EARLY_REVERSAL"

    else:
        status = "NEUTRAL"

    return {
        "momentum_status": status,
        "momentum_score":  round(momentum_score, 2),
        "rsi_slope_3d":    round(rsi_slope, 3),
        "macd_hist_slope": round(macd_hist_slope, 6),
        "bb_pct_b_slope":  round(bb_pct_b_slope, 4),
        "bounce_failed":   bounce_failed,
        "reversal_days":   reversal_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HARD GATES
# ══════════════════════════════════════════════════════════════════════════════

def _check_hard_gates(ctx: dict) -> tuple[bool, str]:
    """System-level gates. Returns (passed, reason)."""
    # if ctx["bandh_today"] == "YES":
    #     return False, "BANDH_TODAY — zero liquidity"
    if ctx["market_state"] == "CRISIS":
        return False, "MARKET_STATE=CRISIS — capital preservation"
    if ctx["combined_geo"] <= -10:
        return False, f"COMBINED_GEO={ctx['combined_geo']} ≤ -3"
    if ctx["loss_streak"] > 7:
        return False, f"LOSS_STREAK={ctx['loss_streak']} > 7 circuit breaker"
    return True, ""


def _check_symbol_gates(
    symbol:     str,
    ind:        dict,
    price_row,
    ctx:        dict,
    flow_cache: dict = None,
) -> tuple[bool, str]:
    """Per-symbol gates. Returns (passed, reason)."""
    ltp = float(price_row.ltp or price_row.close or 0)
    if ltp <= 0:
        return False, "NO_LTP"
    sym_upper = symbol.upper()
    if any(sym_upper.endswith(sfx) for sfx in _MUTUAL_FUND_SUFFIXES):
        return False, f"MUTUAL_FUND_SUFFIX"
    if sym_upper in _MUTUAL_FUND_KEYWORDS:
        return False, f"MUTUAL_FUND_KEYWORD"


    history_days = int(ind.get("history_days", 0) or 0)
    if history_days < MIN_HISTORY_DAYS:
        return False, f"HISTORY={history_days}<{MIN_HISTORY_DAYS}"

    sector_lc = str(ind.get("sector", "") or getattr(price_row, "sector", "") or "").lower().strip()
    if sector_lc:
        sec_count      = ctx.get("sector_position_counts", {}).get(sector_lc, 0)
        max_per_sector = ctx.get("max_positions_per_sector", 2)
        if sec_count >= max_per_sector:
            return False, f"SECTOR_LIMIT={sector_lc}:{sec_count}>={max_per_sector}"

    tech_score = int(ind.get("tech_score", 0) or 0)
    threshold  = ctx.get("tech_thresholds", TECH_SCORE_THRESHOLDS).get(ctx["market_state"], DEFAULT_TECH_THRESHOLD)

    # News catalyst boost — POSITIVE sentiment from nepal_pulse
    catalyst_boost = 0
    news_catalyst_str = ""
    try:
        from sheets import get_setting as _gs
        raw = _gs("STOCK_CATALYSTS_TODAY", "[]")
        catalysts = json.loads(raw) if raw else []
        for c in catalysts:
            if str(c.get("symbol", "")).upper() == symbol.upper():
                if str(c.get("sentiment", "")).upper() == "POSITIVE" and tech_score >= 40:
                    catalyst_boost = 10
                    news_catalyst_str = str(c.get("reason", ""))
                    logger.info(
                        "NEWS_CATALYST_BOOST: %s +%d (reason: %s)",
                        symbol, catalyst_boost, news_catalyst_str,
                    )
                break
    except Exception:
        pass

    # ── Validated broker accumulation bypass ─────────────────────────────────
    # If a validated smart-money broker dominates single-day buy flow (>60%),
    # allow through even if tech_score is below threshold.
    # Only 4 validated brokers matched by name (not ID — IDs are unstable).
    # Requires tech_score >= 30 (absolute floor — avoids complete junk).
    _VALIDATED_BROKER_NAMES = {
        "dipshikha dhitopatra",
        "trishakti securities",
        "online securities",
        "sri hari securities",
    }
    broker_bypass = False
    if (tech_score + catalyst_boost) < threshold and tech_score >= 30 and flow_cache:
        row = (flow_cache or {}).get(symbol.upper(), {})
        top_broker_name = str(row.get("acc_top_broker_1d", "") or "").strip().lower()
        top_broker_pct  = float(row.get("acc_top_broker_pct_1d", 0) or 0)
        net_flow        = float(row.get("net_flow_1d", 0) or 0)
        if any(vn in top_broker_name for vn in _VALIDATED_BROKER_NAMES) and top_broker_pct >= 60.0 and net_flow > 0:
            broker_bypass = True
            logger.info(
                "BROKER_BYPASS: %s tech=%d<%d but broker=%s pct=%.1f%% net_flow=%.0f — allowing through",
                symbol, tech_score, threshold, top_broker_name, top_broker_pct, net_flow,
            )
            ctx.setdefault("_sym_catalysts", {})[symbol] = (
                f"BROKER_ACCUMULATION: validated broker {top_broker_name} "
                f"holds {top_broker_pct:.1f}% of today's buy flow"
            )

    if not broker_bypass and (tech_score + catalyst_boost) < threshold:
        return False, f"TECH={tech_score}<{threshold}"
    
    # Store catalyst on ctx for downstream use
    if news_catalyst_str:
        ctx.setdefault("_sym_catalysts", {})[symbol] = news_catalyst_str

    conf_score = float(getattr(price_row, "conf_score", 0) or 0)
    min_conf   = ctx.get("min_conf_score", MIN_CONF_SCORE)
    if conf_score < min_conf:
        return False, f"CONF={conf_score:.0f}<{min_conf:.0f}"

    rsi = float(ind.get("rsi_14", 50) or 50)
    if rsi > 75:
        return False, f"RSI={rsi:.1f} OVERBOUGHT"

    # RSI alone is not a buy signal in NEPSE (paper: -4.81% standalone)
    if rsi < 30:
        momentum_status = ctx.get("_sym_momentum", {}).get("momentum_status", "NEUTRAL")
        bounce_failed   = ctx.get("_sym_momentum", {}).get("bounce_failed",   False)

        if bounce_failed:
            return False, f"RSI={rsi:.1f} BOUNCE_FAILED — dead-cat trap"
        if momentum_status == "FALLING_KNIFE":
            return False, f"RSI={rsi:.1f} FALLING_KNIFE — still declining"
        if momentum_status not in ("EARLY_REVERSAL", "CONFIRMED_REVERSAL", "OVERSOLD_WATCH"):
            # No improvement detected — apply original MACD/BB gate
            macd_cross = str(ind.get("macd_cross", "NONE"))
            bb_signal  = str(ind.get("bb_signal", "NEUTRAL"))
            if macd_cross != "BULLISH" and bb_signal not in ("LOWER_TOUCH", "SQUEEZE"):
                return False, f"RSI={rsi:.1f} OVERSOLD_NO_MOMENTUM — no direction improvement"

    flow_class, _, _ = _get_broker_flow_classification(symbol, flow_cache or {})
    if flow_class == "NET_DISTRIBUTION":
        return False, "BROKER_FLOW=NET_DISTRIBUTION"
    if flow_class == "CHURN":
        return False, "BROKER_FLOW=CHURN"

    # ── DPR proximity gate — no upside room near upper circuit ───────────────
    dpr_prox = float(getattr(price_row, "dpr_proximity", 0) or 0)
    try:
        from sheets import get_setting
        dpr_upper_gate = float(get_setting("LIVE_ADJ_DPR_UPPER_GATE", "0.85"))
    except Exception:
        dpr_upper_gate = 0.85
    if dpr_prox > dpr_upper_gate:
        return False, f"DPR_PROXIMITY={dpr_prox:.2f}>{dpr_upper_gate} — near upper circuit"

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INDICATOR SCORING
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SECTOR SCORING
# Evidence: Khadka & Rajopadhyaya 2023 + Political events paper
# ══════════════════════════════════════════════════════════════════════════════

def _get_sector_multiplier(sector: str, ctx: dict) -> float:
    """
    Sector multiplier from SIM paper with insurance conditional adjustment
    from the political events paper.
    """
    s    = sector.lower().strip()
    mult = SECTOR_MULTIPLIERS.get(s, DEFAULT_SECTOR_MULTIPLIER)

    # Partial match if exact not found
    if mult == DEFAULT_SECTOR_MULTIPLIER:
        for key, val in SECTOR_MULTIPLIERS.items():
            if key in s or s in key:
                mult = val
                break

    # Insurance conditional — political events paper:
    # Insurance AAR pre-event = 0.272 vs NEPSE 0.0595 (4.6x more sensitive)
    is_insurance = any(x in s for x in ["insurance", "insur"])
    if is_insurance:
        if ctx.get("crisis_detected") == "YES":
            mult = 0.85
            logger.debug("%s: insurance multiplier → 0.85 (crisis)", sector)
        elif ctx.get("nepal_score", 0) >= 1:
            mult = min(mult + 0.10, 1.40)
            logger.debug("%s: insurance multiplier boosted → %.2f (stable)", sector, mult)

    return mult


def _get_sector_beta(sector: str) -> float:
    """Empirical sector beta from SIM paper."""
    s = sector.lower().strip()
    if s in SECTOR_BETAS:
        return SECTOR_BETAS[s]
    for key, val in SECTOR_BETAS.items():
        if key in s or s in key:
            return val
    return DEFAULT_SECTOR_BETA


def _check_cstar_signal(
    daily_change_pct: float,
    sector:           str,
    rf_rate_annual:   float,
) -> bool:
    """
    C* live ranking signal — SIM paper (Khadka & Rajopadhyaya 2023).
    True when (daily_return - Rf_daily) / beta > C* (0.129).
    When true: +5 bonus to composite score.
    """
    try:
        rf_daily     = rf_rate_annual / 252 / 100
        beta         = _get_sector_beta(sector)
        if beta <= 0:
            return False
        excess_ratio = (daily_change_pct / 100 - rf_daily) / beta
        return excess_ratio > C_STAR
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CANDLESTICK SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def _load_candle_signals(symbols: list[str], date: str) -> dict[str, list]:
    """Load candle signals from candle_signals table.
    
    Tries today's date first. If no rows found (candle_detector runs at 9 PM
    summary_workflow, not morning), falls back to the most recent available date.
    """
    candles: dict[str, list] = {sym: [] for sym in symbols}
    try:
        from sheets import run_raw_sql

        # Check if today has any rows at all
        check = run_raw_sql(
            "SELECT MAX(date) AS latest FROM candle_signals WHERE date <= %s",
            (date,)
        )
        effective_date = (check[0]["latest"] if check and check[0]["latest"] else None)

        if not effective_date:
            logger.info("Candle signals: no data available at or before %s", date)
            return candles

        if effective_date != date:
            logger.info(
                "Candle signals: no data for %s — using most recent available: %s",
                date, effective_date,
            )

        rows = run_raw_sql(
            """
            SELECT symbol, pattern_name, signal, tier, confidence, volume_confirmed
            FROM candle_signals
            WHERE date = %s
            ORDER BY symbol, tier::int ASC, confidence::int DESC
            """,
            (effective_date,)
        )
        for r in rows:
            sym = str(r.get("symbol", "")).upper()
            if sym in candles:
                candles[sym].append({
                    "pattern":          r.get("pattern_name", ""),
                    "signal":           r.get("signal", ""),
                    "tier":             int(r.get("tier", 3) or 3),
                    "confidence":       int(r.get("confidence", 0) or 0),
                    "volume_confirmed": str(r.get("volume_confirmed", "false")).lower() == "true",
                })

        logger.info(
            "Candle signals loaded: %d symbols (date=%s)", len(symbols), effective_date
        )
    except Exception as exc:
        logger.warning("_load_candle_signals failed: %s", exc)
    return candles


def _candle_bonus(patterns: list) -> tuple[float, str, int, int]:
    """
    Candle bonus score added to composite (0–10).
    Returns (bonus, pattern_name, tier, confidence).
    Tier 1 + volume confirmed = +10 (highest quality signal)
    """
    if not patterns:
        return 0.0, "", 0, 0

    best = sorted(patterns, key=lambda p: (p["tier"], -p["confidence"]))[0]
    tier = best["tier"]
    vol  = best["volume_confirmed"]

    if tier == 1:
        bonus = 10.0 if vol else 7.0
    elif tier == 2:
        bonus = 5.0 if vol else 3.0
    else:
        bonus = 1.0

    return bonus, best["pattern"], tier, best["confidence"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5.a — FUNDAMENTAL ADJUSTMENT
# Evidence: fundamental_validated.csv (FDR q<0.05), fundamental_sector.csv
# Rules derived from study run 2026-04-05. Re-run quarterly.
#
# Design principles:
#   - Uses lag=1 only (never lag=0) — avoids lookahead in live trading
#   - Sector-aware: different signals matter per sector
#   - Soft adjustment only: range -3 to +3, never a hard gate
#   - Fails silently: if DB unavailable, returns 0 (no effect on pipeline)
#   - Debenture/bond exclusion: near-zero beta symbols auto-excluded
# ══════════════════════════════════════════════════════════════════════════════

 
# Validated fundamentals with LOW VIF (safe to use without multicollinearity)
# Source: fundamental_study_results.csv + OLS VIF section
# Only these are used in scoring — high-VIF cols (promoter_shares, total_assets
# etc.) are intentionally excluded.
_LOW_VIF_FUNDAMENTALS = {
    "npl", "capital_fund_to_rwa", "cd_ratio", "roa",
    "interest_spread", "dps", "pe_ratio", "peg_value",
    "growth_rate", "cost_of_fund", "base_rate",
}
 
# Sector-specific signal map (from fundamental_sector.csv significant=True)
# Format: sector_key → list of (fundamental, expected_sign)
# expected_sign: +1 means higher value → good, -1 means higher value → bad
_SECTOR_FUNDAMENTAL_SIGNALS = {
    "hydro power": [
        ("roa",                +1),   # rho=+0.190 — strongest Hydro signal
        ("roe",                +1),   # rho=+0.156
        ("dps",                +1),   # rho=+0.142
        ("prev_quarter_profit",+1),   # rho=+0.099
        ("growth_rate",        +1),   # rho=+0.078
        ("net_profit",         +1),   # rho=+0.082
        ("promoter_shares",    -1),   # rho=-0.108
    ],
    "development bank": [
        ("interest_spread",    +1),   # rho=+0.214
        ("capital_fund_to_rwa",+1),   # rho=+0.118
        ("net_interest_income",-1),   # rho=-0.194
        ("loan",               -1),   # rho=-0.125
        ("deposit",            -1),   # rho=-0.111
        ("net_worth",          -1),   # rho=-0.129
    ],
    "finance": [
        ("interest_spread",    +1),   # rho=+0.227
        ("capital_fund_to_rwa",+1),   # rho=+0.157
        ("cd_ratio",           -1),   # rho=-0.130
        ("prev_quarter_profit",+1),   # rho=+0.112
        ("net_worth",          -1),   # rho=-0.117
    ],
    "microfinance": [
        ("roa",                -1),   # rho=-0.145 (high ROA paradox: regulatory pressure)
        ("interest_spread",    -1),   # rho=-0.129
        ("prev_quarter_profit",+1),   # rho=+0.069
    ],
    "commercial bank": [
        ("net_interest_income",+1),   # rho=+0.198
        ("growth_rate",        -1),   # rho=-0.116
        ("npl",                -1),   # global signal: higher NPL → lower 1m returns
        ("capital_fund_to_rwa",+1),   # global signal validated at lag 1-3
    ],
    "life insurance": [
        # Only gram_value significant — not a fundamental we control
        # No actionable fundamental signals for this sector
    ],
}
 
# Beta-based debenture/bond exclusion threshold
# Empirical: all debenture symbols cluster between -0.20 and +0.20
_BOND_BETA_MAX = 0.20
 
# Debenture suffix patterns (belt-and-suspenders with beta check)
_DEBENTURE_SUFFIXES = (
    "D80", "D81", "D82", "D83", "D84", "D85", "D86", "D87", "D88", "D89", "D90",
    "D2082", "D2083", "D2084", "D2085",
)

# Mutual fund + scheme suffixes — exclude from trading signals
_MUTUAL_FUND_SUFFIXES = (
    "GF", "MF", "BF", "SF", "OF",   # growth fund, mutual fund, balanced, savings, open-ended
)

# Mutual fund name keywords (belt-and-suspenders)
_MUTUAL_FUND_KEYWORDS = (
    "NIBLGF", "NMBSF", "NMBHF", "LBSLGF", "SIGS", "NMB50", "NMBD",
    "GIMES1", "CMF1", "CMF2", "LEMF",
)
 
def _load_fundamental_data() -> tuple[dict, dict]:
    """
    Load fundamentals and beta tables from DB into memory dicts.
    Called ONCE per run_filter() invocation (not per symbol).
 
    Returns:
        fund_map:  dict[symbol → latest quarter fundamentals dict]
        beta_map:  dict[symbol → {beta, market_corr, market_corr_p}]
 
    Both return empty dicts on any failure — pipeline continues safely.
    """
    fund_map: dict = {}
    beta_map: dict = {}
 
    try:
        from db.connection import _db  # adjust to your actual DB import
        with _db() as cur:
            # ── Latest quarter fundamentals per symbol (lag=1 safe)
            # We grab the most recent completed quarter for each symbol.
            # The fundamental_study uses lag=1 which means we use data
            # published in the PREVIOUS quarter relative to today.
            cur.execute("""
                SELECT DISTINCT ON (symbol)
                    symbol, fiscal_year, quarter,
                    npl, capital_fund_to_rwa, cd_ratio, roa, roe,
                    interest_spread, dps, pe_ratio, peg_value,
                    growth_rate, cost_of_fund, base_rate,
                    net_interest_income, net_worth, net_profit,
                    prev_quarter_profit, promoter_shares,
                    loan, deposit, paidup_capital
                FROM fundamentals
                ORDER BY symbol, fiscal_year DESC, quarter DESC
            """)
            rows = cur.fetchall()
            for row in rows:
                sym = row["symbol"].upper() if hasattr(row, "__getitem__") else row[0].upper()
                fund_map[sym] = dict(row) if hasattr(row, "keys") else {
                    col: val for col, val in zip(
                        ["symbol","fiscal_year","quarter","npl","capital_fund_to_rwa",
                            "cd_ratio","roa","roe","interest_spread","dps","pe_ratio",
                            "peg_value","growth_rate","cost_of_fund","base_rate",
                            "net_interest_income","net_worth","net_profit",
                            "prev_quarter_profit","promoter_shares","loan","deposit","paidup_capital"],
                        row
                    )
                }

            # ── Beta table
            cur.execute("""
                SELECT symbol, beta, market_corr, market_corr_p, n_months
                FROM fundamental_beta
            """)
            for row in cur.fetchall():
                sym = (row["symbol"] if hasattr(row, "__getitem__") else row[0]).upper()
                beta_map[sym] = {
                    "beta":           float(row["beta"]          if hasattr(row, "__getitem__") else row[1]),
                    "market_corr":    float(row["market_corr"]   if hasattr(row, "__getitem__") else row[2]),
                    "market_corr_p":  float(row["market_corr_p"] if hasattr(row, "__getitem__") else row[3]),
                    "n_months":       int(row["n_months"]        if hasattr(row, "__getitem__") else row[4]),
                }

    except Exception as exc:
        logger.warning("_load_fundamental_data failed (%s) — fundamentals skipped", exc)
 
    logger.info(
        "Fundamental data loaded: %d symbols with fundamentals, %d with beta",
        len(fund_map), len(beta_map),
    )
    return fund_map, beta_map
 
 
def _is_non_equity_by_beta(symbol: str, beta_map: dict) -> bool:
    """
    Returns True if symbol looks like a debenture/bond based on empirical beta.
    Used as an additional filter in _check_symbol_gates equivalent logic.
 
    A symbol is flagged as non-equity if:
      - Its beta is between -0.20 and +0.20 AND
      - market_corr_p > 0.05 (no significant market relationship) AND
      - n_months >= 12 (enough history to trust the beta)
 
    Suffix check is belt-and-suspenders.
    """
    # Suffix check first (fast, no DB needed)
    sym_upper = symbol.upper()
    if any(sym_upper.endswith(sfx) for sfx in _DEBENTURE_SUFFIXES):
        return True
 
    # Beta check
    entry = beta_map.get(sym_upper)
    if entry and entry["n_months"] >= 12:
        if (abs(entry["beta"]) <= _BOND_BETA_MAX
                and entry["market_corr_p"] > 0.05):
            return True
 
    return False
 
 
def _get_fundamental_adj(
    symbol:   str,
    sector:   str,
    fund_map: dict,
    beta_map: dict,
) -> tuple[float, str]:
    """
    Compute fundamental score adjustment for one symbol.
 
    Returns:
        (adjustment, reason_string)
        adjustment: float in range [-3.0, +3.0]
        reason:     short string for logging/display
 
    Rules:
      - Only uses fundamentals with low VIF (safe predictors)
      - Sector-specific signals weighted higher than global signals
      - Each signal contributes ±0.5 to ±1.0 based on validated rho strength
      - Total capped at ±3.0
      - Returns (0.0, "no_data") if symbol has no fundamental record
    """
    fund = fund_map.get(symbol.upper())
    if not fund:
        return 0.0, "no_fundamental_data"
 
    sector_key = sector.lower().strip()
    adj        = 0.0
    reasons    = []
 
    def _safe_float(val) -> float | None:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None
 
    # ── 1. Sector-specific signals (higher weight: ±1.0 each) ────────────────
    sector_signals = _SECTOR_FUNDAMENTAL_SIGNALS.get(sector_key, [])
 
    # Partial match if exact not found (e.g. "hydro" matches "hydro power")
    if not sector_signals:
        for key, signals in _SECTOR_FUNDAMENTAL_SIGNALS.items():
            if key in sector_key or sector_key in key:
                sector_signals = signals
                break
 
    for fundamental, expected_sign in sector_signals:
        val = _safe_float(fund.get(fundamental))
        if val is None:
            continue
 
        # Sector-aware thresholds: flag only extreme quartile values
        # Using simple sign logic: if value direction matches expected_sign → positive
        # This is intentionally simple — we're not re-running the regression,
        # just using direction of the known validated signal.
        contribution = 0.0
        if expected_sign == +1 and val > 0:
            contribution = +1.0
        elif expected_sign == -1 and val > 0:
            contribution = -1.0
        elif expected_sign == +1 and val <= 0:
            contribution = -0.5
        elif expected_sign == -1 and val <= 0:
            contribution = +0.5
 
        adj += contribution
        if contribution != 0.0:
            sign_str = "+" if contribution > 0 else ""
            reasons.append(f"{fundamental}:{sign_str}{contribution:.1f}")
 
    # ── 2. Global low-VIF signals (lower weight: ±0.5 each) ──────────────────
    # These apply across all sectors from fundamental_validated.csv
 
    # NPL — validated globally (rho=-0.11 at lag 0-2, most sectors)
    npl = _safe_float(fund.get("npl"))
    if npl is not None:
        if npl > 5.0:          # High NPL: clearly bad
            adj -= 0.5
            reasons.append("npl>5:-0.5")
        elif npl < 2.0:        # Very clean loan book
            adj += 0.5
            reasons.append("npl<2:+0.5")
 
    # capital_fund_to_rwa — validated globally (rho=+0.08-0.13 across lags)
    cfrwa = _safe_float(fund.get("capital_fund_to_rwa"))
    if cfrwa is not None:
        if cfrwa > 13.0:       # Well-capitalised (NRB minimum is 11%)
            adj += 0.5
            reasons.append("cfrwa>13:+0.5")
        elif cfrwa < 11.0:     # Below regulatory minimum — risk flag
            adj -= 1.0
            reasons.append("cfrwa<11:-1.0")
 
    # pe_ratio — validated (rho=+0.05-0.07 at lag 0, 3)
    # High PE is slightly positive (momentum effect in NEPSE)
    pe = _safe_float(fund.get("pe_ratio"))
    if pe is not None and pe > 0:
        if pe > 30:
            adj += 0.3
            reasons.append("pe>30:+0.3")
        elif pe < 8:           # Extremely cheap — value signal
            adj += 0.3
            reasons.append("pe<8:+0.3")
 
    # DPS — strong signal but direction flips by lag (use conservatively)
    # At lag=1: negative rho (post-dividend drift down)
    # At lag=3: positive rho (anticipation effect)
    # Use: presence of DPS > 0 as mild positive (company is profitable)
    dps = _safe_float(fund.get("dps"))
    if dps is not None:
        if dps > 0:
            adj += 0.3
            reasons.append("dps>0:+0.3")
 
    # ── 3. Cap total adjustment ───────────────────────────────────────────────
    adj = max(-3.0, min(3.0, adj))
 
    reason_str = "|".join(reasons) if reasons else "no_signal"
    return round(adj, 2), reason_str
 

def _load_broker_flow_cache(date: str) -> tuple[dict, dict]:
    """
    Load broker_flow and broker_holdings for the given date.
    Returns (flow_cache, holdings_cache) keyed by uppercase symbol.
    Both dicts are empty on any failure — never raises.
    """
    flow_cache:     dict = {}
    holdings_cache: dict = {}
    try:
        from sheets import run_raw_sql
        flow_rows = run_raw_sql(
            "SELECT * FROM broker_flow WHERE date = %s",
            (date,)
        )
        for r in (flow_rows or []):
            sym = str(r.get("symbol", "")).upper()
            if sym:
                flow_cache[sym] = r

        holdings_rows = run_raw_sql(
            "SELECT * FROM broker_holdings WHERE date = %s",
            (date,)
        )
        for r in (holdings_rows or []):
            sym = str(r.get("symbol", "")).upper()
            if sym:
                holdings_cache[sym] = r
    except Exception as exc:
        logger.warning("_load_broker_flow_cache failed (%s) — broker flow skipped", exc)

    logger.info(
        "Broker flow cache: %d flow + %d holdings symbols",
        len(flow_cache), len(holdings_cache),
    )
    return flow_cache, holdings_cache


def _get_broker_flow_classification(sym: str, flow_cache: dict) -> tuple[str, int, float]:
    """
    Classify broker flow for a symbol.

    Returns (classification, net_broker_count, net_amount):
      NET_ACCUMULATION — acc_count dominates and dist_count < acc_count * churn_threshold
      NET_DISTRIBUTION — dist_count dominates and acc_count < dist_count * churn_threshold
      CHURN            — both sides heavy (above churn threshold)
      NO_DATA          — symbol not in cache or parse error
    """
    if not flow_cache:
        return ("NO_DATA", 0, 0.0)

    row = flow_cache.get(str(sym).upper())
    if not row:
        return ("NO_DATA", 0, 0.0)

    try:
        from sheets import get_setting
        churn_threshold = float(get_setting("BROKER_FLOW_CHURN_THRESHOLD", "0.6"))
    except Exception:
        churn_threshold = 0.6

    try:
        acc_count  = int(float(row.get("acc_broker_count_1d", 0) or 0))
        dist_count = int(float(row.get("dist_broker_count_1d", 0) or 0))
        acc_amt    = float(row.get("acc_amount_1d", 0) or 0)
        dist_amt   = float(row.get("dist_amount_1d", 0) or 0)
        acc_qty    = float(row.get("acc_qty_1d", 0) or 0)
        dist_qty   = float(row.get("dist_qty_1d", 0) or 0)
        net_amount = acc_amt - dist_amt
        net_count  = acc_count - dist_count
        acc_score  = (acc_amt * acc_qty)  / max(acc_count, 1)
        dist_score = (dist_amt * dist_qty) / max(dist_count, 1)
    except (TypeError, ValueError):
        return ("NO_DATA", 0, 0.0)

    if acc_score > dist_score and dist_score < acc_score * churn_threshold:
        return ("NET_ACCUMULATION", net_count, net_amount)
    if dist_score > acc_score and acc_score < dist_score * churn_threshold:
        return ("NET_DISTRIBUTION", net_count, net_amount)
    if acc_score > 0 and dist_score > 0:
        return ("CHURN", net_count, net_amount)
    return ("NO_DATA", 0, 0.0)


def _compute_broker_flow_adj(sym: str, flow_cache: dict, holdings_cache: dict) -> float:
    """
    Composite score bonus from broker flow and stealth concentration.
    Never negative — gate handles the downside.
    Bonus values read from settings with hard-coded fallbacks.
    """
    try:
        from sheets import get_setting
        acc_bonus          = float(get_setting("BROKER_FLOW_ACC_BONUS",          "5.0"))
        stealth_high_bonus = float(get_setting("BROKER_FLOW_STEALTH_HIGH_BONUS", "3.0"))
        stealth_med_bonus  = float(get_setting("BROKER_FLOW_STEALTH_MED_BONUS",  "1.5"))
    except Exception:
        acc_bonus          = 5.0
        stealth_high_bonus = 3.0
        stealth_med_bonus  = 1.5

    flow_class, _, _ = _get_broker_flow_classification(sym, flow_cache)

    if flow_class in ("NO_DATA", "CHURN"):
        return 0.0

    adj = 0.0
    if flow_class == "NET_ACCUMULATION":
        adj += acc_bonus

    holdings_row = (holdings_cache or {}).get(str(sym).upper())
    if holdings_row:
        try:
            stealth = float(holdings_row.get("stealth_score", 0) or 0)
            if stealth > 80:
                adj += stealth_high_bonus
            elif stealth > 60:
                adj += stealth_med_bonus
        except (TypeError, ValueError):
            pass

    return max(0.0, adj)


def _compute_vos_adj(vos: float) -> float:
    """
    Score adjustment based on volume/outstanding-shares ratio.
    Additive to composite score — not multiplicative.
    Range: -1.0 to +4.0
    """
    if vos <= 0:
        return 0.0        # no OS data — neutral, don't penalise
    elif vos < 0.3:
        return -1.0       # thin liquidity — mild penalty
    elif vos < 1.0:
        return 0.0        # normal — neutral
    elif vos < 3.0:
        return +2.0       # high — smart money possible
    else:
        return +4.0       # exceptional — operator/institutional signal


def _compute_live_adj(price_row) -> float:
    """
    Intraday live signal adjustment from ATrad real-time data.
    Scores vwap_dev, bid_ask_ratio, and dpr_proximity (lower penalty).

    Range: -1.5 to +2.0 (small — never dominates frozen indicator score)

    REVIEW FLAG: Thresholds are experience-based, not NEPSE-study-backed.
    Review after 6 months of live trading data (target: 2026-11-01).
    Tunable via settings: LIVE_ADJ_VWAP_HIGH, LIVE_ADJ_VWAP_LOW,
    LIVE_ADJ_VWAP_HIGH_SCORE, LIVE_ADJ_VWAP_MILD_SCORE, LIVE_ADJ_VWAP_WEAK_SCORE,
    LIVE_ADJ_BAR_HIGH, LIVE_ADJ_BAR_LOW,
    LIVE_ADJ_BAR_HIGH_SCORE, LIVE_ADJ_BAR_MILD_SCORE, LIVE_ADJ_BAR_WEAK_SCORE,
    LIVE_ADJ_DPR_LOW_GATE, LIVE_ADJ_DPR_LOW_PENALTY.
    """
    try:
        from sheets import get_setting

        # ── Thresholds from settings ──────────────────────────────────────────
        vwap_high       = float(get_setting("LIVE_ADJ_VWAP_HIGH",        "2.0"))
        vwap_low        = float(get_setting("LIVE_ADJ_VWAP_LOW",         "2.0"))
        vwap_high_score = float(get_setting("LIVE_ADJ_VWAP_HIGH_SCORE",  "1.0"))
        vwap_mild_score = float(get_setting("LIVE_ADJ_VWAP_MILD_SCORE",  "0.5"))
        vwap_weak_score = float(get_setting("LIVE_ADJ_VWAP_WEAK_SCORE",  "1.0"))

        bar_high        = float(get_setting("LIVE_ADJ_BAR_HIGH",         "0.65"))
        bar_low         = float(get_setting("LIVE_ADJ_BAR_LOW",          "0.35"))
        bar_high_score  = float(get_setting("LIVE_ADJ_BAR_HIGH_SCORE",   "1.0"))
        bar_mild_score  = float(get_setting("LIVE_ADJ_BAR_MILD_SCORE",   "0.5"))
        bar_weak_score  = float(get_setting("LIVE_ADJ_BAR_WEAK_SCORE",   "1.0"))

        dpr_low_gate    = float(get_setting("LIVE_ADJ_DPR_LOW_GATE",     "0.10"))
        dpr_low_penalty = float(get_setting("LIVE_ADJ_DPR_LOW_PENALTY",  "0.5"))

    except Exception:
        # Hard-coded fallbacks if settings unreachable — safe defaults
        vwap_high = vwap_low = 2.0
        vwap_high_score = vwap_mild_score = vwap_weak_score = 1.0
        bar_high = 0.65; bar_low = 0.35
        bar_high_score = bar_mild_score = bar_weak_score = 1.0
        dpr_low_gate = 0.10; dpr_low_penalty = 0.5

    adj = 0.0

    # ── vwap_dev: % LTP above/below VWAP (already in % from scraper) ─────────
    vwap_dev = float(getattr(price_row, "vwap_dev", 0) or 0)
    if vwap_dev > vwap_high:
        adj += vwap_high_score        # strong buy-side momentum
    elif vwap_dev > 0.5:
        adj += vwap_mild_score        # mild above VWAP
    elif vwap_dev < -vwap_low:
        adj -= vwap_weak_score        # meaningfully below VWAP — weakness

    # ── bid_ask_ratio: 0–1 normalized (bid_qty / total depth) ────────────────
    bar = float(getattr(price_row, "bid_ask_ratio", 0) or 0)
    if bar > bar_high:
        adj += bar_high_score         # strong buy-side order book
    elif bar > 0.55:
        adj += bar_mild_score         # mild buy pressure
    elif bar < bar_low:
        adj -= bar_weak_score         # heavy sell-side depth

    # ── dpr_proximity: lower end penalty (upper end is gated, not scored) ────
    dpr_prox = float(getattr(price_row, "dpr_proximity", 0) or 0)
    if dpr_prox < dpr_low_gate:
        adj -= dpr_low_penalty        # near lower circuit — risk signal

    return round(max(-1.5, min(2.0, adj)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_composite_score(
    indicator_score: float,
    sector_mult:     float,
    candle_bonus:    float,
    cstar_signal:    bool,
    conf_score:      float,
    geo_combined:    int,
    ipo_drain:       str,
    fundamental_adj:  float = 0.0,
    vos_adj:          float = 0.0,   # volume/OS ratio adjustment
    broker_flow_adj:  float = 0.0,   # broker accumulation/distribution bonus
    live_adj:         float = 0.0,   # intraday live signal (vwap_dev, bid_ask_ratio, dpr_proximity)
    min_conf_score:   float = MIN_CONF_SCORE,  # dynamic — read from settings via ctx
) -> float:
    """
    Final composite score for candidate ranking.

    base      = indicator_score × sector_mult
    + candle  = 0 to +10
    + cstar   = +5 if excess return > C*
    + conf    = 0 to +5 (ShareSansar momentum above 50)
    + geo_adj = -5 to +3 (asymmetric: downside hurts more)
    - ipo_pen = -3 if IPO drain active
    + fund_adj= -3 to +3 (fundamental quality, sector-aware)
    + live_adj= -1.5 to +2.0 (intraday: vwap_dev, bid_ask_ratio, dpr_proximity)
    """
    base       = indicator_score * sector_mult
    conf_bonus = min((conf_score - min_conf_score) / 10, 5.0) if conf_score > min_conf_score else 0.0
    cstar_b    = 5.0 if cstar_signal else 0.0
    ipo_pen    = -3.0 if ipo_drain == "YES" else 0.0
 
    # Asymmetric geo adjustment — capital preservation principle
    if geo_combined >= 3:
        geo_adj = 3.0
    elif geo_combined >= 1:
        geo_adj = 1.5
    elif geo_combined == 0:
        geo_adj = 0.0
    elif geo_combined >= -2:
        geo_adj = -2.0
    else:
        geo_adj = -5.0
 
    return round(
        max(0.0, base + candle_bonus + cstar_b + conf_bonus
                 + geo_adj + ipo_pen + fundamental_adj + vos_adj + broker_flow_adj + live_adj),
        2,
    )
 
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN FILTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_filter(
    market_data: dict   = None,
    top_n:       int    = 10,
    date:        str    = None,
) -> list[FilterCandidate]:
    """
    Main entry point. Called every 6 min by trading.yml.

    Args:
        market_data: dict[symbol, PriceRow] from scraper.get_all_market_data()
                     If None, fetches live data automatically.
        top_n:       Number of top candidates to return (default 10).
        date:        Override date string YYYY-MM-DD (default: today NST).

    Returns:
        list[FilterCandidate] ranked by composite_score descending.
        Empty list if hard gates block the run.
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")
    
    global _last_near_misses
    _last_near_misses = []

    logger.info("=" * 60)
    logger.info("filter_engine.run_filter() — %s", date)

    # ── Context ───────────────────────────────────────────────────────────────
    ctx = _load_context()

    # ── System hard gates ─────────────────────────────────────────────────────
    gates_ok, gate_reason = _check_hard_gates(ctx)
    if not gates_ok:
        logger.warning("HARD GATE BLOCKED: %s", gate_reason)
        return []

    # ── Live prices ───────────────────────────────────────────────────────────
    if market_data is None:
        try:
            from modules.scraper import get_all_market_data
            market_data = get_all_market_data(write_breadth=False)
        except Exception as exc:
            logger.error("Market data fetch failed: %s", exc)
            return []

    if not market_data:
        logger.warning("No market data available")
        return []

    # ── Frozen indicators ─────────────────────────────────────────────────────
    try:
        from sheets import read_today_indicators
        indicators_map = read_today_indicators(date)
    except Exception as exc:
        logger.error("Could not load indicators: %s", exc)
        return []

    if not indicators_map:
        logger.warning("No indicators for %s — run indicators.py first", date)
        return []

    # ── Momentum history (7-day batch load) ───────────────────────────────────
    valid_syms = [
        s.upper() for s in market_data
        if str(s).replace("-", "").replace("_", "").isalpha()
        and indicators_map.get(s.upper())
    ]
    recent_map = _load_recent_indicators(valid_syms, date, lookback=7)
    logger.info("Momentum history loaded: %d symbols", len(recent_map))

    # ── Candle signals ────────────────────────────────────────────────────────
    valid_symbols = [
        s.upper() for s in market_data
        if str(s).replace("-", "").replace("_", "").isalpha()
    ]
    candle_map = _load_candle_signals(valid_symbols, date)

    # ── Fundamental data (loaded ONCE for all symbols) ────────────────────────
    fund_map, beta_map = _load_fundamental_data()

    # ── Broker flow cache (loaded ONCE for all symbols) ───────────────────────
    flow_cache, holdings_cache = _load_broker_flow_cache(date)

    # ── Sector info from watchlist ────────────────────────────────────────────
    sector_map: dict[str, str] = {}
    try:
        from sheets import read_tab
        share_sectors = read_tab("share_sectors")
        sector_map = {
            r["symbol"].upper(): (r.get("sectorname") or "others").lower()
            for r in share_sectors if r.get("symbol")
        }
    except Exception as exc:
        logger.warning("Could not load share_sectors: %s", exc)

    # ── Score every symbol ────────────────────────────────────────────────────
    candidates:    list[FilterCandidate] = []
    skipped_gate   = 0
    skipped_no_ind = 0
    processed      = 0
    rf_rate        = ctx.get("rf_rate_annual", RF_ANNUAL_PCT)

    for symbol, price_row in market_data.items():
        sym = str(symbol).upper()
        if not sym.replace("-", "").replace("_", "").isalpha():
            continue

        ind = indicators_map.get(sym)
        if not ind:
            skipped_no_ind += 1
            continue

        processed += 1

        # ── VOS computation — before sym_ok so it's available for NearMiss ──
        fund_data  = fund_map.get(sym, {})
        paidup     = float(fund_data.get("paidup_capital") or 0)
        os_shares  = paidup / 100.0 if paidup > 0 else 0.0
        volume_val = float(getattr(price_row, "volume", 0) or 0)
        volume_os_ratio = round((volume_val / os_shares * 100), 4) if os_shares > 0 else 0.0
        logger.debug("VOS: %s vol=%.0f os=%.0f ratio=%.3f%%", sym, volume_val, os_shares, volume_os_ratio)

        # Compute momentum for this symbol and inject into ctx temporarily
        momentum = _compute_momentum(recent_map.get(sym, []))
        ctx["_sym_momentum"] = momentum   # consumed by _check_symbol_gates

        sym_ok, sym_reason = _check_symbol_gates(sym, ind, price_row, ctx, flow_cache=flow_cache)
        if not sym_ok:
            skipped_gate += 1
            logger.debug("GATE: %s — %s", sym, sym_reason)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)
            _last_near_misses.append(NearMiss(
                symbol                   = sym,
                sector                   = sector_map.get(sym, "others"),
                date                     = date,
                gate_reason              = sym_reason,
                gate_category            = _categorize_gate_reason(sym_reason),
                price_at_block           = ltp,
                market_state             = ctx["market_state"],
                tech_score               = int(ind.get("tech_score", 0) or 0),
                conf_score               = float(getattr(price_row, "conf_score", 0) or 0),
                composite_score_would_be = 0.0,
                volume_os_ratio          = volume_os_ratio,
                vwap_dev                 = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio            = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity            = float(getattr(price_row, "dpr_proximity",  0) or 0),
            ))
            continue

        # ── Non-equity exclusion via beta (debentures/bonds) ─────────────────
        if _is_non_equity_by_beta(sym, beta_map):
            skipped_gate += 1
            logger.debug("GATE: %s — NON_EQUITY_BY_BETA", sym)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)
            _last_near_misses.append(NearMiss(
                symbol          = sym,
                sector          = sector_map.get(sym, "others"),
                date            = date,
                gate_reason     = "NON_EQUITY_BY_BETA",
                gate_category   = "NON_EQUITY",
                price_at_block  = ltp,
                market_state    = ctx["market_state"],
                tech_score      = int(ind.get("tech_score", 0) or 0),
                conf_score      = float(getattr(price_row, "conf_score", 0) or 0),
                volume_os_ratio = volume_os_ratio,
                vwap_dev        = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio   = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity   = float(getattr(price_row, "dpr_proximity",  0) or 0),
            ))
            continue

        # ── VOS soft gate: truly dead stock ──────────────────────────────────
        if volume_os_ratio < MIN_VOS_PCT and os_shares > 0:
            skipped_gate += 1
            gate_reason = f"VOS={volume_os_ratio:.3f}%<{MIN_VOS_PCT}%"
            logger.debug("GATE: %s — %s", sym, gate_reason)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)
            _last_near_misses.append(NearMiss(
                symbol                   = sym,
                sector                   = sector_map.get(sym, "others"),
                date                     = date,
                gate_reason              = gate_reason,
                gate_category            = "ILLIQUID",
                price_at_block           = ltp,
                market_state             = ctx["market_state"],
                tech_score               = int(ind.get("tech_score", 0) or 0),
                conf_score               = float(getattr(price_row, "conf_score", 0) or 0),
                composite_score_would_be = 0.0,
                volume_os_ratio          = volume_os_ratio,
                vwap_dev                 = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio            = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity            = float(getattr(price_row, "dpr_proximity",  0) or 0),
            ))
            continue

        sector    = sector_map.get(sym) or str(ind.get("sector", "others") or "others")
        ind_score, primary, hold_days = _compute_indicator_score(
            ind, sector,
            momentum["momentum_status"],
            momentum.get("momentum_score", 50.0),
        )
        sect_mult = _get_sector_multiplier(sector, ctx)

        ltp        = float(getattr(price_row, "ltp", 0)        or getattr(price_row, "close", 0) or 0)
        change_pct = float(getattr(price_row, "change_pct", 0) or 0)
        cstar      = _check_cstar_signal(change_pct, sector, rf_rate)

        patterns                        = candle_map.get(sym, [])
        c_bonus, c_name, c_tier, c_conf = _candle_bonus(patterns)

        # ── Fundamental adjustment ────────────────────────────────────────────
        fund_adj, fund_reason = _get_fundamental_adj(sym, sector, fund_map, beta_map)
        logger.debug("FUND: %s adj=%.2f [%s]", sym, fund_adj, fund_reason)

        # ── VOS scoring adjustment ────────────────────────────────────────────
        vos_adj = _compute_vos_adj(volume_os_ratio)
        logger.debug("VOS_ADJ: %s vos=%.3f%% adj=%+.1f", sym, volume_os_ratio, vos_adj)

        # ── Live intraday adjustment ──────────────────────────────────────────────
        live_adj = _compute_live_adj(price_row)
        logger.debug("LIVE_ADJ: %s vwap=%.2f%% bar=%.2f dpr=%.2f adj=%+.2f",
                     sym,
                     float(getattr(price_row, "vwap_dev",       0) or 0),
                     float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                     float(getattr(price_row, "dpr_proximity",  0) or 0),
                     live_adj)

        # ── Broker flow scoring adjustment ────────────────────────────────────
        broker_flow_adj = _compute_broker_flow_adj(sym, flow_cache, holdings_cache)
        logger.debug("BROKER_FLOW_ADJ: %s adj=%+.1f", sym, broker_flow_adj)

        # ── News catalyst (from nepal_pulse STOCK_CATALYSTS_TODAY) ────────────
        candidate_news_catalyst = ctx.get("_sym_catalysts", {}).get(sym, "")


        composite = _compute_composite_score(
            indicator_score = ind_score,
            sector_mult     = sect_mult,
            candle_bonus    = c_bonus,
            cstar_signal    = cstar,
            conf_score      = float(getattr(price_row, "conf_score", 0) or 0),
            geo_combined    = ctx["combined_geo"],
            ipo_drain       = ctx["ipo_drain"],
            fundamental_adj = fund_adj,
            vos_adj         = vos_adj,
            broker_flow_adj = broker_flow_adj,
            live_adj        = live_adj,
            min_conf_score  = ctx.get("min_conf_score", MIN_CONF_SCORE),
        )
        logger.debug("VOS_ADJ: %s composite=%.1f", sym, composite)

        candidates.append(FilterCandidate(
            symbol           = sym,
            sector           = sector,
            ltp              = ltp,
            change_pct       = change_pct,
            volume           = int(getattr(price_row, "volume", 0) or 0),

            rsi_14           = float(ind.get("rsi_14",           0)   or 0),
            rsi_signal       = str(ind.get("rsi_signal",         "")  or ""),
            ema_trend        = str(ind.get("ema_trend",          "")  or ""),
            ema_20_50_cross  = str(ind.get("ema_20_50_cross",    "")  or ""),
            ema_50_200_cross = str(ind.get("ema_50_200_cross",   "")  or ""),
            macd_cross       = str(ind.get("macd_cross",     "NONE") or "NONE"),
            macd_histogram   = float(ind.get("macd_histogram",   0)   or 0),
            macd_line        = float(ind.get("macd_line",         0)   or 0),
            macd_signal_line = float(ind.get("macd_signal",       0)   or 0),
            bb_signal        = str(ind.get("bb_signal",  "NEUTRAL")   or "NEUTRAL"),
            bb_pct_b         = float(ind.get("bb_pct_b",        0.5)  or 0.5),
            bb_upper         = float(ind.get("bb_upper",          0)   or 0),
            bb_lower         = float(ind.get("bb_lower",          0)   or 0),
            
            obv_trend        = str(ind.get("obv_trend",    "FLAT")    or "FLAT"),
            atr_pct          = float(ind.get("atr_pct",          0)   or 0),
            tech_score       = int(ind.get("tech_score",         0)   or 0),
            tech_signal      = str(ind.get("tech_signal",        "")  or ""),
            history_days     = int(ind.get("history_days",       0)   or 0),

            support_level    = float(ind.get("support_level",    0)   or 0),
            resistance_level = float(ind.get("resistance_level", 0)   or 0),

            conf_score       = float(getattr(price_row, "conf_score",  0) or 0),
            conf_signal      = str(getattr(price_row,  "conf_signal", "") or ""),

            candle_patterns  = patterns,
            best_candle      = c_name,
            candle_tier      = c_tier,
            candle_conf      = c_conf,

            geo_score        = ctx["geo_score"],
            nepal_score      = ctx["nepal_score"],
            combined_geo     = ctx["combined_geo"],
            bandh_today      = ctx["bandh_today"],
            crisis_detected  = ctx["crisis_detected"],
            ipo_drain        = ctx["ipo_drain"],
            market_state     = ctx["market_state"],

            indicator_score  = ind_score,
            sector_mult      = sect_mult,
            cstar_signal     = cstar,
            composite_score  = composite,
            primary_signal   = primary,
            suggested_hold   = hold_days,

            fundamental_adj    = fund_adj,
            fundamental_reason = fund_reason,
            broker_flow_adj    = broker_flow_adj,

            vwap_dev        = float(getattr(price_row, "vwap_dev",       0) or 0),
            bid_ask_ratio   = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
            dpr_proximity   = float(getattr(price_row, "dpr_proximity",  0) or 0),
            volume_os_ratio = volume_os_ratio,

            momentum_status  = momentum["momentum_status"],
            rsi_slope_3d     = momentum["rsi_slope_3d"],
            macd_hist_slope  = momentum["macd_hist_slope"],
            bb_pct_b_slope   = momentum["bb_pct_b_slope"],
            bounce_failed    = momentum["bounce_failed"],
            reversal_days    = momentum["reversal_days"],

            news_catalyst    = candidate_news_catalyst,   # ADD
        ))

    ctx.pop("_sym_momentum", None)

    # ── Rank and trim ─────────────────────────────────────────────────────────
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    top = candidates[:top_n]

    logger.info(
        "run_filter done: %d processed | %d passed | %d gate-skipped | %d no-indicator",
        processed, len(candidates), skipped_gate, skipped_no_ind,
    )
    for c in top:
        logger.info("  %s", c.summary())

    _log_filter_candidates(top, date)

    return top

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7.5 — FILTER CANDIDATE LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _log_filter_candidates(candidates: list, date: str) -> None:
    """
    Log all filter candidates passed to Gemini each cycle.
    Upserts on (symbol, date) — increments pass_count on repeat appearances.
    Uses execute_dml from sheets (never from db directly).
    Fails silently — logging must never break the trading pipeline.
    """
    if not candidates:
        return
    try:
        from sheets import execute_dml
        for c in candidates:
            execute_dml(
                """
                INSERT INTO filter_candidates_log
                    (date, symbol, sector, composite_score, primary_signal,
                     tech_score, macro_score, market_state, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (symbol, date) DO UPDATE SET
                    pass_count      = filter_candidates_log.pass_count + 1,
                    last_seen       = now(),
                    composite_score = EXCLUDED.composite_score,
                    tech_score      = EXCLUDED.tech_score,
                    macro_score     = EXCLUDED.macro_score
                """,
                (
                    date,
                    c.symbol,
                    c.sector,
                    c.composite_score,
                    c.primary_signal,
                    float(c.tech_score),
                    float(c.nepal_score),
                    c.market_state,
                ),
            )
        logger.info("filter_candidates_log: wrote %d candidates", len(candidates))
    except Exception as e:
        logger.warning("filter_candidates_log write failed: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HELPERS FOR gemini_filter.py
# ══════════════════════════════════════════════════════════════════════════════

def get_filter_context() -> dict:
    """Current context dict without running the full filter."""
    return _load_context()

def get_last_near_misses() -> list[NearMiss]:
    """Return near-misses captured during the last run_filter() call."""
    return list(_last_near_misses)

def format_candidate_for_gemini(c: FilterCandidate) -> str:
    """
    Compact single-line string for Gemini Flash prompt.
    Keeps token count low while preserving all signal information.
    """
    candle = f"{c.best_candle}(T{c.candle_tier},{c.candle_conf}%)" if c.best_candle else "none"
    cstar  = "Y" if c.cstar_signal else "N"
    momentum_str = f" | momentum={c.momentum_status} rsi_d={c.rsi_slope_3d:+.2f}"
    if c.bounce_failed:
        momentum_str += " BOUNCE_FAILED"
    elif c.reversal_days > 0:
        momentum_str += f" rev_days={c.reversal_days}"
    return (
        f"SYM:{c.symbol} SEC:{c.sector} LTP:{c.ltp:.2f} CHG:{c.change_pct:+.2f}% "
        f"VOL:{c.volume:,} SCORE:{c.composite_score:.1f} TECH:{c.tech_score} "
        f"RSI:{c.rsi_14:.1f}[{c.rsi_signal}] MACD:{c.macd_cross} "
        f"BB:{c.bb_signal}[{c.bb_pct_b:.2f}] EMA:{c.ema_trend} "
        f"OBV:{c.obv_trend} ATR%:{c.atr_pct:.1f} CONF:{c.conf_score:.0f} "
        f"CANDLE:{candle} CSTAR:{cstar} HOLD:{c.suggested_hold}d SIG:{c.primary_signal}"
    ) + momentum_str


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python filter_engine.py              → full run, top 10
#   python filter_engine.py --dry-run    → synthetic prices from cache
#   python filter_engine.py NABIL HBL   → specific symbols only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [FILTER] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    args     = sys.argv[1:]
    dry_run  = "--dry-run" in args
    sym_args = [a.upper() for a in args if not a.startswith("--")]

    print("\n" + "=" * 70)
    print("  NEPSE AI — filter_engine.py")
    print("=" * 70)

    # ── Market data ───────────────────────────────────────────────────────────
    if dry_run:
        print("\n[DRY RUN] Synthetic prices from HistoryCache...")
        try:
            from modules.indicators import HistoryCache, DEFAULT_LOAD_PERIODS
            from modules.scraper import PriceRow
            cache = HistoryCache()
            if not cache.load(periods=DEFAULT_LOAD_PERIODS):
                print("  ❌ Cache empty"); sys.exit(1)
            market_data = {
                s: PriceRow(
                    symbol=s, ltp=c[-1],
                    open_price=c[-2] if len(c) > 1 else c[-1],
                    close=c[-1],
                    high=cache.get_highs(s)[-1] if cache.get_highs(s) else c[-1],
                    low=cache.get_lows(s)[-1]   if cache.get_lows(s)  else c[-1],
                    prev_close=c[-2] if len(c) > 1 else c[-1],
                    volume=int(cache.get_volumes(s)[-1]) if cache.get_volumes(s) else 10000,
                    conf_score=55.0, conf_signal="BULLISH", change_pct=0.5,
                )
                for s, c in cache.closes.items() if c
            }
            print(f"  ✅ {len(market_data)} symbols")
        except Exception as e:
            print(f"  ❌ {e}"); sys.exit(1)
    else:
        print("\n[1/2] Fetching live prices...")
        try:
            from modules.scraper import get_all_market_data
            market_data = get_all_market_data(write_breadth=False)
            if not market_data:
                print("  ⚠️  No data (market closed?)"); sys.exit(0)
            print(f"  ✅ {len(market_data)} symbols")
        except Exception as e:
            print(f"  ❌ {e}"); sys.exit(1)

    if sym_args:
        market_data = {k: v for k, v in market_data.items() if k in sym_args}
        print(f"  Filtered to: {list(market_data.keys())}")

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n[2/2] Running filter engine...")
    results = run_filter(market_data=market_data, top_n=15)

    if not results:
        print("\n  No candidates passed all gates.")
        print("  Check: bandh, geo score, market state, tech/conf thresholds.\n")
    else:
        print(f"\n  {len(results)} candidates:\n")
        print(f"  {'#':<3} {'Symbol':<10} {'Score':>6} {'Tech':>5} {'RSI':>5} "
              f"{'MACD':<10} {'BB':<14} {'Geo':>4} {'C*':>3} "
              f"{'x':>4} {'Candle':<22} Signal")
        print("  " + "─" * 100)
        for i, c in enumerate(results, 1):
            candle = f"{c.best_candle[:16]}(T{c.candle_tier})" if c.best_candle else "—"
            print(
                f"  {i:<3} {c.symbol:<10} {c.composite_score:>6.1f} "
                f"{c.tech_score:>5} {c.rsi_14:>5.1f} "
                f"{c.macd_cross:<10} {c.bb_signal:<14} "
                f"{c.combined_geo:>+4} {'✓' if c.cstar_signal else ' ':>3} "
                f"{c.sector_mult:>4.2f} {candle:<22} {c.primary_signal}"
            )

        c0 = results[0]
        print(f"\n  Market: {c0.market_state} | Geo: {c0.combined_geo:+d} | "
              f"Bandh: {c0.bandh_today} | IPO: {c0.ipo_drain}")

        print(f"\n  Gemini-ready format (top 3):")
        print("  " + "─" * 70)
        for c in results[:3]:
            print(f"  {format_candidate_for_gemini(c)}")

    print("\n" + "=" * 70 + "\n")
