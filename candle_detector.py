"""
candle_detector.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2B
Purpose : Detect candlestick patterns for every symbol using historical
          OHLCV data from HistoryCache (indicators.py).

Design decisions:
    - Pure price geometry — no indicators needed, no external calls.
    - Takes last 5 candles from HistoryCache for each symbol.
    - Appends today's live PriceRow as the most recent candle.
    - Returns list of CandlePattern dataclass per symbol.
    - Writes detected patterns to Neon candle_patterns table via upsert_row.
    - Called by filter_engine.py to add candle context to each candidate.

Pattern tiers:
    Tier 1 — Highest reliability (use for entry signals):
        Bullish Engulfing, Bearish Engulfing, Hammer, Shooting Star,
        Morning Star, Evening Star, Doji at Support, Doji at Resistance

    Tier 2 — Good reliability (use for confirmation):
        Three White Soldiers, Three Black Crows, Piercing Line,
        Dark Cloud Cover, Bullish Harami, Bearish Harami

    Tier 3 — Context-dependent (use as additional context only):
        Spinning Top, Marubozu (Bullish/Bearish), Inside Bar

Math note:
    All pattern math uses plain Python — no numpy/pandas.
    Tolerance thresholds are tuned for NEPSE's typical daily volatility
    (higher than NSE/BSE — NEPSE stocks often move 2–5% daily).

Usage:
    from candle_detector import detect_patterns, detect_all_patterns
    from indicators import HistoryCache

    cache = HistoryCache()
    cache.load()

    # Single symbol
    patterns = detect_patterns("NABIL", price_row, cache)
    for p in patterns:
        print(p.name, p.signal, p.confidence)

    # All symbols (called by filter_engine.py)
    all_patterns = detect_all_patterns(market_data, cache)
    nabil_patterns = all_patterns.get("NABIL", [])

─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — NEPSE-tuned thresholds
# ══════════════════════════════════════════════════════════════════════════════

# How many historical candles to use per symbol (including today)
LOOKBACK = 5

# Body/wick ratio thresholds — tuned for NEPSE volatility
# A "small body" means body is < this fraction of the full candle range
SMALL_BODY_RATIO      = 0.25   # body < 25% of high-low range → doji/spinning top
DOJI_BODY_RATIO       = 0.10   # body < 10% of range → strict doji
LONG_BODY_RATIO       = 0.60   # body > 60% of range → marubozu / engulfing

# Wick thresholds for hammer / shooting star
HAMMER_LOWER_WICK     = 2.0    # lower wick must be >= 2x body size
HAMMER_UPPER_WICK     = 0.3    # upper wick must be <= 30% of body
SHOOTING_LOWER_WICK   = 0.3    # lower wick <= 30% of body
SHOOTING_UPPER_WICK   = 2.0    # upper wick >= 2x body

# Volume confirmation — pattern is stronger with above-average volume
# Volume ratio = today's volume / avg of last 5 days
VOLUME_SURGE_RATIO    = 1.5    # 1.5x average = volume surge

# Support/Resistance proximity for Doji context
# Price is "near" S/R if within this % of a recent swing high/low
SR_PROXIMITY_PCT      = 0.03   # within 3%


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE PATTERN RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandlePattern:
    """
    A detected candlestick pattern for one symbol.
    filter_engine.py reads this and adds it to each candidate's context.
    """
    symbol:      str
    name:        str                  # e.g. "Bullish Engulfing"
    signal:      str                  # BULLISH / BEARISH / NEUTRAL
    tier:        int                  # 1 / 2 / 3
    confidence:  int                  # 0–100 (higher = more reliable)
    description: str                  # human-readable explanation
    volume_confirmed: bool = False    # True if pattern has above-avg volume
    candles_used: int = 1             # how many candles the pattern spans

    timestamp: str = field(default_factory=lambda: datetime.now(
        tz=timezone(timedelta(hours=5, minutes=45))
    ).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        vol_tag = " [VOL✓]" if self.volume_confirmed else ""
        return (
            f"T{self.tier} | {self.name} | {self.signal} | "
            f"conf={self.confidence}{vol_tag} | {self.description}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _body(o: float, c: float) -> float:
    """Absolute body size."""
    return abs(c - o)


def _range(h: float, l: float) -> float:
    """Full candle range (high - low)."""
    return h - l if h > l else 0.0


def _upper_wick(o: float, h: float, c: float) -> float:
    """Upper wick = high - max(open, close)."""
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    """Lower wick = min(open, close) - low."""
    return min(o, c) - l


def _is_bullish(o: float, c: float) -> bool:
    return c > o


def _is_bearish(o: float, c: float) -> bool:
    return c < o


def _body_ratio(o: float, h: float, l: float, c: float) -> float:
    """Body as fraction of full range. Returns 0 if range is zero."""
    r = _range(h, l)
    if r == 0:
        return 0.0
    return _body(o, c) / r


def _avg_volume(volumes: list[float], exclude_last: int = 1) -> float:
    """Average volume of all candles except the last N (today's)."""
    hist = volumes[:-exclude_last] if exclude_last else volumes
    if not hist:
        return 0.0
    return sum(hist) / len(hist)


def _volume_ratio(volumes: list[float]) -> float:
    """Today's volume vs average of prior candles."""
    if len(volumes) < 2:
        return 1.0
    avg = _avg_volume(volumes, exclude_last=1)
    if avg == 0:
        return 1.0
    return volumes[-1] / avg


def _near_support(closes: list[float], low: float, pct: float = SR_PROXIMITY_PCT) -> bool:
    """
    True if today's low is near a recent swing low (support).
    Swing low = minimum close in the lookback window.
    """
    if len(closes) < 3:
        return False
    swing_low = min(closes[:-1])  # exclude today
    return abs(low - swing_low) / swing_low <= pct if swing_low > 0 else False


def _near_resistance(closes: list[float], high: float, pct: float = SR_PROXIMITY_PCT) -> bool:
    """
    True if today's high is near a recent swing high (resistance).
    Swing high = maximum close in the lookback window.
    """
    if len(closes) < 3:
        return False
    swing_high = max(closes[:-1])  # exclude today
    return abs(high - swing_high) / swing_high <= pct if swing_high > 0 else False


def _is_downtrend(closes: list[float], periods: int = 3) -> bool:
    """Simple downtrend: last N closes are declining."""
    if len(closes) < periods + 1:
        return False
    window = closes[-(periods + 1):-1]  # exclude today
    return all(window[i] > window[i + 1] for i in range(len(window) - 1))


def _is_uptrend(closes: list[float], periods: int = 3) -> bool:
    """Simple uptrend: last N closes are rising."""
    if len(closes) < periods + 1:
        return False
    window = closes[-(periods + 1):-1]
    return all(window[i] < window[i + 1] for i in range(len(window) - 1))


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 PATTERNS — Highest reliability
# ══════════════════════════════════════════════════════════════════════════════

def _check_bullish_engulfing(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Bullish Engulfing — 2-candle reversal pattern.
    Conditions:
        - Prior candle is bearish (red)
        - Current candle is bullish (green)
        - Current body completely engulfs prior body
        - Ideally appears after a downtrend
    """
    if len(closes) < 2:
        return None

    prev_o, prev_c = opens[-2], closes[-2]
    curr_o, curr_c = opens[-1], closes[-1]

    if not _is_bearish(prev_o, prev_c):
        return None
    if not _is_bullish(curr_o, curr_c):
        return None
    # Current body must engulf prior body
    if curr_o > prev_c or curr_c < prev_o:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(closes, periods=3)

    confidence = 70
    if vol_confirmed:
        confidence += 10
    if in_downtrend:
        confidence += 10
    if _body_ratio(curr_o, highs[-1], lows[-1], curr_c) > LONG_BODY_RATIO:
        confidence += 5
    confidence = min(confidence, 95)

    return CandlePattern(
        symbol=symbol, name="Bullish Engulfing", signal="BULLISH",
        tier=1, confidence=confidence,
        description=(
            f"Green body (O={curr_o:.0f} C={curr_c:.0f}) engulfs prior "
            f"red body (O={prev_o:.0f} C={prev_c:.0f}). "
            f"{'Downtrend context. ' if in_downtrend else ''}"
            "Strong reversal signal."
        ),
        volume_confirmed=vol_confirmed, candles_used=2,
    )


def _check_bearish_engulfing(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Bearish Engulfing — 2-candle reversal pattern.
    Conditions:
        - Prior candle is bullish (green)
        - Current candle is bearish (red)
        - Current body completely engulfs prior body
        - Ideally appears after an uptrend
    """
    if len(closes) < 2:
        return None

    prev_o, prev_c = opens[-2], closes[-2]
    curr_o, curr_c = opens[-1], closes[-1]

    if not _is_bullish(prev_o, prev_c):
        return None
    if not _is_bearish(curr_o, curr_c):
        return None
    if curr_o < prev_c or curr_c > prev_o:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(closes, periods=3)

    confidence = 70
    if vol_confirmed:
        confidence += 10
    if in_uptrend:
        confidence += 10
    if _body_ratio(curr_o, highs[-1], lows[-1], curr_c) > LONG_BODY_RATIO:
        confidence += 5
    confidence = min(confidence, 95)

    return CandlePattern(
        symbol=symbol, name="Bearish Engulfing", signal="BEARISH",
        tier=1, confidence=confidence,
        description=(
            f"Red body (O={curr_o:.0f} C={curr_c:.0f}) engulfs prior "
            f"green body (O={prev_o:.0f} C={prev_c:.0f}). "
            f"{'Uptrend context. ' if in_uptrend else ''}"
            "Strong reversal signal."
        ),
        volume_confirmed=vol_confirmed, candles_used=2,
    )


def _check_hammer(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Hammer — 1-candle bullish reversal at bottom of downtrend.
    Conditions:
        - Small body in upper portion of range
        - Lower wick >= 2x body
        - Upper wick <= 30% of body (very small or absent)
        - Appears after downtrend (context)
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body      = _body(o, c)
    lower_w   = _lower_wick(o, l, c)
    upper_w   = _upper_wick(o, h, c)
    full_r    = _range(h, l)

    if full_r == 0 or body == 0:
        return None
    if lower_w < HAMMER_LOWER_WICK * body:
        return None
    if upper_w > HAMMER_UPPER_WICK * body:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(closes, periods=3)
    at_support    = _near_support(closes, l)

    confidence = 60
    if in_downtrend:
        confidence += 15
    if at_support:
        confidence += 10
    if vol_confirmed:
        confidence += 8
    if _is_bullish(o, c):
        confidence += 5   # green hammer is stronger
    confidence = min(confidence, 92)

    color = "green" if _is_bullish(o, c) else "red"
    return CandlePattern(
        symbol=symbol, name="Hammer", signal="BULLISH",
        tier=1, confidence=confidence,
        description=(
            f"{'Green' if color == 'green' else 'Red'} hammer: "
            f"long lower wick ({lower_w:.0f}), small body ({body:.0f}). "
            f"{'Near support. ' if at_support else ''}"
            f"{'Downtrend context. ' if in_downtrend else ''}"
            "Potential bullish reversal."
        ),
        volume_confirmed=vol_confirmed, candles_used=1,
    )


def _check_shooting_star(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Shooting Star — 1-candle bearish reversal at top of uptrend.
    Conditions:
        - Small body in lower portion of range
        - Upper wick >= 2x body
        - Lower wick <= 30% of body
        - Appears after uptrend (context)
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body    = _body(o, c)
    upper_w = _upper_wick(o, h, c)
    lower_w = _lower_wick(o, l, c)
    full_r  = _range(h, l)

    if full_r == 0 or body == 0:
        return None
    if upper_w < SHOOTING_UPPER_WICK * body:
        return None
    if lower_w > SHOOTING_LOWER_WICK * body:
        return None

    vol_confirmed    = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO
    in_uptrend       = _is_uptrend(closes, periods=3)
    at_resistance    = _near_resistance(closes, h)

    confidence = 60
    if in_uptrend:
        confidence += 15
    if at_resistance:
        confidence += 10
    if vol_confirmed:
        confidence += 8
    if _is_bearish(o, c):
        confidence += 5   # red shooting star is stronger
    confidence = min(confidence, 92)

    return CandlePattern(
        symbol=symbol, name="Shooting Star", signal="BEARISH",
        tier=1, confidence=confidence,
        description=(
            f"Shooting star: long upper wick ({upper_w:.0f}), "
            f"small body ({body:.0f}). "
            f"{'Near resistance. ' if at_resistance else ''}"
            f"{'Uptrend context. ' if in_uptrend else ''}"
            "Potential bearish reversal."
        ),
        volume_confirmed=vol_confirmed, candles_used=1,
    )


def _check_morning_star(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Morning Star — 3-candle bullish reversal.
    Conditions:
        - Candle 1: large bearish body
        - Candle 2: small body (gap down ideally, but NEPSE rarely gaps)
        - Candle 3: large bullish body closing above midpoint of candle 1
    """
    if len(closes) < 3:
        return None

    o1, c1 = opens[-3], closes[-3]
    o2, c2 = opens[-2], closes[-2]
    o3, c3 = opens[-1], closes[-1]
    h3, l3 = highs[-1], lows[-1]

    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)

    if body1 == 0 or body3 == 0:
        return None
    if not _is_bearish(o1, c1):
        return None
    if body2 > body1 * 0.4:  # candle 2 must be clearly smaller
        return None
    if not _is_bullish(o3, c3):
        return None
    # Candle 3 must close above midpoint of candle 1
    midpoint1 = (o1 + c1) / 2
    if c3 < midpoint1:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 72
    if vol_confirmed:
        confidence += 8
    if _is_downtrend(closes, periods=3):
        confidence += 8
    if body3 >= body1 * 0.8:  # strong recovery candle
        confidence += 7
    confidence = min(confidence, 92)

    return CandlePattern(
        symbol=symbol, name="Morning Star", signal="BULLISH",
        tier=1, confidence=confidence,
        description=(
            f"3-candle reversal: bearish ({c1:.0f}), "
            f"indecision ({c2:.0f}), "
            f"bullish recovery ({c3:.0f}) above midpoint. "
            "Strong bottom reversal."
        ),
        volume_confirmed=vol_confirmed, candles_used=3,
    )


def _check_evening_star(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Evening Star — 3-candle bearish reversal.
    Conditions:
        - Candle 1: large bullish body
        - Candle 2: small body (indecision)
        - Candle 3: large bearish body closing below midpoint of candle 1
    """
    if len(closes) < 3:
        return None

    o1, c1 = opens[-3], closes[-3]
    o2, c2 = opens[-2], closes[-2]
    o3, c3 = opens[-1], closes[-1]

    body1 = _body(o1, c1)
    body2 = _body(o2, c2)
    body3 = _body(o3, c3)

    if body1 == 0 or body3 == 0:
        return None
    if not _is_bullish(o1, c1):
        return None
    if body2 > body1 * 0.4:
        return None
    if not _is_bearish(o3, c3):
        return None
    midpoint1 = (o1 + c1) / 2
    if c3 > midpoint1:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 72
    if vol_confirmed:
        confidence += 8
    if _is_uptrend(closes, periods=3):
        confidence += 8
    if body3 >= body1 * 0.8:
        confidence += 7
    confidence = min(confidence, 92)

    return CandlePattern(
        symbol=symbol, name="Evening Star", signal="BEARISH",
        tier=1, confidence=confidence,
        description=(
            f"3-candle reversal: bullish ({c1:.0f}), "
            f"indecision ({c2:.0f}), "
            f"bearish selloff ({c3:.0f}) below midpoint. "
            "Strong top reversal."
        ),
        volume_confirmed=vol_confirmed, candles_used=3,
    )


def _check_doji(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Doji — 1-candle indecision pattern.
    Body is very small relative to range.
    Signal depends on context:
        - At support (after downtrend) → BULLISH
        - At resistance (after uptrend) → BEARISH
        - Otherwise → NEUTRAL
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    br = _body_ratio(o, h, l, c)

    if br > DOJI_BODY_RATIO:
        return None

    full_r = _range(h, l)
    if full_r == 0:
        return None

    at_support    = _near_support(closes, l)
    at_resistance = _near_resistance(closes, h)
    in_downtrend  = _is_downtrend(closes, periods=3)
    in_uptrend    = _is_uptrend(closes, periods=3)

    if at_support and in_downtrend:
        signal     = "BULLISH"
        context    = "at support after downtrend — potential reversal up"
        confidence = 68
        tier       = 1
    elif at_resistance and in_uptrend:
        signal     = "BEARISH"
        context    = "at resistance after uptrend — potential reversal down"
        confidence = 68
        tier       = 1
    else:
        signal     = "NEUTRAL"
        context    = "indecision — wait for confirmation"
        confidence = 45
        tier       = 3

    return CandlePattern(
        symbol=symbol, name="Doji", signal=signal,
        tier=tier, confidence=confidence,
        description=f"Doji (body={_body(o, c):.0f}, range={full_r:.0f}): {context}.",
        volume_confirmed=False, candles_used=1,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 PATTERNS — Good reliability, use for confirmation
# ══════════════════════════════════════════════════════════════════════════════

def _check_three_white_soldiers(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Three White Soldiers — 3 consecutive bullish candles.
    Conditions:
        - Three consecutive green candles
        - Each opens within prior body
        - Each closes higher than prior close
        - Bodies are reasonably large (not tiny)
    """
    if len(closes) < 3:
        return None

    for i in [-3, -2, -1]:
        if not _is_bullish(opens[i], closes[i]):
            return None

    # Each close higher
    if not (closes[-3] < closes[-2] < closes[-1]):
        return None

    # Each open within prior body (opens above prior open but below prior close)
    if not (opens[-3] < opens[-2] < closes[-3]):
        return None
    if not (opens[-2] < opens[-1] < closes[-2]):
        return None

    # Bodies should not be tiny (avoid small-body false positives)
    avg_body = sum(_body(opens[i], closes[i]) for i in [-3, -2, -1]) / 3
    if avg_body < 0:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 68
    if _is_downtrend(closes, periods=4):
        confidence += 10   # after downtrend = reversal
    if vol_confirmed:
        confidence += 8
    confidence = min(confidence, 88)

    return CandlePattern(
        symbol=symbol, name="Three White Soldiers", signal="BULLISH",
        tier=2, confidence=confidence,
        description=(
            f"3 consecutive green candles closing higher: "
            f"{closes[-3]:.0f} → {closes[-2]:.0f} → {closes[-1]:.0f}. "
            "Strong bullish momentum."
        ),
        volume_confirmed=vol_confirmed, candles_used=3,
    )


def _check_three_black_crows(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Three Black Crows — 3 consecutive bearish candles.
    Conditions:
        - Three consecutive red candles
        - Each opens within prior body
        - Each closes lower than prior close
    """
    if len(closes) < 3:
        return None

    for i in [-3, -2, -1]:
        if not _is_bearish(opens[i], closes[i]):
            return None

    if not (closes[-3] > closes[-2] > closes[-1]):
        return None

    if not (closes[-3] > opens[-2] > closes[-3] * 0.97):  # opens near prior close
        return None
    if not (closes[-2] > opens[-1] > closes[-2] * 0.97):
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 68
    if _is_uptrend(closes, periods=4):
        confidence += 10
    if vol_confirmed:
        confidence += 8
    confidence = min(confidence, 88)

    return CandlePattern(
        symbol=symbol, name="Three Black Crows", signal="BEARISH",
        tier=2, confidence=confidence,
        description=(
            f"3 consecutive red candles closing lower: "
            f"{closes[-3]:.0f} → {closes[-2]:.0f} → {closes[-1]:.0f}. "
            "Strong bearish momentum."
        ),
        volume_confirmed=vol_confirmed, candles_used=3,
    )


def _check_piercing_line(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Piercing Line — 2-candle bullish reversal.
    Conditions:
        - Candle 1: bearish
        - Candle 2: opens below candle 1 low, closes above midpoint of candle 1
    """
    if len(closes) < 2:
        return None

    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
    o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1]

    if not _is_bearish(o1, c1):
        return None
    if not _is_bullish(o2, c2):
        return None
    if o2 > l1:  # must open below prior low (or at least near it, NEPSE tolerance)
        if o2 > c1 * 1.005:  # allow tiny gap
            return None
    midpoint1 = (o1 + c1) / 2
    if c2 < midpoint1:
        return None
    if c2 >= o1:  # if it closes above prior open, it's Engulfing (already caught)
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 62
    if _is_downtrend(closes, periods=3):
        confidence += 10
    if vol_confirmed:
        confidence += 8
    confidence = min(confidence, 85)

    return CandlePattern(
        symbol=symbol, name="Piercing Line", signal="BULLISH",
        tier=2, confidence=confidence,
        description=(
            f"Green candle ({c2:.0f}) pierces above midpoint of prior red ({c1:.0f}). "
            "Bullish reversal — weaker than Engulfing."
        ),
        volume_confirmed=vol_confirmed, candles_used=2,
    )


def _check_dark_cloud_cover(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Dark Cloud Cover — 2-candle bearish reversal.
    Conditions:
        - Candle 1: bullish
        - Candle 2: opens above candle 1 high, closes below midpoint of candle 1
    """
    if len(closes) < 2:
        return None

    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
    o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1]

    if not _is_bullish(o1, c1):
        return None
    if not _is_bearish(o2, c2):
        return None
    if o2 < h1 * 0.995:  # must open near or above prior high
        return None
    midpoint1 = (o1 + c1) / 2
    if c2 > midpoint1:
        return None
    if c2 <= o1:  # if it closes below prior open, it's Engulfing
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    confidence = 62
    if _is_uptrend(closes, periods=3):
        confidence += 10
    if vol_confirmed:
        confidence += 8
    confidence = min(confidence, 85)

    return CandlePattern(
        symbol=symbol, name="Dark Cloud Cover", signal="BEARISH",
        tier=2, confidence=confidence,
        description=(
            f"Red candle ({c2:.0f}) closes below midpoint of prior green ({c1:.0f}). "
            "Bearish reversal — weaker than Engulfing."
        ),
        volume_confirmed=vol_confirmed, candles_used=2,
    )


def _check_harami(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Harami — 2-candle reversal (inside candle).
    Bullish Harami: large bearish → small bullish inside it
    Bearish Harami: large bullish → small bearish inside it
    """
    if len(closes) < 2:
        return None

    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
    o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1]

    body1 = _body(o1, c1)
    body2 = _body(o2, c2)

    if body1 == 0:
        return None

    # Candle 2 body must be inside candle 1 body
    high1_body = max(o1, c1)
    low1_body  = min(o1, c1)
    high2_body = max(o2, c2)
    low2_body  = min(o2, c2)

    if high2_body > high1_body or low2_body < low1_body:
        return None

    # Candle 2 body should be meaningfully smaller
    if body2 > body1 * 0.6:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    if _is_bearish(o1, c1) and _is_bullish(o2, c2):
        signal    = "BULLISH"
        name      = "Bullish Harami"
        in_trend  = _is_downtrend(closes, periods=3)
        base_conf = 58
    elif _is_bullish(o1, c1) and _is_bearish(o2, c2):
        signal    = "BEARISH"
        name      = "Bearish Harami"
        in_trend  = _is_uptrend(closes, periods=3)
        base_conf = 58
    else:
        return None

    confidence = base_conf
    if in_trend:
        confidence += 10
    if vol_confirmed:
        confidence += 7
    confidence = min(confidence, 80)

    return CandlePattern(
        symbol=symbol, name=name, signal=signal,
        tier=2, confidence=confidence,
        description=(
            f"Small {'green' if signal == 'BULLISH' else 'red'} candle "
            f"(body={body2:.0f}) inside large prior body (body={body1:.0f}). "
            f"{'Potential bottom reversal.' if signal == 'BULLISH' else 'Potential top reversal.'}"
        ),
        volume_confirmed=vol_confirmed, candles_used=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 PATTERNS — Context-dependent, use as additional information only
# ══════════════════════════════════════════════════════════════════════════════

def _check_spinning_top(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Spinning Top — small body with wicks on both sides.
    Signals indecision. Useful context — not a standalone signal.
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    br    = _body_ratio(o, h, l, c)
    upper = _upper_wick(o, h, c)
    lower = _lower_wick(o, l, c)

    if br > SMALL_BODY_RATIO:
        return None
    if br < DOJI_BODY_RATIO:
        return None  # that's a doji, already caught
    if upper < _range(h, l) * 0.1 or lower < _range(h, l) * 0.1:
        return None  # needs wicks on both sides

    return CandlePattern(
        symbol=symbol, name="Spinning Top", signal="NEUTRAL",
        tier=3, confidence=40,
        description=(
            f"Small body ({_body(o, c):.0f}) with wicks both sides. "
            "Market indecision — watch for breakout direction."
        ),
        volume_confirmed=False, candles_used=1,
    )


def _check_marubozu(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Marubozu — large body with almost no wicks.
    Bullish: opens at low, closes at high (buyers dominated all day)
    Bearish: opens at high, closes at low (sellers dominated all day)
    """
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    br      = _body_ratio(o, h, l, c)
    full_r  = _range(h, l)
    upper_w = _upper_wick(o, h, c)
    lower_w = _lower_wick(o, l, c)

    if br < LONG_BODY_RATIO:
        return None
    if full_r == 0:
        return None
    # Wicks must be tiny — each wick < 5% of range
    if upper_w > full_r * 0.05 and lower_w > full_r * 0.05:
        return None

    vol_confirmed = _volume_ratio(volumes) >= VOLUME_SURGE_RATIO

    if _is_bullish(o, c):
        signal = "BULLISH"
        name   = "Bullish Marubozu"
        desc   = f"Strong green candle (O={o:.0f} C={c:.0f}), minimal wicks. Buyers in full control."
    else:
        signal = "BEARISH"
        name   = "Bearish Marubozu"
        desc   = f"Strong red candle (O={o:.0f} C={c:.0f}), minimal wicks. Sellers in full control."

    confidence = 52
    if vol_confirmed:
        confidence += 8
    confidence = min(confidence, 70)

    return CandlePattern(
        symbol=symbol, name=name, signal=signal,
        tier=3, confidence=confidence,
        description=desc,
        volume_confirmed=vol_confirmed, candles_used=1,
    )


def _check_inside_bar(
    opens: list[float], highs: list[float],
    lows: list[float], closes: list[float],
    volumes: list[float], symbol: str,
) -> Optional[CandlePattern]:
    """
    Inside Bar — today's high-low range is completely inside prior candle's range.
    Signals consolidation — breakout pending in either direction.
    """
    if len(highs) < 2:
        return None

    if highs[-1] >= highs[-2] or lows[-1] <= lows[-2]:
        return None

    return CandlePattern(
        symbol=symbol, name="Inside Bar", signal="NEUTRAL",
        tier=3, confidence=38,
        description=(
            f"Today's range ({lows[-1]:.0f}–{highs[-1]:.0f}) is inside "
            f"prior range ({lows[-2]:.0f}–{highs[-2]:.0f}). "
            "Consolidation — watch for breakout."
        ),
        volume_confirmed=False, candles_used=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN REGISTRY
# All pattern detector functions in priority order.
# Tier 1 first — if a Tier 1 fires, lower tiers are still checked
# (multiple patterns can coexist on the same candle).
# ══════════════════════════════════════════════════════════════════════════════

_DETECTORS = [
    # Tier 1
    _check_bullish_engulfing,
    _check_bearish_engulfing,
    _check_hammer,
    _check_shooting_star,
    _check_morning_star,
    _check_evening_star,
    _check_doji,
    # Tier 2
    _check_three_white_soldiers,
    _check_three_black_crows,
    _check_piercing_line,
    _check_dark_cloud_cover,
    _check_harami,
    # Tier 3
    _check_spinning_top,
    _check_marubozu,
    _check_inside_bar,
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_patterns(
    symbol:    str,
    price_row,              # scraper.PriceRow — today's live data
    cache,                  # indicators.HistoryCache
) -> list[CandlePattern]:
    """
    Detect all candlestick patterns for one symbol.

    Args:
        symbol:    Stock symbol e.g. "NABIL"
        price_row: Today's PriceRow from scraper.py
        cache:     Loaded HistoryCache from indicators.py

    Returns:
        List of CandlePattern (may be empty if no pattern fires).
        Sorted by tier (Tier 1 first), then confidence (highest first).
    """
    sym = symbol.upper()

    # Pull historical OHLCV arrays (oldest → newest, excludes today)
    hist_opens   = cache.get_closes(sym)   # OmitNomis doesn't always have open
    hist_closes  = cache.get_closes(sym)
    hist_highs   = cache.get_highs(sym)
    hist_lows    = cache.get_lows(sym)
    hist_volumes = cache.get_volumes(sym)

    # We need at least 1 prior candle for any 1-candle pattern
    # (we compare today against history for trend context)
    if len(hist_closes) < 1:
        return []

    # ── Build open series ──────────────────────────────────────────────────
    # OmitNomis has open prices — use them if available via cache opens
    # For now the cache only exposes closes/highs/lows/volumes;
    # we approximate historical opens as previous close (conservative).
    # This is accurate enough for body-size calculations.
    # Today's open comes from PriceRow directly.
    hist_approx_opens = hist_closes[:]   # prior close ≈ next open in NEPSE

    # Append today's live candle
    today_open   = price_row.open_price if price_row.open_price > 0 else price_row.prev_close
    today_high   = price_row.high       if price_row.high   > 0 else price_row.ltp
    today_low    = price_row.low        if price_row.low    > 0 else price_row.ltp
    today_close  = price_row.ltp        if price_row.ltp    > 0 else price_row.close
    today_volume = float(price_row.volume)

    # Take last LOOKBACK-1 historical candles + today = LOOKBACK total
    n = LOOKBACK - 1
    opens   = hist_approx_opens[-n:] + [today_open]
    closes  = hist_closes[-n:]       + [today_close]
    highs   = hist_highs[-n:]        + [today_high]
    lows    = hist_lows[-n:]         + [today_low]
    volumes = hist_volumes[-n:]      + [today_volume]

    # Need full close history for trend detection (not just last 5)
    full_closes = hist_closes + [today_close]

    # Replace closes in detector calls with full_closes for trend context
    # but use the windowed arrays for pattern geometry
    patterns: list[CandlePattern] = []

    for detector in _DETECTORS:
        try:
            result = detector(opens, highs, lows, full_closes, volumes, sym)
            if result:
                patterns.append(result)
        except Exception as exc:
            logger.debug(
                "candle_detector: %s raised error in %s — %s",
                sym, detector.__name__, exc,
            )

    # Sort: Tier 1 first, then by confidence descending
    patterns.sort(key=lambda p: (p.tier, -p.confidence))

    if patterns:
        logger.debug(
            "%s: %d pattern(s) detected: %s",
            sym,
            len(patterns),
            ", ".join(p.name for p in patterns),
        )

    return patterns


def detect_all_patterns(
    market_data: dict,   # dict[symbol, PriceRow] from scraper.py
    cache,               # indicators.HistoryCache
) -> dict[str, list[CandlePattern]]:
    """
    Detect patterns for ALL symbols in market_data.
    Called by filter_engine.py after scraper runs.

    Returns:
        dict[symbol, list[CandlePattern]]
        Symbols with no patterns map to empty list.
    """
    results: dict[str, list[CandlePattern]] = {}
    total_patterns = 0

    for symbol, price_row in market_data.items():
        patterns = detect_patterns(symbol, price_row, cache)
        results[symbol] = patterns
        total_patterns += len(patterns)

    tier1_count = sum(
        1 for plist in results.values()
        for p in plist if p.tier == 1
    )
    logger.info(
        "detect_all_patterns: %d symbols checked | %d total patterns | %d Tier 1",
        len(results), total_patterns, tier1_count,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# NEON DB WRITE — called by filter_engine.py or morning_brief.yml
# ══════════════════════════════════════════════════════════════════════════════

def write_patterns_to_db(
    all_patterns: dict[str, list[CandlePattern]],
) -> int:
    """
    Write detected patterns to Neon candle_patterns table.
    One row per (symbol, pattern_name) — upserted on conflict.

    The candle_patterns table is a REFERENCE table that stores
    pattern metadata (win rates, best sectors, etc.) — it is NOT
    a time-series log of every detected pattern per day.

    For per-day signal tracking, filter_engine.py writes the
    pattern name string to market_log.candle_pattern column instead.

    Returns:
        Number of rows written.
    """
    try:
        from db import upsert_row
    except ImportError:
        logger.warning("write_patterns_to_db: db module not available")
        return 0

    written = 0
    for symbol, patterns in all_patterns.items():
        for p in patterns:
            try:
                upsert_row(
                    "candle_patterns",
                    {
                        "pattern_name": p.name,
                        "type":         p.signal,
                        "tier":         str(p.tier),
                        "reliability":  str(p.confidence),
                        "notes":        p.description,
                    },
                    conflict_columns=["pattern_name"],
                )
                written += 1
            except Exception as exc:
                logger.warning(
                    "write_patterns_to_db: failed for %s/%s — %s",
                    symbol, p.name, exc,
                )

    logger.info("write_patterns_to_db: %d rows written to Neon", written)
    return written


def get_top_patterns(
    all_patterns: dict[str, list[CandlePattern]],
    signal: str = "BULLISH",
    min_tier: int = 1,
    max_tier: int = 2,
    min_confidence: int = 60,
) -> list[tuple[str, CandlePattern]]:
    """
    Filter patterns by signal type, tier, and confidence.
    Returns list of (symbol, CandlePattern) sorted by confidence.
    Used by filter_engine.py to shortlist candidates.

    Args:
        all_patterns:   Output of detect_all_patterns()
        signal:         "BULLISH" / "BEARISH" / "NEUTRAL" / "ANY"
        min_tier:       Minimum tier to include (1 = best)
        max_tier:       Maximum tier to include
        min_confidence: Minimum confidence score (0-100)

    Returns:
        List of (symbol, CandlePattern), sorted by confidence desc.
    """
    results: list[tuple[str, CandlePattern]] = []

    for sym, patterns in all_patterns.items():
        for p in patterns:
            if p.tier < min_tier or p.tier > max_tier:
                continue
            if p.confidence < min_confidence:
                continue
            if signal != "ANY" and p.signal != signal:
                continue
            results.append((sym, p))

    results.sort(key=lambda x: x[1].confidence, reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python candle_detector.py NABIL HBL HIDCL   → detect + print patterns
#   python candle_detector.py --all              → scan all symbols (needs live data)
#   python candle_detector.py --bullish          → show only BULLISH Tier 1/2
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [CANDLE] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]

    print("\n" + "=" * 70)
    print("  NEPSE AI — candle_detector.py")
    print("=" * 70)

    # Load history cache
    print("\n[1/3] Loading history cache...")
    try:
        from indicators import HistoryCache, DEFAULT_LOAD_PERIODS
        cache = HistoryCache()
        count = cache.load(periods=DEFAULT_LOAD_PERIODS)
        if count == 0:
            print("  ❌ Cache load failed")
            sys.exit(1)
        print(f"  ✅ {count} symbols | {len(cache.dates)} days")
    except Exception as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Fetch live prices
    print("\n[2/3] Fetching live prices...")
    try:
        from scraper import get_all_market_data, PriceRow
        market_data = get_all_market_data(write_breadth=False)
        if not market_data:
            print("  ⚠️  Market closed — building synthetic prices from cache")
            market_data = {}
            for sym, closes in cache.closes.items():
                if closes:
                    highs = cache.get_highs(sym)
                    lows  = cache.get_lows(sym)
                    market_data[sym] = PriceRow(
                        symbol     = sym,
                        ltp        = closes[-1],
                        open_price = closes[-2] if len(closes) > 1 else closes[-1],
                        close      = closes[-1],
                        high       = highs[-1] if highs else closes[-1],
                        low        = lows[-1]  if lows  else closes[-1],
                        prev_close = closes[-2] if len(closes) > 1 else closes[-1],
                        volume     = 10000,
                    )
        print(f"  ✅ {len(market_data)} symbols")
    except Exception as e:
        print(f"  ❌ Scraper failed: {e}")
        sys.exit(1)

    # Filter symbols if specified
    if args and not args[0].startswith("--"):
        requested   = {a.upper() for a in args}
        market_data = {k: v for k, v in market_data.items() if k in requested}

    # Detect patterns
    print(f"\n[3/3] Detecting patterns...")
    all_patterns = detect_all_patterns(market_data, cache)

    # Collect all detected patterns for display
    found = [
        (sym, p)
        for sym, plist in all_patterns.items()
        for p in plist
    ]

    # Apply --bullish filter
    if "--bullish" in args:
        found = [(s, p) for s, p in found if p.signal == "BULLISH"]
    elif "--bearish" in args:
        found = [(s, p) for s, p in found if p.signal == "BEARISH"]

    found.sort(key=lambda x: (x[1].tier, -x[1].confidence))

    print(f"  ✅ {len(found)} patterns detected across {len(market_data)} symbols\n")

    if not found:
        print("  No patterns found.")
        sys.exit(0)

    # Print results
    print(f"  {'Symbol':<10} {'Tier':<5} {'Pattern':<25} {'Signal':<10} {'Conf':>5} {'Vol':>5}  Description")
    print("  " + "-" * 100)

    for sym, p in found[:40]:
        vol_tag = "✓" if p.volume_confirmed else " "
        print(
            f"  {sym:<10} T{p.tier:<4} {p.name:<25} {p.signal:<10} "
            f"{p.confidence:>5}% {vol_tag:>4}  {p.description[:55]}"
        )

    # Summary
    from collections import Counter
    tier_dist   = Counter(p.tier   for _, p in found)
    signal_dist = Counter(p.signal for _, p in found)

    print(f"\n  Tier breakdown:   " + " | ".join(f"T{k}: {v}" for k, v in sorted(tier_dist.items())))
    print(f"  Signal breakdown: " + " | ".join(f"{k}: {v}" for k, v in sorted(signal_dist.items())))

    top_bullish = get_top_patterns(all_patterns, signal="BULLISH", max_tier=2, min_confidence=65)
    if top_bullish:
        print(f"\n  🟢 Top BULLISH signals (Tier 1-2, conf ≥65%):")
        for sym, p in top_bullish[:10]:
            vol_tag = " [VOL✓]" if p.volume_confirmed else ""
            print(f"    {sym:<10} T{p.tier} {p.name}{vol_tag} — conf={p.confidence}%")

    top_bearish = get_top_patterns(all_patterns, signal="BEARISH", max_tier=2, min_confidence=65)
    if top_bearish:
        print(f"\n  🔴 Top BEARISH signals (Tier 1-2, conf ≥65%):")
        for sym, p in top_bearish[:10]:
            vol_tag = " [VOL✓]" if p.volume_confirmed else ""
            print(f"    {sym:<10} T{p.tier} {p.name}{vol_tag} — conf={p.confidence}%")

    print("=" * 70 + "\n")
