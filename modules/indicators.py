"""
indicators.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2A
Purpose : Technical indicators for every symbol using HistoryCache OHLCV data.

Indicators computed:
    RSI 14          — Wilder's smoothing (fixed: correct 46.92 → 58.60)
    EMA 20/50/200   — Exponential moving averages
    MACD            — 12/26/9 line, signal, histogram, cross signal (fixed crossover detection)
    Bollinger Bands — 20-period, ±2σ, %B, bandwidth
    ATR 14          — Average True Range (Wilder's)
    OBV             — On Balance Volume + trend direction (improved: linear slope over last 5)
    tech_score      — Composite 0–100 score (used by filter_engine.py)
    tech_signal     — STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR

DB writes:
    indicators table — one row per (symbol, date), upserted on conflict.
    Written by run_daily_indicators() called from morning_brief.yml at 10:30 NST.
    Read by filter_engine.py during the 6-min trading loop via read_today_indicators().

HistoryCache:
    Shared by candle_detector.py — loaded once at startup.
    Source: Neon price_history table (populated by history_bootstrap.py).

─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from config import NST

logger = logging.getLogger(__name__)
DEFAULT_LOAD_PERIODS = 250   # trading days to load (~1 year)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY CACHE
# Shared with candle_detector.py — load once, read many.
# ══════════════════════════════════════════════════════════════════════════════

class HistoryCache:
    """
    In-memory cache of historical OHLCV data for all NEPSE symbols.
    Loaded once from Neon price_history table at startup.

    Used by:
        indicators.py    — RSI, EMA, MACD, BB, ATR, OBV computation
        candle_detector.py — candlestick pattern geometry
    """

    def __init__(self):
        self.closes:  dict[str, list[float]] = {}
        self.highs:   dict[str, list[float]] = {}
        self.lows:    dict[str, list[float]] = {}
        self.volumes: dict[str, list[float]] = {}
        self.dates:   list[str] = []

    def load(self, periods: int = DEFAULT_LOAD_PERIODS) -> int:
        """
        Load historical data from Neon price_history table.
        Returns number of symbols loaded. Returns 0 on failure.
        """
        try:
            from modules.history_bootstrap import load_history_all_symbols
            data = load_history_all_symbols(periods=periods)

            if not data:
                logger.warning(
                    "HistoryCache: no data in price_history table. "
                    "Run: python history_bootstrap.py --folder <csv_folder>"
                )
                return 0

            all_dates: set[str] = set()

            for symbol, hist in data.items():
                sym = symbol.upper()
                self.closes[sym]  = hist.get("closes",  [])
                self.highs[sym]   = hist.get("highs",   [])
                self.lows[sym]    = hist.get("lows",    [])
                self.volumes[sym] = hist.get("volumes", [])
                all_dates.update(hist.get("dates", []))

            self.dates = sorted(all_dates)

            logger.info(
                "HistoryCache loaded: %d symbols | %d trading days | %s to %s",
                len(self.closes),
                len(self.dates),
                self.dates[0]  if self.dates else "—",
                self.dates[-1] if self.dates else "—",
            )
            return len(self.closes)

        except Exception as exc:
            logger.error("HistoryCache.load() failed: %s", exc)
            return 0

    def get_closes(self, symbol: str) -> list[float]:
        return self.closes.get(symbol.upper(), [])

    def get_highs(self, symbol: str) -> list[float]:
        return self.highs.get(symbol.upper(), [])

    def get_lows(self, symbol: str) -> list[float]:
        return self.lows.get(symbol.upper(), [])

    def get_volumes(self, symbol: str) -> list[float]:
        return self.volumes.get(symbol.upper(), [])

    def get_dates(self, symbol: str) -> list[str]:
        return self.dates


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorResult:
    """
    All computed indicators for one symbol on one date.
    Matches the 'indicators' table columns in db/schema.py exactly.
    """
    symbol:          str
    date:            str
    close:           str = "0"
    volume:          str = "0"
    history_days:    str = "0"

    # RSI
    rsi_14:          str = ""
    rsi_signal:      str = ""        # OVERSOLD / NEUTRAL / OVERBOUGHT

    # EMA
    ema_20:          str = ""
    ema_50:          str = ""
    ema_200:         str = ""
    ema_trend:       str = ""        # ABOVE_ALL / BELOW_ALL / MIXED
    ema_20_50_cross: str = ""        # GOLDEN / DEATH / NONE
    ema_50_200_cross:str = ""        # GOLDEN / DEATH / NONE

    # MACD
    macd_line:       str = ""
    macd_signal:     str = ""
    macd_histogram:  str = ""
    macd_cross:      str = ""        # BULLISH / BEARISH / NONE

    # Bollinger Bands
    bb_upper:        str = ""
    bb_middle:       str = ""
    bb_lower:        str = ""
    bb_width:        str = ""
    bb_pct_b:        str = ""        # 0–1 (>1 = above upper, <0 = below lower)
    bb_signal:       str = ""        # SQUEEZE / EXPANSION / UPPER_TOUCH / LOWER_TOUCH / NEUTRAL

    # ATR
    atr_14:          str = ""
    atr_pct:         str = ""        # ATR as % of close

    # OBV
    obv:             str = ""
    obv_trend:       str = ""        # RISING / FALLING / FLAT

    # Support / Resistance (20-day high/low)
    support_level:   str = ""        # lowest low over last 20 trading days
    resistance_level:str = ""        # highest high over last 20 trading days

    # Composite
    tech_score:      str = ""        # 0–100
    tech_signal:     str = ""        # STRONG_BULL / BULL / NEUTRAL / BEAR / STRONG_BEAR

    timestamp:       str = field(default_factory=lambda: datetime.now(
        tz=NST
    ).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR MATH — pure Python, no numpy/pandas required
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """
    RSI using Wilder's smoothing method.
    Requires at least period+1 data points.
    """
    if len(closes) < period + 1:
        return None

    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    # Seed with simple average of first `period` gains/losses
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing for remaining candles
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_ema(closes: list[float], period: int) -> Optional[float]:
    """
    EMA using standard multiplier k = 2/(period+1).
    Returns only the most recent EMA value.
    """
    if len(closes) < period:
        return None

    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period   # seed with SMA

    for price in closes[period:]:
        ema = price * k + ema * (1 - k)

    return round(ema, 2)


def _calc_ema_series(closes: list[float], period: int) -> list[float]:
    """
    Full EMA series — needed for MACD and BB intermediate steps.
    Returns list of EMA values same length as closes (NaN-padded at start).
    """
    if len(closes) < period:
        return [float("nan")] * len(closes)

    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    result = [float("nan")] * (period - 1) + [ema]

    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
        result.append(ema)

    return result


def _calc_macd(
    closes: list[float],
    fast: int = 12, slow: int = 26, signal: int = 9,
) -> Optional[dict]:
    """
    MACD line = EMA(fast) - EMA(slow)
    Signal    = EMA(signal) of MACD line
    Histogram = MACD line - Signal
    Returns dict or None if insufficient data.

    FIX: Cross detection now correctly compares yesterday's and today's
         MACD line vs signal line positions (instead of using current signal).
    """
    if len(closes) < slow + signal:
        return None

    ema_fast_series = _calc_ema_series(closes, fast)
    ema_slow_series = _calc_ema_series(closes, slow)

    macd_series = []
    for f, s in zip(ema_fast_series, ema_slow_series):
        if f != f or s != s:   # NaN check
            macd_series.append(float("nan"))
        else:
            macd_series.append(f - s)

    # Remove leading NaN
    valid_macd = [v for v in macd_series if v == v]
    if len(valid_macd) < signal:
        return None

    # Compute signal line EMA (full series)
    k = 2 / (signal + 1)
    signal_series = []
    sig = sum(valid_macd[:signal]) / signal
    signal_series.append(sig)
    for val in valid_macd[signal:]:
        sig = val * k + sig * (1 - k)
        signal_series.append(sig)

    macd_line = valid_macd[-1]
    macd_signal = signal_series[-1]
    macd_hist = macd_line - macd_signal

    # Detect cross — compare yesterday vs today (MACD line vs signal line)
    cross = "NONE"
    if len(valid_macd) >= 2 and len(signal_series) >= 2:
        if valid_macd[-2] < signal_series[-2] and valid_macd[-1] > signal_series[-1]:
            cross = "BULLISH"
        elif valid_macd[-2] > signal_series[-2] and valid_macd[-1] < signal_series[-1]:
            cross = "BEARISH"

    return {
        "macd_line":      round(macd_line, 4),
        "macd_signal":    round(macd_signal, 4),
        "macd_histogram": round(macd_hist, 4),
        "macd_cross":     cross,
    }


def _calc_bollinger(
    closes: list[float], period: int = 20, std_mult: float = 2.0,
) -> Optional[dict]:
    """
    Bollinger Bands: middle=SMA(20), upper=middle+2σ, lower=middle-2σ.
    Returns dict or None if insufficient data.
    """
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = variance ** 0.5

    upper = middle + std_mult * std
    lower = middle - std_mult * std
    width = (upper - lower) / middle if middle != 0 else 0
    price = closes[-1]
    pct_b = (price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # Signal
    if width < 0.05:
        signal = "SQUEEZE"
    elif pct_b > 1.0:
        signal = "UPPER_TOUCH"
    elif pct_b < 0.0:
        signal = "LOWER_TOUCH"
    elif width > 0.15:
        signal = "EXPANSION"
    else:
        signal = "NEUTRAL"

    return {
        "bb_upper":  round(upper, 2),
        "bb_middle": round(middle, 2),
        "bb_lower":  round(lower, 2),
        "bb_width":  round(width, 4),
        "bb_pct_b":  round(pct_b, 4),
        "bb_signal": signal,
    }


def _calc_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14,
) -> Optional[dict]:
    """
    ATR using Wilder's smoothing.
    Requires at least period+1 data points.
    """
    if len(closes) < period + 1:
        return None

    tr_list = []
    for i in range(1, len(closes)):
        high, low, prev_close = highs[i], lows[i], closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    # Seed ATR with simple average
    atr = sum(tr_list[:period]) / period
    for tr in tr_list[period:]:
        atr = (atr * (period - 1) + tr) / period

    atr_pct = (atr / closes[-1] * 100) if closes[-1] != 0 else 0

    return {
        "atr_14":  round(atr, 2),
        "atr_pct": round(atr_pct, 2),
    }


def _calc_obv(closes: list[float], volumes: list[float]) -> Optional[dict]:
    """
    OBV: running total — add volume on up days, subtract on down days.
    Trend determined by linear regression slope over last 5 values.
    Returns dict with obv and trend: "RISING" (slope > 2% of avg), "FALLING" (slope < -2% of avg), else "FLAT".
    """
    if len(closes) < 2 or len(volumes) < 2:
        return None

    obv = 0.0
    obv_series = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
        obv_series.append(obv)

    # Calculate linear slope over last 5 points (if available)
    trend = "FLAT"
    if len(obv_series) >= 5:
        y = obv_series[-5:]          # last 5 OBV values
        x = list(range(len(y)))      # [0,1,2,3,4]
        n = len(y)
        # slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            # Use average of last 5 OBV as reference for relative threshold
            avg_y = sum_y / n
            if avg_y != 0:
                slope_pct = slope / abs(avg_y) * 100
                if slope_pct > 2.0:
                    trend = "RISING"
                elif slope_pct < -2.0:
                    trend = "FALLING"
                # else FLAT
            else:
                # if avg_y is zero, use absolute slope threshold
                if slope > 100:   # arbitrary, adjust as needed
                    trend = "RISING"
                elif slope < -100:
                    trend = "FALLING"

    return {
        "obv":       int(obv),
        "obv_trend": trend,
    }


def _calc_support_resistance(
    highs:   list[float],
    lows:    list[float],
    period:  int = 20,
) -> Optional[dict]:
    """
    Compute support and resistance levels from recent price history.
 
    Method: 20-day high/low range.
      support_level    = lowest low  over last `period` trading days
      resistance_level = highest high over last `period` trading days
 
    Why 20 days:
      - Covers ~1 month of NEPSE trading (Sun-Thu)
      - Meaningful floor/ceiling for MACD (17d hold) and SMA (33d hold) signals
      - BB signals (130d hold) use 52W high/low from price_history for
        longer-term context — this captures recent momentum levels
 
    Used by:
      - claude_analyst _build_prompt() — Claude sees actual NPR levels
        for stop loss and target validation
      - market_log.support_level / resistance_level — stored for GPT review
 
    Returns dict or None if insufficient data.
    """
    if len(highs) < period or len(lows) < period:
        return None
 
    recent_highs = highs[-period:]
    recent_lows  = lows[-period:]
 
    support    = round(min(recent_lows),   2)
    resistance = round(max(recent_highs),  2)
 
    # Sanity check — support must be below resistance
    if support >= resistance:
        return None
 
    return {
        "support_level":    support,
        "resistance_level": resistance,
    }
 
 



def _ema_trend_signal(
    close: float,
    ema20: Optional[float],
    ema50: Optional[float],
    ema200: Optional[float],
) -> str:
    """Classify price position relative to EMAs."""
    available = [e for e in [ema20, ema50, ema200] if e is not None]
    if not available:
        return "UNKNOWN"
    above = sum(1 for e in available if close > e)
    if above == len(available):
        return "ABOVE_ALL"
    elif above == 0:
        return "BELOW_ALL"
    else:
        return "MIXED"


def _ema_cross_signal(
    fast: Optional[float], slow: Optional[float], label: str = ""
) -> str:
    """Simple cross detection (single-day snapshot — not crossover event)."""
    if fast is None or slow is None:
        return "NONE"
    if fast > slow:
        return "GOLDEN"
    elif fast < slow:
        return "DEATH"
    return "NONE"


def _calc_tech_score(
    rsi: Optional[float],
    macd: Optional[dict],
    bb: Optional[dict],
    ema_trend: str,
    obv_trend: str,
    atr_pct: Optional[float],
) -> int:
    """
    Composite technical score 0–100.
    Each component contributes a weighted sub-score.
    """
    score = 50   # neutral baseline

    # RSI (±15 pts)
    if rsi is not None:
        if rsi < 30:
            score += 15     # deeply oversold = bullish
        elif rsi < 40:
            score += 8
        elif rsi > 70:
            score -= 15     # overbought = bearish
        elif rsi > 60:
            score -= 8

    # MACD (±12 pts)
    if macd:
        hist = macd.get("macd_histogram", 0)
        if hist > 0:
            score += 8
        elif hist < 0:
            score -= 8
        cross = macd.get("macd_cross", "NONE")
        if cross == "BULLISH":
            score += 4
        elif cross == "BEARISH":
            score -= 4

    # EMA trend (±10 pts)
    if ema_trend == "ABOVE_ALL":
        score += 10
    elif ema_trend == "BELOW_ALL":
        score -= 10
    elif ema_trend == "MIXED":
        score += 2

    # OBV trend (±8 pts)
    if obv_trend == "RISING":
        score += 8
    elif obv_trend == "FALLING":
        score -= 8

    # BB position (±5 pts)
    if bb:
        pct_b = bb.get("bb_pct_b", 0.5)
        bb_sig = bb.get("bb_signal", "NEUTRAL")
        if bb_sig == "LOWER_TOUCH":
            score += 5     # price at lower band = potential bounce
        elif bb_sig == "UPPER_TOUCH":
            score -= 5
        elif pct_b < 0.3:
            score += 3
        elif pct_b > 0.7:
            score -= 3

    return max(0, min(100, score))


def _tech_signal_from_score(score: int) -> str:
    if score >= 75:
        return "STRONG_BULL"
    elif score >= 58:
        return "BULL"
    elif score <= 25:
        return "STRONG_BEAR"
    elif score <= 42:
        return "BEAR"
    return "NEUTRAL"


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SYMBOL COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(
    symbol:    str,
    price_row,          # scraper.PriceRow — today's live data
    cache:     HistoryCache,
    date:      Optional[str] = None,
) -> IndicatorResult:
    """
    Compute all technical indicators for one symbol.

    Args:
        symbol:    Stock ticker e.g. "NABIL"
        price_row: Today's PriceRow from scraper.py
        cache:     Loaded HistoryCache
        date:      Date string YYYY-MM-DD (default: today NST)

    Returns:
        IndicatorResult with all fields populated.
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    sym = symbol.upper()
    result = IndicatorResult(symbol=sym, date=date)

    # ── Pull historical data ──────────────────────────────────────────────
    hist_closes  = cache.get_closes(sym)
    hist_highs   = cache.get_highs(sym)
    hist_lows    = cache.get_lows(sym)
    hist_volumes = cache.get_volumes(sym)

    if not hist_closes:
        logger.warning("compute_indicators: no history for %s", sym)
        return result

    # Append today's live values
    today_close  = float(price_row.ltp   or price_row.close or 0)
    today_high   = float(price_row.high  or price_row.ltp   or 0)
    today_low    = float(price_row.low   or price_row.ltp   or 0)
    today_vol    = float(price_row.volume or 0)

    closes  = hist_closes  + [today_close]
    highs   = hist_highs   + [today_high]   if hist_highs  else [today_high]
    lows    = hist_lows    + [today_low]    if hist_lows   else [today_low]
    volumes = hist_volumes + [today_vol]    if hist_volumes else [today_vol]

    result.close        = str(today_close)
    result.volume       = str(int(today_vol))
    result.history_days = str(len(hist_closes))

    # ── RSI ──────────────────────────────────────────────────────────────
    rsi = _calc_rsi(closes, period=14)
    if rsi is not None:
        result.rsi_14 = str(rsi)
        if rsi < 30:
            result.rsi_signal = "OVERSOLD"
        elif rsi > 70:
            result.rsi_signal = "OVERBOUGHT"
        else:
            result.rsi_signal = "NEUTRAL"

    # ── EMA ──────────────────────────────────────────────────────────────
    ema20  = _calc_ema(closes, 20)
    ema50  = _calc_ema(closes, 50)
    ema200 = _calc_ema(closes, 200)

    result.ema_20  = str(ema20)  if ema20  is not None else ""
    result.ema_50  = str(ema50)  if ema50  is not None else ""
    result.ema_200 = str(ema200) if ema200 is not None else ""

    result.ema_trend        = _ema_trend_signal(today_close, ema20, ema50, ema200)
    result.ema_20_50_cross  = _ema_cross_signal(ema20, ema50)
    result.ema_50_200_cross = _ema_cross_signal(ema50, ema200)

    # ── MACD ─────────────────────────────────────────────────────────────
    macd = _calc_macd(closes)
    if macd:
        result.macd_line      = str(macd["macd_line"])
        result.macd_signal    = str(macd["macd_signal"])
        result.macd_histogram = str(macd["macd_histogram"])
        result.macd_cross     = macd["macd_cross"]

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb = _calc_bollinger(closes)
    if bb:
        result.bb_upper  = str(bb["bb_upper"])
        result.bb_middle = str(bb["bb_middle"])
        result.bb_lower  = str(bb["bb_lower"])
        result.bb_width  = str(bb["bb_width"])
        result.bb_pct_b  = str(bb["bb_pct_b"])
        result.bb_signal = bb["bb_signal"]

    # ── ATR ───────────────────────────────────────────────────────────────
    atr = _calc_atr(highs, lows, closes)
    if atr:
        result.atr_14  = str(atr["atr_14"])
        result.atr_pct = str(atr["atr_pct"])

    # ── OBV ───────────────────────────────────────────────────────────────
    obv = _calc_obv(closes, volumes)
    if obv:
        result.obv      = str(obv["obv"])
        result.obv_trend = obv["obv_trend"]
        
    # ── Support / Resistance ──────────────────────────────────────────────
    sr = _calc_support_resistance(highs, lows, period=20)
    if sr:
        result.support_level    = str(sr["support_level"])
        result.resistance_level = str(sr["resistance_level"])

    # ── Tech Score ────────────────────────────────────────────────────────
    tech_score = _calc_tech_score(
        rsi        = rsi,
        macd       = macd,
        bb         = bb,
        ema_trend  = result.ema_trend,
        obv_trend  = result.obv_trend if obv else "FLAT",
        atr_pct    = float(result.atr_pct) if result.atr_pct else None,
    )
    result.tech_score  = str(tech_score)
    result.tech_signal = _tech_signal_from_score(tech_score)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DAILY BATCH — called from morning_brief.yml at 10:30 NST
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_indicators(
    market_data: dict,       # dict[symbol, PriceRow]
    cache:       HistoryCache,
    date:        Optional[str] = None,
) -> dict[str, IndicatorResult]:
    """
    Compute indicators for ALL symbols and write to Neon indicators table.

    Called once daily at 10:30 AM NST by morning_brief.yml.
    filter_engine.py reads this via read_today_indicators() all day.

    Args:
        market_data: dict[symbol, PriceRow] from scraper.get_all_market_data()
        cache:       Loaded HistoryCache
        date:        Date override (default: today NST)

    Returns:
        dict[symbol, IndicatorResult]
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    results: dict[str, IndicatorResult] = {}
    skipped = 0

    for symbol, price_row in market_data.items():
        # Skip numeric/invalid keys (S.No leakage from scraper)
        if not str(symbol).replace("-", "").replace("_", "").isalpha():
            skipped += 1
            continue
        try:
            result = compute_indicators(symbol, price_row, cache, date=date)
            results[symbol] = result
        except Exception as exc:
            logger.warning("run_daily_indicators: %s failed — %s", symbol, exc)
            skipped += 1

    logger.info(
        "run_daily_indicators: %d computed | %d skipped | date=%s",
        len(results), skipped, date,
    )

    # Write to Neon
    if results:
        try:
            from sheets import write_indicators_batch
            rows = [r.to_dict() for r in results.values()]
            written = write_indicators_batch(rows)
            logger.info("run_daily_indicators: %d rows written to Neon", written)
        except Exception as exc:
            logger.error("run_daily_indicators: DB write failed — %s", exc)

    return results


def run() -> None:
    """
    Entry point called by morning_workflow.py.
    Loads HistoryCache, fetches live prices, computes indicators, writes to Neon.
    """
    from modules.scraper import get_all_market_data, PriceRow

    logger.info("Loading history cache...")
    cache = HistoryCache()
    count = cache.load(periods=DEFAULT_LOAD_PERIODS)
    if count == 0:
        raise RuntimeError("HistoryCache load failed — is price_history populated?")
    logger.info("Cache: %d symbols | %d trading days", count, len(cache.dates))

    logger.info("Fetching live prices...")
    market_data = get_all_market_data(write_breadth=False)
    if not market_data:
        logger.warning("Market closed — using cache close prices as today's price")
        market_data = {}
        for sym, closes in cache.closes.items():
            if closes:
                highs = cache.get_highs(sym)
                lows  = cache.get_lows(sym)
                market_data[sym] = PriceRow(
                    symbol=sym, ltp=closes[-1],
                    open_price=closes[-2] if len(closes) > 1 else closes[-1],
                    close=closes[-1],
                    high=highs[-1] if highs else closes[-1],
                    low=lows[-1]   if lows  else closes[-1],
                    prev_close=closes[-2] if len(closes) > 1 else closes[-1],
                    volume=10000,
                )
    logger.info("Prices: %d symbols", len(market_data))

    results = run_daily_indicators(market_data, cache)
    logger.info("Indicators complete: %d symbols", len(results))


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.indicators --cache-only    → test HistoryCache load
#   python -m modules.indicators --daily         → compute all + write to Neon
#   python -m modules.indicators --read          → read today's rows from Neon
#   python -m modules.indicators NABIL HBL       → compute specific symbols
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [INDICATORS] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    args = sys.argv[1:]
    print("\n" + "=" * 70)
    print("  NEPSE AI — indicators.py  (RSI / EMA / MACD / BB / ATR / OBV)")
    print("=" * 70)

    # ── Cache-only test ──────────────────────────────────────────────────
    if "--cache-only" in args:
        print("\n[1/1] Loading history cache...")
        cache = HistoryCache()
        count = cache.load(periods=DEFAULT_LOAD_PERIODS)
        if count:
            print(f"  ✅ {count} symbols | {len(cache.dates)} trading days")
            print(f"  Date range: {cache.dates[0]} → {cache.dates[-1]}")
            nabil = cache.get_closes("NABIL")
            print(f"  NABIL closes (last 5): {nabil[-5:]}")
        else:
            print("  ❌ Cache load failed — is price_history table populated?")
        sys.exit(0)

    # ── Read today's indicators from Neon ────────────────────────────────
    if "--read" in args:
        print("\n[1/1] Reading today's indicators from Neon...")
        try:
            from sheets import read_today_indicators
            today = datetime.now(tz=NST).strftime("%Y-%m-%d")
            rows = read_today_indicators(today)
            if not rows:
                print(f"  No indicators for {today} — run --daily first")
            else:
                print(f"  {len(rows)} symbols found for {today}\n")
                print(f"  {'Symbol':<10} {'RSI':>6} {'Signal':<10} {'EMA_trend':<12} {'Tech':<6} {'Score':>5}")
                print("  " + "-" * 60)
                top = sorted(rows.values(), key=lambda r: int(r.get("tech_score","0") or 0), reverse=True)
                for r in top[:20]:
                    print(
                        f"  {r['symbol']:<10} "
                        f"{r.get('rsi_14','—'):>6} "
                        f"{r.get('rsi_signal','—'):<10} "
                        f"{r.get('ema_trend','—'):<12} "
                        f"{r.get('tech_signal','—'):<6} "
                        f"{r.get('tech_score','—'):>5}"
                    )
        except Exception as e:
            print(f"  ❌ {e}")
        sys.exit(0)

    # ── Load cache ───────────────────────────────────────────────────────
    print("\n[1/3] Loading history cache...")
    cache = HistoryCache()
    count = cache.load(periods=DEFAULT_LOAD_PERIODS)
    if count == 0:
        print("  ❌ Cache load failed"); sys.exit(1)
    print(f"  ✅ {count} symbols | {len(cache.dates)} trading days")

    # ── Fetch live prices ────────────────────────────────────────────────
    print("\n[2/3] Fetching live prices...")
    try:
        from modules.scraper import get_all_market_data, PriceRow
        market_data = get_all_market_data(write_breadth=False)
        if not market_data:
            print("  ⚠️  Market closed — using cache close prices as today's price")
            market_data = {}
            for sym, closes in cache.closes.items():
                if closes:
                    highs = cache.get_highs(sym)
                    lows  = cache.get_lows(sym)
                    market_data[sym] = PriceRow(
                        symbol=sym, ltp=closes[-1],
                        open_price=closes[-2] if len(closes) > 1 else closes[-1],
                        close=closes[-1],
                        high=highs[-1] if highs else closes[-1],
                        low=lows[-1]   if lows  else closes[-1],
                        prev_close=closes[-2] if len(closes) > 1 else closes[-1],
                        volume=10000,
                    )
        print(f"  ✅ {len(market_data)} symbols")
    except Exception as e:
        print(f"  ❌ Scraper failed: {e}"); sys.exit(1)

    # Filter to specific symbols if passed on CLI
    sym_args = [a for a in args if not a.startswith("--")]
    if sym_args:
        req = {a.upper() for a in sym_args}
        market_data = {k: v for k, v in market_data.items() if k in req}
        print(f"  Filtered to: {list(market_data.keys())}")

    # ── Compute ──────────────────────────────────────────────────────────
    write_db = "--daily" in args
    print(f"\n[3/3] Computing indicators {'+ writing to Neon' if write_db else '(dry run)'}...")

    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    results: dict[str, IndicatorResult] = {}

    for symbol, price_row in market_data.items():
        if not str(symbol).replace("-", "").replace("_", "").isalpha():
            continue
        try:
            results[symbol] = compute_indicators(symbol, price_row, cache, date=today)
        except Exception as exc:
            logger.warning("%s: compute failed — %s", symbol, exc)

    print(f"  ✅ {len(results)} symbols computed\n")

    if results:
        print(f"  {'Symbol':<10} {'RSI':>6} {'RSI_sig':<10} {'EMA':<12} {'MACD_x':<10} {'Tech':<12} {'Score':>5}")
        print("  " + "-" * 75)
        top = sorted(results.values(), key=lambda r: int(r.tech_score or 0), reverse=True)
        for r in top[:20]:
            print(
                f"  {r.symbol:<10} "
                f"{r.rsi_14 or '—':>6} "
                f"{r.rsi_signal or '—':<10} "
                f"{r.ema_trend or '—':<12} "
                f"{r.macd_cross or '—':<10} "
                f"{r.tech_signal or '—':<12} "
                f"{r.tech_score or '—':>5}"
            )

    if write_db:
        print(f"\n  Writing {len(results)} rows to Neon indicators table...")
        try:
            from sheets import write_indicators_batch
            rows = [r.to_dict() for r in results.values()]
            written = write_indicators_batch(rows)
            print(f"  ✅ {written} rows written")
        except Exception as e:
            print(f"  ❌ DB write failed: {e}")
    else:
        print("  Tip: run with --daily to write results to Neon")

    print("=" * 70 + "\n")