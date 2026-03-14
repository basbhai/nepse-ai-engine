"""
indicators.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2
Purpose : Compute all technical indicators for every symbol.

Run schedule — ONCE DAILY (not every 6 minutes):
    Triggered by morning_brief.yml at 10:30 AM NST (pre-open window).
    Results written to INDICATORS table in Neon PostgreSQL.
    The 6-minute trading loop (trading.yml) reads from that table —
    no heavy DB reads or math during the trading session.

Why once daily is correct:
    RSI, EMA, MACD, Bollinger, ATR are all defined on DAILY closes.
    Recomputing every 6 minutes from intraday ticks adds noise, not
    accuracy. Daily close RSI is the standard used by every serious
    technical analyst. Running once also avoids burning API quota.

What the 6-minute loop does instead (in filter_engine.py):
    Reads pre-computed indicators from indicators table (one DB read)
    Then applies 3 lightweight live checks:
        1. Volume ratio  — live vol vs 180-day average
        2. Price vs EMA  — is LTP currently above/below EMA20/50/200?
        3. Conf. score   — from ShareSansar (already in scraper.PriceRow)

Indicators computed:
    RSI-14          Relative Strength Index (14-period, Wilder's smoothing)
    EMA-20          Exponential Moving Average (20-period)
    EMA-50          Exponential Moving Average (50-period)
    EMA-200         Exponential Moving Average (200-period)
    MACD            MACD Line, Signal Line, Histogram (12/26/9)
    Bollinger Bands Upper, Middle (SMA-20), Lower, %B, Bandwidth
    ATR-14          Average True Range (14-period)
    OBV             On-Balance Volume + trend direction

Fixes applied (v2):
    1. EMA crossover detection — now detects the actual cross EVENT
       (today fast crosses above/below slow) instead of just position.
       Uses _ema_series() to compare yesterday vs today values.
    2. BB signal — now uses bb_pct_b properly:
       NEAR_LOWER (<0.05), NEAR_UPPER (>0.95), SQUEEZE, EXPANSION, NEUTRAL
    3. tech_score BB logic — uses bb_pct_b thresholds not raw price vs band
    4. DEFAULT_LOAD_PERIODS raised 220 → 260 for better EMA-200 warmup

Data sources:
    Historical OHLCV  ← GOOGLE_SHEET_SHARE_DATA (separate Sheets file)
                         One tab per date e.g. "2024-03-04" (485+ tabs)
                         Columns: S.No | Symbol | Conf. | Open | High |
                         Low | Close | VWAP | Vol | Prev.Close | ...

    Live price today  ← PriceRow from scraper.py (in memory, passed in)
                         Today's close appended to historical array before
                         computing — so indicators always reflect today.

Output:
    indicators table in Neon PostgreSQL.
    One row per symbol. Overwritten fresh each morning.
    Read by filter_engine.py during the trading session.

Usage (morning_brief.yml):
    from indicators import run_daily_indicators
    run_daily_indicators()   # full pipeline: load → compute → write to Neon

Usage (filter_engine.py — reading pre-computed results):
    from indicators import read_indicators_from_db
    indicators = read_indicators_from_db()
    nabil = indicators.get('NABIL')

─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
import math

logger = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

HIST_SHEET_ID_ENV = "GOOGLE_SHEET_SHARE_DATA"

# Periods required per indicator
RSI_PERIOD         = 14
EMA_SHORT          = 12    # MACD fast
EMA_LONG           = 26    # MACD slow
EMA_SIGNAL         = 9     # MACD signal
EMA_20_PERIOD      = 20
EMA_50_PERIOD      = 50
EMA_200_PERIOD     = 200
BB_PERIOD          = 20    # Bollinger middle SMA
ATR_PERIOD         = 14

# Minimum history needed for all indicators (EMA-200 is the binding constraint)
MIN_PERIODS_REQUIRED = 200
# FIX 4: Raised from 220 → 260 for proper EMA-200 warmup.
# EMA-200 needs 200 bars just to compute. Extra 60 bars allow Wilder's
# smoothing to converge before we start trusting the values.
DEFAULT_LOAD_PERIODS = 260

# Column names in the OmitNomis / ShareSansar tabs (exact header strings)
COL_SYMBOL = "Symbol"
COL_CLOSE  = "Close"
COL_OPEN   = "Open"
COL_HIGH   = "High"
COL_LOW    = "Low"
COL_VOLUME = "Vol"


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorResult:
    """
    All technical indicators for one symbol at one point in time.
    Passed to candle_detector.py and filter_engine.py.
    """
    symbol:         str

    # RSI
    rsi_14:         float = 0.0
    rsi_signal:     str   = "NEUTRAL"   # OVERSOLD / NEUTRAL / OVERBOUGHT

    # EMAs
    ema_20:         float = 0.0
    ema_50:         float = 0.0
    ema_200:        float = 0.0
    ema_trend:      str   = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
                                        # based on price vs EMA-200

    # EMA crossover signals — TRUE CROSS EVENT (not just position)
    # GOLDEN = fast crossed ABOVE slow TODAY
    # DEATH  = fast crossed BELOW slow TODAY
    # ABOVE  = fast has been above slow (no fresh cross)
    # BELOW  = fast has been below slow (no fresh cross)
    # NONE   = insufficient data
    ema_20_50_cross:  str = "NONE"
    ema_50_200_cross: str = "NONE"

    # MACD
    macd_line:      float = 0.0
    macd_signal:    float = 0.0
    macd_histogram: float = 0.0
    macd_cross:     str   = "NONE"      # BULLISH / BEARISH / NONE

    # Bollinger Bands
    bb_upper:       float = 0.0
    bb_middle:      float = 0.0         # SMA-20
    bb_lower:       float = 0.0
    bb_width:       float = 0.0         # (upper - lower) / middle * 100
    bb_pct_b:       float = 0.0         # where price sits in band (0.0–1.0)
    # FIX 2: bb_signal now uses bb_pct_b properly
    # NEAR_LOWER / NEAR_UPPER / SQUEEZE / EXPANSION / NEUTRAL
    bb_signal:      str   = "NEUTRAL"

    # ATR
    atr_14:         float = 0.0
    atr_pct:        float = 0.0         # ATR as % of price (volatility %)

    # OBV
    obv:            float = 0.0
    obv_trend:      str   = "NEUTRAL"   # RISING / FALLING / NEUTRAL

    # Price context
    ltp:            float = 0.0
    prev_close:     float = 0.0
    volume:         int   = 0
    history_days:   int   = 0

    # Overall technical signal (composite)
    tech_score:     int   = 0           # -10 to +10
    tech_signal:    str   = "NEUTRAL"   # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL

    timestamp: str = field(default_factory=lambda: datetime.now(
        tz=timezone(timedelta(hours=5, minutes=45))
    ).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY CACHE
# ══════════════════════════════════════════════════════════════════════════════

class HistoryCache:
    """
    Loads and holds historical OHLCV data from the OmitNomis Google Sheet.

    Structure after load():
        self.closes  = {symbol: [float, ...]}   oldest → newest
        self.highs   = {symbol: [float, ...]}
        self.lows    = {symbol: [float, ...]}
        self.volumes = {symbol: [float, ...]}
        self.dates   = [str, ...]               sorted date strings loaded

    Call load() once per job. All indicator math reads from these dicts.
    """

    def __init__(self):
        self.closes:  dict[str, list[float]] = {}
        self.highs:   dict[str, list[float]] = {}
        self.lows:    dict[str, list[float]] = {}
        self.volumes: dict[str, list[float]] = {}
        self.dates:   list[str] = []
        self._loaded  = False
        self._periods = 0

    # ── Sheets connection ──────────────────────────────────────────────────

    def _get_hist_sheet(self):
        """Open the historical data Google Sheet."""
        import gspread
        from google.oauth2.service_account import Credentials

        sheet_id = os.getenv(HIST_SHEET_ID_ENV)
        if not sheet_id:
            raise RuntimeError(
                f"Env var {HIST_SHEET_ID_ENV} not set — "
                "add GOOGLE_SHEET_SHARE_DATA to .env / GitHub Secrets"
            )

        service_account_path = os.getenv(
            "GOOGLE_SERVICE_ACCOUNT_PATH", "service_account.json"
        )
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        # Support JSON string in env (GitHub Actions secret style)
        sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT")
        if sa_json:
            import json, tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            tmp.write(sa_json)
            tmp.flush()
            service_account_path = tmp.name

        creds  = Credentials.from_service_account_file(service_account_path, scopes=scopes)
        client = gspread.authorize(creds)
        return client.open_by_key(sheet_id)

    # ── Tab discovery ──────────────────────────────────────────────────────

    def _get_date_tabs(self, sheet, periods: int) -> list[str]:
        """
        Return list of the most recent `periods` date tab names,
        sorted oldest → newest.
        """
        all_tabs = [ws.title for ws in sheet.worksheets()]

        date_tabs = []
        for title in all_tabs:
            try:
                datetime.strptime(title.strip(), "%Y-%m-%d")
                date_tabs.append(title.strip())
            except ValueError:
                pass  # skip non-date tabs

        date_tabs.sort()

        if not date_tabs:
            raise RuntimeError(
                "No date-named tabs found in historical sheet. "
                "Expected tabs named like '2024-03-04'."
            )

        selected = date_tabs[-periods:]
        logger.info(
            "History: found %d date tabs, loading last %d (%s → %s)",
            len(date_tabs), len(selected), selected[0], selected[-1],
        )
        return selected

    # ── Parse one day's tab ────────────────────────────────────────────────

    @staticmethod
    def _parse_tab(records: list[dict]) -> dict[str, dict]:
        """
        Parse one date tab's records into a per-symbol dict.
        Returns {symbol: {close, open, high, low, volume}}
        """
        day: dict[str, dict] = {}

        for row in records:
            symbol = str(row.get(COL_SYMBOL, "")).strip().upper()
            if not symbol:
                continue

            def _f(key: str) -> float:
                val = row.get(key, 0)
                try:
                    return float(str(val).replace(",", "").strip() or 0)
                except (ValueError, TypeError):
                    return 0.0

            close  = _f(COL_CLOSE)
            volume = _f(COL_VOLUME)

            if close <= 0:
                continue

            day[symbol] = {
                "close":  close,
                "open":   _f(COL_OPEN),
                "high":   _f(COL_HIGH),
                "low":    _f(COL_LOW),
                "volume": volume,
            }

        return day

    # ── Main load ──────────────────────────────────────────────────────────

    def load(self, periods: int = DEFAULT_LOAD_PERIODS) -> int:
        """
        Load the last `periods` trading days from the historical sheet.
        Builds per-symbol arrays in chronological order.

        Args:
            periods: Number of date tabs to load (default 260 for EMA-200 warmup)

        Returns:
            Number of symbols loaded.
        """
        logger.info("HistoryCache: loading last %d periods from Sheets...", periods)
        start = time.time()

        try:
            sheet = self._get_hist_sheet()
        except Exception as exc:
            logger.error("HistoryCache: cannot open historical sheet — %s", exc)
            return 0

        try:
            date_tabs = self._get_date_tabs(sheet, periods)
        except Exception as exc:
            logger.error("HistoryCache: tab discovery failed — %s", exc)
            return 0

        closes:  dict[str, list[float]] = {}
        highs:   dict[str, list[float]] = {}
        lows:    dict[str, list[float]] = {}
        volumes: dict[str, list[float]] = {}

        loaded_dates = []
        failed_tabs  = 0

        for i, tab_name in enumerate(date_tabs):
            try:
                ws      = sheet.worksheet(tab_name)
                records = ws.get_all_records()
                day     = self._parse_tab(records)

                for symbol, ohlcv in day.items():
                    closes.setdefault(symbol,  []).append(ohlcv["close"])
                    highs.setdefault(symbol,   []).append(ohlcv["high"])
                    lows.setdefault(symbol,    []).append(ohlcv["low"])
                    volumes.setdefault(symbol, []).append(ohlcv["volume"])

                loaded_dates.append(tab_name)

                if (i + 1) % 50 == 0:
                    logger.info(
                        "HistoryCache: loaded %d/%d tabs...", i + 1, len(date_tabs)
                    )
                    time.sleep(1)

            except Exception as exc:
                logger.warning("HistoryCache: failed to load tab %s — %s", tab_name, exc)
                failed_tabs += 1
                continue

        self.closes  = closes
        self.highs   = highs
        self.lows    = lows
        self.volumes = volumes
        self.dates   = loaded_dates
        self._loaded = True
        self._periods = len(loaded_dates)

        elapsed = round(time.time() - start, 2)
        logger.info(
            "HistoryCache: loaded %d symbols across %d days in %.1fs (%d tabs failed)",
            len(closes), len(loaded_dates), elapsed, failed_tabs,
        )
        return len(closes)

    # ── Accessors ──────────────────────────────────────────────────────────

    def get_closes(self, symbol: str) -> list[float]:
        return self.closes.get(symbol.upper(), [])

    def get_highs(self, symbol: str) -> list[float]:
        return self.highs.get(symbol.upper(), [])

    def get_lows(self, symbol: str) -> list[float]:
        return self.lows.get(symbol.upper(), [])

    def get_volumes(self, symbol: str) -> list[float]:
        return self.volumes.get(symbol.upper(), [])

    def symbol_count(self) -> int:
        return len(self.closes)

    def is_loaded(self) -> bool:
        return self._loaded

    def coverage(self, symbol: str) -> int:
        """How many days of history do we have for this symbol."""
        return len(self.closes.get(symbol.upper(), []))


# ══════════════════════════════════════════════════════════════════════════════
# MATH PRIMITIVES
# All functions operate on plain Python lists — no numpy/pandas dependency.
# GitHub Actions runners may not have them pre-installed and we want to keep
# the requirements.txt minimal.
# ══════════════════════════════════════════════════════════════════════════════

def _sma(prices: list[float], period: int) -> float:
    """Simple Moving Average of the last `period` values."""
    if len(prices) < period:
        return 0.0
    return sum(prices[-period:]) / period


def _ema_series(prices: list[float], period: int) -> list[float]:
    """
    Compute full EMA series for a price list.
    Uses standard smoothing factor: k = 2 / (period + 1)
    Returns list same length as prices (first period-1 values are 0.0).
    """
    if len(prices) < period:
        return [0.0] * len(prices)

    k      = 2.0 / (period + 1)
    result = [0.0] * len(prices)

    # Seed with SMA of first `period` values
    result[period - 1] = sum(prices[:period]) / period

    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)

    return result


def _ema_last(prices: list[float], period: int) -> float:
    """Return only the last (current) EMA value."""
    series = _ema_series(prices, period)
    return series[-1] if series else 0.0


def _rsi(closes: list[float], period: int = 14) -> float:
    """
    RSI using Wilder's smoothing (standard RSI formula).
    Uses the full closes list so Wilder's smoothing loop actually runs
    over all available history — not just the last period+1 values.
    Requires at least period+1 values.
    """
    if len(closes) < period + 1:
        return 0.0

    gains  = []
    losses = []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    # Seed with simple average of first `period` differences
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing over ALL remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _macd(closes: list[float]) -> tuple[float, float, float]:
    """
    MACD Line, Signal Line, Histogram.
    Standard 12/26/9 parameters.
    Returns (macd_line, signal_line, histogram).
    """
    if len(closes) < EMA_LONG + EMA_SIGNAL:
        return 0.0, 0.0, 0.0

    ema12_series = _ema_series(closes, EMA_SHORT)
    ema26_series = _ema_series(closes, EMA_LONG)

    macd_series = []
    for i in range(len(closes)):
        if ema12_series[i] != 0.0 and ema26_series[i] != 0.0:
            macd_series.append(ema12_series[i] - ema26_series[i])
        else:
            macd_series.append(0.0)

    non_zero_start = next(
        (i for i, v in enumerate(macd_series) if v != 0.0), None
    )
    if non_zero_start is None:
        return 0.0, 0.0, 0.0

    macd_clean = macd_series[non_zero_start:]
    if len(macd_clean) < EMA_SIGNAL:
        return 0.0, 0.0, 0.0

    signal_series = _ema_series(macd_clean, EMA_SIGNAL)

    macd_line   = round(macd_series[-1], 4)
    signal_line = round(signal_series[-1], 4)
    histogram   = round(macd_line - signal_line, 4)

    return macd_line, signal_line, histogram


def _bollinger(closes: list[float], period: int = BB_PERIOD) -> tuple[float, float, float]:
    """
    Bollinger Bands: (upper, middle, lower).
    Middle = SMA-20, bands = ±2 standard deviations.
    """
    if len(closes) < period:
        return 0.0, 0.0, 0.0

    subset   = closes[-period:]
    middle   = sum(subset) / period
    variance = sum((p - middle) ** 2 for p in subset) / period
    std_dev  = math.sqrt(variance)

    upper = round(middle + 2 * std_dev, 2)
    lower = round(middle - 2 * std_dev, 2)
    return upper, round(middle, 2), lower


def _atr(highs: list[float], lows: list[float],
         closes: list[float], period: int = ATR_PERIOD) -> float:
    """
    Average True Range using Wilder's smoothing.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        return 0.0

    n      = min(len(highs), len(lows), len(closes))
    highs  = highs[-n:]
    lows   = lows[-n:]
    closes = closes[-n:]

    tr_values = []
    for i in range(1, n):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hpc, lpc))

    if len(tr_values) < period:
        return 0.0

    atr = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr = (atr * (period - 1) + tr) / period

    return round(atr, 2)


def _obv(closes: list[float], volumes: list[float]) -> tuple[float, str]:
    """
    On-Balance Volume and trend direction.
    OBV trend = slope of last 5 OBV values (RISING / FALLING / NEUTRAL).
    Returns (current_obv, trend_signal).
    """
    n = min(len(closes), len(volumes))
    if n < 2:
        return 0.0, "NEUTRAL"

    closes  = closes[-n:]
    volumes = volumes[-n:]

    obv_series = [0.0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv_series.append(obv_series[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv_series.append(obv_series[-1] - volumes[i])
        else:
            obv_series.append(obv_series[-1])

    current_obv = obv_series[-1]

    if len(obv_series) >= 10:
        recent   = sum(obv_series[-5:]) / 5
        previous = sum(obv_series[-10:-5]) / 5
        if recent > previous * 1.001:
            trend = "RISING"
        elif recent < previous * 0.999:
            trend = "FALLING"
        else:
            trend = "NEUTRAL"
    else:
        trend = "NEUTRAL"

    return round(current_obv, 0), trend


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def _rsi_signal(rsi: float) -> str:
    if rsi <= 30:
        return "OVERSOLD"
    if rsi >= 70:
        return "OVERBOUGHT"
    return "NEUTRAL"


def _ema_trend_signal(price: float, ema_200: float) -> str:
    if ema_200 <= 0:
        return "NEUTRAL"
    if price > ema_200 * 1.002:
        return "BULLISH"
    if price < ema_200 * 0.998:
        return "BEARISH"
    return "NEUTRAL"


def _ema_cross_signal(fast_series: list[float], slow_series: list[float]) -> str:
    """
    FIX 1: Detect the actual EMA crossover EVENT — not just current position.

    Compares yesterday's relationship vs today's relationship between
    fast and slow EMA. Returns:
        GOLDEN  — fast crossed ABOVE slow today (bullish event)
        DEATH   — fast crossed BELOW slow today (bearish event)
        ABOVE   — fast is above slow, no fresh cross (existing bull position)
        BELOW   — fast is below slow, no fresh cross (existing bear position)
        NONE    — insufficient data

    Why this matters:
        A golden cross that happened 6 months ago should NOT score the same
        as one that happened today. filter_engine.py gives extra weight to
        fresh crosses (GOLDEN/DEATH) vs existing positions (ABOVE/BELOW).
    """
    # Need at least 2 values of each to detect a cross
    if len(fast_series) < 2 or len(slow_series) < 2:
        return "NONE"

    fast_today = fast_series[-1]
    fast_prev  = fast_series[-2]
    slow_today = slow_series[-1]
    slow_prev  = slow_series[-2]

    # Skip if either series is still in its zero-padding zone
    if fast_today == 0.0 or slow_today == 0.0:
        return "NONE"
    if fast_prev == 0.0 or slow_prev == 0.0:
        return "NONE"

    # Detect cross: yesterday fast was below slow, today fast is above slow
    if fast_prev <= slow_prev and fast_today > slow_today:
        return "GOLDEN"

    # Detect cross: yesterday fast was above slow, today fast is below slow
    if fast_prev >= slow_prev and fast_today < slow_today:
        return "DEATH"

    # No fresh cross — report current position
    if fast_today > slow_today:
        return "ABOVE"
    if fast_today < slow_today:
        return "BELOW"

    return "NONE"


def _macd_cross_signal(macd_line: float, signal_line: float) -> str:
    if macd_line == 0.0 and signal_line == 0.0:
        return "NONE"
    if macd_line > signal_line:
        return "BULLISH"
    if macd_line < signal_line:
        return "BEARISH"
    return "NONE"


def _bb_signal(bb_width: float, bb_pct_b: float) -> str:
    """
    FIX 2: Bollinger Band signal now uses bb_pct_b properly.

    bb_pct_b = (price - lower) / (upper - lower)
        0.0  = price AT lower band
        0.5  = price at middle
        1.0  = price AT upper band

    Signal priority (most actionable first):
        NEAR_LOWER  — price hugging lower band (bb_pct_b < 0.05) → mean reversion long setup
        NEAR_UPPER  — price hugging upper band (bb_pct_b > 0.95) → mean reversion short setup
        SQUEEZE     — very tight bands (bb_width < 5%) → breakout incoming (direction unknown)
        EXPANSION   — very wide bands (bb_width > 20%) → high volatility, trend in motion
        NEUTRAL     — normal conditions
    """
    if bb_pct_b < 0.05:
        return "NEAR_LOWER"
    if bb_pct_b > 0.95:
        return "NEAR_UPPER"
    if bb_width < 5.0:
        return "SQUEEZE"
    if bb_width > 20.0:
        return "EXPANSION"
    return "NEUTRAL"


def _tech_score(result: "IndicatorResult") -> tuple[int, str]:
    """
    Composite technical score from -10 to +10.
    Each indicator contributes ±1 or ±2 points.

    Scoring:
        RSI oversold               +2   (entry signal)
        RSI overbought             -2
        Price > EMA200             +2   (bull trend)
        Price < EMA200             -2
        EMA20/50 GOLDEN cross      +2   (fresh cross — strong signal)
        EMA20/50 ABOVE (position)  +1   (existing bull — weaker)
        EMA20/50 DEATH cross       -2   (fresh cross — strong signal)
        EMA20/50 BELOW (position)  -1   (existing bear — weaker)
        MACD bullish               +1
        MACD bearish               -1
        OBV rising                 +1
        OBV falling                -1
        BB NEAR_LOWER              +1   (FIX 3: mean reversion long setup)
        BB NEAR_UPPER              -1   (FIX 3: mean reversion short setup)
    """
    score = 0

    # RSI
    if result.rsi_signal == "OVERSOLD":
        score += 2
    elif result.rsi_signal == "OVERBOUGHT":
        score -= 2

    # EMA trend (price vs EMA-200)
    if result.ema_trend == "BULLISH":
        score += 2
    elif result.ema_trend == "BEARISH":
        score -= 2

    # FIX 3: EMA cross — differentiate fresh cross vs existing position
    if result.ema_20_50_cross == "GOLDEN":
        score += 2   # fresh bullish cross → strong
    elif result.ema_20_50_cross == "ABOVE":
        score += 1   # already above → weaker ongoing bull
    elif result.ema_20_50_cross == "DEATH":
        score -= 2   # fresh bearish cross → strong
    elif result.ema_20_50_cross == "BELOW":
        score -= 1   # already below → weaker ongoing bear

    # MACD
    if result.macd_cross == "BULLISH":
        score += 1
    elif result.macd_cross == "BEARISH":
        score -= 1

    # OBV
    if result.obv_trend == "RISING":
        score += 1
    elif result.obv_trend == "FALLING":
        score -= 1

    # FIX 3: Bollinger — use bb_signal (which now uses bb_pct_b)
    if result.bb_signal == "NEAR_LOWER":
        score += 1
    elif result.bb_signal == "NEAR_UPPER":
        score -= 1

    # Clamp to -10 / +10
    score = max(-10, min(10, score))

    if score >= 5:
        signal = "STRONG_BUY"
    elif score >= 2:
        signal = "BUY"
    elif score <= -5:
        signal = "STRONG_SELL"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return score, signal


# ══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(
    symbol:    str,
    price_row,              # scraper.PriceRow
    cache:     HistoryCache,
) -> Optional[IndicatorResult]:
    """
    Compute all technical indicators for one symbol.

    Args:
        symbol:    Stock symbol e.g. "NABIL"
        price_row: PriceRow from scraper.py (today's live data)
        cache:     HistoryCache loaded at job start

    Returns:
        IndicatorResult with all indicators populated.
        Returns None if insufficient history (< RSI_PERIOD + 1 days).
    """
    sym = symbol.upper()

    # Pull historical arrays
    hist_closes  = cache.get_closes(sym)
    hist_highs   = cache.get_highs(sym)
    hist_lows    = cache.get_lows(sym)
    hist_volumes = cache.get_volumes(sym)

    # Append today's live values
    today_close  = price_row.ltp if price_row.ltp > 0 else price_row.close
    today_high   = price_row.high  if price_row.high  > 0 else today_close
    today_low    = price_row.low   if price_row.low   > 0 else today_close
    today_volume = float(price_row.volume)

    closes  = hist_closes  + [today_close]
    highs   = hist_highs   + [today_high]
    lows    = hist_lows    + [today_low]
    volumes = hist_volumes + [today_volume]

    n = len(closes)

    if n < RSI_PERIOD + 1:
        logger.debug(
            "%s: insufficient history (%d days, need %d) — skipping",
            sym, n, RSI_PERIOD + 1,
        )
        return None

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi   = _rsi(closes, RSI_PERIOD)
    rsi_s = _rsi_signal(rsi)

    # ── EMAs ───────────────────────────────────────────────────────────────
    ema20_series  = _ema_series(closes, EMA_20_PERIOD)  if n >= EMA_20_PERIOD  else [0.0] * n
    ema50_series  = _ema_series(closes, EMA_50_PERIOD)  if n >= EMA_50_PERIOD  else [0.0] * n
    ema200_series = _ema_series(closes, EMA_200_PERIOD) if n >= EMA_200_PERIOD else [0.0] * n

    ema20  = ema20_series[-1]
    ema50  = ema50_series[-1]
    ema200 = ema200_series[-1]

    ema_trend = _ema_trend_signal(today_close, ema200)

    # FIX 1: Pass full series so cross detector can compare today vs yesterday
    cross_20_50  = _ema_cross_signal(ema20_series,  ema50_series)
    cross_50_200 = _ema_cross_signal(ema50_series,  ema200_series)

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(closes)
    macd_cross = _macd_cross_signal(macd_line, macd_sig)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(closes, BB_PERIOD)
    bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0.0
    bb_pct_b = round(
        (today_close - bb_lower) / (bb_upper - bb_lower), 4
    ) if (bb_upper - bb_lower) > 0 else 0.5
    # FIX 2: _bb_signal now uses bb_pct_b properly
    bb_sig = _bb_signal(bb_width, bb_pct_b)

    # ── ATR ────────────────────────────────────────────────────────────────
    atr     = _atr(highs, lows, closes, ATR_PERIOD)
    atr_pct = round(atr / today_close * 100, 2) if today_close > 0 else 0.0

    # ── OBV ────────────────────────────────────────────────────────────────
    obv_val, obv_trend = _obv(closes, volumes)

    # ── Composite score ────────────────────────────────────────────────────
    result = IndicatorResult(
        symbol           = sym,
        rsi_14           = rsi,
        rsi_signal       = rsi_s,
        ema_20           = round(ema20, 2),
        ema_50           = round(ema50, 2),
        ema_200          = round(ema200, 2),
        ema_trend        = ema_trend,
        ema_20_50_cross  = cross_20_50,
        ema_50_200_cross = cross_50_200,
        macd_line        = macd_line,
        macd_signal      = macd_sig,
        macd_histogram   = macd_hist,
        macd_cross       = macd_cross,
        bb_upper         = bb_upper,
        bb_middle        = bb_mid,
        bb_lower         = bb_lower,
        bb_width         = bb_width,
        bb_pct_b         = bb_pct_b,
        bb_signal        = bb_sig,
        atr_14           = atr,
        atr_pct          = atr_pct,
        obv              = obv_val,
        obv_trend        = obv_trend,
        ltp              = today_close,
        prev_close       = price_row.prev_close,
        volume           = price_row.volume,
        history_days     = n,
    )

    score, signal      = _tech_score(result)
    result.tech_score  = score
    result.tech_signal = signal

    return result


def compute_all_indicators(
    market_data: dict,       # dict[symbol, PriceRow] from scraper.py
    cache:       HistoryCache,
    min_history: int = RSI_PERIOD + 1,
) -> dict[str, IndicatorResult]:
    """
    Compute indicators for ALL symbols in market_data.
    Returns dict[symbol, IndicatorResult].
    Symbols with insufficient history are silently skipped.
    """
    results: dict[str, IndicatorResult] = {}
    skipped = 0

    for symbol, price_row in market_data.items():
        result = compute_indicators(symbol, price_row, cache)
        if result and result.history_days >= min_history:
            results[symbol] = result
        else:
            skipped += 1

    logger.info(
        "compute_all_indicators: %d computed, %d skipped (insufficient history)",
        len(results), skipped,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# NEON DB I/O — write once daily, read during trading session
# ══════════════════════════════════════════════════════════════════════════════

def _result_to_row(r: IndicatorResult) -> dict:
    """Convert IndicatorResult to a flat dict for Neon DB write."""
    return {
        "symbol":           r.symbol,
        "date":             datetime.now(tz=NST).strftime("%Y-%m-%d"),
        "ltp":              str(r.ltp),
        "prev_close":       str(r.prev_close),
        "volume":           str(r.volume),
        "history_days":     str(r.history_days),
        "rsi_14":           str(r.rsi_14),
        "rsi_signal":       r.rsi_signal,
        "ema_20":           str(r.ema_20),
        "ema_50":           str(r.ema_50),
        "ema_200":          str(r.ema_200),
        "ema_trend":        r.ema_trend,
        "ema_20_50_cross":  r.ema_20_50_cross,
        "ema_50_200_cross": r.ema_50_200_cross,
        "macd_line":        str(r.macd_line),
        "macd_signal":      str(r.macd_signal),
        "macd_histogram":   str(r.macd_histogram),
        "macd_cross":       r.macd_cross,
        "bb_upper":         str(r.bb_upper),
        "bb_middle":        str(r.bb_middle),
        "bb_lower":         str(r.bb_lower),
        "bb_width":         str(r.bb_width),
        "bb_pct_b":         str(r.bb_pct_b),
        "bb_signal":        r.bb_signal,
        "atr_14":           str(r.atr_14),
        "atr_pct":          str(r.atr_pct),
        "obv":              str(r.obv),
        "obv_trend":        r.obv_trend,
        "tech_score":       str(r.tech_score),
        "tech_signal":      r.tech_signal,
        "timestamp":        r.timestamp,
    }


def write_indicators_to_db(results: dict[str, IndicatorResult]) -> bool:
    """
    Write all indicator results to Neon indicators table.
    Uses upsert on symbol — safe to run multiple times.
    Returns True on success.
    """
    try:
        from db import write_indicators_batch
        rows = [_result_to_row(r) for r in results.values()]
        write_indicators_batch(rows)
        logger.info("write_indicators_to_db: wrote %d rows to Neon", len(rows))
        return True
    except Exception as exc:
        logger.error("write_indicators_to_db failed: %s", exc)
        return False


def read_indicators_from_db() -> dict[str, IndicatorResult]:
    """
    Read today's pre-computed indicators from Neon indicators table.
    Called by filter_engine.py during the 6-minute trading loop.

    Returns dict[symbol, IndicatorResult].
    Returns empty dict if table is empty or stale.
    """
    try:
        from db import read_today_indicators

        rows = read_today_indicators()
        if not rows:
            logger.warning("read_indicators_from_db: indicators table is empty")
            return {}

        results: dict[str, IndicatorResult] = {}

        for symbol, row in rows.items():
            def _f(key: str, default: float = 0.0) -> float:
                try:
                    return float(str(row.get(key, default)).replace(",", "") or default)
                except (ValueError, TypeError):
                    return default

            def _i(key: str, default: int = 0) -> int:
                try:
                    return int(float(str(row.get(key, default)) or default))
                except (ValueError, TypeError):
                    return default

            def _s(key: str, default: str = "") -> str:
                return str(row.get(key, default)).strip()

            result = IndicatorResult(
                symbol           = symbol,
                ltp              = _f("ltp"),
                prev_close       = _f("prev_close"),
                volume           = _i("volume"),
                history_days     = _i("history_days"),
                rsi_14           = _f("rsi_14"),
                rsi_signal       = _s("rsi_signal", "NEUTRAL"),
                ema_20           = _f("ema_20"),
                ema_50           = _f("ema_50"),
                ema_200          = _f("ema_200"),
                ema_trend        = _s("ema_trend", "NEUTRAL"),
                ema_20_50_cross  = _s("ema_20_50_cross", "NONE"),
                ema_50_200_cross = _s("ema_50_200_cross", "NONE"),
                macd_line        = _f("macd_line"),
                macd_signal      = _f("macd_signal"),
                macd_histogram   = _f("macd_histogram"),
                macd_cross       = _s("macd_cross", "NONE"),
                bb_upper         = _f("bb_upper"),
                bb_middle        = _f("bb_middle"),
                bb_lower         = _f("bb_lower"),
                bb_width         = _f("bb_width"),
                bb_pct_b         = _f("bb_pct_b"),
                bb_signal        = _s("bb_signal", "NEUTRAL"),
                atr_14           = _f("atr_14"),
                atr_pct          = _f("atr_pct"),
                obv              = _f("obv"),
                obv_trend        = _s("obv_trend", "NEUTRAL"),
                tech_score       = _i("tech_score"),
                tech_signal      = _s("tech_signal", "NEUTRAL"),
                timestamp        = _s("timestamp"),
            )
            results[symbol] = result

        logger.info(
            "read_indicators_from_db: loaded %d symbols from Neon", len(results)
        )
        return results

    except Exception as exc:
        logger.error("read_indicators_from_db failed: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# DAILY PIPELINE — called by morning_brief.yml at 10:30 AM NST
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_indicators() -> dict[str, IndicatorResult]:
    """
    Full daily pipeline:
        1. Load 260 days history from GOOGLE_SHEET_SHARE_DATA
        2. Fetch today's live prices from TMS + ShareSansar
        3. Compute all indicators
        4. Write results to Neon indicators table
        5. Return results dict

    Called once per day by morning_brief.yml before market open.
    Returns empty dict on critical failure.
    """
    logger.info("run_daily_indicators: starting daily indicator computation...")
    start = time.time()

    # Step 1: Load history
    cache = HistoryCache()
    symbol_count = cache.load(periods=DEFAULT_LOAD_PERIODS)
    if symbol_count == 0:
        logger.error("run_daily_indicators: history load failed — aborting")
        return {}

    # Step 2: Live prices
    try:
        from scraper import get_all_market_data
        market_data = get_all_market_data(write_breadth=True)
    except Exception as exc:
        logger.error("run_daily_indicators: scraper failed — %s", exc)
        return {}

    if not market_data:
        logger.warning(
            "run_daily_indicators: no live prices (market not open yet?) "
            "— computing from last historical close instead"
        )
        try:
            from scraper import PriceRow
            market_data = {}
            for sym, closes in cache.closes.items():
                if closes:
                    last  = closes[-1]
                    highs = cache.get_highs(sym)
                    lows  = cache.get_lows(sym)
                    market_data[sym] = PriceRow(
                        symbol     = sym,
                        ltp        = last,
                        close      = last,
                        high       = highs[-1] if highs else last,
                        low        = lows[-1]  if lows  else last,
                        prev_close = closes[-2] if len(closes) > 1 else last,
                        volume     = 0,
                    )
        except Exception as exc:
            logger.error("run_daily_indicators: fallback price build failed — %s", exc)
            return {}

    # Step 3: Compute
    results = compute_all_indicators(market_data, cache)
    if not results:
        logger.error("run_daily_indicators: no indicators computed — aborting")
        return {}

    # Step 4: Write to Neon
    ok = write_indicators_to_db(results)
    if not ok:
        logger.warning(
            "run_daily_indicators: Neon write failed — "
            "results still returned in memory for this session"
        )

    elapsed = round(time.time() - start, 2)
    logger.info(
        "run_daily_indicators: complete — %d symbols in %.1fs | Neon write: %s",
        len(results), elapsed, "✅" if ok else "❌",
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python indicators.py --daily          → full daily pipeline (write to Neon)
#   python indicators.py --read           → read back from Neon indicators table
#   python indicators.py --cache-only     → test history loading only
#   python indicators.py NABIL HBL        → compute specific symbols, no write
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from collections import Counter

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [INDICATORS] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]

    print("\n" + "=" * 70)
    print("  NEPSE AI — indicators.py  (v2 — fixed EMA cross + BB signal)")
    print("=" * 70)

    # ── --daily: full pipeline ─────────────────────────────────────────────
    if "--daily" in args:
        print("\n  Running full daily pipeline → will write to Neon indicators table\n")
        results = run_daily_indicators()
        if not results:
            print("  ❌ Daily run failed")
            sys.exit(1)
        print(f"\n  ✅ {len(results)} symbols computed and written to Neon")
        dist = Counter(r.tech_signal for r in results.values())
        print("\n  Signal distribution:")
        for sig, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {sig:<15} {count}")
        sys.exit(0)

    # ── --read: verify what's in Neon ─────────────────────────────────────
    if "--read" in args:
        print("\n  Reading indicators from Neon...\n")
        results = read_indicators_from_db()
        if not results:
            print("  ❌ Nothing found — run --daily first")
            sys.exit(1)
        print(f"  ✅ {len(results)} symbols loaded\n")
        print(f"  {'Symbol':<10} {'RSI':>6} {'EMA_Trend':<10} {'20/50 Cross':<14} "
              f"{'MACD':>8} {'BB_Sig':<12} {'Score':>6} {'Signal'}")
        print("  " + "-" * 80)
        top = sorted(results.values(), key=lambda r: r.tech_score, reverse=True)[:15]
        for r in top:
            print(f"  {r.symbol:<10} {r.rsi_14:>6.1f} {r.ema_trend:<10} "
                  f"{r.ema_20_50_cross:<14} {r.macd_cross:>8} "
                  f"{r.bb_signal:<12} {r.tech_score:>+6}  {r.tech_signal}")
        sys.exit(0)

    # ── --cache-only: test history loading ────────────────────────────────
    if "--cache-only" in args:
        print("\n  Loading history cache only...\n")
        cache = HistoryCache()
        count = cache.load(periods=DEFAULT_LOAD_PERIODS)
        if count == 0:
            print("  ❌ Cache load failed — check GOOGLE_SHEET_SHARE_DATA env var")
            sys.exit(1)
        print(f"  ✅ {count} symbols | {len(cache.dates)} days "
              f"({cache.dates[0]} → {cache.dates[-1]})\n")
        print("  Sample coverage:")
        samples = ["NABIL", "HBL", "HIDCL", "NLIC", "UPPER", "ADBL", "NICA"]
        for s in samples:
            cov    = cache.coverage(s)
            closes = cache.get_closes(s)
            last   = closes[-1] if closes else 0
            print(f"    {s:<10} {cov:>4} days  last_close={last:.2f}")
        sys.exit(0)

    # ── Symbol / test mode: compute without writing ────────────────────────
    print("\n[1/3] Loading history cache...")
    cache = HistoryCache()
    count = cache.load(periods=DEFAULT_LOAD_PERIODS)
    if count == 0:
        print("  ❌ Cache load failed")
        sys.exit(1)
    print(f"  ✅ {count} symbols | {len(cache.dates)} days")

    print("\n[2/3] Fetching live prices...")
    try:
        from scraper import get_all_market_data, PriceRow
        market_data = get_all_market_data(write_breadth=False)
        if not market_data:
            print("  ⚠️  Market closed — using last cache close for testing")
            market_data = {}
            for sym, closes in cache.closes.items():
                if closes:
                    market_data[sym] = PriceRow(
                        symbol=sym, ltp=closes[-1], close=closes[-1],
                        high=closes[-1]*1.01, low=closes[-1]*0.99,
                        prev_close=closes[-2] if len(closes) > 1 else closes[-1],
                        volume=10000,
                    )
    except Exception as e:
        print(f"  ❌ Scraper failed: {e}")
        sys.exit(1)

    if args and not args[0].startswith("--"):
        requested   = {a.upper() for a in args}
        market_data = {k: v for k, v in market_data.items() if k in requested}
        print(f"  Symbols: {list(market_data.keys())}")
    else:
        print(f"  ✅ {len(market_data)} symbols (use --daily to write to Neon)")

    print(f"\n[3/3] Computing indicators...")
    results = compute_all_indicators(market_data, cache)
    print(f"  ✅ {len(results)} computed\n")

    if not results:
        print("  No results")
        sys.exit(0)

    sorted_results = sorted(results.values(), key=lambda r: r.tech_score, reverse=True)
    print(f"  {'Symbol':<10} {'LTP':>8} {'RSI':>6} {'20/50 Cross':<14} "
          f"{'BB_Sig':<12} {'OBV':>8} {'Score':>6} {'Signal'}")
    print("  " + "-" * 85)
    for r in sorted_results[:20]:
        print(
            f"  {r.symbol:<10} {r.ltp:>8.2f} {r.rsi_14:>6.1f} "
            f"{r.ema_20_50_cross:<14} {r.bb_signal:<12} "
            f"{r.obv_trend:>8} {r.tech_score:>+6}  {r.tech_signal}"
        )

    dist = Counter(r.tech_signal for r in results.values())
    print(f"\n  Signal distribution:")
    for sig, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {sig:<15} {count}")

    # Show cross breakdown — useful to verify fix 1 is working
    cross_dist = Counter(r.ema_20_50_cross for r in results.values())
    print(f"\n  EMA 20/50 cross breakdown (should see ABOVE/BELOW/GOLDEN/DEATH):")
    for val, count in sorted(cross_dist.items(), key=lambda x: -x[1]):
        print(f"    {val:<15} {count}")

    print(f"\n  💡 Run with --daily to compute all + write to Neon")
    print("=" * 70 + "\n")