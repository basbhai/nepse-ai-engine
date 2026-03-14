"""
indicators.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2
Purpose : Compute all technical indicators for every symbol.

Run schedule — ONCE DAILY (not every 6 minutes):
    Triggered by morning_brief.yml at 10:30 AM NST (pre-open window).
    Results written to INDICATORS tab in NEPSE_CORE_ENGINE sheet.
    The 6-minute trading loop (trading.yml) reads from that tab —
    no heavy Sheets reads or math during the trading session.

Why once daily is correct:
    RSI, EMA, MACD, Bollinger, ATR are all defined on DAILY closes.
    Recomputing every 6 minutes from intraday ticks adds noise, not
    accuracy. Daily close RSI is the standard used by every serious
    technical analyst. Running once also avoids burning ~8,000 Sheets
    API calls/day (200 tabs × 40 runs).

What the 6-minute loop does instead (in filter_engine.py):
    Reads pre-computed indicators from INDICATORS tab (one Sheets read)
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

Data sources:
    Historical OHLCV  ← GOOGLE_SHEET_SHARE_DATA (separate Sheets file)
                         One tab per date e.g. "2024-03-04" (485+ tabs)
                         Columns: S.No | Symbol | Conf. | Open | High |
                         Low | Close | VWAP | Vol | Prev.Close | ...

    Live price today  ← PriceRow from scraper.py (in memory, passed in)
                         Today's close appended to historical array before
                         computing — so indicators always reflect today.

Output:
    INDICATORS tab in NEPSE_CORE_ENGINE sheet.
    One row per symbol. Overwritten fresh each morning.
    Read by filter_engine.py during the trading session.

Usage (morning_brief.yml):
    from indicators import run_daily_indicators
    run_daily_indicators()   # full pipeline: load → compute → write to Sheets

Usage (filter_engine.py — reading pre-computed results):
    from indicators import read_indicators_from_sheets
    indicators = read_indicators_from_sheets()
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

# INDICATORS tab columns — written once daily to NEPSE_CORE_ENGINE
# Must match TABS["indicators"] in sheets.py (added below)
INDICATORS_TAB_COLUMNS = [
    "Symbol",
    "Date",
    "LTP",
    "Prev_Close",
    "Volume",
    "History_Days",
    # RSI
    "RSI_14",
    "RSI_Signal",
    # EMAs
    "EMA_20",
    "EMA_50",
    "EMA_200",
    "EMA_Trend",
    "EMA_20_50_Cross",
    "EMA_50_200_Cross",
    # MACD
    "MACD_Line",
    "MACD_Signal",
    "MACD_Histogram",
    "MACD_Cross",
    # Bollinger
    "BB_Upper",
    "BB_Middle",
    "BB_Lower",
    "BB_Width",
    "BB_Pct_B",
    "BB_Signal",
    # ATR
    "ATR_14",
    "ATR_Pct",
    # OBV
    "OBV",
    "OBV_Trend",
    # Composite
    "Tech_Score",
    "Tech_Signal",
    "Timestamp",
]

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
# Load a bit extra so EMA-200 is properly "warmed up"
DEFAULT_LOAD_PERIODS = 220

# Column names in the OmitNomis / ShareSansar tabs (exact header strings)
COL_SYMBOL     = "Symbol"
COL_CLOSE      = "Close"
COL_OPEN       = "Open"
COL_HIGH       = "High"
COL_LOW        = "Low"
COL_VOLUME     = "Vol"


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

    # EMA crossover signals
    ema_20_50_cross:  str = "NONE"      # GOLDEN / DEATH / NONE
    ema_50_200_cross: str = "NONE"      # GOLDEN / DEATH / NONE

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
    bb_pct_b:       float = 0.0         # where price sits in band (0-1)
    bb_signal:      str   = "NEUTRAL"   # SQUEEZE / EXPANSION / NEUTRAL

    # ATR
    atr_14:         float = 0.0
    atr_pct:        float = 0.0         # ATR as % of price (volatility %)

    # OBV
    obv:            float = 0.0
    obv_trend:      str   = "NEUTRAL"   # RISING / FALLING / NEUTRAL
                                        # based on OBV slope over last 5 periods

    # Price context
    ltp:            float = 0.0
    prev_close:     float = 0.0
    volume:         int   = 0
    history_days:   int   = 0           # how many days of history were available

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

        Tab names are expected to be date strings like "2024-03-04".
        Non-date tabs (SETTINGS, SCHEMA etc.) are silently ignored.
        """
        all_tabs = [ws.title for ws in sheet.worksheets()]

        date_tabs = []
        for title in all_tabs:
            try:
                datetime.strptime(title.strip(), "%Y-%m-%d")
                date_tabs.append(title.strip())
            except ValueError:
                pass  # skip non-date tabs

        date_tabs.sort()  # chronological order

        if not date_tabs:
            raise RuntimeError(
                "No date-named tabs found in historical sheet. "
                "Expected tabs named like '2024-03-04'."
            )

        # Take the most recent `periods` tabs
        selected = date_tabs[-periods:]
        logger.info(
            "History: found %d date tabs, loading last %d (%s → %s)",
            len(date_tabs), len(selected),
            selected[0], selected[-1],
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
                continue  # skip rows with no closing price

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
            periods: Number of date tabs to load (default 220 for EMA-200 warmup)

        Returns:
            Number of symbols loaded.

        Should be called ONCE per job at startup.
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

        # Read all selected tabs
        # gspread batch approach: read tab by tab with small delays
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

                # Rate limit protection — pause every 50 tabs
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
            "HistoryCache: loaded %d symbols across %d days in %.1fs "
            "(%d tabs failed)",
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

    # Wilder's smoothing over ALL remaining values — this is the key fix.
    # With 200 days of history the loop runs 185 times, properly smoothing
    # the averages before returning the final RSI value.
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

    # MACD line = EMA12 - EMA26 (only where both are non-zero)
    macd_series = []
    for i in range(len(closes)):
        if ema12_series[i] != 0.0 and ema26_series[i] != 0.0:
            macd_series.append(ema12_series[i] - ema26_series[i])
        else:
            macd_series.append(0.0)

    # Signal = EMA9 of MACD line
    # Only compute over non-zero portion
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

    subset = closes[-period:]
    middle = sum(subset) / period
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

    # Align to same length
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

    # Wilder smoothing
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

    # Trend: compare average of last 5 vs previous 5
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


def _ema_cross_signal(fast: float, slow: float) -> str:
    """Detect EMA crossover — requires comparing to previous period."""
    if fast <= 0 or slow <= 0:
        return "NONE"
    if fast > slow * 1.001:
        return "GOLDEN"    # fast above slow = bullish
    if fast < slow * 0.999:
        return "DEATH"     # fast below slow = bearish
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
    Bollinger Band signal.
    Squeeze = low volatility, potential breakout setup.
    """
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
        RSI oversold          +2   (entry signal)
        RSI overbought        -2
        Price > EMA200        +2   (bull trend)
        Price < EMA200        -2
        EMA20 > EMA50 (golden)+1
        EMA20 < EMA50 (death) -1
        MACD bullish cross    +1
        MACD bearish cross    -1
        OBV rising            +1
        OBV falling           -1
        Price < BB lower      +1   (mean reversion setup)
        Price > BB upper      -1
    """
    score = 0

    # RSI
    if result.rsi_signal == "OVERSOLD":
        score += 2
    elif result.rsi_signal == "OVERBOUGHT":
        score -= 2

    # EMA trend
    if result.ema_trend == "BULLISH":
        score += 2
    elif result.ema_trend == "BEARISH":
        score -= 2

    # EMA cross
    if result.ema_20_50_cross == "GOLDEN":
        score += 1
    elif result.ema_20_50_cross == "DEATH":
        score -= 1

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

    # Bollinger
    if result.bb_lower > 0 and result.ltp < result.bb_lower:
        score += 1
    elif result.bb_upper > 0 and result.ltp > result.bb_upper:
        score -= 1

    # Clamp to -10 / +10
    score = max(-10, min(10, score))

    # Signal label
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
    symbol:     str,
    price_row,              # scraper.PriceRow
    cache:      HistoryCache,
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
    today_high   = price_row.high   if price_row.high   > 0 else today_close
    today_low    = price_row.low    if price_row.low    > 0 else today_close
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
    ema20  = _ema_last(closes, EMA_20_PERIOD)  if n >= EMA_20_PERIOD  else 0.0
    ema50  = _ema_last(closes, EMA_50_PERIOD)  if n >= EMA_50_PERIOD  else 0.0
    ema200 = _ema_last(closes, EMA_200_PERIOD) if n >= EMA_200_PERIOD else 0.0

    ema_trend   = _ema_trend_signal(today_close, ema200)
    cross_20_50 = _ema_cross_signal(ema20, ema50)
    cross_50_200 = _ema_cross_signal(ema50, ema200)

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(closes)
    macd_cross = _macd_cross_signal(macd_line, macd_sig)

    # ── Bollinger Bands ────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(closes, BB_PERIOD)
    bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2) if bb_mid > 0 else 0.0
    bb_pct_b = round(
        (today_close - bb_lower) / (bb_upper - bb_lower), 4
    ) if (bb_upper - bb_lower) > 0 else 0.5
    bb_sig = _bb_signal(bb_width, bb_pct_b)

    # ── ATR ────────────────────────────────────────────────────────────────
    atr     = _atr(highs, lows, closes, ATR_PERIOD)
    atr_pct = round(atr / today_close * 100, 2) if today_close > 0 else 0.0

    # ── OBV ────────────────────────────────────────────────────────────────
    obv_val, obv_trend = _obv(closes, volumes)

    # ── Composite score ────────────────────────────────────────────────────
    result = IndicatorResult(
        symbol         = sym,
        rsi_14         = rsi,
        rsi_signal     = rsi_s,
        ema_20         = round(ema20, 2),
        ema_50         = round(ema50, 2),
        ema_200        = round(ema200, 2),
        ema_trend      = ema_trend,
        ema_20_50_cross  = cross_20_50,
        ema_50_200_cross = cross_50_200,
        macd_line      = macd_line,
        macd_signal    = macd_sig,
        macd_histogram = macd_hist,
        macd_cross     = macd_cross,
        bb_upper       = bb_upper,
        bb_middle      = bb_mid,
        bb_lower       = bb_lower,
        bb_width       = bb_width,
        bb_pct_b       = bb_pct_b,
        bb_signal      = bb_sig,
        atr_14         = atr,
        atr_pct        = atr_pct,
        obv            = obv_val,
        obv_trend      = obv_trend,
        ltp            = today_close,
        prev_close     = price_row.prev_close,
        volume         = price_row.volume,
        history_days   = n,
    )

    score, signal      = _tech_score(result)
    result.tech_score  = score
    result.tech_signal = signal

    return result


def compute_all_indicators(
    market_data: dict,       # dict[symbol, PriceRow] from scraper.py
    cache: HistoryCache,
    min_history: int = RSI_PERIOD + 1,
) -> dict[str, IndicatorResult]:
    """
    Compute indicators for ALL symbols in market_data.
    Returns dict[symbol, IndicatorResult].
    Symbols with insufficient history are silently skipped.

    Args:
        market_data:  Output of scraper.get_all_market_data()
        cache:        Loaded HistoryCache
        min_history:  Minimum days required (default 15 for RSI)
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
# SHEETS I/O — write once daily, read during trading session
# ══════════════════════════════════════════════════════════════════════════════

def _result_to_row(r: IndicatorResult) -> dict:
    """Convert IndicatorResult to a flat dict matching INDICATORS_TAB_COLUMNS."""
    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    return {
        "Symbol":           r.symbol,
        "Date":             today,
        "LTP":              r.ltp,
        "Prev_Close":       r.prev_close,
        "Volume":           r.volume,
        "History_Days":     r.history_days,
        "RSI_14":           r.rsi_14,
        "RSI_Signal":       r.rsi_signal,
        "EMA_20":           r.ema_20,
        "EMA_50":           r.ema_50,
        "EMA_200":          r.ema_200,
        "EMA_Trend":        r.ema_trend,
        "EMA_20_50_Cross":  r.ema_20_50_cross,
        "EMA_50_200_Cross": r.ema_50_200_cross,
        "MACD_Line":        r.macd_line,
        "MACD_Signal":      r.macd_signal,
        "MACD_Histogram":   r.macd_histogram,
        "MACD_Cross":       r.macd_cross,
        "BB_Upper":         r.bb_upper,
        "BB_Middle":        r.bb_middle,
        "BB_Lower":         r.bb_lower,
        "BB_Width":         r.bb_width,
        "BB_Pct_B":         r.bb_pct_b,
        "BB_Signal":        r.bb_signal,
        "ATR_14":           r.atr_14,
        "ATR_Pct":          r.atr_pct,
        "OBV":              r.obv,
        "OBV_Trend":        r.obv_trend,
        "Tech_Score":       r.tech_score,
        "Tech_Signal":      r.tech_signal,
        "Timestamp":        r.timestamp,
    }


def write_indicators_to_sheets(
    results: dict[str, IndicatorResult],
) -> bool:
    """
    Write all indicator results to the INDICATORS tab in NEPSE_CORE_ENGINE.

    Strategy:
        Clear the tab (keep header) → batch write all rows at once.
        One write operation per run — not one per symbol.
        Called once daily by run_daily_indicators().

    Returns True on success, False on failure (never raises).
    """
    if not results:
        logger.warning("write_indicators_to_sheets: no results to write")
        return False

    try:
        from sheets import get_setting  # noqa — just to test sheets import
        import gspread
        from sheets import _get_tab, TABS  # noqa

        # Ensure INDICATORS tab exists in TABS dict
        if "indicators" not in TABS:
            TABS["indicators"] = "INDICATORS"

        ws = _get_tab("INDICATORS")

        # Clear existing data (keep header row)
        ws.clear()
        ws.append_row(INDICATORS_TAB_COLUMNS, value_input_option="USER_ENTERED")

        # Build all rows at once for batch write
        rows = []
        for r in results.values():
            row_dict = _result_to_row(r)
            row = [row_dict.get(col, "") for col in INDICATORS_TAB_COLUMNS]
            rows.append(row)

        if rows:
            # Batch write — single API call for all symbols
            start_row = 2  # row 1 = header
            end_row   = start_row + len(rows) - 1
            end_col   = len(INDICATORS_TAB_COLUMNS)

            # Convert col number to letter
            def col_letter(n: int) -> str:
                result = ""
                while n > 0:
                    n, r = divmod(n - 1, 26)
                    result = chr(65 + r) + result
                return result

            range_notation = f"A{start_row}:{col_letter(end_col)}{end_row}"
            ws.update(range_notation, rows, value_input_option="USER_ENTERED")

        logger.info(
            "write_indicators_to_sheets: wrote %d symbols to INDICATORS tab",
            len(rows),
        )
        return True

    except ImportError:
        logger.warning("sheets.py not importable — indicators not written to Sheets")
        return False
    except Exception as exc:
        logger.error("write_indicators_to_sheets failed: %s", exc)
        return False


def read_indicators_from_sheets() -> dict[str, IndicatorResult]:
    """
    Read today's pre-computed indicators from the INDICATORS tab.
    Called by filter_engine.py during the 6-minute trading loop.

    Returns dict[symbol, IndicatorResult].
    Returns empty dict if tab is missing or stale (different date).
    filter_engine.py handles the empty dict gracefully.
    """
    try:
        from sheets import read_tab, TABS  # noqa

        if "indicators" not in TABS:
            TABS["indicators"] = "INDICATORS"

        rows = read_tab("indicators")
        if not rows:
            logger.warning("read_indicators_from_sheets: INDICATORS tab is empty")
            return {}

        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        results: dict[str, IndicatorResult] = {}
        stale = 0

        for row in rows:
            symbol = str(row.get("Symbol", "")).strip().upper()
            if not symbol:
                continue

            # Check freshness — skip if from a previous day
            row_date = str(row.get("Date", "")).strip()
            if row_date and row_date != today:
                stale += 1
                continue

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
                ltp              = _f("LTP"),
                prev_close       = _f("Prev_Close"),
                volume           = _i("Volume"),
                history_days     = _i("History_Days"),
                rsi_14           = _f("RSI_14"),
                rsi_signal       = _s("RSI_Signal", "NEUTRAL"),
                ema_20           = _f("EMA_20"),
                ema_50           = _f("EMA_50"),
                ema_200          = _f("EMA_200"),
                ema_trend        = _s("EMA_Trend", "NEUTRAL"),
                ema_20_50_cross  = _s("EMA_20_50_Cross", "NONE"),
                ema_50_200_cross = _s("EMA_50_200_Cross", "NONE"),
                macd_line        = _f("MACD_Line"),
                macd_signal      = _f("MACD_Signal"),
                macd_histogram   = _f("MACD_Histogram"),
                macd_cross       = _s("MACD_Cross", "NONE"),
                bb_upper         = _f("BB_Upper"),
                bb_middle        = _f("BB_Middle"),
                bb_lower         = _f("BB_Lower"),
                bb_width         = _f("BB_Width"),
                bb_pct_b         = _f("BB_Pct_B"),
                bb_signal        = _s("BB_Signal", "NEUTRAL"),
                atr_14           = _f("ATR_14"),
                atr_pct          = _f("ATR_Pct"),
                obv              = _f("OBV"),
                obv_trend        = _s("OBV_Trend", "NEUTRAL"),
                tech_score       = _i("Tech_Score"),
                tech_signal      = _s("Tech_Signal", "NEUTRAL"),
                timestamp        = _s("Timestamp"),
            )
            results[symbol] = result

        if stale:
            logger.warning(
                "read_indicators_from_sheets: %d stale rows skipped "
                "(from previous day — morning run may not have completed yet)",
                stale,
            )

        logger.info(
            "read_indicators_from_sheets: loaded %d symbols from INDICATORS tab",
            len(results),
        )
        return results

    except ImportError:
        logger.warning("sheets.py not importable — cannot read INDICATORS tab")
        return {}
    except Exception as exc:
        logger.error("read_indicators_from_sheets failed: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# DAILY PIPELINE — called by morning_brief.yml at 10:30 AM NST
# ══════════════════════════════════════════════════════════════════════════════

def run_daily_indicators() -> dict[str, IndicatorResult]:
    """
    Full daily pipeline:
        1. Load 220 days history from GOOGLE_SHEET_SHARE_DATA
        2. Fetch today's live prices from TMS + ShareSansar
        3. Compute all indicators
        4. Write results to INDICATORS tab in NEPSE_CORE_ENGINE
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
        from scraper import get_all_market_data  # noqa
        market_data = get_all_market_data(write_breadth=True)
    except Exception as exc:
        logger.error("run_daily_indicators: scraper failed — %s", exc)
        return {}

    if not market_data:
        logger.warning(
            "run_daily_indicators: no live prices (market not open yet?) "
            "— computing from last historical close instead"
        )
        # Pre-open scenario: use last cache close as today's price
        # This is fine at 10:30 AM before open — indicators are daily anyway
        try:
            from scraper import PriceRow  # noqa
            market_data = {}
            for sym, closes in cache.closes.items():
                if closes:
                    last = closes[-1]
                    highs  = cache.get_highs(sym)
                    lows   = cache.get_lows(sym)
                    market_data[sym] = PriceRow(
                        symbol     = sym,
                        ltp        = last,
                        close      = last,
                        high       = highs[-1]  if highs  else last,
                        low        = lows[-1]   if lows   else last,
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

    # Step 4: Write to Sheets
    ok = write_indicators_to_sheets(results)
    if not ok:
        logger.warning(
            "run_daily_indicators: Sheets write failed — "
            "results still returned in memory for this session"
        )

    elapsed = round(time.time() - start, 2)
    logger.info(
        "run_daily_indicators: complete — %d symbols in %.1fs | Sheets write: %s",
        len(results), elapsed, "✅" if ok else "❌",
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python indicators.py --daily          → full daily pipeline (write to Sheets)
#   python indicators.py --read           → read back from INDICATORS tab
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
    print("  NEPSE AI — indicators.py")
    print("=" * 70)

    # ── --daily: full pipeline ─────────────────────────────────────────────
    if "--daily" in args:
        print("\n  Running full daily pipeline → will write to INDICATORS tab\n")
        results = run_daily_indicators()
        if not results:
            print("  ❌ Daily run failed")
            sys.exit(1)
        print(f"\n  ✅ {len(results)} symbols computed and written to Sheets")
        dist = Counter(r.tech_signal for r in results.values())
        print("\n  Signal distribution:")
        for sig, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {sig:<15} {count}")
        sys.exit(0)

    # ── --read: verify what's in Sheets ───────────────────────────────────
    if "--read" in args:
        print("\n  Reading INDICATORS tab from Sheets...\n")
        results = read_indicators_from_sheets()
        if not results:
            print("  ❌ Nothing found — run --daily first")
            sys.exit(1)
        print(f"  ✅ {len(results)} symbols loaded\n")
        print(f"  {'Symbol':<10} {'RSI':>6} {'EMA_Trend':<10} {'MACD_Cross':<12} "
              f"{'Score':>6} {'Signal'}")
        print("  " + "-" * 60)
        top = sorted(results.values(), key=lambda r: r.tech_score, reverse=True)[:15]
        for r in top:
            print(f"  {r.symbol:<10} {r.rsi_14:>6.1f} {r.ema_trend:<10} "
                  f"{r.macd_cross:<12} {r.tech_score:>+6}  {r.tech_signal}")
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

    # ── Symbol / test mode: compute without writing to Sheets ──────────────
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
        print(f"  ✅ {len(market_data)} symbols (use --daily to write to Sheets)")

    print(f"\n[3/3] Computing indicators...")
    results = compute_all_indicators(market_data, cache)
    print(f"  ✅ {len(results)} computed\n")

    if not results:
        print("  No results")
        sys.exit(0)

    sorted_results = sorted(results.values(), key=lambda r: r.tech_score, reverse=True)
    print(f"  {'Symbol':<10} {'LTP':>8} {'RSI':>6} {'EMA20':>8} {'EMA200':>8} "
          f"{'MACD':>7} {'OBV':>8} {'Score':>6} {'Signal'}")
    print("  " + "-" * 80)
    for r in sorted_results[:20]:
        print(
            f"  {r.symbol:<10} {r.ltp:>8.2f} {r.rsi_14:>6.1f} "
            f"{r.ema_20:>8.2f} {r.ema_200:>8.2f} "
            f"{r.macd_line:>+7.3f} {r.obv_trend:>8} "
            f"{r.tech_score:>+6}  {r.tech_signal}"
        )

    dist = Counter(r.tech_signal for r in results.values())
    print(f"\n  Signal distribution:")
    for sig, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {sig:<15} {count}")

    print(f"\n  💡 Run with --daily to compute all + write to Sheets")
    print("=" * 70 + "\n")
