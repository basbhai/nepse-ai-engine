"""
candle_detector.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2B
Purpose : Detect candlestick patterns for every symbol using historical
          OHLCV data from HistoryCache (indicators.py).

Performance rewrite (v2):
    All pattern math is fully vectorized with NumPy arrays.
    Pattern detection for 335 symbols runs in a single vectorized pass
    instead of Python for-loops — ~10–30x faster than the list-based v1.

    Key vectorization strategy:
        1. detect_all_patterns() builds 2D NumPy arrays for O/H/L/C/V
           with shape (n_symbols, LOOKBACK) in ONE batch.
        2. Each detector receives the full matrix and returns matched rows.
        3. CandlePattern objects are only instantiated for matched symbols.

DB writes (two separate tables):
    candle_patterns      — reference table: one row per pattern name
                           (win rates, tiers, etc.) — upserted, never logs daily.
    candle_signals       — daily time-series: one row per (symbol, date, pattern)
                           This is what gives you historical reference.
                           Written by write_daily_signals_to_db().

Usage:
    from candle_detector import detect_all_patterns, write_daily_signals_to_db
    from indicators import HistoryCache

    cache = HistoryCache()
    cache.load()
    all_patterns = detect_all_patterns(market_data, cache)
    write_daily_signals_to_db(all_patterns)          # ← saves to DB

─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — NEPSE-tuned thresholds
# ══════════════════════════════════════════════════════════════════════════════

LOOKBACK = 5   # candles per symbol (including today)

# Body/wick ratio thresholds — tuned for NEPSE volatility
SMALL_BODY_RATIO    = 0.25   # body < 25% of range → doji / spinning top
DOJI_BODY_RATIO     = 0.10   # body < 10% of range → strict doji
LONG_BODY_RATIO     = 0.60   # body > 60% of range → marubozu / engulfing

# Hammer / Shooting Star wick thresholds
HAMMER_LOWER_WICK   = 2.0    # lower wick >= 2x body
HAMMER_UPPER_WICK   = 0.30   # upper wick <= 30% of body
SHOOTING_UPPER_WICK = 2.0    # upper wick >= 2x body
SHOOTING_LOWER_WICK = 0.30   # lower wick <= 30% of body

# Volume surge: today volume / prior 4-day avg >= this → confirmed
VOLUME_SURGE_RATIO  = 1.5

# Support/Resistance proximity (within 3% of recent swing high/low)
SR_PROXIMITY_PCT    = 0.03


# ══════════════════════════════════════════════════════════════════════════════
# CANDLE PATTERN RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandlePattern:
    """
    A detected candlestick pattern for one symbol.
    filter_engine.py reads this and adds it to each candidate's context.
    """
    symbol:           str
    name:             str       # e.g. "Bullish Engulfing"
    signal:           str       # BULLISH / BEARISH / NEUTRAL
    tier:             int       # 1 / 2 / 3
    confidence:       int       # 0-100
    description:      str
    volume_confirmed: bool = False
    candles_used:     int  = 1

    timestamp: str = field(default_factory=lambda: datetime.now(
        tz=timezone(timedelta(hours=5, minutes=45))
    ).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        vol = " [VOL]" if self.volume_confirmed else ""
        return (
            f"T{self.tier} | {self.name} | {self.signal} | "
            f"conf={self.confidence}{vol} | {self.description}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# VECTORIZED CANDLE MATH PRIMITIVES
# All operate on NumPy arrays of shape (n_symbols,) sliced from the matrix.
# ══════════════════════════════════════════════════════════════════════════════

def _body(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(c - o)

def _candle_range(h: np.ndarray, l: np.ndarray) -> np.ndarray:
    return np.maximum(h - l, 1e-9)   # avoid divide-by-zero

def _upper_wick(o: np.ndarray, h: np.ndarray, c: np.ndarray) -> np.ndarray:
    return h - np.maximum(o, c)

def _lower_wick(o: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.minimum(o, c) - l

def _body_ratio(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _body(o, c) / _candle_range(h, l)

def _is_bullish(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c > o

def _is_bearish(o: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c < o

def _volume_ratio(vol_matrix: np.ndarray) -> np.ndarray:
    """Today's volume / mean of prior LOOKBACK-1 candles."""
    prior_avg = vol_matrix[:, :-1].mean(axis=1)
    today_vol = vol_matrix[:, -1]
    safe_avg  = np.where(prior_avg > 0, prior_avg, 1.0)
    return today_vol / safe_avg

def _is_downtrend(close_matrix: np.ndarray, periods: int = 3) -> np.ndarray:
    """True where last `periods` HISTORICAL closes are declining."""
    if close_matrix.shape[1] < periods + 1:
        return np.zeros(close_matrix.shape[0], dtype=bool)
    window = close_matrix[:, -(periods + 1):-1]
    diffs  = np.diff(window, axis=1)
    return np.all(diffs < 0, axis=1)

def _is_uptrend(close_matrix: np.ndarray, periods: int = 3) -> np.ndarray:
    if close_matrix.shape[1] < periods + 1:
        return np.zeros(close_matrix.shape[0], dtype=bool)
    window = close_matrix[:, -(periods + 1):-1]
    diffs  = np.diff(window, axis=1)
    return np.all(diffs > 0, axis=1)

def _near_support(close_matrix: np.ndarray, low_today: np.ndarray) -> np.ndarray:
    """Today's low is within SR_PROXIMITY_PCT of the recent swing low."""
    prior_closes = close_matrix[:, :-1]
    swing_low    = np.nanmin(prior_closes, axis=1)
    safe_swing   = np.where(swing_low > 0, swing_low, 1.0)
    return np.abs(low_today - swing_low) / safe_swing <= SR_PROXIMITY_PCT

def _near_resistance(close_matrix: np.ndarray, high_today: np.ndarray) -> np.ndarray:
    """Today's high is within SR_PROXIMITY_PCT of the recent swing high."""
    prior_closes = close_matrix[:, :-1]
    swing_high   = np.nanmax(prior_closes, axis=1)
    safe_swing   = np.where(swing_high > 0, swing_high, 1.0)
    return np.abs(high_today - swing_high) / safe_swing <= SR_PROXIMITY_PCT


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX BUILDER
# Converts HistoryCache + today's PriceRow into NumPy matrices.
# ══════════════════════════════════════════════════════════════════════════════

def _build_matrices(symbols: list, market_data: dict, cache) -> tuple:
    """
    Build per-symbol OHLCV matrices of shape (n_valid, LOOKBACK).

    Returns:
        valid_symbols  list[str]
        O, H, L, C, V  np.ndarray (n_valid, LOOKBACK)
        C_full          np.ndarray (n_valid, max_hist) — full history, NaN-padded left
    """
    n_hist = LOOKBACK - 1

    O_rows, H_rows, L_rows, C_rows, V_rows = [], [], [], [], []
    C_full_rows  = []
    valid_symbols = []

    for sym in symbols:
        row = market_data.get(sym)
        if row is None:
            continue

        hist_c = cache.get_closes(sym)
        hist_h = cache.get_highs(sym)
        hist_l = cache.get_lows(sym)
        hist_v = cache.get_volumes(sym)

        if len(hist_c) < 1:
            continue

        today_o = row.open_price if row.open_price > 0 else (row.prev_close or row.ltp)
        today_h = row.high       if row.high       > 0 else row.ltp
        today_l = row.low        if row.low        > 0 else row.ltp
        today_c = row.ltp        if row.ltp        > 0 else row.close
        today_v = float(row.volume)

        # Approximate historical opens as prior close (OmitNomis has no open column)
        o_hist = np.array(hist_c[-n_hist:], dtype=np.float64)
        h_hist = np.array(hist_h[-n_hist:] if hist_h else hist_c[-n_hist:], dtype=np.float64)
        l_hist = np.array(hist_l[-n_hist:] if hist_l else hist_c[-n_hist:], dtype=np.float64)
        c_hist = np.array(hist_c[-n_hist:], dtype=np.float64)
        v_hist = np.array(hist_v[-n_hist:] if hist_v else [today_v] * n_hist, dtype=np.float64)

        # Pad left with NaN if fewer than n_hist days available
        pad = n_hist - len(c_hist)
        if pad > 0:
            nan_pad = np.full(pad, np.nan)
            o_hist  = np.concatenate([nan_pad, o_hist])
            h_hist  = np.concatenate([nan_pad, h_hist])
            l_hist  = np.concatenate([nan_pad, l_hist])
            c_hist  = np.concatenate([nan_pad, c_hist])
            v_hist  = np.concatenate([nan_pad, v_hist])

        O_rows.append(np.append(o_hist, today_o))
        H_rows.append(np.append(h_hist, today_h))
        L_rows.append(np.append(l_hist, today_l))
        C_rows.append(np.append(c_hist, today_c))
        V_rows.append(np.append(v_hist, today_v))

        C_full_rows.append(np.array(hist_c + [today_c], dtype=np.float64))
        valid_symbols.append(sym)

    if not valid_symbols:
        empty = np.empty((0, LOOKBACK), dtype=np.float64)
        return [], empty, empty, empty, empty, empty, empty

    O = np.array(O_rows, dtype=np.float64)
    H = np.array(H_rows, dtype=np.float64)
    L = np.array(L_rows, dtype=np.float64)
    C = np.array(C_rows, dtype=np.float64)
    V = np.array(V_rows, dtype=np.float64)

    # Pad C_full to same width (left-pad with NaN)
    max_hist = max(len(r) for r in C_full_rows)
    C_full = np.full((len(valid_symbols), max_hist), np.nan, dtype=np.float64)
    for i, row in enumerate(C_full_rows):
        C_full[i, max_hist - len(row):] = row

    return valid_symbols, O, H, L, C, V, C_full


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 VECTORIZED DETECTORS
# Each takes full matrices, returns list of (index, CandlePattern) for matches.
# ══════════════════════════════════════════════════════════════════════════════

def _detect_bullish_engulfing(symbols, O, H, L, C, V, C_full):
    prev_o, prev_c = O[:, -2], C[:, -2]
    curr_o, curr_c = O[:, -1], C[:, -1]

    prev_bearish  = _is_bearish(prev_o, prev_c)
    curr_bullish  = _is_bullish(curr_o, curr_c)
    engulfs       = (curr_o <= prev_c) & (curr_c >= prev_o)
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=3)
    big_body      = _body_ratio(curr_o, H[:, -1], L[:, -1], curr_c) > LONG_BODY_RATIO

    match = prev_bearish & curr_bullish & engulfs
    out = []
    for i in np.where(match)[0]:
        conf = min(70 + 10*vol_confirmed[i] + 10*in_downtrend[i] + 5*big_body[i], 95)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Bullish Engulfing", signal="BULLISH",
            tier=1, confidence=int(conf),
            description=(
                f"Green ({curr_o[i]:.0f}->{curr_c[i]:.0f}) engulfs "
                f"red ({prev_o[i]:.0f}->{prev_c[i]:.0f}). "
                f"{'Downtrend. ' if in_downtrend[i] else ''}Strong reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    return out


def _detect_bearish_engulfing(symbols, O, H, L, C, V, C_full):
    prev_o, prev_c = O[:, -2], C[:, -2]
    curr_o, curr_c = O[:, -1], C[:, -1]

    prev_bullish  = _is_bullish(prev_o, prev_c)
    curr_bearish  = _is_bearish(curr_o, curr_c)
    engulfs       = (curr_o >= prev_c) & (curr_c <= prev_o)
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(C_full, periods=3)
    big_body      = _body_ratio(curr_o, H[:, -1], L[:, -1], curr_c) > LONG_BODY_RATIO

    match = prev_bullish & curr_bearish & engulfs
    out = []
    for i in np.where(match)[0]:
        conf = min(70 + 10*vol_confirmed[i] + 10*in_uptrend[i] + 5*big_body[i], 95)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Bearish Engulfing", signal="BEARISH",
            tier=1, confidence=int(conf),
            description=(
                f"Red ({curr_o[i]:.0f}->{curr_c[i]:.0f}) engulfs "
                f"green ({prev_o[i]:.0f}->{prev_c[i]:.0f}). "
                f"{'Uptrend. ' if in_uptrend[i] else ''}Strong reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    return out


def _detect_hammer(symbols, O, H, L, C, V, C_full):
    o, h, l, c = O[:, -1], H[:, -1], L[:, -1], C[:, -1]
    body    = _body(o, c)
    lower_w = _lower_wick(o, l, c)
    upper_w = _upper_wick(o, h, c)
    sb      = np.where(body > 0, body, 1e-9)

    match = (
        (lower_w >= HAMMER_LOWER_WICK * sb) &
        (upper_w <= HAMMER_UPPER_WICK * sb) &
        (body > 0) & (_candle_range(h, l) > 0)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=3)
    at_support    = _near_support(C_full, l)
    is_green      = _is_bullish(o, c)

    out = []
    for i in np.where(match)[0]:
        conf = min(60 + 15*in_downtrend[i] + 10*at_support[i] + 8*vol_confirmed[i] + 5*is_green[i], 92)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Hammer", signal="BULLISH",
            tier=1, confidence=int(conf),
            description=(
                f"{'Green' if is_green[i] else 'Red'} hammer: "
                f"lower_wick={lower_w[i]:.0f}, body={body[i]:.0f}. "
                f"{'Support. ' if at_support[i] else ''}"
                f"{'Downtrend. ' if in_downtrend[i] else ''}Bullish reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=1,
        )))
    return out


def _detect_shooting_star(symbols, O, H, L, C, V, C_full):
    o, h, l, c = O[:, -1], H[:, -1], L[:, -1], C[:, -1]
    body    = _body(o, c)
    upper_w = _upper_wick(o, h, c)
    lower_w = _lower_wick(o, l, c)
    sb      = np.where(body > 0, body, 1e-9)

    match = (
        (upper_w >= SHOOTING_UPPER_WICK * sb) &
        (lower_w <= SHOOTING_LOWER_WICK * sb) &
        (body > 0) & (_candle_range(h, l) > 0)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(C_full, periods=3)
    at_resistance = _near_resistance(C_full, h)
    is_red        = _is_bearish(o, c)

    out = []
    for i in np.where(match)[0]:
        conf = min(60 + 15*in_uptrend[i] + 10*at_resistance[i] + 8*vol_confirmed[i] + 5*is_red[i], 92)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Shooting Star", signal="BEARISH",
            tier=1, confidence=int(conf),
            description=(
                f"Shooting star: upper_wick={upper_w[i]:.0f}, body={body[i]:.0f}. "
                f"{'Resistance. ' if at_resistance[i] else ''}"
                f"{'Uptrend. ' if in_uptrend[i] else ''}Bearish reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=1,
        )))
    return out


def _detect_morning_star(symbols, O, H, L, C, V, C_full):
    if C.shape[1] < 3:
        return []
    o1, c1 = O[:, -3], C[:, -3]
    o2, c2 = O[:, -2], C[:, -2]
    o3, c3 = O[:, -1], C[:, -1]

    body1      = _body(o1, c1)
    body2      = _body(o2, c2)
    body3      = _body(o3, c3)
    sb1        = np.where(body1 > 0, body1, 1e-9)
    midpoint1  = (o1 + c1) / 2

    match = (
        _is_bearish(o1, c1) & (body2 <= sb1 * 0.4) &
        _is_bullish(o3, c3) & (c3 >= midpoint1) &
        (body1 > 0) & (body3 > 0)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=3)
    strong_recov  = body3 >= sb1 * 0.8

    out = []
    for i in np.where(match)[0]:
        conf = min(72 + 8*vol_confirmed[i] + 8*in_downtrend[i] + 7*strong_recov[i], 92)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Morning Star", signal="BULLISH",
            tier=1, confidence=int(conf),
            description=(
                f"3-candle: bearish({c1[i]:.0f}), "
                f"indecision({c2[i]:.0f}), "
                f"recovery({c3[i]:.0f}) above midpoint. Bottom reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=3,
        )))
    return out


def _detect_evening_star(symbols, O, H, L, C, V, C_full):
    if C.shape[1] < 3:
        return []
    o1, c1 = O[:, -3], C[:, -3]
    o2, c2 = O[:, -2], C[:, -2]
    o3, c3 = O[:, -1], C[:, -1]

    body1      = _body(o1, c1)
    body2      = _body(o2, c2)
    body3      = _body(o3, c3)
    sb1        = np.where(body1 > 0, body1, 1e-9)
    midpoint1  = (o1 + c1) / 2

    match = (
        _is_bullish(o1, c1) & (body2 <= sb1 * 0.4) &
        _is_bearish(o3, c3) & (c3 <= midpoint1) &
        (body1 > 0) & (body3 > 0)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(C_full, periods=3)
    strong_drop   = body3 >= sb1 * 0.8

    out = []
    for i in np.where(match)[0]:
        conf = min(72 + 8*vol_confirmed[i] + 8*in_uptrend[i] + 7*strong_drop[i], 92)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Evening Star", signal="BEARISH",
            tier=1, confidence=int(conf),
            description=(
                f"3-candle: bullish({c1[i]:.0f}), "
                f"indecision({c2[i]:.0f}), "
                f"selloff({c3[i]:.0f}) below midpoint. Top reversal."
            ),
            volume_confirmed=bool(vol_confirmed[i]), candles_used=3,
        )))
    return out


def _detect_doji(symbols, O, H, L, C, V, C_full):
    o, h, l, c = O[:, -1], H[:, -1], L[:, -1], C[:, -1]
    br         = _body_ratio(o, h, l, c)
    full_range = _candle_range(h, l)

    is_doji       = (br <= DOJI_BODY_RATIO) & (full_range > 0)
    at_support    = _near_support(C_full, l)
    at_resistance = _near_resistance(C_full, h)
    in_downtrend  = _is_downtrend(C_full, periods=3)
    in_uptrend    = _is_uptrend(C_full, periods=3)

    out = []
    for i in np.where(is_doji)[0]:
        body_val = float(np.abs(c[i] - o[i]))
        rng_val  = float(full_range[i])

        if at_support[i] and in_downtrend[i]:
            sig, ctx, conf, tier = "BULLISH", "support after downtrend", 68, 1
        elif at_resistance[i] and in_uptrend[i]:
            sig, ctx, conf, tier = "BEARISH", "resistance after uptrend", 68, 1
        else:
            sig, ctx, conf, tier = "NEUTRAL", "indecision — wait for confirmation", 45, 3

        out.append((i, CandlePattern(
            symbol=symbols[i], name="Doji", signal=sig,
            tier=tier, confidence=conf,
            description=f"Doji (body={body_val:.0f}, range={rng_val:.0f}): {ctx}.",
            volume_confirmed=False, candles_used=1,
        )))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 VECTORIZED DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_three_white_soldiers(symbols, O, H, L, C, V, C_full):
    if C.shape[1] < 3:
        return []
    o1, c1 = O[:, -3], C[:, -3]
    o2, c2 = O[:, -2], C[:, -2]
    o3, c3 = O[:, -1], C[:, -1]

    match = (
        _is_bullish(o1, c1) & _is_bullish(o2, c2) & _is_bullish(o3, c3) &
        (c1 < c2) & (c2 < c3) &
        (o2 > o1) & (o2 < c1) &
        (o3 > o2) & (o3 < c2)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=4)

    out = []
    for i in np.where(match)[0]:
        conf = min(68 + 10*in_downtrend[i] + 8*vol_confirmed[i], 88)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Three White Soldiers", signal="BULLISH",
            tier=2, confidence=int(conf),
            description=f"3 green: {c1[i]:.0f}>{c2[i]:.0f}>{c3[i]:.0f}. Bullish momentum.",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=3,
        )))
    return out


def _detect_three_black_crows(symbols, O, H, L, C, V, C_full):
    if C.shape[1] < 3:
        return []
    o1, c1 = O[:, -3], C[:, -3]
    o2, c2 = O[:, -2], C[:, -2]
    o3, c3 = O[:, -1], C[:, -1]

    # Standard definition: each candle is bearish, closes lower, and opens within the prior body.
    # For a bearish candle, the body spans from close (low) to open (high). So:
    #   - Second candle's open should be between first's close and first's open.
    #   - Third candle's open should be between second's close and second's open.
    match = (
        _is_bearish(o1, c1) & _is_bearish(o2, c2) & _is_bearish(o3, c3) &
        (c1 > c2) & (c2 > c3) &
        (o2 >= c1) & (o2 <= o1) &           # open inside first body
        (o3 >= c2) & (o3 <= o2)              # open inside second body
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(C_full, periods=4)

    out = []
    for i in np.where(match)[0]:
        conf = min(68 + 10*in_uptrend[i] + 8*vol_confirmed[i], 88)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Three Black Crows", signal="BEARISH",
            tier=2, confidence=int(conf),
            description=f"3 red: {c1[i]:.0f}<{c2[i]:.0f}<{c3[i]:.0f}. Bearish momentum.",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=3,
        )))
    return out


def _detect_piercing_line(symbols, O, H, L, C, V, C_full):
    o1, c1 = O[:, -2], C[:, -2]
    o2, c2 = O[:, -1], C[:, -1]
    l1     = L[:, -2]
    midpt1 = (o1 + c1) / 2

    match = (
        _is_bearish(o1, c1) & _is_bullish(o2, c2) &
        (o2 <= l1 * 1.005) &
        (c2 >= midpt1) & (c2 < o1)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=3)

    out = []
    for i in np.where(match)[0]:
        conf = min(62 + 10*in_downtrend[i] + 8*vol_confirmed[i], 85)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Piercing Line", signal="BULLISH",
            tier=2, confidence=int(conf),
            description=f"Green ({c2[i]:.0f}) pierces above midpoint of red ({c1[i]:.0f}).",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    return out


def _detect_dark_cloud_cover(symbols, O, H, L, C, V, C_full):
    o1, h1, c1 = O[:, -2], H[:, -2], C[:, -2]
    o2, c2     = O[:, -1], C[:, -1]
    midpt1     = (o1 + c1) / 2

    match = (
        _is_bullish(o1, c1) & _is_bearish(o2, c2) &
        (o2 >= h1 * 0.995) &
        (c2 <= midpt1) & (c2 > o1)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_uptrend    = _is_uptrend(C_full, periods=3)

    out = []
    for i in np.where(match)[0]:
        conf = min(62 + 10*in_uptrend[i] + 8*vol_confirmed[i], 85)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Dark Cloud Cover", signal="BEARISH",
            tier=2, confidence=int(conf),
            description=f"Red ({c2[i]:.0f}) closes below midpoint of green ({c1[i]:.0f}).",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    return out


def _detect_harami(symbols, O, H, L, C, V, C_full):
    o1, c1 = O[:, -2], C[:, -2]
    o2, c2 = O[:, -1], C[:, -1]

    body1      = _body(o1, c1)
    body2      = _body(o2, c2)
    sb1        = np.where(body1 > 0, body1, 1e-9)
    high1_body = np.maximum(o1, c1)
    low1_body  = np.minimum(o1, c1)
    high2_body = np.maximum(o2, c2)
    low2_body  = np.minimum(o2, c2)

    inside  = (high2_body <= high1_body) & (low2_body >= low1_body)
    smaller = body2 <= sb1 * 0.6

    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    in_downtrend  = _is_downtrend(C_full, periods=3)
    in_uptrend    = _is_uptrend(C_full, periods=3)

    bull = inside & smaller & _is_bearish(o1, c1) & _is_bullish(o2, c2)
    bear = inside & smaller & _is_bullish(o1, c1) & _is_bearish(o2, c2)

    out = []
    for i in np.where(bull)[0]:
        conf = min(58 + 10*in_downtrend[i] + 7*vol_confirmed[i], 80)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Bullish Harami", signal="BULLISH",
            tier=2, confidence=int(conf),
            description=f"Small green (body={body2[i]:.0f}) inside large red (body={body1[i]:.0f}). Bottom reversal.",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    for i in np.where(bear)[0]:
        conf = min(58 + 10*in_uptrend[i] + 7*vol_confirmed[i], 80)
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Bearish Harami", signal="BEARISH",
            tier=2, confidence=int(conf),
            description=f"Small red (body={body2[i]:.0f}) inside large green (body={body1[i]:.0f}). Top reversal.",
            volume_confirmed=bool(vol_confirmed[i]), candles_used=2,
        )))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 VECTORIZED DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_spinning_top(symbols, O, H, L, C, V, C_full):
    o, h, l, c = O[:, -1], H[:, -1], L[:, -1], C[:, -1]
    br      = _body_ratio(o, h, l, c)
    full_r  = _candle_range(h, l)
    upper_w = _upper_wick(o, h, c)
    lower_w = _lower_wick(o, l, c)

    match = (
        (br > DOJI_BODY_RATIO) & (br <= SMALL_BODY_RATIO) &
        (upper_w >= full_r * 0.1) & (lower_w >= full_r * 0.1)
    )
    out = []
    for i in np.where(match)[0]:
        body_val = float(np.abs(c[i] - o[i]))
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Spinning Top", signal="NEUTRAL",
            tier=3, confidence=40,
            description=f"Small body ({body_val:.0f}) with wicks both sides. Indecision.",
            volume_confirmed=False, candles_used=1,
        )))
    return out


def _detect_marubozu(symbols, O, H, L, C, V, C_full):
    o, h, l, c = O[:, -1], H[:, -1], L[:, -1], C[:, -1]
    br      = _body_ratio(o, h, l, c)
    full_r  = _candle_range(h, l)
    upper_w = _upper_wick(o, h, c)
    lower_w = _lower_wick(o, l, c)

    match = (br >= LONG_BODY_RATIO) & (full_r > 0) & (
        (upper_w <= full_r * 0.05) | (lower_w <= full_r * 0.05)
    )
    vol_confirmed = _volume_ratio(V) >= VOLUME_SURGE_RATIO
    is_green      = _is_bullish(o, c)

    out = []
    for i in np.where(match)[0]:
        conf = min(52 + 8*vol_confirmed[i], 70)
        if is_green[i]:
            name, sig = "Bullish Marubozu", "BULLISH"
            desc = f"Strong green (O={o[i]:.0f} C={c[i]:.0f}), no wicks. Buyers in control."
        else:
            name, sig = "Bearish Marubozu", "BEARISH"
            desc = f"Strong red (O={o[i]:.0f} C={c[i]:.0f}), no wicks. Sellers in control."
        out.append((i, CandlePattern(
            symbol=symbols[i], name=name, signal=sig,
            tier=3, confidence=int(conf), description=desc,
            volume_confirmed=bool(vol_confirmed[i]), candles_used=1,
        )))
    return out


def _detect_inside_bar(symbols, O, H, L, C, V, C_full):
    if H.shape[1] < 2:
        return []
    prev_h, prev_l = H[:, -2], L[:, -2]
    curr_h, curr_l = H[:, -1], L[:, -1]

    match = (curr_h < prev_h) & (curr_l > prev_l)
    out = []
    for i in np.where(match)[0]:
        out.append((i, CandlePattern(
            symbol=symbols[i], name="Inside Bar", signal="NEUTRAL",
            tier=3, confidence=38,
            description=(
                f"Range ({curr_l[i]:.0f}-{curr_h[i]:.0f}) inside "
                f"prior ({prev_l[i]:.0f}-{prev_h[i]:.0f}). Consolidation."
            ),
            volume_confirmed=False, candles_used=2,
        )))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DETECTOR REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

_DETECTORS = [
    _detect_bullish_engulfing,
    _detect_bearish_engulfing,
    _detect_hammer,
    _detect_shooting_star,
    _detect_morning_star,
    _detect_evening_star,
    _detect_doji,
    _detect_three_white_soldiers,
    _detect_three_black_crows,
    _detect_piercing_line,
    _detect_dark_cloud_cover,
    _detect_harami,
    _detect_spinning_top,
    _detect_marubozu,
    _detect_inside_bar,
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def detect_all_patterns(
    market_data: dict,
    cache,
) -> dict[str, list[CandlePattern]]:
    """
    Detect candlestick patterns for ALL symbols in one vectorized pass.

    Args:
        market_data: dict[symbol, PriceRow] from scraper.py
        cache:       HistoryCache loaded from indicators.py

    Returns:
        dict[symbol, list[CandlePattern]] — Tier 1 first, then confidence desc.
        Symbols with no patterns map to [].
    """
    symbols = list(market_data.keys())
    if not symbols:
        return {}

    valid_symbols, O, H, L, C, V, C_full = _build_matrices(symbols, market_data, cache)
    if not valid_symbols:
        return {}

    by_index: dict[int, list[CandlePattern]] = {i: [] for i in range(len(valid_symbols))}

    for detector in _DETECTORS:
        try:
            for idx, pattern in detector(valid_symbols, O, H, L, C, V, C_full):
                by_index[idx].append(pattern)
        except Exception as exc:
            logger.debug("candle_detector: %s — %s", detector.__name__, exc)

    results: dict[str, list[CandlePattern]] = {}
    total = 0
    for i, sym in enumerate(valid_symbols):
        plist = sorted(by_index[i], key=lambda p: (p.tier, -p.confidence))
        results[sym] = plist
        total += len(plist)

    for sym in symbols:
        if sym not in results:
            results[sym] = []

    t1 = sum(1 for pl in results.values() for p in pl if p.tier == 1)
    logger.info(
        "detect_all_patterns: %d symbols | %d patterns | %d Tier 1",
        len(valid_symbols), total, t1,
    )
    return results


def detect_patterns(symbol: str, price_row, cache) -> list[CandlePattern]:
    """Single-symbol wrapper. Used by filter_engine.py."""
    return detect_all_patterns({symbol: price_row}, cache).get(symbol.upper(), [])


def get_top_patterns(
    all_patterns:   dict[str, list[CandlePattern]],
    signal:         str = "BULLISH",
    min_tier:       int = 1,
    max_tier:       int = 2,
    min_confidence: int = 60,
) -> list[tuple[str, CandlePattern]]:
    """Filter + rank by signal, tier, confidence. Used by filter_engine.py."""
    out = [
        (sym, p)
        for sym, plist in all_patterns.items()
        for p in plist
        if min_tier <= p.tier <= max_tier
        and p.confidence >= min_confidence
        and (signal == "ANY" or p.signal == signal)
    ]
    out.sort(key=lambda x: x[1].confidence, reverse=True)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DB WRITES
# ══════════════════════════════════════════════════════════════════════════════

def write_patterns_to_db(all_patterns: dict[str, list[CandlePattern]]) -> int:
    """
    Upsert pattern METADATA to the candle_patterns reference table.
    One row per unique pattern name (e.g. "Hammer") — not per symbol or date.
    Uses sheets.write_candle_pattern() — the proper public API for this table.

    For daily signal history, call write_daily_signals_to_db() instead.
    """
    try:
        from sheets import write_candle_pattern
    except ImportError:
        logger.warning("write_patterns_to_db: sheets not available")
        return 0

    written = 0
    seen: set[str] = set()

    for sym, patterns in all_patterns.items():
        for p in patterns:
            if p.name in seen:
                continue
            seen.add(p.name)
            try:
                write_candle_pattern({
                    "pattern_name": p.name,
                    "type":         p.signal,
                    "tier":         str(p.tier),
                    "reliability":  str(p.confidence),
                    "notes":        p.description,
                })
                written += 1
            except Exception as exc:
                logger.warning("write_patterns_to_db: %s — %s", p.name, exc)

    logger.info("write_patterns_to_db: %d reference rows upserted", written)
    return written


def write_daily_signals_to_db(
    all_patterns: dict[str, list[CandlePattern]],
    date: Optional[str] = None,
) -> int:
    """
    Write today's detected signals to the candle_signals table.
    One row per (symbol, date, pattern_name) — upserted on conflict.

    This is the HISTORICAL LOG — every pattern fired on every symbol
    every day is recorded here for future backtesting and reference.

    Table: candle_signals
        symbol          TEXT
        date            TEXT          (YYYY-MM-DD, NST)
        pattern_name    TEXT
        signal          TEXT          BULLISH / BEARISH / NEUTRAL
        tier            INTEGER
        confidence      INTEGER
        volume_confirmed BOOLEAN
        candles_used    INTEGER
        description     TEXT
        timestamp       TEXT          (detection time, NST)

    Args:
        all_patterns: Output of detect_all_patterns()
        date:         Override date string YYYY-MM-DD (default: today NST)

    Returns:
        Number of rows written.
    """
    try:
        from sheets import upsert_row
    except ImportError:
        logger.warning("write_daily_signals_to_db: db not available")
        return 0

    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    written = 0
    for sym, patterns in all_patterns.items():
        for p in patterns:
            try:
                upsert_row(
                    "candle_signals",
                    {
                        "symbol":           sym,
                        "date":             date,
                        "pattern_name":     p.name,
                        "signal":           p.signal,
                        "tier":             str(p.tier),
                        "confidence":       str(p.confidence),
                        "volume_confirmed": "true" if p.volume_confirmed else "false",
                        "candles_used":     str(p.candles_used),
                        "description":      p.description,
                        "timestamp":        p.timestamp,
                    },
                    conflict_columns=["symbol", "date", "pattern_name"],
                )
                written += 1
            except Exception as exc:
                logger.warning(
                    "write_daily_signals_to_db: %s/%s — %s", sym, p.name, exc
                )

    logger.info(
        "write_daily_signals_to_db: %d signal rows written for %s", written, date
    )
    return written


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.candle_detector            → detect all symbols
#   python -m modules.candle_detector NABIL HBL   → specific symbols
#   python -m modules.candle_detector --bullish   → BULLISH only
#   python -m modules.candle_detector --bearish   → BEARISH only
#   python -m modules.candle_detector --save      → also write to DB
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    from collections import Counter

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [CANDLE] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]
    print("\n" + "=" * 70)
    print("  NEPSE AI — candle_detector.py  (v2 — NumPy vectorized)")
    print("=" * 70)

    print("\n[1/3] Loading history cache...")
    try:
        from modules.indicators import HistoryCache, DEFAULT_LOAD_PERIODS
        cache = HistoryCache()
        count = cache.load(periods=DEFAULT_LOAD_PERIODS)
        if count == 0:
            print("  Cache load failed"); sys.exit(1)
        print(f"  {count} symbols | {len(cache.dates)} days")
    except Exception as e:
        print(f"  {e}"); sys.exit(1)

    print("\n[2/3] Fetching live prices...")
    try:
        from modules.scraper import get_all_market_data, PriceRow
        market_data = get_all_market_data(write_breadth=False)
        if not market_data:
            print("  Market closed — using synthetic prices from cache")
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
                        low=lows[-1] if lows else closes[-1],
                        prev_close=closes[-2] if len(closes) > 1 else closes[-1],
                        volume=10000,
                    )
        print(f"  {len(market_data)} symbols")
    except Exception as e:
        print(f"  Scraper failed: {e}"); sys.exit(1)

    # Filter to specific symbols if passed
    sym_args = [a for a in args if not a.startswith("--")]
    if sym_args:
        req = {a.upper() for a in sym_args}
        market_data = {k: v for k, v in market_data.items() if k in req}
        print(f"  Filtered to: {list(market_data.keys())}")

    print(f"\n[3/3] Detecting patterns...")
    t0 = time.time()
    all_patterns = detect_all_patterns(market_data, cache)
    elapsed = time.time() - t0

    found = [(s, p) for s, pl in all_patterns.items() for p in pl]
    if "--bullish" in args:
        found = [(s, p) for s, p in found if p.signal == "BULLISH"]
    elif "--bearish" in args:
        found = [(s, p) for s, p in found if p.signal == "BEARISH"]
    found.sort(key=lambda x: (x[1].tier, -x[1].confidence))

    sym_per_sec = len(market_data) / elapsed if elapsed > 0 else 0
    print(f"  {len(found)} patterns in {elapsed:.2f}s ({sym_per_sec:.0f} sym/s)\n")

    if found:
        print(f"  {'Symbol':<10} {'Tier':<5} {'Pattern':<25} {'Signal':<10} {'Conf':>5} {'Vol':>4}  Description")
        print("  " + "-" * 100)
        for sym, p in found[:40]:
            v = "V" if p.volume_confirmed else " "
            print(f"  {sym:<10} T{p.tier:<4} {p.name:<25} {p.signal:<10} {p.confidence:>4}% {v:>3}  {p.description[:55]}")

        td = Counter(p.tier   for _, p in found)
        sd = Counter(p.signal for _, p in found)
        print(f"\n  Tier:   " + " | ".join(f"T{k}:{v}" for k, v in sorted(td.items())))
        print(f"  Signal: " + " | ".join(f"{k}:{v}" for k, v in sorted(sd.items())))

        top_bull = get_top_patterns(all_patterns, "BULLISH", max_tier=2, min_confidence=65)
        if top_bull:
            print(f"\n  Top BULLISH (T1-2, conf>=65%):")
            for sym, p in top_bull[:10]:
                print(f"    {sym:<10} T{p.tier} {p.name} conf={p.confidence}%{'[VOL]' if p.volume_confirmed else ''}")

        top_bear = get_top_patterns(all_patterns, "BEARISH", max_tier=2, min_confidence=65)
        if top_bear:
            print(f"\n  Top BEARISH (T1-2, conf>=65%):")
            for sym, p in top_bear[:10]:
                print(f"    {sym:<10} T{p.tier} {p.name} conf={p.confidence}%{'[VOL]' if p.volume_confirmed else ''}")

    # --save flag: write both reference + daily signal rows to DB
    if "--save" in args:
        print(f"\n  Writing to DB...")
        ref_written = write_patterns_to_db(all_patterns)
        sig_written = write_daily_signals_to_db(all_patterns)
        print(f"  candle_patterns (reference): {ref_written} rows upserted")
        print(f"  candle_signals  (daily log): {sig_written} rows written")
    else:
        print(f"\n  Tip: run with --save to write results to Neon DB")

    print("=" * 70 + "\n")