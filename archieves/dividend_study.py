"""
analysis/dividend_study.py
─────────────────────────────────────────────────────────────────────────────
NEPSE Dividend Pattern Study.

Validates Adhikari (2023) research findings against actual NEPSE price data:
  - Abnormal returns start day -6 (information leakage)
  - Day +1 = highest single-day return (~9.30%)
  - Day +2 to +9 = negative adjustment period
  - Pre-book-close retail accumulation pattern

Analyzes TWO windows per announcement:
  - ANNOUNCEMENT window: day -10 to day +9 around announcement_date
  - BOOK_CLOSE window:   day -10 to day +2 around book_close_date

Adds SEASONALITY FLAG to separate genuine accumulation from
seasonal bull market noise (dividend season = March-July).

Writes results → dividend_pattern_study table.
Prints aggregate summary at end.

Usage:
    python -m analysis.dividend_study                          # all announcements
    python -m analysis.dividend_study --dry-run               # compute only, no DB write
    python -m analysis.dividend_study --symbol NABIL          # one symbol
    python -m analysis.dividend_study --from 2021-01-01 --to 2023-12-31
    python -m analysis.dividend_study --summary-only          # skip compute, print DB stats
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("dividend_study")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

NST = timezone(timedelta(hours=5, minutes=45))

# Gregorian months that correspond to NEPSE dividend announcement season
# Chaitra (mid-March) through Ashadh (mid-July) — most AGMs happen here
DIVIDEND_SEASON_MONTHS = {3, 4, 5, 6, 7}


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers — inline to keep sheets.py clean
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_announcements(symbol: str = None,
                         from_date: str = None,
                         to_date: str = None) -> list[dict]:
    """
    Fetch dividend announcements that have BOTH announcement_date and
    book_close_date — only these can be fully analyzed.
    """
    from db.connection import _db
    conditions = [
        "announcement_date IS NOT NULL AND announcement_date != ''",
        "book_close_date   IS NOT NULL AND book_close_date   != ''",
    ]
    params = []
    if symbol:
        conditions.append("symbol = %s")
        params.append(symbol)
    if from_date:
        conditions.append("announcement_date >= %s")
        params.append(from_date)
    if to_date:
        conditions.append("announcement_date <= %s")
        params.append(to_date)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT symbol, company, sector, announcement_date, book_close_date,
               fiscal_year, dividend_type, cash_dividend_pct,
               bonus_share_pct, total_dividend_pct, direction
        FROM dividend_announcements
        WHERE {where}
        ORDER BY announcement_date ASC
    """
    try:
        with _db() as cur:
            cur.execute(sql, params or None)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("_fetch_announcements failed: %s", e)
        return []


def _fetch_price_history(symbol: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch OHLCV rows for symbol between from_date and to_date, sorted ASC."""
    from db.connection import _db
    try:
        with _db() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, ltp, volume, turnover
                FROM price_history
                WHERE symbol = %s AND date >= %s AND date <= %s
                ORDER BY date ASC
            """, (symbol, from_date, to_date))
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("_fetch_price_history(%s) failed: %s", symbol, e)
        return []


def _fetch_nepse_index(from_date: str, to_date: str) -> dict[str, float]:
    """
    Fetch NEPSE composite index (index_id=1) for alpha calculation.
    Returns {date: close_value}.
    """
    from db.connection import _db
    try:
        with _db() as cur:
            cur.execute("""
                SELECT date, current_value
                FROM nepse_indices
                WHERE index_id = '1' AND date >= %s AND date <= %s
                ORDER BY date ASC
            """, (from_date, to_date))
            result = {}
            for r in cur.fetchall():
                try:
                    result[r["date"]] = float(r["current_value"])
                except (TypeError, ValueError):
                    pass
            return result
    except Exception as e:
        log.error("_fetch_nepse_index failed: %s", e)
        return {}


def _upsert_pattern(row: dict) -> bool:
    """Upsert one result row into dividend_pattern_study."""
    from sheets import upsert_row
    try:
        upsert_row(
            "dividend_pattern_study",
            row,
            conflict_columns=["symbol", "announcement_date"],
        )
        return True
    except Exception as e:
        log.error("_upsert_pattern failed %s %s %s: %s",
                  row.get("symbol"), row.get("announcement_date"),
                  row.get("event_type"), e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality
# ─────────────────────────────────────────────────────────────────────────────

def _is_dividend_season(date_str: str) -> str:
    """
    Returns "true" if announcement falls in NEPSE dividend season
    (March-July = Chaitra through Ashadh — when most AGMs happen).

    Off-season pattern rate is the CLEANER signal — no seasonal
    bull market tailwind contaminating the accumulation fingerprint.
    """
    try:
        month = int(date_str[5:7])
        return "true" if month in DIVIDEND_SEASON_MONTHS else "false"
    except (IndexError, ValueError):
        return "false"


# ─────────────────────────────────────────────────────────────────────────────
# Technical indicators — self-contained, no dependency on indicators.py
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default=None):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_close(row: dict) -> Optional[float]:
    """Prefer close, fall back to ltp."""
    return _safe_float(row.get("close")) or _safe_float(row.get("ltp"))


def _get_volume(row: dict) -> Optional[float]:
    return _safe_float(row.get("volume"))


def _calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Wilder smoothing RSI. Returns None if insufficient data."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_rolling_atr(rows: list[dict], period: int = 14) -> Optional[float]:
    """Compute ATR over the last `period` true ranges (rolling, not cumulative)."""
    if len(rows) < period + 1:
        return None
    trs = []
    for i in range(len(rows) - period, len(rows)):
        if i == 0:
            continue
        h  = _safe_float(rows[i].get("high"))
        l  = _safe_float(rows[i].get("low"))
        pc = _get_close(rows[i - 1])
        if h is None or l is None or pc is None:
            continue
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    return sum(trs) / len(trs)


def _calc_avg_rolling_atr(rows: list[dict],
                           period: int = 14,
                           lookback: int = 20) -> Optional[float]:
    """
    Average of the last `lookback` rolling ATR values.
    Used as the ATR baseline to detect compression.
    """
    if len(rows) < period + lookback:
        return None
    atr_vals = []
    for i in range(period, len(rows)):
        window = rows[i - period + 1: i + 1]
        atr = _calc_rolling_atr(window, period)
        if atr is not None:
            atr_vals.append(atr)
    if len(atr_vals) < lookback:
        return None
    return sum(atr_vals[-lookback:]) / lookback


def _calc_obv(rows: list[dict]) -> list[float]:
    """On-Balance Volume series."""
    obv = 0.0
    series = []
    prev_close = None
    for row in rows:
        close = _get_close(row)
        vol = _get_volume(row) or 0
        if prev_close is not None and close is not None:
            if close > prev_close:
                obv += vol
            elif close < prev_close:
                obv -= vol
        series.append(obv)
        prev_close = close
    return series


def _obv_trend(obv_series: list[float]) -> str:
    """
    OBV trend via linear regression slope (relative to average magnitude).
    RISING / FLAT / FALLING.
    """
    if len(obv_series) < 3:
        return "FLAT"
    n      = len(obv_series)
    x      = list(range(n))
    y      = obv_series
    x_sum  = sum(x)
    y_sum  = sum(y)
    xy_sum = sum(xi * yi for xi, yi in zip(x, y))
    x2_sum = sum(xi * xi for xi in x)
    denom  = n * x2_sum - x_sum * x_sum
    if denom == 0:
        return "FLAT"
    slope   = (n * xy_sum - x_sum * y_sum) / denom
    avg_obv = sum(abs(v) for v in y) / n
    if avg_obv == 0:
        return "FLAT"
    rel_slope = slope / avg_obv
    if rel_slope > 0.05:
        return "RISING"
    elif rel_slope < -0.05:
        return "FALLING"
    return "FLAT"


def _avg_volume(rows: list[dict], days: int = 20) -> Optional[float]:
    vols = [_get_volume(r) for r in rows[-days:] if _get_volume(r) is not None]
    if not vols:
        return None
    return sum(vols) / len(vols)


def _macd_state(closes: list[float]) -> str:
    """
    Returns "ABOVE" if MACD line > signal line, else "BELOW".
    Requires at least 35 closes for meaningful result.
    """
    if len(closes) < 35:
        return "UNKNOWN"
    k12 = 2 / 13
    k26 = 2 / 27
    k9  = 2 / 10
    ema12 = closes[0]
    ema26 = closes[0]
    macd_vals = []
    for c in closes[1:]:
        ema12 = c * k12 + ema12 * (1 - k12)
        ema26 = c * k26 + ema26 * (1 - k26)
        macd_vals.append(ema12 - ema26)
    if len(macd_vals) < 9:
        return "UNKNOWN"
    signal = macd_vals[0]
    for m in macd_vals[1:]:
        signal = m * k9 + signal * (1 - k9)
    return "ABOVE" if macd_vals[-1] > signal else "BELOW"


# ─────────────────────────────────────────────────────────────────────────────
# Trading day alignment
# ─────────────────────────────────────────────────────────────────────────────

def _get_trading_day_index(price_rows: list[dict], event_date: str) -> Optional[int]:
    """
    Return index of the trading day for event_date.
    If exact date missing (holiday/weekend), use next available trading day.
    Returns None if no date >= event_date found.
    """
    dates = [r["date"] for r in price_rows]
    if event_date in dates:
        return dates.index(event_date)
    for i, d in enumerate(dates):
        if d >= event_date:
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis — one event window
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_window(
    price_rows: list[dict],
    event_date: str,
    nepse_index: dict[str, float],
    pre_days: int = 10,
    post_days: int = 9,
) -> Optional[dict]:
    """
    Analyze one event window (announcement or book close).
    Returns dict of computed metrics, or None if insufficient data.

    price_rows : ALL rows for the symbol (wide range), sorted ASC.
    event_date : anchor date (day 0).
    nepse_index: {date: value} for alpha vs NEPSE calculation.
    """
    day0_idx = _get_trading_day_index(price_rows, event_date)
    if day0_idx is None:
        log.debug("No trading day found for event_date %s", event_date)
        return None

    # MACD needs 35 warmup + 10 pre-event days minimum
    min_required_pre = pre_days + 35
    if day0_idx < min_required_pre:
        log.debug("Insufficient pre-event data at %s (day0_idx=%d, need %d)",
                  event_date, day0_idx, min_required_pre)
        return None

    # ── Index helpers ─────────────────────────────────────────────────────────
    def idx(offset: int) -> Optional[int]:
        i = day0_idx + offset
        return i if 0 <= i < len(price_rows) else None

    def price_at(offset: int) -> Optional[float]:
        i = idx(offset)
        return _get_close(price_rows[i]) if i is not None else None

    def vol_at(offset: int) -> Optional[float]:
        i = idx(offset)
        return _get_volume(price_rows[i]) if i is not None else None

    def date_at(offset: int) -> Optional[str]:
        i = idx(offset)
        return price_rows[i]["date"] if i is not None else None

    def pct_change(base, end) -> Optional[str]:
        if base is not None and end is not None and base != 0:
            return str(round((end - base) / base * 100, 4))
        return None

    # ── Price drift ───────────────────────────────────────────────────────────
    p0   = price_at(0)
    p_10 = price_at(-10)
    p_6  = price_at(-6)
    p_4  = price_at(-4)
    p_2  = price_at(-2)

    price_drift_d10 = pct_change(p_10, p0)
    price_drift_d6  = pct_change(p_6,  p0)
    price_drift_d4  = pct_change(p_4,  p0)
    price_drift_d2  = pct_change(p_2,  p0)

    # ── Volume baseline: 20 trading days ending at day -7 ────────────────────
    # Cutoff at day -7 keeps baseline uncontaminated by accumulation window
    pre_window_end   = max(0, day0_idx - 7)
    pre_window_start = max(0, pre_window_end - 20)
    avg_vol_rows = price_rows[pre_window_start:pre_window_end]
    if len(avg_vol_rows) < 20:
        log.debug("Insufficient volume baseline for %s (only %d days before day -7)",
                  event_date, len(avg_vol_rows))
        return None
    avg_vol = _avg_volume(avg_vol_rows, days=20)
    if avg_vol is None or avg_vol == 0:
        log.debug("Zero average volume for %s", event_date)
        return None

    def vol_ratio(offset: int) -> Optional[str]:
        v = vol_at(offset)
        if v is not None and avg_vol > 0:
            return str(round(v / avg_vol, 4))
        return None

    vol_ratio_d6 = vol_ratio(-6)
    vol_ratio_d4 = vol_ratio(-4)
    vol_ratio_d2 = vol_ratio(-2)
    vol_ratio_d1 = vol_ratio(-1)

    # ── RSI at day -6 and -4 ─────────────────────────────────────────────────
    def rsi_at(offset: int) -> Optional[str]:
        i = idx(offset)
        if i is None or i < 15:
            return None
        closes = [_get_close(r) for r in price_rows[:i + 1]]
        closes = [c for c in closes if c is not None]
        if len(closes) < 15:
            return None
        rsi = _calc_rsi(closes, period=14)
        return str(round(rsi, 2)) if rsi is not None else None

    rsi_d6 = rsi_at(-6)
    rsi_d4 = rsi_at(-4)

    # ── OBV trend (day -6 to day -1) — linear regression slope ──────────────
    i_d6 = idx(-6)
    i_d1 = idx(-1)
    obv_trend_val = "FLAT"
    if i_d6 is not None and i_d1 is not None and i_d6 < i_d1:
        obv_window    = price_rows[i_d6:i_d1 + 1]
        obv_series    = _calc_obv(obv_window)
        obv_trend_val = _obv_trend(obv_series)

    # ── ATR compression at day -6 (rolling) ──────────────────────────────────
    atr_ratio_d6 = None
    i_d6_idx = idx(-6)
    if i_d6_idx is not None and i_d6_idx >= 14:
        rows_up_to = price_rows[:i_d6_idx + 1]
        atr_now    = _calc_rolling_atr(rows_up_to, period=14)
        atr_avg    = _calc_avg_rolling_atr(rows_up_to, period=14, lookback=20)
        if atr_now is not None and atr_avg is not None and atr_avg > 0:
            atr_ratio_d6 = str(round(atr_now / atr_avg, 4))

    # ── MACD state at day -4 ──────────────────────────────────────────────────
    macd_state_d4 = "UNKNOWN"
    i_d4_idx = idx(-4)
    if i_d4_idx is not None and i_d4_idx >= 34:
        closes = [_get_close(r) for r in price_rows[:i_d4_idx + 1]]
        closes = [c for c in closes if c is not None]
        if len(closes) >= 35:
            macd_state_d4 = _macd_state(closes)

    # ── Post-event returns ────────────────────────────────────────────────────
    return_d0    = pct_change(price_at(-1), p0)
    return_d1    = pct_change(p0, price_at(1))
    p_d1         = price_at(1)
    p_d9         = price_at(9)
    return_d2_d9 = pct_change(p_d1, p_d9) if p_d1 and p_d9 else None

    # ── Alpha vs NEPSE (day 0 → day +1) ──────────────────────────────────────
    vs_nepse_d1 = None
    d0_date = date_at(0)
    d1_date = date_at(1)
    if (d0_date and d1_date
            and d0_date in nepse_index
            and d1_date in nepse_index
            and nepse_index[d0_date] != 0):
        nepse_ret = ((nepse_index[d1_date] - nepse_index[d0_date])
                     / nepse_index[d0_date] * 100)
        if return_d1 is not None:
            vs_nepse_d1 = str(round(float(return_d1) - nepse_ret, 4))

    # ── Pattern classification ────────────────────────────────────────────────
    # 4 signals — pattern fires when 3+ present
    signals_fired = 0
    vr_d2_val  = _safe_float(vol_ratio_d2)
    atr_val    = _safe_float(atr_ratio_d6)
    rsi_d4_val = _safe_float(rsi_d4)

    if vr_d2_val is not None and vr_d2_val > 1.3:          # volume building
        signals_fired += 1
    if obv_trend_val == "RISING":                           # OBV leading price
        signals_fired += 1
    if atr_val is not None and atr_val < 1.0:              # ATR compressing
        signals_fired += 1
    if rsi_d4_val is not None and 45 <= rsi_d4_val <= 65:  # RSI neutral-rising
        signals_fired += 1

    pattern_detected = signals_fired >= 3
    if signals_fired == 4:
        pattern_strength = "STRONG"
    elif signals_fired == 3:
        pattern_strength = "WEAK"
    else:
        pattern_strength = "NONE"

    # T+3: must buy by day -4 to be in demat by day -1 (eligible for dividend)
    entry_feasible = price_at(-4) is not None

    # Optimal entry: lowest close in day -6 to day -3 window
    best_day   = None
    best_price = None
    for offset in range(-6, -2):
        p = price_at(offset)
        if p is not None and (best_price is None or p < best_price):
            best_price = p
            best_day   = str(offset)

    return {
        "price_drift_d10":    price_drift_d10,
        "price_drift_d6":     price_drift_d6,
        "price_drift_d4":     price_drift_d4,
        "price_drift_d2":     price_drift_d2,
        "vol_ratio_d6":       vol_ratio_d6,
        "vol_ratio_d4":       vol_ratio_d4,
        "vol_ratio_d2":       vol_ratio_d2,
        "vol_ratio_d1":       vol_ratio_d1,
        "rsi_d6":             rsi_d6,
        "rsi_d4":             rsi_d4,
        "obv_trend_d6_to_d1": obv_trend_val,
        "atr_ratio_d6":       atr_ratio_d6,
        "macd_state_d4":      macd_state_d4,
        "return_d0":          return_d0,
        "return_d1":          return_d1,
        "return_d2_d9":       return_d2_d9,
        "vs_nepse_d1":        vs_nepse_d1,
        "pattern_detected":   "true" if pattern_detected else "false",
        "pattern_type":       "ACCUMULATION" if pattern_detected else "NONE",
        "pattern_strength":   pattern_strength,
        "entry_feasible":     "true" if entry_feasible else "false",
        "optimal_entry_day":  best_day,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]):
    if not results:
        print("\nNo results to summarize.")
        return

    def avg(vals):
        v = [float(x) for x in vals if x is not None and x != ""]
        return round(sum(v) / len(v), 4) if v else None

    def pct_true(vals):
        v = [x == "true" for x in vals if x is not None]
        return round(sum(v) / len(v) * 100, 1) if v else None

    for event_type in ["ANNOUNCEMENT", "BOOK_CLOSE"]:
        rows = [r for r in results if r.get("event_type") == event_type]
        if not rows:
            continue

        print(f"\n{'='*65}")
        print(f"  {event_type} WINDOW  (n={len(rows)})")
        print(f"{'='*65}")

        # ── Overall pattern rate ──────────────────────────────────────────────
        pattern_pct = pct_true([r.get("pattern_detected") for r in rows])
        print(f"\nPattern Detection Rate (overall):  {pattern_pct}%")
        print(f"  → 70%+ = build detector | 50-69% = tiebreaker | <50% = negative seed")

        # ── Season split — KEY METRIC ─────────────────────────────────────────
        in_season  = [r for r in rows if r.get("in_dividend_season") == "true"]
        off_season = [r for r in rows if r.get("in_dividend_season") == "false"]
        season_pct     = pct_true([r.get("pattern_detected") for r in in_season])
        off_season_pct = pct_true([r.get("pattern_detected") for r in off_season])
        print(f"\nSeasonality Split:")
        print(f"  IN season  (Mar-Jul) n={len(in_season):3d}:  pattern={season_pct}%")
        print(f"  OFF season           n={len(off_season):3d}:  pattern={off_season_pct}%")
        print(f"  ↳ Off-season rate is the cleaner signal (no seasonal bull tailwind)")
        if season_pct and off_season_pct:
            gap = round(season_pct - off_season_pct, 1)
            if gap > 15:
                print(f"  ⚠  Large gap ({gap}pp) — in-season pattern inflated by seasonal noise")
            elif gap < 5:
                print(f"  ✓  Small gap ({gap}pp) — pattern genuine, not seasonal artifact")
            else:
                print(f"  ~  Moderate gap ({gap}pp) — partial seasonal contribution")

        # ── Day +1 returns ────────────────────────────────────────────────────
        detected     = [r for r in rows if r.get("pattern_detected") == "true"]
        not_detected = [r for r in rows if r.get("pattern_detected") == "false"]
        print(f"\nDay +1 Return:")
        print(f"  All:               {avg([r.get('return_d1') for r in rows])}%")
        print(f"  Pattern fired:     {avg([r.get('return_d1') for r in detected])}%")
        print(f"  Pattern not fired: {avg([r.get('return_d1') for r in not_detected])}%")
        print(f"  IN season:         {avg([r.get('return_d1') for r in in_season])}%")
        print(f"  OFF season:        {avg([r.get('return_d1') for r in off_season])}%")

        # ── Indicator averages ────────────────────────────────────────────────
        print(f"\nIndicator Averages:")
        print(f"  Price drift day -6 to 0:    {avg([r.get('price_drift_d6') for r in rows])}%")
        print(f"  Vol ratio day -2:            {avg([r.get('vol_ratio_d2') for r in rows])}x")
        print(f"  RSI at day -4:               {avg([r.get('rsi_d4') for r in rows])}")
        print(f"  OBV RISING rate:             {pct_true(['true' if r.get('obv_trend_d6_to_d1') == 'RISING' else 'false' for r in rows])}%")
        print(f"  ATR compressed (<1.0) rate:  {pct_true(['true' if _safe_float(r.get('atr_ratio_d6'), 99) < 1.0 else 'false' for r in rows])}%")
        print(f"  MACD ABOVE signal rate:      {pct_true(['true' if r.get('macd_state_d4') == 'ABOVE' else 'false' for r in rows])}%")

        # ── Post-event ────────────────────────────────────────────────────────
        print(f"\nPost-Event:")
        print(f"  Cumulative day +2 to +9:     {avg([r.get('return_d2_d9') for r in rows])}%")
        print(f"  Alpha vs NEPSE day +1:       {avg([r.get('vs_nepse_d1') for r in rows])}%")
        print(f"  Entry feasible (T+3):        {pct_true([r.get('entry_feasible') for r in rows])}%")

        # ── Pattern strength breakdown ────────────────────────────────────────
        print(f"\nPattern Strength:")
        for strength in ["STRONG", "WEAK", "NONE"]:
            sub = [r for r in rows if r.get("pattern_strength") == strength]
            d1  = avg([r.get("return_d1") for r in sub])
            print(f"  {strength:6s} n={len(sub):3d} | avg_d1={d1}%")

        # ── By dividend type ──────────────────────────────────────────────────
        print(f"\nBy Dividend Type:")
        for dtype in ["CASH", "BONUS", "BOTH"]:
            sub = [r for r in rows if r.get("dividend_type") == dtype]
            if sub:
                print(f"  {dtype:5s} n={len(sub):3d} | "
                      f"pattern={pct_true([r.get('pattern_detected') for r in sub])}% | "
                      f"d1={avg([r.get('return_d1') for r in sub])}%")

        # ── By direction ──────────────────────────────────────────────────────
        print(f"\nBy Direction:")
        for direction in ["INCREASE", "DECREASE", "UNCHANGED", ""]:
            label = direction if direction else "UNKNOWN"
            sub   = [r for r in rows if r.get("direction") == direction]
            if sub:
                print(f"  {label:10s} n={len(sub):3d} | "
                      f"pattern={pct_true([r.get('pattern_detected') for r in sub])}% | "
                      f"d1={avg([r.get('return_d1') for r in sub])}%")

        # ── By sector (n >= 5) ────────────────────────────────────────────────
        sector_groups = defaultdict(list)
        for r in rows:
            sector_groups[r.get("sector") or "UNKNOWN"].append(r)
        print(f"\nBy Sector (n≥5):")
        for sector, sub in sorted(sector_groups.items(), key=lambda x: -len(x[1])):
            if len(sub) >= 5:
                print(f"  {sector:28s} n={len(sub):3d} | "
                      f"pattern={pct_true([r.get('pattern_detected') for r in sub])}% | "
                      f"d1={avg([r.get('return_d1') for r in sub])}%")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("VERDICT GUIDE (use OFF-SEASON rate as primary decision metric):")
    print("  Off-season pattern rate >= 70% → build dividend_detector.py")
    print("  Off-season pattern rate 50-69% → tiebreaker in claude_analyst")
    print("  Off-season pattern rate  < 50% → seed as negative finding in Learning Hub")
    print(f"{'='*65}\n")


def _print_db_summary():
    """Print summary from existing dividend_pattern_study rows (--summary-only mode)."""
    from db.connection import _db
    try:
        with _db() as cur:
            cur.execute("""
                SELECT
                    event_type,
                    in_dividend_season,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pattern_detected = 'true' THEN 1 ELSE 0 END) AS detected,
                    AVG(return_d1::numeric)    AS avg_d1,
                    AVG(return_d2_d9::numeric) AS avg_d2_d9
                FROM dividend_pattern_study
                WHERE return_d1 IS NOT NULL AND return_d1 != ''
                GROUP BY event_type, in_dividend_season
                ORDER BY event_type, in_dividend_season
            """)
            rows = cur.fetchall()
            print("\n=== dividend_pattern_study DB Summary ===")
            for r in rows:
                total    = r["total"]
                detected = r["detected"]
                pct      = round(detected / total * 100, 1) if total else 0
                season   = "IN-season" if r["in_dividend_season"] == "true" else "OFF-season"
                print(f"\n{r['event_type']} | {season}:")
                print(f"  Total rows:    {total}")
                print(f"  Pattern rate:  {pct}% ({detected}/{total})")
                print(f"  Avg Day +1:    {round(float(r['avg_d1']), 4) if r['avg_d1'] else 'N/A'}%")
                print(f"  Avg Day +2-+9: {round(float(r['avg_d2_d9']), 4) if r['avg_d2_d9'] else 'N/A'}%")
    except Exception as e:
        log.error("_print_db_summary failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run(symbol: str = None,
        from_date: str = None,
        to_date: str = None,
        dry_run: bool = False) -> list[dict]:
    """
    Main study runner.
    Returns list of all result dicts written (or computed in dry_run).
    """
    announcements = _fetch_announcements(symbol=symbol,
                                         from_date=from_date,
                                         to_date=to_date)
    log.info("Announcements to analyze: %d", len(announcements))

    if not announcements:
        log.warning("No announcements found with both dates. Exiting.")
        return []

    # Fetch NEPSE index for full date range (wide buffer)
    all_dates     = ([a["announcement_date"] for a in announcements]
                     + [a["book_close_date"] for a in announcements])
    nepse_from_dt = datetime.strptime(min(all_dates), "%Y-%m-%d") - timedelta(days=5)
    nepse_to_dt   = datetime.strptime(max(all_dates), "%Y-%m-%d") + timedelta(days=15)
    nepse_index   = _fetch_nepse_index(
        nepse_from_dt.strftime("%Y-%m-%d"),
        nepse_to_dt.strftime("%Y-%m-%d"),
    )
    log.info("NEPSE index rows loaded: %d", len(nepse_index))

    all_results = []
    written     = 0
    skipped     = 0

    for ann in announcements:
        sym           = ann["symbol"]
        ann_date      = ann["announcement_date"]
        book_date     = ann["book_close_date"]
        fiscal_year   = ann.get("fiscal_year", "")
        dividend_type = ann.get("dividend_type", "")
        direction     = ann.get("direction", "")
        sector        = ann.get("sector", "")

        # Wide fetch: 90 days before announcement + 15 days after book close
        earliest   = ann_date
        latest     = book_date if book_date > ann_date else ann_date
        fetch_from = (datetime.strptime(earliest, "%Y-%m-%d")
                      - timedelta(days=90)).strftime("%Y-%m-%d")
        fetch_to   = (datetime.strptime(latest, "%Y-%m-%d")
                      + timedelta(days=15)).strftime("%Y-%m-%d")

        price_rows = _fetch_price_history(sym, fetch_from, fetch_to)
        if len(price_rows) < 20:
            log.debug("Skipping %s %s — insufficient price history (%d rows)",
                      sym, ann_date, len(price_rows))
            skipped += 1
            continue

        run_date = datetime.now(NST).strftime("%Y-%m-%d")

        for event_type, event_date in [("ANNOUNCEMENT", ann_date),
                                        ("BOOK_CLOSE",   book_date)]:
            metrics = _analyze_window(price_rows, event_date, nepse_index)
            if metrics is None:
                log.debug("Skipping %s %s %s — insufficient window data",
                          sym, event_type, event_date)
                skipped += 1
                continue

            row = {
                "symbol":             sym,
                "announcement_date":  ann_date,
                "event_type":         event_type,
                "sector":             sector,
                "direction":          direction,
                "dividend_type":      dividend_type,
                "fiscal_year":        fiscal_year,
                "in_dividend_season": _is_dividend_season(event_date),
                "run_date":           run_date,
                **metrics,
            }

            all_results.append(row)

            if dry_run:
                log.info("[DRY-RUN] %s | %s | %s | season=%s | pattern=%s | d1=%s%%",
                         sym, event_type, event_date,
                         row["in_dividend_season"],
                         metrics["pattern_detected"],
                         metrics.get("return_d1", "N/A"))
            else:
                if _upsert_pattern(row):
                    written += 1
                else:
                    skipped += 1

    log.info("Complete | analyzed=%d written=%d skipped=%d dry_run=%s",
             len(all_results), written, skipped, dry_run)
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Dividend Pattern Study — validates Adhikari (2023) findings"
    )
    parser.add_argument("--symbol",       help="Analyze one symbol only e.g. NABIL")
    parser.add_argument("--from",         dest="from_date", help="Start date YYYY-MM-DD")
    parser.add_argument("--to",           dest="to_date",   help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Compute only, no DB write")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print DB stats only, skip compute")
    args = parser.parse_args()

    if args.summary_only:
        _print_db_summary()
        sys.exit(0)

    results = run(
        symbol=args.symbol,
        from_date=args.from_date,
        to_date=args.to_date,
        dry_run=args.dry_run,
    )

    _print_summary(results)
    sys.exit(0)


if __name__ == "__main__":
    main()