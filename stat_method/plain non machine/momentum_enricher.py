"""
momentum_enricher.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Momentum Enricher
Takes momentum_hits.json from momentum_screener.py and enriches each hit
with all available data:

  - Full technicals at trigger date (RSI-14, EMA 20/50/200, MACD 12/26/9,
    Bollinger Bands 20, ATR-14, OBV trend)
  - Indicator trajectory: snapshots every 3 trading days from day -15 to
    day 0 (checkpoints: -15, -12, -9, -6, -3, 0) — each with close,
    volume, RSI, MACD histogram, BB %B, OBV trend
  - Floorsheet accumulation signals on trigger date (buyer pressure, broker
    concentration, institutional flag, large order %)
  - Pre-move floorsheet: avg buyer pressure + institutional days over the
    15 trading days before trigger
  - Fundamentals (EPS, ROE, PE, NPL) with lag>=1 quarter
  - Sector from share_sectors
  - Forward returns: +5, +10, +20 trading days from trigger date

Output: stat_method/output/momentum_enriched.json

Usage:
    cd ~/nepse-engine
    python stat_method/momentum_enricher.py
    python stat_method/momentum_enricher.py --limit 50   # test with 50 hits
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ENRICHER] %(message)s",
)
log = logging.getLogger(__name__)

HISTORY_WINDOW = 200   # trading days of OHLCV to load per symbol for indicators


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR MATH — mirrors indicators.py logic exactly
# ══════════════════════════════════════════════════════════════════════════════

def _calc_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    # Wilder's smoothing — seed with simple average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_ema(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2 / (period + 1)
    emas = [sum(closes[:period]) / period]
    for price in closes[period:]:
        emas.append(price * k + emas[-1] * (1 - k))
    return emas


def _calc_macd(closes: list[float]) -> dict | None:
    ema12 = _calc_ema(closes, 12)
    ema26 = _calc_ema(closes, 26)
    if not ema12 or not ema26:
        return None
    # align — ema26 is shorter
    diff = len(ema12) - len(ema26)
    ema12_aligned = ema12[diff:]
    macd_line = [e12 - e26 for e12, e26 in zip(ema12_aligned, ema26)]
    if len(macd_line) < 9:
        return None
    signal_line = _calc_ema(macd_line, 9)
    if not signal_line:
        return None
    macd_val    = macd_line[-1]
    signal_val  = signal_line[-1]
    hist        = macd_val - signal_val
    # Cross detection: previous histogram sign vs current
    prev_hist   = macd_line[-2] - signal_line[-2] if len(signal_line) >= 2 else 0
    if prev_hist < 0 and hist >= 0:
        cross = "BULLISH"
    elif prev_hist > 0 and hist <= 0:
        cross = "BEARISH"
    else:
        cross = "NONE"
    return {
        "macd_line":      round(macd_val, 4),
        "macd_signal":    round(signal_val, 4),
        "macd_histogram": round(hist, 4),
        "macd_cross":     cross,
    }


def _calc_bollinger(closes: list[float], period: int = 20) -> dict | None:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid    = sum(window) / period
    std    = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
    upper  = mid + 2 * std
    lower  = mid - 2 * std
    width  = (upper - lower) / mid if mid > 0 else 0
    pct_b  = (closes[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    if pct_b > 1.0:
        signal = "UPPER_TOUCH"
    elif pct_b < 0.0:
        signal = "LOWER_TOUCH"
    elif width < 0.05:
        signal = "SQUEEZE"
    else:
        signal = "NEUTRAL"
    return {
        "bb_upper":  round(upper, 2),
        "bb_middle": round(mid, 2),
        "bb_lower":  round(lower, 2),
        "bb_width":  round(width, 4),
        "bb_pct_b":  round(pct_b, 4),
        "bb_signal": signal,
    }


def _calc_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    # Wilder's ATR
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


def _calc_obv_trend(closes: list[float], volumes: list[float], lookback: int = 10) -> str:
    """OBV over last lookback+1 bars, linear slope to determine trend direction."""
    if len(closes) < lookback + 1 or len(volumes) < lookback + 1:
        return "UNKNOWN"
    c = closes[-(lookback + 1):]
    v = volumes[-(lookback + 1):]
    obv = [0.0]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            obv.append(obv[-1] + v[i])
        elif c[i] < c[i - 1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    n   = len(obv)
    x_m = (n - 1) / 2
    y_m = sum(obv) / n
    num = sum((i - x_m) * (obv[i] - y_m) for i in range(n))
    den = sum((i - x_m) ** 2 for i in range(n))
    if den == 0:
        return "FLAT"
    slope = num / den
    if slope > 0:
        return "RISING"
    elif slope < 0:
        return "FALLING"
    return "FLAT"


def _calc_obv_value(closes: list[float], volumes: list[float]) -> float | None:
    """Cumulative OBV value at the last bar."""
    if len(closes) < 2:
        return None
    obv = 0.0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    return round(obv, 0)


def _snapshot_at(
    dates: list[str],
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    target_idx: int,
) -> dict:
    """
    Compute a lightweight indicator snapshot at a specific index.
    Returns: close, volume, rsi, macd_histogram, bb_pct_b, obv_trend.
    """
    if target_idx < 1:
        return {}
    c = closes[:target_idx + 1]
    h = highs[:target_idx + 1]
    l = lows[:target_idx + 1]
    v = volumes[:target_idx + 1]

    if len(c) < 20:
        return {
            "close":  round(c[-1], 2),
            "volume": round(v[-1], 0),
        }

    rsi = _calc_rsi(c)
    bb  = _calc_bollinger(c)
    macd = _calc_macd(c)
    obv_trend = _calc_obv_trend(c, v)

    return {
        "date":           dates[target_idx],
        "close":          round(c[-1], 2),
        "volume":         round(v[-1], 0),
        "rsi_14":         rsi,
        "macd_histogram": macd["macd_histogram"] if macd else None,
        "bb_pct_b":       bb["bb_pct_b"]         if bb   else None,
        "obv_trend":      obv_trend,
    }


def compute_trajectory(
    dates:   list[str],
    closes:  list[float],
    highs:   list[float],
    lows:    list[float],
    volumes: list[float],
    trigger_date: str,
    interval: int = 3,
) -> list[dict]:
    """
    Compute indicator snapshots at every `interval` trading days from
    day -15 through day 0 (trigger_date), inclusive.
    Checkpoints: -15, -12, -9, -6, -3, 0  (6 snapshots at interval=3)

    Returns list of snapshot dicts ordered oldest → newest.
    Each snapshot: {date, day_offset, close, volume, rsi_14,
                    macd_histogram, bb_pct_b, obv_trend}
    """
    try:
        idx0 = dates.index(trigger_date)
    except ValueError:
        idx0 = max((i for i, d in enumerate(dates) if d <= trigger_date), default=-1)
    if idx0 < 0:
        return []

    LOOKBACK = 15
    offsets = list(range(-LOOKBACK, 1, interval))   # [-15,-12,-9,-6,-3, 0]

    snapshots = []
    for offset in offsets:
        target_idx = idx0 + offset
        if target_idx < 0:
            continue
        snap = _snapshot_at(dates, closes, highs, lows, volumes, target_idx)
        if snap:
            snap["day_offset"] = offset
            # Reorder for readability
            ordered = {"day_offset": snap.pop("day_offset"), **snap}
            snapshots.append(ordered)

    return snapshots


def compute_technicals(
    dates:   list[str],
    closes:  list[float],
    highs:   list[float],
    lows:    list[float],
    volumes: list[float],
    trigger_date: str,
) -> dict:
    """
    Compute all indicators up to and including trigger_date.
    Uses only data up to trigger_date (no lookahead).
    """
    # Slice to trigger_date inclusive
    try:
        idx = dates.index(trigger_date)
    except ValueError:
        # Find nearest earlier date
        idx = max((i for i, d in enumerate(dates) if d <= trigger_date), default=-1)
    if idx < 0:
        return {}

    c = closes[:idx + 1]
    h = highs[:idx + 1]
    l = lows[:idx + 1]
    v = volumes[:idx + 1]

    if len(c) < 20:
        return {}

    ema20_list = _calc_ema(c, 20)
    ema50_list = _calc_ema(c, 50)
    ema200_list = _calc_ema(c, 200)

    ema20  = ema20_list[-1]  if ema20_list  else None
    ema50  = ema50_list[-1]  if ema50_list  else None
    ema200 = ema200_list[-1] if ema200_list else None

    close_now = c[-1]

    # EMA trend
    if ema20 and ema50 and ema200:
        if close_now > ema20 > ema50 > ema200:
            ema_trend = "ABOVE_ALL"
        elif close_now < ema20 < ema50 < ema200:
            ema_trend = "BELOW_ALL"
        else:
            ema_trend = "MIXED"
    else:
        ema_trend = "INSUFFICIENT"

    # EMA crosses (last two values of aligned EMAs)
    def _last_cross(short_list, long_list, label_golden, label_death):
        if len(short_list) < 2 or len(long_list) < 2:
            return "NONE"
        diff = len(short_list) - len(long_list)
        s = short_list[diff:] if diff >= 0 else short_list
        lo = long_list[-diff:] if diff < 0 else long_list
        n = min(len(s), len(lo))
        if n < 2:
            return "NONE"
        if s[-2] <= lo[-2] and s[-1] > lo[-1]:
            return label_golden
        if s[-2] >= lo[-2] and s[-1] < lo[-1]:
            return label_death
        return "NONE"

    ema_20_50_cross  = _last_cross(ema20_list, ema50_list,  "GOLDEN", "DEATH")
    ema_50_200_cross = _last_cross(ema50_list, ema200_list, "GOLDEN", "DEATH")

    rsi   = _calc_rsi(c)
    rsi_signal = (
        "OVERSOLD"   if rsi is not None and rsi < 35 else
        "OVERBOUGHT" if rsi is not None and rsi > 65 else
        "NEUTRAL"
    )

    macd = _calc_macd(c)
    bb   = _calc_bollinger(c)
    atr  = _calc_atr(h, l, c)
    atr_pct = round(atr / close_now * 100, 2) if atr and close_now > 0 else None
    obv_trend = _calc_obv_trend(c, v)

    # Support / resistance: 20-day high/low
    support    = min(l[-20:]) if len(l) >= 20 else min(l)
    resistance = max(h[-20:]) if len(h) >= 20 else max(h)

    result = {
        "rsi_14":          rsi,
        "rsi_signal":      rsi_signal,
        "ema_20":          round(ema20,  2) if ema20  else None,
        "ema_50":          round(ema50,  2) if ema50  else None,
        "ema_200":         round(ema200, 2) if ema200 else None,
        "ema_trend":       ema_trend,
        "ema_20_50_cross": ema_20_50_cross,
        "ema_50_200_cross":ema_50_200_cross,
        "atr_14":          atr,
        "atr_pct":         atr_pct,
        "obv_trend":       obv_trend,
        "support_level":   round(support,    2),
        "resistance_level":round(resistance, 2),
    }

    if macd:
        result.update(macd)
    if bb:
        result.update(bb)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_full_ohlcv(symbols: set[str]) -> dict[str, dict]:
    """
    Load full OHLCV history for given symbols (from 2019 for indicator warmup).
    Returns: symbol → {dates, closes, highs, lows, volumes}
    """
    log.info("Loading full OHLCV for %d symbols...", len(symbols))
    sym_list = list(symbols)

    with _db() as cur:
        cur.execute("""
            SELECT symbol, date, close, high, low, volume
            FROM price_history
            WHERE symbol = ANY(%s)
              AND close  IS NOT NULL AND close  != '' AND close  ~ '^[0-9]+\\.?[0-9]*$'
              AND high   IS NOT NULL AND high   != '' AND high   ~ '^[0-9]+\\.?[0-9]*$'
              AND low    IS NOT NULL AND low    != '' AND low    ~ '^[0-9]+\\.?[0-9]*$'
              AND volume IS NOT NULL AND volume != '' AND volume ~ '^[0-9]+\\.?[0-9]*$'
            ORDER BY symbol, date ASC
        """, (sym_list,))
        rows = cur.fetchall()

    data: dict[str, dict] = defaultdict(lambda: {
        "dates": [], "closes": [], "highs": [], "lows": [], "volumes": []
    })

    for r in rows:
        sym = str(r["symbol"]).upper()
        try:
            data[sym]["dates"].append(r["date"])
            data[sym]["closes"].append(float(r["close"]))
            data[sym]["highs"].append(float(r["high"]))
            data[sym]["lows"].append(float(r["low"]))
            data[sym]["volumes"].append(float(r["volume"]))
        except (ValueError, TypeError):
            continue

    log.info("OHLCV loaded for %d symbols", len(data))
    return dict(data)


def load_floorsheet_signals(symbols: set[str]) -> dict[str, dict]:
    """
    Load floorsheet_signals for given symbols.
    Returns: (symbol, date) → signal dict
    """
    log.info("Loading floorsheet_signals...")
    sym_list = list(symbols)

    with _db() as cur:
        cur.execute("""
            SELECT
                symbol, date,
                total_trades, total_volume, total_turnover,
                avg_trade_size, large_order_count, large_order_pct,
                buyer_pressure, seller_pressure,
                broker_concentration, institutional_flag,
                vwap
            FROM floorsheet_signals
            WHERE symbol = ANY(%s)
              AND date >= '2025-07-01'
            ORDER BY symbol, date ASC
        """, (sym_list,))
        rows = cur.fetchall()

    result = {}
    for r in rows:
        key = (str(r["symbol"]).upper(), r["date"])
        result[key] = {
            "fs_total_trades":       _safe_float(r["total_trades"]),
            "fs_total_volume":       _safe_float(r["total_volume"]),
            "fs_large_order_pct":    _safe_float(r["large_order_pct"]),
            "fs_buyer_pressure":     _safe_float(r["buyer_pressure"]),
            "fs_seller_pressure":    _safe_float(r["seller_pressure"]),
            "fs_broker_concentration": _safe_float(r["broker_concentration"]),
            "fs_institutional_flag": str(r["institutional_flag"]).lower() if r["institutional_flag"] else None,
            "fs_vwap":               _safe_float(r["vwap"]),
        }

    log.info("Floorsheet signals: %d (symbol, date) pairs", len(result))
    return result


def load_fundamentals(symbols: set[str]) -> dict[str, dict]:
    """
    Load latest fundamentals per symbol with lag>=1 quarter.
    Returns: symbol → fundamentals dict
    """
    log.info("Loading fundamentals...")
    sym_list = list(symbols)

    # Get most recent quarter before 2025-07-01 (strict lag)
    with _db() as cur:
        cur.execute("""
            SELECT DISTINCT ON (symbol)
                symbol, fiscal_year, quarter,
                eps, roe, roa, pe_ratio,
                npl, net_profit, paidup_capital,
                cd_ratio, cost_of_fund, base_rate
            FROM fundamentals
            WHERE symbol = ANY(%s)
              AND scraped_at < '2025-07-01'
            ORDER BY symbol, fiscal_year DESC, quarter DESC
        """, (sym_list,))
        rows = cur.fetchall()

    result = {}
    for r in rows:
        sym = str(r["symbol"]).upper()
        result[sym] = {
            "fund_fiscal_year":  r["fiscal_year"],
            "fund_quarter":      r["quarter"],
            "fund_eps":          _safe_float(r["eps"]),
            "fund_roe":          _safe_float(r["roe"]),
            "fund_roa":          _safe_float(r["roa"]),
            "fund_pe_ratio":     _safe_float(r["pe_ratio"]),
            "fund_npl":          _safe_float(r["npl"]),
            "fund_net_profit":   _safe_float(r["net_profit"]),
            "fund_paidup_capital": _safe_float(r["paidup_capital"]),
            "fund_cd_ratio":     _safe_float(r["cd_ratio"]),
        }

    log.info("Fundamentals loaded for %d symbols", len(result))
    return result


def load_sectors(symbols: set[str]) -> dict[str, str]:
    """Load symbol → sector mapping."""
    sym_list = list(symbols)
    with _db() as cur:
        cur.execute("""
            SELECT symbol, sectorname
            FROM share_sectors
            WHERE symbol = ANY(%s)
        """, (sym_list,))
        rows = cur.fetchall()
    return {str(r["symbol"]).upper(): r["sectorname"] for r in rows}


def compute_forward_returns(
    ohlcv: dict,
    symbol: str,
    trigger_date: str,
    windows: list[int] = [5, 10, 20],
) -> dict:
    """
    Compute forward returns from trigger_date close for given windows.
    Returns dict: fwd_5d_pct, fwd_10d_pct, fwd_20d_pct
    """
    if symbol not in ohlcv:
        return {}

    dates  = ohlcv[symbol]["dates"]
    closes = ohlcv[symbol]["closes"]

    try:
        idx0 = dates.index(trigger_date)
    except ValueError:
        idx0 = max((i for i, d in enumerate(dates) if d <= trigger_date), default=-1)

    if idx0 < 0 or closes[idx0] <= 0:
        return {}

    base = closes[idx0]
    result = {}
    for w in windows:
        fwd_idx = idx0 + w
        if fwd_idx < len(closes):
            result[f"fwd_{w}d_pct"] = round((closes[fwd_idx] - base) / base * 100, 2)
            result[f"fwd_{w}d_date"] = dates[fwd_idx]
        else:
            result[f"fwd_{w}d_pct"]  = None  # not enough future data
            result[f"fwd_{w}d_date"] = None
    return result


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return round(f, 4) if f == f else None  # NaN check
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENRICHMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def enrich(hits: list[dict], limit: int | None = None) -> list[dict]:
    if limit:
        hits = hits[:limit]

    symbols = set(h["symbol"] for h in hits)
    log.info("Enriching %d hits across %d symbols", len(hits), len(symbols))

    # Load all supporting data once
    ohlcv        = load_full_ohlcv(symbols)
    floorsheet   = load_floorsheet_signals(symbols)
    fundamentals = load_fundamentals(symbols)
    sectors      = load_sectors(symbols)

    enriched = []
    for i, hit in enumerate(hits):
        sym          = hit["symbol"]
        trigger_date = hit["trigger_date"]

        if (i + 1) % 100 == 0:
            log.info("  Enriching %d / %d...", i + 1, len(hits))

        row = dict(hit)

        # Sector
        row["sector"] = sectors.get(sym)

        # Technicals at trigger date (full snapshot)
        if sym in ohlcv:
            tech = compute_technicals(
                ohlcv[sym]["dates"],
                ohlcv[sym]["closes"],
                ohlcv[sym]["highs"],
                ohlcv[sym]["lows"],
                ohlcv[sym]["volumes"],
                trigger_date,
            )
            row.update(tech)

            # Trajectory: indicator snapshots every 3 days from day -15 to day 0
            row["trajectory"] = compute_trajectory(
                ohlcv[sym]["dates"],
                ohlcv[sym]["closes"],
                ohlcv[sym]["highs"],
                ohlcv[sym]["lows"],
                ohlcv[sym]["volumes"],
                trigger_date,
                interval=3,
            )

        # Floorsheet signals for trigger_date
        fs = floorsheet.get((sym, trigger_date), {})
        row.update(fs)

        # Pre-move floorsheet: last 15 trading days before trigger
        sym_dates = ohlcv.get(sym, {}).get("dates", [])
        try:
            t_idx = sym_dates.index(trigger_date)
        except ValueError:
            t_idx = max((i for i, d in enumerate(sym_dates) if d <= trigger_date), default=-1)

        pre_window_dates = sym_dates[max(0, t_idx - 15):t_idx] if t_idx > 0 else []

        pre_bp   = []
        pre_inst = 0
        for d in pre_window_dates:
            fs_pre = floorsheet.get((sym, d), {})
            if not fs_pre:
                continue
            bp = fs_pre.get("fs_buyer_pressure")
            if bp is not None:
                pre_bp.append(bp)
            if fs_pre.get("fs_institutional_flag") == "true":
                pre_inst += 1

        if pre_bp:
            row["pre_move_avg_buyer_pressure"]   = round(sum(pre_bp) / len(pre_bp), 4)
            row["pre_move_institutional_days"]   = pre_inst
            row["pre_move_floorsheet_days_avail"] = len(pre_bp)

        # Fundamentals
        fund = fundamentals.get(sym, {})
        row.update(fund)

        # Forward returns
        fwd = compute_forward_returns(ohlcv, sym, trigger_date)
        row.update(fwd)

        enriched.append(row)

    log.info("Enrichment complete: %d rows", len(enriched))
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit hits for testing")
    args = parser.parse_args()

    in_path = Path(__file__).parent / "output" / "momentum_hits.json"
    if not in_path.exists():
        log.error("momentum_hits.json not found. Run momentum_screener.py first.")
        sys.exit(1)

    with open(in_path) as f:
        data = json.load(f)

    hits     = data["hits"]
    meta     = data["meta"]
    log.info("Loaded %d hits from screener", len(hits))

    enriched = enrich(hits, limit=args.limit)

    out_dir  = Path(__file__).parent / "output"
    out_path = out_dir / "momentum_enriched.json"

    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                **meta,
                "enriched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "enriched_count": len(enriched),
            },
            "hits": enriched,
        }, f, indent=2)

    log.info("Written to %s", out_path)


if __name__ == "__main__":
    main()
