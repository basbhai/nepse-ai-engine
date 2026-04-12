"""
analysis/backtester.py — NEPSE AI Engine
═══════════════════════════════════════════════════════════════════════════════
Phase 1 — Math only, $0 cost. No AI calls.

Two simulations:
  SIM1 — Capital Constrained: NPR 2,00,000 | 3 slots | circuit breaker
  SIM2 — Free Run:            Unlimited capital | unlimited slots | no breaker

Dynamic threshold optimization:
  Training:   2020-07-01 → 2022-12-31
  Validation: 2023-01-01 → 2024-06-30
  Test:       2024-07-01 → 2026-03-29

Key mechanics:
  Trailing stop — activates once profit crosses trail_activate_pct (default 6%)
                  trails trail_pct (default 3%) below peak price from that point
                  below trail_activate_pct: hard stop at entry * (1 - stop_pct)
  T+4 minimum   — position cannot be CLOSED before 4 trading days after entry
                  (NEPSE settlement reality). Stop/target levels still update
                  intraday; exit is deferred to T+4 if triggered earlier.
  Indicator opt — RSI period, BB period, EMA spans, MACD params all optimized
                  per signal. Indicators recomputed per combination in optimizer.

Performance:
  Indicator cache — computed ONCE per combo set, reused across threshold combos
  3 parallel workers — safe for 8GB RAM (~500MB total usage)
  Optimize time: ~45-75 min (indicator recompute adds time vs previous version)
  Full run time:  ~20-30 min

Fees: Real Nepal tiered brokerage + SEBON + DP + CGT
Symbols: From price_history, avg volume > 50k (liquidity filter)
Candle signals: Disabled (PF=0.63 in initial backtest — noise on daily data)

Run commands:
  python -m analysis.backtester db-check              # ← run this FIRST
  python -m analysis.backtester test [SYMBOL]         # single symbol (default: NABIL)
  python -m analysis.backtester optimize              # grid search on training period
  python -m analysis.backtester run                   # full backtest with default thresholds
  python -m analysis.backtester full                  # optimize + validate + full backtest
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import sys
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from datetime import datetime
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backtester")

# ── Env ───────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── DB imports ────────────────────────────────────────────────────────────────
from db.connection import _db
from sheets import upsert_row


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — DB HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def run_db_check() -> bool:
    """
    Pre-flight check before any backtest run.
    Tests every DB dependency the backtester needs.
    Returns True only if ALL checks pass.

    Usage: python -m analysis.backtester db-check
    """
    print("\n" + "=" * 65)
    print("  NEPSE BACKTESTER — DB PRE-FLIGHT CHECK")
    print("=" * 65)

    all_passed = True

    def _check(label: str, fn) -> bool:
        try:
            result = fn()
            print(f"  ✅  {label}: {result}")
            return True
        except Exception as e:
            print(f"  ❌  {label}: {e}")
            return False

    # ── 1. Basic connection ───────────────────────────────────────────────────
    def check_connection():
        with _db() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        return "Connected to Neon PostgreSQL"

    if not _check("DB Connection", check_connection):
        all_passed = False
        print("\n  FATAL: Cannot connect. Check DATABASE_URL in .env")
        print("=" * 65)
        return False  # no point continuing if can't connect

    # ── 2. price_history table exists and has rows ────────────────────────────
    def check_price_history():
        with _db() as cur:
            cur.execute("""
                SELECT COUNT(*) as cnt,
                       COUNT(DISTINCT symbol) as syms,
                       MIN(date) as earliest,
                       MAX(date) as latest
                FROM price_history
                WHERE close IS NOT NULL
            """)
            r = cur.fetchone()
        if r["cnt"] == 0:
            raise ValueError("Table exists but has 0 rows")
        return (f"{r['cnt']:,} rows | {r['syms']} symbols | "
                f"{r['earliest']} → {r['latest']}")

    if not _check("price_history", check_price_history):
        all_passed = False

    # ── 3. Liquid symbols (avg volume > 50k) ─────────────────────────────────
    def check_liquid_symbols():
        with _db() as cur:
            cur.execute("""
                SELECT COUNT(*) as cnt
                FROM (
                    SELECT symbol
                    FROM price_history
                    WHERE close IS NOT NULL
                      AND volume IS NOT NULL
                      AND volume != ''
                    GROUP BY symbol
                    HAVING AVG(NULLIF(volume, '')::numeric) > 50000
                ) sub
            """)
            r = cur.fetchone()
        count = r["cnt"]
        skip  = ("BF", "MF", "SF", "DB", "DEBD")
        # rough estimate after suffix filter
        note  = "⚠ low — consider reducing volume threshold" if count < 30 else "OK"
        return f"{count} symbols pass 50k avg-volume filter ({note})"

    if not _check("Liquid symbols (50k vol)", check_liquid_symbols):
        all_passed = False

    # ── 4. nepse_indices — index_id = 58 ─────────────────────────────────────
    def check_nepse_index():
        with _db() as cur:
            cur.execute("""
                SELECT COUNT(*) as cnt,
                       MIN(date)  as earliest,
                       MAX(date)  as latest
                FROM nepse_indices
                WHERE index_id = '58'
                  AND current_value IS NOT NULL
            """)
            r = cur.fetchone()
        if r["cnt"] == 0:
            raise ValueError("No rows for index_id=58 — NEPSE composite missing")
        return f"{r['cnt']} rows | {r['earliest']} → {r['latest']}"

    if not _check("nepse_indices (id=58)", check_nepse_index):
        all_passed = False

    # ── 5. backtest_results table — check if it exists ───────────────────────
    def check_backtest_results():
        with _db() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'backtest_results'
                ) as exists
            """)
            r = cur.fetchone()
        if not r["exists"]:
            raise ValueError(
                "Table does not exist — run db.migrations first.\n"
                "         Add 'backtest_results' to schema.prisma, then:\n"
                "         python -m db.codegen && python -m db.migrations"
            )
        # check required columns exist
        with _db() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'backtest_results'
            """)
            cols = {row["column_name"] for row in cur.fetchall()}

        required = {
            "test_name", "parameter_tested", "optimal_value",
            "win_rate_at_optimal", "sample_size", "confidence",
            "date_run", "notes",
        }
        extended = {
            "sim_mode", "period_start", "period_end", "total_trades",
            "wins", "losses", "win_rate_pct", "profit_factor",
            "annual_ret_pct", "total_pnl_npr", "total_fees_npr",
            "sharpe_ratio", "max_drawdown_pct", "alpha_vs_nepse",
            "signal_breakdown",
        }
        missing_base  = required - cols
        missing_extra = extended - cols

        if missing_base:
            raise ValueError(f"Missing base columns: {missing_base}")

        note = ""
        if missing_extra:
            note = (f" | ⚠ extended columns missing: {missing_extra} "
                    f"— save_to_db will fall back to base columns only")
        return f"Table exists with {len(cols)} columns{note}"

    if not _check("backtest_results table", check_backtest_results):
        all_passed = False

    # ── 6. settings table (needed by some callers) ────────────────────────────
    def check_settings():
        with _db() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'settings'
                ) as exists
            """)
            r = cur.fetchone()
        if not r["exists"]:
            raise ValueError("settings table missing")
        return "OK"

    if not _check("settings table", check_settings):
        all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 65)
    if all_passed:
        print("  ✅  ALL CHECKS PASSED — safe to run backtester\n")
    else:
        print("  ❌  SOME CHECKS FAILED — fix above errors before running\n")
    print("=" * 65 + "\n")
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_symbols() -> list:
    """
    Pull symbols from price_history.
    Filters: avg daily volume > 50,000 (liquidity filter).
    Excludes: mutual funds / bond funds (BF/MF/SF/DB/DEBD suffixes).
    """
    try:
        with _db() as cur:
            cur.execute("""
                SELECT symbol
                FROM "price_history"
                WHERE close IS NOT NULL
                  AND close != ''
                  AND volume IS NOT NULL
                  AND volume != ''
                GROUP BY symbol
                HAVING AVG(NULLIF(volume, '')::numeric) > 50000
                ORDER BY symbol
            """)
            rows = cur.fetchall()

        symbols = [r["symbol"] for r in rows]
        skip    = ("BF", "MF", "SF", "DB", "DEBD")
        symbols = [s for s in symbols
                   if not any(s.endswith(sfx) for sfx in skip)]
        log.info("Loaded %d liquid symbols from price_history", len(symbols))
        return symbols

    except Exception as e:
        log.error("load_all_symbols failed: %s", e)
        return []


def load_price_history_raw(symbol: str) -> pd.DataFrame:
    """
    Load full OHLCV for one symbol from price_history.
    Returns DataFrame indexed by date (datetime), sorted ASC.
    Columns: open, high, low, close, volume (all float).
    """
    try:
        with _db() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, ltp, volume
                FROM "price_history"
                WHERE symbol = %s
                  AND date IS NOT NULL
                ORDER BY date ASC
            """, (symbol,))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        for col in ["open", "high", "low", "close", "ltp", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["close"]  = df["close"].fillna(df["ltp"])
        df["open"]   = df["open"].fillna(df["close"])
        df["high"]   = df["high"].fillna(df["close"])
        df["low"]    = df["low"].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0)

        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        return df[df["close"] > 0]

    except Exception as e:
        log.error("load_price_history_raw(%s) failed: %s", symbol, e)
        return pd.DataFrame()


def load_nepse_index() -> pd.DataFrame:
    """Load NEPSE composite index (index_id=58)."""
    try:
        with _db() as cur:
            cur.execute("""
                SELECT date, current_value
                FROM "nepse_indices"
                WHERE index_id = '58'
                ORDER BY date ASC
            """)
            rows = cur.fetchall()

        if not rows:
            log.warning("No NEPSE index data found for index_id=58")
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["current_value"] = pd.to_numeric(df["current_value"], errors="coerce")
        df = df.dropna(subset=["current_value"])
        return df.set_index("date").sort_index()

    except Exception as e:
        log.error("load_nepse_index failed: %s", e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INDICATOR COMPUTATION (ON-THE-FLY, NO LOOK-AHEAD)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IndicatorParams:
    """
    Indicator construction parameters — all optimizable.
    These control HOW indicators are built, not how they're traded.
    Separate from SignalThresholds (entry/exit rules).
    """
    # RSI
    rsi_period:    int   = 14

    # MACD
    macd_fast:     int   = 12
    macd_slow:     int   = 26
    macd_signal:   int   = 9

    # Bollinger Bands
    bb_period:     int   = 20
    bb_std:        float = 2.0

    # EMA spans (used for golden cross + trend filter)
    ema_fast:      int   = 20
    ema_slow:      int   = 50
    ema_trend:     int   = 200

    # Volume ratio lookback
    vol_lookback:  int   = 20


DEFAULT_IND_PARAMS = IndicatorParams()

# Grid for indicator parameter optimization
INDICATOR_GRID = {
    "rsi_period":  [10, 14, 21],
    "macd_fast":   [8,  12, 16],
    "macd_slow":   [21, 26, 30],
    "macd_signal": [7,  9,  12],
    "bb_period":   [15, 20, 25],
    "bb_std":      [1.8, 2.0, 2.2],
    "ema_fast":    [15, 20, 25],
    "ema_slow":    [40, 50, 60],
    "vol_lookback":[15, 20, 25],
}


def compute_indicators(df: pd.DataFrame,
                       params: IndicatorParams = None) -> pd.DataFrame:
    """
    Compute RSI, EMA, MACD, Bollinger Bands, Volume Ratio from OHLCV.
    Each row uses ONLY past data — zero look-ahead bias.
    All indicator parameters are driven by IndicatorParams (default or optimized).
    """
    if df.empty or len(df) < 30:
        return df

    p      = params or DEFAULT_IND_PARAMS
    close  = df["close"]
    volume = df["volume"]

    # ── RSI (Wilder smoothing, parameterized period) ──────────────────────────
    delta    = close.diff()
    alpha    = 1 / p.rsi_period
    avg_gain = delta.clip(lower=0).ewm(
        alpha=alpha, min_periods=p.rsi_period, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(
        alpha=alpha, min_periods=p.rsi_period, adjust=False).mean()
    rs            = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"]  = 100 - (100 / (1 + rs))   # keep column name "rsi_14" for compat

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema_20"]  = close.ewm(span=p.ema_fast,  min_periods=p.ema_fast,  adjust=False).mean()
    df["ema_50"]  = close.ewm(span=p.ema_slow,  min_periods=p.ema_slow,  adjust=False).mean()
    df["ema_200"] = close.ewm(span=p.ema_trend, min_periods=p.ema_trend, adjust=False).mean()

    # ── MACD (parameterized fast/slow/signal) ─────────────────────────────────
    ema_fast_s     = close.ewm(span=p.macd_fast,   min_periods=p.macd_fast,   adjust=False).mean()
    ema_slow_s     = close.ewm(span=p.macd_slow,   min_periods=p.macd_slow,   adjust=False).mean()
    df["macd_line"]   = ema_fast_s - ema_slow_s
    df["macd_signal"] = df["macd_line"].ewm(
        span=p.macd_signal, min_periods=p.macd_signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

    # Cross: histogram neg→pos = BULLISH
    prev_hist = df["macd_hist"].shift(1)
    df["macd_cross"] = "NONE"
    df.loc[(prev_hist < 0) & (df["macd_hist"] > 0), "macd_cross"] = "BULLISH"
    df.loc[(prev_hist > 0) & (df["macd_hist"] < 0), "macd_cross"] = "BEARISH"

    # ── Bollinger Bands (parameterized period + std multiplier) ───────────────
    bb_mid = close.rolling(p.bb_period).mean()
    bb_std = close.rolling(p.bb_period).std()
    df["bb_upper"]  = bb_mid + p.bb_std * bb_std
    df["bb_lower"]  = bb_mid - p.bb_std * bb_std
    df["bb_middle"] = bb_mid

    # LOWER_TOUCH: price below lower band yesterday, back inside today (bounce)
    prev_close = close.shift(1)
    prev_lower = df["bb_lower"].shift(1)
    df["bb_signal"] = "NEUTRAL"
    df.loc[
        (prev_close < prev_lower) & (close > df["bb_lower"]),
        "bb_signal"
    ] = "LOWER_TOUCH"

    # ── EMA fast/slow Golden Cross ────────────────────────────────────────────
    prev_e_fast = df["ema_20"].shift(1)
    prev_e_slow = df["ema_50"].shift(1)
    df["ema_20_50_cross"] = "NONE"
    df.loc[
        (prev_e_fast < prev_e_slow) & (df["ema_20"] > df["ema_50"]),
        "ema_20_50_cross"
    ] = "GOLDEN"
    df.loc[
        (prev_e_fast > prev_e_slow) & (df["ema_20"] < df["ema_50"]),
        "ema_20_50_cross"
    ] = "DEATH"

    # ── Volume ratio (today / rolling avg) ────────────────────────────────────
    df["volume_ratio"] = volume / volume.rolling(p.vol_lookback).mean().replace(0, np.nan)

    return df


def load_sector_map() -> dict:
    """
    Load symbol → sector mapping from share_sectors table.
    Falls back to empty dict if table missing or empty.
    Called once at startup — result passed to all engine.run() calls.
    """
    try:
        with _db() as cur:
            cur.execute("""
                SELECT symbol, "sectorname"
                FROM share_sectors
                WHERE symbol IS NOT NULL
                  AND "sectorname" IS NOT NULL
            """)
            rows = cur.fetchall()
        sector_map = {r["symbol"]: r["sectorname"] for r in rows}
        log.info("Loaded sector map: %d symbols", len(sector_map))
        return sector_map
    except Exception as e:
        log.warning("load_sector_map failed (%s) — sector breakdown will show 'Unknown'", e)
        return {}


def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candle patterns — DISABLED.
    Initial backtest result: PF=0.63 (noise on daily NEPSE data).
    Re-enable after sector data populated in watchlist.
    """
    df["candle_signal"] = "NONE"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIGNAL THRESHOLDS & DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalThresholds:
    """
    Per-signal entry/exit thresholds + trailing stop params.
    All optimized independently per signal type.

    Trailing stop mechanic (per signal):
      - While profit < trail_activate_pct  → hard stop at entry * (1 - stop_pct)
      - Once profit >= trail_activate_pct  → trailing stop = peak * (1 - trail_pct)
        The trailing stop only moves UP, never down.
    """

    # ── MACD Bullish Cross ────────────────────────────────────────────────────
    macd_rsi_max:            float = 65.0
    macd_vol_min:            float = 1.2
    macd_stop_pct:           float = 0.03   # hard stop below entry
    macd_target_pct:         float = 0.10
    macd_hold_days:          int   = 17
    macd_trail_activate_pct: float = 0.06   # profit % to activate trailing
    macd_trail_pct:          float = 0.03   # trail distance below peak

    # ── Bollinger Band Lower Touch ────────────────────────────────────────────
    bb_rsi_max:              float = 40.0
    bb_vol_min:              float = 1.5
    bb_stop_pct:             float = 0.04
    bb_target_pct:           float = 0.20
    bb_hold_days:            int   = 130
    bb_trail_activate_pct:   float = 0.06
    bb_trail_pct:            float = 0.03

    # ── SMA Golden Cross (EMA fast > EMA slow) ────────────────────────────────
    sma_rsi_max:             float = 65.0
    sma_vol_min:             float = 1.2
    sma_stop_pct:            float = 0.03
    sma_target_pct:          float = 0.12
    sma_hold_days:           int   = 33
    sma_trail_activate_pct:  float = 0.06
    sma_trail_pct:           float = 0.03

    # ── Global ────────────────────────────────────────────────────────────────
    rsi_herding_block:       float = 72.0


DEFAULT_THRESHOLDS = SignalThresholds()

OPTIMIZATION_GRID = {
    "macd": {
        "stop_pct":           [0.02, 0.03, 0.04, 0.05],
        "target_pct":         [0.06, 0.08, 0.10, 0.12, 0.15],
        "rsi_max":            [60.0, 65.0, 70.0, 72.0],
        "vol_min":            [1.0,  1.2,  1.5,  2.0],
        "hold_days":          [10,   17,   25,   35],
        "trail_activate_pct": [0.05, 0.06, 0.08, 0.10],
        "trail_pct":          [0.02, 0.03, 0.04, 0.05],
    },
    "bb": {
        "stop_pct":           [0.02, 0.03, 0.04, 0.05],
        "target_pct":         [0.10, 0.15, 0.20, 0.25],
        "rsi_max":            [35.0, 40.0, 45.0, 50.0],
        "vol_min":            [1.2,  1.5,  2.0],
        "hold_days":          [60,   90,   130,  180],
        "trail_activate_pct": [0.06, 0.08, 0.10, 0.12],
        "trail_pct":          [0.03, 0.04, 0.05, 0.06],
    },
    "sma": {
        "stop_pct":           [0.02, 0.03, 0.04, 0.05],
        "target_pct":         [0.08, 0.10, 0.12, 0.15],
        "rsi_max":            [60.0, 65.0, 70.0],
        "vol_min":            [1.0,  1.2,  1.5],
        "hold_days":          [20,   33,   45,   60],
        "trail_activate_pct": [0.05, 0.06, 0.08, 0.10],
        "trail_pct":          [0.02, 0.03, 0.04, 0.05],
    },
}


def detect_signals(row: pd.Series, thresholds: SignalThresholds) -> list:
    """
    Detect signals on a given indicator row.
    Returns list of (signal_name, stop_pct, target_pct, hold_days, score,
                     trail_activate_pct, trail_pct)
    sorted by score descending — highest quality signal first.
    """
    signals   = []
    rsi       = row.get("rsi_14", np.nan)
    vol_ratio = row.get("volume_ratio", 0)

    # Global herding block — never enter when RSI > 72
    if pd.notna(rsi) and rsi > thresholds.rsi_herding_block:
        return []

    # ── MACD Bullish Cross ────────────────────────────────────────────────────
    if (row.get("macd_cross") == "BULLISH" and
            (pd.isna(rsi) or rsi < thresholds.macd_rsi_max) and
            vol_ratio >= thresholds.macd_vol_min):
        score = 70 + (thresholds.macd_rsi_max - rsi) * 0.3 if pd.notna(rsi) else 70
        signals.append((
            "MACD_CROSS",
            thresholds.macd_stop_pct,
            thresholds.macd_target_pct,
            thresholds.macd_hold_days,
            round(score, 1),
            thresholds.macd_trail_activate_pct,
            thresholds.macd_trail_pct,
        ))

    # ── BB Lower Touch ────────────────────────────────────────────────────────
    if (row.get("bb_signal") == "LOWER_TOUCH" and
            (pd.isna(rsi) or rsi < thresholds.bb_rsi_max) and
            vol_ratio >= thresholds.bb_vol_min):
        score = 85 + (thresholds.bb_rsi_max - rsi) * 0.5 if pd.notna(rsi) else 85
        signals.append((
            "BB_LOWER_TOUCH",
            thresholds.bb_stop_pct,
            thresholds.bb_target_pct,
            thresholds.bb_hold_days,
            round(score, 1),
            thresholds.bb_trail_activate_pct,
            thresholds.bb_trail_pct,
        ))

    # ── SMA Golden Cross ──────────────────────────────────────────────────────
    if (row.get("ema_20_50_cross") == "GOLDEN" and
            pd.notna(row.get("ema_200")) and
            row.get("ema_50", 0) > row.get("ema_200", 0) and
            (pd.isna(rsi) or rsi < thresholds.sma_rsi_max) and
            vol_ratio >= thresholds.sma_vol_min):
        score = 60 + (thresholds.sma_rsi_max - rsi) * 0.2 if pd.notna(rsi) else 60
        signals.append((
            "SMA_GOLDEN_CROSS",
            thresholds.sma_stop_pct,
            thresholds.sma_target_pct,
            thresholds.sma_hold_days,
            round(score, 1),
            thresholds.sma_trail_activate_pct,
            thresholds.sma_trail_pct,
        ))

    # Candle signals: DISABLED (PF=0.63). Re-enable after sector filter.

    signals.sort(key=lambda x: x[4], reverse=True)
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NEPAL FEE CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_brokerage(amount: float) -> float:
    """NEPSE tiered brokerage commission."""
    if amount <= 2_500:
        return 10.0
    elif amount <= 50_000:
        return amount * 0.0036
    elif amount <= 500_000:
        return amount * 0.0033
    elif amount <= 2_000_000:
        return amount * 0.0031
    elif amount <= 10_000_000:
        return amount * 0.0027
    else:
        return amount * 0.0024


def _calc_buy_fees(amount: float) -> float:
    """Buy-side: brokerage + SEBON (0.015%) + DP charge (NPR 25)."""
    return _calc_brokerage(amount) + amount * 0.00015 + 25.0


def _calc_sell_fees(amount: float, profit: float) -> float:
    """Sell-side: brokerage + SEBON (0.015%) + DP (NPR 25) + CGT (7.5% on profit)."""
    cgt = max(0.0, profit * 0.075)
    return _calc_brokerage(amount) + amount * 0.00015 + 25.0 + cgt


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol:              str
    entry_date:          object   # pd.Timestamp
    entry_price:         float
    shares:              int
    allocation:          float
    fees_entry:          float
    primary_signal:      str
    stop_loss:           float    # hard stop — active until trailing kicks in
    target:              float
    max_hold_date:       object   # pd.Timestamp
    score:               float
    trail_activate_pct:  float    # e.g. 0.06 = 6% profit needed to activate
    trail_pct:           float    # e.g. 0.03 = trail 3% below peak
    peak_price:          float    = 0.0   # highest price seen since entry
    trail_activated:     bool     = False # True once profit >= trail_activate_pct
    earliest_exit_date:  object   = None  # T+4 — cannot close before this date


class BacktestEngine:
    """
    Core backtest simulation engine.

    mode='constrained' → Sim1: NPR 2L, 3 slots max, circuit breaker active
    mode='free'        → Sim2: unlimited capital, unlimited slots, no breaker

    Trailing stop mechanic:
        Each day in _process_exits:
          1. Update peak_price = max(peak_price, today's high)
          2. If profit from entry to peak >= trail_activate_pct:
               trail_activated = True
               effective_stop  = peak_price * (1 - trail_pct)
          3. Else: effective_stop = entry_price * (1 - stop_pct)  [hard stop]
          4. Check if today's LOW <= effective_stop → STOP_LOSS
        This means once trailing activates, the stop can only move upward
        (it tracks peak, never retreats).

    T+4 minimum hold:
        earliest_exit_date = entry_date + 4 trading days.
        If stop/target fires before this date, the price is RECORDED (position
        is marked as pending close) but the actual close is deferred until T+4.
        This mirrors NEPSE settlement — you cannot sell shares before T+4.

    Indicator caching:
        Inject precomputed indicators via engine._symbol_cache before run().
        The optimizer builds the cache once per indicator-param combo and
        shares it across all threshold combinations.
    """

    SLOT_CAPITAL           = 66_666   # NPR per position slot
    MAX_SLOTS              = 3
    MIN_HOLD_TRADING_DAYS  = 4        # T+4 NEPSE settlement minimum
    CIRCUIT_BREAKER_STREAK = 7        # consecutive losses before pause
    CIRCUIT_BREAKER_PAUSE  = 5        # trading days to pause

    def __init__(
        self,
        mode:            str   = "constrained",
        initial_capital: float = 200_000,
        thresholds:      SignalThresholds = None,
        ind_params:      IndicatorParams  = None,
        period_start:    str   = "2020-07-01",
        period_end:      str   = "2026-03-29",
    ):
        self.mode            = mode
        self.initial_capital = initial_capital
        self.thresholds      = thresholds or DEFAULT_THRESHOLDS
        self.ind_params      = ind_params  or DEFAULT_IND_PARAMS
        self.period_start    = pd.Timestamp(period_start)
        self.period_end      = pd.Timestamp(period_end)
        self._symbol_cache   = None   # inject from outside to skip recompute

        # Runtime state
        self.free_balance:          float = initial_capital
        self.open_positions:        dict  = {}
        self.closed_trades:         list  = []
        self.loss_streak:           int   = 0
        self.circuit_breaker_until        = None
        self.equity_curve:          list  = []

    def _slot_available(self) -> bool:
        if self.mode == "free":
            return True
        return len(self.open_positions) < self.MAX_SLOTS

    def _allocation_for_trade(self) -> float:
        if self.mode == "free":
            return self.SLOT_CAPITAL
        slots_left = self.MAX_SLOTS - len(self.open_positions)
        if slots_left <= 0 or self.free_balance <= 0:
            return 0
        return min(self.free_balance / slots_left, self.free_balance)

    def _circuit_breaker_active(self, date) -> bool:
        if self.mode == "free":
            return False
        return bool(
            self.circuit_breaker_until and
            date <= self.circuit_breaker_until
        )

    def _get_nth_trading_date(self, after_date, n: int, all_dates: list):
        """Return the Nth trading date after after_date from the sorted date list."""
        future = [d for d in all_dates if d > after_date]
        if len(future) >= n:
            return future[n - 1]
        return future[-1] if future else after_date

    def run(
        self,
        symbols:    list,
        nepse_df:   pd.DataFrame,
        sector_map: dict = None,
        verbose:    bool = False,
    ) -> list:
        """
        Walk every trading date sequentially.
        For each date: check exits → record equity → scan signals → enter.
        Returns list of ClosedTrade.
        """
        sector_map = sector_map or {}

        # Reset state
        self.free_balance          = self.initial_capital
        self.open_positions        = {}
        self.closed_trades         = []
        self.loss_streak           = 0
        self.circuit_breaker_until = None
        self.equity_curve          = []

        # Load indicators — use cache if available
        if self._symbol_cache:
            symbol_data = self._symbol_cache
            log.info("[%s] Using cached indicators: %d symbols",
                     self.mode.upper(), len(symbol_data))
        else:
            symbol_data = self._load_and_compute(symbols)

        # Collect all trading dates in backtest period
        all_dates = set()
        for df in symbol_data.values():
            in_period = df.index[
                (df.index >= self.period_start) &
                (df.index <= self.period_end)
            ]
            all_dates.update(in_period)
        all_dates = sorted(all_dates)

        log.info("[%s] Walking %d trading dates (%s → %s)",
                 self.mode.upper(), len(all_dates),
                 self.period_start.date(), self.period_end.date())

        # Main loop — every trading day
        for date in all_dates:

            # 1. Process exits first (trailing stop + T+4 guard)
            self._process_exits(date, symbol_data, nepse_df, sector_map,
                                 all_dates)

            # 2. Record equity curve
            equity = self.free_balance
            for sym, pos in self.open_positions.items():
                df = symbol_data.get(sym)
                if df is not None and date in df.index:
                    equity += pos.shares * df.loc[date, "close"]
            self.equity_curve.append((date, equity))

            # 3. Gates
            if self._circuit_breaker_active(date):
                continue
            if not self._slot_available():
                continue

            # 4. Scan all symbols for signals today
            candidates = []
            for sym, df in symbol_data.items():
                if date not in df.index:
                    continue
                if sym in self.open_positions:
                    continue
                sigs = detect_signals(df.loc[date], self.thresholds)
                if sigs:
                    candidates.append((sym, sigs[0], df))

            # 5. Rank by score, enter best available
            candidates.sort(key=lambda x: x[1][4], reverse=True)

            for sym, best_signal, df in candidates:
                if not self._slot_available():
                    break
                if self._circuit_breaker_active(date):
                    break

                allocation = self._allocation_for_trade()
                if allocation < 1_000:
                    continue

                # T+1 realistic entry — next day open
                future = df.index[df.index > date]
                if len(future) == 0:
                    continue
                next_date = future[0]
                if next_date > self.period_end:
                    continue

                entry_price = df.loc[next_date, "open"]
                if pd.isna(entry_price) or entry_price <= 0:
                    entry_price = df.loc[next_date, "close"]
                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                shares = int(allocation / entry_price)
                if shares <= 0:
                    continue

                actual_cost = shares * entry_price
                fees_entry  = _calc_buy_fees(actual_cost)

                # Unpack 7-tuple (added trail params)
                sig_name, stop_pct, target_pct, hold_days, score, \
                    trail_act_pct, trail_pct = best_signal

                # T+4 earliest exit — 4 trading days after entry
                earliest_exit = self._get_nth_trading_date(
                    next_date, self.MIN_HOLD_TRADING_DAYS, all_dates
                )

                pos = Position(
                    symbol              = sym,
                    entry_date          = next_date,
                    entry_price         = entry_price,
                    shares              = shares,
                    allocation          = actual_cost + fees_entry,
                    fees_entry          = fees_entry,
                    primary_signal      = sig_name,
                    stop_loss           = entry_price * (1 - stop_pct),
                    target              = entry_price * (1 + target_pct),
                    max_hold_date       = next_date + pd.Timedelta(days=hold_days * 2),
                    score               = score,
                    trail_activate_pct  = trail_act_pct,
                    trail_pct           = trail_pct,
                    peak_price          = entry_price,
                    trail_activated     = False,
                    earliest_exit_date  = earliest_exit,
                )
                self.open_positions[sym] = pos
                if self.mode == "constrained":
                    self.free_balance -= pos.allocation

                if verbose:
                    log.info(
                        "  ENTRY [%s] %s @ %.2f | %s | score=%.0f | "
                        "trail_activate=%.0f%% trail=%.0f%% | earliest_exit=%s",
                        self.mode.upper(), sym, entry_price, sig_name, score,
                        trail_act_pct * 100, trail_pct * 100,
                        str(earliest_exit.date()) if earliest_exit else "N/A"
                    )

        # Force-close any remaining open positions at period end
        self._force_close_all(
            self.period_end, symbol_data, nepse_df, sector_map, all_dates
        )

        log.info("[%s] Complete. %d trades closed.",
                 self.mode.upper(), len(self.closed_trades))
        return self.closed_trades

    def _load_and_compute(self, symbols: list) -> dict:
        """Load price history and compute indicators for all symbols."""
        log.info("[%s] Loading & computing indicators for %d symbols...",
                 self.mode.upper(), len(symbols))
        data = {}
        for i, sym in enumerate(symbols):
            raw = load_price_history_raw(sym)
            if raw.empty or len(raw) < 50:
                continue
            ind = compute_indicators(raw, self.ind_params)
            ind = detect_candle_patterns(ind)
            data[sym] = ind
            if (i + 1) % 50 == 0:
                log.info("  Processed %d / %d symbols", i + 1, len(symbols))
        log.info("[%s] Loaded %d symbols with data",
                 self.mode.upper(), len(data))
        return data

    def _process_exits(self, date, symbol_data, nepse_df, sector_map,
                       all_dates: list):
        """
        Check all open positions for trailing stop / hard stop / target / max-hold.

        Trailing stop logic per position:
          1. Update peak_price to max(peak_price, today high)
          2. Check if profit_from_entry_to_peak >= trail_activate_pct
               → trail_activated = True going forward
          3. If trail_activated:
               effective_stop = peak_price * (1 - trail_pct)
             Else:
               effective_stop = hard stop_loss (set at entry, never changes)
          4. If today LOW <= effective_stop AND date >= earliest_exit_date → STOP_LOSS
          5. If today HIGH >= target AND date >= earliest_exit_date → TARGET_HIT
          6. If date >= max_hold_date AND date >= earliest_exit_date → MAX_HOLD

        T+4 guard:
          If a stop/target fires before earliest_exit_date, the exit is
          deferred: we close AT earliest_exit_date using that day's close price.
          This represents the earliest we can physically settle the sale.
        """
        to_close  = []   # (sym, exit_date, exit_price, reason)
        to_defer  = []   # positions triggered before T+4 — close at T+4

        for sym, pos in self.open_positions.items():
            df = symbol_data.get(sym)
            if df is None or date not in df.index:
                continue

            row  = df.loc[date]
            low  = row["low"]
            high = row["high"]
            ltp  = row["close"]

            if pd.isna(ltp) or ltp <= 0:
                continue

            # 1. Update peak price
            if pd.notna(high) and high > pos.peak_price:
                pos.peak_price = high

            # 2. Check trailing activation
            peak_profit_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
            if not pos.trail_activated and peak_profit_pct >= pos.trail_activate_pct:
                pos.trail_activated = True

            # 3. Compute effective stop
            if pos.trail_activated:
                effective_stop = pos.peak_price * (1 - pos.trail_pct)
            else:
                effective_stop = pos.stop_loss  # original hard stop

            # 4. Determine exit trigger
            exit_reason = None
            exit_price  = None

            if pd.notna(low) and low <= effective_stop:
                exit_reason = "STOP_LOSS"
                exit_price  = effective_stop
            elif pd.notna(high) and high >= pos.target:
                exit_reason = "TARGET_HIT"
                exit_price  = pos.target
            elif date >= pos.max_hold_date:
                exit_reason = "MAX_HOLD"
                exit_price  = ltp

            if exit_reason is None:
                continue

            # 5. T+4 guard — cannot physically close before earliest_exit_date
            if pos.earliest_exit_date and date < pos.earliest_exit_date:
                # Defer to earliest_exit_date — find that date's close
                to_defer.append((sym, pos.earliest_exit_date, exit_reason))
            else:
                to_close.append((sym, date, exit_price, exit_reason))

        # Process deferred closes at T+4 price
        for sym, exit_date, original_reason in to_defer:
            pos = self.open_positions.get(sym)
            if pos is None:
                continue
            df = symbol_data.get(sym)
            if df is None:
                continue
            # Find the exit date or nearest available
            available = df.index[(df.index >= exit_date) & (df.index <= self.period_end)]
            if len(available) == 0:
                continue
            actual_exit_date  = available[0]
            actual_exit_price = df.loc[actual_exit_date, "close"]
            if pd.isna(actual_exit_price) or actual_exit_price <= 0:
                continue
            reason_label = f"{original_reason}_DEFERRED_T4"
            to_close.append((sym, actual_exit_date, actual_exit_price, reason_label))

        for sym, d, price, reason in to_close:
            if sym in self.open_positions:  # guard: deferred close may duplicate
                self._close_trade(sym, d, price, reason,
                                  symbol_data, nepse_df, sector_map)

    def _close_trade(self, sym, date, exit_price, exit_reason,
                     symbol_data, nepse_df, sector_map):
        pos         = self.open_positions.pop(sym)
        gross_pnl   = (exit_price - pos.entry_price) * pos.shares
        sell_amount = exit_price * pos.shares
        fees_exit   = _calc_sell_fees(sell_amount, max(0, gross_pnl))
        net_pnl     = gross_pnl - pos.fees_entry - fees_exit
        return_pct  = (net_pnl / pos.allocation * 100) if pos.allocation > 0 else 0
        hold_days   = (date - pos.entry_date).days

        if net_pnl > 0:
            result = "WIN"
            self.loss_streak           = 0
            self.circuit_breaker_until = None
        elif net_pnl < 0:
            result = "LOSS"
            self.loss_streak += 1
            if (self.mode == "constrained" and
                    self.loss_streak >= self.CIRCUIT_BREAKER_STREAK):
                df     = symbol_data.get(sym)
                future = df.index[df.index > date] if df is not None else []
                if len(future) >= self.CIRCUIT_BREAKER_PAUSE:
                    self.circuit_breaker_until = (
                        future[self.CIRCUIT_BREAKER_PAUSE - 1]
                    )
                log.warning(
                    "  CIRCUIT BREAKER: %d losses. Paused until %s",
                    self.loss_streak, self.circuit_breaker_until
                )
        else:
            result = "BREAKEVEN"

        if self.mode == "constrained":
            self.free_balance += pos.allocation + net_pnl

        capital_after = (
            self.free_balance +
            sum(p.allocation for p in self.open_positions.values())
            if self.mode == "constrained" else self.initial_capital
        )

        self.closed_trades.append(ClosedTrade(
            symbol               = sym,
            sector               = sector_map.get(sym, "Unknown"),
            primary_signal       = pos.primary_signal,
            entry_date           = str(pos.entry_date.date()),
            entry_price          = round(pos.entry_price, 2),
            exit_date            = str(date.date()),
            exit_price           = round(exit_price, 2),
            exit_reason          = exit_reason,
            shares               = pos.shares,
            allocation           = round(pos.allocation, 2),
            gross_pnl            = round(gross_pnl, 2),
            fees_total           = round(pos.fees_entry + fees_exit, 2),
            net_pnl              = round(net_pnl, 2),
            return_pct           = round(return_pct, 2),
            hold_days            = hold_days,
            result               = result,
            loss_streak_at_entry = self.loss_streak,
            capital_after        = round(capital_after, 2),
            trail_activated      = pos.trail_activated,
            peak_price           = round(pos.peak_price, 2),
        ))

    def _force_close_all(self, date, symbol_data, nepse_df, sector_map,
                         all_dates: list):
        """Force-close all open positions at end of backtest period."""
        for sym in list(self.open_positions.keys()):
            df = symbol_data.get(sym)
            if df is None:
                continue
            available = df.index[df.index <= date]
            if len(available) == 0:
                continue
            last_price = df.loc[available[-1], "close"]
            if pd.notna(last_price) and last_price > 0:
                self._close_trade(
                    sym, available[-1], last_price,
                    "PERIOD_END", symbol_data, nepse_df, sector_map
                )


@dataclass
class ClosedTrade:
    symbol:               str
    sector:               str
    primary_signal:       str
    entry_date:           str
    entry_price:          float
    exit_date:            str
    exit_price:           float
    exit_reason:          str
    shares:               int
    allocation:           float
    gross_pnl:            float
    fees_total:           float
    net_pnl:              float
    return_pct:           float
    hold_days:            int
    result:               str
    loss_streak_at_entry: int
    capital_after:        float
    trail_activated:      bool  = False
    peak_price:           float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RESULTS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_results(
    trades:          list,
    initial_capital: float,
    nepse_df:        pd.DataFrame,
    label:           str = "",
) -> dict:
    """Compute all backtest KPIs from a list of ClosedTrade."""
    if not trades:
        return {"error": "No trades", "label": label}

    df = pd.DataFrame([asdict(t) for t in trades])
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    total  = len(df)
    wins   = (df["result"] == "WIN").sum()
    losses = (df["result"] == "LOSS").sum()
    be     = (df["result"] == "BREAKEVEN").sum()

    win_rate   = wins / total * 100 if total > 0 else 0
    sum_wins   = df.loc[df["result"] == "WIN",  "net_pnl"].sum()
    sum_losses = abs(df.loc[df["result"] == "LOSS", "net_pnl"].sum())
    pf         = round(sum_wins / sum_losses, 2) if sum_losses > 0 else float("inf")

    total_pnl     = df["net_pnl"].sum()
    total_fees    = df["fees_total"].sum()
    total_ret_pct = total_pnl / initial_capital * 100

    days    = (df["exit_date"].max() - df["entry_date"].min()).days
    years   = days / 365.25
    ann_ret = (
        ((1 + total_ret_pct / 100) ** (1 / years) - 1) * 100
        if years > 0 else 0
    )

    # Sharpe — annualized, RF=5.5%
    excess = df["return_pct"].values - (5.5 / 252)
    sharpe = (
        np.mean(excess) / np.std(excess) * np.sqrt(252)
        if np.std(excess) > 0 else 0
    )

    # Max drawdown
    cum_pnl     = df.sort_values("exit_date")["net_pnl"].cumsum()
    running_max = cum_pnl.expanding().max()
    max_dd_pct  = (
        (cum_pnl - running_max).min() / initial_capital * 100
        if initial_capital > 0 else 0
    )

    # Alpha vs NEPSE
    alpha = None
    if not nepse_df.empty:
        nw = nepse_df[
            (nepse_df.index >= df["entry_date"].min()) &
            (nepse_df.index <= df["exit_date"].max())
        ]["current_value"]
        if len(nw) >= 2:
            alpha = round(
                total_ret_pct - (nw.iloc[-1] / nw.iloc[0] - 1) * 100, 2
            )

    # Trailing stop stats
    trail_trades = df["trail_activated"].sum() if "trail_activated" in df.columns else 0
    trail_wins   = (
        df.loc[df["trail_activated"] == True, "result"] == "WIN"
    ).sum() if "trail_activated" in df.columns else 0

    # Signal breakdown
    signal_breakdown = {}
    for sig in df["primary_signal"].unique():
        sub = df[df["primary_signal"] == sig]
        sw  = sub.loc[sub["result"] == "WIN",  "net_pnl"].sum()
        sl  = abs(sub.loc[sub["result"] == "LOSS", "net_pnl"].sum())
        signal_breakdown[sig] = {
            "trades":         len(sub),
            "win_rate":       round(
                (sub["result"] == "WIN").sum() / len(sub) * 100, 1
            ),
            "pf":             round(sw / sl, 2) if sl > 0 else float("inf"),
            "net_pnl":        round(sub["net_pnl"].sum(), 0),
            "avg_hold":       round(sub["hold_days"].mean(), 1),
            "trail_used":     int(sub["trail_activated"].sum()) if "trail_activated" in sub.columns else 0,
        }

    # Sector breakdown
    sector_breakdown = {}
    for sec in df["sector"].unique():
        sub = df[df["sector"] == sec]
        sw  = sub.loc[sub["result"] == "WIN",  "net_pnl"].sum()
        sl  = abs(sub.loc[sub["result"] == "LOSS", "net_pnl"].sum())
        sector_breakdown[sec] = {
            "trades":   len(sub),
            "win_rate": round(
                (sub["result"] == "WIN").sum() / len(sub) * 100, 1
            ),
            "pf":       round(sw / sl, 2) if sl > 0 else float("inf"),
            "net_pnl":  round(sub["net_pnl"].sum(), 0),
        }

    return {
        "label":              label,
        "total_trades":       int(total),
        "wins":               int(wins),
        "losses":             int(losses),
        "breakevens":         int(be),
        "win_rate_pct":       round(win_rate, 2),
        "profit_factor":      pf,
        "total_pnl_npr":      round(total_pnl, 0),
        "total_fees_npr":     round(total_fees, 0),
        "total_ret_pct":      round(total_ret_pct, 2),
        "annual_ret_pct":     round(ann_ret, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "max_drawdown_pct":   round(max_dd_pct, 2),
        "alpha_vs_nepse":     alpha,
        "trail_trades":       int(trail_trades),
        "trail_wins":         int(trail_wins),
        "signal_breakdown":   signal_breakdown,
        "sector_breakdown":   sector_breakdown,
        "trades_df":          df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — THRESHOLD OPTIMIZER (indicator cache + parallel workers)
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level dict shared with worker processes via initializer
_WORKER_CACHE = {}


def _worker_init(symbol_cache, nepse_data, t_start, t_end):
    """Initialize each worker with shared read-only data."""
    global _WORKER_CACHE
    _WORKER_CACHE["symbol_cache"] = symbol_cache
    _WORKER_CACHE["nepse_df"]     = nepse_data
    _WORKER_CACHE["t_start"]      = t_start
    _WORKER_CACHE["t_end"]        = t_end


def _eval_combo(args):
    """
    Worker function — evaluate one threshold combination.
    Uses _WORKER_CACHE set by _worker_init.
    Returns (profit_factor, win_rate, trade_count, params, sig_name) or None.
    """
    sig_name, keys, combo, base_thresholds, min_trades = args
    params = dict(zip(keys, combo))

    t = deepcopy(base_thresholds)
    setattr(t, f"{sig_name}_stop_pct",           params["stop_pct"])
    setattr(t, f"{sig_name}_target_pct",         params["target_pct"])
    setattr(t, f"{sig_name}_rsi_max",            params["rsi_max"])
    setattr(t, f"{sig_name}_vol_min",            params["vol_min"])
    setattr(t, f"{sig_name}_hold_days",          params["hold_days"])
    setattr(t, f"{sig_name}_trail_activate_pct", params["trail_activate_pct"])
    setattr(t, f"{sig_name}_trail_pct",          params["trail_pct"])

    symbol_cache = _WORKER_CACHE["symbol_cache"]
    nepse_df     = _WORKER_CACHE["nepse_df"]
    t_start      = _WORKER_CACHE["t_start"]
    t_end        = _WORKER_CACHE["t_end"]

    engine = BacktestEngine(
        mode="free", initial_capital=200_000,
        thresholds=t, period_start=t_start, period_end=t_end,
    )
    engine._symbol_cache = symbol_cache   # inject cache — skip recompute

    trades = engine.run(
        list(symbol_cache.keys()), nepse_df,
        sector_map={}, verbose=False,
    )

    sig_key = (
        "MACD_CROSS"       if sig_name == "macd" else
        "BB_LOWER_TOUCH"   if sig_name == "bb"   else
        "SMA_GOLDEN_CROSS"
    )
    sig_trades = [tr for tr in trades if tr.primary_signal == sig_key]
    if len(sig_trades) < min_trades:
        return None

    res = compute_results(sig_trades, 200_000, nepse_df)
    if res.get("error"):
        return None

    return (res["profit_factor"], res["win_rate_pct"],
            len(sig_trades), params, sig_name)


def _worker_init_ind(raw_data, nepse_data, t_start, t_end, ind_params_dict):
    """Worker initializer for indicator optimization — recomputes indicators."""
    global _WORKER_CACHE
    _WORKER_CACHE["raw_data"]      = raw_data
    _WORKER_CACHE["nepse_df"]      = nepse_data
    _WORKER_CACHE["t_start"]       = t_start
    _WORKER_CACHE["t_end"]         = t_end
    _WORKER_CACHE["ind_params_dict"] = ind_params_dict


def _eval_ind_combo(args):
    """
    Worker for indicator parameter optimization.
    Recomputes indicators for every combo — more expensive but finds
    the indicator construction (RSI period, MACD params, BB period etc.)
    that maximizes profit factor across all signals combined.
    """
    keys, combo, base_thresholds, min_trades = args
    params = dict(zip(keys, combo))

    # Build IndicatorParams from combo
    ip = IndicatorParams(
        rsi_period   = params.get("rsi_period",   14),
        macd_fast    = params.get("macd_fast",    12),
        macd_slow    = params.get("macd_slow",    26),
        macd_signal  = params.get("macd_signal",   9),
        bb_period    = params.get("bb_period",    20),
        bb_std       = params.get("bb_std",       2.0),
        ema_fast     = params.get("ema_fast",     20),
        ema_slow     = params.get("ema_slow",     50),
        vol_lookback = params.get("vol_lookback", 20),
    )

    # Validate MACD: fast must be < slow
    if ip.macd_fast >= ip.macd_slow:
        return None

    raw_data = _WORKER_CACHE["raw_data"]
    nepse_df = _WORKER_CACHE["nepse_df"]
    t_start  = _WORKER_CACHE["t_start"]
    t_end    = _WORKER_CACHE["t_end"]

    # Recompute indicators with these params
    symbol_cache = {}
    for sym, raw_df in raw_data.items():
        if raw_df.empty or len(raw_df) < 50:
            continue
        ind = compute_indicators(raw_df.copy(), ip)
        ind = detect_candle_patterns(ind)
        symbol_cache[sym] = ind

    if not symbol_cache:
        return None

    engine = BacktestEngine(
        mode="free", initial_capital=200_000,
        thresholds=base_thresholds,
        ind_params=ip,
        period_start=t_start, period_end=t_end,
    )
    engine._symbol_cache = symbol_cache

    trades = engine.run(list(symbol_cache.keys()), nepse_df,
                        sector_map={}, verbose=False)
    if len(trades) < min_trades:
        return None

    res = compute_results(trades, 200_000, nepse_df)
    if res.get("error"):
        return None

    return (res["profit_factor"], res["win_rate_pct"],
            len(trades), params, "all_signals")


def optimize_indicators(
    symbols:        list,
    nepse_df:       pd.DataFrame,
    base_thresholds: SignalThresholds,
    training_start: str = "2019-07-15",
    training_end:   str = "2022-12-31",
    min_trades:     int = 20,
    n_workers:      int = 4,
) -> IndicatorParams:
    """
    Grid search over indicator construction parameters on training period.
    Finds the RSI period, MACD params, BB period etc. that maximize profit
    factor when combined with the base signal thresholds.

    Note: This is more expensive than threshold optimization because indicators
    must be recomputed per combination. Run after threshold optimization.
    """
    from multiprocessing import Pool

    log.info("=" * 60)
    log.info("INDICATOR OPTIMIZATION — %s → %s", training_start, training_end)
    log.info("Optimizing: RSI period, MACD fast/slow/signal, BB period/std,")
    log.info("            EMA fast/slow, volume lookback")
    log.info("Workers: %d", n_workers)
    log.info("=" * 60)

    # Load raw data once — workers recompute indicators per combo
    log.info("Loading raw price data (one-time)...")
    raw_data = {}
    for sym in symbols:
        raw = load_price_history_raw(sym)
        if not raw.empty and len(raw) >= 50:
            raw_data[sym] = raw
    log.info("Raw data loaded: %d symbols", len(raw_data))

    # Build all combinations
    keys   = list(INDICATOR_GRID.keys())
    vals   = list(INDICATOR_GRID.values())
    combos = list(product(*vals))
    log.info("%d indicator combinations to evaluate", len(combos))

    work_args = [
        (keys, combo, base_thresholds, min_trades)
        for combo in combos
    ]

    with Pool(
        processes   = n_workers,
        initializer = _worker_init_ind,
        initargs    = (raw_data, nepse_df, training_start, training_end, {}),
    ) as pool:
        results = pool.map(_eval_ind_combo, work_args)

    best_pf     = 0
    best_params = None
    best_wr     = 0

    for result in results:
        if result is None:
            continue
        pf, wr, trade_count, params, _ = result
        if pf > best_pf and wr >= 45:
            best_pf     = pf
            best_wr     = wr
            best_params = params
            log.info("  New best — PF=%.2f WR=%.1f%% trades=%d | %s",
                     pf, wr, trade_count, params)

    if best_params:
        opt = IndicatorParams(
            rsi_period   = best_params.get("rsi_period",   14),
            macd_fast    = best_params.get("macd_fast",    12),
            macd_slow    = best_params.get("macd_slow",    26),
            macd_signal  = best_params.get("macd_signal",   9),
            bb_period    = best_params.get("bb_period",    20),
            bb_std       = best_params.get("bb_std",       2.0),
            ema_fast     = best_params.get("ema_fast",     20),
            ema_slow     = best_params.get("ema_slow",     50),
            vol_lookback = best_params.get("vol_lookback", 20),
        )
        log.info("\nBEST INDICATOR PARAMS (PF=%.2f WR=%.1f%%):", best_pf, best_wr)
        log.info("  RSI period=%d", opt.rsi_period)
        log.info("  MACD fast=%d slow=%d signal=%d",
                 opt.macd_fast, opt.macd_slow, opt.macd_signal)
        log.info("  BB period=%d std=%.1f", opt.bb_period, opt.bb_std)
        log.info("  EMA fast=%d slow=%d trend=%d",
                 opt.ema_fast, opt.ema_slow, opt.ema_trend)
        log.info("  Vol lookback=%d", opt.vol_lookback)
        return opt
    else:
        log.warning("No valid indicator combo found — using defaults")
        return DEFAULT_IND_PARAMS


def optimize_thresholds(
    symbols:        list,
    nepse_df:       pd.DataFrame,
    sector_map:     dict,
    ind_params:     IndicatorParams = None,
    training_start: str = "2019-07-15",
    training_end:   str = "2022-12-31",
    min_trades:     int = 20,
    n_workers:      int = 4,
) -> SignalThresholds:
    """
    Grid search over threshold combinations on training period only.
    Optimizes MACD, BB, and SMA independently (including trailing params).

    Key optimizations:
      1. Indicator cache — computed once, shared across all combinations
      2. Parallel workers — safe for 8GB RAM
    """
    from multiprocessing import Pool

    log.info("=" * 60)
    log.info("THRESHOLD OPTIMIZATION — %s → %s", training_start, training_end)
    log.info("Workers: %d  |  RAM-safe for 8GB", n_workers)
    log.info("=" * 60)

    # ── Build indicator cache ONCE ────────────────────────────────────────────
    log.info("Building indicator cache (one-time, ~2-3 min)...")
    temp = BacktestEngine(mode="free", ind_params=ind_params or DEFAULT_IND_PARAMS)
    symbol_cache = temp._load_and_compute(symbols)
    log.info("Cache ready: %d symbols", len(symbol_cache))

    best_thresholds = deepcopy(DEFAULT_THRESHOLDS)

    # ── Optimize each signal independently ───────────────────────────────────
    for sig_name in ["macd", "bb", "sma"]:
        log.info("\nOptimizing %s (including trailing stop params)...",
                 sig_name.upper())
        grid   = OPTIMIZATION_GRID[sig_name]
        keys   = list(grid.keys())
        vals   = list(grid.values())
        combos = list(product(*vals))
        log.info("  %d combinations, %d workers", len(combos), n_workers)

        work_args = [
            (sig_name, keys, combo, best_thresholds, min_trades)
            for combo in combos
        ]

        with Pool(
            processes   = n_workers,
            initializer = _worker_init,
            initargs    = (symbol_cache, nepse_df, training_start, training_end),
        ) as pool:
            results = pool.map(_eval_combo, work_args)

        best_pf    = 0
        best_combo = None

        for result in results:
            if result is None:
                continue
            pf, wr, trade_count, params, _ = result
            if pf > best_pf and wr >= 45:
                best_pf    = pf
                best_combo = params
                log.info("  Best — %s: PF=%.2f WR=%.1f%% trades=%d | %s",
                         sig_name.upper(), pf, wr, trade_count, params)

        if best_combo:
            setattr(best_thresholds, f"{sig_name}_stop_pct",
                    best_combo["stop_pct"])
            setattr(best_thresholds, f"{sig_name}_target_pct",
                    best_combo["target_pct"])
            setattr(best_thresholds, f"{sig_name}_rsi_max",
                    best_combo["rsi_max"])
            setattr(best_thresholds, f"{sig_name}_vol_min",
                    best_combo["vol_min"])
            setattr(best_thresholds, f"{sig_name}_hold_days",
                    best_combo["hold_days"])
            setattr(best_thresholds, f"{sig_name}_trail_activate_pct",
                    best_combo["trail_activate_pct"])
            setattr(best_thresholds, f"{sig_name}_trail_pct",
                    best_combo["trail_pct"])
            log.info("  LOCKED %s: PF=%.2f | %s",
                     sig_name.upper(), best_pf, best_combo)
        else:
            log.warning("  No valid combo for %s — keeping defaults",
                        sig_name.upper())

    log.info("\nFINAL OPTIMIZED THRESHOLDS:")
    log.info("  MACD: stop=%.0f%% target=%.0f%% rsi<%.0f vol>%.1fx hold=%dd "
             "trail_act=%.0f%% trail=%.0f%%",
             best_thresholds.macd_stop_pct * 100,
             best_thresholds.macd_target_pct * 100,
             best_thresholds.macd_rsi_max,
             best_thresholds.macd_vol_min,
             best_thresholds.macd_hold_days,
             best_thresholds.macd_trail_activate_pct * 100,
             best_thresholds.macd_trail_pct * 100)
    log.info("  BB:   stop=%.0f%% target=%.0f%% rsi<%.0f vol>%.1fx hold=%dd "
             "trail_act=%.0f%% trail=%.0f%%",
             best_thresholds.bb_stop_pct * 100,
             best_thresholds.bb_target_pct * 100,
             best_thresholds.bb_rsi_max,
             best_thresholds.bb_vol_min,
             best_thresholds.bb_hold_days,
             best_thresholds.bb_trail_activate_pct * 100,
             best_thresholds.bb_trail_pct * 100)
    log.info("  SMA:  stop=%.0f%% target=%.0f%% rsi<%.0f vol>%.1fx hold=%dd "
             "trail_act=%.0f%% trail=%.0f%%",
             best_thresholds.sma_stop_pct * 100,
             best_thresholds.sma_target_pct * 100,
             best_thresholds.sma_rsi_max,
             best_thresholds.sma_vol_min,
             best_thresholds.sma_hold_days,
             best_thresholds.sma_trail_activate_pct * 100,
             best_thresholds.sma_trail_pct * 100)

    return best_thresholds


def validate_thresholds(
    thresholds:       SignalThresholds,
    symbols:          list,
    nepse_df:         pd.DataFrame,
    sector_map:       dict,
    symbol_cache:     dict = None,
    ind_params:       IndicatorParams = None,
    validation_start: str   = "2023-01-01",
    validation_end:   str   = "2024-06-30",
    training_pf:      dict  = None,
    max_degradation:  float = 0.30,
) -> tuple:
    """
    Validate on hold-out period. Fails if PF degrades > 30%.
    Accepts optional symbol_cache to avoid recomputing indicators.
    """
    log.info("\nVALIDATION — %s → %s", validation_start, validation_end)

    engine = BacktestEngine(
        mode="free", initial_capital=200_000,
        thresholds=thresholds,
        ind_params=ind_params or DEFAULT_IND_PARAMS,
        period_start=validation_start, period_end=validation_end,
    )
    if symbol_cache:
        engine._symbol_cache = symbol_cache
        log.info("  [VALIDATION] Using cached indicators")

    trades = engine.run(symbols, nepse_df, sector_map, verbose=False)
    res    = compute_results(trades, 200_000, nepse_df, label="VALIDATION")

    if res.get("error"):
        log.warning("  Validation: no trades")
        return False, res

    passed = True
    if training_pf:
        train_pf    = training_pf.get("profit_factor", 1.0)
        val_pf      = res["profit_factor"]
        degradation = (train_pf - val_pf) / train_pf if train_pf > 0 else 1.0
        if degradation > max_degradation:
            log.warning("  FAILED: degraded %.0f%% (%.2f→%.2f)",
                        degradation * 100, train_pf, val_pf)
            passed = False
        else:
            log.info("  PASSED ✅ degradation=%.0f%% (%.2f→%.2f)",
                     degradation * 100, train_pf, val_pf)

    log.info("  Trades=%d WR=%.1f%% PF=%.2f AnnRet=%.1f%%",
             res["total_trades"], res["win_rate_pct"],
             res["profit_factor"], res["annual_ret_pct"])
    return passed, res


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DB PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence_label(trades: int) -> str:
    if trades >= 100:
        return "HIGH"
    if trades >= 30:
        return "MEDIUM"
    return "LOW"


def _thresholds_to_json(t: SignalThresholds,
                        ip: IndicatorParams = None) -> str:
    d = {
        "macd": {
            "stop_pct":           t.macd_stop_pct,
            "target_pct":         t.macd_target_pct,
            "rsi_max":            t.macd_rsi_max,
            "vol_min":            t.macd_vol_min,
            "hold_days":          t.macd_hold_days,
            "trail_activate_pct": t.macd_trail_activate_pct,
            "trail_pct":          t.macd_trail_pct,
        },
        "bb": {
            "stop_pct":           t.bb_stop_pct,
            "target_pct":         t.bb_target_pct,
            "rsi_max":            t.bb_rsi_max,
            "vol_min":            t.bb_vol_min,
            "hold_days":          t.bb_hold_days,
            "trail_activate_pct": t.bb_trail_activate_pct,
            "trail_pct":          t.bb_trail_pct,
        },
        "sma": {
            "stop_pct":           t.sma_stop_pct,
            "target_pct":         t.sma_target_pct,
            "rsi_max":            t.sma_rsi_max,
            "vol_min":            t.sma_vol_min,
            "hold_days":          t.sma_hold_days,
            "trail_activate_pct": t.sma_trail_activate_pct,
            "trail_pct":          t.sma_trail_pct,
        },
    }
    if ip:
        d["indicator_params"] = {
            "rsi_period":   ip.rsi_period,
            "macd_fast":    ip.macd_fast,
            "macd_slow":    ip.macd_slow,
            "macd_signal":  ip.macd_signal,
            "bb_period":    ip.bb_period,
            "bb_std":       ip.bb_std,
            "ema_fast":     ip.ema_fast,
            "ema_slow":     ip.ema_slow,
            "vol_lookback": ip.vol_lookback,
        }
    return json.dumps(d)


def save_to_db(
    r1:           dict,
    r2:           dict,
    thresholds:   SignalThresholds,
    ind_params:   IndicatorParams = None,
    period_start: str = "2020-07-01",
    period_end:   str = "2026-03-29",
) -> None:
    """
    Persist backtest results to backtest_results table.
    One row per simulation (SIM1 + SIM2).
    Upserts on (test_name, date_run) — safe to re-run on same day.

    Falls back gracefully if extended columns don't exist yet:
      base columns only = always works if table exists
      extended columns  = added via schema.prisma migration
    """
    # First check which extended columns actually exist
    try:
        with _db() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'backtest_results'
            """)
            existing_cols = {row["column_name"] for row in cur.fetchall()}
    except Exception as e:
        log.error("Cannot check backtest_results schema: %s", e)
        return

    date_run       = datetime.now().strftime("%Y-%m-%d")
    threshold_json = _thresholds_to_json(thresholds, ind_params)

    for r, sim_mode in [(r1, "constrained"), (r2, "free")]:
        if r.get("error"):
            log.warning("Skipping DB save for %s — no trades", sim_mode)
            continue

        sig_bd_json = json.dumps({
            sig: {
                "trades":      s["trades"],
                "win_rate":    s["win_rate"],
                "pf":          s["pf"],
                "net_pnl":     s["net_pnl"],
                "avg_hold":    s.get("avg_hold"),
                "trail_used":  s.get("trail_used", 0),
            }
            for sig, s in r["signal_breakdown"].items()
        })

        # ── Always-present base columns ───────────────────────────────────────
        row = {
            "test_name":           r["label"],
            "parameter_tested":    threshold_json,
            "optimal_value":       str(r["profit_factor"]),
            "win_rate_at_optimal": str(round(r["win_rate_pct"], 2)),
            "sample_size":         str(r["total_trades"]),
            "confidence":          _confidence_label(r["total_trades"]),
            "date_run":            date_run,
            "notes": (
                f"alpha={r['alpha_vs_nepse']}% "
                f"sharpe={r['sharpe_ratio']} "
                f"maxDD={r['max_drawdown_pct']}% "
                f"annRet={r['annual_ret_pct']}% "
                f"fees=NPR {r['total_fees_npr']:.0f} "
                f"trail_trades={r.get('trail_trades', 0)} "
                f"trail_wins={r.get('trail_wins', 0)}"
            ),
        }

        # ── Extended columns — only add if they exist in DB ───────────────────
        extended = {
            "sim_mode":          sim_mode,
            "period_start":      period_start,
            "period_end":        period_end,
            "total_trades":      str(r["total_trades"]),
            "wins":              str(r["wins"]),
            "losses":            str(r["losses"]),
            "win_rate_pct":      str(round(r["win_rate_pct"], 2)),
            "profit_factor":     str(r["profit_factor"]),
            "annual_ret_pct":    str(round(r["annual_ret_pct"], 2)),
            "total_pnl_npr":     str(round(r["total_pnl_npr"], 0)),
            "total_fees_npr":    str(round(r["total_fees_npr"], 0)),
            "sharpe_ratio":      str(round(r["sharpe_ratio"], 2)),
            "max_drawdown_pct":  str(round(r["max_drawdown_pct"], 2)),
            "alpha_vs_nepse":    str(r["alpha_vs_nepse"]),
            "signal_breakdown":  sig_bd_json,
        }
        for col, val in extended.items():
            if col in existing_cols:
                row[col] = val
            else:
                log.debug("  Column '%s' not in DB — skipping", col)

        try:
            upsert_row("backtest_results", row,
                       conflict_columns=["test_name", "date_run"])
            log.info("  ✅ DB saved: %s (%d trades, PF=%s)",
                     r["label"], r["total_trades"], r["profit_factor"])
        except Exception as e:
            log.error("  DB save failed for %s: %s", r["label"], e)
            log.error("  Row keys attempted: %s", list(row.keys()))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT & REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(r: dict):
    if r.get("error"):
        print(f"\n[{r.get('label','')}] No trades generated.")
        return
    trail_info = (f"  Trail stop used:  {r.get('trail_trades', 0)} trades "
                  f"({r.get('trail_wins', 0)} wins with trail)\n")
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  BACKTEST RESULTS — {r['label']:<41}║
╚══════════════════════════════════════════════════════════════╝

Trades:          {r['total_trades']}  (W:{r['wins']} L:{r['losses']} BE:{r['breakevens']})
Win Rate:        {r['win_rate_pct']:.1f}%
Profit Factor:   {r['profit_factor']}
Total P&L:       NPR {r['total_pnl_npr']:>10,.0f}
Total Fees:      NPR {r['total_fees_npr']:>10,.0f}
Total Return:    {r['total_ret_pct']:.2f}%
Annual Return:   {r['annual_ret_pct']:.2f}%
Sharpe Ratio:    {r['sharpe_ratio']:.2f}
Max Drawdown:    {r['max_drawdown_pct']:.2f}%
Alpha vs NEPSE:  {str(r['alpha_vs_nepse']) + '%' if r['alpha_vs_nepse'] is not None else 'N/A'}
{trail_info}""")
    print("── Signal Breakdown ──────────────────────────────────────")
    for sig, s in r["signal_breakdown"].items():
        print(f"  {sig:<22} trades={s['trades']:>4}  "
              f"WR={s['win_rate']:>5.1f}%  PF={s['pf']:>5.2f}  "
              f"PnL=NPR {s['net_pnl']:>8,.0f}  avgHold={s['avg_hold']:.0f}d  "
              f"trail={s.get('trail_used', 0)}")
    print("\n── Sector Breakdown ──────────────────────────────────────")
    for sec, s in sorted(r["sector_breakdown"].items(),
                         key=lambda x: x[1]["net_pnl"], reverse=True):
        print(f"  {sec:<25} trades={s['trades']:>4}  "
              f"WR={s['win_rate']:>5.1f}%  PF={s['pf']:>5.2f}  "
              f"PnL=NPR {s['net_pnl']:>8,.0f}")


def print_comparison(r1: dict, r2: dict):
    ann1 = f"{r1['annual_ret_pct']:.2f}%" if r1.get("annual_ret_pct") else "N/A"
    ann2 = f"{r2['annual_ret_pct']:.2f}%" if r2.get("annual_ret_pct") else "N/A"
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SIMULATION COMPARISON                                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Metric              │  SIM1 (Constrained)    │  SIM2 (Free Run)            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Total Trades        │  {r1['total_trades']:<22} │  {r2['total_trades']:<28}║
║  Win Rate            │  {r1['win_rate_pct']:<21.1f}% │  {r2['win_rate_pct']:<27.1f}%║
║  Profit Factor       │  {r1['profit_factor']:<22} │  {r2['profit_factor']:<28}║
║  Annual Return       │  {ann1:<22} │  {ann2:<28}║
║  Total P&L (NPR)     │  {r1['total_pnl_npr']:<22,.0f} │  {r2['total_pnl_npr']:<28,.0f}║
║  Max Drawdown        │  {r1['max_drawdown_pct']:<21.2f}% │  {r2['max_drawdown_pct']:<27.2f}%║
║  Sharpe Ratio        │  {r1['sharpe_ratio']:<22.2f} │  {r2['sharpe_ratio']:<28.2f}║
║  Alpha vs NEPSE      │  {str(r1['alpha_vs_nepse'])+'%':<22} │  {str(r2['alpha_vs_nepse'])+'%':<28}║
║  Trail Stop Used     │  {r1.get('trail_trades',0):<22} │  {r2.get('trail_trades',0):<28}║
╚══════════════════════════════════════════════════════════════════════════════╝

GAP ANALYSIS:
  Win rate gap:        {abs(r2['win_rate_pct'] - r1['win_rate_pct']):.1f}%
  Missed signals:      {r2['total_trades'] - r1['total_trades']} trades
  Cost of constraints: NPR {abs(r2['total_pnl_npr'] - r1['total_pnl_npr']):,.0f}
""")
    gap = abs(r2["win_rate_pct"] - r1["win_rate_pct"])
    if gap < 5:
        print("  ✅ Small gap — capital efficient. System works well at NPR 2L.")
    elif gap < 10:
        print("  ⚠️  Moderate gap — some signals missed. Consider more capital.")
    else:
        print("  ❌ Large gap — capital is bottleneck. Scale when confident.")


def save_results(
    r1:         dict,
    r2:         dict,
    thresholds: SignalThresholds,
    ind_params: IndicatorParams = None,
    output_dir: str = "outputs",
) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    p1 = os.path.join(output_dir, "backtest_sim1_constrained_trades.csv")
    p2 = os.path.join(output_dir, "backtest_sim2_free_trades.csv")
    pt = os.path.join(output_dir, "backtest_summary.txt")

    if not r1.get("error"):
        r1["trades_df"].to_csv(p1, index=False)
    if not r2.get("error"):
        r2["trades_df"].to_csv(p2, index=False)

    ip = ind_params or DEFAULT_IND_PARAMS

    with open(pt, "w") as f:
        f.write("NEPSE AI ENGINE — BACKTEST RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("INDICATOR PARAMS\n" + "-" * 40 + "\n")
        f.write(f"RSI period={ip.rsi_period}\n")
        f.write(f"MACD fast={ip.macd_fast} slow={ip.macd_slow} signal={ip.macd_signal}\n")
        f.write(f"BB period={ip.bb_period} std={ip.bb_std}\n")
        f.write(f"EMA fast={ip.ema_fast} slow={ip.ema_slow} trend={ip.ema_trend}\n")
        f.write(f"Vol lookback={ip.vol_lookback}\n\n")

        f.write("SIGNAL THRESHOLDS\n" + "-" * 40 + "\n")
        f.write(f"MACD: stop={thresholds.macd_stop_pct*100:.0f}% "
                f"target={thresholds.macd_target_pct*100:.0f}% "
                f"rsi<{thresholds.macd_rsi_max:.0f} "
                f"vol>{thresholds.macd_vol_min:.1f}x "
                f"hold={thresholds.macd_hold_days}d "
                f"trail_act={thresholds.macd_trail_activate_pct*100:.0f}% "
                f"trail={thresholds.macd_trail_pct*100:.0f}%\n")
        f.write(f"BB:   stop={thresholds.bb_stop_pct*100:.0f}% "
                f"target={thresholds.bb_target_pct*100:.0f}% "
                f"rsi<{thresholds.bb_rsi_max:.0f} "
                f"vol>{thresholds.bb_vol_min:.1f}x "
                f"hold={thresholds.bb_hold_days}d "
                f"trail_act={thresholds.bb_trail_activate_pct*100:.0f}% "
                f"trail={thresholds.bb_trail_pct*100:.0f}%\n")
        f.write(f"SMA:  stop={thresholds.sma_stop_pct*100:.0f}% "
                f"target={thresholds.sma_target_pct*100:.0f}% "
                f"rsi<{thresholds.sma_rsi_max:.0f} "
                f"vol>{thresholds.sma_vol_min:.1f}x "
                f"hold={thresholds.sma_hold_days}d "
                f"trail_act={thresholds.sma_trail_activate_pct*100:.0f}% "
                f"trail={thresholds.sma_trail_pct*100:.0f}%\n\n")

        for r, label in [(r1, "SIM1 CONSTRAINED"), (r2, "SIM2 FREE RUN")]:
            f.write(f"\n{label}\n" + "-" * 40 + "\n")
            if r.get("error"):
                f.write(f"ERROR: {r['error']}\n")
                continue
            f.write(f"Trades:        {r['total_trades']} "
                    f"(W:{r['wins']} L:{r['losses']} BE:{r['breakevens']})\n")
            f.write(f"Win Rate:      {r['win_rate_pct']:.1f}%\n")
            f.write(f"Profit Factor: {r['profit_factor']}\n")
            f.write(f"Total P&L:     NPR {r['total_pnl_npr']:,.0f}\n")
            f.write(f"Annual Return: {r['annual_ret_pct']:.2f}%\n")
            f.write(f"Sharpe:        {r['sharpe_ratio']:.2f}\n")
            f.write(f"Max Drawdown:  {r['max_drawdown_pct']:.2f}%\n")
            f.write(f"Alpha:         {r['alpha_vs_nepse']}%\n")
            f.write(f"Trail used:    {r.get('trail_trades',0)} trades "
                    f"({r.get('trail_wins',0)} wins)\n")
            f.write("\nSignal Breakdown:\n")
            for sig, s in r["signal_breakdown"].items():
                f.write(f"  {sig:<22} trades={s['trades']:>4} "
                        f"WR={s['win_rate']:>5.1f}% PF={s['pf']:>5.2f} "
                        f"PnL=NPR {s['net_pnl']:>8,.0f} "
                        f"trail={s.get('trail_used',0)}\n")

    log.info("CSV + summary saved: %s | %s | %s", p1, p2, pt)
    return p1, p2, pt


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

PERIOD_START = "2020-07-01"
PERIOD_END   = "2026-03-29"


def run_test_mode(symbol: str = "NABIL"):
    log.info("TEST MODE — %s", symbol)
    nepse_df   = load_nepse_index()
    sector_map = load_sector_map()
    engine     = BacktestEngine(
        mode="constrained", initial_capital=200_000,
        thresholds=DEFAULT_THRESHOLDS,
        ind_params=DEFAULT_IND_PARAMS,
        period_start=PERIOD_START, period_end=PERIOD_END,
    )
    trades = engine.run([symbol], nepse_df, sector_map=sector_map, verbose=True)
    print(f"\n{'='*60}\nTEST — {symbol}  ({len(trades)} trades)\n{'='*60}")
    print(f"{'Date':<12} {'Signal':<22} {'Entry':>8} {'Exit':>8} "
          f"{'Hold':>5} {'PnL':>10} {'Trail':>6} Result")
    print("-" * 85)
    for t in trades[:20]:
        trail_mark = "✓" if t.trail_activated else "-"
        print(f"{t.entry_date:<12} {t.primary_signal:<22} "
              f"{t.entry_price:>8.2f} {t.exit_price:>8.2f} "
              f"{t.hold_days:>5}d {t.net_pnl:>10,.0f}  {trail_mark:>5}  {t.result}")
    if trades:
        print_results(
            compute_results(trades, 200_000, nepse_df,
                            label=f"TEST {symbol}")
        )


def run_optimize_mode() -> tuple:
    """
    Full optimization pipeline:
      1. Optimize signal thresholds (entry/exit rules + trailing params)
      2. Optimize indicator params (RSI period, MACD spans, BB period etc.)
      3. Validate on hold-out period
    Returns (best_thresholds, best_ind_params, symbol_cache, symbols, nepse_df)
    """
    log.info("OPTIMIZE MODE — threshold + indicator optimization")
    symbols    = load_all_symbols()
    nepse_df   = load_nepse_index()
    sector_map = load_sector_map()

    # Step 1: optimize signal thresholds with default indicator params
    log.info("\nStep 1/2: Optimizing signal thresholds...")
    best_thresh = optimize_thresholds(
        symbols=symbols, nepse_df=nepse_df, sector_map=sector_map,
        ind_params=DEFAULT_IND_PARAMS,
        training_start="2019-07-15", training_end="2022-12-31",
        n_workers=4,
    )

    # Step 2: optimize indicator params using best thresholds
    log.info("\nStep 2/2: Optimizing indicator construction params...")
    best_ind = optimize_indicators(
        symbols=symbols, nepse_df=nepse_df,
        base_thresholds=best_thresh,
        training_start="2019-07-15", training_end="2022-12-31",
        n_workers=4,
    )

    # Rebuild cache with optimized indicator params for validation
    temp = BacktestEngine(mode="free", ind_params=best_ind)
    symbol_cache = temp._load_and_compute(symbols)

    passed, _ = validate_thresholds(
        thresholds=best_thresh, symbols=symbols, nepse_df=nepse_df,
        sector_map=sector_map, symbol_cache=symbol_cache,
        ind_params=best_ind,
        validation_start="2023-01-01", validation_end="2024-06-30",
    )
    if not passed:
        log.warning("Validation failed — using defaults")
        best_thresh = DEFAULT_THRESHOLDS
        best_ind    = DEFAULT_IND_PARAMS

    return best_thresh, best_ind, symbol_cache, symbols, nepse_df


def run_full_backtest(
    thresholds:   SignalThresholds = None,
    ind_params:   IndicatorParams  = None,
    symbol_cache: dict = None,
    symbols:      list = None,
    nepse_df:     pd.DataFrame = None,
):
    """
    Run both simulations.
    Accepts pre-built cache/symbols/nepse_df from optimize mode to avoid
    redundant DB calls and indicator recomputation.
    """
    log.info("FULL BACKTEST MODE")

    if symbols is None:
        symbols = load_all_symbols()
    if nepse_df is None:
        nepse_df = load_nepse_index()

    thresholds = thresholds or DEFAULT_THRESHOLDS
    ind_params = ind_params  or DEFAULT_IND_PARAMS

    # Load sector map once for both sims
    sector_map = load_sector_map()

    # Build cache once if not already provided
    if symbol_cache is None:
        log.info("Building indicator cache for full backtest...")
        temp = BacktestEngine(mode="free", ind_params=ind_params)
        symbol_cache = temp._load_and_compute(symbols)

    # ── SIM 1 — CONSTRAINED ──────────────────────────────────────────────────
    log.info("\n%s\nSIM 1 — CONSTRAINED\n%s", "="*60, "="*60)
    sim1 = BacktestEngine(
        mode="constrained", initial_capital=200_000,
        thresholds=thresholds, ind_params=ind_params,
        period_start=PERIOD_START, period_end=PERIOD_END,
    )
    sim1._symbol_cache = symbol_cache
    r1 = compute_results(
        sim1.run(symbols, nepse_df, sector_map=sector_map),
        200_000, nepse_df,
        label="SIM1 CONSTRAINED (2020-2026)",
    )

    # ── SIM 2 — FREE RUN ─────────────────────────────────────────────────────
    log.info("\n%s\nSIM 2 — FREE RUN\n%s", "="*60, "="*60)
    sim2 = BacktestEngine(
        mode="free", initial_capital=200_000,
        thresholds=thresholds, ind_params=ind_params,
        period_start=PERIOD_START, period_end=PERIOD_END,
    )
    sim2._symbol_cache = symbol_cache
    r2 = compute_results(
        sim2.run(symbols, nepse_df, sector_map=sector_map),
        200_000, nepse_df,
        label="SIM2 FREE RUN (2020-2026)",
    )

    print_results(r1)
    print_results(r2)
    print_comparison(r1, r2)

    paths = save_results(r1, r2, thresholds, ind_params)
    save_to_db(r1, r2, thresholds, ind_params,
               period_start=PERIOD_START, period_end=PERIOD_END)

    return r1, r2, paths


# ── Windows multiprocessing guard ────────────────────────────────────────────
if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    import multiprocessing
    multiprocessing.freeze_support()   # required on Windows

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "run"

    if mode == "db-check":
        ok = run_db_check()
        sys.exit(0 if ok else 1)

    elif mode == "test":
        symbol = sys.argv[2] if len(sys.argv) > 2 else "NABIL"
        run_test_mode(symbol)

    elif mode == "optimize":
        best_t, best_i, _, _, _ = run_optimize_mode()
        log.info("Optimization done. Best thresholds and indicator params locked.")
        log.info("Run 'python -m analysis.backtester run' to backtest with them.")

    elif mode == "run":
        run_full_backtest()

    elif mode == "full":
        log.info("FULL PIPELINE: optimize → validate → backtest → DB save")
        best_t, best_i, cache, syms, nepse = run_optimize_mode()
        run_full_backtest(
            thresholds=best_t,
            ind_params=best_i,
            symbol_cache=cache,
            symbols=syms,
            nepse_df=nepse,
        )
        log.info("Done. Results in DB + outputs/")

    else:
        print("""
Usage:
  python -m analysis.backtester db-check              # pre-flight DB test (run first!)
  python -m analysis.backtester test [SYMBOL]         # single symbol (default: NABIL)
  python -m analysis.backtester optimize              # grid search, cached indicators
  python -m analysis.backtester run                   # full backtest, default thresholds
  python -m analysis.backtester full                  # optimize + validate + full backtest
        """)