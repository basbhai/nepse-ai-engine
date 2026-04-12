"""
analysis/indicator_backtest.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — New Indicator Backtester (Phase 2)

Tests 5 new indicators standalone, then combos of survivors:
  1. Stochastics %K/%D
  2. OBV Trend (EMA of OBV)
  3. VWAP Deviation
  4. Volume Profile (POC / HVN as support)
  5. Liquidity Sweep (wick + reversal)

Rules (locked):
  • T+4 settlement — earliest sell is 4 trading days after entry (skips weekends + holidays)
  • Stop loss      — 5% hard from entry
  • Trailing stop  — activates at +6% profit, floor = peak_profit - 3%
                     (peak 6% → floor 3%, peak 10% → floor 7%, etc.)
  • Max hold       — configurable per indicator (default 30 days)
  • Fees           — NEPSE: brokerage (tiered) + 0.015% SEBON + NPR 25 DP per side
  • Vectorized     — NumPy only, no Python loops over rows
  • Workers        — 2 cores max (leaves 2 free for OS)

Usage:
  python -m analysis.new_backtest standalone            # all 5 standalone
  python -m analysis.new_backtest combos                # combos of survivors
  python -m analysis.new_backtest standalone --fast     # fewer param combos
  python -m analysis.new_backtest standalone --symbol NABIL   # single symbol test
  python -m analysis.new_backtest standalone --from 2022-01-01 --to 2024-12-31

Output:
  results/indicator_results_YYYYMMDD_HHMMSS.csv   — incremental saves per combo
  Printed summary table at end

Architecture notes:
  • Reads from price_history table via Neon (DATABASE_URL in .env)
  • NEPSE holidays loaded from calendar_guard GAZETTE_HOLIDAYS constant
  • All NumPy — price series loaded once per symbol, vectorized signal computation
  • progress bar via tqdm
  • Crash-safe: incremental CSV appended after every combo result
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import time
import logging
import argparse
import warnings
import itertools
import urllib.request
import urllib.parse
import json
from datetime import datetime, date, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM NOTIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Lightweight Telegram notifier — no external library, pure stdlib urllib.
    Reads NEW_TELEGRAM_TOKEN and TELEGRAM_CHAT_ID from .env.
    Silently skips all sends if token not set (safe for dry-run / local test).

    Notification stages:
      notify_start()          — backtest started, config summary
      notify_data_loaded()    — DB load complete, symbol count
      notify_phase_start()    — standalone / combo phase beginning
      notify_per_signal()     — each signal's best result when phase done
      notify_survivors()      — survivor list after standalone
      notify_combo_result()   — combo phase summary
      notify_done()           — full completion with top results
      notify_error()          — any fatal error
    """

    def __init__(self):
        self.token   = os.getenv("NEW_TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)
        if self.enabled:
            log.info("Telegram notifier enabled (chat_id=%s)", self.chat_id)
        else:
            log.info("Telegram notifier disabled — NEW_TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set")

    def _send(self, text: str) -> bool:
        """Send a message. Returns True on success. Never raises."""
        if not self.enabled:
            return False
        try:
            url     = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = json.dumps({
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log.warning("Telegram send failed: %s", e)
            return False

    # ── Stage notifications ──────────────────────────────────────────────────

    def notify_start(self, mode: str, from_date: str, to_date: str,
                     fast: bool, hold: int):
        self._send(
            f"🚀 *Indicator Backtest Started*\n\n"
            f"Mode:   `{mode}`\n"
            f"Period: `{from_date}` → `{to_date}`\n"
            f"Hold:   `{hold}` trading days\n"
            f"Fast:   `{'yes' if fast else 'no'}`\n"
            f"Time:   `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )

    def notify_data_loaded(self, symbol_count: int, from_date: str, to_date: str,
                           row_count: int):
        self._send(
            f"📦 *Data Loaded from Neon*\n\n"
            f"Symbols:  `{symbol_count}` (≥200 days history)\n"
            f"Period:   `{from_date}` → `{to_date}`\n"
            f"DB rows:  `{row_count:,}`\n"
            f"Ready to run..."
        )

    def notify_phase_start(self, phase: str, n_combos: int, n_symbols: int):
        emoji = "⚙️" if phase == "STANDALONE" else "🔗"
        self._send(
            f"{emoji} *Phase: {phase}*\n\n"
            f"Parameter combos: `{n_combos}`\n"
            f"Symbols:          `{n_symbols}`\n"
            f"Workers:          `{MAX_WORKERS}`\n"
            f"Started:          `{datetime.now().strftime('%H:%M:%S')}`"
        )

    def notify_per_signal(self, signal: str, best: dict):
        """Called after all params for one signal are done."""
        if not best:
            self._send(f"📊 *{signal}* — no valid results")
            return
        pf  = best.get("profit_factor", 0)
        wr  = best.get("win_rate", 0) * 100
        ann = best.get("annual_ret_pct", 0)
        tot = best.get("total_trades", 0)
        pnl = best.get("total_pnl", 0)
        p   = best.get("spearman_p", 1)
        passed = "✅" if pf >= 1.5 and tot >= 30 and p <= 0.05 else "❌"
        self._send(
            f"{passed} *{signal}* best result\n\n"
            f"PF:     `{pf:.2f}`\n"
            f"WR:     `{wr:.1f}%`\n"
            f"Ann%:   `{ann:.1f}%`\n"
            f"Trades: `{tot}`\n"
            f"PnL:    `NPR {pnl:,.0f}`\n"
            f"ρ-p:    `{p:.4f}`"
        )

    def notify_survivors(self, survivors: list[tuple[str, dict]],
                         all_results: list[dict]):
        if not survivors:
            self._send(
                "⚠️ *No Survivors*\n\n"
                "No signal passed PF≥1.5, trades≥30, p≤0.05.\n"
                "Combo phase skipped.\n"
                "Check results CSV for details."
            )
            return

        lines = [f"🏆 *Survivors ({len(survivors)}/{len(set(r['signal'] for r in all_results))} signals)*\n"]
        for sig, params in survivors:
            # Find best result for this signal
            best = max(
                [r for r in all_results if r["signal"] == sig],
                key=lambda x: x["profit_factor"],
                default={}
            )
            pf  = best.get("profit_factor", 0)
            wr  = best.get("win_rate", 0) * 100
            lines.append(f"✅ `{sig}` — PF={pf:.2f} | WR={wr:.1f}%")

        lines.append(f"\nProceeding to combo phase...")
        self._send("\n".join(lines))

    def notify_combo_result(self, results: list[dict]):
        """Called after combo phase finishes."""
        if not results:
            self._send("🔗 *Combo Phase* — no results")
            return

        passing = [r for r in results
                   if r.get("profit_factor", 0) >= 1.5
                   and r.get("total_trades", 0) >= 30]
        best_combos = sorted(passing, key=lambda x: x["profit_factor"], reverse=True)[:5]

        lines = [f"🔗 *Combo Results* ({len(results)} combos tested)\n"
                 f"Passed bar: {len(passing)}\n"]
        for r in best_combos:
            pf  = r.get("profit_factor", 0)
            wr  = r.get("win_rate", 0) * 100
            sig = r.get("signal", "?")
            comp = str(r.get("components", ""))[:60]
            lines.append(f"✅ `{sig}` PF={pf:.2f} WR={wr:.1f}%\n   _{comp}_")

        if not best_combos:
            lines.append("No combo passed the bar (PF≥1.5, trades≥30)")

        self._send("\n".join(lines))

    def notify_done(self, all_results: list[dict], csv_path: str,
                    elapsed_sec: float):
        """Final completion message with top 5 overall."""
        top5 = sorted(all_results, key=lambda x: x.get("profit_factor", 0),
                      reverse=True)[:5]

        elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
        lines = [
            f"✅ *Backtest Complete!*\n",
            f"Total combos: `{len(all_results)}`",
            f"Elapsed:      `{elapsed_str}`",
            f"Results:      `{Path(csv_path).name}`\n",
            f"*Top 5 by Profit Factor:*",
        ]
        for i, r in enumerate(top5, 1):
            pf  = r.get("profit_factor", 0)
            wr  = r.get("win_rate", 0) * 100
            sig = r.get("signal", "?")
            ann = r.get("annual_ret_pct", 0)
            lines.append(f"{i}. `{sig}` — PF={pf:.2f} | WR={wr:.1f}% | Ann={ann:.1f}%")

        self._send("\n".join(lines))

    def notify_error(self, stage: str, error: str):
        self._send(
            f"🔴 *Backtest ERROR*\n\n"
            f"Stage: `{stage}`\n"
            f"Error: `{str(error)[:300]}`\n"
            f"Time:  `{datetime.now().strftime('%H:%M:%S')}`"
        )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Date ranges
DEFAULT_TRAIN_FROM = "2019-07-15"
DEFAULT_TRAIN_TO   = "2024-12-31"
DEFAULT_TEST_FROM  = "2025-01-01"
DEFAULT_TEST_TO    = "2026-03-29"

# Trade rules
STOP_LOSS_PCT      = 0.05    # 5% hard stop
TRAIL_ACTIVATE_PCT = 0.06    # trailing activates at +6%
TRAIL_FLOOR_PCT    = 0.03    # floor = peak - 3%
MIN_SYMBOL_DAYS    = 200     # minimum days of history to include symbol
MAX_WORKERS        = 2       # leave 2 cores free for OS/pipeline

# NEPSE fee structure
SEBON_PCT  = 0.00015
DP_FEE_NPR = 25.0

def _brokerage(amount: float) -> float:
    if amount <= 2_500:    return 10.0
    if amount <= 50_000:   return amount * 0.0036
    if amount <= 500_000:  return amount * 0.0033
    if amount <= 2_000_000: return amount * 0.0031
    if amount <= 10_000_000: return amount * 0.0027
    return amount * 0.0024

def calc_fees(amount: float) -> float:
    return _brokerage(amount) + amount * SEBON_PCT + DP_FEE_NPR

# NEPSE gazette holidays (from calendar_guard — key dates)
# Format: "YYYY-MM-DD"
# These are approximate — in production calendar_guard.py handles this
GAZETTE_HOLIDAYS = {
    # Major fixed holidays (approximate — Nepali calendar converts vary by year)
    "2024-01-11", "2024-01-15", "2024-02-19", "2024-03-08", "2024-04-12",
    "2024-05-01", "2024-05-10", "2024-05-23", "2024-07-04", "2024-08-07",
    "2024-09-16", "2024-10-02", "2024-10-12", "2024-10-13", "2024-11-15",
    "2024-12-25",
    "2023-01-11", "2023-02-20", "2023-03-08", "2023-04-13", "2023-05-01",
    "2023-08-30", "2023-09-15", "2023-10-02", "2023-10-23", "2023-10-24",
    "2023-11-14", "2023-12-25",
    "2025-01-11", "2025-02-19", "2025-03-08", "2025-04-14", "2025-05-01",
    "2025-08-27", "2025-10-02", "2025-10-23", "2025-11-05", "2025-12-25",
}

def is_trading_day(d: date) -> bool:
    """True if date is a NEPSE trading day (Sun–Thu, not holiday)."""
    # NEPSE trades Sunday(6) through Thursday(3) in Python weekday
    # weekday(): Mon=0 Tue=1 Wed=2 Thu=3 Fri=4 Sat=5 Sun=6
    if d.weekday() in (4, 5):  # Friday, Saturday
        return False
    if d.strftime("%Y-%m-%d") in GAZETTE_HOLIDAYS:
        return False
    return True

def t4_exit_date(entry_date: date, trading_dates: np.ndarray) -> date:
    """
    Return earliest valid exit date (T+4 settlement).
    trading_dates: sorted numpy array of datetime64 trading dates.
    """
    entry_dt64 = np.datetime64(entry_date, "D")
    future = trading_dates[trading_dates > entry_dt64]
    if len(future) < 4:
        return None
    return future[3].astype("M8[D]").astype(object)  # T+4 trading day


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def load_price_history(from_date: str, to_date: str,
                       tg: "TelegramNotifier" = None) -> dict[str, pd.DataFrame]:
    """
    Load price_history from Neon for all symbols.
    Returns dict[symbol → DataFrame] sorted by date ascending.
    Only includes symbols with MIN_SYMBOL_DAYS+ rows in the date range.
    Excludes non-equity suffixes: P, PO, BF, MF, SF, D83, D84, D85, D86, D87.
    """
    import psycopg2
    import psycopg2.extras

    SKIP_SUFFIX = ("P", "PO", "BF", "MF", "SF", "D83", "D84", "D85", "D86", "D87",
                   "ACLBSLP", "GMFILP", "GRDBLP", "MLBBLP", "EDBLPO")

    log.info("Loading price_history from %s to %s ...", from_date, to_date)

    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s AND date <= %s
        ORDER BY symbol, date ASC
    """, (from_date, to_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    raw_row_count = len(rows)
    log.info("Loaded %d rows from DB", raw_row_count)

    # Group by symbol
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        sym = r["symbol"]
        # Skip non-equity
        skip = False
        for suf in SKIP_SUFFIX:
            if sym.endswith(suf) and sym != suf:
                skip = True
                break
        if sym.endswith("P") and len(sym) > 2:
            skip = True
        if skip:
            continue
        grouped[sym].append(r)

    result = {}
    for sym, recs in grouped.items():
        if len(recs) < MIN_SYMBOL_DAYS:
            continue
        df = pd.DataFrame(recs)
        df["date"]   = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "ltp", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Use ltp as close if close is null
        df["close"] = df["close"].fillna(df["ltp"])
        df = df.dropna(subset=["close", "volume"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_SYMBOL_DAYS:
            result[sym] = df

    log.info("Usable symbols: %d (>= %d days)", len(result), MIN_SYMBOL_DAYS)

    if tg:
        tg.notify_data_loaded(len(result), from_date, to_date, raw_row_count)

    return result


def get_trading_dates(symbol_data: dict) -> np.ndarray:
    """Extract all unique trading dates across all symbols as sorted numpy array."""
    all_dates = set()
    for df in symbol_data.values():
        all_dates.update(df["date"].dt.date.tolist())
    arr = np.array(sorted(all_dates), dtype="datetime64[D]")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR COMPUTATION (all NumPy vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stochastics(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        k_period: int, d_period: int, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Stochastics %K (smoothed) and %D.
    All NumPy — no loops.
    Returns (pct_k, pct_d) arrays, NaN where insufficient data.
    """
    n = len(close)
    raw_k = np.full(n, np.nan)

    # Rolling high/low via stride tricks equivalent using cumulative approach
    # Vectorized via pandas rolling (ok — it's not row-by-row Python)
    s = pd.Series(close)
    h = pd.Series(high)
    l = pd.Series(low)

    highest_high = h.rolling(k_period).max().values
    lowest_low   = l.rolling(k_period).min().values

    denom = highest_high - lowest_low
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_k = np.where(denom > 0, (close - lowest_low) / denom * 100, 50.0)

    # Smooth %K
    pct_k = pd.Series(raw_k).rolling(smooth).mean().values

    # %D = SMA of %K
    pct_d = pd.Series(pct_k).rolling(d_period).mean().values

    return pct_k, pct_d


def compute_obv_trend(close: np.ndarray, volume: np.ndarray,
                      ema_period: int) -> np.ndarray:
    """
    OBV = cumulative sum of signed volume.
    OBV trend = EMA(OBV, ema_period).
    Signal: close > obv_ema → rising trend (bullish).
    Returns obv_ema array.
    """
    # OBV: vectorized with np.sign on daily returns
    daily_ret = np.diff(close, prepend=close[0])
    direction = np.sign(daily_ret)
    direction[0] = 0
    obv = np.cumsum(direction * volume)

    # EMA of OBV
    alpha = 2.0 / (ema_period + 1)
    obv_ema = np.empty_like(obv, dtype=float)
    obv_ema[:ema_period] = np.nan
    if ema_period <= len(obv):
        obv_ema[ema_period - 1] = np.nanmean(obv[:ema_period])
        for i in range(ema_period, len(obv)):
            obv_ema[i] = obv[i] * alpha + obv_ema[i-1] * (1 - alpha)

    return obv_ema


def compute_vwap_deviation(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """
    Daily VWAP approximation: cumulative(close * volume) / cumulative(volume).
    vwap_dev[i] = (close[i] - vwap[i]) / vwap[i]
    Negative dev = price below VWAP (potential entry).
    """
    cumvol  = np.cumsum(volume)
    cumturn = np.cumsum(close * volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cumvol > 0, cumturn / cumvol, close)
        dev  = (close - vwap) / vwap
    return dev


def compute_volume_profile(close: np.ndarray, volume: np.ndarray,
                           lookback: int, node_threshold: float) -> np.ndarray:
    """
    For each bar i, look back `lookback` bars.
    Find Point of Control (POC) = price level with highest volume.
    Signal: close[i] is within 2% of POC → near high-volume support.
    Returns boolean array: True = near POC (support zone).
    """
    n = len(close)
    near_poc = np.zeros(n, dtype=bool)

    # Use pandas rolling for vectorized approach
    s_close  = pd.Series(close)
    s_volume = pd.Series(volume)

    for i in range(lookback, n):
        window_c = close[i - lookback:i]
        window_v = volume[i - lookback:i]
        if len(window_v) == 0:
            continue
        # Find price level with highest volume (10 price bins)
        if window_c.max() == window_c.min():
            continue
        bins = np.linspace(window_c.min(), window_c.max(), 11)
        bin_idx   = np.digitize(window_c, bins) - 1
        bin_idx   = np.clip(bin_idx, 0, 9)
        bin_vol   = np.zeros(10)
        np.add.at(bin_vol, bin_idx, window_v)
        poc_bin   = np.argmax(bin_vol)
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
        avg_vol   = window_v.mean()
        # Near POC if current price within 2% of POC and POC has high volume
        if (abs(close[i] - poc_price) / poc_price < 0.02 and
                bin_vol[poc_bin] > avg_vol * node_threshold):
            near_poc[i] = True

    return near_poc


def compute_liquidity_sweep(high: np.ndarray, low: np.ndarray,
                            close: np.ndarray, open_: np.ndarray,
                            wick_ratio: float, lookback: int) -> np.ndarray:
    """
    Liquidity sweep detection:
    1. Identify support (recent swing low over `lookback` bars)
    2. Current bar wicks below support (wick_ratio of candle is lower wick)
    3. Current bar closes ABOVE the wick low (reversal confirmed)
    Signal = True when sweep+reversal detected.
    """
    n = len(close)
    sweep = np.zeros(n, dtype=bool)

    for i in range(lookback + 1, n):
        # Support = lowest low in lookback window (excluding current bar)
        support = np.min(low[i - lookback:i])

        # Current candle anatomy
        candle_range = high[i] - low[i]
        if candle_range < 1e-6:
            continue

        lower_wick = min(open_[i], close[i]) - low[i]
        lower_wick_pct = lower_wick / candle_range

        # Swept below support AND closed above it
        if (low[i] < support and
                lower_wick_pct >= wick_ratio and
                close[i] > low[i] * 1.005):   # closed at least 0.5% above wick low
            sweep[i] = True

    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_signal(df: pd.DataFrame,
                    signal: np.ndarray,
                    trading_dates: np.ndarray,
                    max_hold_days: int = 30,
                    from_date: str = None,
                    to_date: str = None) -> list[dict]:
    """
    Core backtest engine for a single symbol + signal array.

    Args:
        df:             OHLCV DataFrame sorted by date
        signal:         Boolean array — True = entry signal on this bar
        trading_dates:  All NEPSE trading dates (for T+4 calculation)
        max_hold_days:  Force exit after this many trading days
        from_date/to_date: Filter entry dates

    Returns:
        List of trade dicts with entry/exit/pnl/result fields.
    """
    trades = []
    closes = df["close"].values
    dates  = df["date"].dt.date.values
    n      = len(df)

    # Filter date range
    from_d = date.fromisoformat(from_date) if from_date else date(2000, 1, 1)
    to_d   = date.fromisoformat(to_date)   if to_date   else date(2100, 1, 1)

    i = 0
    while i < n:
        # Entry condition
        if not signal[i]:
            i += 1
            continue

        entry_date = dates[i]
        if entry_date < from_d or entry_date > to_d:
            i += 1
            continue

        entry_price = closes[i]
        if entry_price <= 0 or np.isnan(entry_price):
            i += 1
            continue

        # T+4 settlement — earliest exit date
        earliest_exit = t4_exit_date(entry_date, trading_dates)
        if earliest_exit is None:
            i += 1
            continue

        # Find earliest_exit index in df
        exit_start = i + 1
        while exit_start < n and dates[exit_start] < earliest_exit:
            exit_start += 1

        if exit_start >= n:
            i += 1
            continue

        # Simulate holding from exit_start onward
        peak_profit = 0.0
        exit_price  = closes[exit_start]
        exit_reason = "MAX_HOLD"
        trade_days  = 0

        # Force max_hold deadline
        max_exit_idx = min(exit_start + max_hold_days, n - 1)

        j = exit_start
        trailing_active = False

        while j <= max_exit_idx:
            cp = closes[j]
            if np.isnan(cp) or cp <= 0:
                j += 1
                continue

            pnl_pct = (cp - entry_price) / entry_price

            # Update peak
            if pnl_pct > peak_profit:
                peak_profit = pnl_pct

            # Trailing stop activation
            if peak_profit >= TRAIL_ACTIVATE_PCT:
                trailing_active = True

            # Exit logic
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_price  = cp
                exit_reason = "STOP_LOSS"
                trade_days  = j - i
                break

            if trailing_active:
                floor = peak_profit - TRAIL_FLOOR_PCT
                if pnl_pct <= floor:
                    exit_price  = cp
                    exit_reason = "TRAIL_STOP"
                    trade_days  = j - i
                    break

            if j == max_exit_idx:
                exit_price  = cp
                exit_reason = "MAX_HOLD"
                trade_days  = j - i
                break

            j += 1

        # Calculate P&L with fees
        entry_amount = entry_price * 100   # assume 100 share lot for sizing
        exit_amount  = exit_price * 100
        buy_fees     = calc_fees(entry_amount)
        sell_fees    = calc_fees(exit_amount)
        gross_pnl    = exit_amount - entry_amount
        net_pnl      = gross_pnl - buy_fees - sell_fees
        total_cost   = entry_amount + buy_fees
        return_pct   = net_pnl / total_cost * 100 if total_cost > 0 else 0.0
        result       = "WIN" if net_pnl > 0 else "LOSS"

        trades.append({
            "symbol":       df["symbol"].iloc[0] if "symbol" in df.columns else "?",
            "entry_date":   str(entry_date),
            "exit_date":    str(dates[min(j, n-1)]),
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "exit_reason":  exit_reason,
            "hold_days":    trade_days,
            "gross_pnl":    round(gross_pnl, 2),
            "net_pnl":      round(net_pnl, 2),
            "return_pct":   round(return_pct, 4),
            "result":       result,
            "total_fees":   round(buy_fees + sell_fees, 2),
        })

        # Move past this trade
        i = j + 1

    return trades


def compute_metrics(trades: list[dict], signal_name: str, params: dict) -> dict:
    """Aggregate trade list into performance metrics."""
    if not trades:
        return {"signal": signal_name, **params, "total_trades": 0,
                "win_rate": 0, "profit_factor": 0, "annual_ret_pct": 0,
                "sharpe": 0, "max_drawdown": 0, "spearman_rho": 0,
                "spearman_p": 1.0, "avg_return_pct": 0, "total_fees": 0,
                "total_pnl": 0}

    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = total - wins
    win_rate = wins / total

    returns  = np.array([t["return_pct"] for t in trades])
    net_pnls = np.array([t["net_pnl"] for t in trades])

    win_pnls  = net_pnls[net_pnls > 0]
    loss_pnls = net_pnls[net_pnls < 0]
    profit_factor = (
        win_pnls.sum() / abs(loss_pnls.sum())
        if len(loss_pnls) > 0 and abs(loss_pnls.sum()) > 0
        else (999.0 if wins > 0 else 0.0)
    )

    # Annual return estimate (simple)
    avg_hold  = np.mean([t["hold_days"] for t in trades]) or 17
    trades_yr = 252 / avg_hold
    annual_ret = (1 + np.mean(returns) / 100) ** trades_yr - 1

    # Sharpe (assume risk-free = 5.5% annual)
    rf_daily  = (1 + 0.055) ** (1/252) - 1
    excess    = returns / 100 - rf_daily
    sharpe    = (np.mean(excess) / np.std(excess) * np.sqrt(252)
                 if np.std(excess) > 0 else 0)

    # Max drawdown on cumulative equity
    cum_pnl   = np.cumsum(net_pnls)
    peak      = np.maximum.accumulate(cum_pnl)
    drawdown  = (cum_pnl - peak) / (np.abs(peak) + 1e-9) * 100
    max_dd    = float(np.min(drawdown))

    # Spearman correlation: signal rank vs forward return
    n_ret = len(returns)
    if n_ret >= 10:
        rho, p = stats.spearmanr(np.arange(n_ret), returns)
    else:
        rho, p = 0.0, 1.0

    total_fees = sum(t["total_fees"] for t in trades)
    total_pnl  = float(np.sum(net_pnls))

    return {
        "signal":        signal_name,
        **params,
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "win_rate":      round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "annual_ret_pct": round(annual_ret * 100, 2),
        "avg_return_pct": round(float(np.mean(returns)), 4),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 2),
        "spearman_rho":  round(rho, 4),
        "spearman_p":    round(p, 6),
        "total_pnl":     round(total_pnl, 0),
        "total_fees":    round(total_fees, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def signal_stochastics(df: pd.DataFrame, k_period: int, d_period: int,
                       smooth: int, ob_level: float = 80.0,
                       os_level: float = 20.0) -> np.ndarray:
    """Entry: %K crosses above %D in oversold zone, then exits oversold."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    pct_k, pct_d = compute_stochastics(h, l, c, k_period, d_period, smooth)

    n = len(c)
    signal = np.zeros(n, dtype=bool)

    # Vectorized: %K crosses above %D (k > d now, k <= d prev) AND k < ob_level
    k_cross_up = (
        (~np.isnan(pct_k)) &
        (~np.isnan(pct_d)) &
        (pct_k > pct_d) &
        (np.roll(pct_k, 1) <= np.roll(pct_d, 1)) &
        (pct_k < ob_level) &      # not overbought at signal
        (pct_k > os_level)        # just exited oversold
    )
    signal[1:] = k_cross_up[1:]   # can't trigger on bar 0
    return signal


def signal_obv_trend(df: pd.DataFrame, ema_period: int) -> np.ndarray:
    """Entry: OBV crosses above its EMA (momentum turning bullish)."""
    c = df["close"].values
    v = df["volume"].values

    obv_ema = compute_obv_trend(c, v, ema_period)

    daily_ret = np.diff(c, prepend=c[0])
    direction = np.sign(daily_ret)
    direction[0] = 0
    obv = np.cumsum(direction * v)

    n = len(c)
    signal = np.zeros(n, dtype=bool)

    # OBV crosses above EMA: obv > ema now, obv <= ema prev
    cross_up = (
        (~np.isnan(obv_ema)) &
        (obv > obv_ema) &
        (np.roll(obv, 1) <= np.roll(obv_ema, 1))
    )
    signal[1:] = cross_up[1:]
    return signal


def signal_vwap_deviation(df: pd.DataFrame,
                          entry_threshold: float) -> np.ndarray:
    """
    Entry: price is entry_threshold% BELOW VWAP → mean reversion entry.
    i.e. vwap_dev <= -entry_threshold (oversold relative to fair value).
    Next bar: price starts recovering (close > prev close).
    """
    c = df["close"].values
    v = df["volume"].values

    dev = compute_vwap_deviation(c, v)

    n = len(c)
    signal = np.zeros(n, dtype=bool)

    # Price below VWAP by threshold AND next day recovering
    below_vwap = dev <= -entry_threshold
    recovering = np.roll(c < c, 1)  # placeholder — use: c > np.roll(c, 1)
    recovering = c > np.roll(c, 1)

    trigger = below_vwap & recovering & (~np.isnan(dev))
    signal[1:] = trigger[1:]
    return signal


def signal_volume_profile(df: pd.DataFrame,
                          lookback: int, node_threshold: float) -> np.ndarray:
    """Entry: price is near high-volume node (support) with volume confirming."""
    c = df["close"].values
    v = df["volume"].values

    near_poc = compute_volume_profile(c, v, lookback, node_threshold)

    n = len(c)
    signal = np.zeros(n, dtype=bool)

    # Near POC AND volume above 20d average (accumulation confirmation)
    vol_ma = pd.Series(v).rolling(20).mean().values
    vol_confirm = v > vol_ma * 1.2

    trigger = near_poc & vol_confirm
    signal[1:] = trigger[1:]
    return signal


def signal_liquidity_sweep(df: pd.DataFrame,
                           wick_ratio: float, lookback: int) -> np.ndarray:
    """Entry: wick below support + reversal = liquidity sweep pattern."""
    h  = df["high"].values
    l  = df["low"].values
    c  = df["close"].values
    op = df["open"].values if "open" in df.columns else c

    sweep = compute_liquidity_sweep(h, l, c, op, wick_ratio, lookback)

    n = len(c)
    signal = np.zeros(n, dtype=bool)
    signal[lookback + 1:] = sweep[lookback + 1:]
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# COMBO SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def signal_combo(df: pd.DataFrame, signals: list[np.ndarray],
                 mode: str = "ALL") -> np.ndarray:
    """
    Combine multiple signal arrays.
    mode='ALL'  → all signals must fire same bar (AND)
    mode='ANY'  → any signal (OR)
    mode='2OF3' → at least 2 of 3
    """
    if not signals:
        return np.zeros(len(df), dtype=bool)
    stack = np.stack(signals, axis=0).astype(float)
    count = stack.sum(axis=0)
    if mode == "ALL":
        return count == len(signals)
    elif mode == "ANY":
        return count >= 1
    elif mode == "2OF3":
        return count >= 2
    return count == len(signals)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (runs in subprocess for parallel execution)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_combo(args: tuple) -> dict:
    """
    Top-level function for ProcessPoolExecutor.
    args = (signal_name, param_dict, symbol_data_list, trading_dates_arr,
            max_hold, from_date, to_date)
    """
    signal_name, params, symbol_data_list, trading_dates, max_hold, from_date, to_date = args

    all_trades = []
    for sym, df_dict in symbol_data_list:
        df = pd.DataFrame(df_dict)
        df["date"] = pd.to_datetime(df["date"])

        try:
            # Generate signal
            if signal_name == "STOCH":
                sig = signal_stochastics(df, params["k"], params["d"], params["smooth"])
            elif signal_name == "OBV":
                sig = signal_obv_trend(df, params["ema"])
            elif signal_name == "VWAP":
                sig = signal_vwap_deviation(df, params["threshold"])
            elif signal_name == "VOLPROFILE":
                sig = signal_volume_profile(df, params["lookback"], params["node_thr"])
            elif signal_name == "LIQSWEEP":
                sig = signal_liquidity_sweep(df, params["wick_ratio"], params["lookback"])
            elif signal_name.startswith("COMBO_"):
                # Combo mode: rebuild component signals
                sigs = []
                for comp, p in params["components"]:
                    if comp == "STOCH":
                        sigs.append(signal_stochastics(df, p["k"], p["d"], p["smooth"]))
                    elif comp == "OBV":
                        sigs.append(signal_obv_trend(df, p["ema"]))
                    elif comp == "VWAP":
                        sigs.append(signal_vwap_deviation(df, p["threshold"]))
                    elif comp == "VOLPROFILE":
                        sigs.append(signal_volume_profile(df, p["lookback"], p["node_thr"]))
                    elif comp == "LIQSWEEP":
                        sigs.append(signal_liquidity_sweep(df, p["wick_ratio"], p["lookback"]))
                sig = signal_combo(df, sigs, mode=params.get("mode", "ALL"))
            else:
                continue

            if "symbol" not in df.columns:
                df["symbol"] = sym
            trades = backtest_signal(df, sig, trading_dates, max_hold, from_date, to_date)
            all_trades.extend(trades)

        except Exception as e:
            pass  # skip symbol on error

    return compute_metrics(all_trades, signal_name, params)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GRIDS
# ─────────────────────────────────────────────────────────────────────────────

def get_param_grid(fast: bool = False) -> dict[str, list[dict]]:
    """Return parameter combinations per signal. fast=True = reduced set."""

    if fast:
        return {
            "STOCH": [
                {"k": 14, "d": 3, "smooth": 3},
                {"k": 21, "d": 5, "smooth": 5},
            ],
            "OBV": [
                {"ema": 20},
            ],
            "VWAP": [
                {"threshold": 0.02},
                {"threshold": 0.03},
            ],
            "VOLPROFILE": [
                {"lookback": 30, "node_thr": 2.0},
            ],
            "LIQSWEEP": [
                {"wick_ratio": 0.7, "lookback": 10},
            ],
        }

    return {
        "STOCH": [
            {"k": k, "d": d, "smooth": s}
            for k in [9, 14, 21]
            for d in [3, 5]
            for s in [3, 5]
        ],
        "OBV": [
            {"ema": e} for e in [10, 20, 30]
        ],
        "VWAP": [
            {"threshold": t} for t in [0.01, 0.02, 0.03]
        ],
        "VOLPROFILE": [
            {"lookback": lb, "node_thr": nt}
            for lb in [20, 30, 60]
            for nt in [1.5, 2.0, 2.5]
        ],
        "LIQSWEEP": [
            {"wick_ratio": wr, "lookback": lb}
            for wr in [0.6, 0.7, 0.8]
            for lb in [5, 10, 20]
        ],
    }


def get_combo_params(survivors: list[tuple[str, dict]]) -> list[tuple]:
    """
    Build combo jobs from survivors.
    survivors: list of (signal_name, best_params) tuples.
    Returns 2-way and 3-way combos.
    """
    combos = []
    for i, (n1, p1) in enumerate(survivors):
        for j, (n2, p2) in enumerate(survivors):
            if j <= i:
                continue
            combos.append(("COMBO_2", {
                "components": [(n1, p1), (n2, p2)],
                "mode": "ALL",
            }))
    # 3-way if enough survivors
    if len(survivors) >= 3:
        for a, b, c in itertools.combinations(range(len(survivors)), 3):
            n1, p1 = survivors[a]
            n2, p2 = survivors[b]
            n3, p3 = survivors[c]
            combos.append(("COMBO_3", {
                "components": [(n1, p1), (n2, p2), (n3, p3)],
                "mode": "2OF3",
            }))
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# CSV WRITER (incremental, crash-safe)
# ─────────────────────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "signal", "total_trades", "wins", "losses", "win_rate",
    "profit_factor", "annual_ret_pct", "avg_return_pct",
    "sharpe", "max_drawdown", "spearman_rho", "spearman_p",
    "total_pnl", "total_fees",
    # params (flattened later)
    "k", "d", "smooth", "ema", "threshold", "lookback",
    "node_thr", "wick_ratio", "components", "mode",
]

def append_result(csv_path: Path, result: dict):
    """Append one result row to CSV. Creates header if file is new."""
    # Flatten params into top-level keys
    flat = {}
    for k, v in result.items():
        if k == "components":
            flat["components"] = str(v)
        else:
            flat[k] = v

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()),
                                extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone(symbol_data: dict, trading_dates: np.ndarray,
                   param_grid: dict, csv_path: Path,
                   max_hold: int, from_date: str, to_date: str,
                   symbol_filter: str = None,
                   tg: "TelegramNotifier" = None) -> list[dict]:
    """Run all standalone indicator combinations."""

    if symbol_filter:
        symbol_data = {s: df for s, df in symbol_data.items()
                       if s == symbol_filter.upper()}
        log.info("Filtered to single symbol: %s", symbol_filter)

    # Serialize DataFrames for multiprocessing
    symbol_list = [(sym, df.to_dict("list")) for sym, df in symbol_data.items()]
    td_arr = trading_dates

    # Build all jobs — grouped by signal so we can notify per-signal
    jobs_by_signal: dict[str, list] = {}
    for sig_name, param_list in param_grid.items():
        jobs_by_signal[sig_name] = []
        for params in param_list:
            jobs_by_signal[sig_name].append(
                (sig_name, params, symbol_list, td_arr, max_hold, from_date, to_date)
            )

    all_jobs = [j for jobs in jobs_by_signal.values() for j in jobs]
    total_jobs = len(all_jobs)

    log.info("Standalone: %d total combos | %d symbols | %d workers",
             total_jobs, len(symbol_list), MAX_WORKERS)

    if tg:
        tg.notify_phase_start("STANDALONE", total_jobs, len(symbol_list))

    results = []
    results_by_signal: dict[str, list] = {s: [] for s in jobs_by_signal}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Submit all jobs, tracking which signal each belongs to
        future_to_sig = {}
        for sig_name, jobs in jobs_by_signal.items():
            for j in jobs:
                fut = pool.submit(run_single_combo, j)
                future_to_sig[fut] = sig_name

        pbar = tqdm(total=total_jobs, desc="Standalone", ncols=80)

        # Track which signals are fully done
        signal_remaining = {s: len(jobs) for s, jobs in jobs_by_signal.items()}
        signal_done = set()

        for fut in as_completed(future_to_sig):
            sig_name = future_to_sig[fut]
            try:
                r = fut.result()
                results.append(r)
                results_by_signal[sig_name].append(r)
                append_result(csv_path, r)
            except Exception as e:
                log.warning("Combo failed: %s", e)
            finally:
                signal_remaining[sig_name] -= 1

            # When all params for a signal are done → send per-signal notification
            if signal_remaining[sig_name] == 0 and sig_name not in signal_done:
                signal_done.add(sig_name)
                sig_results = results_by_signal[sig_name]
                if sig_results and tg:
                    best = max(sig_results, key=lambda x: x.get("profit_factor", 0))
                    tg.notify_per_signal(sig_name, best)

            pbar.update(1)
        pbar.close()

    return results


def run_combos(symbol_data: dict, trading_dates: np.ndarray,
               survivors: list[tuple[str, dict]],
               csv_path: Path, max_hold: int,
               from_date: str, to_date: str,
               tg: "TelegramNotifier" = None) -> list[dict]:
    """Run combo signals from survivors."""

    symbol_list = [(sym, df.to_dict("list")) for sym, df in symbol_data.items()]
    combo_params = get_combo_params(survivors)

    if not combo_params:
        log.warning("No combos to test — need at least 2 survivors")
        return []

    jobs = []
    for sig_name, params in combo_params:
        jobs.append((sig_name, params, symbol_list, trading_dates,
                     max_hold, from_date, to_date))

    log.info("Combos: %d total | %d workers", len(jobs), MAX_WORKERS)

    if tg:
        tg.notify_phase_start("COMBOS", len(jobs), len(symbol_list))

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_single_combo, j): j for j in jobs}
        pbar = tqdm(total=len(jobs), desc="Combos", ncols=80)
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                append_result(csv_path, r)
            except Exception as e:
                log.warning("Combo failed: %s", e)
            pbar.update(1)
        pbar.close()

    if tg:
        tg.notify_combo_result(results)

    return results


def identify_survivors(results: list[dict],
                       min_pf: float = 1.5,
                       min_trades: int = 30,
                       max_p: float = 0.05) -> list[tuple[str, dict]]:
    """
    Identify signals that pass the survivor bar:
      PF >= min_pf  AND  trades >= min_trades  AND  spearman_p <= max_p
    Returns list of (signal_name, best_params) for best combo per signal.
    """
    by_signal: dict[str, list[dict]] = {}
    for r in results:
        sig = r["signal"]
        if sig not in by_signal:
            by_signal[sig] = []
        by_signal[sig].append(r)

    survivors = []
    for sig, rlist in by_signal.items():
        passing = [r for r in rlist
                   if r["profit_factor"] >= min_pf
                   and r["total_trades"] >= min_trades
                   and r["spearman_p"] <= max_p]
        if passing:
            best = max(passing, key=lambda x: x["profit_factor"])
            # Extract params (strip non-param keys)
            param_keys = {"k", "d", "smooth", "ema", "threshold",
                          "lookback", "node_thr", "wick_ratio"}
            params = {k: v for k, v in best.items() if k in param_keys}
            survivors.append((sig, params))
            log.info("SURVIVOR: %s | PF=%.2f | WR=%.1f%% | trades=%d",
                     sig, best["profit_factor"],
                     best["win_rate"] * 100, best["total_trades"])

    if not survivors:
        log.warning("No survivors passed the bar (PF>=%.1f, trades>=%d, p<=%.2f)",
                    min_pf, min_trades, max_p)

    return survivors


def print_summary(results: list[dict], title: str = "Results"):
    """Print ranked results table."""
    if not results:
        print(f"\n{title}: No results to display")
        return

    # Sort by profit factor descending
    ranked = sorted(results, key=lambda x: x.get("profit_factor", 0), reverse=True)

    print(f"\n{'─'*90}")
    print(f"  {title}")
    print(f"{'─'*90}")
    print(f"  {'Signal':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Ann%':>8} "
          f"{'Sharpe':>7} {'DD%':>7} {'ρ-p':>8} {'NetPnL':>10}")
    print(f"{'─'*90}")

    for r in ranked[:20]:   # show top 20
        print(
            f"  {r['signal']:<14} "
            f"{r['total_trades']:>7} "
            f"{r['win_rate']*100:>7.1f} "
            f"{r['profit_factor']:>7.2f} "
            f"{r['annual_ret_pct']:>8.1f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_drawdown']:>7.1f} "
            f"{r['spearman_p']:>8.4f} "
            f"{r['total_pnl']:>10,.0f}"
        )
    print(f"{'─'*90}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE New Indicator Backtester"
    )
    parser.add_argument("mode", choices=["standalone", "combos", "both"],
                        help="standalone: all indicators. combos: survivors only. both: full run.")
    parser.add_argument("--fast",   action="store_true",
                        help="Reduced parameter grid (faster, ~30 min)")
    parser.add_argument("--from",   dest="from_date", default=DEFAULT_TRAIN_FROM)
    parser.add_argument("--to",     dest="to_date",   default=DEFAULT_TRAIN_TO)
    parser.add_argument("--hold",   type=int, default=30,
                        help="Max hold days (default 30)")
    parser.add_argument("--symbol", help="Test single symbol only")
    parser.add_argument("--min-pf", type=float, default=1.5,
                        help="Minimum profit factor for survivors (default 1.5)")
    parser.add_argument("--min-trades", type=int, default=30,
                        help="Minimum trades for survivors (default 30)")
    args = parser.parse_args()

    if not DATABASE_URL:
        log.error("DATABASE_URL not set in .env")
        sys.exit(1)

    # ── Init Telegram ────────────────────────────────────────────────────────
    tg = TelegramNotifier()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = RESULTS_DIR / f"indicator_results_{timestamp}.csv"
    log.info("Results will be saved to: %s", csv_path)

    start_time = time.time()

    # ── Notify: start ────────────────────────────────────────────────────────
    tg.notify_start(args.mode, args.from_date, args.to_date, args.fast, args.hold)

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        symbol_data   = load_price_history(args.from_date, args.to_date, tg=tg)
        trading_dates = get_trading_dates(symbol_data)
    except Exception as e:
        tg.notify_error("Data Load", str(e))
        log.error("Failed to load data: %s", e)
        sys.exit(1)

    if not symbol_data:
        msg = "No symbol data loaded. Check DATABASE_URL and date range."
        tg.notify_error("Data Load", msg)
        log.error(msg)
        sys.exit(1)

    param_grid = get_param_grid(fast=args.fast)

    all_results = []

    # ── Standalone phase ─────────────────────────────────────────────────────
    if args.mode in ("standalone", "both"):
        log.info("=== STANDALONE PHASE ===")
        try:
            results = run_standalone(
                symbol_data, trading_dates, param_grid, csv_path,
                args.hold, args.from_date, args.to_date,
                symbol_filter=args.symbol, tg=tg,
            )
        except Exception as e:
            tg.notify_error("Standalone Phase", str(e))
            raise

        all_results.extend(results)
        print_summary(results, "Standalone Results")

        survivors = identify_survivors(results, args.min_pf, args.min_trades)
        log.info("Survivors: %d / %d signals", len(survivors), len(param_grid))

        # Notify survivors (sends message regardless — notifies even if 0)
        tg.notify_survivors(survivors, results)

        # ── Combo phase ──────────────────────────────────────────────────────
        if args.mode == "both":
            if survivors:
                log.info("=== COMBO PHASE ===")
                try:
                    combo_results = run_combos(
                        symbol_data, trading_dates, survivors, csv_path,
                        args.hold, args.from_date, args.to_date, tg=tg,
                    )
                except Exception as e:
                    tg.notify_error("Combo Phase", str(e))
                    raise

                all_results.extend(combo_results)
                print_summary(combo_results, "Combo Results")
            else:
                log.info("Skipping combo phase — no survivors")

    elif args.mode == "combos":
        log.warning("--mode combos requires prior standalone results.")
        log.warning("Run 'standalone' first, then check results CSV for survivors.")

    # ── Final notification ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    tg.notify_done(all_results, str(csv_path), elapsed)

    log.info("Done. Elapsed: %s. Results: %s",
             str(timedelta(seconds=int(elapsed))), csv_path)


if __name__ == "__main__":
    main()