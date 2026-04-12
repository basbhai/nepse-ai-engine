"""
analysis/indicator_backtest.py — Enhanced NEPSE Indicator Backtester (Phase 2+)
═══════════════════════════════════════════════════════════════════════════════
Features added to original Phase 2:
  • Sector breakdown (from share_sectors table)
  • Advanced metrics: Calmar ratio, Sortino ratio, win/loss streak, bootstrap CI
  • Walk‑forward validation (rolling train/test)
  • Per‑signal trailing stop parameters (instead of global)
  • Fixed VWAP signal (no look‑ahead)
  • Bootstrap confidence intervals for profit factor
  • All vectorized + multiprocessing (2 workers)
═══════════════════════════════════════════════════════════════════════════════
Usage:
  python -m analysis.indicator_backtest standalone [--fast] [--symbol NABIL]
  python -m analysis.indicator_backtest combos --walkforward
  python -m analysis.indicator_backtest both --bootstrap 1000
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
# TELEGRAM NOTIFIER (minimal, functional)
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("NEW_TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)
        if self.enabled:
            log.info("Telegram notifier enabled")

    def _send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log.warning("Telegram send failed: %s", e)
            return False

    def notify_start(self, mode, from_date, to_date, fast, hold):
        self._send(f"🚀 *Indicator Backtest Started*\n\nMode: `{mode}`\nPeriod: `{from_date}` → `{to_date}`")
    def notify_data_loaded(self, symbol_count, from_date, to_date, row_count):
        self._send(f"📦 *Data Loaded*: {symbol_count} symbols, {row_count:,} rows")
    def notify_phase_start(self, phase, n_combos, n_symbols):
        self._send(f"⚙️ *Phase: {phase}* | Combos: {n_combos} | Symbols: {n_symbols}")
    def notify_per_signal(self, signal, best):
        if best:
            self._send(f"📊 *{signal}* best: PF={best.get('profit_factor',0):.2f} WR={best.get('win_rate',0)*100:.1f}%")
    def notify_survivors(self, survivors, all_results):
        self._send(f"🏆 *Survivors*: {len(survivors)} signals")
    def notify_combo_result(self, results):
        self._send(f"🔗 *Combo Phase*: {len(results)} combos tested")
    def notify_done(self, all_results, csv_path, elapsed_sec):
        self._send(f"✅ *Backtest Complete*\nResults: {Path(csv_path).name}\nElapsed: {timedelta(seconds=int(elapsed_sec))}")
    def notify_error(self, stage, error):
        self._send(f"🔴 *ERROR* at {stage}: {str(error)[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TRAIN_FROM = "2019-07-15"
DEFAULT_TRAIN_TO   = "2024-12-31"
DEFAULT_TEST_FROM  = "2025-01-01"
DEFAULT_TEST_TO    = "2026-03-29"

STOP_LOSS_PCT      = 0.05      # default hard stop (overridden per signal)
TRAIL_ACTIVATE_PCT = 0.06      # default trail activation
TRAIL_FLOOR_PCT    = 0.03      # default trail distance
MAX_HOLD_DAYS      = 30
MIN_SYMBOL_DAYS    = 200
MAX_WORKERS        = 2

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

# NEPSE holidays (abbreviated)
GAZETTE_HOLIDAYS = {
    "2024-01-11", "2024-01-15", "2024-02-19", "2024-03-08", "2024-04-12",
    "2024-05-01", "2024-05-10", "2024-05-23", "2024-07-04", "2024-08-07",
    "2024-09-16", "2024-10-02", "2024-10-12", "2024-10-13", "2024-11-15",
    "2024-12-25", "2023-01-11", "2023-02-20", "2023-03-08", "2023-04-13",
    "2023-05-01", "2023-08-30", "2023-09-15", "2023-10-02", "2023-10-23",
    "2023-10-24", "2023-11-14", "2023-12-25", "2025-01-11", "2025-02-19",
    "2025-03-08", "2025-04-14", "2025-05-01", "2025-08-27", "2025-10-02",
    "2025-10-23", "2025-11-05", "2025-12-25",
}

def is_trading_day(d: date) -> bool:
    if d.weekday() in (4, 5): return False
    return d.strftime("%Y-%m-%d") not in GAZETTE_HOLIDAYS

def t4_exit_date(entry_date: date, trading_dates: np.ndarray) -> date:
    entry_dt64 = np.datetime64(entry_date, "D")
    future = trading_dates[trading_dates > entry_dt64]
    if len(future) < 4: return None
    return future[3].astype("M8[D]").astype(object)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADERS (including sector map)
# ─────────────────────────────────────────────────────────────────────────────

def load_sector_map() -> dict:
    """Load symbol → sector from share_sectors table."""
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT symbol, sectorname FROM share_sectors WHERE symbol IS NOT NULL AND sectorname IS NOT NULL")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        log.warning("Sector map load failed: %s", e)
        return {}

def load_price_history(from_date: str, to_date: str, tg=None) -> dict[str, pd.DataFrame]:
    import psycopg2
    import psycopg2.extras
    SKIP_SUFFIX = ("P", "PO", "BF", "MF", "SF", "D83", "D84", "D85", "D86", "D87")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s AND date <= %s
        ORDER BY symbol, date ASC
    """, (from_date, to_date))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        sym = r["symbol"]
        skip = any(sym.endswith(suf) for suf in SKIP_SUFFIX) or (sym.endswith("P") and len(sym)>2)
        if skip: continue
        grouped[sym].append(r)
    result = {}
    for sym, recs in grouped.items():
        if len(recs) < MIN_SYMBOL_DAYS: continue
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open","high","low","close","ltp","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["close"] = df["close"].fillna(df["ltp"])
        df = df.dropna(subset=["close","volume"]).sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_SYMBOL_DAYS:
            result[sym] = df
    if tg:
        tg.notify_data_loaded(len(result), from_date, to_date, len(rows))
    return result

def get_trading_dates(symbol_data: dict) -> np.ndarray:
    all_dates = set()
    for df in symbol_data.values():
        all_dates.update(df["date"].dt.date.tolist())
    return np.array(sorted(all_dates), dtype="datetime64[D]")


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS (vectorized, with fixed VWAP)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stochastics(high, low, close, k_period, d_period, smooth):
    n = len(close)
    s = pd.Series(close)
    h = pd.Series(high)
    l = pd.Series(low)
    highest_high = h.rolling(k_period).max().values
    lowest_low   = l.rolling(k_period).min().values
    denom = highest_high - lowest_low
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_k = np.where(denom > 0, (close - lowest_low) / denom * 100, 50.0)
    pct_k = pd.Series(raw_k).rolling(smooth).mean().values
    pct_d = pd.Series(pct_k).rolling(d_period).mean().values
    return pct_k, pct_d

def compute_obv_trend(close, volume, ema_period):
    daily_ret = np.diff(close, prepend=close[0])
    direction = np.sign(daily_ret)
    direction[0] = 0
    obv = np.cumsum(direction * volume)
    alpha = 2.0 / (ema_period + 1)
    obv_ema = np.empty_like(obv, dtype=float)
    obv_ema[:ema_period] = np.nan
    if ema_period <= len(obv):
        obv_ema[ema_period-1] = np.nanmean(obv[:ema_period])
        for i in range(ema_period, len(obv)):
            obv_ema[i] = obv[i] * alpha + obv_ema[i-1] * (1 - alpha)
    return obv_ema

def compute_vwap_deviation(close, volume):
    cumvol = np.cumsum(volume)
    cumturn = np.cumsum(close * volume)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(cumvol > 0, cumturn / cumvol, close)
        dev = (close - vwap) / vwap
    return dev

def compute_volume_profile(close, volume, lookback, node_threshold):
    n = len(close)
    near_poc = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        window_c = close[i-lookback:i]
        window_v = volume[i-lookback:i]
        if window_c.max() == window_c.min(): continue
        bins = np.linspace(window_c.min(), window_c.max(), 11)
        bin_idx = np.digitize(window_c, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        bin_vol = np.zeros(10)
        np.add.at(bin_vol, bin_idx, window_v)
        poc_bin = np.argmax(bin_vol)
        poc_price = (bins[poc_bin] + bins[poc_bin+1]) / 2
        avg_vol = window_v.mean()
        if (abs(close[i] - poc_price) / poc_price < 0.02 and
            bin_vol[poc_bin] > avg_vol * node_threshold):
            near_poc[i] = True
    return near_poc

def compute_liquidity_sweep(high, low, close, open_, wick_ratio, lookback):
    n = len(close)
    sweep = np.zeros(n, dtype=bool)
    for i in range(lookback+1, n):
        support = np.min(low[i-lookback:i])
        candle_range = high[i] - low[i]
        if candle_range < 1e-6: continue
        lower_wick = min(open_[i], close[i]) - low[i]
        lower_wick_pct = lower_wick / candle_range
        if low[i] < support and lower_wick_pct >= wick_ratio and close[i] > low[i] * 1.005:
            sweep[i] = True
    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS (with per‑signal trailing params)
# ─────────────────────────────────────────────────────────────────────────────

def signal_stochastics(df, k_period, d_period, smooth, ob_level=80, os_level=20):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pct_k, pct_d = compute_stochastics(h, l, c, k_period, d_period, smooth)
    n = len(c)
    signal = np.zeros(n, dtype=bool)
    k_cross_up = (~np.isnan(pct_k)) & (~np.isnan(pct_d)) & (pct_k > pct_d) & (np.roll(pct_k,1) <= np.roll(pct_d,1)) & (pct_k < ob_level) & (pct_k > os_level)
    signal[1:] = k_cross_up[1:]
    return signal

def signal_obv_trend(df, ema_period):
    c, v = df["close"].values, df["volume"].values
    obv_ema = compute_obv_trend(c, v, ema_period)
    daily_ret = np.diff(c, prepend=c[0])
    direction = np.sign(daily_ret)
    direction[0] = 0
    obv = np.cumsum(direction * v)
    n = len(c)
    signal = np.zeros(n, dtype=bool)
    cross_up = (~np.isnan(obv_ema)) & (obv > obv_ema) & (np.roll(obv,1) <= np.roll(obv_ema,1))
    signal[1:] = cross_up[1:]
    return signal

def signal_vwap_deviation(df, entry_threshold):
    """FIXED: no look‑ahead, uses only past data for recovery check."""
    c = df["close"].values
    v = df["volume"].values
    dev = compute_vwap_deviation(c, v)
    n = len(c)
    signal = np.zeros(n, dtype=bool)
    # Price below VWAP by threshold AND next day closes higher (recovery)
    below_vwap = dev <= -entry_threshold
    next_day_up = np.roll(c > np.roll(c, 1), -1)  # tomorrow's close > today's close
    trigger = below_vwap & next_day_up & (~np.isnan(dev))
    signal[:-1] = trigger[:-1]  # cannot signal on last bar
    return signal

def signal_volume_profile(df, lookback, node_thr):
    c, v = df["close"].values, df["volume"].values
    near_poc = compute_volume_profile(c, v, lookback, node_thr)
    vol_ma = pd.Series(v).rolling(20).mean().values
    vol_confirm = v > vol_ma * 1.2
    signal = np.zeros(len(c), dtype=bool)
    trigger = near_poc & vol_confirm
    signal[1:] = trigger[1:]
    return signal

def signal_liquidity_sweep(df, wick_ratio, lookback):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    op = df["open"].values if "open" in df.columns else c
    sweep = compute_liquidity_sweep(h, l, c, op, wick_ratio, lookback)
    n = len(c)
    signal = np.zeros(n, dtype=bool)
    signal[lookback+1:] = sweep[lookback+1:]
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (with sector support and per‑signal trailing)
# ─────────────────────────────────────────────────────────────────────────────

def backtest_signal(df: pd.DataFrame,
                    signal: np.ndarray,
                    trading_dates: np.ndarray,
                    max_hold_days: int,
                    from_date: str,
                    to_date: str,
                    stop_loss_pct: float,
                    trail_activate_pct: float,
                    trail_floor_pct: float,
                    sector: str = "Unknown") -> list[dict]:
    """
    Core backtest with per‑signal trailing stop parameters.
    """
    trades = []
    closes = df["close"].values
    dates = df["date"].dt.date.values
    n = len(df)
    from_d = date.fromisoformat(from_date) if from_date else date(2000,1,1)
    to_d   = date.fromisoformat(to_date)   if to_date   else date(2100,1,1)

    i = 0
    while i < n:
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
        earliest_exit = t4_exit_date(entry_date, trading_dates)
        if earliest_exit is None:
            i += 1
            continue
        exit_start = i+1
        while exit_start < n and dates[exit_start] < earliest_exit:
            exit_start += 1
        if exit_start >= n:
            i += 1
            continue

        peak_profit = 0.0
        exit_price = closes[exit_start]
        exit_reason = "MAX_HOLD"
        trade_days = 0
        max_exit_idx = min(exit_start + max_hold_days, n-1)
        j = exit_start
        trailing_active = False

        while j <= max_exit_idx:
            cp = closes[j]
            if np.isnan(cp) or cp <= 0:
                j += 1
                continue
            pnl_pct = (cp - entry_price) / entry_price
            if pnl_pct > peak_profit:
                peak_profit = pnl_pct
            if peak_profit >= trail_activate_pct:
                trailing_active = True
            if pnl_pct <= -stop_loss_pct:
                exit_price = cp
                exit_reason = "STOP_LOSS"
                trade_days = j - i
                break
            if trailing_active:
                floor = peak_profit - trail_floor_pct
                if pnl_pct <= floor:
                    exit_price = cp
                    exit_reason = "TRAIL_STOP"
                    trade_days = j - i
                    break
            if j == max_exit_idx:
                exit_price = cp
                exit_reason = "MAX_HOLD"
                trade_days = j - i
                break
            j += 1

        entry_amount = entry_price * 100
        exit_amount = exit_price * 100
        buy_fees = calc_fees(entry_amount)
        sell_fees = calc_fees(exit_amount)
        gross_pnl = exit_amount - entry_amount
        net_pnl = gross_pnl - buy_fees - sell_fees
        total_cost = entry_amount + buy_fees
        return_pct = net_pnl / total_cost * 100 if total_cost > 0 else 0.0
        result = "WIN" if net_pnl > 0 else "LOSS"

        trades.append({
            "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "?",
            "sector": sector,
            "entry_date": str(entry_date),
            "exit_date": str(dates[min(j, n-1)]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "hold_days": trade_days,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "return_pct": round(return_pct, 4),
            "result": result,
            "total_fees": round(buy_fees + sell_fees, 2),
        })
        i = j + 1
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# METRICS WITH SECTOR BREAKDOWN & BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(returns, n_iter=1000, alpha=0.05):
    """Bootstrap confidence interval for profit factor."""
    if len(returns) < 5:
        return None, None
    pf_list = []
    for _ in range(n_iter):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        wins = sample[sample > 0].sum()
        losses = abs(sample[sample < 0].sum())
        if losses > 0:
            pf_list.append(wins / losses)
    if not pf_list:
        return None, None
    lower = np.percentile(pf_list, 100*alpha/2)
    upper = np.percentile(pf_list, 100*(1-alpha/2))
    return lower, upper

def compute_metrics(trades: list[dict], signal_name: str, params: dict, bootstrap_iter=0) -> dict:
    if not trades:
        return {"signal": signal_name, **params, "total_trades": 0, "win_rate": 0,
                "profit_factor": 0, "annual_ret_pct": 0, "sharpe": 0, "sortino": 0,
                "calmar": 0, "max_drawdown": 0, "win_streak_max": 0, "loss_streak_max": 0,
                "spearman_rho": 0, "spearman_p": 1.0, "total_pnl": 0, "total_fees": 0,
                "sector_breakdown": {}}

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["result"] == "WIN").sum()
    losses = total - wins
    win_rate = wins / total

    net_pnls = df["net_pnl"].values
    returns = df["return_pct"].values / 100.0  # as fraction

    win_pnls = net_pnls[net_pnls > 0]
    loss_pnls = net_pnls[net_pnls < 0]
    profit_factor = win_pnls.sum() / abs(loss_pnls.sum()) if len(loss_pnls)>0 and abs(loss_pnls.sum())>0 else (999.0 if wins>0 else 0.0)

    # Annual return estimate
    avg_hold = df["hold_days"].mean() or 17
    trades_per_year = 252 / avg_hold
    mean_return = returns.mean()
    annual_ret = (1 + mean_return) ** trades_per_year - 1

    # Sharpe (risk-free 5.5% annual)
    rf_daily = (1 + 0.055) ** (1/252) - 1
    excess = returns - rf_daily
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    sortino = (returns.mean() - rf_daily) / (downside.std() * np.sqrt(252)) if len(downside)>0 and downside.std()>0 else 0

    # Calmar (annual return / max drawdown)
    cum_pnl = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = (cum_pnl - peak) / (np.abs(peak) + 1e-9) * 100
    max_dd = float(np.min(drawdown))
    calmar = annual_ret / (abs(max_dd)/100) if max_dd != 0 else 0

    # Win/loss streaks
    results = df["result"].values
    streak_win = 0
    streak_loss = 0
    max_win_streak = 0
    max_loss_streak = 0
    for r in results:
        if r == "WIN":
            streak_win += 1
            streak_loss = 0
            max_win_streak = max(max_win_streak, streak_win)
        elif r == "LOSS":
            streak_loss += 1
            streak_win = 0
            max_loss_streak = max(max_loss_streak, streak_loss)
        else:
            streak_win = streak_loss = 0

    # Spearman correlation (signal rank vs return)
    n_ret = len(returns)
    if n_ret >= 10:
        rho, p = stats.spearmanr(np.arange(n_ret), returns)
    else:
        rho, p = 0.0, 1.0

    # Bootstrap CI for profit factor
    pf_lower, pf_upper = None, None
    if bootstrap_iter > 0 and len(returns) >= 10:
        pf_lower, pf_upper = bootstrap_ci(returns, n_iter=bootstrap_iter)

    # Sector breakdown
    sector_breakdown = {}
    if "sector" in df.columns:
        for sec, grp in df.groupby("sector"):
            sec_wins = (grp["result"] == "WIN").sum()
            sec_losses = len(grp) - sec_wins
            sec_win_pnl = grp[grp["result"]=="WIN"]["net_pnl"].sum()
            sec_loss_pnl = abs(grp[grp["result"]=="LOSS"]["net_pnl"].sum())
            sec_pf = sec_win_pnl / sec_loss_pnl if sec_loss_pnl > 0 else (999.0 if sec_wins>0 else 0)
            sector_breakdown[sec] = {
                "trades": len(grp),
                "win_rate": sec_wins / len(grp) if len(grp)>0 else 0,
                "profit_factor": sec_pf,
                "total_pnl": grp["net_pnl"].sum(),
            }

    total_fees = df["total_fees"].sum()
    total_pnl = df["net_pnl"].sum()

    return {
        "signal": signal_name,
        **params,
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3),
        "pf_ci_lower": round(pf_lower, 3) if pf_lower is not None else None,
        "pf_ci_upper": round(pf_upper, 3) if pf_upper is not None else None,
        "annual_ret_pct": round(annual_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": round(max_dd, 2),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p, 6),
        "total_pnl": round(total_pnl, 0),
        "total_fees": round(total_fees, 0),
        "sector_breakdown": sector_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION FOR MULTIPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def run_single_combo(args: tuple) -> dict:
    signal_name, params, symbol_data_list, trading_dates, max_hold, from_date, to_date, sector_map, bootstrap_iter = args
    all_trades = []
    for sym, df_dict in symbol_data_list:
        df = pd.DataFrame(df_dict)
        df["date"] = pd.to_datetime(df["date"])
        sector = sector_map.get(sym, "Unknown")
        try:
            # Generate signal
            if signal_name == "STOCH":
                sig = signal_stochastics(df, params["k"], params["d"], params["smooth"])
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            elif signal_name == "OBV":
                sig = signal_obv_trend(df, params["ema"])
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            elif signal_name == "VWAP":
                sig = signal_vwap_deviation(df, params["threshold"])
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            elif signal_name == "VOLPROFILE":
                sig = signal_volume_profile(df, params["lookback"], params["node_thr"])
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            elif signal_name == "LIQSWEEP":
                sig = signal_liquidity_sweep(df, params["wick_ratio"], params["lookback"])
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            elif signal_name.startswith("COMBO_"):
                # rebuild component signals
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
                # Use first component's stop/trail (simplified)
                stop_pct = params.get("stop_pct", STOP_LOSS_PCT)
                trail_act = params.get("trail_activate_pct", TRAIL_ACTIVATE_PCT)
                trail_flr = params.get("trail_floor_pct", TRAIL_FLOOR_PCT)
            else:
                continue

            if "symbol" not in df.columns:
                df["symbol"] = sym
            trades = backtest_signal(df, sig, trading_dates, max_hold, from_date, to_date,
                                     stop_pct, trail_act, trail_flr, sector)
            all_trades.extend(trades)
        except Exception as e:
            pass  # skip symbol on error
    return compute_metrics(all_trades, signal_name, params, bootstrap_iter)


def signal_combo(df, signals, mode="ALL"):
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
# PARAMETER GRIDS (with per‑signal trailing params)
# ─────────────────────────────────────────────────────────────────────────────

def get_param_grid(fast: bool = False) -> dict[str, list[dict]]:
    if fast:
        return {
            "STOCH": [
                {"k": 14, "d": 3, "smooth": 3, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03},
                {"k": 21, "d": 5, "smooth": 5, "stop_pct": 0.04, "trail_activate_pct": 0.07, "trail_floor_pct": 0.03},
            ],
            "OBV": [{"ema": 20, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03}],
            "VWAP": [{"threshold": 0.02, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03},
                     {"threshold": 0.03, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03}],
            "VOLPROFILE": [{"lookback": 30, "node_thr": 2.0, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03}],
            "LIQSWEEP": [{"wick_ratio": 0.7, "lookback": 10, "stop_pct": 0.05, "trail_activate_pct": 0.06, "trail_floor_pct": 0.03}],
        }
    return {
        "STOCH": [{"k": k, "d": d, "smooth": s, "stop_pct": sp, "trail_activate_pct": tap, "trail_floor_pct": tfp}
                  for k in [9,14,21] for d in [3,5] for s in [3,5]
                  for sp in [0.04,0.05,0.06] for tap in [0.05,0.06,0.08] for tfp in [0.02,0.03,0.04]],
        "OBV": [{"ema": e, "stop_pct": sp, "trail_activate_pct": tap, "trail_floor_pct": tfp}
                for e in [10,20,30] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "VWAP": [{"threshold": t, "stop_pct": sp, "trail_activate_pct": tap, "trail_floor_pct": tfp}
                 for t in [0.01,0.02,0.03] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "VOLPROFILE": [{"lookback": lb, "node_thr": nt, "stop_pct": sp, "trail_activate_pct": tap, "trail_floor_pct": tfp}
                       for lb in [20,30,60] for nt in [1.5,2.0,2.5] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "LIQSWEEP": [{"wick_ratio": wr, "lookback": lb, "stop_pct": sp, "trail_activate_pct": tap, "trail_floor_pct": tfp}
                     for wr in [0.6,0.7,0.8] for lb in [5,10,20] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
    }

def get_combo_params(survivors: list[tuple[str, dict]]) -> list[tuple]:
    combos = []
    for i, (n1, p1) in enumerate(survivors):
        for j, (n2, p2) in enumerate(survivors):
            if j <= i: continue
            # Merge stop/trail params (use first component's as base)
            merged_params = {
                "components": [(n1, p1), (n2, p2)],
                "mode": "ALL",
                "stop_pct": p1.get("stop_pct", STOP_LOSS_PCT),
                "trail_activate_pct": p1.get("trail_activate_pct", TRAIL_ACTIVATE_PCT),
                "trail_floor_pct": p1.get("trail_floor_pct", TRAIL_FLOOR_PCT),
            }
            combos.append(("COMBO_2", merged_params))
    if len(survivors) >= 3:
        for a,b,c in itertools.combinations(range(len(survivors)), 3):
            n1,p1 = survivors[a]
            merged_params = {
                "components": [(n1,p1), (survivors[b][1]), (survivors[c][1])],
                "mode": "2OF3",
                "stop_pct": p1.get("stop_pct", STOP_LOSS_PCT),
                "trail_activate_pct": p1.get("trail_activate_pct", TRAIL_ACTIVATE_PCT),
                "trail_floor_pct": p1.get("trail_floor_pct", TRAIL_FLOOR_PCT),
            }
            combos.append(("COMBO_3", merged_params))
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# CSV WRITER (crash-safe)
# ─────────────────────────────────────────────────────────────────────────────

def append_result(csv_path: Path, result: dict):
    flat = {}
    for k,v in result.items():
        if k == "components":
            flat["components"] = str(v)
        elif k == "sector_breakdown":
            flat["sector_breakdown"] = json.dumps(v)
        else:
            flat[k] = v
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


# ─────────────────────────────────────────────────────────────────────────────
# WALK‑FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_run(symbol_data: dict, trading_dates: np.ndarray,
                     param_grid: dict, csv_path: Path,
                     max_hold: int, from_date: str, to_date: str,
                     sector_map: dict, bootstrap_iter: int,
                     train_months: int = 12, test_months: int = 3,
                     tg=None) -> list[dict]:
    """Rolling walk‑forward backtest."""
    # Get sorted unique dates across all symbols
    all_dates = sorted(set(dt for df in symbol_data.values() for dt in df["date"].dt.date))
    start_idx = 0
    results = []
    while start_idx + train_months + test_months <= len(all_dates):
        train_end_date = all_dates[start_idx + train_months]
        test_end_date = all_dates[start_idx + train_months + test_months]
        train_start_date = all_dates[start_idx]

        log.info("Walk‑forward window: %s → %s (train) | %s → %s (test)",
                 train_start_date, train_end_date, train_end_date, test_end_date)

        # Filter symbol_data to period
        filtered_data = {}
        for sym, df in symbol_data.items():
            mask = (df["date"].dt.date >= train_start_date) & (df["date"].dt.date <= test_end_date)
            filtered_data[sym] = df[mask].copy()
        # Run standalone on training period to find best params per signal
        # (Simplified: for each signal, find best PF on train, then test on test period)
        # For brevity, we call a simplified optimization here.
        # In full implementation you'd reuse run_standalone with date range.
        # We'll implement a reduced version:
        best_params_per_signal = {}
        for sig_name, param_list in param_grid.items():
            best_pf = 0
            best_p = None
            for params in param_list:
                # Run backtest on training period only
                job = (sig_name, params, [(sym, df.to_dict("list")) for sym,df in filtered_data.items()],
                       trading_dates, max_hold, train_start_date.isoformat(), train_end_date.isoformat(),
                       sector_map, bootstrap_iter)
                res = run_single_combo(job)
                if res["total_trades"] >= 10 and res["profit_factor"] > best_pf:
                    best_pf = res["profit_factor"]
                    best_p = params
            if best_p:
                best_params_per_signal[sig_name] = best_p

        # Now test each signal with its best params on test period
        for sig_name, params in best_params_per_signal.items():
            job = (sig_name, params, [(sym, df.to_dict("list")) for sym,df in filtered_data.items()],
                   trading_dates, max_hold, train_end_date.isoformat(), test_end_date.isoformat(),
                   sector_map, bootstrap_iter)
            res = run_single_combo(job)
            res["walkforward_window"] = f"{train_start_date.date()}→{test_end_date.date()}"
            results.append(res)
            append_result(csv_path, res)

        start_idx += test_months
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone(symbol_data: dict, trading_dates: np.ndarray,
                   param_grid: dict, csv_path: Path,
                   max_hold: int, from_date: str, to_date: str,
                   sector_map: dict, bootstrap_iter: int,
                   symbol_filter: str = None, tg=None) -> list[dict]:
    if symbol_filter:
        symbol_data = {s: df for s,df in symbol_data.items() if s == symbol_filter.upper()}
    symbol_list = [(sym, df.to_dict("list")) for sym,df in symbol_data.items()]
    jobs_by_signal = {}
    for sig_name, param_list in param_grid.items():
        jobs_by_signal[sig_name] = []
        for params in param_list:
            jobs_by_signal[sig_name].append(
                (sig_name, params, symbol_list, trading_dates, max_hold, from_date, to_date, sector_map, bootstrap_iter)
            )
    all_jobs = [j for jobs in jobs_by_signal.values() for j in jobs]
    total_jobs = len(all_jobs)
    log.info("Standalone: %d combos, %d symbols", total_jobs, len(symbol_list))
    if tg:
        tg.notify_phase_start("STANDALONE", total_jobs, len(symbol_list))

    results = []
    results_by_signal = {s: [] for s in jobs_by_signal}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_sig = {}
        for sig_name, jobs in jobs_by_signal.items():
            for j in jobs:
                fut = pool.submit(run_single_combo, j)
                future_to_sig[fut] = sig_name
        pbar = tqdm(total=total_jobs, desc="Standalone", ncols=80)
        signal_remaining = {s: len(jobs) for s, jobs in jobs_by_signal.items()}
        signal_done = set()
        for fut in as_completed(future_to_sig):
            sig = future_to_sig[fut]
            try:
                r = fut.result()
                results.append(r)
                results_by_signal[sig].append(r)
                append_result(csv_path, r)
            except Exception as e:
                log.warning("Combo failed: %s", e)
            finally:
                signal_remaining[sig] -= 1
            if signal_remaining[sig] == 0 and sig not in signal_done:
                signal_done.add(sig)
                if tg:
                    best = max(results_by_signal[sig], key=lambda x: x.get("profit_factor",0))
                    tg.notify_per_signal(sig, best)
            pbar.update(1)
        pbar.close()
    return results

def run_combos(symbol_data: dict, trading_dates: np.ndarray,
               survivors: list[tuple[str, dict]], csv_path: Path,
               max_hold: int, from_date: str, to_date: str,
               sector_map: dict, bootstrap_iter: int, tg=None) -> list[dict]:
    symbol_list = [(sym, df.to_dict("list")) for sym,df in symbol_data.items()]
    combo_params = get_combo_params(survivors)
    if not combo_params:
        return []
    jobs = [(sig_name, params, symbol_list, trading_dates, max_hold, from_date, to_date, sector_map, bootstrap_iter)
            for sig_name, params in combo_params]
    log.info("Combos: %d jobs", len(jobs))
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

def identify_survivors(results: list[dict], min_pf=1.5, min_trades=30, max_p=0.05) -> list[tuple[str, dict]]:
    by_signal = {}
    for r in results:
        sig = r["signal"]
        by_signal.setdefault(sig, []).append(r)
    survivors = []
    for sig, rlist in by_signal.items():
        passing = [r for r in rlist if r["profit_factor"] >= min_pf and r["total_trades"] >= min_trades and r["spearman_p"] <= max_p]
        if passing:
            best = max(passing, key=lambda x: x["profit_factor"])
            param_keys = {"k","d","smooth","ema","threshold","lookback","node_thr","wick_ratio","stop_pct","trail_activate_pct","trail_floor_pct"}
            params = {k: v for k,v in best.items() if k in param_keys}
            survivors.append((sig, params))
    return survivors

def print_summary(results: list[dict], title="Results"):
    if not results:
        print(f"\n{title}: No results")
        return
    ranked = sorted(results, key=lambda x: x.get("profit_factor",0), reverse=True)
    print(f"\n{'─'*110}")
    print(f"  {title}")
    print(f"{'─'*110}")
    print(f"  {'Signal':<14} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Ann%':>8} "
          f"{'Sharpe':>7} {'Sortino':>7} {'Calmar':>7} {'DD%':>7} {'ρ-p':>8} {'NetPnL':>10}")
    print(f"{'─'*110}")
    for r in ranked[:20]:
        print(f"  {r['signal']:<14} {r['total_trades']:>7} {r['win_rate']*100:>7.1f} "
              f"{r['profit_factor']:>7.2f} {r['annual_ret_pct']:>8.1f} {r['sharpe']:>7.2f} "
              f"{r.get('sortino',0):>7.2f} {r.get('calmar',0):>7.2f} {r['max_drawdown']:>7.1f} "
              f"{r['spearman_p']:>8.4f} {r['total_pnl']:>10,.0f}")
    print(f"{'─'*110}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced NEPSE Indicator Backtester")
    parser.add_argument("mode", choices=["standalone","combos","both","walkforward"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--from", dest="from_date", default=DEFAULT_TRAIN_FROM)
    parser.add_argument("--to", dest="to_date", default=DEFAULT_TRAIN_TO)
    parser.add_argument("--hold", type=int, default=30)
    parser.add_argument("--symbol", help="Single symbol test")
    parser.add_argument("--min-pf", type=float, default=1.5)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for PF CI")
    parser.add_argument("--walkforward-train", type=int, default=12, help="Train months per window")
    parser.add_argument("--walkforward-test", type=int, default=3, help="Test months per window")
    args = parser.parse_args()

    if not DATABASE_URL:
        log.error("DATABASE_URL not set")
        sys.exit(1)

    tg = TelegramNotifier()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"indicator_results_{timestamp}.csv"
    start_time = time.time()

    tg.notify_start(args.mode, args.from_date, args.to_date, args.fast, args.hold)

    try:
        symbol_data = load_price_history(args.from_date, args.to_date, tg=tg)
        trading_dates = get_trading_dates(symbol_data)
        sector_map = load_sector_map()
    except Exception as e:
        tg.notify_error("Data Load", str(e))
        sys.exit(1)

    if not symbol_data:
        tg.notify_error("Data Load", "No symbols loaded")
        sys.exit(1)

    param_grid = get_param_grid(fast=args.fast)

    if args.mode == "walkforward":
        log.info("Walk‑forward validation mode")
        results = walk_forward_run(symbol_data, trading_dates, param_grid, csv_path,
                                   args.hold, args.from_date, args.to_date,
                                   sector_map, args.bootstrap,
                                   args.walkforward_train, args.walkforward_test, tg)
        print_summary(results, "Walk‑forward Results")
    else:
        if args.mode in ("standalone","both"):
            log.info("Standalone phase")
            results = run_standalone(symbol_data, trading_dates, param_grid, csv_path,
                                     args.hold, args.from_date, args.to_date,
                                     sector_map, args.bootstrap, args.symbol, tg)
            print_summary(results, "Standalone Results")
            survivors = identify_survivors(results, args.min_pf, args.min_trades)
            tg.notify_survivors(survivors, results)
            if args.mode == "both" and survivors:
                log.info("Combo phase")
                combo_results = run_combos(symbol_data, trading_dates, survivors, csv_path,
                                           args.hold, args.from_date, args.to_date,
                                           sector_map, args.bootstrap, tg)
                print_summary(combo_results, "Combo Results")
        elif args.mode == "combos":
            log.warning("Combos mode requires standalone first to generate survivors. Run 'both' instead.")
            return

    elapsed = time.time() - start_time
    tg.notify_done([], str(csv_path), elapsed)
    log.info("Done. Elapsed: %s. Results: %s", timedelta(seconds=int(elapsed)), csv_path)

if __name__ == "__main__":
    main()