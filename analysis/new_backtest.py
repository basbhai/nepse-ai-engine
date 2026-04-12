"""
analysis/indicator_backtest.py — Enhanced NEPSE Indicator Backtester (Phase 2+)
═══════════════════════════════════════════════════════════════════════════════
Full pipeline mode: runs all three simulation scenarios (unlimited, constrained_200k, constrained_500k)
Exports detailed text summary, CSV results, and sends Telegram notifications (with diagnostics).
═══════════════════════════════════════════════════════════════════════════════
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
# TELEGRAM NOTIFIER (with diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("NEW_TELEGRAM_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.warning("Telegram disabled: missing NEW_TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in .env")
        else:
            log.info("Telegram notifier enabled for chat_id=%s", self.chat_id)

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
                success = resp.status == 200
                if not success:
                    log.warning("Telegram send returned status %d", resp.status)
                return success
        except Exception as e:
            log.warning("Telegram send failed: %s", e)
            return False

    def notify_start(self, mode, from_date, to_date, fast, hold, sim_mode):
        self._send(f"🚀 *Backtest Started*\nMode: `{mode}`\nSim: `{sim_mode}`\nPeriod: `{from_date}` → `{to_date}`")
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
# CONFIGURATION (same as before, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_TRAIN_FROM = "2019-07-15"
DEFAULT_TRAIN_TO   = "2024-12-31"
DEFAULT_TEST_FROM  = "2025-01-01"
DEFAULT_TEST_TO    = "2026-03-29"

STOP_LOSS_PCT      = 0.05
TRAIL_ACTIVATE_PCT = 0.06
TRAIL_FLOOR_PCT    = 0.03
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

GAZETTE_HOLIDAYS = { ... }  # keep the same set as before (abbreviated for brevity)
def is_trading_day(d: date) -> bool:
    if d.weekday() in (4,5): return False
    return d.strftime("%Y-%m-%d") not in GAZETTE_HOLIDAYS

def t4_exit_date(entry_date: date, trading_dates: np.ndarray) -> date:
    entry_dt64 = np.datetime64(entry_date, "D")
    future = trading_dates[trading_dates > entry_dt64]
    if len(future) < 4: return None
    return future[3].astype("M8[D]").astype(object)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE LOADERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_sector_map() -> dict:
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
# INDICATORS (unchanged, includes OBV)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stochastics(high, low, close, k_period, d_period, smooth):
    # ... same as before
    pass

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
    # ... same
    pass

def compute_volume_profile(close, volume, lookback, node_threshold):
    # ... same
    pass

def compute_liquidity_sweep(high, low, close, open_, wick_ratio, lookback):
    # ... same
    pass


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATORS (OBV included)
# ─────────────────────────────────────────────────────────────────────────────

def signal_stochastics(df, k_period, d_period, smooth, ob_level=80, os_level=20):
    # ... same
    pass

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
    # ... fixed version
    pass

def signal_volume_profile(df, lookback, node_thr):
    # ... same
    pass

def signal_liquidity_sweep(df, wick_ratio, lookback):
    # ... same
    pass


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINES (unlimited and constrained) – same as before
# ─────────────────────────────────────────────────────────────────────────────

def backtest_signal(df, signal, trading_dates, max_hold_days, from_date, to_date,
                    stop_loss_pct, trail_activate_pct, trail_floor_pct, sector):
    # ... unchanged (the unlimited version)
    pass

class PortfolioBacktestEngine:
    # ... unchanged (constrained version)
    pass


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(returns, n_iter=1000, alpha=0.05):
    # ... same
    pass

def compute_metrics(trades, signal_name, params, bootstrap_iter=0):
    # ... same (includes sector breakdown)
    pass


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def signal_combo(df, signals, mode="ALL"):
    # ... same
    pass

def run_single_combo(args):
    # ... same as provided in the previous answer
    pass


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER GRIDS (includes OBV)
# ─────────────────────────────────────────────────────────────────────────────

def get_param_grid(fast: bool = False) -> dict[str, list[dict]]:
    if fast:
        return {
            "STOCH": [{"k":14,"d":3,"smooth":3,"stop_pct":0.05,"trail_activate_pct":0.06,"trail_floor_pct":0.03}],
            "OBV":   [{"ema":20, "stop_pct":0.05,"trail_activate_pct":0.06,"trail_floor_pct":0.03}],
            "VWAP":  [{"threshold":0.02,"stop_pct":0.05,"trail_activate_pct":0.06,"trail_floor_pct":0.03}],
            "VOLPROFILE":[{"lookback":30,"node_thr":2.0,"stop_pct":0.05,"trail_activate_pct":0.06,"trail_floor_pct":0.03}],
            "LIQSWEEP":[{"wick_ratio":0.7,"lookback":10,"stop_pct":0.05,"trail_activate_pct":0.06,"trail_floor_pct":0.03}],
        }
    # Full grid (includes OBV with multiple EMAs)
    return {
        "STOCH": [{"k":k,"d":d,"smooth":s,"stop_pct":sp,"trail_activate_pct":tap,"trail_floor_pct":tfp}
                  for k in [9,14,21] for d in [3,5] for s in [3,5]
                  for sp in [0.04,0.05,0.06] for tap in [0.05,0.06,0.08] for tfp in [0.02,0.03,0.04]],
        "OBV": [{"ema":e, "stop_pct":sp, "trail_activate_pct":tap, "trail_floor_pct":tfp}
                for e in [10,20,30] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "VWAP": [{"threshold":t, "stop_pct":sp, "trail_activate_pct":tap, "trail_floor_pct":tfp}
                 for t in [0.01,0.02,0.03] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "VOLPROFILE": [{"lookback":lb, "node_thr":nt, "stop_pct":sp, "trail_activate_pct":tap, "trail_floor_pct":tfp}
                       for lb in [20,30,60] for nt in [1.5,2.0,2.5] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
        "LIQSWEEP": [{"wick_ratio":wr, "lookback":lb, "stop_pct":sp, "trail_activate_pct":tap, "trail_floor_pct":tfp}
                     for wr in [0.6,0.7,0.8] for lb in [5,10,20] for sp in [0.04,0.05] for tap in [0.05,0.06] for tfp in [0.02,0.03]],
    }

def get_combo_params(survivors):
    # ... same
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CSV WRITER
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
# DETAILED TEXT SUMMARY EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

def export_text_summary(results: list[dict], sim_mode: str, output_path: Path):
    """Export a human-readable summary with top signals, sector breakdown, etc."""
    if not results:
        with open(output_path, "w") as f:
            f.write(f"No results for {sim_mode}\n")
        return

    ranked = sorted(results, key=lambda x: x.get("profit_factor",0), reverse=True)
    with open(output_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"NEPSE BACKTEST SUMMARY — {sim_mode.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write("TOP 10 SIGNALS (by Profit Factor)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<5} {'Signal':<14} {'Trades':>8} {'WR%':>8} {'PF':>8} {'Ann%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'NetPnL':>12}\n")
        for i, r in enumerate(ranked[:10], 1):
            f.write(f"{i:<5} {r['signal']:<14} {r['total_trades']:>8} {r['win_rate']*100:>7.1f}% "
                    f"{r['profit_factor']:>8.2f} {r['annual_ret_pct']:>7.1f}% {r['sharpe']:>8.2f} "
                    f"{r['max_drawdown']:>7.1f}% {r['total_pnl']:>11,.0f}\n")
        f.write("\n")

        # Best overall signal
        best = ranked[0]
        f.write("BEST OVERALL SIGNAL\n")
        f.write("-"*80 + "\n")
        f.write(f"Signal:          {best['signal']}\n")
        f.write(f"Profit Factor:   {best['profit_factor']:.2f}\n")
        f.write(f"Win Rate:        {best['win_rate']*100:.1f}%\n")
        f.write(f"Total Trades:    {best['total_trades']}\n")
        f.write(f"Annual Return:   {best['annual_ret_pct']:.2f}%\n")
        f.write(f"Sharpe Ratio:    {best['sharpe']:.2f}\n")
        f.write(f"Sortino Ratio:   {best.get('sortino',0):.2f}\n")
        f.write(f"Calmar Ratio:    {best.get('calmar',0):.2f}\n")
        f.write(f"Max Drawdown:    {best['max_drawdown']:.2f}%\n")
        f.write(f"Total Net P&L:   NPR {best['total_pnl']:,.0f}\n")
        f.write(f"Total Fees Paid: NPR {best['total_fees']:,.0f}\n")
        f.write(f"Spearman p-value:{best['spearman_p']:.6f}\n")
        if best.get('pf_ci_lower'):
            f.write(f"PF 95% CI:       [{best['pf_ci_lower']}, {best['pf_ci_upper']}]\n")
        f.write("\n")

        # Sector breakdown for best signal
        if "sector_breakdown" in best and best["sector_breakdown"]:
            f.write("SECTOR BREAKDOWN (Best Signal)\n")
            f.write("-"*80 + "\n")
            try:
                sec_data = json.loads(best["sector_breakdown"]) if isinstance(best["sector_breakdown"], str) else best["sector_breakdown"]
                for sec, stats_ in sorted(sec_data.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
                    f.write(f"{sec:<25} trades={stats_['trades']:>4}  WR={stats_['win_rate']*100:>5.1f}%  "
                            f"PF={stats_['profit_factor']:>6.2f}  PnL=NPR {stats_['total_pnl']:>10,.0f}\n")
            except:
                pass
            f.write("\n")

        # Survivors (if any)
        survivors = [r for r in results if r["profit_factor"] >= 1.5 and r["total_trades"] >= 30 and r["spearman_p"] <= 0.05]
        if survivors:
            f.write(f"SURVIVORS (PF≥1.5, trades≥30, p≤0.05) — {len(survivors)} signals\n")
            f.write("-"*80 + "\n")
            for r in sorted(survivors, key=lambda x: x["profit_factor"], reverse=True):
                f.write(f"{r['signal']:<14} PF={r['profit_factor']:.2f}  WR={r['win_rate']*100:.1f}%  trades={r['total_trades']}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("END OF SUMMARY\n")
    log.info("Text summary exported to %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS (standalone, combos, walkforward) – same as before
# ─────────────────────────────────────────────────────────────────────────────

def run_standalone(symbol_data, trading_dates, param_grid, csv_path, max_hold,
                   from_date, to_date, sector_map, bootstrap_iter, sim_mode,
                   symbol_filter=None, tg=None):
    # ... same as previous implementation (already includes OBV)
    pass

def run_combos(symbol_data, trading_dates, survivors, csv_path, max_hold,
               from_date, to_date, sector_map, bootstrap_iter, sim_mode, tg=None):
    # ... same as before
    pass

def identify_survivors(results, min_pf=1.5, min_trades=30, max_p=0.05):
    # ... same
    pass

def print_summary(results, title="Results"):
    # ... same
    pass


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE: runs all three simulation modes sequentially
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(args, tg):
    """Run standalone + combos for all three simulation modes."""
    sim_modes = ["unlimited", "constrained_200k", "constrained_500k"]
    all_summaries = []

    # Load data once (same for all sims)
    symbol_data = load_price_history(args.from_date, args.to_date, tg=tg)
    trading_dates = get_trading_dates(symbol_data)
    sector_map = load_sector_map()

    if not symbol_data:
        tg.notify_error("Data Load", "No symbols loaded")
        return

    param_grid = get_param_grid(fast=args.fast)

    for sim_mode in sim_modes:
        log.info("="*60)
        log.info("RUNNING SIMULATION: %s", sim_mode)
        log.info("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f"indicator_results_{sim_mode}_{timestamp}.csv"
        txt_path = RESULTS_DIR / f"summary_{sim_mode}_{timestamp}.txt"

        # Standalone phase
        log.info("Standalone phase for %s", sim_mode)
        results = run_standalone(
            symbol_data, trading_dates, param_grid, csv_path,
            args.hold, args.from_date, args.to_date,
            sector_map, args.bootstrap, sim_mode,
            args.symbol, tg
        )
        print_summary(results, f"Standalone Results ({sim_mode})")

        survivors = identify_survivors(results, args.min_pf, args.min_trades)
        tg.notify_survivors(survivors, results)

        combo_results = []
        if survivors and args.mode != "standalone":
            log.info("Combo phase for %s", sim_mode)
            combo_results = run_combos(
                symbol_data, trading_dates, survivors, csv_path,
                args.hold, args.from_date, args.to_date,
                sector_map, args.bootstrap, sim_mode, tg
            )
            print_summary(combo_results, f"Combo Results ({sim_mode})")

        # Export text summary
        all_res = results + combo_results
        export_text_summary(all_res, sim_mode, txt_path)

        all_summaries.append({
            "sim_mode": sim_mode,
            "standalone_trades": sum(r["total_trades"] for r in results),
            "best_pf": max((r["profit_factor"] for r in results), default=0),
            "survivors": len(survivors),
            "csv": str(csv_path),
            "txt": str(txt_path),
        })

    # Write master summary
    master_path = RESULTS_DIR / f"full_pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(master_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("FULL PIPELINE MASTER SUMMARY\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        for s in all_summaries:
            f.write(f"Simulation: {s['sim_mode']}\n")
            f.write(f"  Total standalone trades: {s['standalone_trades']}\n")
            f.write(f"  Best profit factor:      {s['best_pf']:.3f}\n")
            f.write(f"  Number of survivors:     {s['survivors']}\n")
            f.write(f"  CSV file:                {s['csv']}\n")
            f.write(f"  Text summary:            {s['txt']}\n\n")
    log.info("Master summary saved to %s", master_path)
    tg._send(f"✅ *Full Pipeline Complete*\nResults saved to {master_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enhanced NEPSE Indicator Backtester")
    parser.add_argument("mode", choices=["standalone","combos","both","walkforward","full_pipeline"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--from", dest="from_date", default=DEFAULT_TRAIN_FROM)
    parser.add_argument("--to", dest="to_date", default=DEFAULT_TRAIN_TO)
    parser.add_argument("--hold", type=int, default=30)
    parser.add_argument("--symbol", help="Single symbol test")
    parser.add_argument("--min-pf", type=float, default=1.5)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--walkforward-train", type=int, default=12)
    parser.add_argument("--walkforward-test", type=int, default=3)
    parser.add_argument("--sim", choices=["unlimited","constrained_200k","constrained_500k"],
                        default="unlimited", help="Simulation mode (ignored in full_pipeline)")
    parser.add_argument("--export-summary", action="store_true", help="Export detailed text summary")
    args = parser.parse_args()

    if not DATABASE_URL:
        log.error("DATABASE_URL not set")
        sys.exit(1)

    tg = TelegramNotifier()

    if args.mode == "full_pipeline":
        # Run all three sims
        run_full_pipeline(args, tg)
        return

    # Single simulation mode
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"indicator_results_{args.sim}_{timestamp}.csv"
    start_time = time.time()

    tg.notify_start(args.mode, args.from_date, args.to_date, args.fast, args.hold, args.sim)

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
        log.info("Walk‑forward validation mode, sim=%s", args.sim)
        results = walk_forward_run(symbol_data, trading_dates, param_grid, csv_path,
                                   args.hold, args.from_date, args.to_date,
                                   sector_map, args.bootstrap, args.sim,
                                   args.walkforward_train, args.walkforward_test, tg)
        print_summary(results, "Walk‑forward Results")
        if args.export_summary:
            txt_path = RESULTS_DIR / f"summary_{args.sim}_{timestamp}.txt"
            export_text_summary(results, args.sim, txt_path)
    else:
        if args.mode in ("standalone","both"):
            log.info("Standalone phase, sim=%s", args.sim)
            results = run_standalone(symbol_data, trading_dates, param_grid, csv_path,
                                     args.hold, args.from_date, args.to_date,
                                     sector_map, args.bootstrap, args.sim,
                                     args.symbol, tg)
            print_summary(results, "Standalone Results")
            survivors = identify_survivors(results, args.min_pf, args.min_trades)
            tg.notify_survivors(survivors, results)
            if args.export_summary:
                txt_path = RESULTS_DIR / f"summary_{args.sim}_{timestamp}.txt"
                export_text_summary(results, args.sim, txt_path)
            if args.mode == "both" and survivors:
                log.info("Combo phase, sim=%s", args.sim)
                combo_results = run_combos(symbol_data, trading_dates, survivors, csv_path,
                                           args.hold, args.from_date, args.to_date,
                                           sector_map, args.bootstrap, args.sim, tg)
                print_summary(combo_results, "Combo Results")
                if args.export_summary:
                    all_res = results + combo_results
                    export_text_summary(all_res, args.sim, txt_path)
        elif args.mode == "combos":
            log.warning("Combos mode requires standalone first. Run 'both' instead.")
            return

    elapsed = time.time() - start_time
    tg.notify_done([], str(csv_path), elapsed)
    log.info("Done. Elapsed: %s. Results: %s", timedelta(seconds=int(elapsed)), csv_path)

if __name__ == "__main__":
    main()