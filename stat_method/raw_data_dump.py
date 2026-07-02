# -*- coding: utf-8 -*-
"""
raw_data_dump.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Research Tool (stat_method)

RAW DATA ONLY. No scoring, no slopes, no statistical comparison — this just
lists which symbols hit each of 3 outcome categories over a forward 20-day
window, and dumps the raw indicator readings for the 6 trading days
immediately preceding (and including) T0 for each hit.

Categories (forward 20-day return from T0), MUTUALLY EXCLUSIVE — checked in
this order so a -15% move lands in LOSE_10 only, not both LOSE_10 and LOSE_5:
    WINNER   — forward_return_pct >= +10%
    LOSE_10  — forward_return_pct <= -10%
    LOSE_5   — forward_return_pct <= -5%   (and > -10%, since LOSE_10 already
               claimed anything -10% or worse)
Anything that doesn't hit one of these 3 is discarded — this script only
dumps the requested categories, there's no NEUTRAL output file.

Want LOSE_5 to include everything at or beyond -5% instead (so a -15% event
shows up in BOTH lose10_*.csv and lose5_*.csv)? Swap the order of the two
checks in _classify_category() below — that's the only change needed.

Same event-scan shape as the earlier v2_weight_derivation.py:
  - non-overlapping FORWARD_DAYS-day tiles per symbol (NOT a sliding window)
    — avoids the sample-inflation problem already diagnosed in
    winner_loser_events.csv
  - corporate-action guard: |forward_return_pct| > 100% discarded (split /
    bonus / rights artifact, not a real price move)

BB / MACD column lookup is PREFIX-based (startswith), not exact string
match. Confirmed on shanvi@pc: pandas_ta 0.4.71b0 names bbands() columns
with a doubled std suffix (BBP_20_2.0_2.0, not BBP_20_2.0) — exact-match
lookup silently returns None and drops every event. Prefix lookup survives
that and any future pandas_ta renaming.

Output — three CSVs under data/v2/ (relative to project root):
    winners_<timestamp>.csv
    lose10_<timestamp>.csv
    lose5_<timestamp>.csv
Each row = one event: symbol, t0_date, forward_return_pct, category, then
raw close / RSI / MACD-line / MACD-signal / MACD-hist / BB-lower / BB-mid /
BB-upper / BB-%B / OBV values for each of the 6 days T-5..T0 (wide format,
one column per indicator per day — nothing aggregated, nothing scored).

This script is READ-ONLY. No DB writes. Nothing wired into the live pipeline.

Run:
    python -m stat_method.raw_data_dump
    python -m stat_method.raw_data_dump --dry-run   (first 15 symbols only)

Import rule: from sheets import ... — NEVER from db import ...
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import os
import csv
from datetime import datetime

import pandas as pd
import pandas_ta as ta

from config import NST
from sheets import run_raw_sql

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE          = "2021-01-01"
FORWARD_DAYS         = 20     # forward window used to decide the 3 categories
SIGNAL_DAYS          = 6      # backward window T-5..T0 dumped raw (6 points)
MIN_HISTORY_DAYS     = 60     # min prior bars before T0 for stable indicators
WINNER_THRESHOLD     = 10.0
LOSE10_THRESHOLD     = -10.0
LOSE5_THRESHOLD      = -5.0
CORP_ACTION_ABS_PCT  = 100.0

DRY_RUN = "--dry-run" in sys.argv

OUTPUT_DIR = os.path.join(os.path.expanduser("~/nepse-engine"), "data", "v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PRICE HISTORY  (same pattern as spring_loading_precision.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_all_price_history() -> dict[str, pd.DataFrame]:
    """Load all price history since START_DATE. Returns {symbol: DataFrame}."""
    log.info("Loading price_history since %s...", START_DATE)
    rows = run_raw_sql(
        """
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s
        ORDER BY symbol, date ASC
        """,
        (START_DATE,),
    )
    log.info("Loaded %d rows", len(rows))

    def _f(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    sym_data: dict[str, list] = {}
    for r in rows:
        sym   = str(r["symbol"]).upper()
        close = _f(r["close"]) or _f(r["ltp"])
        if close is None:
            continue
        sym_data.setdefault(sym, []).append({
            "date":   r["date"],
            "open":   _f(r["open"]) or close,
            "high":   _f(r["high"]) or close,
            "low":    _f(r["low"])  or close,
            "close":  close,
            "volume": _f(r["volume"]) or 0.0,
        })

    result = {}
    for sym, records in sym_data.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_HISTORY_DAYS + SIGNAL_DAYS + FORWARD_DAYS:
            result[sym] = df

    log.info("Symbols with enough history: %d", len(result))
    if DRY_RUN:
        result = dict(list(result.items())[:15])
        log.info("[DRY RUN] Trimmed to %d symbols", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INDICATOR LOOKUP HELPER (prefix-based — pandas_ta version safe)
# ══════════════════════════════════════════════════════════════════════════════

def _col_by_prefix(df: pd.DataFrame, prefix: str):
    """
    Find the first column starting with `prefix`. Version-safe — pandas_ta's
    suffix formatting (BBP_20_2.0 vs BBP_20_2.0_2.0) varies by installed
    version, so exact-string lookup is fragile. Prefixes here (MACD_, MACDs_,
    MACDh_, BBL_, BBM_, BBU_, BBP_) don't collide with each other.
    """
    for c in df.columns:
        if c.startswith(prefix):
            return df[c]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAW INDICATOR VALUES FOR THE SIGNAL_DAYS WINDOW ENDING AT T0
# ══════════════════════════════════════════════════════════════════════════════

def _compute_raw_at_t0(hist_df: pd.DataFrame):
    """
    hist_df: full price history UP TO AND INCLUDING T0 (so indicators get
    correct warm-up), sorted ascending by date.

    Returns a dict of raw day-by-day values for T-(SIGNAL_DAYS-1)..T0, wide
    format, or None if indicators aren't computable yet. No slopes, no
    aggregation — literally the raw readings for each day.
    """
    if len(hist_df) < MIN_HISTORY_DAYS:
        return None

    close  = hist_df["close"]
    volume = hist_df["volume"]

    try:
        rsi  = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        bb   = ta.bbands(close, length=20, std=2)
        obv  = ta.obv(close, volume)
    except Exception as exc:
        log.debug("_compute_raw_at_t0: indicator calc failed: %s", exc)
        return None

    if rsi is None or macd is None or bb is None or obv is None:
        return None

    cols = {
        "close":       close,
        "rsi":         rsi,
        "macd_line":   _col_by_prefix(macd, "MACD_"),
        "macd_signal": _col_by_prefix(macd, "MACDs_"),
        "macd_hist":   _col_by_prefix(macd, "MACDh_"),
        "bb_lower":    _col_by_prefix(bb, "BBL_"),
        "bb_mid":      _col_by_prefix(bb, "BBM_"),
        "bb_upper":    _col_by_prefix(bb, "BBU_"),
        "bb_pctb":     _col_by_prefix(bb, "BBP_"),
        "obv":         obv,
    }

    missing = [k for k, v in cols.items() if v is None]
    if missing:
        log.debug("_compute_raw_at_t0: missing columns %s", missing)
        return None

    tails = {}
    for name, series in cols.items():
        vals = series.dropna().iloc[-SIGNAL_DAYS:].tolist()
        if len(vals) < SIGNAL_DAYS:
            return None
        tails[name] = vals

    day_labels = [f"t_minus{SIGNAL_DAYS - 1 - i}" for i in range(SIGNAL_DAYS - 1)] + ["t0"]

    raw = {}
    for name, vals in tails.items():
        for i, day in enumerate(day_labels):
            raw[f"{name}_{day}"] = round(vals[i], 4)

    return raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NON-OVERLAPPING EVENT SCAN PER SYMBOL
# ══════════════════════════════════════════════════════════════════════════════

def _classify_category(forward_return_pct: float):
    """
    Mutually exclusive: LOSE_10 claims anything <= -10% before LOSE_5 gets a
    chance at it, so LOSE_5 only ever contains -10% < x <= -5%.
    Swap the order of the two `if` checks below for overlapping buckets.
    """
    if abs(forward_return_pct) > CORP_ACTION_ABS_PCT:
        return None
    if forward_return_pct >= WINNER_THRESHOLD:
        return "WINNER"
    if forward_return_pct <= LOSE10_THRESHOLD:
        return "LOSE_10"
    if forward_return_pct <= LOSE5_THRESHOLD:
        return "LOSE_5"
    return None  # doesn't hit any of the 3 requested categories — discard


def _scan_symbol(sym: str, df: pd.DataFrame) -> list:
    """Non-overlapping tiling: T0 anchors advance by FORWARD_DAYS each step."""
    events = []
    n = len(df)
    t0_idx = MIN_HISTORY_DAYS

    while t0_idx + FORWARD_DAYS < n:
        t0_close  = float(df.iloc[t0_idx]["close"])
        fwd_close = float(df.iloc[t0_idx + FORWARD_DAYS]["close"])

        if t0_close <= 0:
            t0_idx += FORWARD_DAYS
            continue

        forward_return_pct = (fwd_close - t0_close) / t0_close * 100.0
        category = _classify_category(forward_return_pct)

        if category is not None:
            hist_slice = df.iloc[: t0_idx + 1]  # up to & including T0
            raw = _compute_raw_at_t0(hist_slice)
            if raw:
                events.append({
                    "symbol":             sym,
                    "t0_date":            df.iloc[t0_idx]["date"].strftime("%Y-%m-%d"),
                    "forward_return_pct": round(forward_return_pct, 2),
                    "category":           category,
                    **raw,
                })

        t0_idx += FORWARD_DAYS  # non-overlapping jump

    return events


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(rows: list, path: str):
    if not rows:
        log.warning("No rows to write — skipping %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), path)


def run() -> dict:
    ts = datetime.now(tz=NST).strftime("%Y%m%d_%H%M%S")
    log.info("=" * 70)
    log.info("raw_data_dump — forward=%dd, signal=%dd%s",
              FORWARD_DAYS, SIGNAL_DAYS, " [DRY RUN]" if DRY_RUN else "")

    sym_data = _load_all_price_history()
    if not sym_data:
        log.error("No price data — aborting")
        return {}

    all_events: list = []
    for sym, df in sym_data.items():
        all_events.extend(_scan_symbol(sym, df))

    log.info("Total events across 3 categories: %d", len(all_events))
    if not all_events:
        log.warning("No events produced — check thresholds / history depth")
        return {}

    winners = [e for e in all_events if e["category"] == "WINNER"]
    lose10  = [e for e in all_events if e["category"] == "LOSE_10"]
    lose5   = [e for e in all_events if e["category"] == "LOSE_5"]

    log.info("WINNER=%d  LOSE_10=%d  LOSE_5=%d", len(winners), len(lose10), len(lose5))
    log.info("Winner symbols (%d unique): %s", len(set(e["symbol"] for e in winners)),
              sorted(set(e["symbol"] for e in winners)))
    log.info("Lose_10 symbols (%d unique): %s", len(set(e["symbol"] for e in lose10)),
              sorted(set(e["symbol"] for e in lose10)))
    log.info("Lose_5 symbols (%d unique): %s", len(set(e["symbol"] for e in lose5)),
              sorted(set(e["symbol"] for e in lose5)))

    winners_path = os.path.join(OUTPUT_DIR, f"winners_{ts}.csv")
    lose10_path  = os.path.join(OUTPUT_DIR, f"lose10_{ts}.csv")
    lose5_path   = os.path.join(OUTPUT_DIR, f"lose5_{ts}.csv")

    _write_csv(winners, winners_path)
    _write_csv(lose10, lose10_path)
    _write_csv(lose5, lose5_path)

    log.info("Output directory: %s", OUTPUT_DIR)

    return {
        "winners_path": winners_path,
        "lose10_path":  lose10_path,
        "lose5_path":   lose5_path,
        "n_winner":     len(winners),
        "n_lose10":     len(lose10),
        "n_lose5":      len(lose5),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from log_config import attach_file_handler
        attach_file_handler(__name__)
    except Exception:
        pass
    run()