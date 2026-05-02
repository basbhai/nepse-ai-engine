# -*- coding: utf-8 -*-
"""
modules/execution_monitor.py — NEPSE AI Engine
═══════════════════════════════════════════════════════════════════════════════
Real-time intraday position intelligence. Runs as a separate process during
market hours (10:45 AM – 3:00 PM NST, Mon–Fri).

Two responsibilities in one loop:
  1. GUARD     (every 30s)  — hard stop -5%, trailing stop (activates +6%, floor peak-3%)
  2. INTELLIGENCE (every 60s) — GIS score: OBI + TIR + VWAP + Microprice → HOLD/CAUTION/EXIT

Mathematical framework (validated across Gemini, ChatGPT, DeepSeek):
  OBI  — exponential decay weighted order book imbalance (top 10 levels, λ=0.5)
  TIR  — tick rule trade imbalance (5-min rolling, qty≥10 filter)
  VWAP — tanh clamped at ±2% deviation
  Micro— Stoikov microprice using W_bid/W_ask from OBI computation

  GIS  = conf × (0.30×OBI + 0.30×TIR + 0.20×VWAP + 0.20×Micro)
         + 0.20×circuit_risk + 0.20×spread_score

State machine with hysteresis (DeepSeek framework):
  HOLD_STRONG → HOLD if S < 0.25
  HOLD        → CAUTION if S < -0.05
  CAUTION     → EXIT_EARLY if S < -0.25
  EXIT_EARLY  → IMMEDIATE if S < -0.45
  (recovery requires stronger positive signal to prevent whipsawing)

NEPSE-specific:
  - DPR magnet: within 1.5% of upper circuit → suppress exits unless depth collapses
  - Opening 15 min: VWAP + Microprice only (no OBI/TIR — order book chaotic)
  - Closing 15 min: full score, double spread + circuit penalty
  - Odd lot filter: drop ticks where qty < 10
  - Pre-session fragility: loaded from yesterday's floorsheet_signals at startup
  - Confidence multiplicative: low volume → score attenuated, not just penalised

Architecture:
  - No AI calls — pure math only
  - Reads open positions from paper_portfolio (PAPER_MODE) or portfolio (LIVE)
  - LTP from atrad_market_watch DB (no API call for price)
  - getOrderBook + getTradesOfDay called per open position (max 3 symbols)
  - Telegram alert on state change only (never spam)
  - Paper mode → alert only; Live mode → ATrad placeOrder (NOT wired until 55% WR)

Run:
  python -m modules.execution_monitor          # normal
  python -m modules.execution_monitor --once   # single cycle, exit (for testing)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import math
import os
import sys
import time
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional
from modules.atrad_scraper import get_ltp_live
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GUARD_INTERVAL_SEC       = 30     # how often guard + intelligence runs
INTELLIGENCE_INTERVAL_SEC = 60   # how often TIR (5-min tick window) recomputes
MARKET_OPEN_H, MARKET_OPEN_M   = 10, 45
MARKET_CLOSE_H, MARKET_CLOSE_M = 15,  0

# Stop / trail parameters (from backtester & design doc)
HARD_STOP_PCT      = -5.0   # % from entry → immediate exit
TRAIL_ACTIVATE_PCT =  6.0   # % profit → trailing activates
TRAIL_FLOOR_PCT    =  3.0   # trailing floor = peak − 3%

# GIS weights
W_OBI   = 0.30
W_TIR   = 0.30
W_VWAP  = 0.20
W_MICRO = 0.20

# Penalty weights
W_CIRCUIT = 0.20
W_SPREAD  = 0.20

# OBI exponential decay (λ=0.5 → level 10 ≈ 1% weight)
OBI_LAMBDA   = 0.5
OBI_LEVELS   = 10

# VWAP clamp threshold (2% → tanh saturates at ±1)
VWAP_CLAMP   = 0.02

# Minimum 5-min volume for full TIR confidence
TIR_MIN_VOL  = 500
TIR_WINDOW_S = 300   # 5 minutes

# Spread thresholds (basis points)
SPREAD_OK_BP  = 20
SPREAD_MAX_BP = 80

# DPR magnet zone
DPR_MAGNET_PCT = 0.015   # within 1.5% of upper circuit

# Signal state hysteresis thresholds
# Each state has (up_threshold, down_threshold) to move to next state
# These create asymmetric bands — once defensive, need stronger reversal to recover
STATE_THRESHOLDS = {
    # (threshold_to_recover_from_this_state, threshold_to_enter_this_state)
    "HOLD_STRONG": {"drop_to_hold":     0.25},
    "HOLD":        {"rise_to_strong":   0.45, "drop_to_caution": -0.05},
    "CAUTION":     {"rise_to_hold":     0.25, "drop_to_exit":    -0.25},
    "EXIT_EARLY":  {"rise_to_caution":  0.05, "drop_to_immed":   -0.45},
    "IMMEDIATE":   {"rise_to_exit":     0.05},
}

# Instant exit (skip 2-cycle persistence)
INSTANT_EXIT_S   = -0.45
INSTANT_EXIT_TIR = -0.50

# Liquidity shock threshold
LIQUIDITY_DROP_THRESH = -0.35

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (after env loaded)
# ─────────────────────────────────────────────────────────────────────────────

from db.connection import _db
from sheets import get_setting, run_raw_sql


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIME HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _now_nst() -> datetime:
    return datetime.now(tz=NST)


def _is_market_open() -> bool:
    now = _now_nst()
    open_  = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_ = now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return open_ <= now <= close_


def _market_phase() -> str:
    """Return OPENING (first 15 min), CLOSING (last 15 min), or NORMAL."""
    now   = _now_nst()
    open_ = now.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
    close_= now.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    if now < open_ + timedelta(minutes=15):
        return "OPENING"
    if now > close_ - timedelta(minutes=15):
        return "CLOSING"
    return "NORMAL"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — POSITION LOADER
# ═══════════════════════════════════════════════════════════════════════════

def _load_open_positions(paper_mode: bool) -> list[dict]:
    """
    Load all open positions. Max 3 (enforced upstream by gemini_filter).
    Paper mode: paper_portfolio. Live mode: portfolio.
    Returns list of dicts with keys: symbol, entry_price, shares, stop_loss, target,
    peak_price, trail_active, telegram_id (paper only).
    """
    try:
        if paper_mode:
            rows = run_raw_sql("""
                SELECT pp.id, pp.telegram_id, pp.symbol,
                       pp.wacc         AS entry_price,
                       pp.total_shares AS shares,
                       pp.total_cost,
                       pp.first_buy_date,
                       pp.updated_at
                FROM paper_portfolio pp
                WHERE pp.status = 'OPEN'
                ORDER BY pp.first_buy_date ASC
                LIMIT 3
            """)
            # Enrich with stop/target from latest market_log BUY signal
            for row in rows:
                _enrich_from_market_log(row)
            return rows or []
        else:
            rows = run_raw_sql("""
                SELECT id, symbol, entry_price, shares, stop_level AS stop_loss,
                       peak_price, trail_active, trail_stop
                FROM portfolio
                WHERE status = 'OPEN'
                ORDER BY entry_date ASC
                LIMIT 3
            """)
            return rows or []
    except Exception as e:
        log.error("load_open_positions failed: %s", e)
        return []


def _enrich_from_market_log(row: dict) -> None:
    """Pull stop_loss + target from the most recent BUY signal for this symbol."""
    try:
        symbol = row.get("symbol", "")
        rows = run_raw_sql("""
            SELECT stop_loss, target, entry_price
            FROM market_log
            WHERE symbol = %s AND action = 'BUY'
            ORDER BY id DESC LIMIT 1
        """, (symbol,))
        if rows:
            ml = rows[0]
            row["stop_loss"] = float(ml.get("stop_loss") or 0) or None
            row["target"]    = float(ml.get("target")    or 0) or None
        else:
            row["stop_loss"] = None
            row["target"]    = None
        # peak_price: use current entry as starting peak if not tracked
        row["peak_price"]   = float(row.get("entry_price") or 0)
        row["trail_active"] = False
    except Exception as e:
        log.warning("enrich_from_market_log(%s): %s", row.get("symbol"), e)
        row["stop_loss"] = None
        row["target"]    = None
        row["peak_price"] = float(row.get("entry_price") or 0)
        row["trail_active"] = False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LTP & MARKET WATCH (DB read — no API call)
# ═══════════════════════════════════════════════════════════════════════════

def _get_ltp_from_db(symbol: str) -> Optional[dict]:
    """
    Read latest row for symbol from atrad_market_watch.
    Returns dict with ltp, vwap, vwap_dev, bid_price, bid_qty, ask_price, ask_qty,
    low_dpr, high_dpr, dpr_proximity, or None if not found.
    """
    try:
        rows = run_raw_sql("""
            SELECT ltp, vwap, vwap_dev, bid_price, bid_qty, ask_price, ask_qty,
                   low_dpr, high_dpr, dpr_proximity, time
            FROM atrad_market_watch
            WHERE symbol = %s
            ORDER BY id DESC LIMIT 1
        """, (symbol,))
        if not rows:
            return None
        r = rows[0]
        return {k: float(v) if v not in (None, "", "None") else 0.0 for k, v in r.items()
                if k != "time"} | {"time": r.get("time", "")}
    except Exception as e:
        log.error("get_ltp_from_db(%s): %s", symbol, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRE-SESSION FRAGILITY FLAG (from yesterday's floorsheet_signals)
# ═══════════════════════════════════════════════════════════════════════════

def _load_fragility_flags(symbols: list[str]) -> dict[str, bool]:
    """
    Load yesterday's floorsheet_signals broker_concentration for open positions.
    Returns {symbol: is_fragile} where is_fragile = broker_concentration > 0.6
    for 3+ consecutive days (single dominant buyer).
    Called once at startup — not every cycle.
    """
    flags = {s: False for s in symbols}
    if not symbols:
        return flags
    try:
        placeholders = ",".join(["%s"] * len(symbols))
        rows = run_raw_sql(f"""
            SELECT symbol, broker_concentration, date
            FROM floorsheet_signals
            WHERE symbol IN ({placeholders})
            ORDER BY symbol, date DESC
        """, tuple(symbols))

        from itertools import groupby
        for symbol, group in groupby(rows, key=lambda r: r["symbol"]):
            recent = list(group)[:5]   # last 5 trading days
            high_conc_days = sum(
                1 for r in recent
                if float(r.get("broker_concentration") or 0) > 0.6
            )
            # Fragile if dominant broker for 3+ of last 5 days
            flags[symbol] = high_conc_days >= 3
            if flags[symbol]:
                log.info("FRAGILE flag set for %s (%d/5 days high broker concentration)",
                         symbol, high_conc_days)
    except Exception as e:
        log.warning("load_fragility_flags failed: %s — all flags = False", e)
    return flags


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — MATH: OBI (Exponential Decay Weighted Order Book Imbalance)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_obi(bids: list, asks: list) -> tuple[float, float, float, float, float]:
    """
    Exponential decay weighted OBI using top OBI_LEVELS levels.
    λ=0.5 → level 1 weight=1.0, level 10 weight≈0.011

    Returns:
        obi      ∈ [-1, 1]
        W_bid    (weighted bid qty — used for microprice)
        W_ask    (weighted ask qty — used for microprice)
        best_bid (level 1 bid price)
        best_ask (level 1 ask price)
    """
    if not bids or not asks:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def _safe(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    W_bid = 0.0
    W_ask = 0.0

    for i, b in enumerate(bids[:OBI_LEVELS]):
        w = math.exp(-OBI_LAMBDA * i)
        W_bid += w * _safe(b.get("qty", 0))

    for i, a in enumerate(asks[:OBI_LEVELS]):
        w = math.exp(-OBI_LAMBDA * i)
        W_ask += w * _safe(a.get("qty", 0))

    total = W_bid + W_ask
    obi   = (W_bid - W_ask) / total if total > 0 else 0.0

    best_bid = _safe(bids[0].get("price", 0)) if bids else 0.0
    best_ask = _safe(asks[0].get("price", 0)) if asks else 0.0

    return obi, W_bid, W_ask, best_bid, best_ask


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — MATH: Microprice (Stoikov)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_microprice(W_bid: float, W_ask: float,
                        best_bid: float, best_ask: float,
                        ltp: float) -> float:
    """
    Stoikov microprice using exponential-decay weighted quantities from OBI.
    microprice = mid + (imbalance - 0.5) × spread

    Returns micro_score ∈ [-1, 1]:
        positive → upward pressure (microprice above LTP)
        negative → downward pressure (microprice below LTP)
    """
    if best_bid <= 0 or best_ask <= 0 or ltp <= 0:
        return 0.0

    mid        = (best_bid + best_ask) / 2.0
    spread     = best_ask - best_bid
    total      = W_bid + W_ask
    imbalance  = W_bid / total if total > 0 else 0.5
    microprice = mid + (imbalance - 0.5) * spread

    if ltp <= 0:
        return 0.0

    micro_dev = (microprice - ltp) / ltp
    # tanh clamp at 1% deviation → ±1
    return math.tanh(micro_dev / 0.01)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — MATH: TIR (Tick Rule Trade Imbalance, 5-min rolling)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_tir(trades: list) -> tuple[float, float]:
    """
    Tick rule aggressor classification on trades from last TIR_WINDOW_S seconds.
    Filters out odd lots (qty < 10).
    Returns:
        tir     ∈ [-1, 1]  (buy pressure minus sell pressure)
        v_total (total volume in window — used for confidence factor)
    """
    if not trades:
        return 0.0, 0.0

    now_ts = time.time()
    cutoff = now_ts - TIR_WINDOW_S

    # Filter to window and non-odd-lot
    recent = []
    for t in trades:
        try:
            qty = float(t.get("qty", 0) or t.get("quantity", 0) or 0)
            if qty < 10:
                continue
            ts = t.get("timestamp", 0) or t.get("time", 0)
            if ts and float(ts) < cutoff:
                continue
            recent.append({"price": float(t.get("price", 0) or t.get("rate", 0) or 0),
                           "qty":   qty})
        except (TypeError, ValueError):
            continue

    if not recent:
        return 0.0, 0.0

    v_buy   = 0.0
    v_sell  = 0.0
    last_dir = "buy"   # initial assumption: first tick is buy
    prev_price = None

    for tick in recent:
        price = tick["price"]
        qty   = tick["qty"]
        if prev_price is None:
            direction = last_dir
        elif price > prev_price:
            direction = "buy"
        elif price < prev_price:
            direction = "sell"
        else:
            direction = last_dir   # zero-tick carry forward

        if direction == "buy":
            v_buy += qty
        else:
            v_sell += qty

        last_dir   = direction
        prev_price = price

    v_total = v_buy + v_sell
    tir     = (v_buy - v_sell) / v_total if v_total > 0 else 0.0
    return tir, v_total


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — MATH: VWAP Score
# ═══════════════════════════════════════════════════════════════════════════

def _compute_vwap_score(ltp: float, vwap: float) -> float:
    """
    VWAP deviation score, tanh clamped at ±2%.
    Returns ∈ [-1, 1].
    """
    if vwap <= 0 or ltp <= 0:
        return 0.0
    dev = (ltp - vwap) / vwap
    return math.tanh(dev / VWAP_CLAMP)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — MATH: Penalty Scores (Circuit Risk + Spread)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_circuit_risk(ltp: float, low_dpr: float, high_dpr: float) -> float:
    """
    Penalty when approaching upper circuit (last 10% of band).
    Returns ∈ [-1, 0].
    """
    band = high_dpr - low_dpr
    if band <= 0 or high_dpr <= 0:
        return 0.0
    upper_dist = (high_dpr - ltp) / band
    if upper_dist < 0.10:
        return -((0.10 - upper_dist) / 0.10)
    return 0.0


def _compute_spread_score(best_bid: float, best_ask: float, mid: float,
                          phase: str) -> float:
    """
    Liquidity penalty for wide spread.
    Closing phase: doubled penalty.
    Returns ∈ [-1, 0].
    """
    if mid <= 0 or best_bid <= 0 or best_ask <= 0:
        return 0.0
    spread_bps = (best_ask - best_bid) / mid * 10_000
    raw = -min(1.0, max(0.0, (spread_bps - SPREAD_OK_BP) / (SPREAD_MAX_BP - SPREAD_OK_BP)))
    return raw * 2 if phase == "CLOSING" else raw


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — COMPOSITE GIS SCORE
# ═══════════════════════════════════════════════════════════════════════════

def _compute_gis(
    ltp: float,
    vwap: float,
    low_dpr: float,
    high_dpr: float,
    bids: list,
    asks: list,
    trades: list,
    phase: str,
    is_fragile: bool,
) -> dict:
    """
    Compute Global Intraday Score (GIS) for one symbol.
    Returns dict with all component scores + final S + TIR (for instant-exit check).
    """
    result = {
        "obi": 0.0, "tir": 0.0, "vwap_score": 0.0, "micro_score": 0.0,
        "circuit_risk": 0.0, "spread_score": 0.0, "conf": 0.0,
        "S": 0.0, "v_total": 0.0,
        "best_bid": 0.0, "best_ask": 0.0, "W_bid": 0.0, "W_ask": 0.0,
        "dpr_magnet": False,
    }

    # ── DPR magnet check ─────────────────────────────────────────────────────
    band = high_dpr - low_dpr
    upper_dist_pct = (high_dpr - ltp) / ltp if ltp > 0 else 1.0
    dpr_magnet = (upper_dist_pct < DPR_MAGNET_PCT) and (band > 0)
    result["dpr_magnet"] = dpr_magnet

    # ── OBI + microprice ──────────────────────────────────────────────────────
    obi, W_bid, W_ask, best_bid, best_ask = _compute_obi(bids, asks)
    result.update({"obi": obi, "W_bid": W_bid, "W_ask": W_ask,
                   "best_bid": best_bid, "best_ask": best_ask})

    micro_score = _compute_microprice(W_bid, W_ask, best_bid, best_ask, ltp)
    result["micro_score"] = micro_score

    # ── TIR ──────────────────────────────────────────────────────────────────
    tir, v_total = _compute_tir(trades)
    result["tir"]     = tir
    result["v_total"] = v_total

    # ── VWAP ─────────────────────────────────────────────────────────────────
    vwap_score = _compute_vwap_score(ltp, vwap)
    result["vwap_score"] = vwap_score

    # ── Confidence (multiplicative) ───────────────────────────────────────────
    conf = min(1.0, v_total / TIR_MIN_VOL)
    result["conf"] = conf

    # ── Penalties ────────────────────────────────────────────────────────────
    mid          = (best_bid + best_ask) / 2.0 if best_bid and best_ask else ltp
    circuit_risk = _compute_circuit_risk(ltp, low_dpr, high_dpr)
    spread_score = _compute_spread_score(best_bid, best_ask, mid, phase)
    result["circuit_risk"] = circuit_risk
    result["spread_score"] = spread_score

    # ── Phase-adjusted weights ────────────────────────────────────────────────
    if phase == "OPENING":
        # First 15 min: no microstructure — VWAP + microprice only
        S_base = 0.50 * vwap_score + 0.50 * micro_score
        S_adj  = S_base   # no confidence factor (no tick data yet)
    elif dpr_magnet:
        # Near upper circuit: OBI suppressed, TIR dominates
        S_base = 0.00 * obi + 1.00 * tir + 0.0 * vwap_score + 0.0 * micro_score
        S_adj  = conf * S_base
    else:
        w_obi = W_OBI
        w_tir = W_TIR
        # Fragile flag (high broker concentration from floorsheet): reduce OBI, boost TIR
        if is_fragile:
            w_obi -= 0.10
            w_tir += 0.10
        S_base = w_obi * obi + w_tir * tir + W_VWAP * vwap_score + W_MICRO * micro_score
        S_adj  = conf * S_base

    # Closing phase: double penalties
    penalty_mult = 2.0 if phase == "CLOSING" else 1.0
    S = S_adj + penalty_mult * W_CIRCUIT * circuit_risk + penalty_mult * W_SPREAD * spread_score

    # Clamp to [-1, 1]
    S = max(-1.0, min(1.0, S))
    result["S"] = S

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — STATE MACHINE WITH HYSTERESIS
# ═══════════════════════════════════════════════════════════════════════════

STATES = ["HOLD_STRONG", "HOLD", "CAUTION", "EXIT_EARLY", "IMMEDIATE"]

def _next_state(current: str, S: float) -> str:
    """
    State transition with hysteresis (asymmetric bands).
    Requires 2-cycle persistence (handled by caller).
    """
    if current == "HOLD_STRONG":
        if S < 0.25:  return "HOLD"

    elif current == "HOLD":
        if S > 0.45:   return "HOLD_STRONG"
        if S < -0.05:  return "CAUTION"

    elif current == "CAUTION":
        if S > 0.25:   return "HOLD"
        if S < -0.25:  return "EXIT_EARLY"

    elif current == "EXIT_EARLY":
        if S > 0.05:   return "CAUTION"
        if S < -0.45:  return "IMMEDIATE"

    elif current == "IMMEDIATE":
        if S > 0.05:   return "EXIT_EARLY"

    return current   # no change


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — GUARD: Hard Stop + Trailing Stop
# ═══════════════════════════════════════════════════════════════════════════

def _check_guard(pos: dict, ltp: float, state_store: dict) -> Optional[str]:
    """
    Pure math guard. Checks hard stop and trailing stop.
    Returns exit reason string or None.
    Mutates pos["peak_price"] and pos["trail_active"] in-place.
    """
    entry = float(pos.get("entry_price") or pos.get("wacc") or 0)
    if entry <= 0 or ltp <= 0:
        return None

    profit_pct = (ltp - entry) / entry * 100

    # Hard stop: -5% from entry (always fires regardless of GIS)
    if profit_pct <= HARD_STOP_PCT:
        return "HARD_STOP"

    # Check claude-set stop_loss (from market_log)
    stop_loss = pos.get("stop_loss")
    if stop_loss and ltp <= float(stop_loss):
        return "STOP_LOSS"

    # Check claude-set target (from market_log)
    target = pos.get("target")
    if target and ltp >= float(target):
        return "TARGET_HIT"

    # Trailing stop
    peak = float(pos.get("peak_price") or entry)
    if ltp > peak:
        pos["peak_price"] = ltp
        peak = ltp

    peak_profit_pct = (peak - entry) / entry * 100
    if peak_profit_pct >= TRAIL_ACTIVATE_PCT:
        pos["trail_active"] = True

    if pos.get("trail_active"):
        floor = peak * (1 - TRAIL_FLOOR_PCT / 100)
        if ltp <= floor:
            return "TRAILING_STOP"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13 — TELEGRAM ALERTS
# ═══════════════════════════════════════════════════════════════════════════

def _send_telegram(text: str) -> None:
    """Send Telegram message to admin. Non-blocking best-effort."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — alert suppressed: %s", text[:80])
        return
    try:
        import requests
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.error("Telegram send failed: %s", e)


def _format_guard_alert(symbol: str, reason: str, ltp: float,
                        entry: float, paper_mode: bool) -> str:
    pnl_pct = (ltp - entry) / entry * 100 if entry > 0 else 0
    mode    = "📄 PAPER" if paper_mode else "⚡ LIVE"
    emoji   = {"HARD_STOP": "🛑", "STOP_LOSS": "🛑", "TRAILING_STOP": "🔒",
               "TARGET_HIT": "🎯"}.get(reason, "⚠️")
    return (
        f"{emoji} *{reason}* — {symbol} | {mode}\n"
        f"LTP: {ltp:.2f} | Entry: {entry:.2f} | P&L: {pnl_pct:+.1f}%"
    )


def _format_intel_alert(symbol: str, old_state: str, new_state: str,
                        gis: dict, paper_mode: bool) -> str:
    mode  = "📄 PAPER" if paper_mode else "⚡ LIVE"
    emoji = {
        "HOLD_STRONG": "🟢", "HOLD": "🟢",
        "CAUTION": "🟡", "EXIT_EARLY": "🔴", "IMMEDIATE": "🚨"
    }.get(new_state, "❓")
    S    = gis["S"]
    obi  = gis["obi"]
    tir  = gis["tir"]
    vwap = gis["vwap_score"]
    mic  = gis["micro_score"]
    conf = gis["conf"]
    mag  = " 🧲DPR" if gis.get("dpr_magnet") else ""
    return (
        f"{emoji} *{new_state}* — {symbol} | {mode}{mag}\n"
        f"Score: {S:+.3f} (was {old_state})\n"
        f"OBI: {obi:+.2f} | TIR: {tir:+.2f} | VWAP: {vwap:+.2f} | Micro: {mic:+.2f}\n"
        f"Confidence: {conf:.0%}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14 — LIQUIDITY SHOCK DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def _check_liquidity_shock(symbol: str, current_depth: float,
                           prev_depth_store: dict) -> bool:
    """
    Detect sudden depth collapse (>35% drop in total order book depth).
    Returns True if liquidity shock detected.
    """
    prev = prev_depth_store.get(symbol, current_depth)
    shock = False
    if prev > 0:
        drop = (current_depth - prev) / prev
        if drop < LIQUIDITY_DROP_THRESH:
            shock = True
            log.warning("%s: Liquidity shock detected (depth drop %.1f%%)",
                        symbol, drop * 100)
    prev_depth_store[symbol] = current_depth
    return shock


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15 — MAIN MONITOR LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_monitor(paper_mode: bool = True, once: bool = False) -> None:
    """
    Main execution monitor loop.
    paper_mode=True  → alerts only, no real orders
    once=True        → single cycle then exit (for testing)
    """
    log.info("=" * 65)
    log.info("EXECUTION MONITOR starting | mode=%s", "PAPER" if paper_mode else "LIVE")
    log.info("=" * 65)

    from modules.atrad_scraper import fetch_order_book, _ensure_session

    # Per-symbol rolling state
    intel_states:    dict[str, str]          = {}   # current GIS state per symbol
    pending_states:  dict[str, tuple]        = {}   # (proposed_state, cycle_count)
    prev_depth:      dict[str, float]        = {}   # for liquidity shock detection
    tir_cache:       dict[str, tuple]        = {}   # (tir, v_total, computed_at)
    fragility_flags: dict[str, bool]         = {}   # loaded once at startup
    fragility_loaded: bool                   = False

    last_intel_time: float = 0.0

    _ensure_session()

    while True:
        cycle_start = time.time()

        if not _is_market_open():
            if once:
                log.info("Market closed — exiting (--once mode)")
                return
            log.info("Market closed — sleeping 60s")
            time.sleep(60)
            continue

        phase = _market_phase()

        # ── Load open positions ───────────────────────────────────────────────
        positions = _load_open_positions(paper_mode)
        if not positions:
            log.info("No open positions — sleeping %ds", GUARD_INTERVAL_SEC)
            if once:
                return
            time.sleep(GUARD_INTERVAL_SEC)
            continue

        symbols = [p.get("symbol", "") for p in positions]

        # ── Load fragility flags once at startup or when positions change ─────
        if not fragility_loaded or set(symbols) != set(fragility_flags.keys()):
            fragility_flags  = _load_fragility_flags(symbols)
            fragility_loaded = True

        # ── Decide whether to recompute TIR this cycle ────────────────────────
        now_ts          = time.time()
        run_intelligence = (now_ts - last_intel_time) >= INTELLIGENCE_INTERVAL_SEC

        # ─────────────────────────────────────────────────────────────────────
        # Per-symbol processing
        # ─────────────────────────────────────────────────────────────────────
        for pos in positions:
            symbol = pos.get("symbol", "")
            if not symbol:
                continue

            # ── 1. Get LTP from DB (no API call) ─────────────────────────────
            mw = get_ltp_live(symbol)
            if not mw:
                log.warning("%s: no market watch data — skipping", symbol)
                continue

            ltp      = mw["ltp"]
            vwap     = mw["vwap"]
            low_dpr  = mw["low_dpr"]
            high_dpr = mw["high_dpr"]
            entry    = float(pos.get("entry_price") or pos.get("wacc") or 0)

            if ltp <= 0:
                log.warning("%s: LTP=0 in DB — skipping", symbol)
                continue

            # ── 2. GUARD — hard stop / trailing (every cycle) ─────────────────
            guard_reason = _check_guard(pos, ltp, {})
            if guard_reason:
                alert = _format_guard_alert(symbol, guard_reason, ltp, entry, paper_mode)
                log.warning(alert.replace("*", "").replace("_", " "))
                _send_telegram(alert)
                if not paper_mode:
                    log.warning("[LIVE] Would placeOrder SELL %s @ market — NOT WIRED YET", symbol)
                # Don't skip intelligence after guard — still want GIS info
                if once:
                    return

            # ── 3. INTELLIGENCE — GIS score ───────────────────────────────────
            if run_intelligence:
                # Fetch order book (live API call)
                book   = fetch_order_book(symbol)
                bids   = book.get("bids", [])
                asks   = book.get("asks", [])

                # Fetch trades (live API call) — only if not opening phase
                if phase != "OPENING":
                    raw_trades = fetch_trades(symbol) if hasattr(fetch_trades, "__call__") else []
                    # Cache TIR result
                    tir, v_total = _compute_tir(raw_trades)
                    tir_cache[symbol] = (tir, v_total, now_ts)
                else:
                    tir, v_total = 0.0, 0.0
                    tir_cache[symbol] = (0.0, 0.0, now_ts)
            else:
                # Use cached TIR, fetch fresh order book only
                book  = fetch_order_book(symbol)
                bids  = book.get("bids", [])
                asks  = book.get("asks", [])
                cached = tir_cache.get(symbol, (0.0, 0.0, 0))
                tir, v_total = cached[0], cached[1]

            # Depth for liquidity shock check
            total_depth = book.get("total_bid_qty", 0) + book.get("total_ask_qty", 0)
            liq_shock   = _check_liquidity_shock(symbol, total_depth, prev_depth)

            # Compute GIS
            # Build trade dicts from cache if needed
            dummy_trades = [{"price": ltp, "qty": v_total, "timestamp": now_ts}] if v_total else []
            gis = _compute_gis(
                ltp=ltp, vwap=vwap,
                low_dpr=low_dpr, high_dpr=high_dpr,
                bids=bids, asks=asks,
                trades=dummy_trades if not run_intelligence else [],
                phase=phase,
                is_fragile=fragility_flags.get(symbol, False),
            )
            # Inject cached TIR into gis if not freshly computed
            if not run_intelligence:
                gis["tir"]     = tir
                gis["v_total"] = v_total
                gis["conf"]    = min(1.0, v_total / TIR_MIN_VOL)
                # Recompute S with cached TIR
                S_base = W_OBI * gis["obi"] + W_TIR * tir + W_VWAP * gis["vwap_score"] + W_MICRO * gis["micro_score"]
                S_adj  = gis["conf"] * S_base
                penalty_mult = 2.0 if phase == "CLOSING" else 1.0
                S = S_adj + penalty_mult * W_CIRCUIT * gis["circuit_risk"] + penalty_mult * W_SPREAD * gis["spread_score"]
                gis["S"] = max(-1.0, min(1.0, S))

            S   = gis["S"]
            tir = gis["tir"]

            # DPR magnet: suppress exit signals (but not hard stops already handled above)
            dpr_magnet = gis.get("dpr_magnet", False)

            # ── 4. Instant exit override (skip 2-cycle persistence) ───────────
            if S < INSTANT_EXIT_S and tir < INSTANT_EXIT_TIR and not dpr_magnet:
                old_state = intel_states.get(symbol, "HOLD")
                if old_state != "IMMEDIATE":
                    intel_states[symbol] = "IMMEDIATE"
                    alert = _format_intel_alert(symbol, old_state, "IMMEDIATE", gis, paper_mode)
                    log.warning(alert.replace("*", "").replace("_", " "))
                    _send_telegram(alert)
                    if not paper_mode:
                        log.warning("[LIVE] IMMEDIATE EXIT %s — NOT WIRED YET", symbol)
                if once:
                    return
                continue

            # Liquidity shock: force upgrade toward EXIT
            if liq_shock:
                current = intel_states.get(symbol, "HOLD")
                upgrade_map = {"HOLD_STRONG": "CAUTION", "HOLD": "CAUTION",
                               "CAUTION": "EXIT_EARLY"}
                upgraded = upgrade_map.get(current, current)
                if upgraded != current:
                    intel_states[symbol] = upgraded
                    alert = (f"⚠️ *LIQUIDITY SHOCK* — {symbol}\n"
                             f"Depth collapsed. State: {current} → {upgraded}")
                    _send_telegram(alert)

            # DPR magnet: only allow CAUTION at most during magnet zone
            if dpr_magnet:
                proposed = _next_state(intel_states.get(symbol, "HOLD"), S)
                if proposed in ("EXIT_EARLY", "IMMEDIATE"):
                    # Check if bid depth collapsed (>40% drop) or microprice negative
                    bid_collapse  = (prev_depth.get(f"{symbol}_bid", total_depth) > 0 and
                                     total_depth / prev_depth.get(f"{symbol}_bid", total_depth) < 0.60)
                    micro_neg     = gis["micro_score"] < 0
                    if not (bid_collapse or micro_neg):
                        proposed = "CAUTION"   # suppress to CAUTION in magnet zone
                intel_states[symbol] = proposed
                prev_depth[f"{symbol}_bid"] = total_depth

            # ── 5. 2-cycle persistence state machine ─────────────────────────
            current_state = intel_states.get(symbol, "HOLD")
            proposed      = _next_state(current_state, S)

            if proposed != current_state:
                prev_prop, count = pending_states.get(symbol, (proposed, 0))
                if prev_prop == proposed:
                    count += 1
                else:
                    count = 1
                pending_states[symbol] = (proposed, count)

                if count >= 2:
                    # Confirmed — transition
                    old_state = current_state
                    intel_states[symbol] = proposed
                    pending_states.pop(symbol, None)

                    alert = _format_intel_alert(symbol, old_state, proposed, gis, paper_mode)
                    log.info(alert.replace("*", "").replace("_", " "))
                    _send_telegram(alert)

                    if proposed in ("EXIT_EARLY", "IMMEDIATE") and not paper_mode:
                        log.warning("[LIVE] EXIT signal for %s — NOT WIRED YET", symbol)
            else:
                pending_states.pop(symbol, None)   # reset pending if back to current

            log.info("%s | S=%+.3f | OBI=%+.2f TIR=%+.2f VWAP=%+.2f Micro=%+.2f | "
                     "State=%s | Phase=%s | Fragile=%s",
                     symbol, S, gis["obi"], gis["tir"], gis["vwap_score"],
                     gis["micro_score"], intel_states.get(symbol, "HOLD"),
                     phase, fragility_flags.get(symbol, False))

        if run_intelligence:
            last_intel_time = now_ts

        # ── Cycle timing ──────────────────────────────────────────────────────
        elapsed = time.time() - cycle_start
        log.info("Monitor cycle: %.2fs | %d positions | phase=%s",
                 elapsed, len(positions), phase)

        if once:
            log.info("--once mode: exiting after single cycle")
            return

        sleep_time = max(0, GUARD_INTERVAL_SEC - elapsed)
        time.sleep(sleep_time)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 16 — FETCH TRADES (wrapper — uses atrad_scraper)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_trades(symbol: str) -> list:
    """
    Fetch tick trades for symbol via ATrad getTradesOfDay.
    Returns list of dicts with price, qty, timestamp keys.
    Fails silently — returns [] on error.
    """
    try:
        from modules.atrad_scraper import _session, BASE_URL, _parse
        import time as _time
        r = _session.get(
            f"{BASE_URL}/marketdetails",
            params={
                "action": "getTradesOfDay", "format": "json",
                "security": symbol, "board": "1",
                "dojo.preventCache": str(int(_time.time() * 1000))
            },
            timeout=15,
        )
        data = _parse(r)
        if data.get("code") != "0":
            return []
        trades_raw = data.get("data", {}).get("trade", [])
        result = []
        for t in trades_raw:
            try:
                result.append({
                    "price":     float(t.get("price") or t.get("rate") or 0),
                    "qty":       float(t.get("qty") or t.get("quantity") or 0),
                    "timestamp": float(t.get("timestamp") or 0),
                })
            except (TypeError, ValueError):
                continue
        return result
    except Exception as e:
        log.error("fetch_trades(%s): %s", symbol, e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    parser = argparse.ArgumentParser(description="NEPSE Execution Monitor")
    parser.add_argument("--live",  action="store_true", help="Live mode (default: paper)")
    parser.add_argument("--once",  action="store_true", help="Single cycle then exit (testing)")
    args = parser.parse_args()

    paper_mode = not args.live
    if not paper_mode:
        log.warning("⚡ LIVE MODE — exits will trigger real ATrad orders (NOT WIRED YET)")

    # Pre-flight check — exit immediately if no open positions
    positions = _load_open_positions(paper_mode)
    if not positions:
        log.info("No open positions — execution monitor exiting (nothing to guard)")
        sys.exit(0)

    log.info("Found %d open position(s) — starting monitor", len(positions))

    try:
        run_monitor(paper_mode=paper_mode, once=args.once)
    except KeyboardInterrupt:
        log.info("Interrupted by user — monitor stopped")


if __name__ == "__main__":
    main()
