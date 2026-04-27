# -*- coding: utf-8 -*-
"""
recommendation_tracker.py -- NEPSE AI Engine
============================================
Tracks ALL signals from market_log daily: BUY, WAIT, AVOID.
Stamps outcome after hold period expires.
Captures macro snapshot + key news at evaluation time.
GPT Sunday reviewer reads this alongside trade_journal.

Changes from original:
  - Covers BUY rows too (not just WAIT/AVOID) -- GPT needs full picture
  - AVOID: stamped immediately as CLOSED on write (no hold tracking needed)
  - WAIT: auto-expires after 5 calendar days if still PENDING
  - BUY: evaluated for alpha vs NEPSE after hold period (signal quality, not WIN/LOSS)
  - All outcomes flow to learning_hub for GPT review

FIX 4: All upsert_row calls replaced with update_row.
FIX 5: Expired WAIT branch now writes all eval fields (was missing
  eval_nepse_change_pct, eval_alpha, eval_geo_delta, eval_nepal_delta,
  eval_nepse_index). Indentation corrected.
FIX 6: trading_days_between() updated to Mon-Fri (2026 NEPSE schedule).
  Previously counted Sun-Thu which under-counted days elapsed.

Run modes:
    python -m analysis.recommendation_tracker            # normal daily run
    python -m analysis.recommendation_tracker --status   # show all signals summary
    python -m analysis.recommendation_tracker --dry-run  # compute outcomes, write nothing

Called by:
    eod_workflow.py  (3:15 PM NST -- final daily evaluation)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from sheets import (
    read_tab, update_row, run_raw_sql,
    get_setting, get_latest_geo, get_latest_pulse
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

HOLD_DAYS = {
    "MACD":           17,
    "MACD_CROSS":     17,
    "BB":             60,
    "BB_LOWER":       60,
    "BB_LOWER_TOUCH": 60,
    "SMA":            33,
    "SMA_GOLDEN":     33,
    "SMA_CROSS":      33,
    "CANDLE":         17,
    "COMBO":          17,
    "DEFAULT":        17,
}

# WAIT auto-expiry: if still PENDING after N calendar days, force stamp
WAIT_EXPIRY_DAYS = 5

# Outcome thresholds
CORRECT_AVOID_THRESHOLD = -2.0
FALSE_AVOID_THRESHOLD   =  5.0
MISSED_ENTRY_THRESHOLD  =  8.0
CORRECT_WAIT_THRESHOLD  = -2.0

# BUY signal quality thresholds (alpha vs NEPSE)
BUY_STRONG_ALPHA =  5.0
BUY_WEAK_ALPHA   = -5.0


# ---------------------------------------------------------------------------
# DATE HELPERS
# ---------------------------------------------------------------------------

def now_nst() -> datetime:
    return datetime.now(NST)

def today_str() -> str:
    return now_nst().strftime("%Y-%m-%d")

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def calendar_days_between(start: str, end: str) -> int:
    try:
        return (parse_date(end) - parse_date(start)).days
    except Exception:
        return 0

def trading_days_between(start: str, end: str) -> int:
    """
    Count NEPSE trading days between two date strings.
    FIX 6: Updated to Mon-Fri (0-4) for 2026 schedule.
    Previously used Sun-Thu (0,1,2,3,6) which was wrong.
    """
    s = parse_date(start)
    e = parse_date(end)
    count = 0
    current = s
    while current < e:
        if current.weekday() in (0, 1, 2, 3, 4):  # Mon-Fri
            count += 1
        current += timedelta(days=1)
    return count


# ---------------------------------------------------------------------------
# SIGNAL TYPE RESOLVER
# ---------------------------------------------------------------------------

def resolve_hold_days(row: dict) -> int:
    cp        = (row.get("candle_pattern") or "").upper()
    reasoning = (row.get("reasoning")      or "").upper()
    for key in HOLD_DAYS:
        if key in cp or key in reasoning:
            return HOLD_DAYS[key]
    return HOLD_DAYS["DEFAULT"]


# ---------------------------------------------------------------------------
# PRICE / INDEX FETCHERS
# ---------------------------------------------------------------------------

def get_price_on_or_before(symbol: str, target_date: str) -> float | None:
    rows = run_raw_sql(
        """
        SELECT ltp FROM price_history
        WHERE symbol = %s AND date <= %s AND ltp IS NOT NULL
        ORDER BY date DESC LIMIT 5
        """,
        (symbol, target_date)
    )
    if not rows:
        return None
    try:
        return float(rows[0]["ltp"])
    except (ValueError, TypeError):
        return None


def get_nepse_index_on_or_before(target_date: str) -> float | None:
    rows = run_raw_sql(
        """
        SELECT current_value FROM nepse_indices
        WHERE index_id = '58' AND date <= %s AND current_value IS NOT NULL
        ORDER BY date DESC LIMIT 3
        """,
        (target_date,)
    )
    if not rows:
        return None
    try:
        return float(rows[0]["current_value"])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# MACRO SNAPSHOT
# ---------------------------------------------------------------------------

def get_macro_snapshot(eval_date: str) -> dict:
    snapshot = {
        "eval_date":         eval_date,
        "eval_geo_score":    None,
        "eval_nepal_score":  None,
        "eval_nepse_index":  None,
        "eval_market_state": None,
        "eval_policy_rate":  None,
        "eval_fd_rate_pct":  None,
        "eval_key_news":     None,
    }
    try:
        geo_rows = run_raw_sql(
            "SELECT geo_score FROM geopolitical_data "
            "WHERE date <= %s AND geo_score IS NOT NULL "
            "ORDER BY date DESC, id DESC LIMIT 1",
            (eval_date,)
        )
        if geo_rows:
            snapshot["eval_geo_score"] = geo_rows[0]["geo_score"]
    except Exception as e:
        log.warning("geo snapshot failed: %s", e)

    try:
        pulse_rows = run_raw_sql(
            "SELECT nepal_score, key_event FROM nepal_pulse "
            "WHERE date <= %s AND nepal_score IS NOT NULL "
            "ORDER BY date DESC, id DESC LIMIT 1",
            (eval_date,)
        )
        if pulse_rows:
            snapshot["eval_nepal_score"] = pulse_rows[0]["nepal_score"]
            raw   = pulse_rows[0].get("key_event") or ""
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            snapshot["eval_key_news"] = " | ".join(lines[:3]) if lines else raw[:300]
    except Exception as e:
        log.warning("nepal_pulse snapshot failed: %s", e)

    try:
        nepse_val = get_nepse_index_on_or_before(eval_date)
        if nepse_val:
            snapshot["eval_nepse_index"] = str(nepse_val)
    except Exception as e:
        log.warning("nepse index snapshot failed: %s", e)

    try:
        snapshot["eval_market_state"] = get_setting("MARKET_STATE") or "SIDEWAYS"
    except Exception as e:
        log.warning("market state fetch failed: %s", e)

    try:
        nrb_rows = run_raw_sql(
            "SELECT policy_rate FROM nrb_monthly "
            "WHERE policy_rate IS NOT NULL "
            "ORDER BY fiscal_year DESC, month_number DESC LIMIT 1"
        )
        if nrb_rows:
            snapshot["eval_policy_rate"] = nrb_rows[0]["policy_rate"]
    except Exception as e:
        log.warning("nrb policy rate fetch failed: %s", e)

    try:
        snapshot["eval_fd_rate_pct"] = get_setting("FD_RATE_PCT") or "8.5"
    except Exception as e:
        log.warning("fd rate fetch failed: %s", e)

    return snapshot


# ---------------------------------------------------------------------------
# OUTCOME CLASSIFIERS
# ---------------------------------------------------------------------------

def classify_wait_avoid(action: str, price_change_pct: float) -> str:
    """
    AVOID: CORRECT_AVOID | FALSE_AVOID | EXPIRED_AVOID
    WAIT:  MISSED_ENTRY  | CORRECT_WAIT | EXPIRED_WAIT
    """
    if action == "AVOID":
        if price_change_pct <= CORRECT_AVOID_THRESHOLD:
            return "CORRECT_AVOID"
        elif price_change_pct >= FALSE_AVOID_THRESHOLD:
            return "FALSE_AVOID"
        else:
            return "EXPIRED_AVOID"
    elif action == "WAIT":
        if price_change_pct >= MISSED_ENTRY_THRESHOLD:
            return "MISSED_ENTRY"
        elif price_change_pct <= CORRECT_WAIT_THRESHOLD:
            return "CORRECT_WAIT"
        else:
            return "EXPIRED_WAIT"
    return "EXPIRED"


def classify_buy_alpha(price_change_pct: float, nepse_change_pct: float | None) -> str:
    """
    BUY signal quality -- based on alpha vs NEPSE over hold period.
    NOT WIN/LOSS (that is auditor.py's job from portfolio table).
    """
    if nepse_change_pct is None:
        return "UNVERIFIED_BUY"
    alpha = price_change_pct - nepse_change_pct
    if alpha >= BUY_STRONG_ALPHA:
        return "STRONG_BUY_SIGNAL"
    elif alpha <= BUY_WEAK_ALPHA:
        return "WEAK_BUY_SIGNAL"
    else:
        return "NEUTRAL_BUY_SIGNAL"


# ---------------------------------------------------------------------------
# AVOID -- immediate close (called from claude_analyst after write)
# ---------------------------------------------------------------------------

def stamp_avoid_closed(row_id: int, symbol: str, dry_run: bool = False) -> None:
    """
    Call immediately after writing an AVOID row to market_log.
    AVOID = 'we decided not to act'. Nothing to track. Close it now.
    """
    if dry_run:
        log.info("[DRY-RUN] Would close AVOID immediately: %s id=%s", symbol, row_id)
        return
    try:
        updated = update_row(
            "market_log",
            updates={"outcome": "CLOSED", "eval_date": today_str()},
            where={"id": str(row_id)},
        )
        if updated:
            log.info("AVOID %s stamped CLOSED immediately (id=%s)", symbol, row_id)
        else:
            log.warning("stamp_avoid_closed: no row updated for %s id=%s", symbol, row_id)
    except Exception as e:
        log.error("stamp_avoid_closed failed for %s id=%s: %s", symbol, row_id, e)


# ---------------------------------------------------------------------------
# SHARED EVAL BUILDER — avoids duplicating alpha/delta logic across branches
# ---------------------------------------------------------------------------

def _build_eval_update(
    outcome:          str,
    today:            str,
    price_change_pct: float | None,
    nepse_change_pct: float | None,
    eval_alpha:       float | None,
    geo_at_signal:    float | None,
    nepal_at_signal:  float | None,
    macro:            dict,
    price_now:        float | None = None,
) -> dict:
    """
    Build the full update dict for any evaluated signal.
    Centralises all eval field writes — no branch can miss a field.
    """
    eval_geo_score   = _safe_float(macro.get("eval_geo_score"))
    eval_nepal_score = _safe_float(macro.get("eval_nepal_score"))
    eval_geo_delta   = (
        eval_geo_score - geo_at_signal
        if geo_at_signal is not None and eval_geo_score is not None else None
    )
    eval_nepal_delta = (
        eval_nepal_score - nepal_at_signal
        if nepal_at_signal is not None and eval_nepal_score is not None else None
    )

    update = {
        "outcome":               outcome,
        "eval_date":             today,
        "eval_geo_score":        _str(eval_geo_score),
        "eval_nepal_score":      _str(eval_nepal_score),
        "eval_nepse_index":      macro.get("eval_nepse_index"),
        "eval_market_state":     macro.get("eval_market_state"),
        "eval_policy_rate":      macro.get("eval_policy_rate"),
        "eval_fd_rate_pct":      macro.get("eval_fd_rate_pct"),
        "eval_key_news":         macro.get("eval_key_news"),
        "eval_geo_delta":        _str(eval_geo_delta),
        "eval_nepal_delta":      _str(eval_nepal_delta),
        "eval_price_change_pct": _str(round(price_change_pct, 4)) if price_change_pct is not None else None,
        "eval_nepse_change_pct": _str(round(nepse_change_pct, 4)) if nepse_change_pct is not None else None,
        "eval_alpha":            _str(round(eval_alpha, 4))        if eval_alpha        is not None else None,
    }
    if price_now is not None:
        update["exit_price"] = str(round(price_now, 2))
        update["exit_date"]  = today

    return update


# ---------------------------------------------------------------------------
# EVALUATOR: WAIT / AVOID
# ---------------------------------------------------------------------------

def evaluate_wait_avoid(dry_run: bool = False) -> list[dict]:
    """
    1. AVOIDs still PENDING -> force CLOSED (safety net)
    2. WAITs >= WAIT_EXPIRY_DAYS calendar days -> stamp outcome
    3. WAITs at full hold period -> classify outcome
    """
    today     = today_str()
    evaluated = []

    try:
        all_rows = run_raw_sql(
            """
            SELECT * FROM market_log
            WHERE action IN ('WAIT', 'AVOID')
              AND outcome = 'PENDING'
              AND date IS NOT NULL
            ORDER BY date ASC
            """
        )
    except Exception as e:
        log.error("Failed to fetch pending WAIT/AVOID rows: %s", e)
        return []

    if not all_rows:
        log.info("No pending WAIT/AVOID signals.")
        return []

    log.info("Found %d pending WAIT/AVOID signals.", len(all_rows))

    for row in all_rows:
        symbol   = row.get("symbol", "UNKNOWN")
        action   = row.get("action", "WAIT")
        sig_date = row.get("date", "")
        row_id   = row.get("id")

        if not sig_date or not row_id:
            continue

        cal_days = calendar_days_between(sig_date, today)

        # ── AVOID safety net ─────────────────────────────────────────────────
        if action == "AVOID":
            log.info("AVOID %s still PENDING after %d days -- force CLOSED", symbol, cal_days)
            if not dry_run:
                try:
                    update_row(
                        "market_log",
                        updates={"outcome": "CLOSED", "eval_date": today},
                        where={"id": str(row_id)},
                    )
                except Exception as e:
                    log.error("Force-close AVOID failed: %s", e)
            evaluated.append({"symbol": symbol, "action": action,
                               "outcome": "CLOSED", "signal_date": sig_date,
                               "price_change_pct": None, "eval_alpha": None})
            continue

        # ── WAIT: force expiry after WAIT_EXPIRY_DAYS ────────────────────────
        if action == "WAIT" and cal_days >= WAIT_EXPIRY_DAYS:
            log.info("WAIT %s expired after %d calendar days -- stamping", symbol, cal_days)

            price_at_signal  = _safe_float(row.get("entry_price")) or get_price_on_or_before(symbol, sig_date)
            price_now        = get_price_on_or_before(symbol, today)
            price_change_pct = None
            outcome          = "EXPIRED_WAIT"

            if price_at_signal and price_now:
                price_change_pct = ((price_now - price_at_signal) / price_at_signal) * 100
                outcome = classify_wait_avoid("WAIT", price_change_pct)

            nepse_at_signal  = get_nepse_index_on_or_before(sig_date)
            nepse_now        = get_nepse_index_on_or_before(today)
            nepse_change_pct = (
                (nepse_now - nepse_at_signal) / nepse_at_signal * 100
                if nepse_at_signal and nepse_now else None
            )
            eval_alpha = (
                price_change_pct - nepse_change_pct
                if price_change_pct is not None and nepse_change_pct is not None else None
            )

            macro  = get_macro_snapshot(today)
            update = _build_eval_update(
                outcome          = outcome,
                today            = today,
                price_change_pct = price_change_pct,
                nepse_change_pct = nepse_change_pct,
                eval_alpha       = eval_alpha,
                geo_at_signal    = _safe_float(row.get("geo_score")),
                nepal_at_signal  = _safe_float(row.get("macro_score")),
                macro            = macro,
                price_now        = price_now,
            )

            if not dry_run:
                try:
                    updated = update_row("market_log", updates=update, where={"id": str(row_id)})
                    if updated:
                        log.info("WAIT %s stamped %s (expired) alpha=%s",
                                 symbol, outcome,
                                 f"{eval_alpha:+.2f}" if eval_alpha is not None else "N/A")
                    else:
                        log.warning("WAIT expiry stamp: no row updated for %s id=%s", symbol, row_id)
                except Exception as e:
                    log.error("WAIT expiry stamp failed for %s: %s", symbol, e)
            else:
                log.info("[DRY-RUN] Would stamp WAIT %s -> %s", symbol, outcome)

            evaluated.append({
                "symbol":           symbol,
                "action":           action,
                "outcome":          outcome,
                "signal_date":      sig_date,
                "price_change_pct": round(price_change_pct, 2) if price_change_pct is not None else None,
                "nepse_change_pct": round(nepse_change_pct, 2) if nepse_change_pct is not None else None,
                "eval_alpha":       round(eval_alpha, 2)        if eval_alpha        is not None else None,
            })
            continue

        # ── WAIT within expiry window: check hold period ─────────────────────
        hold_period  = resolve_hold_days(row)
        days_elapsed = trading_days_between(sig_date, today)
        if days_elapsed < hold_period:
            log.debug("%s WAIT -- %d/%d trading days. Skipping.", symbol, days_elapsed, hold_period)
            continue

        price_at_signal = _safe_float(row.get("entry_price")) or get_price_on_or_before(symbol, sig_date)
        price_now       = get_price_on_or_before(symbol, today)

        if not price_at_signal or not price_now:
            log.warning("%s -- cannot find prices. Skipping.", symbol)
            continue

        price_change_pct = ((price_now - price_at_signal) / price_at_signal) * 100
        nepse_at_signal  = get_nepse_index_on_or_before(sig_date)
        nepse_now        = get_nepse_index_on_or_before(today)
        nepse_change_pct = (
            (nepse_now - nepse_at_signal) / nepse_at_signal * 100
            if nepse_at_signal and nepse_now else None
        )
        eval_alpha = (
            price_change_pct - nepse_change_pct
            if nepse_change_pct is not None else None
        )
        outcome = classify_wait_avoid(action, price_change_pct)

        log.info("%s %s -> price_chg=%.2f%% alpha=%s outcome=%s",
                 symbol, action, price_change_pct,
                 f"{eval_alpha:+.2f}" if eval_alpha is not None else "N/A", outcome)

        macro  = get_macro_snapshot(today)
        update = _build_eval_update(
            outcome          = outcome,
            today            = today,
            price_change_pct = price_change_pct,
            nepse_change_pct = nepse_change_pct,
            eval_alpha       = eval_alpha,
            geo_at_signal    = _safe_float(row.get("geo_score")),
            nepal_at_signal  = _safe_float(row.get("macro_score")),
            macro            = macro,
            price_now        = price_now,
        )

        evaluated.append({
            "symbol":           symbol,
            "action":           action,
            "outcome":          outcome,
            "signal_date":      sig_date,
            "eval_date":        today,
            "hold_days":        hold_period,
            "days_elapsed":     days_elapsed,
            "price_change_pct": round(price_change_pct, 2),
            "nepse_change_pct": round(nepse_change_pct, 2) if nepse_change_pct is not None else None,
            "eval_alpha":       round(eval_alpha, 2)        if eval_alpha        is not None else None,
        })

        if not dry_run:
            try:
                updated = update_row("market_log", updates=update, where={"id": str(row_id)})
                if updated:
                    log.info("Stamped %s -> %s", symbol, outcome)
                else:
                    log.warning("Stamp: no row updated for %s id=%s", symbol, row_id)
            except Exception as e:
                log.error("Failed to stamp %s id=%s: %s", symbol, row_id, e)
        else:
            log.info("[DRY-RUN] Would stamp %s -> %s", symbol, outcome)

    return evaluated


# ---------------------------------------------------------------------------
# EVALUATOR: BUY (signal quality / alpha tracking)
# ---------------------------------------------------------------------------

def evaluate_buy_signals(dry_run: bool = False) -> list[dict]:
    """
    For BUY rows still PENDING after their hold period:
    Compute price change + alpha vs NEPSE.
    Stamps STRONG/NEUTRAL/WEAK_BUY_SIGNAL for GPT learning.
    Does NOT stamp WIN/LOSS -- that is auditor.py from portfolio table.
    """
    today     = today_str()
    evaluated = []

    try:
        all_rows = run_raw_sql(
            """
            SELECT * FROM market_log
            WHERE action = 'BUY'
              AND outcome = 'PENDING'
              AND date IS NOT NULL
            ORDER BY date ASC
            """
        )
    except Exception as e:
        log.error("Failed to fetch pending BUY rows: %s", e)
        return []

    if not all_rows:
        log.info("No pending BUY signals to evaluate.")
        return []

    log.info("Found %d pending BUY signals.", len(all_rows))

    for row in all_rows:
        symbol   = row.get("symbol", "UNKNOWN")
        sig_date = row.get("date", "")
        row_id   = row.get("id")

        if not sig_date or not row_id:
            continue

        hold_period  = resolve_hold_days(row)
        days_elapsed = trading_days_between(sig_date, today)

        if days_elapsed < hold_period:
            log.debug("BUY %s -- %d/%d trading days. Skipping.", symbol, days_elapsed, hold_period)
            continue

        price_at_signal = _safe_float(row.get("entry_price")) or get_price_on_or_before(symbol, sig_date)
        price_now       = get_price_on_or_before(symbol, today)

        if not price_at_signal or not price_now:
            log.warning("BUY %s -- cannot find prices. Skipping.", symbol)
            continue

        price_change_pct = ((price_now - price_at_signal) / price_at_signal) * 100
        nepse_at_signal  = get_nepse_index_on_or_before(sig_date)
        nepse_now        = get_nepse_index_on_or_before(today)
        nepse_change_pct = (
            (nepse_now - nepse_at_signal) / nepse_at_signal * 100
            if nepse_at_signal and nepse_now else None
        )
        eval_alpha = (
            price_change_pct - nepse_change_pct
            if nepse_change_pct is not None else None
        )
        outcome = classify_buy_alpha(price_change_pct, nepse_change_pct)

        log.info("BUY %s -> price_chg=%.2f%% alpha=%s outcome=%s",
                 symbol, price_change_pct,
                 f"{eval_alpha:+.2f}" if eval_alpha is not None else "N/A", outcome)

        macro  = get_macro_snapshot(today)
        update = _build_eval_update(
            outcome          = outcome,
            today            = today,
            price_change_pct = price_change_pct,
            nepse_change_pct = nepse_change_pct,
            eval_alpha       = eval_alpha,
            geo_at_signal    = _safe_float(row.get("geo_score")),
            nepal_at_signal  = _safe_float(row.get("macro_score")),
            macro            = macro,
        )

        evaluated.append({
            "symbol":           symbol,
            "action":           "BUY",
            "outcome":          outcome,
            "signal_date":      sig_date,
            "eval_date":        today,
            "hold_days":        hold_period,
            "days_elapsed":     days_elapsed,
            "price_change_pct": round(price_change_pct, 2),
            "nepse_change_pct": round(nepse_change_pct, 2) if nepse_change_pct is not None else None,
            "eval_alpha":       round(eval_alpha, 2)        if eval_alpha        is not None else None,
        })

        if not dry_run:
            try:
                updated = update_row("market_log", updates=update, where={"id": str(row_id)})
                if updated:
                    log.info("Stamped BUY %s -> %s", symbol, outcome)
                else:
                    log.warning("Stamp BUY: no row updated for %s id=%s", symbol, row_id)
            except Exception as e:
                log.error("Failed to stamp BUY %s id=%s: %s", symbol, row_id, e)
        else:
            log.info("[DRY-RUN] Would stamp BUY %s -> %s", symbol, outcome)

    return evaluated


# ---------------------------------------------------------------------------
# STATUS REPORTER
# ---------------------------------------------------------------------------

def print_status():
    today = today_str()
    try:
        rows = run_raw_sql(
            """
            SELECT symbol, action, date, outcome, confidence,
                   eval_price_change_pct, eval_alpha, eval_key_news,
                   eval_geo_delta, eval_nepal_delta
            FROM market_log
            WHERE action IN ('BUY', 'WAIT', 'AVOID')
            ORDER BY date DESC
            LIMIT 60
            """
        )
    except Exception as e:
        log.error("Status query failed: %s", e)
        return

    pending   = [r for r in rows if r.get("outcome") == "PENDING"]
    evaluated = [r for r in rows if r.get("outcome") not in ("PENDING", None)]

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION TRACKER STATUS -- {today}")
    print(f"{'='*70}")

    print(f"\nPENDING ({len(pending)}):")
    print(f"  {'Symbol':<10} {'Action':<8} {'Date':<12} Conf")
    print(f"  {'-'*50}")
    for r in pending:
        print(f"  {r.get('symbol',''):<10} {r.get('action',''):<8} "
              f"{r.get('date',''):<12} {r.get('confidence','')}")

    print(f"\nEVALUATED ({len(evaluated)}):")
    print(f"  {'Symbol':<10} {'Action':<8} {'Date':<12} {'Outcome':<25} {'Price%':>7} {'Alpha%':>7}")
    print(f"  {'-'*75}")
    for r in evaluated:
        print(
            f"  {r.get('symbol',''):<10} {r.get('action',''):<8} "
            f"{r.get('date',''):<12} {r.get('outcome',''):<25} "
            f"{_fmt(r.get('eval_price_change_pct')):>7} "
            f"{_fmt(r.get('eval_alpha')):>7}"
        )

    if evaluated:
        from collections import Counter
        counts = Counter(r.get("outcome") for r in evaluated)
        print(f"\nOutcome breakdown:")
        for outcome, count in sorted(counts.items()):
            print(f"   {outcome:<30} {count}")
    print()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def _str(val) -> str | None:
    if val is None:
        return None
    return str(val)

def _fmt(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):+.2f}"
    except (ValueError, TypeError):
        return str(val)


# ---------------------------------------------------------------------------
# ENTRY POINTS
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    """Entry point called by eod_workflow.py."""
    log.info("Starting recommendation_tracker -- %s", today_str())
    wa_evaluated  = evaluate_wait_avoid(dry_run=dry_run)
    buy_evaluated = evaluate_buy_signals(dry_run=dry_run)
    total = len(wa_evaluated) + len(buy_evaluated)
    if total:
        log.info("Evaluated %d signals total (%d WAIT/AVOID, %d BUY):",
                 total, len(wa_evaluated), len(buy_evaluated))
        for r in wa_evaluated + buy_evaluated:
            log.info(
                "  %s %s -> %s (price: %s%%, alpha: %s%%)",
                r.get("action","?"), r.get("symbol","?"), r.get("outcome","?"),
                f"{r['price_change_pct']:+.2f}" if r.get("price_change_pct") is not None else "N/A",
                f"{r['eval_alpha']:+.2f}"        if r.get("eval_alpha")       is not None else "N/A",
            )
    else:
        log.info("No signals ready for evaluation today.")
    log.info("recommendation_tracker complete.")


def main():
    parser = argparse.ArgumentParser(description="NEPSE Recommendation Tracker")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write")
    parser.add_argument("--status",  action="store_true", help="Show all signals summary")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    run(dry_run=args.dry_run)

    if args.dry_run:
        log.info("[DRY-RUN] No writes performed.")


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    main()
    