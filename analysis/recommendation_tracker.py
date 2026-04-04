"""
recommendation_tracker.py — NEPSE AI Engine
============================================
Tracks WAIT and AVOID signals from market_log daily.
Stamps outcome after hold period expires.
Captures macro snapshot + key news at evaluation time.
GPT Sunday reviewer reads this alongside trade_journal.

Run modes:
    python -m analysis.recommendation_tracker            # normal daily run
    python -m analysis.recommendation_tracker --status   # show pending signals summary
    python -m analysis.recommendation_tracker --dry-run  # compute outcomes, write nothing

Called by:
    trading.yml  (every 6 min — checks for new pending)
    eod.yml      (3:15 PM NST — final daily evaluation)
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

from sheets import (
    read_tab, upsert_row, run_raw_sql,
    get_setting, get_latest_geo, get_latest_pulse
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Hold periods per signal type — consistent with backtester optimized values
HOLD_DAYS = {
    "MACD":       17,
    "MACD_CROSS": 17,
    "BB":         60,
    "BB_LOWER":   60,
    "BB_LOWER_TOUCH": 60,
    "SMA":        33,
    "SMA_GOLDEN": 33,
    "SMA_CROSS":  33,
    "CANDLE":     17,
    "COMBO":      17,
    "DEFAULT":    17,
}

# Outcome thresholds
CORRECT_AVOID_THRESHOLD  = -2.0   # price fell > 2% → avoid was right
FALSE_AVOID_THRESHOLD    =  5.0   # price rose > 5% → avoid was wrong
MISSED_ENTRY_THRESHOLD   =  8.0   # price rose > 8% on WAIT → missed opportunity
CORRECT_WAIT_THRESHOLD   = -2.0   # price fell > 2% on WAIT → wait was right


# ─────────────────────────────────────────────────────────────────────────────
# DATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def now_nst() -> datetime:
    return datetime.now(NST)


def today_str() -> str:
    return now_nst().strftime("%Y-%m-%d")


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def trading_days_between(start: str, end: str) -> int:
    """
    Count approximate NEPSE trading days (Sun-Thu) between two date strings.
    Does not account for public holidays — close enough for hold period checks.
    """
    s = parse_date(start)
    e = parse_date(end)
    count = 0
    current = s
    while current < e:
        # NEPSE trades Sun(6) Mon(0) Tue(1) Wed(2) Thu(3)
        if current.weekday() in (0, 1, 2, 3, 6):
            count += 1
        current += timedelta(days=1)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TYPE RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def resolve_hold_days(row: dict) -> int:
    """
    Determine hold period from market_log row.
    Reads reasoning field for signal type hints if candle_pattern not set.
    """
    # Try candle_pattern field first (gemini_filter sometimes sets this)
    cp = (row.get("candle_pattern") or "").upper()

    # Try to infer from reasoning text
    reasoning = (row.get("reasoning") or "").upper()

    for key in HOLD_DAYS:
        if key in cp or key in reasoning:
            return HOLD_DAYS[key]

    return HOLD_DAYS["DEFAULT"]


# ─────────────────────────────────────────────────────────────────────────────
# PRICE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def get_price_on_or_before(symbol: str, target_date: str) -> float | None:
    """
    Get LTP for symbol on target_date or closest prior trading day.
    Looks back up to 5 days for a valid price.
    """
    rows = run_raw_sql(
        """
        SELECT ltp, date FROM price_history
        WHERE symbol = %s
          AND date <= %s
          AND ltp IS NOT NULL
        ORDER BY date DESC
        LIMIT 5
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
    """
    Get NEPSE composite index value on or before target_date.
    Uses nepse_indices table — index_name contains 'NEPSE'.
    """
    rows = run_raw_sql(
        """
        SELECT current_value, date FROM nepse_indices
        WHERE index_name ILIKE '%NEPSE%'
          AND date <= %s
          AND current_value IS NOT NULL
        ORDER BY date DESC
        LIMIT 3
        """,
        (target_date,)
    )
    if not rows:
        return None
    try:
        return float(rows[0]["current_value"])
    except (ValueError, TypeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MACRO SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_snapshot(eval_date: str) -> dict:
    """
    Capture macro state at evaluation date.
    Reads: geo_sentiment, nepal_pulse, nrb_monthly (latest), settings.
    Returns flat dict of eval_* fields.
    """
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

    # geo score — latest row on or before eval_date
    try:
        geo_rows = run_raw_sql(
            """
            SELECT geo_score, date FROM geopolitical_data
            WHERE date <= %s AND geo_score IS NOT NULL
            ORDER BY date DESC, id DESC
            LIMIT 1
            """,
            (eval_date,)
        )
        if geo_rows:
            snapshot["eval_geo_score"] = geo_rows[0]["geo_score"]
    except Exception as e:
        log.warning("geo snapshot failed: %s", e)

    # nepal score + key news — latest nepal_pulse on or before eval_date
    try:
        pulse_rows = run_raw_sql(
            """
            SELECT nepal_score, key_event FROM nepal_pulse
            WHERE date <= %s AND nepal_score IS NOT NULL
            ORDER BY date DESC, id DESC
            LIMIT 1
            """,
            (eval_date,)
        )
        if pulse_rows:
            snapshot["eval_nepal_score"] = pulse_rows[0]["nepal_score"]
            raw_news = pulse_rows[0].get("key_event") or ""
            # Trim to 3 lines max for GPT token efficiency
            lines = [ln.strip() for ln in raw_news.split("\n") if ln.strip()]
            snapshot["eval_key_news"] = " | ".join(lines[:3]) if lines else raw_news[:300]
    except Exception as e:
        log.warning("nepal_pulse snapshot failed: %s", e)

    # NEPSE index
    try:
        nepse_val = get_nepse_index_on_or_before(eval_date)
        if nepse_val:
            snapshot["eval_nepse_index"] = str(nepse_val)
    except Exception as e:
        log.warning("nepse index snapshot failed: %s", e)

    # market state from settings
    try:
        snapshot["eval_market_state"] = get_setting("MARKET_STATE") or "SIDEWAYS"
    except Exception as e:
        log.warning("market state fetch failed: %s", e)

    # policy rate — latest nrb_monthly
    try:
        nrb_rows = run_raw_sql(
            """
            SELECT policy_rate FROM nrb_monthly
            WHERE policy_rate IS NOT NULL
            ORDER BY fiscal_year DESC, month_number DESC
            LIMIT 1
            """
        )
        if nrb_rows:
            snapshot["eval_policy_rate"] = nrb_rows[0]["policy_rate"]
    except Exception as e:
        log.warning("nrb policy rate fetch failed: %s", e)

    # FD rate from settings (updated monthly by interest_scraper)
    try:
        snapshot["eval_fd_rate_pct"] = get_setting("FD_RATE_PCT") or "8.5"
    except Exception as e:
        log.warning("fd rate fetch failed: %s", e)

    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_outcome(action: str, price_change_pct: float) -> str:
    """
    Stamp outcome based on action type and price movement.

    AVOID outcomes:
        CORRECT_AVOID   — price fell, avoid was right
        FALSE_AVOID     — price rose strongly, avoid was wrong
        EXPIRED_AVOID   — hold period elapsed, move ambiguous

    WAIT outcomes:
        MISSED_ENTRY    — price rose strongly, should have bought
        CORRECT_WAIT    — price fell, wait protected capital
        EXPIRED_WAIT    — hold period elapsed, move ambiguous
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

    # Fallback
    return "EXPIRED"


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pending(dry_run: bool = False) -> list[dict]:
    """
    Find all PENDING WAIT/AVOID rows in market_log whose hold period has elapsed.
    Compute outcome, macro snapshot, price delta, alpha.
    Write back to market_log via upsert on id.
    Returns list of evaluated rows (for logging/status).
    """
    today = today_str()

    # Fetch all PENDING WAIT/AVOID rows
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
        log.error("Failed to fetch pending rows: %s", e)
        return []

    if not all_rows:
        log.info("No pending WAIT/AVOID signals found.")
        return []

    log.info("Found %d pending WAIT/AVOID signals to check.", len(all_rows))
    evaluated = []

    for row in all_rows:
        symbol    = row.get("symbol", "UNKNOWN")
        action    = row.get("action", "WAIT")
        sig_date  = row.get("date", "")
        row_id    = row.get("id")

        if not sig_date or not row_id:
            log.warning("Skipping row with missing date or id: %s", row)
            continue

        hold_days = resolve_hold_days(row)

        # Count trading days elapsed since signal
        days_elapsed = trading_days_between(sig_date, today)

        if days_elapsed < hold_days:
            log.debug(
                "%s %s — %d/%d trading days elapsed. Skipping.",
                symbol, action, days_elapsed, hold_days
            )
            continue

        log.info(
            "Evaluating %s %s (signal: %s, %d days elapsed, hold: %d)",
            action, symbol, sig_date, days_elapsed, hold_days
        )

        # ── Price at signal date ──────────────────────────────────────────────
        entry_price_raw = row.get("entry_price")
        price_at_signal = None
        if entry_price_raw:
            try:
                price_at_signal = float(entry_price_raw)
            except (ValueError, TypeError):
                pass

        if not price_at_signal:
            price_at_signal = get_price_on_or_before(symbol, sig_date)

        if not price_at_signal:
            log.warning("%s — cannot find price at signal date %s. Skipping.", symbol, sig_date)
            continue

        # ── Price now (eval date) ─────────────────────────────────────────────
        price_now = get_price_on_or_before(symbol, today)
        if not price_now:
            log.warning("%s — cannot find current price. Skipping.", symbol)
            continue

        price_change_pct = ((price_now - price_at_signal) / price_at_signal) * 100

        # ── NEPSE index delta ─────────────────────────────────────────────────
        nepse_at_signal = get_nepse_index_on_or_before(sig_date)
        nepse_now       = get_nepse_index_on_or_before(today)

        nepse_change_pct = None
        eval_alpha       = None

        if nepse_at_signal and nepse_now:
            nepse_change_pct = ((nepse_now - nepse_at_signal) / nepse_at_signal) * 100
            eval_alpha       = price_change_pct - nepse_change_pct

        # ── Macro snapshot at eval date ───────────────────────────────────────
        macro = get_macro_snapshot(today)

        # ── Geo / Nepal deltas ────────────────────────────────────────────────
        geo_at_signal   = _safe_float(row.get("geo_score"))
        nepal_at_signal = _safe_float(row.get("macro_score"))  # stored as macro_score in market_log

        eval_geo_score   = _safe_float(macro.get("eval_geo_score"))
        eval_nepal_score = _safe_float(macro.get("eval_nepal_score"))

        eval_geo_delta   = None
        eval_nepal_delta = None

        if geo_at_signal is not None and eval_geo_score is not None:
            eval_geo_delta = eval_geo_score - geo_at_signal

        if nepal_at_signal is not None and eval_nepal_score is not None:
            eval_nepal_delta = eval_nepal_score - nepal_at_signal

        # ── Classify outcome ──────────────────────────────────────────────────
        outcome = classify_outcome(action, price_change_pct)

        log.info(
            "%s %s → price_chg=%.2f%% | nepse_chg=%s%% | alpha=%s%% | outcome=%s",
            symbol, action,
            price_change_pct,
            f"{nepse_change_pct:.2f}" if nepse_change_pct is not None else "N/A",
            f"{eval_alpha:.2f}" if eval_alpha is not None else "N/A",
            outcome
        )

        # ── Build update payload ──────────────────────────────────────────────
        update = {
            "id":                    row_id,
            "outcome":               outcome,
            "exit_date":             today,
            "exit_price":            str(round(price_now, 2)),

            # eval macro snapshot
            "eval_date":             today,
            "eval_geo_score":        _str(eval_geo_score),
            "eval_nepal_score":      _str(eval_nepal_score),
            "eval_nepse_index":      macro.get("eval_nepse_index"),
            "eval_market_state":     macro.get("eval_market_state"),
            "eval_policy_rate":      macro.get("eval_policy_rate"),
            "eval_fd_rate_pct":      macro.get("eval_fd_rate_pct"),
            "eval_key_news":         macro.get("eval_key_news"),

            # deltas
            "eval_geo_delta":        _str(eval_geo_delta),
            "eval_nepal_delta":      _str(eval_nepal_delta),
            "eval_price_change_pct": _str(round(price_change_pct, 4)),
            "eval_nepse_change_pct": _str(round(nepse_change_pct, 4)) if nepse_change_pct is not None else None,
            "eval_alpha":            _str(round(eval_alpha, 4)) if eval_alpha is not None else None,
        }

        result_row = {
            "symbol":            symbol,
            "action":            action,
            "signal_date":       sig_date,
            "eval_date":         today,
            "hold_days":         hold_days,
            "days_elapsed":      days_elapsed,
            "price_at_signal":   price_at_signal,
            "price_now":         price_now,
            "price_change_pct":  round(price_change_pct, 2),
            "nepse_change_pct":  round(nepse_change_pct, 2) if nepse_change_pct else None,
            "eval_alpha":        round(eval_alpha, 2) if eval_alpha else None,
            "outcome":           outcome,
            "geo_delta":         eval_geo_delta,
            "nepal_delta":       eval_nepal_delta,
            "key_news":          macro.get("eval_key_news"),
        }
        evaluated.append(result_row)

        if not dry_run:
            try:
                upsert_row("market_log", update, conflict_columns=["id"])
                log.info("Stamped %s → %s", symbol, outcome)
            except Exception as e:
                log.error("Failed to write outcome for %s (id=%s): %s", symbol, row_id, e)
        else:
            log.info("[DRY-RUN] Would stamp %s → %s", symbol, outcome)

    return evaluated


# ─────────────────────────────────────────────────────────────────────────────
# STATUS REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def print_status():
    """
    Show summary of all pending + recently evaluated WAIT/AVOID signals.
    """
    today = today_str()

    try:
        rows = run_raw_sql(
            """
            SELECT symbol, action, date, outcome, confidence,
                   eval_price_change_pct, eval_alpha, eval_key_news,
                   eval_geo_delta, eval_nepal_delta
            FROM market_log
            WHERE action IN ('WAIT', 'AVOID')
            ORDER BY date DESC
            LIMIT 50
            """
        )
    except Exception as e:
        log.error("Status query failed: %s", e)
        return

    pending   = [r for r in rows if r.get("outcome") == "PENDING"]
    evaluated = [r for r in rows if r.get("outcome") != "PENDING"]

    print(f"\n{'='*70}")
    print(f"RECOMMENDATION TRACKER STATUS — {today}")
    print(f"{'='*70}")

    print(f"\n📋 PENDING ({len(pending)} signals awaiting evaluation):")
    print(f"  {'Symbol':<10} {'Action':<8} {'Date':<12} {'Confidence'}")
    print(f"  {'-'*50}")
    for r in pending:
        print(f"  {r.get('symbol',''):<10} {r.get('action',''):<8} "
              f"{r.get('date',''):<12} {r.get('confidence','')}")

    print(f"\n✅ EVALUATED ({len(evaluated)} signals):")
    print(f"  {'Symbol':<10} {'Action':<8} {'Date':<12} {'Outcome':<20} "
          f"{'Price%':>7} {'Alpha%':>7} {'GeoDelta':>9} {'NepalDelta':>11}")
    print(f"  {'-'*85}")
    for r in evaluated:
        print(
            f"  {r.get('symbol',''):<10} {r.get('action',''):<8} "
            f"{r.get('date',''):<12} {r.get('outcome',''):<20} "
            f"{_fmt(r.get('eval_price_change_pct')):>7} "
            f"{_fmt(r.get('eval_alpha')):>7} "
            f"{_fmt(r.get('eval_geo_delta')):>9} "
            f"{_fmt(r.get('eval_nepal_delta')):>11}"
        )

    # Outcome breakdown
    if evaluated:
        from collections import Counter
        counts = Counter(r.get("outcome") for r in evaluated)
        print(f"\n📊 Outcome breakdown:")
        for outcome, count in sorted(counts.items()):
            print(f"   {outcome:<25} {count}")

    # Learning signal
    false_avoids  = sum(1 for r in evaluated if r.get("outcome") == "FALSE_AVOID")
    missed_entries = sum(1 for r in evaluated if r.get("outcome") == "MISSED_ENTRY")
    correct        = sum(1 for r in evaluated if r.get("outcome") in ("CORRECT_AVOID", "CORRECT_WAIT"))

    if evaluated:
        accuracy = (correct / len(evaluated)) * 100
        print(f"\n🎯 Accuracy: {correct}/{len(evaluated)} correct ({accuracy:.1f}%)")
        if false_avoids > 0:
            print(f"   ⚠  {false_avoids} FALSE_AVOID(s) — Claude was too conservative")
        if missed_entries > 0:
            print(f"   ⚠  {missed_entries} MISSED_ENTRY(s) — WAIT signals cost real gains")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NEPSE Recommendation Tracker")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute outcomes but do not write to DB")
    parser.add_argument("--status",  action="store_true",
                        help="Show summary of pending and evaluated signals")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    log.info("Starting recommendation_tracker — %s", today_str())

    evaluated = evaluate_pending(dry_run=args.dry_run)

    if evaluated:
        log.info("Evaluated %d signals:", len(evaluated))
        for r in evaluated:
            log.info(
                "  %s %s → %s (price: %+.2f%%, alpha: %s%%)",
                r["action"], r["symbol"], r["outcome"],
                r["price_change_pct"],
                f"{r['eval_alpha']:+.2f}" if r["eval_alpha"] is not None else "N/A"
            )
    else:
        log.info("No signals ready for evaluation today.")

    if args.dry_run:
        log.info("[DRY-RUN] No writes performed.")

    log.info("recommendation_tracker complete.")


if __name__ == "__main__":
    main()