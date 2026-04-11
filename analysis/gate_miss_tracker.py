"""
gate_miss_tracker.py — NEPSE AI Engine
=======================================
Tracks gate_misses outcomes over 30 trading days.
Runs at EOD after auditor.py.

For every stock that filter_engine.py blocked today:
  - Track its price for up to 30 trading days
  - Stamp outcome at day 30:
      FALSE_BLOCK   → price rose > +5%  (filter was too aggressive)
      CORRECT_BLOCK → price fell > -3%  (filter was right)
      NEUTRAL       → move was ambiguous

Results feed into:
  - learning_hub.py GPT Sunday review (gate_proposals section)
  - daily_context_summarizer.py (gate stats per day)
  - Telegram /gate_stats command

Architecture rules:
  - from sheets import ...   ← all DB reads/writes
  - Never raw psycopg2 outside _db()
  - Fail silently — never block EOD pipeline

CLI:
    python -m analysis.gate_miss_tracker            # normal EOD run
    python -m analysis.gate_miss_tracker --status   # summary table
    python -m analysis.gate_miss_tracker --dry-run  # compute, no writes
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sheets import run_raw_sql, upsert_row, update_row

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

MAX_TRACKING_DAYS    = 30    # trading days before stamping outcome
FALSE_BLOCK_THRESHOLD  = 5.0   # price up >5% = filter was wrong
CORRECT_BLOCK_THRESHOLD = -3.0  # price down >3% = filter was right
# Everything in between = NEUTRAL

# Gate categories that are meaningful to track
# (MUTUAL_FUND and NON_EQUITY are structural — never proposal candidates)
TRACKABLE_CATEGORIES = {
    "CONF_SCORE",
    "TECH_SCORE",
    "RSI_OVERBOUGHT",
    "RSI_NO_CONFIRM",
    "HISTORY",
}

# Minimum sample size before a proposal is surfaced
MIN_SAMPLE_FOR_PROPOSAL = 20
# False block rate above this triggers a proposal
PROPOSAL_THRESHOLD = 0.40


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
    Does not account for public holidays — close enough for tracking.
    Mirrors recommendation_tracker.trading_days_between() exactly.
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
# PRICE FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_price_on_or_before(symbol: str, target_date: str) -> float | None:
    """
    Get LTP for symbol on target_date or closest prior trading day.
    Looks back up to 5 days. Mirrors recommendation_tracker pattern.
    """
    rows = run_raw_sql(
        """
        SELECT ltp, close, date FROM price_history
        WHERE symbol = %s
          AND date <= %s
          AND (ltp IS NOT NULL OR close IS NOT NULL)
        ORDER BY date DESC
        LIMIT 5
        """,
        (symbol, target_date)
    )
    if not rows:
        return None
    for row in rows:
        val = row.get("ltp") or row.get("close")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB READERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_unresolved() -> list[dict]:
    """
    Fetch all gate_misses where outcome IS NULL.
    These are blocks that haven't been evaluated yet.
    """
    try:
        rows = run_raw_sql(
            """
            SELECT id, date, symbol, sector, gate_category, gate_reason,
                   price_at_block, market_state, tech_score, conf_score,
                   tracking_days
            FROM gate_misses
            WHERE outcome IS NULL
              AND date IS NOT NULL
            ORDER BY date ASC
            """
        )
        return rows or []
    except Exception as e:
        log.error("_get_unresolved failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def _classify_outcome(return_pct: float) -> str:
    """
    Classify gate miss outcome based on price movement.

    FALSE_BLOCK   → stock rose strongly — filter was too aggressive
    CORRECT_BLOCK → stock fell — filter was right to block it
    NEUTRAL       → ambiguous move — no strong signal either way
    """
    if return_pct >= FALSE_BLOCK_THRESHOLD:
        return "FALSE_BLOCK"
    elif return_pct <= CORRECT_BLOCK_THRESHOLD:
        return "CORRECT_BLOCK"
    else:
        return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# DB WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def _increment_tracking_days(row_id: int, new_days: int, dry_run: bool) -> None:
    """Update tracking_days counter without stamping outcome yet."""
    if dry_run:
        return
    try:
        update_row(
            "gate_misses",
            {"tracking_days": str(new_days)},
            where={"id": row_id},
        )
    except Exception as e:
        log.warning("_increment_tracking_days failed for id=%s: %s", row_id, e)


def _stamp_outcome(
    row_id: int,
    outcome: str,
    return_pct: float,
    tracking_days: int,
    dry_run: bool,
) -> None:
    """Write outcome, return_pct, tracking_days, and stamped_at to gate_misses row."""
    if dry_run:
        log.info("[DRY-RUN] Would stamp id=%s → %s (%.2f%%)", row_id, outcome, return_pct)
        return
    try:
        update_row(
            "gate_misses",
            {
                "outcome":             outcome,
                "outcome_return_pct":  str(round(return_pct, 4)),
                "tracking_days":       str(tracking_days),
                "outcome_stamped_at":  today_str(),
            },
            where={"id": row_id},
        )
        log.info("Stamped id=%s → %s (%.2f%%)", row_id, outcome, return_pct)
    except Exception as e:
        log.error("_stamp_outcome failed for id=%s: %s", row_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# CORE EOD RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_eod(dry_run: bool = False) -> dict:
    """
    Main EOD function called by eod_workflow.py.

    For each unresolved gate_miss:
      1. Count trading days elapsed since block date
      2. If < MAX_TRACKING_DAYS: increment counter, skip stamping
      3. If >= MAX_TRACKING_DAYS:
           - Get price at block date and price today
           - Compute return_pct
           - Stamp outcome

    Returns summary dict for logging.
    """
    today = today_str()

    log.info("=" * 60)
    log.info("gate_miss_tracker.run_eod() — %s (dry_run=%s)", today, dry_run)
    log.info("=" * 60)

    unresolved = _get_unresolved()
    if not unresolved:
        log.info("No unresolved gate_misses — nothing to evaluate.")
        return {"evaluated": 0, "false_blocks": 0, "correct_blocks": 0, "neutral": 0, "skipped": 0}

    log.info("Found %d unresolved gate_misses to check.", len(unresolved))

    summary = {
        "evaluated":      0,
        "false_blocks":   0,
        "correct_blocks": 0,
        "neutral":        0,
        "skipped":        0,
        "no_price":       0,
    }

    for row in unresolved:
        row_id     = row.get("id")
        symbol     = row.get("symbol", "?")
        block_date = row.get("date", "")
        category   = row.get("gate_category", "OTHER")

        if not block_date or not row_id:
            log.warning("Skipping row with missing date or id: %s", row)
            summary["skipped"] += 1
            continue

        # Count trading days elapsed
        days_elapsed = trading_days_between(block_date, today)

        if days_elapsed < MAX_TRACKING_DAYS:
            # Still within tracking window — increment counter
            _increment_tracking_days(row_id, days_elapsed, dry_run)
            log.debug(
                "%s [%s] — %d/%d trading days. Still tracking.",
                symbol, category, days_elapsed, MAX_TRACKING_DAYS
            )
            summary["skipped"] += 1
            continue

        # Tracking period complete — evaluate outcome
        log.info(
            "Evaluating %s [%s] (blocked %s, %d days elapsed)",
            symbol, category, block_date, days_elapsed
        )

        # Price at block date
        price_at_block_raw = row.get("price_at_block")
        price_at_block = None
        if price_at_block_raw:
            try:
                price_at_block = float(price_at_block_raw)
            except (ValueError, TypeError):
                pass

        if not price_at_block:
            price_at_block = _get_price_on_or_before(symbol, block_date)

        if not price_at_block:
            log.warning("%s — cannot find price at block date %s. Skipping.", symbol, block_date)
            summary["no_price"] += 1
            summary["skipped"] += 1
            continue

        # Current price
        price_now = _get_price_on_or_before(symbol, today)
        if not price_now:
            log.warning("%s — cannot find current price. Skipping.", symbol)
            summary["no_price"] += 1
            summary["skipped"] += 1
            continue

        # Compute return
        return_pct = ((price_now - price_at_block) / price_at_block) * 100

        # Classify
        outcome = _classify_outcome(return_pct)

        log.info(
            "%s [%s] → price_at_block=%.2f price_now=%.2f return=%.2f%% outcome=%s",
            symbol, category, price_at_block, price_now, return_pct, outcome
        )

        # Stamp
        _stamp_outcome(row_id, outcome, return_pct, days_elapsed, dry_run)

        summary["evaluated"] += 1
        if outcome == "FALSE_BLOCK":
            summary["false_blocks"] += 1
        elif outcome == "CORRECT_BLOCK":
            summary["correct_blocks"] += 1
        else:
            summary["neutral"] += 1

    log.info(
        "gate_miss_tracker EOD complete — evaluated=%d false=%d correct=%d neutral=%d skipped=%d no_price=%d",
        summary["evaluated"], summary["false_blocks"], summary["correct_blocks"],
        summary["neutral"], summary["skipped"], summary["no_price"],
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY FOR GPT (called by learning_hub.py)
# ─────────────────────────────────────────────────────────────────────────────


def _write_gate_proposals(proposals: list[dict]) -> None:
    """
    Upsert gate_proposals rows.
    Conflict key: (parameter_name, review_week) — overwrites if re-run same week.
    review_week format: "2026-W15"
    """
    if not proposals:
        return
    from sheets import upsert_row
    review_week = datetime.now(NST).strftime("%G-W%V")   # ISO week
    written = 0
    for i, p in enumerate(proposals, start=1):
        try:
            upsert_row(
                "gate_proposals",
                {
                    "review_week":      review_week,
                    "proposal_number":  str(i),
                    "parameter_name":   p["parameter"],
                    "current_value":    str(p["current"]),
                    "proposed_value":   str(p["suggested"]),
                    "evidence":         p["evidence"],
                    "false_block_rate": str(p["false_block_rate"]),
                    "sample_size":      str(p["sample_size"]),
                    "status":           "PENDING",
                    "created_at":       datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S"),
                },
                conflict_columns=["parameter_name", "review_week"],
            )
            written += 1
        except Exception as exc:
            log.warning("_write_gate_proposals: failed for %s — %s", p["parameter"], exc)
    log.info("_write_gate_proposals: wrote %d/%d proposals to gate_proposals", written, len(proposals))


def get_summary_for_gpt(days: int = 90) -> dict:
    """
    Called by learning_hub._load_gate_miss_summary().

    Aggregates gate_misses outcomes over the last `days` calendar days.
    Returns structured dict ready for GPT prompt injection.

    Returns:
    {
      "total_misses": 245,
      "total_stamped": 189,
      "by_category": {
        "CONF_SCORE": {
          "total": 89,
          "false_block": 38,
          "correct_block": 42,
          "neutral": 9,
          "false_block_rate": 0.43,
          "correct_block_rate": 0.47,
        },
        ...
      },
      "by_market_state": {
        "SIDEWAYS": {"false_block_rate": 0.51, "n": 67},
        ...
      },
      "worst_category": "CONF_SCORE",
      "worst_false_block_rate": 0.43,
      "proposal_candidates": [
        {
          "parameter": "MIN_CONF_SCORE",
          "current": 50,
          "suggested": 45,
          "evidence": "43% of CONF_SCORE blocks in SIDEWAYS market were false positives (n=31)"
        }
      ]
    }
    """
    cutoff = (datetime.now(NST) - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        all_rows = run_raw_sql(
            """
            SELECT gate_category, market_state, outcome, outcome_return_pct,
                   date, symbol, gate_reason
            FROM gate_misses
            WHERE date >= %s
            ORDER BY date ASC
            """,
            (cutoff,)
        ) or []
    except Exception as e:
        log.error("get_summary_for_gpt DB query failed: %s", e)
        return {}

    if not all_rows:
        return {
            "total_misses": 0,
            "total_stamped": 0,
            "by_category": {},
            "by_market_state": {},
            "worst_category": None,
            "worst_false_block_rate": 0.0,
            "proposal_candidates": [],
        }

    total_misses  = len(all_rows)
    stamped_rows  = [r for r in all_rows if r.get("outcome")]
    total_stamped = len(stamped_rows)

    # ── By category — all rows for totals, stamped rows for outcome breakdown ─
    by_category: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "stamped": 0, "false_block": 0, "correct_block": 0, "neutral": 0,
    })

    for row in all_rows:
        cat = row.get("gate_category") or "OTHER"
        by_category[cat]["total"] += 1

    for row in stamped_rows:
        cat = row.get("gate_category") or "OTHER"
        by_category[cat]["stamped"] += 1
        outcome = row.get("outcome", "")
        if outcome == "FALSE_BLOCK":
            by_category[cat]["false_block"] += 1
        elif outcome == "CORRECT_BLOCK":
            by_category[cat]["correct_block"] += 1
        else:
            by_category[cat]["neutral"] += 1

    # Compute rates (based on stamped rows only)
    for cat, stats in by_category.items():
        n = stats["stamped"]
        if n > 0:
            stats["false_block_rate"]   = round(stats["false_block"]   / n, 3)
            stats["correct_block_rate"] = round(stats["correct_block"] / n, 3)
        else:
            stats["false_block_rate"]   = 0.0
            stats["correct_block_rate"] = 0.0

    # ── By market state — all rows for totals, stamped rows for outcome rates ─
    by_market_state: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "stamped": 0, "false_block": 0,
    })

    for row in all_rows:
        ms = row.get("market_state") or "UNKNOWN"
        by_market_state[ms]["total"] += 1

    for row in stamped_rows:
        ms = row.get("market_state") or "UNKNOWN"
        by_market_state[ms]["stamped"] += 1
        if row.get("outcome") == "FALSE_BLOCK":
            by_market_state[ms]["false_block"] += 1

    for ms, stats in by_market_state.items():
        n = stats["stamped"]
        stats["false_block_rate"] = round(stats["false_block"] / n, 3) if n > 0 else 0.0
        stats["n"] = stats["total"]

    # ── Worst category ────────────────────────────────────────────────────────
    worst_category       = None
    worst_false_block_rate = 0.0

    for cat, stats in by_category.items():
        if (cat in TRACKABLE_CATEGORIES
                and stats["total"] >= MIN_SAMPLE_FOR_PROPOSAL
                and stats["false_block_rate"] > worst_false_block_rate):
            worst_false_block_rate = stats["false_block_rate"]
            worst_category = cat

    # ── Proposal candidates ───────────────────────────────────────────────────
    proposal_candidates = []

    # Map gate_category → settings parameter name
    CATEGORY_TO_PARAM = {
        "CONF_SCORE":     "MIN_CONF_SCORE",
        "TECH_SCORE":     "TECH_SCORE_THRESHOLDS",
        "RSI_OVERBOUGHT": "RSI_OVERBOUGHT_THRESHOLD",
        "RSI_NO_CONFIRM": "RSI_CONFIRM_REQUIRED",
        "HISTORY":        "MIN_HISTORY_DAYS",
    }

    # Current threshold values (read from settings for context)
    CURRENT_DEFAULTS = {
        "MIN_CONF_SCORE":           50,
        "TECH_SCORE_THRESHOLDS":    "varies by market state",
        "RSI_OVERBOUGHT_THRESHOLD": 75,
        "RSI_CONFIRM_REQUIRED":     "MACD or BB required",
        "MIN_HISTORY_DAYS":         20,
    }

    for cat, stats in by_category.items():
        if (cat not in TRACKABLE_CATEGORIES):
            continue
        if stats["total"] < MIN_SAMPLE_FOR_PROPOSAL:
            continue
        if stats["false_block_rate"] < PROPOSAL_THRESHOLD:
            continue

        param   = CATEGORY_TO_PARAM.get(cat, cat)
        current = CURRENT_DEFAULTS.get(param, "?")
        n       = stats["total"]
        fb_rate = stats["false_block_rate"]
        fb_pct  = round(fb_rate * 100, 1)

        # Generate a conservative suggestion
        suggestion = _suggest_adjustment(cat, current, fb_rate, by_market_state)

        proposal_candidates.append({
            "parameter": param,
            "current":   current,
            "suggested": suggestion,
            "evidence":  (
                f"{fb_pct}% of {cat} blocks were false positives (n={n}, "
                f"last {days} days). Worst market state: "
                f"{_worst_state_for_category(cat, all_rows)}"
            ),
            "false_block_rate": fb_rate,
            "sample_size": n,
        })

    # ── Fix 2: persist proposals to gate_proposals table ─────────────────────
    _write_gate_proposals(proposal_candidates)

    return {
        "total_misses":          total_misses,
        "total_stamped":         total_stamped,
        "by_category":           dict(by_category),
        "by_market_state":       dict(by_market_state),
        "worst_category":        worst_category,
        "worst_false_block_rate": worst_false_block_rate,
        "proposal_candidates":   proposal_candidates,
    }


def _worst_state_for_category(category: str, all_rows: list[dict]) -> str:
    """Find which market_state had the highest false block rate for this category."""
    state_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "false": 0})
    for row in all_rows:
        if row.get("gate_category") != category or not row.get("outcome"):
            continue
        ms = row.get("market_state", "UNKNOWN")
        state_stats[ms]["total"] += 1
        if row.get("outcome") == "FALSE_BLOCK":
            state_stats[ms]["false"] += 1

    worst = "UNKNOWN"
    worst_rate = 0.0
    for ms, s in state_stats.items():
        if s["total"] >= 5:
            rate = s["false"] / s["total"]
            if rate > worst_rate:
                worst_rate = rate
                worst = f"{ms} ({rate*100:.0f}% false, n={s['total']})"
    return worst


def _suggest_adjustment(
    category: str,
    current,
    false_block_rate: float,
    by_market_state: dict,
) -> str:
    """
    Generate a conservative threshold suggestion.
    Max change: ±5 on numeric thresholds.
    Never removes a gate — only loosens.
    """
    if category == "CONF_SCORE":
        try:
            curr = int(current)
            # Loosen by 5 if >40% false blocks, by 3 if 40-50%
            delta = 5 if false_block_rate > 0.50 else 3
            return str(max(30, curr - delta))  # never go below 30
        except (ValueError, TypeError):
            return "45"

    if category == "TECH_SCORE":
        # Suggest market-state-specific loosening
        worst_ms = max(
            by_market_state.items(),
            key=lambda x: x[1].get("false_block_rate", 0)
        )[0] if by_market_state else "SIDEWAYS"
        return f"Lower {worst_ms} threshold by 5 (e.g. 65→60)"

    if category == "RSI_OVERBOUGHT":
        return "80 (raise from 75 — RSI overbought gate firing too early)"

    if category == "RSI_NO_CONFIRM":
        return "Allow MACD histogram > 0 as confirmation (not just BULLISH cross)"

    if category == "HISTORY":
        return "15 (lower from 20 — newer listings being excluded)"

    return f"Loosen slightly (current: {current})"


# ─────────────────────────────────────────────────────────────────────────────
# STATUS REPORTER (CLI)
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    """CLI status table showing gate miss stats by category."""
    today = today_str()

    try:
        # Overall counts
        overall = run_raw_sql(
            """
            SELECT
                COUNT(*) as total,
                COUNT(outcome) as stamped,
                SUM(CASE WHEN outcome = 'FALSE_BLOCK'   THEN 1 ELSE 0 END) as false_blocks,
                SUM(CASE WHEN outcome = 'CORRECT_BLOCK' THEN 1 ELSE 0 END) as correct_blocks,
                SUM(CASE WHEN outcome = 'NEUTRAL'       THEN 1 ELSE 0 END) as neutral,
                MIN(date) as earliest,
                MAX(date) as latest
            FROM gate_misses
            """
        )

        # By category
        by_cat = run_raw_sql(
            """
            SELECT
                gate_category,
                COUNT(*) as total,
                COUNT(outcome) as stamped,
                SUM(CASE WHEN outcome = 'FALSE_BLOCK'   THEN 1 ELSE 0 END) as false_blocks,
                SUM(CASE WHEN outcome = 'CORRECT_BLOCK' THEN 1 ELSE 0 END) as correct_blocks,
                SUM(CASE WHEN outcome = 'NEUTRAL'       THEN 1 ELSE 0 END) as neutral
            FROM gate_misses
            GROUP BY gate_category
            ORDER BY total DESC
            """
        ) or []

        # Pending (unresolved)
        pending = run_raw_sql(
            "SELECT COUNT(*) as cnt FROM gate_misses WHERE outcome IS NULL"
        )
        pending_count = int(pending[0]["cnt"]) if pending else 0

    except Exception as e:
        print(f"\n  Error reading gate_misses: {e}\n")
        return

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GATE MISS TRACKER STATUS — {today}")
    print(sep)

    if overall and overall[0]["total"]:
        o = overall[0]
        total    = int(o["total"]   or 0)
        stamped  = int(o["stamped"] or 0)
        fb       = int(o["false_blocks"]   or 0)
        cb       = int(o["correct_blocks"] or 0)
        nt       = int(o["neutral"] or 0)
        fb_rate  = round(fb / stamped * 100, 1) if stamped > 0 else 0

        print(f"\n  Total tracked:     {total}")
        print(f"  Pending outcome:   {pending_count}")
        print(f"  Stamped:           {stamped}")
        print(f"    FALSE_BLOCK:     {fb} ({fb_rate}%)")
        print(f"    CORRECT_BLOCK:   {cb} ({round(cb/stamped*100,1) if stamped else 0}%)")
        print(f"    NEUTRAL:         {nt}")
        print(f"  Date range:        {o['earliest']} → {o['latest']}")
    else:
        print("\n  No data in gate_misses table yet.")
        print("  Gates will be recorded from next trading day.")

    if by_cat:
        print(f"\n  {'Category':<22} {'Total':>6} {'Stamped':>8} {'FalseBlk':>9} {'CorrectBlk':>11} {'FB%':>5}  {'Proposal?'}")
        print(f"  {'─'*70}")
        for row in by_cat:
            cat      = row.get("gate_category", "?") or "?"
            total    = int(row["total"]   or 0)
            stamped  = int(row["stamped"] or 0)
            fb       = int(row["false_blocks"]   or 0)
            cb       = int(row["correct_blocks"] or 0)
            fb_rate  = round(fb / stamped * 100, 1) if stamped > 0 else 0.0
            proposal = "⚠️  YES" if (
                cat in TRACKABLE_CATEGORIES
                and stamped >= MIN_SAMPLE_FOR_PROPOSAL
                and fb_rate / 100 >= PROPOSAL_THRESHOLD
            ) else ""
            print(
                f"  {cat:<22} {total:>6} {stamped:>8} {fb:>9} {cb:>11} "
                f"{fb_rate:>5.1f}%  {proposal}"
            )

    # Show any pending proposals
    try:
        proposals = run_raw_sql(
            """
            SELECT review_week, proposal_number, parameter_name,
                   current_value, proposed_value, status
            FROM gate_proposals
            WHERE status = 'PENDING'
            ORDER BY review_week DESC, proposal_number
            LIMIT 10
            """
        ) or []
        if proposals:
            print(f"\n  Pending proposals ({len(proposals)}):")
            for p in proposals:
                print(
                    f"    [{p['review_week']}] #{p['proposal_number']} "
                    f"{p['parameter_name']}: {p['current_value']} → {p['proposed_value']}"
                )
    except Exception:
        pass

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Gate Miss Tracker — EOD outcome stamping"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute outcomes but do not write to DB"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show summary table and exit"
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    log.info("Starting gate_miss_tracker — %s", today_str())

    result = run_eod(dry_run=args.dry_run)

    log.info(
        "Done — evaluated=%d false_blocks=%d correct_blocks=%d neutral=%d skipped=%d",
        result["evaluated"], result["false_blocks"],
        result["correct_blocks"], result["neutral"], result["skipped"],
    )

    if args.dry_run:
        log.info("[DRY-RUN] No writes performed.")


if __name__ == "__main__":
    main()