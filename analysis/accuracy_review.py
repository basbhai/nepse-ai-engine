# -*- coding: utf-8 -*-
"""
analysis/accuracy_review.py — NEPSE AI Engine
──────────────────────────────────────────────
Monthly statistical accuracy review using DeepSeek R1.
Answers: what would make the system more statistically accurate?
Not market opinion — pure math on trade outcomes.

TRIGGER:
  First Sunday of every month EXCEPT months 3, 6, 9, 12
  Minimum 30 closed paper trades in trade_journal

Run modes:
    python -m analysis.accuracy_review              # production run
    python -m analysis.accuracy_review --dry-run    # no API, no DB, token estimate
    python -m analysis.accuracy_review --prompt     # print full DeepSeek prompt
    python -m analysis.accuracy_review --status     # show last run + pending proposals
    python -m analysis.accuracy_review --force      # skip first-Sunday and month guards
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from sheets import run_raw_sql, write_row, get_setting

NST = ZoneInfo("Asia/Kathmandu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ACCURACY] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek/deepseek-r1"
MIN_TRADES     = 30

MOCK_DEEPSEEK_RESULT = {
    "trade_count": 0,
    "analysis_period_days": 0,
    "win_rate": 0.0,
    "avg_return_pct": 0.0,
    "avg_loss_pct": 0.0,
    "profit_factor": 0.0,
    "signal_accuracy": {},
    "sector_accuracy": {},
    "market_state_accuracy": {},
    "confidence_calibration": {
        "is_calibrated": False,
        "high_confidence_win_rate": 0.0,
        "medium_confidence_win_rate": 0.0,
        "low_confidence_win_rate": 0.0,
        "finding": "DRY_RUN",
    },
    "false_block_analysis": {
        "total_false_blocks": 0,
        "false_block_rate": 0.0,
        "worst_gate": "DRY_RUN",
        "worst_sector": "DRY_RUN",
    },
    "system_proposals": [],
    "data_gaps": [],
    "statistical_warnings": ["DRY_RUN — no real data analyzed"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_accuracy_review_month() -> bool:
    """True when current NST month is NOT a quarterly council month (3/6/9/12)."""
    return datetime.now(NST).month not in (3, 6, 9, 12)


def _is_first_sunday_of_month() -> bool:
    """True only on the first Sunday of the calendar month (NST)."""
    now = datetime.now(NST)
    return now.weekday() == 6 and now.day <= 7


# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATE GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _check_already_run(run_month: str) -> bool:
    """True if accuracy_review_log already has an entry for this month."""
    try:
        rows = run_raw_sql(
            "SELECT COUNT(*) AS cnt FROM accuracy_review_log WHERE run_month = %s",
            (run_month,),
        )
        return int((rows[0].get("cnt") or 0)) > 0 if rows else False
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — all return [] or {} on exception, never raise
# ═══════════════════════════════════════════════════════════════════════════════

def _load_trade_outcomes() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT symbol, sector, entry_date, exit_date, result, return_pct,
                   primary_signal, confidence_at_entry, exit_reason, loss_cause,
                   market_state_entry, market_state_exit, volume_ratio_entry,
                   geo_score_entry, nepal_score_entry, paper_mode
            FROM trade_journal
            WHERE paper_mode = 'true'
            ORDER BY entry_date DESC
            """,
        ) or []
        log.info("Loaded %d trade_journal rows", len(rows))
        return rows
    except Exception as e:
        log.error("trade_journal load failed: %s", e)
        return []


def _load_gate_misses() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT symbol, sector,
                   gate_reason    AS failure_reason,
                   price_at_block AS price,
                   market_state,
                   outcome,
                   date           AS created_at,
                   outcome_stamped_at
            FROM gate_misses
            WHERE outcome IN ('FALSE_BLOCK', 'CORRECT_BLOCK')
            ORDER BY date DESC
            LIMIT 200
            """,
        ) or []
        log.info("Loaded %d gate_misses rows (stamped outcomes)", len(rows))
        return rows
    except Exception as e:
        log.error("gate_misses load failed: %s", e)
        return []


def _load_market_log_signals() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT symbol, sector, action, confidence, date,
                   eval_outcome, eval_return_pct
            FROM market_log
            WHERE eval_outcome IS NOT NULL AND eval_outcome != ''
            ORDER BY date DESC
            LIMIT 150
            """,
        ) or []
        log.info("Loaded %d market_log evaluated signals", len(rows))
        return rows
    except Exception as e:
        log.error("market_log load failed: %s", e)
        return []


def _load_claude_audit_history() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT review_week, buy_win_rate, avoid_accuracy,
                   overall_accuracy, false_avoid_rate, missed_entry_rate
            FROM claude_audit
            ORDER BY review_week DESC
            LIMIT 12
            """,
        ) or []
        log.info("Loaded %d claude_audit rows", len(rows))
        return rows
    except Exception as e:
        log.error("claude_audit load failed: %s", e)
        return []


def _load_active_lessons() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT id, lesson_type AS lesson, source, confidence_level,
                   source_weight, validation_count
            FROM learning_hub
            WHERE active = 'true'
            ORDER BY source_weight DESC, confidence_level DESC
            LIMIT 30
            """,
        ) or []
        log.info("Loaded %d active learning_hub lessons", len(rows))
        return rows
    except Exception as e:
        log.error("learning_hub load failed: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_deepseek_prompt(
    trade_outcomes: list[dict],
    gate_misses: list[dict],
    market_signals: list[dict],
    audit_history: list[dict],
    lessons: list[dict],
    run_month: str,
) -> str:
    system = (
        "You are a quantitative analyst reviewing paper trading performance for NEPSE. "
        "Statistical analysis only — no market opinions. "
        "Respond ONLY in valid JSON. No markdown fences. No preamble."
    )

    schema_example = json.dumps({
        "trade_count": 0,
        "analysis_period_days": 0,
        "win_rate": 0.0,
        "avg_return_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "signal_accuracy": {
            "<signal_name>": {
                "count": 0, "win_rate": 0.0, "avg_return": 0.0,
                "recommendation": "INCREASE_WEIGHT|DECREASE_WEIGHT|REMOVE|KEEP",
            }
        },
        "sector_accuracy": {
            "<sector_name>": {
                "trade_count": 0, "win_rate": 0.0, "false_block_rate": 0.0,
                "recommendation": "OVERWEIGHT|UNDERWEIGHT|NEUTRAL|INVESTIGATE",
            }
        },
        "market_state_accuracy": {
            "<state>": {
                "trade_count": 0, "win_rate": 0.0,
                "recommendation": "TRADE_NORMALLY|REDUCE_EXPOSURE|AVOID",
            }
        },
        "confidence_calibration": {
            "is_calibrated": True,
            "high_confidence_win_rate": 0.0,
            "medium_confidence_win_rate": 0.0,
            "low_confidence_win_rate": 0.0,
            "finding": "string",
        },
        "false_block_analysis": {
            "total_false_blocks": 0, "false_block_rate": 0.0,
            "worst_gate": "string", "worst_sector": "string",
        },
        "system_proposals": [
            {
                "priority": 1, "component": "string",
                "proposal_type": "string",
                "current_behavior": "string", "proposed_change": "string",
                "data_evidence": "string",
                "requires_new_data": False, "new_data_source": None,
                "confidence": "HIGH|MEDIUM|LOW", "minimum_trades_to_validate": 0,
            }
        ],
        "data_gaps": [
            {
                "missing_field": "string", "why_it_matters": "string",
                "suggested_source": "string",
                "implementation_complexity": "LOW|MEDIUM|HIGH",
            }
        ],
        "statistical_warnings": ["string"],
    }, ensure_ascii=False, indent=2)

    prompt = (
        f"SYSTEM: {system}\n\n"
        f"=== ACCURACY REVIEW REQUEST — {run_month} ===\n\n"
        f"=== TRADE OUTCOMES ({len(trade_outcomes)} rows) ===\n"
        f"{json.dumps(trade_outcomes[:80], ensure_ascii=False)}\n\n"
        f"=== GATE MISSES — STAMPED OUTCOMES ({len(gate_misses)} rows) ===\n"
        f"{json.dumps(gate_misses[:100], ensure_ascii=False)}\n\n"
        f"=== EVALUATED MARKET LOG SIGNALS ({len(market_signals)} rows) ===\n"
        f"{json.dumps(market_signals[:80], ensure_ascii=False)}\n\n"
        f"=== CLAUDE ACCURACY AUDIT HISTORY ({len(audit_history)} rows) ===\n"
        f"{json.dumps(audit_history, ensure_ascii=False)}\n\n"
        f"=== ACTIVE LEARNING LESSONS ({len(lessons)} rows) ===\n"
        f"{json.dumps(lessons, ensure_ascii=False)}\n\n"
        f"=== TASK ===\n"
        f"Compute accuracy statistics across signals, sectors, and market states.\n"
        f"Identify calibration gaps in the confidence scoring.\n"
        f"Analyse gate_misses false block rate by gate category and sector.\n"
        f"Propose specific system improvements backed by the data.\n"
        f"Flag missing data fields that would improve future analysis.\n\n"
        f"Respond ONLY with valid JSON matching this exact schema:\n"
        f"{schema_example}"
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_review_log(run_month: str, result: dict) -> None:
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    try:
        write_row("accuracy_review_log", {
            "run_month":             run_month,
            "trade_count":           str(result.get("trade_count", 0)),
            "signal_accuracy":       json.dumps(result.get("signal_accuracy", {}),   ensure_ascii=False),
            "sector_accuracy":       json.dumps(result.get("sector_accuracy", {}),   ensure_ascii=False),
            "market_state_accuracy": json.dumps(result.get("market_state_accuracy", {}), ensure_ascii=False),
            "confidence_calibration":json.dumps(result.get("confidence_calibration", {}), ensure_ascii=False),
            "false_block_analysis":  json.dumps(result.get("false_block_analysis",  {}), ensure_ascii=False),
            "deepseek_proposals":    json.dumps(result.get("system_proposals",       []), ensure_ascii=False),
            "status":                "PENDING_REVIEW",
            "inserted_at":           now_nst,
        })
        log.info("[DB] accuracy_review_log written — run_month=%s", run_month)
    except Exception as e:
        log.error("_write_review_log failed: %s", e)


def _write_system_proposals(
    run_month: str,
    proposals: list[dict],
    source: str = "accuracy_review",
) -> int:
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    written = 0
    for p in proposals:
        try:
            write_row("system_proposals", {
                "run_month":         run_month,
                "source":            source,
                "component":         str(p.get("component", "")),
                "proposal_type":     str(p.get("proposal_type", "")),
                "current_behavior":  str(p.get("current_behavior", "")),
                "proposed_change":   str(p.get("proposed_change", "")),
                "data_evidence":     str(p.get("data_evidence", "")),
                "requires_new_data": str(p.get("requires_new_data", False)).lower(),
                "new_data_source":   str(p.get("new_data_source") or ""),
                "confidence":        str(p.get("confidence", "LOW")),
                "status":            "PENDING",
                "inserted_at":       now_nst,
            })
            written += 1
        except Exception as e:
            log.error("_write_system_proposals row failed: %s", e)
    log.info("[DB] %d system_proposals written (source=%s)", written, source)
    return written


def _write_data_gaps(run_month: str, gaps: list[dict]) -> int:
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    written = 0
    for g in gaps:
        try:
            write_row("system_proposals", {
                "run_month":         run_month,
                "source":            "accuracy_review",
                "component":         "scraper / schema",
                "proposal_type":     "new_data_source",
                "current_behavior":  "",
                "proposed_change":   str(g.get("suggested_source", "")),
                "data_evidence":     str(g.get("why_it_matters", "")),
                "requires_new_data": "true",
                "new_data_source":   str(g.get("suggested_source", "")),
                "confidence":        "MEDIUM",
                "status":            "PENDING",
                "inserted_at":       now_nst,
            })
            written += 1
        except Exception as e:
            log.error("_write_data_gaps row failed: %s", e)
    log.info("[DB] %d data_gap proposals written", written)
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM NOTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _send_telegram(run_month: str, result: dict, n_proposals: int, n_gaps: int) -> None:
    try:
        from helper.notifier import _send_admin_only
    except ImportError:
        log.warning("Could not import notifier — skipping Telegram")
        return

    trade_count  = result.get("trade_count", 0)
    win_rate     = result.get("win_rate", 0.0) * 100
    profit_factor = result.get("profit_factor", 0.0)
    warnings     = result.get("statistical_warnings", [])

    proposals = result.get("system_proposals", [])
    top_props  = proposals[:2]

    prop_lines = ""
    for i, p in enumerate(top_props, start=1):
        change = str(p.get("proposed_change", ""))[:80]
        comp   = p.get("component", "?")
        prop_lines += f"\n{i}. {comp}: {change}"

    msg = (
        f"📊 *Accuracy Review — {run_month}*\n"
        f"Trades analyzed: {trade_count} | Win rate: {win_rate:.1f}%\n"
        f"Profit factor: {profit_factor:.2f}\n"
        f"\nTop proposals:{prop_lines or ' None'}\n"
        f"\nData gaps found: {n_gaps}\n"
        f"Statistical warnings: {len(warnings)}"
    )
    try:
        _send_admin_only(msg, parse_mode="Markdown")
        log.info("Accuracy review Telegram notification sent")
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS CLI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _print_status() -> None:
    print("\n  Accuracy Review Status\n  " + "─" * 40)
    try:
        rows = run_raw_sql(
            "SELECT run_month, trade_count, status, inserted_at "
            "FROM accuracy_review_log ORDER BY run_month DESC LIMIT 1",
        )
        if rows:
            r = rows[0]
            print(f"  Last run:     {r.get('run_month', '?')}")
            print(f"  Trade count:  {r.get('trade_count', '?')}")
            print(f"  Status:       {r.get('status', '?')}")
            print(f"  Inserted at:  {r.get('inserted_at', '?')}")
        else:
            print("  No accuracy_review_log entries found.")
    except Exception as e:
        print(f"  accuracy_review_log query failed: {e}")

    try:
        rows = run_raw_sql(
            "SELECT COUNT(*) AS cnt FROM system_proposals WHERE status = 'PENDING'",
        )
        cnt = int((rows[0].get("cnt") or 0)) if rows else 0
        print(f"\n  Pending system_proposals: {cnt}")
    except Exception as e:
        print(f"  system_proposals query failed: {e}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    dry_run: bool = False,
    force: bool = False,
    print_prompt: bool = False,
) -> None:
    """
    Monthly accuracy review pipeline.
    dry_run=True  — load data, print token estimate, skip API + DB.
    force=True    — skip first-Sunday and month guards.
    print_prompt  — print full DeepSeek prompt, skip API + DB.
    """
    now       = datetime.now(NST)
    run_month = now.strftime("%Y-%m")

    log.info("=" * 65)
    log.info("NEPSE ACCURACY REVIEW — %s", run_month)
    log.info("=" * 65)

    # ── Month guard ───────────────────────────────────────────────────────────
    if not force and not _is_accuracy_review_month():
        log.info(
            "Accuracy review skipped — quarterly council month (%d). "
            "Use --force to override.",
            now.month,
        )
        return

    # ── Day guard ─────────────────────────────────────────────────────────────
    if not force and not _is_first_sunday_of_month():
        log.info("Not the first Sunday of the month — skipping. Use --force to override.")
        return

    # ── Duplicate guard ───────────────────────────────────────────────────────
    if not dry_run and not print_prompt and _check_already_run(run_month):
        log.warning(
            "accuracy_review_log already has an entry for %s — aborting duplicates.",
            run_month,
        )
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    trade_outcomes  = _load_trade_outcomes()
    gate_misses     = _load_gate_misses()
    market_signals  = _load_market_log_signals()
    audit_history   = _load_claude_audit_history()
    lessons         = _load_active_lessons()

    trade_count = len(trade_outcomes)
    log.info("Paper trades loaded: %d", trade_count)

    # ── Minimum trade guard ───────────────────────────────────────────────────
    if trade_count < MIN_TRADES:
        log.warning(
            "Insufficient data — minimum %d trades required. "
            "First eligible run June 2026.",
            MIN_TRADES,
        )
        print(
            f"Insufficient data — minimum {MIN_TRADES} trades required. "
            "First eligible run June 2026."
        )
        return

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = _build_deepseek_prompt(
        trade_outcomes, gate_misses, market_signals,
        audit_history, lessons, run_month,
    )
    est_tokens = len(prompt) // 4

    if print_prompt:
        print(f"\n{'='*65}")
        print(f"DEEPSEEK R1 PROMPT — {run_month} (~{est_tokens} tokens)")
        print("=" * 65)
        print(prompt)
        return

    if dry_run:
        print(f"\n{'='*65}")
        print(f"ACCURACY REVIEW DRY RUN — {run_month}")
        print(f"Trade count: {trade_count}")
        print(f"[DRY RUN] Would call DeepSeek R1 — ~{est_tokens} tokens")
        print(f"Zero DB writes confirmed.")
        print("=" * 65)
        return

    # ── Call DeepSeek R1 ──────────────────────────────────────────────────────
    from AI.openrouter import _call
    messages = [
        {
            "role": "system",
            "content": (
                "You are a quantitative analyst reviewing paper trading performance for NEPSE. "
                "Statistical analysis only — no market opinions. "
                "Respond ONLY in valid JSON. No markdown fences. No preamble."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    log.info("[accuracy_review] Calling DeepSeek R1 (~%d tokens)...", est_tokens)
    raw = _call(DEEPSEEK_MODEL, messages, max_tokens=2000, temperature=0.1,
                context="accuracy_review")

    if not raw:
        log.error("DeepSeek R1 returned no response — aborting accuracy review")
        return

    # ── Parse JSON ────────────────────────────────────────────────────────────
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        log.error("DeepSeek JSON parse failed: %s | raw[:400]: %s", e, raw[:400])
        return

    proposals = result.get("system_proposals", [])
    gaps      = result.get("data_gaps", [])
    warnings  = result.get("statistical_warnings", [])

    log.info(
        "DeepSeek analysis complete — win_rate=%.1f%% profit_factor=%.2f "
        "proposals=%d gaps=%d warnings=%d",
        result.get("win_rate", 0.0) * 100,
        result.get("profit_factor", 0.0),
        len(proposals), len(gaps), len(warnings),
    )

    # ── DB writes ─────────────────────────────────────────────────────────────
    _write_review_log(run_month, result)
    n_proposals = _write_system_proposals(run_month, proposals, source="accuracy_review")
    n_gaps      = _write_data_gaps(run_month, gaps)

    # ── Telegram ──────────────────────────────────────────────────────────────
    _send_telegram(run_month, result, n_proposals, n_gaps)

    log.info(
        "Accuracy review complete — %d proposals + %d data gap entries written",
        n_proposals, n_gaps,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="NEPSE monthly accuracy review")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Load data, print token estimate, no API, no DB")
    parser.add_argument("--prompt",   action="store_true",
                        help="Print full DeepSeek prompt with real data, no API")
    parser.add_argument("--status",   action="store_true",
                        help="Print last run + pending proposals count")
    parser.add_argument("--force",    action="store_true",
                        help="Skip first-Sunday and month guards")
    args = parser.parse_args()

    if args.status:
        _print_status()
        sys.exit(0)

    run(dry_run=args.dry_run, force=args.force, print_prompt=args.prompt)


if __name__ == "__main__":
    try:
        from log_config import attach_file_handler
        attach_file_handler(__name__)
    except ImportError:
        pass
    main()
