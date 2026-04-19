# -*- coding: utf-8 -*-
"""
analysis/monthly_council.py — NEPSE AI Engine
==============================================
Multi-model deliberation council. Runs once per month (first Sunday).

Pipeline:
  Stage -1 : GPT-4o   — hindsight audit of last month's council vs outcomes
  Stage 0a : GPT-4o   — draft 3-5 agenda items informed by audit
  Stage 0b : Sonnet   — review, approve, reorder agenda → write to DB
  Stage 1-4: Per agenda item: Grok 3 → GPT-4o → Gemini 2.5 Pro → Opus 4 [A]
             Each model reads full data + item + all prior responses.
             Adversarial instruction: engage with one claim from prior model.
             Each outputs JSON: direction, confidence, key_driver, risk_factor
  Stage 5  : Opus 4 [B] — red team (separate call, no shared context with [A])
  Stage 6  : Opus 4 [C] — chairman synthesis (separate call, independent)
  Stage 7  : Telegram notification

Inputs (30-day lookback):
  daily_context_log, trade_journal, gate_misses, nrb_monthly,
  claude_audit, learning_hub (active, limit 20), portfolio (open positions),
  monthly_council_log (last 3 runs for continuity)

Outputs:
  monthly_council_agenda    — approved agenda items
  monthly_council_log       — per-model responses per stage
  monthly_council_checklist — trading checklist + NEPSE confidence score
  learning_hub              — new lessons from Chairman synthesis

Run modes:
    python -m analysis.monthly_council              # production run
    python -m analysis.monthly_council --dry-run    # no API, no DB, token estimates
    python -m analysis.monthly_council --force      # skip first-Sunday guard
    python -m analysis.monthly_council --prompt     # print prompts, no API, no DB
"""

import argparse
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from AI.openrouter import _call
from sheets import run_raw_sql, write_row, upsert_row, get_setting

NST = ZoneInfo("Asia/Kathmandu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Council-exclusive model IDs — never read from env, council use only ───────
COUNCIL_GPT_MODEL        = "openai/gpt-4o"
COUNCIL_GROK_MODEL       = "x-ai/grok-3"
COUNCIL_GEMINI_PRO_MODEL = "google/gemini-2.5-pro"
COUNCIL_OPUS_MODEL       = "anthropic/claude-opus-4-5"
COUNCIL_SONNET_MODEL     = "anthropic/claude-sonnet-4-6"

# ── Token budget (hard cost cap: $2.00/run) ───────────────────────────────────
MAX_DATA_TOKENS       = 2000   # data context payload sent to any single model
MAX_DISCUSSION_TOKENS = 600    # Stage 1-4 per-model discussion responses
MAX_CHAIRMAN_TOKENS   = 1500   # Stage 6 chairman synthesis
MAX_REDTEAM_TOKENS    = 800    # Stage 5 red team
MAX_AGENDA_TOKENS     = 400    # Stage 0a/0b agenda drafting and review

DATA_LOOKBACK_DAYS = 30


# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _is_first_sunday_of_month() -> bool:
    """True only on the first Sunday of the calendar month (NST)."""
    now = datetime.now(NST)
    return now.weekday() == 6 and now.day <= 7


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — all return [] on exception, never raise
# ═══════════════════════════════════════════════════════════════════════════════

def _load_daily_context() -> list[dict]:
    try:
        cutoff = (datetime.now(NST) - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        rows = run_raw_sql(
            """
            SELECT date, market_state,
                   nepal_score_eod AS nepal_score,
                   geo_score_eod   AS geo_score,
                   key_events_summary AS summary
            FROM daily_context_log
            WHERE date >= %s
            ORDER BY date DESC
            LIMIT 30
            """,
            (cutoff,),
        ) or []
        log.info("Loaded %d daily_context_log rows", len(rows))
        return rows
    except Exception as e:
        log.error("daily_context_log load failed: %s", e)
        return []


def _load_trade_journal() -> list[dict]:
    try:
        cutoff = (datetime.now(NST) - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        rows = run_raw_sql(
            """
            SELECT symbol, sector, entry_date, result,
                   return_pct, exit_reason, loss_cause
            FROM trade_journal
            WHERE entry_date >= %s
            ORDER BY entry_date DESC
            LIMIT 30
            """,
            (cutoff,),
        ) or []
        log.info("Loaded %d trade_journal rows", len(rows))
        return rows
    except Exception as e:
        log.error("trade_journal load failed: %s", e)
        return []


def _load_gate_misses() -> list[dict]:
    try:
        cutoff = (datetime.now(NST) - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        rows = run_raw_sql(
            """
            SELECT symbol, sector,
                   gate_reason AS failure_reason,
                   outcome
            FROM gate_misses
            WHERE date >= %s
            ORDER BY date DESC
            LIMIT 30
            """,
            (cutoff,),
        ) or []
        log.info("Loaded %d gate_misses rows", len(rows))
        return rows
    except Exception as e:
        log.error("gate_misses load failed: %s", e)
        return []


def _load_nrb_monthly() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT period, lending_rate_pct, deposit_rate_pct,
                   npl_ratio_pct, m2_growth_yoy_pct, liquidity_status
            FROM nrb_monthly
            ORDER BY id DESC
            LIMIT 3
            """,
        ) or []
        log.info("Loaded %d nrb_monthly rows", len(rows))
        return rows
    except Exception as e:
        log.error("nrb_monthly load failed: %s", e)
        return []


def _load_claude_audit() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT review_week, buy_win_rate, avoid_accuracy,
                   overall_accuracy, audit_summary
            FROM claude_audit
            ORDER BY review_week DESC
            LIMIT 4
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
            SELECT finding, condition, action,
                   confidence_level, source, review_week
            FROM learning_hub
            WHERE active = 'true'
            ORDER BY created_at DESC
            LIMIT 20
            """,
        ) or []
        log.info("Loaded %d active learning_hub lessons", len(rows))
        return rows
    except Exception as e:
        log.error("learning_hub load failed: %s", e)
        return []


def _load_open_positions() -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT symbol, entry_price, current_price,
                   pnl_pct AS unrealized_pnl_pct
            FROM portfolio
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
            """,
        ) or []
        log.info("Loaded %d open portfolio positions", len(rows))
        return rows
    except Exception as e:
        log.error("portfolio load failed: %s", e)
        return []


def _load_prior_councils(run_month: str) -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT DISTINCT run_month
            FROM monthly_council_log
            WHERE run_month != %s
            ORDER BY run_month DESC
            LIMIT 3
            """,
            (run_month,),
        ) or []
        log.info("Loaded %d prior council month keys", len(rows))
        return rows
    except Exception as e:
        log.error("monthly_council_log prior load failed: %s", e)
        return []


def _load_accuracy_review() -> dict:
    """Last accuracy_review_log row. Returns {} if no rows. Never raises."""
    try:
        rows = run_raw_sql(
            """
            SELECT run_month, trade_count, signal_accuracy, sector_accuracy,
                   false_block_analysis, deepseek_proposals, inserted_at
            FROM accuracy_review_log
            ORDER BY run_month DESC
            LIMIT 1
            """,
        )
        result = rows[0] if rows else {}
        log.info("accuracy_review loaded: %d rows", 1 if result else 0)
        return result
    except Exception as e:
        log.error("accuracy_review_log load failed: %s", e)
        log.info("accuracy_review loaded: 0 rows")
        return {}


def _load_pending_proposals() -> list[dict]:
    """Unreviewed system proposals (status=PENDING). Returns [] if none. Never raises."""
    try:
        rows = run_raw_sql(
            """
            SELECT id, component, proposal_type, proposed_change,
                   data_evidence, confidence, source, inserted_at
            FROM system_proposals
            WHERE status = 'PENDING'
            ORDER BY inserted_at DESC
            LIMIT 10
            """,
        ) or []
        log.info("pending_proposals loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("system_proposals load failed: %s", e)
        log.info("pending_proposals loaded: 0 rows")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUDGET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_rows_to_budget(rows: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Trim oldest rows (front of list) until serialized size fits budget."""
    if not rows:
        return rows, 0
    max_chars = max_tokens * 4
    if len(json.dumps(rows, ensure_ascii=False)) <= max_chars:
        return rows, 0
    trimmed = list(rows)
    omitted = 0
    while trimmed and len(json.dumps(trimmed, ensure_ascii=False)) > max_chars:
        trimmed.pop(0)
        omitted += 1
    return trimmed, omitted


def _build_data_context(
    daily_context: list[dict],
    trade_journal: list[dict],
    gate_misses: list[dict],
    nrb: list[dict],
    audit: list[dict],
    lessons: list[dict],
) -> tuple[str, dict]:
    """
    Assemble all loaded data into a compact context string for model prompts.
    Trims oldest rows to stay within MAX_DATA_TOKENS total.
    Returns (context_string, section_counts).
    """
    budget = MAX_DATA_TOKENS // 6

    dc_rows,     dc_omit     = _trim_rows_to_budget(daily_context, budget)
    tj_rows,     tj_omit     = _trim_rows_to_budget(trade_journal, budget)
    gm_rows,     gm_omit     = _trim_rows_to_budget(gate_misses, budget)
    nrb_rows,    nrb_omit    = _trim_rows_to_budget(nrb, budget)
    audit_rows,  audit_omit  = _trim_rows_to_budget(audit, budget)
    lesson_rows, lesson_omit = _trim_rows_to_budget(lessons, budget)

    def _section(title: str, rows: list[dict], omitted: int) -> str:
        note = f" [{omitted} oldest omitted]" if omitted else ""
        lines = [f"=== {title}{note} ==="]
        for r in rows:
            lines.append(json.dumps({k: v for k, v in r.items() if v is not None}, ensure_ascii=False))
        return "\n".join(lines)

    parts = []
    if dc_rows:
        parts.append(_section("DAILY CONTEXT (last 30 days)", dc_rows, dc_omit))
    if tj_rows:
        parts.append(_section("TRADE JOURNAL (last 30 days)", tj_rows, tj_omit))
    if gm_rows:
        parts.append(_section("GATE MISSES (last 30 days)", gm_rows, gm_omit))
    if nrb_rows:
        parts.append(_section("NRB MONTHLY (last 3)", nrb_rows, nrb_omit))
    if audit_rows:
        parts.append(_section("CLAUDE ACCURACY AUDIT (last 4 weeks)", audit_rows, audit_omit))
    if lesson_rows:
        parts.append(_section("ACTIVE LEARNING LESSONS (top 20)", lesson_rows, lesson_omit))

    context = "\n\n".join(parts)
    counts = {
        "daily_context": len(dc_rows),
        "trade_journal": len(tj_rows),
        "gate_misses": len(gm_rows),
        "nrb": len(nrb_rows),
        "audit": len(audit_rows),
        "lessons": len(lesson_rows),
        "est_tokens": len(context) // 4,
    }
    return context, counts


# ═══════════════════════════════════════════════════════════════════════════════
# COUNCIL CALL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def _council_call(
    model: str,
    messages: list,
    max_tokens: int,
    context: str,
    temperature: float = 0.3,
) -> Optional[str]:
    """Wrapper around AI.openrouter._call for council-exclusive model calls."""
    return _call(model, messages, max_tokens, temperature, context)


def _parse_json_safe(raw: str, context: str = "") -> Optional[dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("[%s] JSON parse failed: %s | raw[:300]: %s", context, e, raw[:300])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DUPLICATE GUARD
# ═══════════════════════════════════════════════════════════════════════════════

def _check_already_run(run_month: str) -> bool:
    """True if council for this month already has log entries."""
    try:
        rows = run_raw_sql(
            "SELECT COUNT(*) AS cnt FROM monthly_council_log WHERE run_month = %s",
            (run_month,),
        )
        cnt = int(rows[0].get("cnt", 0)) if rows else 0
        return cnt > 0
    except Exception:
        return False  # fail open — better to re-run than skip


# ═══════════════════════════════════════════════════════════════════════════════
# DISAGREEMENT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def _disagreement_score(confidences: list) -> float:
    """Standard deviation of confidence integers across 4 discussion models."""
    valid = []
    for c in confidences:
        try:
            valid.append(float(c))
        except (TypeError, ValueError):
            pass
    if len(valid) < 2:
        return 0.0
    return statistics.stdev(valid)


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_agenda(run_month: str, items: list[str]) -> None:
    for i, item in enumerate(items, start=1):
        try:
            upsert_row(
                "monthly_council_agenda",
                {
                    "run_month":   run_month,
                    "item_number": str(i),
                    "agenda_item": item,
                    "approved_by": "sonnet",
                },
                conflict_columns=["run_month", "item_number"],
            )
        except Exception as e:
            log.error("Failed to write agenda item %d: %s", i, e)
    log.info("[DB] agenda written — %d items for %s", len(items), run_month)


def _write_log_entry(row_dict: dict) -> None:
    try:
        write_row("monthly_council_log", row_dict)
        log.info("[DB] log entry written — stage=%s", row_dict.get("stage"))
    except Exception as e:
        log.error("Failed to write monthly_council_log entry: %s", e)


def _write_checklist(run_month: str, checklist: dict) -> None:
    try:
        upsert_row(
            "monthly_council_checklist",
            {
                "run_month":        run_month,
                "stop_trigger":     checklist.get("stop_trigger", ""),
                "go_trigger":       checklist.get("go_trigger", ""),
                "noise_items":      json.dumps(checklist.get("noise_items", [])),
                "confidence_score": str(checklist.get("confidence_score", 50)),
            },
            conflict_columns=["run_month"],
        )
        log.info("[DB] checklist written — confidence=%s", checklist.get("confidence_score"))
    except Exception as e:
        log.error("Failed to write monthly_council_checklist: %s", e)


def _write_monthly_override(
    run_month: str,
    confidence_score: int,
    market_state: str,
    checklist: dict,
    dry_run: bool = False,
) -> None:
    """Upsert monthly_override row after Stage 6 completes."""
    buy_blocked = (
        "true"
        if confidence_score <= 20 and market_state in ("BEAR", "CRISIS")
        else "false"
    )
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    if dry_run:
        log.info(
            "[DRY RUN] Would write monthly_override — run_month=%s confidence=%d buy_blocked=%s",
            run_month, confidence_score, buy_blocked,
        )
        return
    try:
        upsert_row(
            "monthly_override",
            {
                "run_month":        run_month,
                "confidence_score": str(confidence_score),
                "buy_blocked":      buy_blocked,
                "market_state":     market_state,
                "stop_trigger":     checklist.get("stop_trigger", ""),
                "go_trigger":       checklist.get("go_trigger", ""),
                "inserted_at":      now_nst,
                "last_read_at":     "",
            },
            conflict_columns=["run_month"],
        )
        log.info(
            "[monthly_override] written — confidence=%d buy_blocked=%s",
            confidence_score, buy_blocked,
        )
    except Exception as e:
        log.error("_write_monthly_override failed: %s", e)


# ── Lesson writer (exact same pattern as learning_hub.py _write_lessons) ──────

_REQUIRED_LESSON_FIELDS = {"lesson_type", "condition", "finding", "action", "confidence_level"}

_VALID_ACTIONS = {
    "MONITOR", "REDUCE_CONFIDENCE_BY_15", "REDUCE_CONFIDENCE_BY_25",
    "REQUIRE_VOLUME_CONFIRM", "REQUIRE_MACRO_STABLE",
    "WAIT_FOR_CONFIRMATION", "TIGHTEN_STOP", "BLOCK_ENTRY",
}


def _validate_lesson(lesson: dict, index: int) -> bool:
    missing = _REQUIRED_LESSON_FIELDS - set(lesson.keys())
    if missing:
        log.warning("Council lesson #%d missing required fields: %s — skipping", index, missing)
        return False
    if lesson.get("confidence_level") not in ("LOW", "MEDIUM", "HIGH"):
        log.warning("Council lesson #%d invalid confidence_level: %s — skipping",
                    index, lesson.get("confidence_level"))
        return False
    if lesson.get("action") not in _VALID_ACTIONS:
        log.warning("Council lesson #%d invalid action: %s — skipping",
                    index, lesson.get("action"))
        return False
    if lesson.get("action") == "BLOCK_ENTRY":
        try:
            tc = int(lesson.get("trade_count", 0) or 0)
            if tc < 25:
                log.warning("Council lesson #%d: BLOCK_ENTRY with %d trades — "
                            "downgrading to REDUCE_CONFIDENCE_BY_25", index, tc)
                lesson["action"] = "REDUCE_CONFIDENCE_BY_25"
                lesson["gpt_reasoning"] = (
                    (lesson.get("gpt_reasoning") or "") +
                    f" [AUTO-DOWNGRADED: BLOCK_ENTRY requires 25+ trades, had {tc}]"
                )
        except (ValueError, TypeError):
            pass
    return True


def _write_lessons(lessons: list[dict], run_month: str, dry_run: bool = False) -> int:
    """
    Insert Chairman lessons into learning_hub.
    source='monthly_council', validation_count=1. Returns written count.
    """
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    written = 0

    for i, lesson in enumerate(lessons):
        if not _validate_lesson(lesson, i):
            continue

        columns = {
            "created_at":           now_nst,
            "lesson_type":          lesson.get("lesson_type"),
            "source":               "monthly_council",
            "symbol":               lesson.get("symbol", "MARKET"),
            "sector":               lesson.get("sector", "ALL"),
            "applies_to":           lesson.get("applies_to", "ALL"),
            "condition":            lesson.get("condition"),
            "finding":              lesson.get("finding"),
            "action":               lesson.get("action"),
            "trade_count":          str(lesson.get("trade_count", "")),
            "win_count":            str(lesson.get("win_count", "")),
            "loss_count":           str(lesson.get("loss_count", "")),
            "win_rate":             str(lesson.get("win_rate", "")),
            "avg_return_pct":       str(lesson.get("avg_return_pct", "")),
            "confidence_level":     lesson.get("confidence_level", "LOW"),
            "loss_cause_primary":   lesson.get("loss_cause_primary"),
            "active":               "true",
            "superseded_by":        None,
            "supersedes_lesson_id": str(lesson["supersedes_lesson_id"])
                                    if lesson.get("supersedes_lesson_id") else None,
            "review_week":          run_month,
            "evidence_window_days": str(DATA_LOOKBACK_DAYS),
            "gpt_reasoning":        lesson.get("gpt_reasoning"),
            "last_validated":       now_nst[:10],
            "validation_count":     "1",
        }

        if dry_run:
            log.info("[DRY RUN] Would write lesson: %s | %s | %s | %s",
                     columns.get("lesson_type"), columns.get("sector"),
                     columns.get("action"), columns.get("confidence_level"))
            written += 1
            continue

        try:
            valid_cols = {k: v for k, v in columns.items() if v is not None}
            col_names  = list(valid_cols.keys())
            col_sql    = ", ".join(f'"{c}"' for c in col_names)
            val_sql    = ", ".join("%s" for _ in col_names)
            values     = [str(v) if v is not None else None for v in valid_cols.values()]

            new_rows = run_raw_sql(
                f'INSERT INTO learning_hub ({col_sql}) VALUES ({val_sql}) RETURNING id',
                tuple(values),
            )
            new_id = new_rows[0]["id"] if new_rows else None
            if new_id:
                written += 1
                old_id = lesson.get("supersedes_lesson_id")
                if old_id:
                    try:
                        upsert_row(
                            "learning_hub",
                            {"id": str(old_id), "superseded_by": str(new_id), "active": "false"},
                            conflict_columns=["id"],
                        )
                    except Exception as e:
                        log.warning("Failed to link superseded_by %s→%s: %s", old_id, new_id, e)
                log.info("[DB] council lesson written id=%s: %s | %s",
                         new_id, columns.get("lesson_type"), columns.get("confidence_level"))
            else:
                log.error("INSERT returned no id for lesson: %s", columns.get("finding"))
        except Exception as e:
            log.error("Failed to write council lesson: %s | error: %s",
                      columns.get("finding"), e)

    return written


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_audit_messages(
    data_context: str,
    prior_councils: list[dict],
    run_month: str,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Moderator conducting a hindsight audit. "
        "NEPSE (Nepal Stock Exchange) trades Thu-Fri in NPR. "
        "You are reviewing how well last month's council direction predictions matched "
        "actual market outcomes. Be specific, factual, and brief."
    )
    prior_str = json.dumps(prior_councils, ensure_ascii=False) if prior_councils else "No prior council data."
    user = f"""HINDSIGHT AUDIT — {run_month}

Prior council run summaries:
{prior_str}

Current market data (last 30 days):
{data_context}

Task:
1. Compare prior council confidence/direction calls vs what actually happened in trades and market.
2. Identify which agenda themes were correct vs wrong.
3. Note any systematic overconfidence or underconfidence.
4. List 2-3 recurring blind spots the council should address this month.

Write a concise audit (4-6 paragraphs). No JSON needed — plain analytical text."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_agenda_draft_messages(
    data_context: str,
    audit_text: str,
    run_month: str,
    accuracy_review: dict = None,
    pending_proposals: list[dict] = None,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Moderator. "
        "Based on the hindsight audit and current market data, draft 3-5 focused agenda items "
        "for this month's council deliberation. "
        "Each item should be a specific, debatable market question — not a generic topic. "
        "Output ONLY valid JSON.\n\n"
        "AGENDA ROTATION RULE: If any agenda item is identical or substantially similar to an "
        "item from last month's council, you must either: (a) provide 30-day evidence that the "
        "situation has materially changed and justify keeping it, or (b) replace it with a new "
        "item. Do not repeat agenda items without justification.\n\n"
        "You have been provided last month's accuracy review findings "
        "and pending system proposals (if any exist).\n\n"
        "Generate TWO tracks of agenda items:\n\n"
        "TRACK A — Market direction (2 items max):\n"
        "  What does NEPSE look like for the next 30 days?\n"
        "  Label each: [TRACK_A]\n\n"
        "TRACK B — System improvement (2 items max):\n"
        "  Based on accuracy_review findings and pending proposals:\n"
        "  What is structurally wrong with the current pipeline?\n"
        "  What data is missing that would improve decisions?\n"
        "  What signal or filter needs changing based on evidence?\n"
        "  Label each: [TRACK_B]\n\n"
        "If no accuracy review exists yet: generate Track B from "
        "trade_journal loss patterns and gate_misses false block rate only.\n"
        "If no data exists for Track B: generate 1 item asking the council "
        "to establish a baseline for system accuracy measurement.\n\n"
        "Format each item as:\n"
        "[TRACK_A|TRACK_B] Item N: <clear one-sentence agenda item>"
    )

    accuracy_block = ""
    if accuracy_review:
        accuracy_block = (
            f"\nLAST ACCURACY REVIEW ({accuracy_review.get('run_month', '?')}):\n"
            + json.dumps(
                {k: v for k, v in accuracy_review.items() if v is not None},
                ensure_ascii=False,
            )
        )
    else:
        accuracy_block = "\nACCURACY REVIEW: No prior accuracy review available."

    proposals_block = ""
    if pending_proposals:
        proposals_block = (
            f"\nPENDING SYSTEM PROPOSALS ({len(pending_proposals)} items):\n"
            + json.dumps(pending_proposals, ensure_ascii=False)
        )
    else:
        proposals_block = "\nPENDING SYSTEM PROPOSALS: None."

    user = f"""AGENDA DRAFTING — {run_month}

Hindsight audit findings:
{audit_text}

Current market data:
{data_context}
{accuracy_block}
{proposals_block}

Output JSON:
{{
  "agenda_items": [
    "[TRACK_A] Item 1: Specific debatable question (max 120 chars)",
    "[TRACK_B] Item 2: System improvement question (max 120 chars)",
    ...
  ],
  "draft_rationale": "1-2 sentences on why these items were selected"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_agenda_review_messages(
    proposed_items: list[str],
    data_context: str,
    run_month: str,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Secretary reviewing the proposed agenda. "
        "Your role is to ensure agenda items are: (1) clearly debatable, (2) relevant to current NEPSE "
        "conditions, (3) ordered by strategic importance, (4) actionable by a retail investor. "
        "Approve, reorder, or lightly rephrase items. Output ONLY valid JSON."
    )
    items_str = json.dumps(proposed_items, ensure_ascii=False)
    user = f"""AGENDA REVIEW — {run_month}

Proposed agenda items:
{items_str}

Current market data context:
{data_context}

Approve and finalize the agenda. You may reorder, merge near-duplicates, or rephrase for clarity.
Keep 3-5 items. Output JSON:
{{
  "approved_agenda": [
    "Final item 1",
    "Final item 2",
    ...
  ],
  "review_notes": "Brief note on any changes made"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_POSITION_ANCHOR_GUARD = (
    "\n\nYou have been provided current open positions. Do not assume these positions are "
    "correct. Evaluate the agenda item as if entering the market fresh with no existing "
    "positions. Your role is objective analysis, not justification of current holdings."
)

_MODEL_PERSONAS = {
    COUNCIL_GROK_MODEL: (
        "You are Grok 3, a contrarian market analyst on the NEPSE Monthly Council. "
        "You tend to challenge conventional wisdom and look for non-obvious drivers.\n\n"
        "CONTRARIAN STALENESS RULE: If your contrarian view is similar to last month's Stage 1 "
        "response, you must first state whether the 30-day evidence strengthened or weakened the "
        "case. Then decide to maintain or revise. Do not simply explain why the market has not "
        "priced it in — that presumes you were correct. Re-evaluate from evidence first."
        + _POSITION_ANCHOR_GUARD
    ),
    COUNCIL_GPT_MODEL: (
        "You are GPT-4o, a systematic quantitative analyst on the NEPSE Monthly Council. "
        "You anchor in data, statistics, and historical patterns."
        + _POSITION_ANCHOR_GUARD
    ),
    COUNCIL_GEMINI_PRO_MODEL: (
        "You are Gemini 2.5 Pro, a macro-focused analyst on the NEPSE Monthly Council. "
        "You prioritize global macro, NRB policy, and sector rotation signals.\n\n"
        "VOLUME DATA CONSTRAINT: Volume data and advance/decline breadth are NOT available in "
        "daily_context_log. Your technical analysis must rely on sector-level patterns from "
        "trade_journal, market_state trends, and nepal_score. When you would normally cite "
        "volume, state explicitly: 'VOLUME DATA UNAVAILABLE — using market_state proxy' and "
        "reduce your confidence score by 10 points.\n\n"
        "CIRCUIT LIMIT EXCEPTION: Do NOT apply the low-volume trap flag to any stock showing "
        "price at upper or lower circuit limit. Circuit-limit moves suppress volume artificially "
        "and must not be treated as low-volume traps."
        + _POSITION_ANCHOR_GUARD
    ),
    COUNCIL_OPUS_MODEL: (
        "You are Claude Opus, a risk-focused senior analyst on the NEPSE Monthly Council. "
        "You stress-test positions, weigh tail risks, and consider second-order effects."
        + _POSITION_ANCHOR_GUARD
    ),
}


def _build_discussion_messages(
    model: str,
    agenda_item: str,
    item_number: int,
    total_items: int,
    data_context: str,
    prior_responses: list[dict],
    open_positions: list[dict] = None,
) -> list[dict]:
    persona = _MODEL_PERSONAS.get(model, "You are a senior NEPSE market analyst." + _POSITION_ANCHOR_GUARD)
    system = (
        f"{persona}\n"
        "Output ONLY valid JSON with keys: direction (Bullish/Bearish/Neutral), "
        "confidence (integer 0-100), key_driver (string ≤120 chars), risk_factor (string ≤120 chars)."
    )

    prior_block = ""
    if prior_responses:
        lines = []
        for pr in prior_responses:
            lines.append(
                f"[{pr['model_label']}] direction={pr['direction']}, "
                f"confidence={pr['confidence']}, key_driver={pr['key_driver']}"
            )
        last = prior_responses[-1]
        adversarial = (
            f"\nADVERSARIAL INSTRUCTION: Engage directly with this specific claim by "
            f"{last['model_label']}: \"{last['key_driver']}\". "
            "Agree, refine, or rebut it with evidence from the data."
        )
        prior_block = "\nPRIOR COUNCIL RESPONSES:\n" + "\n".join(lines) + adversarial

    positions_block = ""
    if open_positions:
        positions_block = (
            "\nCURRENT OPEN POSITIONS (for context — evaluate independently):\n"
            + json.dumps(open_positions, ensure_ascii=False)
        )

    user = f"""MONTHLY COUNCIL DELIBERATION — Agenda item {item_number}/{total_items}

AGENDA ITEM: {agenda_item}

MARKET DATA (last 30 days):
{data_context}
{positions_block}
{prior_block}

Analyse this agenda item and output your structured assessment as JSON."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_redteam_messages(
    transcript: str,
    open_positions: list[dict],
    run_month: str,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Red Team analyst (Opus instance B, independent). "
        "You have NOT participated in the discussion. Your job is adversarial review: "
        "find the two most conflicting viewpoints and assess which poses highest risk to open positions. "
        "Output ONLY valid JSON."
    )
    positions_str = json.dumps(open_positions, ensure_ascii=False) if open_positions else "No open positions."
    user = f"""RED TEAM REVIEW — {run_month}

FULL COUNCIL DISCUSSION TRANSCRIPT:
{transcript}

CURRENT OPEN POSITIONS:
{positions_str}

Identify the two sharpest conflicting viewpoints in the transcript.
Determine which conflict represents the highest risk to the current open positions.

Output JSON:
{{
  "conflict_1": {{
    "model_a": "model label",
    "claim_a": "their specific claim",
    "model_b": "model label",
    "claim_b": "their specific claim"
  }},
  "conflict_2": {{
    "model_a": "model label",
    "claim_a": "their specific claim",
    "model_b": "model label",
    "claim_b": "their specific claim"
  }},
  "highest_risk": {{
    "conflicting_viewpoint": "which conflict matters most",
    "risk_to_positions": "explain specific risk to open positions",
    "recommended_action": "HOLD / REDUCE / HEDGE / MONITOR"
  }}
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_chairman_messages(
    transcript: str,
    agenda_items: list[str],
    run_month: str,
    disagreement_scores: dict,
    open_positions: list[dict] = None,
    pending_proposals: list[dict] = None,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Chairman (Opus instance C, fully independent). "
        "You have NOT participated in the discussion or red team. "
        "Synthesise all evidence into a final council decision. "
        "Output ONLY valid JSON — no markdown, no extra text.\n\n"
        "Lesson schema (for lessons array):\n"
        "  lesson_type: SIGNAL_FILTER|SECTOR_FILTER|MACRO_FILTER|ENTRY_TIMING|"
        "STOP_CALC|PORTFOLIO_RULE|DIVIDEND_PATTERN|CALENDAR_EFFECT|FAILURE_MODE\n"
        "  condition: machine-readable trigger\n"
        "  finding: human-readable observation\n"
        "  action: MONITOR|REDUCE_CONFIDENCE_BY_15|REDUCE_CONFIDENCE_BY_25|"
        "REQUIRE_VOLUME_CONFIRM|REQUIRE_MACRO_STABLE|WAIT_FOR_CONFIRMATION|TIGHTEN_STOP|BLOCK_ENTRY\n"
        "  confidence_level: LOW|MEDIUM|HIGH\n"
        "  BLOCK_ENTRY requires 25+ supporting trades — use REDUCE_CONFIDENCE_BY_25 otherwise.\n\n"
        "UNVALIDATED LESSON CHECK: For every conclusion you draw, check if it rests solely on a "
        "lesson in learning_hub that has not been validated against this month's trade_journal or "
        "gate_misses data. If so, flag it in unvalidated_lesson_warnings.\n\n"
        "POSITION ANCHORING GUARD: After your synthesis, explicitly evaluate each current open "
        "position and state whether the council's findings support holding or exiting, even if "
        "the conclusion contradicts the current portfolio direction.\n\n"
        "After your market synthesis, review all pending system proposals provided. "
        "For each proposal produce a council assessment. "
        "Also generate new findings if the discussion revealed system improvement opportunities "
        "not already in the proposals.\n\n"
        "Add to your JSON output:\n"
        "'system_verdict': {\n"
        "  'proposals_reviewed': [\n"
        "    {\n"
        "      'proposal_id': N,\n"
        "      'component': 'string',\n"
        "      'council_assessment': 'ENDORSE|REJECT|NEEDS_MORE_DATA',\n"
        "      'reasoning': 'string',\n"
        "      'confidence': 'HIGH|MEDIUM|LOW'\n"
        "    }\n"
        "  ],\n"
        "  'new_system_findings': [\n"
        "    {\n"
        "      'component': 'string',\n"
        "      'finding': 'string',\n"
        "      'proposed_change': 'string',\n"
        "      'data_evidence': 'string',\n"
        "      'requires_new_data': bool,\n"
        "      'new_data_source': 'string or null',\n"
        "      'confidence': 'HIGH|MEDIUM|LOW'\n"
        "    }\n"
        "  ],\n"
        "  'accuracy_improvement_priority': [\n"
        "    '1. Fix X — highest impact',\n"
        "    '2. Add Y — medium impact'\n"
        "  ]\n"
        "}"
    )
    ds_str = json.dumps(disagreement_scores, ensure_ascii=False)
    items_str = "\n".join(f"{i+1}. {item}" for i, item in enumerate(agenda_items))
    positions_str = json.dumps(open_positions, ensure_ascii=False) if open_positions else "No open positions."
    proposals_str = (
        json.dumps(pending_proposals, ensure_ascii=False)
        if pending_proposals
        else "No pending proposals."
    )
    user = f"""CHAIRMAN SYNTHESIS — {run_month}

AGENDA ITEMS:
{items_str}

DISAGREEMENT SCORES (std dev of confidences per item):
{ds_str}

CURRENT OPEN POSITIONS:
{positions_str}

PENDING SYSTEM PROPOSALS:
{proposals_str}

FULL COUNCIL DISCUSSION TRANSCRIPT:
{transcript}

Produce the final council synthesis. Output JSON:
{{
  "strategic_narratives": [
    {{
      "agenda_item": "item text",
      "narrative": "2-3 sentence strategic assessment",
      "council_direction": "Bullish|Bearish|Neutral",
      "council_confidence": 0-100
    }}
  ],
  "confidence_score": 0-100,
  "lessons": [
    {{
      "lesson_type": "...",
      "condition": "...",
      "finding": "...",
      "action": "...",
      "confidence_level": "LOW|MEDIUM|HIGH",
      "gpt_reasoning": "why this lesson is warranted by this month's evidence"
    }}
  ],
  "overruled_lessons": [
    {{"lesson_id": 0, "lesson_text": "...", "reason": "why council evidence overrules this lesson"}}
  ],
  "unvalidated_lesson_warnings": [
    "lesson_id N conclusion rests on lesson not validated against this month's data"
  ],
  "trading_checklist": {{
    "stop_trigger": "one specific condition that should stop all new entries",
    "go_trigger": "one specific condition that confirms it is safe to enter",
    "noise_items": ["item to ignore 1", "item to ignore 2", "item to ignore 3"]
  }},
  "position_evaluations": [
    {{"symbol": "...", "holding_supported": true, "reasoning": "..."}}
  ],
  "system_verdict": {{
    "proposals_reviewed": [
      {{
        "proposal_id": 0,
        "component": "string",
        "council_assessment": "ENDORSE|REJECT|NEEDS_MORE_DATA",
        "reasoning": "string",
        "confidence": "HIGH|MEDIUM|LOW"
      }}
    ],
    "new_system_findings": [
      {{
        "component": "string",
        "finding": "string",
        "proposed_change": "string",
        "data_evidence": "string",
        "requires_new_data": false,
        "new_data_source": null,
        "confidence": "HIGH|MEDIUM|LOW"
      }}
    ],
    "accuracy_improvement_priority": [
      "1. Fix X — highest impact",
      "2. Add Y — medium impact"
    ]
  }}
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# QUARTERLY LESSON WEIGHT REVIEW (1h)
# ═══════════════════════════════════════════════════════════════════════════════

COUNCIL_DEEPSEEK_MODEL = "deepseek/deepseek-r1"


def _run_weight_review(dry_run: bool = False) -> Optional[dict]:
    """
    Quarterly lesson weight review.
    Calls DeepSeek R1 to compute per-source hit-rates, then Opus [C] to validate.
    Writes gate_proposal on endorsement.  dry_run=True → zero API calls, zero DB writes.
    """
    # ── Step 0: sample size check ─────────────────────────────────────────────
    try:
        rows = run_raw_sql(
            """
            SELECT COUNT(*) AS cnt FROM learning_hub
            WHERE active = 'true'
              AND source IN ('gpt_weekly', 'monthly_council', 'learning_seeder')
            """,
        )
        sample_size = int((rows[0].get("cnt") or 0)) if rows else 0
    except Exception as e:
        log.error("_run_weight_review: sample size check failed: %s", e)
        sample_size = 0

    if sample_size < 20:
        log.warning(
            "Weight review skipped — insufficient lesson history (%d lessons). Minimum 20 required.",
            sample_size,
        )
        return None

    # ── Load lesson stats grouped by source ──────────────────────────────────
    try:
        lessons_rows = run_raw_sql(
            """
            SELECT id, source, confidence_level, created_at, active,
                   validation_count, source_weight
            FROM learning_hub
            WHERE source IN ('gpt_weekly', 'monthly_council', 'learning_seeder')
            ORDER BY source, created_at
            LIMIT 100
            """,
        ) or []
    except Exception as e:
        log.error("_run_weight_review: lesson load failed: %s", e)
        lessons_rows = []

    # ── Load trade_journal performance ────────────────────────────────────────
    try:
        trade_perf = run_raw_sql(
            """
            SELECT return_pct, result, exit_reason, entry_date
            FROM trade_journal
            ORDER BY entry_date DESC
            LIMIT 100
            """,
        ) or []
    except Exception as e:
        log.error("_run_weight_review: trade_journal load failed: %s", e)
        trade_perf = []

    # ── Load claude_audit last 8 rows ─────────────────────────────────────────
    try:
        audit_rows = run_raw_sql(
            """
            SELECT buy_win_rate, avoid_accuracy, overall_accuracy
            FROM claude_audit
            ORDER BY review_week DESC
            LIMIT 8
            """,
        ) or []
    except Exception as e:
        log.error("_run_weight_review: claude_audit load failed: %s", e)
        audit_rows = []

    # ── Load current weights ──────────────────────────────────────────────────
    w_council = get_setting("LESSON_WEIGHT_COUNCIL",    "1.5")
    w_gpt     = get_setting("LESSON_WEIGHT_GPT_WEEKLY", "1.0")
    w_seeder  = get_setting("LESSON_WEIGHT_SEEDER",     "0.8")

    deepseek_prompt = (
        f"You are a quantitative analyst reviewing lesson source weights for the NEPSE AI Engine.\n\n"
        f"CURRENT WEIGHTS:\n"
        f"- monthly_council: {w_council}\n"
        f"- gpt_weekly: {w_gpt}\n"
        f"- learning_seeder: {w_seeder}\n\n"
        f"LESSON DATA BY SOURCE ({len(lessons_rows)} rows):\n"
        f"{json.dumps(lessons_rows[:50], ensure_ascii=False)}\n\n"
        f"TRADE JOURNAL PERFORMANCE (last 100 trades, {len(trade_perf)} rows):\n"
        f"{json.dumps(trade_perf[:50], ensure_ascii=False)}\n\n"
        f"CLAUDE AUDIT ACCURACY (last 8 weeks):\n"
        f"{json.dumps(audit_rows, ensure_ascii=False)}\n\n"
        f"TASK:\n"
        f"Compute per source: hit_rate, false_rate, avg_return_when_followed.\n"
        f"Compare against current weights. Propose new weights with statistical reasoning.\n"
        f"Confidence: HIGH (>=30 samples), MEDIUM (20-29 samples), LOW (<20 — set proposed "
        f"weights to current values and explain why sample is insufficient).\n"
        f"Return ONLY valid JSON, no markdown fences:\n"
        f'{{"proposed_weight_council": float, "proposed_weight_gpt_weekly": 1.0, '
        f'"proposed_weight_seeder": float, "sample_size": {sample_size}, '
        f'"council_hit_rate": float, "gpt_hit_rate": float, "seeder_hit_rate": float, '
        f'"reasoning": "string", "confidence": "HIGH|MEDIUM|LOW"}}'
    )
    est_tokens = len(deepseek_prompt) // 4

    if dry_run:
        print(f"\n{'='*65}")
        print(f"WEIGHT REVIEW — sample_size={sample_size} (≥20 → proceeding)")
        print(f"DeepSeek R1 prompt (~{est_tokens} tokens input):")
        print(deepseek_prompt[:500] + "...\n")
        print(f"[DRY RUN] Would call DeepSeek R1 for weight stats — ~{est_tokens} tokens input")
        print(f"[DRY RUN] Would call Opus [C] for weight validation")
        return None

    # ── Step 1: DeepSeek R1 computes stats ───────────────────────────────────
    from AI.openrouter import ask_deepseek
    deepseek_result = ask_deepseek(
        deepseek_prompt, max_tokens=800, context="council_weight_review",
    )

    if not deepseek_result:
        log.error("_run_weight_review: DeepSeek returned no result — aborting")
        return None

    if deepseek_result.get("confidence") == "LOW":
        log.warning(
            "Weight review aborted by DeepSeek — confidence=LOW. Reasoning: %s",
            deepseek_result.get("reasoning", "none"),
        )
        return None

    # ── Step 2: Opus [C] validates ────────────────────────────────────────────
    opus_msgs = [
        {
            "role": "system",
            "content": (
                "You are the NEPSE Monthly Council Chairman validating a lesson weight review. "
                "Check: is sample size sufficient, do proposed weights fit current market regime, "
                "any conflict with recent council findings? "
                "Return ONLY valid JSON, no markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"WEIGHT REVIEW VALIDATION\n\n"
                f"DeepSeek analysis:\n{json.dumps(deepseek_result, ensure_ascii=False)}\n\n"
                f"Council run month: {datetime.now(NST).strftime('%Y-%m')}\n\n"
                f"Validate and return JSON:\n"
                f'{{"endorsed": true_or_false, "reasoning": "string", '
                f'"final_weight_council": float, "final_weight_seeder": float}}'
            ),
        },
    ]
    opus_raw    = _council_call(COUNCIL_OPUS_MODEL, opus_msgs, 500, "council_weight_validation")
    opus_result = _parse_json_safe(opus_raw, "council_weight_validation") or {}

    endorsed = opus_result.get("endorsed", False)
    log.info(
        "[weight_review] Opus endorsed=%s | council_hit=%.2f%% gpt_hit=%.2f%%",
        endorsed,
        deepseek_result.get("council_hit_rate", 0) * 100,
        deepseek_result.get("gpt_hit_rate",     0) * 100,
    )

    # ── Step 3: Write gate_proposal ───────────────────────────────────────────
    final_w_council = opus_result.get(
        "final_weight_council", deepseek_result.get("proposed_weight_council", 1.5)
    )
    final_w_seeder  = opus_result.get(
        "final_weight_seeder", deepseek_result.get("proposed_weight_seeder", 0.8)
    )
    combined_reasoning = (
        f"DeepSeek: {deepseek_result.get('reasoning', '')} | "
        f"Opus: {opus_result.get('reasoning', '')}"
    )[:500]

    proposal_id: Optional[int] = None
    try:
        write_row("gate_proposals", {
            "review_week":      datetime.now(NST).strftime("%Y-%m"),
            "proposal_number":  "1",
            "parameter_name":   "LESSON_WEIGHT_COUNCIL / LESSON_WEIGHT_SEEDER",
            "current_value":    f"council={w_council} seeder={w_seeder}",
            "proposed_value":   f"council={final_w_council} seeder={final_w_seeder}",
            "reasoning":        combined_reasoning,
            "sample_size":      str(deepseek_result.get("sample_size", sample_size)),
            "status":           "PENDING",
        })
        prop_rows = run_raw_sql(
            "SELECT id FROM gate_proposals WHERE parameter_name = %s ORDER BY id DESC LIMIT 1",
            ("LESSON_WEIGHT_COUNCIL / LESSON_WEIGHT_SEEDER",),
        )
        proposal_id = prop_rows[0]["id"] if prop_rows else None
        log.info("[weight_review] gate_proposal written id=%s", proposal_id)
    except Exception as e:
        log.error("_run_weight_review: gate_proposal write failed: %s", e)

    # ── Step 4: Update settings ───────────────────────────────────────────────
    try:
        today_nst = datetime.now(NST).strftime("%Y-%m-%d")
        upsert_row("settings",
                   {"key": "LESSON_WEIGHT_LAST_REVIEW", "value": today_nst},
                   conflict_columns=["key"])
        upsert_row("settings",
                   {"key": "LESSON_WEIGHT_SAMPLE_SIZE",
                    "value": str(deepseek_result.get("sample_size", sample_size))},
                   conflict_columns=["key"])
        log.info("[weight_review] settings LESSON_WEIGHT_LAST_REVIEW + SAMPLE_SIZE updated")
    except Exception as e:
        log.error("_run_weight_review: settings update failed: %s", e)

    # ── Step 5: Telegram notification ─────────────────────────────────────────
    try:
        from helper.notifier import _send_admin_only
        endorsed_str   = "Yes" if endorsed else "No"
        council_hr_pct = deepseek_result.get("council_hit_rate", 0.0) * 100
        gpt_hr_pct     = deepseek_result.get("gpt_hit_rate",     0.0) * 100
        prop_suffix    = f"/approve_{proposal_id} | Reject: /reject_{proposal_id}" if proposal_id else "N/A"
        msg = (
            f"⚖️ *Lesson Weight Review Complete*\n\n"
            f"Council hit rate: {council_hr_pct:.1f}% | GPT hit rate: {gpt_hr_pct:.1f}%\n"
            f"Proposed: council={final_w_council} seeder={final_w_seeder}\n"
            f"Opus endorsed: {endorsed_str}\n"
            f"Approve: /{prop_suffix}"
        )
        _send_admin_only(msg, parse_mode="Markdown")
        log.info("[weight_review] Telegram notification sent")
    except Exception as e:
        log.warning("_run_weight_review: Telegram notification failed: %s", e)

    return {
        "deepseek":    deepseek_result,
        "opus":        opus_result,
        "sample_size": sample_size,
        "endorsed":    endorsed,
    }


def _write_council_system_proposals(
    run_month: str,
    system_verdict: dict,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Write system_verdict proposals to DB after Stage 6.
    Endorsed proposals → UPDATE system_proposals status to COUNCIL_ENDORSED.
    New findings → INSERT into system_proposals with source='monthly_council'.
    Returns (n_endorsed, n_new).
    """
    if not system_verdict:
        return 0, 0

    now_nst     = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    n_endorsed  = 0
    n_new       = 0

    for review in system_verdict.get("proposals_reviewed", []):
        if review.get("council_assessment") != "ENDORSE":
            continue
        proposal_id = review.get("proposal_id")
        if not proposal_id:
            continue
        if dry_run:
            log.info("[DRY RUN] Would endorse proposal id=%s", proposal_id)
            n_endorsed += 1
            continue
        try:
            run_raw_sql(
                "UPDATE system_proposals SET status = 'COUNCIL_ENDORSED' WHERE id = %s",
                (int(proposal_id),),
            )
            n_endorsed += 1
            log.info("[DB] system_proposals id=%s endorsed", proposal_id)
        except Exception as e:
            log.error("_write_council_system_proposals endorse id=%s failed: %s", proposal_id, e)

    for finding in system_verdict.get("new_system_findings", []):
        if dry_run:
            log.info("[DRY RUN] Would write council system finding: %s", finding.get("component"))
            n_new += 1
            continue
        try:
            write_row("system_proposals", {
                "run_month":         run_month,
                "source":            "monthly_council",
                "component":         str(finding.get("component", "")),
                "proposal_type":     "add_signal",
                "current_behavior":  "",
                "proposed_change":   str(finding.get("proposed_change", "")),
                "data_evidence":     str(finding.get("data_evidence", "")),
                "requires_new_data": str(finding.get("requires_new_data", False)).lower(),
                "new_data_source":   str(finding.get("new_data_source") or ""),
                "confidence":        str(finding.get("confidence", "LOW")),
                "status":            "PENDING",
                "inserted_at":       now_nst,
            })
            n_new += 1
            log.info("[DB] council system_finding written — component=%s", finding.get("component"))
        except Exception as e:
            log.error("_write_council_system_proposals new finding failed: %s", e)

    log.info(
        "[system_verdict] %d proposals endorsed, %d new findings written",
        n_endorsed, n_new,
    )
    return n_endorsed, n_new


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM NOTIFICATION (Stage 7)
# ═══════════════════════════════════════════════════════════════════════════════

def _send_council_notification(
    run_month: str,
    confidence_score: int,
    agenda_items: list[str],
    lessons_written: int,
    checklist: dict,
    system_verdict: dict = None,
) -> None:
    try:
        from helper.notifier import _send_admin_only
    except ImportError:
        log.warning("Could not import notifier — skipping Telegram notification")
        return

    items_str = "\n".join(f"  {i+1}. {item[:80]}" for i, item in enumerate(agenda_items))

    system_verdict_block = ""
    if system_verdict:
        sv        = system_verdict
        endorsed  = sum(
            1 for p in sv.get("proposals_reviewed", [])
            if p.get("council_assessment") == "ENDORSE"
        )
        n_new     = len(sv.get("new_system_findings", []))
        priority  = sv.get("accuracy_improvement_priority", [])
        prio_str  = priority[0] if priority else "N/A"
        system_verdict_block = (
            f"\n⚙️ System proposals: *{endorsed} endorsed* | *{n_new} new findings*\n"
            f"Priority: _{prio_str}_"
        )

    msg = (
        f"🏛 *NEPSE MONTHLY COUNCIL — {run_month}*\n\n"
        f"📊 NEPSE Confidence Score: *{confidence_score}/100*\n\n"
        f"📋 Agenda ({len(agenda_items)} items):\n{items_str}\n\n"
        f"✅ GO trigger: _{checklist.get('go_trigger', 'N/A')}_\n"
        f"🛑 STOP trigger: _{checklist.get('stop_trigger', 'N/A')}_\n\n"
        f"📚 Lessons written to learning_hub: *{lessons_written}*"
        f"{system_verdict_block}\n"
        f"_Next council: first Sunday of next month._"
    )
    try:
        _send_admin_only(msg, parse_mode="Markdown")
        log.info("Stage 7: Telegram notification sent")
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _format_transcript(
    agenda_items: list[str],
    discussion_log: list[dict],
) -> str:
    """Format all discussion responses into a readable transcript for Stage 5/6."""
    blocks = []
    for item_idx, item in enumerate(agenda_items, start=1):
        blocks.append(f"\n── AGENDA ITEM {item_idx}: {item} ──")
        for entry in discussion_log:
            if entry.get("agenda_item") == item:
                blocks.append(
                    f"[{entry['model_label']}] "
                    f"direction={entry['direction']}, "
                    f"confidence={entry['confidence']}\n"
                    f"  key_driver: {entry['key_driver']}\n"
                    f"  risk_factor: {entry['risk_factor']}"
                )
    return "\n".join(blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False, force: bool = False, print_prompts: bool = False) -> None:
    """
    Full monthly council pipeline.
    dry_run=True  — load data, build prompts, print token estimates, skip all API/DB calls.
    force=True    — skip first-Sunday guard (for manual runs).
    print_prompts — print all stage prompts without making API calls.
    """
    now = datetime.now(NST)
    run_month = now.strftime("%Y-%m")

    log.info("=" * 65)
    log.info("NEPSE MONTHLY COUNCIL — %s", run_month)
    log.info("=" * 65)

    # ── Trigger guard ─────────────────────────────────────────────────────────
    if not force and not _is_first_sunday_of_month():
        log.info("Not the first Sunday of the month — skipping council run.")
        log.info("Use --force to run manually.")
        return

    # ── Duplicate guard ───────────────────────────────────────────────────────
    if not dry_run and not print_prompts and _check_already_run(run_month):
        log.warning("Council for %s already has log entries — aborting to prevent duplicates.", run_month)
        log.warning("Delete existing rows from monthly_council_log to re-run.")
        return

    # ── Load all data ─────────────────────────────────────────────────────────
    daily_context     = _load_daily_context()
    trade_journal     = _load_trade_journal()
    gate_misses       = _load_gate_misses()
    nrb               = _load_nrb_monthly()
    audit_history     = _load_claude_audit()
    lessons           = _load_active_lessons()
    open_positions    = _load_open_positions()
    prior_councils    = _load_prior_councils(run_month)
    accuracy_review   = _load_accuracy_review()
    pending_proposals = _load_pending_proposals()

    data_context, counts = _build_data_context(
        daily_context, trade_journal, gate_misses,
        nrb, audit_history, lessons,
    )

    log.info(
        "Data loaded — dc=%d tj=%d gm=%d nrb=%d audit=%d lessons=%d est_tokens=%d",
        counts["daily_context"], counts["trade_journal"], counts["gate_misses"],
        counts["nrb"], counts["audit"], counts["lessons"], counts["est_tokens"],
    )

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE -1: GPT-4o hindsight audit
    # ═════════════════════════════════════════════════════════════════════════
    audit_msgs = _build_audit_messages(data_context, prior_councils, run_month)
    est_audit = (sum(len(m["content"]) for m in audit_msgs)) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE -1 (GPT-4o hindsight audit) — ~{est_audit} tokens input")
        if print_prompts:
            print(f"SYSTEM: {audit_msgs[0]['content'][:400]}...")
            print(f"USER: {audit_msgs[1]['content'][:600]}...")
        print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {COUNCIL_GPT_MODEL} "
              f"for stage_audit — ~{est_audit} tokens input, max {MAX_AGENDA_TOKENS} out")
        audit_text = "[DRY RUN audit placeholder]"
    else:
        log.info("[stage_audit] Calling GPT-4o for hindsight audit...")
        audit_text = _council_call(
            COUNCIL_GPT_MODEL, audit_msgs, MAX_AGENDA_TOKENS, "council_audit",
        ) or ""
        if not audit_text:
            log.warning("Stage -1 audit returned empty — continuing with blank audit")
            audit_text = "No audit available."

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0a: GPT-4o agenda drafting
    # ═════════════════════════════════════════════════════════════════════════
    draft_msgs = _build_agenda_draft_messages(
        data_context, audit_text, run_month,
        accuracy_review=accuracy_review, pending_proposals=pending_proposals,
    )
    est_draft = (sum(len(m["content"]) for m in draft_msgs)) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 0a (GPT-4o agenda draft) — ~{est_draft} tokens input")
        if print_prompts:
            print(f"USER: {draft_msgs[1]['content'][:600]}...")
        print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {COUNCIL_GPT_MODEL} "
              f"for stage_0a_draft — ~{est_draft} tokens input, max {MAX_AGENDA_TOKENS} out")
        proposed_items = [
            "Is NEPSE banking sector liquidity improving enough to support re-entry? [DRY RUN]",
            "Should we reduce position sizing given current NRB NPL trends? [DRY RUN]",
            "Is the current market state SIDEWAYS or transitioning to BULL? [DRY RUN]",
        ]
    else:
        log.info("[stage_0a_draft] Calling GPT-4o for agenda drafting...")
        draft_raw = _council_call(
            COUNCIL_GPT_MODEL, draft_msgs, MAX_AGENDA_TOKENS, "council_0a_draft",
        )
        draft_json = _parse_json_safe(draft_raw, "council_0a_draft")
        proposed_items = (
            draft_json.get("agenda_items", []) if draft_json else []
        )
        if not proposed_items:
            log.warning("Stage 0a returned no agenda items — using fallback")
            proposed_items = ["General NEPSE market outlook for next month [FALLBACK]"]

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0b: Sonnet agenda review and approval
    # ═════════════════════════════════════════════════════════════════════════
    review_msgs = _build_agenda_review_messages(proposed_items, data_context, run_month)
    est_review = (sum(len(m["content"]) for m in review_msgs)) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 0b (Sonnet agenda review) — ~{est_review} tokens input")
        if print_prompts:
            print(f"USER: {review_msgs[1]['content'][:600]}...")
        print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {COUNCIL_SONNET_MODEL} "
              f"for stage_0b_review — ~{est_review} tokens input, max {MAX_AGENDA_TOKENS} out")
        approved_items = proposed_items
    else:
        log.info("[stage_0b_review] Calling Sonnet for agenda review...")
        review_raw = _council_call(
            COUNCIL_SONNET_MODEL, review_msgs, MAX_AGENDA_TOKENS, "council_0b_review",
        )
        review_json = _parse_json_safe(review_raw, "council_0b_review")
        approved_items = (
            review_json.get("approved_agenda", proposed_items)
            if review_json else proposed_items
        )
        if not approved_items:
            approved_items = proposed_items
        _write_agenda(run_month, approved_items)

    n_items = len(approved_items)
    log.info("NEPSE MONTHLY COUNCIL — %s — %d agenda items", run_month, n_items)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGES 1-4: Per-agenda-item discussion
    # ═════════════════════════════════════════════════════════════════════════
    discussion_log: list[dict] = []   # all responses, used for transcript
    disagreement_scores: dict  = {}   # agenda_item → disagreement score

    _DISCUSSION_MODELS = [
        (COUNCIL_GROK_MODEL,       "grok_3",    "stage_1_grok"),
        (COUNCIL_GPT_MODEL,        "gpt_4o",    "stage_2_gpt"),
        (COUNCIL_GEMINI_PRO_MODEL, "gemini_pro","stage_3_gemini"),
        (COUNCIL_OPUS_MODEL,       "opus_4",    "stage_4_opus_a"),
    ]

    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

    for item_idx, agenda_item in enumerate(approved_items, start=1):
        log.info("── discussion item %d/%d: %s", item_idx, n_items, agenda_item[:60])
        item_responses:  list[dict] = []   # prior responses for this agenda item
        item_confidences: list[int] = []

        for model, model_label, stage_key in _DISCUSSION_MODELS:
            msgs = _build_discussion_messages(
                model, agenda_item, item_idx, n_items,
                data_context, item_responses,
                open_positions=open_positions,
            )
            est_tokens = (sum(len(m["content"]) for m in msgs)) // 4

            if dry_run or print_prompts:
                print(f"\n{'='*65}")
                print(f"[{stage_key}] item {item_idx}/{n_items}: {agenda_item[:60]}")
                if print_prompts:
                    print(f"SYSTEM: {msgs[0]['content'][:300]}...")
                    print(f"USER: {msgs[1]['content'][:400]}...")
                print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {model} "
                      f"for {stage_key} — ~{est_tokens} tokens input, max {MAX_DISCUSSION_TOKENS} out")
                direction  = "Neutral"
                confidence = 50
                key_driver = f"[DRY RUN — {model_label}]"
                risk_factor = "[DRY RUN]"
                full_response = json.dumps({
                    "direction": direction, "confidence": confidence,
                    "key_driver": key_driver, "risk_factor": risk_factor,
                })
            else:
                log.info("[%s] item %d/%d: %s", stage_key, item_idx, n_items, agenda_item[:60])
                raw = _council_call(model, msgs, MAX_DISCUSSION_TOKENS, f"council_{stage_key}")
                parsed = _parse_json_safe(raw, f"council_{stage_key}") if raw else None
                if parsed:
                    direction   = str(parsed.get("direction", "Neutral"))
                    confidence  = int(parsed.get("confidence", 50))
                    key_driver  = str(parsed.get("key_driver", ""))
                    risk_factor = str(parsed.get("risk_factor", ""))
                else:
                    direction, confidence, key_driver, risk_factor = "Neutral", 50, "", ""
                full_response = raw or ""

            response_entry = {
                "model_label": model_label,
                "stage":       stage_key,
                "agenda_item": agenda_item,
                "direction":   direction,
                "confidence":  confidence,
                "key_driver":  key_driver,
                "risk_factor": risk_factor,
            }
            item_responses.append(response_entry)
            item_confidences.append(confidence)
            discussion_log.append(response_entry)

            if not dry_run and not print_prompts:
                _write_log_entry({
                    "run_month":     run_month,
                    "stage":         stage_key,
                    "agenda_item":   agenda_item,
                    "model":         model,
                    "direction":     direction,
                    "confidence":    str(confidence),
                    "key_driver":    key_driver,
                    "risk_factor":   risk_factor,
                    "full_response": full_response,
                    "inserted_at":   now_nst,
                })

        # ── Disagreement score for this agenda item ───────────────────────
        ds = _disagreement_score(item_confidences)
        disagreement_scores[agenda_item[:80]] = round(ds, 2)

        if ds > 25:
            log.warning("High uncertainty on agenda item %d (DS=%.1f): %s",
                        item_idx, ds, agenda_item[:60])
        elif ds < 8:
            log.warning("Possible groupthink on agenda item %d (DS=%.1f): %s",
                        item_idx, ds, agenda_item[:60])

        if not dry_run and not print_prompts:
            _write_log_entry({
                "run_month":     run_month,
                "stage":         "disagreement_score",
                "agenda_item":   agenda_item,
                "model":         "computed",
                "confidence":    str(round(ds, 2)),
                "full_response": json.dumps({
                    "confidences": item_confidences,
                    "stdev": round(ds, 2),
                }),
                "inserted_at": now_nst,
            })

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5: Opus [B] red team
    # ═════════════════════════════════════════════════════════════════════════
    transcript = _format_transcript(approved_items, discussion_log)
    redteam_msgs = _build_redteam_messages(transcript, open_positions, run_month)
    est_redteam = (sum(len(m["content"]) for m in redteam_msgs)) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 5 (Opus[B] red team) — ~{est_redteam} tokens input")
        if print_prompts:
            print(f"USER (first 600): {redteam_msgs[1]['content'][:600]}...")
        print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {COUNCIL_OPUS_MODEL} "
              f"for stage_5_redteam — ~{est_redteam} tokens input, max {MAX_REDTEAM_TOKENS} out")
        redteam_result = {"highest_risk": {"recommended_action": "MONITOR"}}
    else:
        log.info("[stage_5_redteam] Calling Opus[B] for red team review...")
        redteam_raw = _council_call(
            COUNCIL_OPUS_MODEL, redteam_msgs, MAX_REDTEAM_TOKENS, "council_5_redteam",
        )
        redteam_result = _parse_json_safe(redteam_raw, "council_5_redteam") or {}
        _write_log_entry({
            "run_month":     run_month,
            "stage":         "stage_5_redteam",
            "agenda_item":   "ALL",
            "model":         COUNCIL_OPUS_MODEL,
            "full_response": redteam_raw or "",
            "inserted_at":   now_nst,
        })

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6: Opus [C] chairman synthesis
    # ═════════════════════════════════════════════════════════════════════════
    chairman_msgs = _build_chairman_messages(
        transcript, approved_items, run_month, disagreement_scores,
        open_positions=open_positions,
        pending_proposals=pending_proposals,
    )
    est_chairman = (sum(len(m["content"]) for m in chairman_msgs)) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 6 (Opus[C] chairman) — ~{est_chairman} tokens input")
        if print_prompts:
            print(f"SYSTEM: {chairman_msgs[0]['content'][:400]}...")
            print(f"USER (first 600): {chairman_msgs[1]['content'][:600]}...")
        print(f"[{'DRY RUN' if dry_run else 'PROMPT'}] Would call {COUNCIL_OPUS_MODEL} "
              f"for stage_6_chairman — ~{est_chairman} tokens input, max {MAX_CHAIRMAN_TOKENS} out")
        confidence_score = 50
        lessons_from_chairman: list[dict] = []
        overruled_lessons: list[dict] = []
        unvalidated_warnings: list[str] = []
        checklist = {"stop_trigger": "DRY_RUN", "go_trigger": "DRY_RUN", "noise_items": []}
        system_verdict: dict = {}
    else:
        log.info("[stage_6_chairman] Calling Opus[C] for chairman synthesis...")
        chairman_raw = _council_call(
            COUNCIL_OPUS_MODEL, chairman_msgs, MAX_CHAIRMAN_TOKENS, "council_6_chairman",
        )
        chairman_json = _parse_json_safe(chairman_raw, "council_6_chairman") or {}

        confidence_score      = int(chairman_json.get("confidence_score", 50))
        lessons_from_chairman = chairman_json.get("lessons", [])
        overruled_lessons     = chairman_json.get("overruled_lessons", [])
        unvalidated_warnings  = chairman_json.get("unvalidated_lesson_warnings", [])
        system_verdict        = chairman_json.get("system_verdict", {})
        checklist_raw = chairman_json.get("trading_checklist", {})
        checklist = {
            "stop_trigger":     checklist_raw.get("stop_trigger", ""),
            "go_trigger":       checklist_raw.get("go_trigger", ""),
            "noise_items":      checklist_raw.get("noise_items", []),
            "confidence_score": confidence_score,
        }

        if overruled_lessons:
            log.warning("[chairman] %d lessons overruled by Chairman", len(overruled_lessons))
        if unvalidated_warnings:
            log.warning("[chairman] %d unvalidated lesson warnings", len(unvalidated_warnings))

        _write_log_entry({
            "run_month":     run_month,
            "stage":         "stage_6_chairman",
            "agenda_item":   "ALL",
            "model":         COUNCIL_OPUS_MODEL,
            "confidence":    str(confidence_score),
            "full_response": chairman_raw or "",
            "inserted_at":   now_nst,
        })
        _write_checklist(run_month, checklist)

    # ── Extract market_state for monthly_override ─────────────────────────────
    market_state_now = (
        daily_context[0].get("market_state", "SIDEWAYS")
        if daily_context else "SIDEWAYS"
    )

    # ── Write lessons from Chairman ────────────────────────────────────────
    lessons_written = _write_lessons(lessons_from_chairman, run_month, dry_run=dry_run)

    # ── Write monthly_override (1g) ────────────────────────────────────────
    _write_monthly_override(
        run_month, confidence_score, market_state_now, checklist, dry_run=dry_run,
    )

    # ── Write council system_verdict proposals (2e) ────────────────────────
    _write_council_system_proposals(run_month, system_verdict, dry_run=dry_run)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 7: Telegram notification
    # ═════════════════════════════════════════════════════════════════════════
    if not dry_run and not print_prompts:
        _send_council_notification(
            run_month, confidence_score, approved_items, lessons_written,
            checklist, system_verdict=system_verdict,
        )

    log.info(
        "Council complete — Confidence: %d — %d lessons inserted",
        confidence_score, lessons_written,
    )

    if dry_run:
        print(f"\n{'='*65}")
        print(f"[DRY RUN SUMMARY] run_month={run_month}")
        print(f"  Data: {counts}")
        print(f"  Agenda items (proposed): {len(proposed_items)}")
        print(f"  Token estimates:")
        print(f"    Stage -1  audit:     ~{est_audit} in, {MAX_AGENDA_TOKENS} out")
        print(f"    Stage 0a  draft:     ~{est_draft} in, {MAX_AGENDA_TOKENS} out")
        print(f"    Stage 0b  review:    ~{est_review} in, {MAX_AGENDA_TOKENS} out")
        print(f"    Stage 1-4 per item:  ~{est_tokens} in, {MAX_DISCUSSION_TOKENS} out × 4 models × {n_items} items")
        print(f"    Stage 5   redteam:   ~{est_redteam} in, {MAX_REDTEAM_TOKENS} out")
        print(f"    Stage 6   chairman:  ~{est_chairman} in, {MAX_CHAIRMAN_TOKENS} out")
        print(f"  No API calls made. No DB writes performed.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="NEPSE Monthly Council")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Load data, print token estimates, no API, no DB")
    parser.add_argument("--force",         action="store_true",
                        help="Skip first-Sunday guard (manual run)")
    parser.add_argument("--prompt",        action="store_true",
                        help="Print all stage prompts, no API, no DB")
    parser.add_argument("--weight-review", action="store_true",
                        help="Run quarterly lesson weight review in dry-run mode (no API, no DB)")
    args = parser.parse_args()

    if args.weight_review:
        result = _run_weight_review(dry_run=True)
        if result is None:
            # Either skipped (< 20 lessons) or aborted — message already logged/printed
            sys.exit(0)
        sys.exit(0)

    run(dry_run=args.dry_run, force=args.force, print_prompts=args.prompt)


if __name__ == "__main__":
    try:
        from log_config import attach_file_handler
        attach_file_handler(__name__)
    except ImportError:
        pass
    main()
