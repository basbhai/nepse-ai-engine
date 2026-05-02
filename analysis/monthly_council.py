# -*- coding: utf-8 -*-
"""
analysis/monthly_council.py — NEPSE AI Engine
==============================================
Multi-model deliberation council. Runs once per month (first Sunday).

Pipeline:
  Stage -1 : GPT-5.4-nano   — hindsight audit of last month's council vs outcomes
  Stage 0a : GPT-5.4-nano   — draft 3-5 agenda items informed by audit
  Stage 0b : Haiku-4.5      — review, approve, reorder agenda → write to DB
  Stage 1-5: Per agenda item (5 models, sequential, adversarial):
               Grok 4.20        [web]  — contrarian/sentiment
               GPT-5.4          [web]  — macro/narrative
               DeepSeek V4 Pro  [no]   — quant/math/technical
               Gemini 3.1 Pro   [web]  — DEVIL'S ADVOCATE (mandatory counter-narrative)
               Sonnet 4.5       [web]  — deep fundamental
             Each model reads full data + all prior responses.
             Adversarial instruction: engage with prior model's claim.
             Gemini (Stage 4) MUST challenge consensus if all prior are same direction.
  Stage 6  : Opus 4.6 [web]  — red team (independent, no shared context)
  Stage 7  : Opus 4.7 [web]  — chairman synthesis (independent, upgraded prompt)
  Stage 8  : Telegram notification

PRE-COUNCIL (Saturday before first Sunday):
  Runs Stage -1 + Stage 0a → sends draft agenda to Telegram for review.
  You can reply /agenda_add <item> to add items, /agenda_ok to approve.
  If no response by Sunday 9 AM NST → auto-approves.

Inputs (30-day lookback):
  daily_context_log, trade_journal, gate_misses, nrb_monthly,
  claude_audit, learning_hub (active, limit 20), portfolio (open positions),
  monthly_council_log (last 3 runs for continuity)

Outputs:
  monthly_council_agenda    — approved agenda items
  monthly_council_log       — per-model responses per stage
  monthly_council_checklist — trading checklist + NEPSE confidence score
  learning_hub              — new lessons from Chairman synthesis
  monthly_override          — buy_blocked / buy_cautious flags

Run modes:
    python -m analysis.monthly_council              # production (first Sunday only)
    python -m analysis.monthly_council --dry-run    # no API, no DB, token estimates
    python -m analysis.monthly_council --force      # skip first-Sunday guard
    python -m analysis.monthly_council --prompt     # print prompts, no API, no DB
    python -m analysis.monthly_council --preview    # run pre-council agenda preview (Saturday)
    python -m analysis.monthly_council --weight-review  # quarterly weight review
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

# ── Council model constants ───────────────────────────────────────────────────
# Pre-council / agenda stages (cheap)
COUNCIL_AUDIT_MODEL    = "openai/gpt-5.4-nano"      # Stage -1, 0a
COUNCIL_REVIEW_MODEL   = "anthropic/claude-haiku-4.5" # Stage 0b

# Discussion models (5 flagship models — different families)
COUNCIL_GROK_MODEL     = "x-ai/grok-4.20"                    # Stage 1 — sentiment/news [web]
COUNCIL_GPT_MODEL      = "openai/gpt-5.4"                    # Stage 2 — macro/narrative [web]
COUNCIL_DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"           # Stage 3 — quant/math [no web]
COUNCIL_GEMINI_MODEL   = "google/gemini-3.1-pro-preview"      # Stage 4 — devil's advocate [web]
COUNCIL_SONNET_MODEL   = "anthropic/claude-sonnet-4.5"        # Stage 5 — deep fundamental [web]

# Post-discussion (flagship)
COUNCIL_REDTEAM_MODEL  = "anthropic/claude-opus-4.6"          # Stage 6 — red team [web]
COUNCIL_CHAIRMAN_MODEL = "anthropic/claude-opus-4.7"          # Stage 7 — chairman [web]

# Weight review (quarterly)
COUNCIL_WEIGHT_DEEPSEEK = "deepseek/deepseek-r1"

# ── Token budget ──────────────────────────────────────────────────────────────
MAX_DATA_TOKENS       = 2000
MAX_DISCUSSION_TOKENS = 800    # increased for flagship models
MAX_CHAIRMAN_TOKENS   = 2000   # increased — upgraded chairman
MAX_REDTEAM_TOKENS    = 1200   # increased — upgraded red team
MAX_AGENDA_TOKENS     = 600

DATA_LOOKBACK_DAYS = 30

# ── Pre-council agenda preview DB key ────────────────────────────────────────
AGENDA_PREVIEW_SETTING = "COUNCIL_AGENDA_PREVIEW"
AGENDA_PREVIEW_OK_SETTING = "COUNCIL_AGENDA_APPROVED"


# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER GUARDS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_first_sunday_of_month() -> bool:
    """True only on the first Sunday of the calendar month (NST)."""
    now = datetime.now(NST)
    return now.weekday() == 6 and now.day <= 7


def _is_saturday_before_first_sunday() -> bool:
    """True on Saturday that immediately precedes first Sunday of month (NST)."""
    now = datetime.now(NST)
    if now.weekday() != 5:  # not Saturday
        return False
    tomorrow = now + timedelta(days=1)
    return tomorrow.weekday() == 6 and tomorrow.day <= 7


def _is_quarterly_review_month() -> bool:
    """True in March, June, September, December — quarterly pattern council months."""
    return datetime.now(NST).month in (3, 6, 9, 12)


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
        return False


# ── Permanent agenda items (injected in Python, not as prompt instructions) ───
MONTHLY_PERMANENT_ITEMS = [
    "Political event pattern accuracy: review last 30 days of lag predictions vs actual "
    "NEPSE moves — flag any patterns with declining weighted_accuracy trend",
]

QUARTERLY_PERMANENT_ITEMS = [
    "Quarterly pattern council: promote/demote/disable news_effect_patterns based on "
    "weighted_accuracy — ACTIVE ≥70% keep, 50-69% demote to MONITOR_ONLY, "
    "<50% with ≥5 resolved predictions disable; <5 predictions no change",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _cutoff() -> str:
    return (datetime.now(NST) - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")


def _load_daily_context() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM daily_context_log WHERE date >= %s ORDER BY date DESC LIMIT 30",
            (_cutoff(),),
        ) or []
        log.info("daily_context loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("daily_context load failed: %s", e)
        return []


def _load_trade_journal() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM trade_journal WHERE entry_date >= %s ORDER BY entry_date DESC LIMIT 30",
            (_cutoff(),),
        ) or []
        log.info("trade_journal loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("trade_journal load failed: %s", e)
        return []


def _load_gate_misses() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM gate_misses WHERE date >= %s ORDER BY date DESC LIMIT 50",
            (_cutoff(),),
        ) or []
        log.info("gate_misses loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("gate_misses load failed: %s", e)
        return []


def _load_nrb_monthly() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM nrb_monthly ORDER BY id DESC LIMIT 3",
        ) or []
        log.info("nrb_monthly loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("nrb_monthly load failed: %s", e)
        return []


def _load_claude_audit() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM claude_audit ORDER BY id DESC LIMIT 4",
        ) or []
        log.info("claude_audit loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("claude_audit load failed: %s", e)
        return []


def _load_active_lessons() -> list[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM learning_hub WHERE active = 'true' ORDER BY id DESC LIMIT 20",
        ) or []
        log.info("learning_hub loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("learning_hub load failed: %s", e)
        return []


def _load_open_positions() -> list[dict]:
    try:
        paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
        if paper_mode:
            rows = run_raw_sql(
                "SELECT symbol, wacc, total_shares, first_buy_date "
                "FROM paper_portfolio WHERE status='OPEN'"
            ) or []
        else:
            rows = run_raw_sql(
                "SELECT symbol, entry_price, shares, entry_date "
                "FROM portfolio WHERE status='OPEN'"
            ) or []
        log.info("open_positions loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("open_positions load failed: %s", e)
        return []


def _load_prior_councils(run_month: str) -> list[dict]:
    try:
        rows = run_raw_sql(
            """
            SELECT run_month, stage, agenda_item, direction, confidence,
                   key_driver, risk_factor
            FROM monthly_council_log
            WHERE run_month != %s
            ORDER BY run_month DESC
            LIMIT 60
            """,
            (run_month,),
        ) or []
        log.info("prior_councils loaded: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("prior_councils load failed: %s", e)
        return []


def _load_accuracy_review() -> Optional[dict]:
    try:
        rows = run_raw_sql(
            "SELECT * FROM accuracy_review_log ORDER BY id DESC LIMIT 1"
        )
        return rows[0] if rows else None
    except Exception:
        return None


def _load_pending_proposals() -> list[dict]:
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
        return []


def _load_pattern_validation_data() -> list[dict]:
    """
    Load political event pattern accuracy for council context.
    Groups pattern_validation_log by event_type with weighted accuracy.
    Returns [] on error. Never raises.
    """
    try:
        rows = run_raw_sql(
            """
            SELECT
                pvl.event_type,
                nep.status                                                         AS pattern_status,
                nep.evidence_quality,
                COUNT(*)                                                            AS total,
                SUM(CASE WHEN pvl.outcome = 'PENDING'         THEN 1 ELSE 0 END)  AS pending,
                SUM(CASE WHEN pvl.outcome = 'CORRECT'         THEN 1 ELSE 0 END)  AS correct,
                SUM(CASE WHEN pvl.outcome = 'WRONG_DIRECTION' THEN 1 ELSE 0 END)  AS wrong_direction,
                SUM(CASE WHEN pvl.outcome = 'WRONG_TIMING'    THEN 1 ELSE 0 END)  AS wrong_timing,
                ROUND(
                    (
                        SUM(CASE WHEN pvl.outcome = 'CORRECT'       THEN 1.0 ELSE 0 END) +
                        SUM(CASE WHEN pvl.outcome = 'WRONG_TIMING'  THEN 0.5 ELSE 0 END)
                    ) / NULLIF(
                        SUM(CASE WHEN pvl.outcome != 'PENDING' THEN 1 ELSE 0 END), 0
                    ), 3
                )                                                                  AS weighted_accuracy
            FROM pattern_validation_log pvl
            LEFT JOIN news_effect_patterns nep
                   ON nep.event_type = pvl.event_type AND nep.active = 'true'
            GROUP BY pvl.event_type, nep.status, nep.evidence_quality
            ORDER BY pvl.event_type
            """
        ) or []
        log.info("pattern_validation_data loaded: %d rows", len(rows))
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("_load_pattern_validation_data failed (non-fatal): %s", e)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUDGET HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _trim_rows_to_budget(rows: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    if not rows:
        return rows, 0
    max_chars = max_tokens * 4
    if len(json.dumps(rows, ensure_ascii=False, default=str)) <= max_chars:
        return rows, 0
    trimmed = list(rows)
    omitted = 0
    while trimmed and len(json.dumps(trimmed, ensure_ascii=False, default=str)) > max_chars:
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
    pattern_validation: list[dict] | None = None,
) -> tuple[str, dict]:
    budget = MAX_DATA_TOKENS // 6

    dc_rows,     dc_omit     = _trim_rows_to_budget(daily_context, budget)
    tj_rows,     tj_omit     = _trim_rows_to_budget(trade_journal, budget)
    gm_rows,     gm_omit     = _trim_rows_to_budget(gate_misses,   budget)
    nrb_rows,    nrb_omit    = _trim_rows_to_budget(nrb,           budget)
    audit_rows,  audit_omit  = _trim_rows_to_budget(audit,         budget)
    lesson_rows, lesson_omit = _trim_rows_to_budget(lessons,       budget)
    pv_rows,     pv_omit     = _trim_rows_to_budget(pattern_validation or [], budget)

    def _section(title, rows, omitted):
        note  = f" [{omitted} oldest omitted]" if omitted else ""
        lines = [f"=== {title}{note} ==="]
        for r in rows:
            lines.append(json.dumps(
                {k: v for k, v in r.items() if v is not None},
                ensure_ascii=False, default=str,
            ))
        return "\n".join(lines)

    parts = []
    if dc_rows:     parts.append(_section("DAILY CONTEXT (last 30 days)",        dc_rows,     dc_omit))
    if tj_rows:     parts.append(_section("TRADE JOURNAL (last 30 days)",         tj_rows,     tj_omit))
    if gm_rows:     parts.append(_section("GATE MISSES (last 30 days)",           gm_rows,     gm_omit))
    if nrb_rows:    parts.append(_section("NRB MONTHLY (last 3)",                 nrb_rows,    nrb_omit))
    if audit_rows:  parts.append(_section("CLAUDE ACCURACY AUDIT (last 4 weeks)", audit_rows,  audit_omit))
    if lesson_rows: parts.append(_section("ACTIVE LEARNING LESSONS (top 20)",     lesson_rows, lesson_omit))
    if pv_rows:     parts.append(_section("POLITICAL EVENT PATTERN ACCURACY",     pv_rows,     pv_omit))

    context = "\n\n".join(parts)
    counts  = {
        "daily_context": len(dc_rows), "trade_journal": len(tj_rows),
        "gate_misses":   len(gm_rows), "nrb":           len(nrb_rows),
        "audit":         len(audit_rows), "lessons":    len(lesson_rows),
        "pattern_validation": len(pv_rows),
        "est_tokens":    len(context) // 4,
    }
    return context, counts


# ═══════════════════════════════════════════════════════════════════════════════
# COUNCIL API CALL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def _council_call(
    model: str,
    messages: list,
    max_tokens: int,
    context: str,
    temperature: float = 0.1,       # lower temp = less hallucination
    use_search: bool = False,
) -> Optional[str]:
    """Wrapper around AI.openrouter._call for council-exclusive model calls."""
    return _call(model, messages, max_tokens, temperature, context,
                 use_search=use_search)


def _parse_json_safe(raw: str, context: str = "") -> Optional[dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text  = "\n".join(inner).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("[%s] JSON parse failed: %s | raw[:300]: %s", context, e, raw[:300])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_log_entry(entry: dict) -> None:
    try:
        write_row("monthly_council_log", entry)
    except Exception as e:
        log.error("monthly_council_log write failed: %s", e)


def _write_agenda(run_month: str, items: list[str]) -> None:
    try:
        for i, item in enumerate(items, 1):
            upsert_row("monthly_council_agenda", {
                "run_month":   run_month,
                "item_number": str(i),
                "agenda_item": item,
                "approved_by": "haiku_4.5",
            }, conflict_columns=["run_month", "item_number"])
    except Exception as e:
        log.error("monthly_council_agenda write failed: %s", e)


def _write_checklist(run_month: str, checklist: dict) -> None:
    try:
        upsert_row("monthly_council_checklist", {
            "run_month":        run_month,
            "stop_trigger":     checklist.get("stop_trigger", ""),
            "go_trigger":       checklist.get("go_trigger", ""),
            "noise_items":      json.dumps(checklist.get("noise_items", [])),
            "confidence_score": str(checklist.get("confidence_score", 50)),
        }, conflict_columns=["run_month"])
    except Exception as e:
        log.error("monthly_council_checklist write failed: %s", e)


def _write_monthly_override(
    run_month: str,
    confidence_score: int,
    market_state: str,
    checklist: dict,
    dry_run: bool = False,
) -> None:
    # Capital preservation override: confidence ≤ 20 AND BEAR/CRISIS → block all BUY
    buy_blocked = (
        "true"
        if confidence_score <= 20 and market_state in ("BEAR", "CRISIS")
        else "false"
    )
    buy_cautious = "true" if confidence_score <= 50 else "false"
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

    if dry_run:
        log.info(
            "[DRY RUN] monthly_override — run_month=%s confidence=%d "
            "buy_blocked=%s buy_cautious=%s",
            run_month, confidence_score, buy_blocked, buy_cautious,
        )
        return
    try:
        upsert_row(
            "monthly_override",
            {
                "run_month":        run_month,
                "confidence_score": str(confidence_score),
                "buy_blocked":      buy_blocked,
                "buy_cautious":     buy_cautious,
                "market_state":     market_state,
                "stop_trigger":     checklist.get("stop_trigger", ""),
                "go_trigger":       checklist.get("go_trigger", ""),
                "inserted_at":      now_nst,
                "last_read_at":     "",
            },
            conflict_columns=["run_month"],
        )
        log.info(
            "[monthly_override] written — confidence=%d buy_blocked=%s buy_cautious=%s",
            confidence_score, buy_blocked, buy_cautious,
        )
    except Exception as e:
        log.error("_write_monthly_override failed: %s", e)


_REQUIRED_LESSON_FIELDS = {"lesson_type", "condition", "finding", "action", "confidence_level"}
_VALID_ACTIONS = {
    "MONITOR", "REDUCE_CONFIDENCE_BY_15", "REDUCE_CONFIDENCE_BY_25",
    "REQUIRE_VOLUME_CONFIRM", "REQUIRE_MACRO_STABLE",
    "WAIT_FOR_CONFIRMATION", "TIGHTEN_STOP", "BLOCK_ENTRY",
}


def _validate_lesson(lesson: dict, index: int) -> bool:
    missing = _REQUIRED_LESSON_FIELDS - set(lesson.keys())
    if missing:
        log.warning("Council lesson #%d missing fields: %s — skipping", index, missing)
        return False
    if lesson.get("confidence_level") not in ("LOW", "MEDIUM", "HIGH"):
        log.warning("Council lesson #%d invalid confidence_level — skipping", index)
        return False
    return True


def _write_lessons(
    lessons: list[dict],
    run_month: str,
    dry_run: bool = False,
) -> int:
    written = 0
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    for i, lesson in enumerate(lessons or []):
        if not _validate_lesson(lesson, i):
            continue
        columns = {
            "date":             now_nst[:10],
            "symbol":           "MARKET",
            "sector":           lesson.get("sector", "ALL"),
            "source":           "monthly_council",
            "lesson_type":      lesson.get("lesson_type"),
            "applies_to":       lesson.get("applies_to", "ALL"),
            "condition":        lesson.get("condition"),
            "finding":          lesson.get("finding"),
            "action":           lesson.get("action"),
            "confidence_level": lesson.get("confidence_level"),
            "active":           "true",
            "review_week":      run_month,
            "last_validated":   now_nst[:10],
            "validation_count": "1",
            "gpt_reasoning":    lesson.get("gpt_reasoning"),
        }
        if dry_run:
            log.info("[DRY RUN] Would write lesson: %s | %s",
                     columns.get("lesson_type"), columns.get("action"))
            written += 1
            continue
        try:
            new_rows = run_raw_sql(
                'INSERT INTO learning_hub ({cols}) VALUES ({vals}) RETURNING id'.format(
                    cols=", ".join(f'"{c}"' for c in columns if columns[c] is not None),
                    vals=", ".join("%s" for c in columns if columns[c] is not None),
                ),
                tuple(v for v in columns.values() if v is not None),
            )
            if new_rows:
                written += 1
                log.info("[DB] council lesson written id=%s", new_rows[0].get("id"))
        except Exception as e:
            log.error("Failed to write lesson: %s", e)
    return written


def _write_council_system_proposals(
    run_month: str,
    system_verdict: dict,
    dry_run: bool = False,
) -> tuple[int, int]:
    if not system_verdict:
        return 0, 0
    now_nst    = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    n_endorsed = 0
    n_new      = 0

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
        except Exception as e:
            log.error("Endorse proposal id=%s failed: %s", proposal_id, e)

    for finding in system_verdict.get("new_system_findings", []):
        if dry_run:
            log.info("[DRY RUN] Would write finding: %s", finding.get("component"))
            n_new += 1
            continue
        try:
            write_row("system_proposals", {
                "run_month":       run_month,
                "source":          "monthly_council",
                "component":       str(finding.get("component", "")),
                "proposal_type":   "add_signal",
                "proposed_change": str(finding.get("proposed_change", "")),
                "data_evidence":   str(finding.get("data_evidence", "")),
                "confidence":      str(finding.get("confidence", "LOW")),
                "status":          "PENDING",
                "inserted_at":     now_nst,
            })
            n_new += 1
        except Exception as e:
            log.error("Write council finding failed: %s", e)

    log.info("[system_verdict] %d endorsed, %d new findings", n_endorsed, n_new)
    return n_endorsed, n_new


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
        "NEPSE (Nepal Stock Exchange) trades Mon-Fri in NPR. "
        "Review how well last month's council direction predictions matched actual outcomes. "
        "Be specific, factual, and brief. Plain text only — no JSON."
    )
    prior_str = json.dumps(prior_councils[:20], ensure_ascii=False, default=str) if prior_councils else "No prior council data."
    user = (
        f"HINDSIGHT AUDIT — {run_month}\n\n"
        f"PRIOR COUNCIL RESPONSES:\n{prior_str}\n\n"
        f"CURRENT MARKET DATA:\n{data_context}\n\n"
        f"Summarise in ≤200 words: what did last month's council get right and wrong?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_agenda_draft_messages(
    data_context: str,
    audit_text: str,
    run_month: str,
    accuracy_review: Optional[dict] = None,
    pending_proposals: Optional[list] = None,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Agenda Coordinator. "
        "Draft exactly 3-5 specific, debatable agenda items for the monthly council. "
        "Items must reference actual data — specific symbols, figures, or patterns from the context. "
        "Cover: market direction, system improvements, risk management, macro context. "
        "Output ONLY valid JSON: {\"agenda_items\": [...], \"draft_rationale\": \"...\"}"
    )
    accuracy_block = ""
    if accuracy_review:
        accuracy_block = f"\nACCURACY REVIEW:\n{json.dumps(accuracy_review, ensure_ascii=False, default=str)}"
    proposals_block = ""
    if pending_proposals:
        proposals_block = f"\nPENDING PROPOSALS:\n{json.dumps(pending_proposals, ensure_ascii=False)}"

    user = (
        f"AGENDA DRAFT — {run_month}\n\n"
        f"AUDIT FINDINGS:\n{audit_text}\n\n"
        f"MARKET DATA:\n{data_context}"
        f"{accuracy_block}{proposals_block}\n\n"
        f"Draft 3-5 specific agenda items as JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_agenda_review_messages(
    proposed_items: list[str],
    data_context: str,
    run_month: str,
    user_additions: Optional[list[str]] = None,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Secretary. "
        "Review the proposed agenda. Approve, reorder, or lightly rephrase items. "
        "Incorporate any user-added items. Keep 3-5 items total. "
        "Output ONLY valid JSON: {\"approved_agenda\": [...], \"review_notes\": \"...\"}"
    )
    additions_block = ""
    if user_additions:
        additions_block = f"\nUSER-ADDED ITEMS (must include):\n{json.dumps(user_additions)}"

    user = (
        f"AGENDA REVIEW — {run_month}\n\n"
        f"PROPOSED:\n{json.dumps(proposed_items, ensure_ascii=False)}"
        f"{additions_block}\n\n"
        f"MARKET CONTEXT:\n{data_context[:600]}\n\n"
        f"Approve and finalize as JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ── Position anchor guard (all discussion models) ─────────────────────────────
_POSITION_ANCHOR_GUARD = (
    "\n\nYou have been provided current open positions. Do not assume these positions are "
    "correct. Evaluate the agenda item as if entering fresh with no existing positions. "
    "Your role is objective analysis, not justification of current holdings.\n\n"
    "UNCERTAINTY RULE: If you are not certain of a fact, state it explicitly as uncertain "
    "rather than asserting it. Never fabricate specific statistics or figures."
)

# ── Model personas ─────────────────────────────────────────────────────────────
_MODEL_PERSONAS = {

    COUNCIL_GROK_MODEL: (
        "You are Grok 4.20, a contrarian market analyst on the NEPSE Monthly Council. "
        "You have web search access — use it to verify current NEPSE news, NRB announcements, "
        "and political events before making claims.\n\n"
        "CONTRARIAN STALENESS RULE: If your contrarian view matches last month's, first state "
        "whether 30-day evidence strengthened or weakened the case. Re-evaluate from evidence."
        + _POSITION_ANCHOR_GUARD
    ),

    COUNCIL_GPT_MODEL: (
        "You are GPT-5.4, a macro analyst on the NEPSE Monthly Council. "
        "You have web search access — use it to verify current NRB policy, inflation figures, "
        "remittance data, and forex reserves before citing them.\n\n"
        "Anchor in data, statistics, and historical patterns. "
        "When you cite a specific number, verify it."
        + _POSITION_ANCHOR_GUARD
    ),

    COUNCIL_DEEPSEEK_MODEL: (
        "You are DeepSeek V4 Pro, a quantitative analyst on the NEPSE Monthly Council. "
        "You work purely from the data context provided — no web search. "
        "Focus on: statistical patterns in the trade data, signal quality metrics, "
        "risk-adjusted returns, drawdown analysis, mean-reversion signals, Kelly fractions.\n\n"
        "Challenge narrative claims with math. Show your calculations. "
        "Sceptical of qualitative arguments without quantitative backing."
        + _POSITION_ANCHOR_GUARD
    ),

    COUNCIL_GEMINI_MODEL: (
        "You are Gemini 3.1 Pro, the DEVIL'S ADVOCATE on the NEPSE Monthly Council. "
        "You have web search access — use it to find counter-evidence.\n\n"
        "MANDATORY DEVIL'S ADVOCATE RULE:\n"
        "1. Read all prior analyst responses carefully.\n"
        "2. If all prior analysts have the same direction (all Bearish or all Bullish), "
        "you MUST present the strongest possible counter-argument, supported by data.\n"
        "3. Search the web for evidence that contradicts the consensus.\n"
        "4. If you find genuine counter-evidence, present it as your key_driver.\n"
        "5. If after searching you genuinely cannot find a counter-argument, explicitly state: "
        "'DEVIL'S ADVOCATE: After searching, I cannot find credible counter-evidence because [reason].' "
        "Then provide your honest assessment.\n\n"
        "VOLUME DATA CONSTRAINT: Volume data not available in daily_context_log. "
        "When you would normally cite volume, state: 'VOLUME DATA UNAVAILABLE' and "
        "reduce confidence by 10 points.\n\n"
        "CIRCUIT LIMIT EXCEPTION: Do NOT apply low-volume trap flag to stocks at circuit limit."
        + _POSITION_ANCHOR_GUARD
    ),

    COUNCIL_SONNET_MODEL: (
        "You are Claude Sonnet 4.5, a fundamental analyst on the NEPSE Monthly Council. "
        "You have web search access — use it to verify sector fundamentals, company news, "
        "NRB circulars, and dividend announcements.\n\n"
        "Stress-test positions, weigh tail risks, consider second-order effects. "
        "Focus on: sector-level fundamentals, company-specific drivers, NRB policy impact."
        + _POSITION_ANCHOR_GUARD
    ),
}

# ── Web search flags per model ────────────────────────────────────────────────
_MODEL_USE_SEARCH = {
    COUNCIL_GROK_MODEL:     True,   # news/sentiment verification
    COUNCIL_GPT_MODEL:      True,   # macro data verification
    COUNCIL_DEEPSEEK_MODEL: False,  # pure math — no web distraction
    COUNCIL_GEMINI_MODEL:   True,   # devil's advocate needs counter-evidence
    COUNCIL_SONNET_MODEL:   True,   # fundamental verification
}

# ── Discussion model sequence ─────────────────────────────────────────────────
_DISCUSSION_MODELS = [
    (COUNCIL_GROK_MODEL,     "grok_4.20",    "stage_1_grok"),
    (COUNCIL_GPT_MODEL,      "gpt_5.4",      "stage_2_gpt"),
    (COUNCIL_DEEPSEEK_MODEL, "deepseek_v4",  "stage_3_deepseek"),
    (COUNCIL_GEMINI_MODEL,   "gemini_3.1",   "stage_4_gemini_devil"),
    (COUNCIL_SONNET_MODEL,   "sonnet_4.5",   "stage_5_sonnet"),
]


def _build_discussion_messages(
    model: str,
    agenda_item: str,
    item_number: int,
    total_items: int,
    data_context: str,
    prior_responses: list[dict],
    open_positions: Optional[list[dict]] = None,
) -> list[dict]:
    persona = _MODEL_PERSONAS.get(model, "You are a senior NEPSE market analyst." + _POSITION_ANCHOR_GUARD)
    system  = (
        f"{persona}\n\n"
        "Output ONLY valid JSON with keys: "
        "direction (Bullish/Bearish/Neutral), "
        "confidence (integer 0-100), "
        "key_driver (string ≤150 chars), "
        "risk_factor (string ≤150 chars). "
        "No markdown. No preamble."
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
            "Agree with evidence, refine it, or rebut it using data."
        )
        prior_block = "\nPRIOR COUNCIL RESPONSES:\n" + "\n".join(lines) + adversarial

    positions_block = ""
    if open_positions:
        positions_block = (
            "\nCURRENT OPEN POSITIONS:\n"
            + json.dumps(open_positions, ensure_ascii=False, default=str)
        )

    user = (
        f"NEPSE MONTHLY COUNCIL — Agenda item {item_number}/{total_items}\n\n"
        f"AGENDA ITEM: {agenda_item}\n\n"
        f"MARKET DATA (last 30 days):\n{data_context}\n"
        f"{positions_block}"
        f"{prior_block}\n\n"
        f"Analyse this agenda item and output your JSON assessment."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_redteam_messages(
    transcript: str,
    open_positions: list[dict],
    run_month: str,
) -> list[dict]:
    system = (
        "You are the NEPSE Monthly Council Red Team analyst (Claude Opus 4.6 — independent). "
        "You have NOT participated in the discussion. You have web search access — use it "
        "to verify disputed facts and find current market data that may contradict or support claims.\n\n"
        "Your job:\n"
        "1. Find the two sharpest conflicting viewpoints across all analysts.\n"
        "2. Assess which conflict represents the highest risk to open positions.\n"
        "3. If any conflict involves a factual claim about the trading system itself "
        "(e.g., whether a filter exists), flag it as requires_human_review=true.\n"
        "4. If any factual claim can be verified via web search, verify it and state whether "
        "the analyst was correct.\n\n"
        "Output ONLY valid JSON:\n"
        "{\n"
        "  \"conflict_1\": \"...\",\n"
        "  \"conflict_2\": \"...\",\n"
        "  \"highest_risk\": {\"recommended_action\": \"...\", \"rationale\": \"...\"},\n"
        "  \"red_team_verdict\": \"... ≤250 chars\",\n"
        "  \"requires_human_review\": true|false,\n"
        "  \"human_review_reason\": \"... (only if requires_human_review=true)\",\n"
        "  \"factual_corrections\": [{\"claim\": \"...\", \"analyst\": \"...\", \"correction\": \"...\"}]\n"
        "}"
    )
    positions_str = json.dumps(open_positions, ensure_ascii=False, default=str) if open_positions else "No open positions."
    user = (
        f"RED TEAM REVIEW — {run_month}\n\n"
        f"FULL COUNCIL DISCUSSION TRANSCRIPT:\n{transcript}\n\n"
        f"CURRENT OPEN POSITIONS:\n{positions_str}\n\n"
        f"Identify the two sharpest conflicts. Assess highest risk. "
        f"Flag any system-design contradictions for human review. Output JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_chairman_messages(
    transcript: str,
    agenda_items: list[str],
    run_month: str,
    disagreement_scores: dict,
    open_positions: Optional[list[dict]] = None,
    pending_proposals: Optional[list[dict]] = None,
    redteam_result: Optional[dict] = None,
) -> list[dict]:
    system = (
        "You are the Chairman of the NEPSE Monthly Council (Claude Opus 4.7 — independent). "
        "You have web search access — use it to verify key facts before synthesising.\n\n"
        "CRITICAL RESPONSIBILITIES:\n"
        "1. RESOLVE RED TEAM CONTRADICTIONS: If the Red Team flagged a contradiction about "
        "the trading system's capabilities, you MUST explicitly state which view is correct "
        "based on your knowledge. If unresolvable from data, output requires_human_review=true "
        "in system_verdict.\n\n"
        "2. JUSTIFY CONFIDENCE SCORE: Your confidence score must be supported by explicit "
        "evidence from the discussion. State 2-3 specific reasons in market_assessment.\n\n"
        "3. SPECIFIC LESSONS WITH THRESHOLDS: Each lesson must include concrete, actionable "
        "thresholds — not generic advice. Example of GOOD: 'Block entry if Nepal Score ≤ -3 "
        "AND market_state = SIDEWAYS AND confidence < 50'. Example of BAD: 'Be more careful "
        "in sideways markets'.\n\n"
        "4. CAPITAL PRESERVATION: If confidence_score ≤ 20 AND market_state is BEAR or CRISIS, "
        "explicitly state in trading_checklist that all BUY signals should be blocked.\n\n"
        "5. DEVIL'S ADVOCATE: Explicitly note whether Gemini's counter-narrative changed any "
        "conclusions or was correctly rejected.\n\n"
        "UNCERTAINTY RULE: If you are not certain of a fact, state it as uncertain. "
        "Never fabricate statistics."
    )

    items_str     = "\n".join(f"{i+1}. {item}" for i, item in enumerate(agenda_items))
    ds_str        = json.dumps(disagreement_scores, ensure_ascii=False)
    positions_str = json.dumps(open_positions, ensure_ascii=False, default=str) if open_positions else "No open positions."
    proposals_str = json.dumps(pending_proposals, ensure_ascii=False) if pending_proposals else "No pending proposals."
    redteam_str   = json.dumps(redteam_result, ensure_ascii=False) if redteam_result else "No red team result."

    user = (
        f"CHAIRMAN SYNTHESIS — {run_month}\n\n"
        f"AGENDA ITEMS:\n{items_str}\n\n"
        f"DISAGREEMENT SCORES (stdev per item):\n{ds_str}\n\n"
        f"RED TEAM RESULT:\n{redteam_str}\n\n"
        f"CURRENT OPEN POSITIONS:\n{positions_str}\n\n"
        f"PENDING SYSTEM PROPOSALS:\n{proposals_str}\n\n"
        f"FULL COUNCIL DISCUSSION TRANSCRIPT:\n{transcript}\n\n"
        f"Produce final council synthesis. Output JSON:\n"
        "{\n"
        "  \"strategic_narratives\": [{\"agenda_item\": \"...\", \"narrative\": \"...\", "
        "\"council_direction\": \"Bullish|Bearish|Neutral\", \"council_confidence\": 0-100}],\n"
        "  \"confidence_score\": 0-100,\n"
        "  \"confidence_justification\": \"2-3 specific reasons from discussion evidence\",\n"
        "  \"devils_advocate_assessment\": \"Did Gemini's counter-narrative change conclusions? Why/why not?\",\n"
        "  \"market_assessment\": \"≤300 chars\",\n"
        "  \"lessons\": [{\"lesson_type\": \"...\", \"condition\": \"specific threshold condition\", "
        "\"finding\": \"...\", \"action\": \"...\", \"confidence_level\": \"LOW|MEDIUM|HIGH\", "
        "\"gpt_reasoning\": \"why this lesson is warranted\"}],\n"
        "  \"trading_checklist\": {\"stop_trigger\": \"specific condition with threshold\", "
        "\"go_trigger\": \"specific condition with threshold\", \"noise_items\": [...]},\n"
        "  \"position_evaluations\": [{\"symbol\": \"...\", \"holding_supported\": true, \"reasoning\": \"...\"}],\n"
        "  \"system_verdict\": {\"proposals_reviewed\": [...], \"new_system_findings\": [...], "
        "\"accuracy_improvement_priority\": [...], \"requires_human_review\": false, "
        "\"human_review_items\": []}\n"
        "}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSCRIPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _format_transcript(
    agenda_items: list[str],
    discussion_log: list[dict],
) -> str:
    blocks = []
    for item_idx, item in enumerate(agenda_items, start=1):
        blocks.append(f"\n── AGENDA ITEM {item_idx}: {item} ──")
        for entry in discussion_log:
            if entry.get("agenda_item") == item:
                blocks.append(
                    f"[{entry['model_label']}] direction={entry['direction']}, "
                    f"confidence={entry['confidence']}\n"
                    f"  key_driver: {entry['key_driver']}\n"
                    f"  risk_factor: {entry['risk_factor']}"
                )
    return "\n".join(blocks)


def _disagreement_score(confidences: list) -> float:
    if len(confidences) < 2:
        return 0.0
    try:
        return statistics.stdev(confidences)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COUNCIL AGENDA PREVIEW (Saturday before first Sunday)
# ═══════════════════════════════════════════════════════════════════════════════

def _send_agenda_preview(
    run_month: str,
    proposed_items: list[str],
    dry_run: bool = False,
) -> None:
    """
    Send draft agenda to Telegram for human review on Saturday.
    Stores proposed items in settings DB for retrieval on Sunday.
    """
    items_str = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(proposed_items))
    msg = (
        f"🗓 *NEPSE MONTHLY COUNCIL — Draft Agenda*\n"
        f"_{run_month} — Preview (council runs tomorrow)_\n\n"
        f"*Proposed agenda items:*\n{items_str}\n\n"
        f"📝 To add an item: `/agenda_add Your item here`\n"
        f"✅ To approve as-is: `/agenda_ok`\n\n"
        f"_If no response by 9:00 AM NST tomorrow, agenda auto-approves._"
    )

    if dry_run:
        log.info("[DRY RUN] Would send agenda preview:\n%s", msg)
        return

    # Store proposed agenda in settings for Sunday retrieval
    try:
        upsert_row("settings", {
            "key":        AGENDA_PREVIEW_SETTING,
            "value":      json.dumps(proposed_items, ensure_ascii=False),
            "updated_at": datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_by": "monthly_council_preview",
        }, conflict_columns=["key"])
        upsert_row("settings", {
            "key":        AGENDA_PREVIEW_OK_SETTING,
            "value":      "pending",
            "updated_at": datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_by": "monthly_council_preview",
        }, conflict_columns=["key"])
        log.info("Agenda preview stored in settings DB")
    except Exception as e:
        log.error("Failed to store agenda preview in settings: %s", e)

    # Send Telegram alert
    try:
        from helper.notifier import _send_admin_only
        _send_admin_only(msg, parse_mode="Markdown")
        log.info("Agenda preview sent via Telegram")
    except Exception as e:
        log.warning("Telegram agenda preview failed: %s", e)


def _load_agenda_preview() -> tuple[list[str], list[str]]:
    """
    Load Saturday's proposed agenda + any user additions from settings.
    Returns (proposed_items, user_additions).
    """
    proposed = []
    additions = []
    try:
        rows = run_raw_sql(
            "SELECT key, value FROM settings WHERE key IN (%s, %s, 'COUNCIL_AGENDA_ADDITIONS')",
            (AGENDA_PREVIEW_SETTING, AGENDA_PREVIEW_OK_SETTING),
        ) or []
        settings = {r["key"]: r["value"] for r in rows}

        raw_proposed = settings.get(AGENDA_PREVIEW_SETTING, "")
        if raw_proposed:
            proposed = json.loads(raw_proposed)

        raw_additions = settings.get("COUNCIL_AGENDA_ADDITIONS", "")
        if raw_additions:
            additions = json.loads(raw_additions)

        status = settings.get(AGENDA_PREVIEW_OK_SETTING, "pending")
        log.info("Agenda preview: %d items, %d additions, status=%s",
                 len(proposed), len(additions), status)
    except Exception as e:
        log.error("Failed to load agenda preview: %s", e)

    return proposed, additions


def run_preview(dry_run: bool = False) -> None:
    """
    Pre-council Saturday preview run.
    Runs Stage -1 + Stage 0a → sends draft agenda to Telegram.
    """
    now_nst   = datetime.now(NST)
    run_month = now_nst.strftime("%Y-%m")

    log.info("=" * 65)
    log.info("NEPSE MONTHLY COUNCIL — AGENDA PREVIEW (%s)", run_month)
    log.info("=" * 65)

    daily_context  = _load_daily_context()
    trade_journal  = _load_trade_journal()
    gate_misses    = _load_gate_misses()
    nrb            = _load_nrb_monthly()
    audit_history  = _load_claude_audit()
    lessons        = _load_active_lessons()
    prior_councils = _load_prior_councils(run_month)
    accuracy_review = _load_accuracy_review()
    pending_proposals = _load_pending_proposals()
    pattern_validation = _load_pattern_validation_data()

    data_context, counts = _build_data_context(
        daily_context, trade_journal, gate_misses, nrb, audit_history, lessons,
        pattern_validation=pattern_validation,
    )
    log.info("Data: dc=%d tj=%d gm=%d nrb=%d ~%d tokens",
             counts["daily_context"], counts["trade_journal"],
             counts["gate_misses"], counts["nrb"], counts["est_tokens"])

    # Stage -1: Audit
    audit_msgs = _build_audit_messages(data_context, prior_councils, run_month)
    if dry_run:
        audit_text = "[DRY RUN audit]"
    else:
        log.info("[preview] Stage -1: audit...")
        audit_text = _council_call(
            COUNCIL_AUDIT_MODEL, audit_msgs, MAX_AGENDA_TOKENS, "preview_audit",
            use_search=False,
        ) or ""

    # Stage 0a: Agenda draft
    draft_msgs = _build_agenda_draft_messages(
        data_context, audit_text, run_month,
        accuracy_review=accuracy_review,
        pending_proposals=pending_proposals,
    )
    if dry_run:
        proposed_items = [
            "Is NEPSE BEAR sustained or approaching reversal? [DRY RUN]",
            "Should position sizing reduce given current win rate? [DRY RUN]",
            "Which sectors show accumulation signals? [DRY RUN]",
        ]
    else:
        log.info("[preview] Stage 0a: agenda draft...")
        draft_raw    = _council_call(
            COUNCIL_AUDIT_MODEL, draft_msgs, MAX_AGENDA_TOKENS, "preview_0a",
            use_search=False,
        )
        draft_json   = _parse_json_safe(draft_raw, "preview_0a")
        proposed_items = draft_json.get("agenda_items", []) if draft_json else []
        if not proposed_items:
            proposed_items = ["General NEPSE market outlook [FALLBACK]"]

    log.info("[preview] Proposed agenda (%d items):", len(proposed_items))
    for i, item in enumerate(proposed_items, 1):
        log.info("  %d. %s", i, item)

    _send_agenda_preview(run_month, proposed_items, dry_run=dry_run)
    log.info("[preview] Complete — agenda sent for review")


# ═══════════════════════════════════════════════════════════════════════════════
# TELEGRAM NOTIFICATION (Stage 8)
# ═══════════════════════════════════════════════════════════════════════════════

def _send_council_notification(
    run_month: str,
    confidence_score: int,
    agenda_items: list[str],
    lessons_written: int,
    checklist: dict,
    system_verdict: Optional[dict] = None,
    requires_human_review: bool = False,
) -> None:
    try:
        from helper.notifier import _send_admin_only
    except ImportError:
        log.warning("Could not import notifier — skipping Telegram notification")
        return

    items_str = "\n".join(f"  {i+1}. {item[:80]}" for i, item in enumerate(agenda_items))

    system_block = ""
    if system_verdict:
        endorsed = sum(
            1 for p in system_verdict.get("proposals_reviewed", [])
            if p.get("council_assessment") == "ENDORSE"
        )
        n_new    = len(system_verdict.get("new_system_findings", []))
        system_block = f"\n⚙️ Proposals: *{endorsed} endorsed* | *{n_new} new findings*"

    review_block = ""
    if requires_human_review:
        review_block = "\n\n🔴 *REQUIRES HUMAN REVIEW* — Red Team flagged system-design contradiction."

    msg = (
        f"🏛 *NEPSE MONTHLY COUNCIL — {run_month}*\n\n"
        f"📊 Confidence: *{confidence_score}/100*\n\n"
        f"📋 Agenda ({len(agenda_items)} items):\n{items_str}\n\n"
        f"✅ GO: _{checklist.get('go_trigger', 'N/A')}_\n"
        f"🛑 STOP: _{checklist.get('stop_trigger', 'N/A')}_\n\n"
        f"📚 Lessons written: *{lessons_written}*"
        f"{system_block}"
        f"{review_block}\n\n"
        f"_Next council: first Sunday of next month._"
    )
    try:
        _send_admin_only(msg, parse_mode="Markdown")
        log.info("Stage 8: Telegram notification sent")
    except Exception as e:
        log.warning("Telegram notification failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False, force: bool = False, print_prompts: bool = False) -> None:
    """
    Full monthly council pipeline.
    dry_run=True  — load data, build prompts, print token estimates, skip API/DB.
    force=True    — skip first-Sunday guard.
    print_prompts — print all stage prompts without making API calls.
    """
    now_nst   = datetime.now(NST)
    run_month = now_nst.strftime("%Y-%m")
    now_str   = now_nst.strftime("%Y-%m-%d %H:%M:%S")

    # ── Day guard ─────────────────────────────────────────────────────────────
    if not force and not dry_run and not print_prompts:
        if not _is_first_sunday_of_month():
            log.warning("Today is not the first Sunday of the month. Use --force to override.")
            return

    # ── Duplicate guard ───────────────────────────────────────────────────────
    if not dry_run and not print_prompts and _check_already_run(run_month):
        log.warning("Council for %s already has log entries — aborting.", run_month)
        return

    log.info("=" * 65)
    log.info("NEPSE MONTHLY COUNCIL — %s", run_month)
    log.info("Stack: Grok4.20 + GPT5.4 + DeepSeekV4 + Gemini3.1(devil) + Sonnet4.5")
    log.info("       + Opus4.6(redteam) + Opus4.7(chairman)")
    log.info("=" * 65)

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
    pattern_validation = _load_pattern_validation_data()

    data_context, counts = _build_data_context(
        daily_context, trade_journal, gate_misses, nrb, audit_history, lessons,
        pattern_validation=pattern_validation,
    )
    log.info(
        "Data loaded — dc=%d tj=%d gm=%d nrb=%d audit=%d lessons=%d est_tokens=%d",
        counts["daily_context"], counts["trade_journal"], counts["gate_misses"],
        counts["nrb"], counts["audit"], counts["lessons"], counts["est_tokens"],
    )

    # ── Check for Saturday agenda preview (user-added items + pre-approved) ──
    preview_proposed, user_additions = _load_agenda_preview()
    if preview_proposed:
        log.info("Found Saturday agenda preview: %d items, %d user additions",
                 len(preview_proposed), len(user_additions))

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE -1: Hindsight audit
    # ═════════════════════════════════════════════════════════════════════════
    audit_msgs = _build_audit_messages(data_context, prior_councils, run_month)
    est_audit  = sum(len(m["content"]) for m in audit_msgs) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE -1 ({COUNCIL_AUDIT_MODEL} hindsight audit) — ~{est_audit} tokens")
        if print_prompts:
            print(f"USER: {audit_msgs[1]['content'][:400]}...")
        audit_text = "[DRY RUN audit]"
    else:
        log.info("[stage_audit] Calling %s for audit...", COUNCIL_AUDIT_MODEL)
        audit_text = _council_call(
            COUNCIL_AUDIT_MODEL, audit_msgs, MAX_AGENDA_TOKENS, "council_audit",
            use_search=False,
        ) or ""
        log.info("Stage -1 complete (%d chars)", len(audit_text))

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0a: Agenda draft
    # ═════════════════════════════════════════════════════════════════════════
    if preview_proposed:
        # Saturday preview already ran — use that agenda
        proposed_items = preview_proposed
        log.info("Using Saturday preview agenda (%d items)", len(proposed_items))
    else:
        draft_msgs = _build_agenda_draft_messages(
            data_context, audit_text, run_month,
            accuracy_review=accuracy_review,
            pending_proposals=pending_proposals,
        )
        est_draft = sum(len(m["content"]) for m in draft_msgs) // 4

        if dry_run or print_prompts:
            print(f"\n{'='*65}")
            print(f"STAGE 0a ({COUNCIL_AUDIT_MODEL} agenda draft) — ~{est_draft} tokens")
            if print_prompts:
                print(f"USER: {draft_msgs[1]['content'][:400]}...")
            proposed_items = [
                "Is NEPSE banking sector liquidity improving enough to support re-entry? [DRY RUN]",
                "Should we reduce position sizing given current NRB NPL trends? [DRY RUN]",
                "Is the current market state SIDEWAYS or transitioning to BULL? [DRY RUN]",
            ]
        else:
            log.info("[stage_0a_draft] Calling %s for agenda draft...", COUNCIL_AUDIT_MODEL)
            draft_raw      = _council_call(
                COUNCIL_AUDIT_MODEL, draft_msgs, MAX_AGENDA_TOKENS, "council_0a_draft",
                use_search=False,
            )
            draft_json     = _parse_json_safe(draft_raw, "council_0a_draft")
            proposed_items = draft_json.get("agenda_items", []) if draft_json else []
            if not proposed_items:
                proposed_items = ["General NEPSE market outlook [FALLBACK]"]
            log.info("Stage 0a: %d proposed items", len(proposed_items))

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0b: Haiku agenda review
    # ═════════════════════════════════════════════════════════════════════════
    review_msgs = _build_agenda_review_messages(
        proposed_items, data_context, run_month,
        user_additions=user_additions if user_additions else None,
    )
    est_review = sum(len(m["content"]) for m in review_msgs) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 0b ({COUNCIL_REVIEW_MODEL} agenda review) — ~{est_review} tokens")
        approved_items = proposed_items
    else:
        log.info("[stage_0b_review] Calling %s for agenda review...", COUNCIL_REVIEW_MODEL)
        review_raw    = _council_call(
            COUNCIL_REVIEW_MODEL, review_msgs, MAX_AGENDA_TOKENS, "council_0b_review",
            use_search=False,
        )
        review_json   = _parse_json_safe(review_raw, "council_0b_review")
        approved_items = review_json.get("approved_agenda", proposed_items) if review_json else proposed_items
        if not approved_items:
            approved_items = proposed_items
        _write_agenda(run_month, approved_items)

    # ── Inject permanent agenda items (Python-level, not prompt instructions) ──
    approved_items = list(approved_items)  # ensure mutable copy
    for item in MONTHLY_PERMANENT_ITEMS:
        if item not in approved_items:
            approved_items.append(item)
    if _is_quarterly_review_month():
        for item in QUARTERLY_PERMANENT_ITEMS:
            if item not in approved_items:
                approved_items.append(item)
        log.info("Quarterly review month — added %d quarterly permanent items",
                 len(QUARTERLY_PERMANENT_ITEMS))

    n_items = len(approved_items)
    log.info("NEPSE MONTHLY COUNCIL — %s — %d agenda items", run_month, n_items)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGES 1-5: Per-agenda-item discussion (5 flagship models)
    # ═════════════════════════════════════════════════════════════════════════
    discussion_log:      list[dict] = []
    disagreement_scores: dict       = {}

    for item_idx, agenda_item in enumerate(approved_items, start=1):
        log.info("── discussion item %d/%d: %s", item_idx, n_items, agenda_item[:70])
        item_responses:   list[dict] = []
        item_confidences: list[int]  = []

        for model, model_label, stage_key in _DISCUSSION_MODELS:
            msgs       = _build_discussion_messages(
                model, agenda_item, item_idx, n_items,
                data_context, item_responses,
                open_positions=open_positions,
            )
            est_tokens = sum(len(m["content"]) for m in msgs) // 4
            use_search = _MODEL_USE_SEARCH.get(model, False)

            if dry_run or print_prompts:
                print(f"\n{'='*65}")
                print(f"[{stage_key}] item {item_idx}/{n_items}: {agenda_item[:60]}")
                if print_prompts:
                    print(f"SYSTEM: {msgs[0]['content'][:300]}...")
                    print(f"USER: {msgs[1]['content'][:300]}...")
                print(f"[{'DRY RUN'}] Would call {model} (web_search={use_search}) "
                      f"~{est_tokens} tokens")
                direction   = "Neutral"
                confidence  = 50
                key_driver  = f"[DRY RUN — {model_label}]"
                risk_factor = "[DRY RUN]"
                full_response = json.dumps({
                    "direction": direction, "confidence": confidence,
                    "key_driver": key_driver, "risk_factor": risk_factor,
                })
            else:
                log.info("[%s] item %d/%d | web_search=%s", stage_key, item_idx, n_items, use_search)
                raw    = _council_call(model, msgs, MAX_DISCUSSION_TOKENS, f"council_{stage_key}",
                                       use_search=use_search)
                parsed = _parse_json_safe(raw, f"council_{stage_key}")
                direction     = str(parsed.get("direction",   "Neutral")) if parsed else "Neutral"
                confidence    = int(parsed.get("confidence",  50))        if parsed else 50
                key_driver    = str(parsed.get("key_driver",  ""))        if parsed else ""
                risk_factor   = str(parsed.get("risk_factor", ""))        if parsed else ""
                full_response = raw or ""
                log.info("[%s] direction=%s confidence=%d", stage_key, direction, confidence)

            entry = {
                "model_label":   model_label,
                "agenda_item":   agenda_item,
                "direction":     direction,
                "confidence":    confidence,
                "key_driver":    key_driver,
                "risk_factor":   risk_factor,
            }
            item_responses.append(entry)
            item_confidences.append(confidence)
            discussion_log.append(entry)

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
                    "inserted_at":   now_str,
                })

        ds = _disagreement_score(item_confidences)
        disagreement_scores[agenda_item[:80]] = round(ds, 2)
        log.info("Item %d DS=%.1f confidences=%s", item_idx, ds, item_confidences)
        if ds > 20:
            log.info("  ✅ HIGH disagreement — genuine debate")
        elif ds < 8 and not dry_run:
            log.warning("  ⚠️  LOW disagreement (%.1f) — check for groupthink", ds)

        if not dry_run and not print_prompts:
            _write_log_entry({
                "run_month":     run_month,
                "stage":         "disagreement_score",
                "agenda_item":   agenda_item,
                "model":         "computed",
                "confidence":    str(round(ds, 2)),
                "full_response": json.dumps({"confidences": item_confidences, "stdev": round(ds, 2)}),
                "inserted_at":   now_str,
            })

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6: Opus 4.6 red team
    # ═════════════════════════════════════════════════════════════════════════
    transcript   = _format_transcript(approved_items, discussion_log)
    redteam_msgs = _build_redteam_messages(transcript, open_positions, run_month)
    est_redteam  = sum(len(m["content"]) for m in redteam_msgs) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 6 ({COUNCIL_REDTEAM_MODEL} red team) — ~{est_redteam} tokens [web_search=True]")
        if print_prompts:
            print(f"USER: {redteam_msgs[1]['content'][:400]}...")
        redteam_result = {
            "highest_risk": {"recommended_action": "MONITOR"},
            "requires_human_review": False,
            "red_team_verdict": "[DRY RUN]",
        }
    else:
        log.info("[stage_6_redteam] Calling %s (web_search=True)...", COUNCIL_REDTEAM_MODEL)
        redteam_raw    = _council_call(
            COUNCIL_REDTEAM_MODEL, redteam_msgs, MAX_REDTEAM_TOKENS,
            "council_6_redteam", use_search=True,
        )
        redteam_result = _parse_json_safe(redteam_raw, "council_6_redteam") or {}
        log.info("Stage 6 complete — requires_human_review=%s",
                 redteam_result.get("requires_human_review", False))
        _write_log_entry({
            "run_month":     run_month,
            "stage":         "stage_6_redteam",
            "agenda_item":   "ALL",
            "model":         COUNCIL_REDTEAM_MODEL,
            "full_response": redteam_raw or "",
            "inserted_at":   now_str,
        })

    requires_human_review = bool(redteam_result.get("requires_human_review", False))

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 7: Opus 4.7 chairman synthesis
    # ═════════════════════════════════════════════════════════════════════════
    chairman_msgs = _build_chairman_messages(
        transcript, approved_items, run_month, disagreement_scores,
        open_positions=open_positions,
        pending_proposals=pending_proposals,
        redteam_result=redteam_result,
    )
    est_chairman = sum(len(m["content"]) for m in chairman_msgs) // 4

    if dry_run or print_prompts:
        print(f"\n{'='*65}")
        print(f"STAGE 7 ({COUNCIL_CHAIRMAN_MODEL} chairman) — ~{est_chairman} tokens [web_search=True]")
        if print_prompts:
            print(f"USER: {chairman_msgs[1]['content'][:400]}...")
        chairman_json = {
            "strategic_narratives": [],
            "confidence_score": 50,
            "confidence_justification": "[DRY RUN]",
            "devils_advocate_assessment": "[DRY RUN]",
            "market_assessment": "[DRY RUN]",
            "lessons": [],
            "trading_checklist": {
                "stop_trigger": "[DRY RUN]",
                "go_trigger":   "[DRY RUN]",
                "noise_items":  [],
            },
            "position_evaluations": [],
            "system_verdict": {
                "proposals_reviewed": [],
                "new_system_findings": [],
                "accuracy_improvement_priority": [],
                "requires_human_review": False,
                "human_review_items": [],
            },
        }
        confidence_score = 50
        checklist        = chairman_json["trading_checklist"]
        lessons_from_chairman = []
        system_verdict   = chairman_json.get("system_verdict", {})
    else:
        log.info("[stage_7_chairman] Calling %s (web_search=True)...", COUNCIL_CHAIRMAN_MODEL)
        chairman_raw  = _council_call(
            COUNCIL_CHAIRMAN_MODEL, chairman_msgs, MAX_CHAIRMAN_TOKENS,
            "council_7_chairman", temperature=0.1, use_search=True,
        )
        chairman_json = _parse_json_safe(chairman_raw, "council_7_chairman") or {}

        confidence_score       = int(chairman_json.get("confidence_score", 50))
        checklist              = chairman_json.get("trading_checklist", {})
        lessons_from_chairman  = chairman_json.get("lessons", [])
        system_verdict         = chairman_json.get("system_verdict", {})

        # Chairman may also flag human review
        if chairman_json.get("system_verdict", {}).get("requires_human_review", False):
            requires_human_review = True

        log.info("Stage 7 complete — confidence=%d devil_advocate=%s",
                 confidence_score,
                 chairman_json.get("devils_advocate_assessment", "?")[:80])

        _write_log_entry({
            "run_month":     run_month,
            "stage":         "stage_7_chairman",
            "agenda_item":   "ALL",
            "model":         COUNCIL_CHAIRMAN_MODEL,
            "confidence":    str(confidence_score),
            "full_response": chairman_raw or "",
            "inserted_at":   now_str,
        })
        _write_checklist(run_month, checklist)

    # ── Extract market_state ──────────────────────────────────────────────────
    market_state_now = (
        daily_context[0].get("market_state", "SIDEWAYS")
        if daily_context else "SIDEWAYS"
    )

    # ── Write lessons ─────────────────────────────────────────────────────────
    lessons_written = _write_lessons(lessons_from_chairman, run_month, dry_run=dry_run)

    # ── Quarterly pattern status review ───────────────────────────────────────
    if _is_quarterly_review_month():
        log.info("Running quarterly political pattern review...")
        _pattern_tally = _run_quarterly_pattern_review(dry_run=dry_run)
        log.info("Quarterly pattern review tally: %s", _pattern_tally)

    # ── Write monthly_override ────────────────────────────────────────────────
    _write_monthly_override(
        run_month, confidence_score, market_state_now, checklist, dry_run=dry_run,
    )

    # ── Write system proposals ────────────────────────────────────────────────
    _write_council_system_proposals(run_month, system_verdict, dry_run=dry_run)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 8: Telegram notification
    # ═════════════════════════════════════════════════════════════════════════
    if not dry_run and not print_prompts:
        _send_council_notification(
            run_month, confidence_score, approved_items, lessons_written,
            checklist, system_verdict=system_verdict,
            requires_human_review=requires_human_review,
        )

    log.info("=" * 65)
    log.info("COUNCIL COMPLETE — %s", run_month)
    log.info("  Confidence:          %d/100", confidence_score)
    log.info("  Lessons written:     %d", lessons_written)
    log.info("  Requires human review: %s", requires_human_review)
    log.info("=" * 65)

    if dry_run:
        print(f"\n{'='*65}")
        print(f"[DRY RUN SUMMARY] run_month={run_month}")
        print(f"  Data: {counts}")
        print(f"  Agenda items: {len(proposed_items)}")
        print(f"  Discussion models: 5 (Grok+GPT+DeepSeek+Gemini+Sonnet)")
        print(f"  Token estimates:")
        print(f"    Stage -1  audit:     ~{est_audit} in")
        print(f"    Stage 0b  review:    ~{est_review} in")
        print(f"    Stage 1-5 per item:  ~est_tokens in × 5 models × {n_items} items")
        print(f"    Stage 6   redteam:   ~{est_redteam} in")
        print(f"    Stage 7   chairman:  ~{est_chairman} in")
        print(f"  No API calls. No DB writes.")


# ═══════════════════════════════════════════════════════════════════════════════
# QUARTERLY WEIGHT REVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def _run_weight_review(dry_run: bool = False) -> Optional[dict]:
    """Quarterly lesson weight review using DeepSeek R1 + Opus validation."""
    log.info("Weight review not yet implemented — skipping")
    return None


def _run_quarterly_pattern_review(dry_run: bool = False) -> dict:
    """
    Quarterly pattern status update based on weighted_accuracy thresholds.
    Called in quarterly months (Mar/Jun/Sep/Dec) after council lessons are written.

    Promotion rules (applied to non-DISABLED patterns with active='true'):
      - resolved ≥ 5 AND weighted_accuracy ≥ 0.70 AND status == 'MONITOR_ONLY' → promote to ACTIVE
      - resolved ≥ 5 AND weighted_accuracy in [0.50, 0.70) → no change (WATCH)
      - resolved ≥ 5 AND weighted_accuracy < 0.50 → demote: ACTIVE→MONITOR_ONLY, MONITOR_ONLY→DISABLED
      - resolved < 5 → no change (insufficient data)

    Returns tally dict. Never raises.
    """
    tally = {"promoted": 0, "demoted": 0, "disabled": 0, "unchanged": 0, "skipped": 0}
    try:
        rows = run_raw_sql(
            """
            SELECT
                nep.id,
                nep.event_type,
                nep.status,
                COUNT(pvl.id)                                                        AS total,
                SUM(CASE WHEN pvl.outcome != 'PENDING' THEN 1 ELSE 0 END)           AS resolved,
                ROUND(
                    (
                        SUM(CASE WHEN pvl.outcome = 'CORRECT'       THEN 1.0 ELSE 0 END) +
                        SUM(CASE WHEN pvl.outcome = 'WRONG_TIMING'  THEN 0.5 ELSE 0 END)
                    ) / NULLIF(
                        SUM(CASE WHEN pvl.outcome != 'PENDING' THEN 1 ELSE 0 END), 0
                    ), 3
                )                                                                    AS weighted_accuracy
            FROM news_effect_patterns nep
            LEFT JOIN pattern_validation_log pvl ON pvl.event_type = nep.event_type
            WHERE nep.active = 'true' AND nep.status != 'DISABLED'
            GROUP BY nep.id, nep.event_type, nep.status
            """
        ) or []

        now_str = datetime.now(NST).strftime("%Y-%m-%d")

        for r in rows:
            pat_id   = r["id"]
            et       = r["event_type"]
            status   = r.get("status", "")
            resolved = int(r.get("resolved") or 0)
            acc      = float(r.get("weighted_accuracy") or 0.0)

            if resolved < 5:
                log.info("Pattern %s: insufficient resolved predictions (%d) — skip", et, resolved)
                tally["skipped"] += 1
                continue

            new_status = None
            if acc >= 0.70 and status == "MONITOR_ONLY":
                new_status = "ACTIVE"
                tally["promoted"] += 1
                log.info("Pattern %s: PROMOTED to ACTIVE (acc=%.3f, n=%d)", et, acc, resolved)
            elif acc < 0.50:
                if status == "ACTIVE":
                    new_status = "MONITOR_ONLY"
                    tally["demoted"] += 1
                    log.info("Pattern %s: DEMOTED to MONITOR_ONLY (acc=%.3f, n=%d)", et, acc, resolved)
                elif status == "MONITOR_ONLY":
                    new_status = "DISABLED"
                    tally["disabled"] += 1
                    log.info("Pattern %s: DISABLED (acc=%.3f, n=%d)", et, acc, resolved)
            else:
                tally["unchanged"] += 1
                log.info("Pattern %s: unchanged %s (acc=%.3f, n=%d)", et, status, acc, resolved)

            if new_status and not dry_run:
                upsert_row(
                    "news_effect_patterns",
                    {"id": str(pat_id)},
                    {
                        "id":               str(pat_id),
                        "status":           new_status,
                        "weighted_accuracy": str(acc),
                        "occurrence_count":  str(resolved),
                        "notes": (
                            f"[{now_str} quarterly-review] {status}→{new_status} "
                            f"acc={acc:.3f} resolved={resolved}"
                        ),
                    },
                )
            elif new_status and dry_run:
                log.info("[DRY-RUN] Would update %s: %s→%s", et, status, new_status)

        log.info("Quarterly pattern review: %s", tally)
    except Exception as exc:
        log.error("_run_quarterly_pattern_review failed: %s", exc)

    return tally


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    parser = argparse.ArgumentParser(description="NEPSE Monthly Council")
    parser.add_argument("--dry-run",       action="store_true", help="No API, no DB")
    parser.add_argument("--force",         action="store_true", help="Skip first-Sunday guard")
    parser.add_argument("--prompt",        action="store_true", help="Print all prompts")
    parser.add_argument("--preview",       action="store_true", help="Saturday agenda preview")
    parser.add_argument("--weight-review", action="store_true", help="Quarterly weight review")
    args = parser.parse_args()

    if args.preview:
        run_preview(dry_run=args.dry_run)
    elif args.weight_review:
        _run_weight_review(dry_run=args.dry_run)
    else:
        run(dry_run=args.dry_run, force=args.force, print_prompts=args.prompt)


if __name__ == "__main__":
    main()