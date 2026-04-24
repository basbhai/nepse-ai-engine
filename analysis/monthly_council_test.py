# -*- coding: utf-8 -*-
"""
analysis/monthly_council_test.py — NEPSE AI Engine
==============================================
TEST VERSION of monthly_council.py using free models only.
Validates the 5-model pipeline end-to-end before committing to paid models.

FREE MODEL STACK:
  Stage -1 : GPT-OSS-120b    (openai/gpt-oss-120b:free)       — hindsight audit
  Stage 0a : GPT-OSS-120b    (openai/gpt-oss-120b:free)       — agenda draft
  Stage 0b : Hy3-Preview     (tencent/hy3-preview:free)       — agenda review
  Stage 1  : DeepSeek R1     (ask_deepseek_text — Playwright)  — math/contrarian
  Stage 2  : Hy3-Preview     (tencent/hy3-preview:free)       — sentiment/news
  Stage 3  : Minimax-M2.5    (minimax/minimax-m2.5:free)      — technical
  Stage 4  : GPT-OSS-120b    (openai/gpt-oss-120b:free)       — macro/narrative
  Stage 5  : Gemini Flash     (ask_gemini_text — native SDK)   — red team
  Stage 6  : Gemini Flash     (ask_gemini_text — native SDK)   — chairman
  Stage 7  : Telegram notification

THREE RUN MODES:
  python -m analysis.monthly_council_test            # full API + read DB + NO writes + full logs
  python -m analysis.monthly_council_test --write    # full API + read DB + WITH writes + full logs
  python -m analysis.monthly_council_test --dry-run  # no API + no writes (structure test only)
  python -m analysis.monthly_council_test --prompt   # print prompts only

KEY DESIGN:
  - dry_run=True  → skips ALL API calls (structure/data test)
  - write_db=True → enables DB writes (off by default — safe to run anytime)
  - Logs EVERYTHING regardless of dry_run or write_db
  - DB tag: TEST-YYYY-MM (never conflicts with production)
  - $0 cost (all free models)
"""

import argparse
import json
import logging
import os
import statistics
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from AI.openrouter import _call
from AI import ask_gemini_text, ask_deepseek_text
from sheets import run_raw_sql, write_row, upsert_row, get_setting

NST = ZoneInfo("Asia/Kathmandu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── TEST model constants — all free ──────────────────────────────────────────
TEST_GPT_MODEL      = "qwen/qwen3-coder:free"
TEST_HY3_MODEL      = "tencent/hy3-preview:free"
TEST_MINIMAX_MODEL  = "minimax/minimax-m2.5:free"
# Stage 1   → ask_deepseek_text() Playwright browser (free)
# Stage 5,6 → ask_gemini_text()   native SDK free keys

TEST_RUN_PREFIX = "TEST"

# Token budget (same as production)
MAX_DATA_TOKENS       = 2000
MAX_DISCUSSION_TOKENS = 600
MAX_CHAIRMAN_TOKENS   = 1500
MAX_REDTEAM_TOKENS    = 800
MAX_AGENDA_TOKENS     = 400
DATA_LOOKBACK_DAYS    = 30

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CALL WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _openrouter(model: str, messages: list, max_tokens: int, context: str,
                temperature: float = 0.3) -> Optional[str]:
    log.info("[API] OpenRouter → %s (%s)", model, context)
    result = _call(model, messages, max_tokens, temperature, context)
    if result:
        log.info("[API] %s responded — %d chars", model, len(result))
        # Full response logged separately in caller
    else:
        log.warning("[API] %s returned None", model)
    return result


def _gemini(prompt: str, system: str, context: str) -> Optional[str]:
    log.info("[API] Gemini Flash native SDK (%s)", context)
    result = ask_gemini_text(prompt=prompt, system=system, context=context)
    if result:
        log.info("[API] Gemini responded — %d chars", len(result))
    else:
        log.warning("[API] Gemini returned None")
    return result


def _deepseek(prompt: str, system: str, context: str) -> Optional[str]:
    log.info("[API] DeepSeek Playwright browser (%s)", context)
    result = ask_deepseek_text(prompt=prompt, system=system, context=context)
    if result is None:
        log.warning("[API] DeepSeek returned None")
        return None
    if isinstance(result, dict):
        serialized = json.dumps(result, ensure_ascii=False)
        log.info("[API] DeepSeek responded (dict) — %d chars", len(serialized))
        return serialized
    log.info("[API] DeepSeek responded — %d chars", len(str(result)))
    return str(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — JSON HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_json_safe(raw: str, context: str = "") -> Optional[dict]:
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
        parsed = json.loads(text)
        log.info("[JSON] %s parsed OK", context)
        return parsed
    except json.JSONDecodeError as e:
        log.error("[JSON] %s parse failed: %s | raw[:200]: %s", context, e, raw[:200])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADERS (logging suppressed per user request)
# ═══════════════════════════════════════════════════════════════════════════════

def _cutoff() -> str:
    return (datetime.now(NST) - timedelta(days=DATA_LOOKBACK_DAYS)).strftime("%Y-%m-%d")


def _load(query: str, params=None, label: str = "") -> list[dict]:
    try:
        rows = run_raw_sql(query, params) or []
        # Logging removed per user request: "skip logging like fetching data etc."
        return rows
    except Exception as e:
        log.error("[DB] %s load failed: %s", label, e)
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
        return rows
    except Exception as e:
        log.error("[DB] open_positions load failed: %s", e)
        return []


def _load_all_data() -> dict:
    log.info("[DATA] Loading all context data...")
    return {
        "daily_context":  _load(
            "SELECT * FROM daily_context_log WHERE date >= %s ORDER BY date DESC LIMIT 30",
            (_cutoff(),), "daily_context"),
        "trade_journal":  _load(
            "SELECT * FROM trade_journal WHERE entry_date >= %s ORDER BY entry_date DESC LIMIT 30",
            (_cutoff(),), "trade_journal"),
        "gate_misses":    _load(
            "SELECT * FROM gate_misses WHERE date >= %s ORDER BY date DESC LIMIT 50",
            (_cutoff(),), "gate_misses"),
        "nrb":            _load(
            "SELECT * FROM nrb_monthly ORDER BY id DESC LIMIT 3",
            label="nrb_monthly"),
        "audit":          _load(
            "SELECT * FROM claude_audit ORDER BY id DESC LIMIT 4",
            label="claude_audit"),
        "lessons":        _load(
            "SELECT * FROM learning_hub WHERE active = 'true' ORDER BY id DESC LIMIT 20",
            label="learning_hub"),
        "open_positions": _load_open_positions(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA CONTEXT BUILDER (logging suppressed)
# ═══════════════════════════════════════════════════════════════════════════════

def _trim(rows: list, max_tokens: int) -> tuple[list, int]:
    if not rows:
        return rows, 0
    max_chars = max_tokens * 4
    trimmed, omitted = list(rows), 0
    while trimmed and len(json.dumps(trimmed, ensure_ascii=False, default=str)) > max_chars:
        trimmed.pop(0)
        omitted += 1
    return trimmed, omitted


def _build_data_context(data: dict) -> tuple[str, dict]:
    budget = MAX_DATA_TOKENS // 6

    def _section(title, rows, omitted):
        note  = f" [{omitted} oldest omitted]" if omitted else ""
        lines = [f"=== {title}{note} ==="]
        for r in rows:
            lines.append(json.dumps(
                {k: v for k, v in r.items() if v is not None},
                ensure_ascii=False, default=str,
            ))
        return "\n".join(lines)

    sections = [
        ("DAILY CONTEXT (last 30 days)",        "daily_context"),
        ("TRADE JOURNAL (last 30 days)",         "trade_journal"),
        ("GATE MISSES (last 30 days)",           "gate_misses"),
        ("NRB MONTHLY (last 3)",                 "nrb"),
        ("CLAUDE ACCURACY AUDIT (last 4 weeks)", "audit"),
        ("ACTIVE LEARNING LESSONS (top 20)",     "lessons"),
    ]

    parts, counts = [], {}
    for title, key in sections:
        rows, omit = _trim(data.get(key, []), budget)
        counts[key] = len(rows)
        if rows:
            parts.append(_section(title, rows, omit))

    context = "\n\n".join(parts)
    counts["est_tokens"] = len(context) // 4
    # Removed logging of estimated tokens per user request
    return context, counts


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_GUARD = "\n\nOutput ONLY valid JSON. No markdown fences. No preamble."

_PERSONAS = {
    "deepseek": (
        "You are a rigorous quantitative analyst specialising in emerging market microstructure. "
        "Challenge consensus with mathematical evidence. Deeply sceptical of narrative-driven analysis. "
        "Focus on: statistical patterns, liquidity conditions, mean-reversion, risk-adjusted metrics."
        + _JSON_GUARD
    ),
    "hy3": (
        "You are a NEPSE sentiment analyst with deep knowledge of Nepal's political economy. "
        "Focus on: news sentiment, political risk, remittance flows, retail investor behaviour, social signals."
        + _JSON_GUARD
    ),
    "minimax": (
        "You are a technical analysis specialist for NEPSE. "
        "Focus on: price patterns, volume profiles, MACD/RSI/Bollinger, sector rotation, DPR dynamics."
        + _JSON_GUARD
    ),
    "gpt_oss": (
        "You are a macro analyst covering Nepal's economy and NEPSE. "
        "Focus on: NRB monetary policy, inflation, forex reserves, remittances, banking health, BOP."
        + _JSON_GUARD
    ),
}

_DISCUSSION_SCHEMA = (
    "Output ONLY valid JSON with keys: "
    "direction (Bullish/Bearish/Neutral), "
    "confidence (integer 0-100), "
    "key_driver (string ≤120 chars), "
    "risk_factor (string ≤120 chars)."
)


def _discussion_prompt(persona_key, agenda_item, item_idx, n_items,
                       data_context, prior_responses, open_positions):
    persona = _PERSONAS.get(persona_key, "You are a NEPSE analyst." + _JSON_GUARD)
    system  = f"{persona}\n{_DISCUSSION_SCHEMA}"

    prior_block = ""
    if prior_responses:
        lines = [
            f"[{r['model_label']}] direction={r['direction']}, "
            f"confidence={r['confidence']}, key_driver={r['key_driver']}"
            for r in prior_responses
        ]
        last = prior_responses[-1]
        adversarial = (
            f"\nADVERSARIAL INSTRUCTION: Engage directly with this claim by "
            f"{last['model_label']}: \"{last['key_driver']}\". "
            "Agree, refine, or rebut it with evidence from the data."
        )
        prior_block = "\nPRIOR COUNCIL RESPONSES:\n" + "\n".join(lines) + adversarial

    pos_block = ""
    if open_positions:
        pos_block = (
            "\nCURRENT OPEN POSITIONS:\n"
            + json.dumps(open_positions, ensure_ascii=False, default=str)
        )

    user = (
        f"MONTHLY COUNCIL TEST — Agenda item {item_idx}/{n_items}\n\n"
        f"AGENDA ITEM: {agenda_item}\n\n"
        f"MARKET DATA (last 30 days):\n{data_context}\n"
        f"{pos_block}\n{prior_block}\n\n"
        f"Analyse this agenda item and output your structured assessment as JSON."
    )
    return system, user


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DB WRITERS (only called when write_db=True)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_log(entry: dict, write_db: bool) -> None:
    if not write_db:
        log.info("[WRITE SKIPPED] monthly_council_log — stage=%s", entry.get("stage", "?"))
        return
    try:
        write_row("monthly_council_log", entry)
        log.info("[DB WRITE] monthly_council_log — stage=%s", entry.get("stage", "?"))
    except Exception as e:
        log.error("[DB WRITE FAILED] monthly_council_log: %s", e)


def _write_agenda(run_month: str, items: list, write_db: bool) -> None:
    if not write_db:
        log.info("[WRITE SKIPPED] monthly_council_agenda — %d items", len(items))
        return
    try:
        for i, item in enumerate(items, 1):
            upsert_row("monthly_council_agenda", {
                "run_month":   run_month,
                "item_number": str(i),
                "agenda_item": item,
                "approved_by": "hy3_test",
            }, conflict_columns=["run_month", "item_number"])
        log.info("[DB WRITE] monthly_council_agenda — %d items", len(items))
    except Exception as e:
        log.error("[DB WRITE FAILED] monthly_council_agenda: %s", e)


def _write_checklist(run_month: str, chairman: dict, write_db: bool) -> None:
    if not write_db:
        log.info("[WRITE SKIPPED] monthly_council_checklist")
        return
    try:
        upsert_row("monthly_council_checklist", {
            "run_month":        run_month,
            "stop_trigger":     chairman.get("stop_trigger", ""),
            "go_trigger":       chairman.get("go_trigger", ""),
            "noise_items":      json.dumps(chairman.get("noise_items", [])),
            "confidence_score": str(chairman.get("nepse_confidence_score", 50)),
        }, conflict_columns=["run_month"])
        log.info("[DB WRITE] monthly_council_checklist")
    except Exception as e:
        log.error("[DB WRITE FAILED] monthly_council_checklist: %s", e)


def _write_override(run_month: str, confidence: int, market_state: str,
                    chairman: dict, write_db: bool) -> None:
    buy_blocked  = confidence <= 35 and market_state in ("BEAR", "CRISIS")
    buy_cautious = confidence <= 50
    log.info("[OVERRIDE] confidence=%d blocked=%s cautious=%s",
             confidence, buy_blocked, buy_cautious)
    if not write_db:
        log.info("[WRITE SKIPPED] monthly_override")
        return
    try:
        upsert_row("monthly_override", {
            "run_month":    run_month,
            "confidence":   str(confidence),
            "buy_blocked":  str(buy_blocked).lower(),
            "buy_cautious": str(buy_cautious).lower(),
            "go_trigger":   chairman.get("go_trigger", ""),
            "stop_trigger": chairman.get("stop_trigger", ""),
        }, conflict_columns=["run_month"])
        log.info("[DB WRITE] monthly_override")
    except Exception as e:
        log.error("[DB WRITE FAILED] monthly_override: %s", e)


def _write_lessons(run_month: str, chairman: dict, now_str: str,
                   write_db: bool) -> int:
    lessons = chairman.get("lessons", [])
    count   = 0
    for lesson in lessons:
        if not lesson or len(lesson.strip()) <= 10:
            continue
        log.info("[LESSON] %s", lesson.strip()[:120])  # kept reasonable length
        if not write_db:
            log.info("[WRITE SKIPPED] learning_hub lesson")
            continue
        try:
            write_row("learning_hub", {
                "week":        run_month,
                "lesson_type": "COUNCIL_TEST",
                "lesson":      lesson.strip(),
                "active":      "true",
                "source":      "monthly_council_test",
                "inserted_at": now_str,
            })
            log.info("[DB WRITE] learning_hub lesson written")
            count += 1
        except Exception as e:
            log.error("[DB WRITE FAILED] learning_hub: %s", e)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _format_transcript(approved_items: list, discussion_log: list) -> str:
    lines = []
    for item in approved_items:
        lines.append(f"\n── AGENDA: {item} ──")
        for e in discussion_log:
            if e.get("agenda_item") == item:
                lines.append(
                    f"  [{e['model_label']}] {e['direction']} "
                    f"({e['confidence']}%) — {e['key_driver']}"
                )
    return "\n".join(lines)


def _disagreement_score(confidences: list) -> float:
    if len(confidences) < 2:
        return 0.0
    try:
        return statistics.stdev(confidences)
    except Exception:
        return 0.0


def _send_telegram(run_month: str, confidence: int, approved_items: list,
                   lessons_count: int, chairman: dict) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("[TELEGRAM] Not configured — skipping notification")
        return
    try:
        import requests
        verdict = chairman.get("system_verdict", "UNKNOWN")
        emoji   = {"PROCEED": "🟢", "CAUTION": "🟡", "HALT": "🔴"}.get(verdict, "⚪")
        text = (
            f"🧪 *MONTHLY COUNCIL TEST — {run_month}*\n\n"
            f"{emoji} Verdict: *{verdict}*\n"
            f"Confidence: {confidence}/100\n"
            f"Agenda: {len(approved_items)} items | Lessons: {lessons_count}\n\n"
            f"Go: {chairman.get('go_trigger','?')[:80]}\n"
            f"Stop: {chairman.get('stop_trigger','?')[:80]}"
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
        log.info("[TELEGRAM] Notification sent")
    except Exception as e:
        log.error("[TELEGRAM] Failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False, print_prompts: bool = False,
        write_db: bool = False) -> None:
    """
    dry_run=True  → skip all API calls (structure test only)
    write_db=True → enable DB writes (off by default)
    Logs everything regardless of mode.
    """
    now_nst   = datetime.now(NST)
    run_month = f"{TEST_RUN_PREFIX}-{now_nst.strftime('%Y-%m')}"
    now_str   = now_nst.strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 65)
    log.info("NEPSE MONTHLY COUNCIL TEST — %s", run_month)
    log.info("Stack: DeepSeek(browser) + Hy3 + Minimax + GPT-OSS + Gemini(SDK)")
    log.info("Mode:  dry_run=%-5s | write_db=%-5s | print_prompts=%s",
             dry_run, write_db, print_prompts)
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    data           = _load_all_data()
    data_context, counts = _build_data_context(data)
    open_positions = data["open_positions"]
    daily_context  = data["daily_context"]

    # ── Data logging suppressed per user request (no counts logged) ──────────

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE -1: GPT-OSS hindsight audit
    # ═════════════════════════════════════════════════════════════════════════
    log.info("── STAGE -1: GPT-OSS hindsight audit")
    audit_msgs = [
        {"role": "system", "content": (
            "You are a NEPSE trading system auditor. "
            "Summarise last month's trading outcomes vs predictions in ≤200 words. Plain text."
        )},
        {"role": "user", "content": (
            f"AUDIT — {run_month}\n\nDATA:\n{data_context[:1000]}\n\n"
            f"What worked and what failed last month?"
        )},
    ]
    if dry_run:
        log.info("[DRY RUN] Stage -1 skipped — would call %s", TEST_GPT_MODEL)
        audit_text = "[DRY RUN]"
    else:
        audit_text = _openrouter(TEST_GPT_MODEL, audit_msgs, MAX_AGENDA_TOKENS, "test_audit") or ""
        # Full response logged without truncation
        log.info("Stage -1 audit response:\n%s", audit_text)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0a: GPT-OSS agenda draft
    # ═════════════════════════════════════════════════════════════════════════
    log.info("── STAGE 0a: GPT-OSS agenda draft")
    draft_msgs = [
        {"role": "system", "content": (
            "You are a NEPSE market analyst. "
            "Draft 3-5 specific, actionable agenda items for the monthly council deliberation. "
            "Output ONLY JSON: {\"agenda_items\": [\"item1\", ...]}"
        )},
        {"role": "user", "content": (
            f"MONTHLY COUNCIL TEST — {run_month}\n\n"
            f"MARKET DATA:\n{data_context}\n\n"
            f"Draft 3-5 agenda items as JSON."
        )},
    ]
    if print_prompts:
        print(f"\n{'='*65}\nSTAGE 0a PROMPT:\n{draft_msgs[1]['content']}\n")

    if dry_run:
        log.info("[DRY RUN] Stage 0a skipped — would call %s", TEST_GPT_MODEL)
        proposed_items = [
            "Is NEPSE in sustained BEAR or nearing reversal? [DRY RUN]",
            "Should position sizing reduce given 25% win rate? [DRY RUN]",
            "Which sectors show accumulation in floorsheet data? [DRY RUN]",
        ]
    else:
        raw            = _openrouter(TEST_GPT_MODEL, draft_msgs, MAX_AGENDA_TOKENS, "test_0a")
        log.info("Stage 0a raw response:\n%s", raw)  # full response
        parsed         = _parse_json_safe(raw, "test_0a")
        proposed_items = parsed.get("agenda_items", []) if parsed else []
        if not proposed_items:
            log.warning("Stage 0a returned no items — using fallback")
            proposed_items = ["General NEPSE market outlook [FALLBACK]"]
        log.info("Stage 0a complete — %d proposed items: %s",
                 len(proposed_items), proposed_items)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0b: Hy3 agenda review (replaced GLM)
    # ═════════════════════════════════════════════════════════════════════════
    log.info("── STAGE 0b: Hy3 agenda review")
    review_msgs = [
        {"role": "system", "content": (
            "You are reviewing a NEPSE council agenda. "
            "Approve, reorder, or refine the proposed items. "
            "Output ONLY JSON: {\"approved_agenda\": [\"item1\", ...]}"
        )},
        {"role": "user", "content": (
            f"PROPOSED AGENDA:\n{json.dumps(proposed_items, ensure_ascii=False)}\n\n"
            f"MARKET CONTEXT:\n{data_context[:600]}\n\n"
            f"Approve the agenda as JSON."
        )},
    ]
    if print_prompts:
        print(f"\n{'='*65}\nSTAGE 0b PROMPT:\n{review_msgs[1]['content']}\n")

    if dry_run:
        log.info("[DRY RUN] Stage 0b skipped — would call %s", TEST_HY3_MODEL)
        approved_items = proposed_items
    else:
        raw            = _openrouter(TEST_HY3_MODEL, review_msgs, MAX_AGENDA_TOKENS, "test_0b")
        log.info("Stage 0b raw response:\n%s", raw)  # full response
        parsed         = _parse_json_safe(raw, "test_0b")
        approved_items = parsed.get("approved_agenda", proposed_items) if parsed else proposed_items
        if not approved_items:
            approved_items = proposed_items
        log.info("Stage 0b complete — %d approved items: %s",
                 len(approved_items), approved_items)
        _write_agenda(run_month, approved_items, write_db)

    n_items = len(approved_items)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGES 1-4: Per-agenda-item discussion (4 models)
    # ═════════════════════════════════════════════════════════════════════════
    discussion_log:      list[dict] = []
    disagreement_scores: dict       = {}

    # (persona_key, model_label, stage_key, call_type, model_str_or_None)
    _MODELS = [
        ("deepseek", "deepseek_r1",   "stage_1_deepseek",  "deepseek",   None),
        ("hy3",      "hy3_preview",   "stage_2_hy3",       "openrouter", TEST_HY3_MODEL),
        ("minimax",  "minimax_m2.5",  "stage_3_minimax",   "openrouter", TEST_MINIMAX_MODEL),
        ("gpt_oss",  "gpt_oss_120b",  "stage_4_gpt_oss",   "openrouter", TEST_GPT_MODEL),
    ]

    for item_idx, agenda_item in enumerate(approved_items, start=1):
        log.info("── Discussion %d/%d: %s", item_idx, n_items, agenda_item)
        item_responses:   list[dict] = []
        item_confidences: list[int]  = []

        for persona_key, model_label, stage_key, call_type, model_str in _MODELS:
            system, user = _discussion_prompt(
                persona_key, agenda_item, item_idx, n_items,
                data_context, item_responses, open_positions,
            )

            if print_prompts:
                print(f"\n{'='*65}\n{stage_key} PROMPT:")
                print(f"SYSTEM: {system}")
                print(f"USER:   {user}\n")

            if dry_run:
                log.info("[DRY RUN] %s skipped — would call %s (%s)",
                         stage_key, call_type, model_str or "browser")
                direction, confidence = "Neutral", 50
                key_driver  = f"[DRY RUN — {model_label}]"
                risk_factor = "[DRY RUN]"
                raw = json.dumps({"direction": direction, "confidence": confidence,
                                  "key_driver": key_driver, "risk_factor": risk_factor})
            else:
                log.info("[%s] calling %s (%s)...", stage_key, call_type,
                         model_str or "playwright")
                if call_type == "deepseek":
                    raw = _deepseek(user, system, f"test_{stage_key}")
                else:
                    msgs = [{"role": "system", "content": system},
                            {"role": "user",   "content": user}]
                    raw  = _openrouter(model_str, msgs,
                                       MAX_DISCUSSION_TOKENS, f"test_{stage_key}")

                # Full raw response logged
                log.info("[%s] full response:\n%s", stage_key, raw)

                parsed      = _parse_json_safe(raw, stage_key) if raw else None
                direction   = str(parsed.get("direction",   "Neutral")) if parsed else "Neutral"
                confidence  = int(parsed.get("confidence",  50))        if parsed else 50
                key_driver  = str(parsed.get("key_driver",  ""))        if parsed else ""
                risk_factor = str(parsed.get("risk_factor", ""))        if parsed else ""

                log.info("[%s] result: direction=%s confidence=%d key_driver=%s",
                         stage_key, direction, confidence, key_driver)

            entry = {
                "model_label": model_label,
                "stage":       stage_key,
                "agenda_item": agenda_item,
                "direction":   direction,
                "confidence":  confidence,
                "key_driver":  key_driver,
                "risk_factor": risk_factor,
            }
            item_responses.append(entry)
            item_confidences.append(confidence)
            discussion_log.append(entry)

            _write_log({
                "run_month":     run_month,
                "stage":         stage_key,
                "agenda_item":   agenda_item,
                "model":         model_str or "deepseek_browser",
                "direction":     direction,
                "confidence":    str(confidence),
                "key_driver":    key_driver,
                "risk_factor":   risk_factor,
                "full_response": raw or "",
                "inserted_at":   now_str,
            }, write_db)

        ds = _disagreement_score(item_confidences)
        disagreement_scores[agenda_item[:80]] = round(ds, 2)
        log.info("Item %d disagreement score: %.1f | confidences: %s",
                 item_idx, ds, item_confidences)
        if ds > 25:
            log.warning("HIGH disagreement on item %d (DS=%.1f) — good debate", item_idx, ds)
        elif ds < 8 and not dry_run:
            log.warning("LOW disagreement on item %d (DS=%.1f) — possible groupthink", item_idx, ds)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5: Gemini Flash red team
    # ═════════════════════════════════════════════════════════════════════════
    log.info("── STAGE 5: Gemini Flash red team")
    transcript = _format_transcript(approved_items, discussion_log)
    log.info("Transcript built — %d chars", len(transcript))

    rt_sys = (
        "You are the NEPSE Monthly Council Red Team analyst (independent — not part of the discussion). "
        "Find the two sharpest conflicting viewpoints and assess highest risk to open positions. "
        "Output ONLY valid JSON with keys: "
        "conflict_1 (string), conflict_2 (string), "
        "highest_risk (dict with recommended_action string), "
        "red_team_verdict (string ≤200 chars)."
    )
    rt_user = (
        f"RED TEAM REVIEW — {run_month}\n\n"
        f"FULL COUNCIL DISCUSSION TRANSCRIPT:\n{transcript}\n\n"
        f"CURRENT OPEN POSITIONS:\n"
        f"{json.dumps(open_positions, ensure_ascii=False, default=str)}\n\n"
        f"Identify conflicts. Assess risk. Output JSON."
    )

    if print_prompts:
        print(f"\n{'='*65}\nSTAGE 5 RED TEAM PROMPT:\n{rt_user}\n")

    if dry_run:
        log.info("[DRY RUN] Stage 5 skipped — would call ask_gemini_text()")
        redteam_result = {"highest_risk": {"recommended_action": "MONITOR"},
                          "red_team_verdict": "[DRY RUN]"}
    else:
        rt_raw         = _gemini(rt_user, rt_sys, "test_5_redteam")
        log.info("Stage 5 full response:\n%s", rt_raw)  # full response
        redteam_result = _parse_json_safe(rt_raw, "test_5_redteam") or {}
        log.info("Stage 5 complete — verdict: %s | risk: %s",
                 redteam_result.get("red_team_verdict", "?")[:80],
                 redteam_result.get("highest_risk", {}).get("recommended_action", "?"))
        _write_log({
            "run_month":     run_month,
            "stage":         "stage_5_redteam",
            "agenda_item":   "ALL",
            "model":         "gemini_flash_native",
            "full_response": rt_raw or "",
            "inserted_at":   now_str,
        }, write_db)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6: Gemini Flash chairman synthesis
    # ═════════════════════════════════════════════════════════════════════════
    log.info("── STAGE 6: Gemini Flash chairman synthesis")
    ch_sys = (
        "You are the Chairman of the NEPSE Monthly Council. "
        "Synthesise all discussion into final actionable guidance. "
        "Output ONLY valid JSON with keys: "
        "nepse_confidence_score (integer 0-100), "
        "market_assessment (string ≤300 chars), "
        "go_trigger (string ≤150 chars), "
        "stop_trigger (string ≤150 chars), "
        "noise_items (list ≤3 strings), "
        "lessons (list ≤5 strings — specific lessons for the trading system), "
        "system_verdict (PROCEED / CAUTION / HALT)."
    )
    ch_user = (
        f"CHAIRMAN SYNTHESIS — {run_month}\n\n"
        f"AGENDA ITEMS:\n{json.dumps(approved_items, ensure_ascii=False)}\n\n"
        f"DISAGREEMENT SCORES (stdev per item):\n{json.dumps(disagreement_scores)}\n\n"
        f"FULL DISCUSSION TRANSCRIPT:\n{transcript}\n\n"
        f"CURRENT OPEN POSITIONS:\n"
        f"{json.dumps(open_positions, ensure_ascii=False, default=str)}\n\n"
        f"Synthesise into final council output. Output JSON."
    )

    if print_prompts:
        print(f"\n{'='*65}\nSTAGE 6 CHAIRMAN PROMPT:\n{ch_user}\n")

    if dry_run:
        log.info("[DRY RUN] Stage 6 skipped — would call ask_gemini_text()")
        chairman = {
            "nepse_confidence_score": 50,
            "market_assessment":      "[DRY RUN]",
            "go_trigger":             "[DRY RUN]",
            "stop_trigger":           "[DRY RUN]",
            "noise_items":            [],
            "lessons":                [],
            "system_verdict":         "CAUTION",
        }
        confidence_score = 50
    else:
        ch_raw       = _gemini(ch_user, ch_sys, "test_6_chairman")
        log.info("Stage 6 full response:\n%s", ch_raw)  # full response
        chairman     = _parse_json_safe(ch_raw, "test_6_chairman") or {}
        confidence_score = int(chairman.get("nepse_confidence_score", 50))

        log.info("Stage 6 complete:")
        log.info("  Verdict:    %s", chairman.get("system_verdict", "?"))
        log.info("  Confidence: %d/100", confidence_score)
        log.info("  Assessment: %s", chairman.get("market_assessment", "?"))
        log.info("  Go trigger: %s", chairman.get("go_trigger", "?"))
        log.info("  Stop:       %s", chairman.get("stop_trigger", "?"))
        log.info("  Noise:      %s", chairman.get("noise_items", []))
        for i, lesson in enumerate(chairman.get("lessons", []), 1):
            log.info("  Lesson %d: %s", i, lesson)

        _write_log({
            "run_month":     run_month,
            "stage":         "stage_6_chairman",
            "agenda_item":   "ALL",
            "model":         "gemini_flash_native",
            "confidence":    str(confidence_score),
            "full_response": ch_raw or "",
            "inserted_at":   now_str,
        }, write_db)
        _write_checklist(run_month, chairman, write_db)

        market_state = (daily_context[0].get("market_state", "SIDEWAYS")
                        if daily_context else "SIDEWAYS")
        _write_override(run_month, confidence_score, market_state, chairman, write_db)

    # Write lessons
    lessons_written = _write_lessons(run_month, chairman, now_str, write_db)

    # ── Stage 7: Telegram ─────────────────────────────────────────────────────
    log.info("── STAGE 7: Telegram notification")
    if not dry_run:
        _send_telegram(run_month, confidence_score, approved_items,
                       lessons_written, chairman)
    else:
        log.info("[DRY RUN] Telegram skipped")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("COUNCIL TEST COMPLETE")
    log.info("  run_month:   %s", run_month)
    log.info("  confidence:  %d/100", confidence_score)
    log.info("  verdict:     %s", chairman.get("system_verdict", "?"))
    log.info("  lessons:     %d written", lessons_written)
    log.info("  write_db:    %s", write_db)
    log.info("  DB tag:      %s (safe — never conflicts with production)", run_month)
    log.info("=" * 65)

    if dry_run:
        print(f"\n{'='*65}")
        print(f"[DRY RUN SUMMARY] run_month={run_month}")
        print(f"  Data:     {counts}")
        print(f"  Agenda:   {len(proposed_items)} proposed items")
        print(f"  Models:   DeepSeek(browser) + Hy3 + Minimax + GPT-OSS")
        print(f"  Chairman: Gemini Flash (native SDK)")
        print(f"  Cost:     $0.00 (all free)")
        print(f"  Writes:   NONE")
        print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Monthly Council TEST (free models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analysis.monthly_council_test            # full API, read DB, NO writes
  python -m analysis.monthly_council_test --write    # full API, read DB, WITH writes
  python -m analysis.monthly_council_test --dry-run  # no API, no writes (structure test)
  python -m analysis.monthly_council_test --prompt   # print prompts only
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all API calls (structure/data test only)")
    parser.add_argument("--prompt",  action="store_true",
                        help="Print all prompts before API calls")
    parser.add_argument("--write",   action="store_true",
                        help="Enable DB writes (off by default — safe to run anytime)")
    parser.add_argument("--force",   action="store_true",
                        help="No-op (test always runs regardless of day)")
    args = parser.parse_args()

    from log_config import attach_file_handler
    attach_file_handler(__name__)

    run(
        dry_run      = args.dry_run,
        print_prompts = args.prompt,
        write_db     = args.write,
    )


if __name__ == "__main__":
    main()