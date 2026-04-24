# -*- coding: utf-8 -*-
"""
analysis/monthly_council_test.py — NEPSE AI Engine
==============================================
TEST VERSION — Full 4-trade review council.
Analyses all 4 closed paper trades: AHL (WIN), NHPC (LOSS), PPCL (LOSS), CHL (LOSS).

CHEAP MULTI-MODEL STACK:
  Stage -1 : Qwen3.5-flash    (qwen/qwen3.5-flash-02-23)    — hindsight audit
  Stage 0a : GPT-4.1-nano     (openai/gpt-4.1-nano)         — agenda draft
  Stage 0b : Qwen3.5-flash    (qwen/qwen3.5-flash-02-23)    — agenda review
  Stage 1  : DeepSeek v3.1    (deepseek/deepseek-chat-v3.1) — quant/contrarian
  Stage 2  : Grok-4.1-fast    (x-ai/grok-4.1-fast)          — sentiment/news
  Stage 3  : Qwen3.5-flash    (qwen/qwen3.5-flash-02-23)    — technical
  Stage 4  : GPT-4.1-nano     (openai/gpt-4.1-nano)         — macro
  Stage 5  : Gemini Flash      (native SDK — FREE)           — red team
  Stage 6  : Claude-3-haiku   (anthropic/claude-3-haiku)    — chairman

DESIGN:
  - Always writes to DB (no flag needed)
  - DB tag: TEST-FULL-4TRADES-2026-04 (isolated from production)
  - Full prompt + response logging — zero truncation
  - Resume: skips agenda items already written to DB on rerun
  - Agenda explicitly focused on all 4 trades

RUN:
  python -m analysis.monthly_council_test            # full run
  python -m analysis.monthly_council_test --dry-run  # structure test, no API, no DB
  python -m analysis.monthly_council_test --prompt   # print all prompts, no API, no DB
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
from AI import ask_gemini_text
from sheets import run_raw_sql, write_row, upsert_row, get_setting

NST = ZoneInfo("Asia/Kathmandu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────
M_QWEN     = "qwen/qwen3.5-flash-02-23"
M_GPT      = "openai/gpt-4.1-nano"
M_DEEPSEEK = "deepseek/deepseek-chat-v3.1"
M_GROK     = "x-ai/grok-4.1-fast"
M_HAIKU    = "anthropic/claude-3-haiku"
# Stage 5 → ask_gemini_text() native SDK (free)

RUN_MONTH_TAG  = "TEST-FULL-4TRADES-2026-04"
MAX_DATA_TOK   = 3000   # increased — 4 trades need more context
MAX_DISC_TOK   = 800
MAX_CHAIR_TOK  = 2000
MAX_RED_TOK    = 1000
MAX_AGENDA_TOK = 600
DATA_LOOKBACK  = 45     # 45 days to ensure all 4 trades captured

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CALL WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _or(model: str, messages: list, max_tokens: int, ctx: str,
        temp: float = 0.3) -> Optional[str]:
    log.info("[API] %s (%s)", model, ctx)
    result = _call(model, messages, max_tokens, temp, ctx)
    if result:
        log.info("[API] %s → %d chars", model.split("/")[-1], len(result))
    else:
        log.warning("[API] %s returned None", model.split("/")[-1])
    return result


def _gemini(prompt: str, system: str, ctx: str) -> Optional[str]:
    log.info("[API] Gemini Flash native SDK (%s)", ctx)
    result = ask_gemini_text(prompt=prompt, system=system, context=ctx)
    if result:
        log.info("[API] Gemini → %d chars", len(result))
    else:
        log.warning("[API] Gemini returned None")
    return result


def _msgs(system: str, user: str) -> list:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — JSON HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _parse(raw: str, ctx: str = "") -> Optional[dict]:
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
        log.info("[JSON] %s OK", ctx)
        return parsed
    except json.JSONDecodeError as e:
        log.error("[JSON] %s failed: %s\nFull raw response:\n%s", ctx, e, raw)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AGENDA FLATTENER
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten(items: list) -> list[str]:
    """
    Ensure all agenda items are plain strings.
    GPT-nano sometimes returns {item:..., details:...} dicts — extract text.
    """
    result = []
    for item in items:
        if isinstance(item, dict):
            text = (
                item.get("item") or item.get("title") or
                item.get("agenda_item") or item.get("topic") or
                next((v for v in item.values() if isinstance(v, str)), None) or
                json.dumps(item, ensure_ascii=False)
            )
            result.append(str(text).strip())
        else:
            result.append(str(item).strip())
    return [i for i in result if i]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def _cutoff() -> str:
    return (datetime.now(NST) - timedelta(days=DATA_LOOKBACK)).strftime("%Y-%m-%d")


def _load(query: str, params=None, label: str = "") -> list[dict]:
    try:
        rows = run_raw_sql(query, params) or []
        log.info("[DB] %s: %d rows", label, len(rows))
        return rows
    except Exception as e:
        log.error("[DB] %s failed: %s", label, e)
        return []


def _load_positions() -> list[dict]:
    try:
        paper = get_setting("PAPER_MODE", "true").lower() == "true"
        if paper:
            rows = run_raw_sql(
                "SELECT symbol, wacc, total_shares, first_buy_date, status "
                "FROM paper_portfolio ORDER BY id DESC LIMIT 10"
            ) or []
        else:
            rows = run_raw_sql(
                "SELECT symbol, entry_price, shares, entry_date, status "
                "FROM portfolio ORDER BY id DESC LIMIT 10"
            ) or []
        log.info("[DB] positions: %d rows", len(rows))
        return rows
    except Exception as e:
        log.error("[DB] positions failed: %s", e)
        return []


def _load_all() -> dict:
    c = _cutoff()
    log.info("[DATA] Loading context (cutoff=%s, lookback=%dd)...", c, DATA_LOOKBACK)
    return {
        # All closed trades — extended lookback to catch all 4
        "trade_journal":  _load(
            "SELECT * FROM trade_journal WHERE entry_date >= %s ORDER BY entry_date ASC",
            (c,), "trade_journal"),
        "daily_context":  _load(
            "SELECT * FROM daily_context_log WHERE date >= %s ORDER BY date DESC LIMIT 30",
            (c,), "daily_context"),
        "gate_misses":    _load(
            "SELECT * FROM gate_misses WHERE date >= %s ORDER BY date DESC LIMIT 50",
            (c,), "gate_misses"),
        "nrb":            _load(
            "SELECT * FROM nrb_monthly ORDER BY id DESC LIMIT 3",
            label="nrb_monthly"),
        "audit":          _load(
            "SELECT * FROM claude_audit ORDER BY id DESC LIMIT 4",
            label="claude_audit"),
        "lessons":        _load(
            "SELECT * FROM learning_hub WHERE active = 'true' ORDER BY id DESC LIMIT 20",
            label="learning_hub"),
        "positions":      _load_positions(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DATA CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _trim(rows: list, max_tok: int) -> tuple[list, int]:
    if not rows:
        return rows, 0
    mc = max_tok * 4
    t, om = list(rows), 0
    while t and len(json.dumps(t, ensure_ascii=False, default=str)) > mc:
        t.pop(0); om += 1
    return t, om


def _build_context(data: dict) -> tuple[str, dict]:
    # Trade journal gets bigger budget — it's the focus
    trade_budget   = MAX_DATA_TOK // 3
    other_budget   = MAX_DATA_TOK // 8

    def _sec(title, rows, om):
        note  = f" [{om} oldest omitted]" if om else ""
        lines = [f"=== {title}{note} ==="]
        for r in rows:
            lines.append(json.dumps(
                {k: v for k, v in r.items() if v is not None},
                ensure_ascii=False, default=str,
            ))
        return "\n".join(lines)

    parts, counts = [], {}

    # Trade journal — primary focus, full budget
    tj, tj_om = _trim(data.get("trade_journal", []), trade_budget)
    counts["trade_journal"] = len(tj)
    if tj:
        parts.append(_sec("CLOSED TRADES — ALL 4 PAPER TRADES (primary review focus)", tj, tj_om))

    # Others with smaller budget
    for title, key, budget in [
        ("DAILY CONTEXT (30d)",   "daily_context", other_budget),
        ("GATE MISSES (30d)",     "gate_misses",   other_budget),
        ("NRB MONTHLY (3)",       "nrb",           other_budget),
        ("CLAUDE AUDIT (4wk)",    "audit",         other_budget),
        ("LESSONS (top 20)",      "lessons",       other_budget),
    ]:
        rows, om = _trim(data.get(key, []), budget)
        counts[key] = len(rows)
        if rows:
            parts.append(_sec(title, rows, om))

    ctx = "\n\n".join(parts)
    counts["est_tokens"] = len(ctx) // 4
    log.info("[DATA] Context built: %d tokens | trades=%d dc=%d gm=%d nrb=%d lessons=%d",
             counts["est_tokens"], counts["trade_journal"], counts.get("daily_context", 0),
             counts.get("gate_misses", 0), counts.get("nrb", 0), counts.get("lessons", 0))
    return ctx, counts


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PERSONAS & DISCUSSION PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_J = "\n\nOutput ONLY valid JSON. No markdown fences. No preamble. No explanation."

_PERSONAS = {
    "deepseek": (
        "You are a rigorous quantitative analyst specialising in NEPSE microstructure. "
        "Challenge consensus with mathematical evidence. Deeply sceptical of narrative. "
        "Focus: trade statistics, risk-adjusted returns, signal quality, drawdown analysis, "
        "liquidity conditions, mean-reversion." + _J
    ),
    "grok": (
        "You are a NEPSE sentiment and macro news analyst. "
        "Focus: political risk, market sentiment, remittance flows, retail behaviour, "
        "NRB policy signals, news context around each trade's entry/exit." + _J
    ),
    "qwen": (
        "You are a technical analysis specialist for NEPSE. "
        "Focus: price patterns, volume profile, MACD/RSI/OBV/Bollinger signals, "
        "sector rotation, DPR proximity, entry/exit timing quality." + _J
    ),
    "gpt": (
        "You are a macro analyst covering Nepal's economy and NEPSE. "
        "Focus: NRB monetary policy, inflation, forex reserves, remittances, "
        "banking sector health, BOP impact on market conditions during each trade." + _J
    ),
}

_DISC_SCHEMA = (
    "Output ONLY valid JSON with exactly these keys:\n"
    "{\n"
    '  "direction": "Bullish" | "Bearish" | "Neutral",\n'
    '  "confidence": <integer 0-100>,\n'
    '  "key_driver": "<string, max 150 chars>",\n'
    '  "risk_factor": "<string, max 150 chars>"\n'
    "}"
)


def _disc_prompt(persona: str, agenda_item: str, idx: int, n: int,
                 ctx: str, prior: list, positions: list) -> tuple[str, str]:
    system = f"{_PERSONAS.get(persona, 'You are a NEPSE analyst.' + _J)}\n\n{_DISC_SCHEMA}"

    prior_block = ""
    if prior:
        lines = [
            f"[{r['label']}] direction={r['direction']}, "
            f"confidence={r['confidence']}%, "
            f"key_driver={r['key_driver']}"
            for r in prior
        ]
        last = prior[-1]
        adversarial = (
            f"\n\nADVERSARIAL INSTRUCTION:\n"
            f"The previous analyst ({last['label']}) claimed: \"{last['key_driver']}\"\n"
            f"You MUST directly engage with this claim — agree with evidence, refine it, "
            f"or rebut it using data from the trades and market context provided."
        )
        prior_block = "\n\nPRIOR COUNCIL RESPONSES:\n" + "\n".join(lines) + adversarial

    pos_block = ""
    if positions:
        pos_block = (
            "\n\nPORTFOLIO CONTEXT:\n"
            + json.dumps(positions, ensure_ascii=False, default=str)
        )

    user = (
        f"NEPSE MONTHLY COUNCIL — 4-TRADE REVIEW\n"
        f"Agenda Item {idx} of {n}\n\n"
        f"TOPIC: {agenda_item}\n\n"
        f"MARKET & TRADE DATA (last {DATA_LOOKBACK} days):\n"
        f"{ctx}\n"
        f"{pos_block}"
        f"{prior_block}\n\n"
        f"Analyse this agenda item using the trade data above. Output your JSON assessment."
    )
    return system, user


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — DB WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _wlog(entry: dict) -> None:
    try:
        write_row("monthly_council_log", entry)
        log.info("[DB] monthly_council_log — stage=%s", entry.get("stage", "?"))
    except Exception as e:
        log.error("[DB FAIL] monthly_council_log: %s", e)


def _wagenda(run_month: str, items: list[str]) -> None:
    try:
        for i, item in enumerate(items, 1):
            upsert_row("monthly_council_agenda", {
                "run_month":   run_month,
                "item_number": str(i),
                "agenda_item": item,
                "approved_by": "council_test_qwen",
            }, conflict_columns=["run_month", "item_number"])
        log.info("[DB] agenda — %d items written", len(items))
    except Exception as e:
        log.error("[DB FAIL] agenda: %s", e)


def _wchecklist(run_month: str, ch: dict) -> None:
    try:
        upsert_row("monthly_council_checklist", {
            "run_month":        run_month,
            "stop_trigger":     ch.get("stop_trigger", ""),
            "go_trigger":       ch.get("go_trigger", ""),
            "noise_items":      json.dumps(ch.get("noise_items", [])),
            "confidence_score": str(ch.get("nepse_confidence_score", 50)),
        }, conflict_columns=["run_month"])
        log.info("[DB] checklist written")
    except Exception as e:
        log.error("[DB FAIL] checklist: %s", e)


def _woverride(run_month: str, conf: int, mstate: str, ch: dict) -> None:
    blocked  = conf <= 35 and mstate in ("BEAR", "CRISIS")
    cautious = conf <= 50
    log.info("[OVERRIDE] confidence=%d buy_blocked=%s buy_cautious=%s",
             conf, blocked, cautious)
    try:
        upsert_row("monthly_override", {
            "run_month":    run_month,
            "confidence":   str(conf),
            "buy_blocked":  str(blocked).lower(),
            "buy_cautious": str(cautious).lower(),
            "go_trigger":   ch.get("go_trigger", ""),
            "stop_trigger": ch.get("stop_trigger", ""),
        }, conflict_columns=["run_month"])
        log.info("[DB] monthly_override written")
    except Exception as e:
        log.error("[DB FAIL] override: %s", e)


def _wlessons(run_month: str, ch: dict, now_str: str) -> int:
    count = 0
    for lesson in ch.get("lessons", []):
        if not lesson or len(lesson.strip()) <= 10:
            continue
        log.info("[LESSON] %s", lesson.strip())
        try:
            write_row("learning_hub", {
                "week":        run_month,
                "lesson_type": "COUNCIL_TEST",
                "lesson":      lesson.strip(),
                "active":      "true",
                "source":      "monthly_council_test",
                "inserted_at": now_str,
            })
            log.info("[DB] learning_hub lesson written")
            count += 1
        except Exception as e:
            log.error("[DB FAIL] lesson: %s", e)
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RESUME HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_completed_items(run_month: str) -> set[str]:
    """Return set of agenda_item strings already written to monthly_council_log."""
    try:
        rows = run_raw_sql(
            "SELECT DISTINCT agenda_item FROM monthly_council_log "
            "WHERE run_month = %s AND agenda_item != 'ALL'",
            (run_month,),
        ) or []
        completed = {r["agenda_item"] for r in rows if r.get("agenda_item")}
        if completed:
            log.info("[RESUME] Found %d completed items in DB — will skip them", len(completed))
            for item in completed:
                log.info("[RESUME]   SKIP: %s", item[:80])
        return completed
    except Exception as e:
        log.error("[RESUME] Failed to check completed items: %s", e)
        return set()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _transcript(items: list[str], dlog: list) -> str:
    lines = ["FULL COUNCIL DISCUSSION TRANSCRIPT", "=" * 60]
    for item in items:
        lines.append(f"\n── AGENDA ITEM: {item} ──")
        for e in dlog:
            if e.get("agenda_item") == item:
                lines.append(
                    f"  [{e['label']}] {e['direction']} ({e['confidence']}%)\n"
                    f"    KEY DRIVER:  {e['key_driver']}\n"
                    f"    RISK FACTOR: {e['risk_factor']}"
                )
    return "\n".join(lines)


def _ds(confs: list) -> float:
    if len(confs) < 2:
        return 0.0
    try:
        return statistics.stdev(confs)
    except Exception:
        return 0.0


def _telegram(run_month: str, conf: int, items: list, lessons: int, ch: dict) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("[TELEGRAM] Not configured — skipping")
        return
    try:
        import requests
        verdict = ch.get("system_verdict", "?")
        emoji   = {"PROCEED": "🟢", "CAUTION": "🟡", "HALT": "🔴"}.get(verdict, "⚪")
        text = (
            f"🧪 *COUNCIL TEST — {run_month}*\n\n"
            f"{emoji} *{verdict}* | Confidence: {conf}/100\n"
            f"Agenda: {len(items)} items | Lessons: {lessons}\n\n"
            f"📈 Go:   {ch.get('go_trigger','?')[:100]}\n"
            f"🛑 Stop: {ch.get('stop_trigger','?')[:100]}\n\n"
            f"Assessment: {ch.get('market_assessment','?')[:200]}"
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
# SECTION 10 — MAIN RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False, print_prompts: bool = False) -> None:
    """
    dry_run=True  → skip all API calls, no DB writes
    Always writes to DB unless dry_run.
    Full prompt + response logging — no truncation.
    """
    now_nst   = datetime.now(NST)
    run_month = RUN_MONTH_TAG
    now_str   = now_nst.strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 65)
    log.info("NEPSE MONTHLY COUNCIL TEST — %s", run_month)
    log.info("Focus: All 4 closed paper trades (AHL WIN, NHPC/PPCL/CHL LOSS)")
    log.info("Stack: DeepSeek + Grok + Qwen + GPT-nano + Gemini(free) + Haiku")
    log.info("Mode:  dry_run=%s | write_db=%s", dry_run, not dry_run)
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    data      = _load_all()
    ctx, cnts = _build_context(data)
    positions = data["positions"]
    dc        = data["daily_context"]

    # ── Resume: find completed items ──────────────────────────────────────────
    completed_items: set[str] = set()
    if not dry_run:
        completed_items = _get_completed_items(run_month)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE -1: Qwen hindsight audit
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("STAGE -1: Qwen hindsight audit")
    log.info("=" * 65)

    audit_sys = (
        "You are a NEPSE paper trading system auditor. "
        "You have 4 closed trades to review: AHL (WIN +4.34%), NHPC (LOSS), PPCL (LOSS), CHL (LOSS). "
        "Summarise what worked and what failed across all 4 trades in ≤250 words. "
        "Focus on: entry timing, market state alignment, confidence scores, signal quality, "
        "position sizing, stop loss adherence. Plain text only — no JSON."
    )
    audit_user = (
        f"AUDIT — {run_month}\n\n"
        f"TRADE DATA:\n{ctx}\n\n"
        f"Review all 4 closed trades. What patterns explain the 1 win and 3 losses?"
    )

    log.info("[STAGE -1] System prompt:\n%s", audit_sys)
    log.info("[STAGE -1] User prompt:\n%s", audit_user)

    if dry_run:
        audit_text = "[DRY RUN — audit skipped]"
        log.info("[DRY RUN] Stage -1 skipped")
    else:
        audit_text = _or(M_QWEN, _msgs(audit_sys, audit_user), MAX_AGENDA_TOK, "audit") or ""
        log.info("[STAGE -1] Full response:\n%s", audit_text)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0a: GPT-nano agenda draft
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("STAGE 0a: GPT-nano agenda draft")
    log.info("=" * 65)

    draft_sys = (
        "You are a NEPSE monthly council agenda coordinator. "
        "Draft exactly 5 specific, actionable agenda items for council deliberation. "
        "The council must review 4 closed paper trades: "
        "AHL (WIN +4.34%), NHPC (LOSS -820.94 NPR), PPCL (LOSS -311.99 NPR), CHL (LOSS -2396.20 NPR). "
        "Agenda items must be specific — reference actual trade data, signals, and patterns. "
        "Cover: trade execution quality, signal failures, risk management, system improvements, "
        "macro context. "
        "Output ONLY valid JSON: {\"agenda_items\": [\"item1\", \"item2\", ...]}"
        + _J
    )
    draft_user = (
        f"COUNCIL AGENDA DRAFT — {run_month}\n\n"
        f"AUDIT SUMMARY:\n{audit_text}\n\n"
        f"FULL TRADE & MARKET DATA:\n{ctx}\n\n"
        f"Draft 5 specific agenda items for council review of all 4 trades."
    )

    log.info("[STAGE 0a] System prompt:\n%s", draft_sys)
    log.info("[STAGE 0a] User prompt:\n%s", draft_user)

    if dry_run:
        log.info("[DRY RUN] Stage 0a skipped")
        proposed = [
            "AHL WIN vs NHPC/PPCL/CHL LOSS: What signals differentiated the winner from the losers? [DRY RUN]",
            "CHL: Largest loss (-2396 NPR, 100 shares). Was position sizing proportionate to confidence? [DRY RUN]",
            "Market state was BEAR for all 4 trades. Should BEAR state block entries entirely? [DRY RUN]",
            "RSI lesson #16 (-4.81% drag): Were any of the 3 losses RSI-influenced entries? [DRY RUN]",
            "Gate misses (CORBLP/CITPO/CFCLPO NO_LTP): Does the filter engine need liquidity pre-screening? [DRY RUN]",
        ]
    else:
        raw      = _or(M_GPT, _msgs(draft_sys, draft_user), MAX_AGENDA_TOK, "0a_draft")
        log.info("[STAGE 0a] Full response:\n%s", raw)
        parsed   = _parse(raw, "0a_draft")
        proposed = _flatten(parsed.get("agenda_items", []) if parsed else [])
        if not proposed:
            log.warning("[STAGE 0a] No items returned — using fallback")
            proposed = [
                "Review AHL WIN vs 3 losses: signal quality comparison across all 4 trades",
                "CHL loss analysis: was -2396 NPR loss preventable given entry signals?",
                "BEAR market state and trade entry: should market state gate be stricter?",
                "RSI lesson applicability: did RSI influence any of the 3 losing trades?",
                "Position sizing review: was allocation proportionate to confidence scores?",
            ]
        log.info("[STAGE 0a] Proposed items (%d):", len(proposed))
        for i, item in enumerate(proposed, 1):
            log.info("  %d. %s", i, item)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 0b: Qwen agenda review
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("STAGE 0b: Qwen agenda review")
    log.info("=" * 65)

    review_sys = (
        "You are reviewing a NEPSE council agenda focused on 4 closed paper trades. "
        "Approve, reorder, or improve the proposed agenda items. "
        "Ensure items are specific and actionable — not generic. "
        "Output ONLY valid JSON: {\"approved_agenda\": [\"item1\", ...]}"
        + _J
    )
    review_user = (
        f"PROPOSED AGENDA:\n{json.dumps(proposed, ensure_ascii=False)}\n\n"
        f"TRADE CONTEXT:\n{ctx[:1000]}\n\n"
        f"Review and approve as plain string list in JSON."
    )

    log.info("[STAGE 0b] System prompt:\n%s", review_sys)
    log.info("[STAGE 0b] User prompt:\n%s", review_user)

    if dry_run:
        log.info("[DRY RUN] Stage 0b skipped")
        approved = proposed
    else:
        raw      = _or(M_QWEN, _msgs(review_sys, review_user), MAX_AGENDA_TOK, "0b_review")
        log.info("[STAGE 0b] Full response:\n%s", raw)
        parsed   = _parse(raw, "0b_review")
        approved = _flatten(parsed.get("approved_agenda", proposed) if parsed else proposed)
        if not approved:
            approved = proposed
        log.info("[STAGE 0b] Approved items (%d):", len(approved))
        for i, item in enumerate(approved, 1):
            log.info("  %d. %s", i, item)
        _wagenda(run_month, approved)

    n = len(approved)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGES 1-4: Per-agenda-item discussion (4 models, adversarial)
    # ═════════════════════════════════════════════════════════════════════════
    _DISC = [
        ("deepseek", "DeepSeek_Quant",  "stage_1_deepseek", M_DEEPSEEK),
        ("grok",     "Grok_Sentiment",  "stage_2_grok",     M_GROK),
        ("qwen",     "Qwen_Tech",       "stage_3_qwen",     M_QWEN),
        ("gpt",      "GPT_Macro",       "stage_4_gpt",      M_GPT),
    ]

    dlog:   list[dict] = []
    ds_map: dict       = {}

    for idx, agenda_item in enumerate(approved, start=1):
        agenda_item = str(agenda_item).strip()

        # ── Resume: skip if already completed ────────────────────────────────
        if agenda_item in completed_items:
            log.info("=" * 65)
            log.info("SKIP item %d/%d (already in DB): %s", idx, n, agenda_item[:80])
            log.info("=" * 65)
            # Still need to populate dlog for transcript building
            dlog.append({
                "label": "RESUMED", "stage": "skipped",
                "agenda_item": agenda_item,
                "direction": "N/A", "confidence": 0,
                "key_driver": "[resumed from previous run]",
                "risk_factor": "",
            })
            ds_map[agenda_item[:80]] = 0.0
            continue

        log.info("=" * 65)
        log.info("DISCUSSION %d/%d: %s", idx, n, agenda_item)
        log.info("=" * 65)

        prior:  list[dict] = []
        confs:  list[int]  = []

        for persona, label, stage_key, model in _DISC:
            system, user = _disc_prompt(persona, agenda_item, idx, n, ctx, prior, positions)

            log.info("── [%s] ─────────────────────────────────────────────", stage_key)
            log.info("[%s] System prompt:\n%s", stage_key, system)
            log.info("[%s] User prompt:\n%s", stage_key, user)

            if print_prompts:
                print(f"\n{'='*65}\n[{stage_key}] SYSTEM:\n{system}\nUSER:\n{user}\n")

            if dry_run:
                log.info("[DRY RUN] %s skipped", stage_key)
                direction, confidence = "Neutral", 50
                key_driver  = f"[DRY RUN — {label}]"
                risk_factor = "[DRY RUN]"
                raw = "{}"
            else:
                raw    = _or(model, _msgs(system, user), MAX_DISC_TOK, stage_key)
                log.info("[%s] Full response:\n%s", stage_key, raw)
                parsed = _parse(raw, stage_key) if raw else None
                direction   = str(parsed.get("direction",   "Neutral")) if parsed else "Neutral"
                confidence  = int(parsed.get("confidence",  50))        if parsed else 50
                key_driver  = str(parsed.get("key_driver",  ""))        if parsed else ""
                risk_factor = str(parsed.get("risk_factor", ""))        if parsed else ""
                log.info("[%s] Parsed: direction=%s confidence=%d",
                         stage_key, direction, confidence)
                log.info("[%s] key_driver:  %s", stage_key, key_driver)
                log.info("[%s] risk_factor: %s", stage_key, risk_factor)

            entry = {
                "label":       label,
                "stage":       stage_key,
                "agenda_item": agenda_item,
                "direction":   direction,
                "confidence":  confidence,
                "key_driver":  key_driver,
                "risk_factor": risk_factor,
            }
            prior.append(entry)
            confs.append(confidence)
            dlog.append(entry)

            if not dry_run:
                _wlog({
                    "run_month":     run_month,
                    "stage":         stage_key,
                    "agenda_item":   agenda_item,
                    "model":         model,
                    "direction":     direction,
                    "confidence":    str(confidence),
                    "key_driver":    key_driver,
                    "risk_factor":   risk_factor,
                    "full_response": raw or "",
                    "inserted_at":   now_str,
                })

        score = _ds(confs)
        ds_map[str(agenda_item)[:80]] = round(score, 2)
        log.info("[DISCUSSION %d] Disagreement score: %.1f | confidences: %s",
                 idx, score, confs)
        if score > 20:
            log.info("  ✅ HIGH disagreement — genuine debate")
        elif score < 8 and not dry_run:
            log.warning("  ⚠️  LOW disagreement (%.1f) — possible groupthink", score)

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 5: Gemini red team (FREE)
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("STAGE 5: Gemini Flash red team")
    log.info("=" * 65)

    ts = _transcript(approved, dlog)
    log.info("[STAGE 5] Transcript (%d chars):\n%s", len(ts), ts)

    rt_sys = (
        "You are the NEPSE Monthly Council Red Team analyst — independent, not part of the discussion. "
        "You have read the full discussion transcript of 4 closed paper trades. "
        "Your job: find the two sharpest conflicting viewpoints across all analysts. "
        "Assess which conflict represents the highest risk to future trading. "
        "Output ONLY valid JSON:\n"
        "{\n"
        '  "conflict_1": "<string — describe first conflict>",\n'
        '  "conflict_2": "<string — describe second conflict>",\n'
        '  "highest_risk": {"recommended_action": "<string>", "rationale": "<string>"},\n'
        '  "red_team_verdict": "<string, max 250 chars>"\n'
        "}"
        + _J
    )
    rt_user = (
        f"RED TEAM REVIEW — {run_month}\n\n"
        f"{ts}\n\n"
        f"PORTFOLIO:\n{json.dumps(positions, ensure_ascii=False, default=str)}\n\n"
        f"Identify the two sharpest conflicts. Output JSON."
    )

    log.info("[STAGE 5] System prompt:\n%s", rt_sys)
    log.info("[STAGE 5] User prompt:\n%s", rt_user)

    if dry_run:
        log.info("[DRY RUN] Stage 5 skipped")
        rt_result = {
            "highest_risk": {"recommended_action": "MONITOR", "rationale": "[DRY RUN]"},
            "red_team_verdict": "[DRY RUN]"
        }
    else:
        rt_raw    = _gemini(rt_user, rt_sys, "5_redteam")
        log.info("[STAGE 5] Full response:\n%s", rt_raw)
        rt_result = _parse(rt_raw, "5_redteam") or {}
        log.info("[STAGE 5] verdict: %s", rt_result.get("red_team_verdict", "?"))
        log.info("[STAGE 5] action:  %s",
                 rt_result.get("highest_risk", {}).get("recommended_action", "?"))
        _wlog({
            "run_month":     run_month,
            "stage":         "stage_5_redteam",
            "agenda_item":   "ALL",
            "model":         "gemini_flash_native",
            "full_response": rt_raw or "",
            "inserted_at":   now_str,
        })

    # ═════════════════════════════════════════════════════════════════════════
    # STAGE 6: Claude Haiku chairman synthesis
    # ═════════════════════════════════════════════════════════════════════════
    log.info("=" * 65)
    log.info("STAGE 6: Claude Haiku chairman synthesis")
    log.info("=" * 65)

    ch_sys = (
        "You are the Chairman of the NEPSE Monthly Council. "
        "You have reviewed 4 closed paper trades (AHL WIN, NHPC/PPCL/CHL LOSS) "
        "through a full multi-analyst deliberation. "
        "Synthesise into final actionable guidance for the trading system. "
        "Be specific — reference actual trade outcomes and signal patterns. "
        "Output ONLY valid JSON:\n"
        "{\n"
        '  "nepse_confidence_score": <integer 0-100>,\n'
        '  "market_assessment": "<string, max 300 chars>",\n'
        '  "go_trigger": "<string, max 150 chars — specific conditions to enter next trade>",\n'
        '  "stop_trigger": "<string, max 150 chars — specific conditions to block entry>",\n'
        '  "noise_items": ["<item1>", "<item2>", "<item3>"],\n'
        '  "lessons": [\n'
        '    "<lesson1 — specific, actionable>",\n'
        '    "<lesson2>",\n'
        '    "<lesson3>",\n'
        '    "<lesson4>",\n'
        '    "<lesson5>"\n'
        '  ],\n'
        '  "system_verdict": "PROCEED" | "CAUTION" | "HALT"\n'
        "}"
        + _J
    )
    ch_user = (
        f"CHAIRMAN SYNTHESIS — {run_month}\n\n"
        f"AGENDA ITEMS:\n{json.dumps(approved, ensure_ascii=False)}\n\n"
        f"DISAGREEMENT SCORES (stdev of confidence per item):\n{json.dumps(ds_map)}\n\n"
        f"RED TEAM VERDICT:\n{json.dumps(rt_result, ensure_ascii=False)}\n\n"
        f"{ts}\n\n"
        f"PORTFOLIO:\n{json.dumps(positions, ensure_ascii=False, default=str)}\n\n"
        f"Synthesise final council output as JSON."
    )

    log.info("[STAGE 6] System prompt:\n%s", ch_sys)
    log.info("[STAGE 6] User prompt:\n%s", ch_user)

    if dry_run:
        log.info("[DRY RUN] Stage 6 skipped")
        chairman   = {
            "nepse_confidence_score": 50,
            "market_assessment":      "[DRY RUN]",
            "go_trigger":             "[DRY RUN]",
            "stop_trigger":           "[DRY RUN]",
            "noise_items":            [],
            "lessons":                [],
            "system_verdict":         "CAUTION",
        }
        conf_score = 50
    else:
        ch_raw   = _or(M_HAIKU, _msgs(ch_sys, ch_user), MAX_CHAIR_TOK, "6_chairman")
        log.info("[STAGE 6] Full response:\n%s", ch_raw)
        chairman   = _parse(ch_raw, "6_chairman") or {}
        conf_score = int(chairman.get("nepse_confidence_score", 50))

        log.info("[STAGE 6] FINAL COUNCIL OUTPUT:")
        log.info("  Verdict:    %s", chairman.get("system_verdict", "?"))
        log.info("  Confidence: %d/100", conf_score)
        log.info("  Assessment: %s", chairman.get("market_assessment", "?"))
        log.info("  Go trigger: %s", chairman.get("go_trigger", "?"))
        log.info("  Stop:       %s", chairman.get("stop_trigger", "?"))
        log.info("  Noise:      %s", chairman.get("noise_items", []))
        for i, lesson in enumerate(chairman.get("lessons", []), 1):
            log.info("  Lesson %d:  %s", i, lesson)

        _wlog({
            "run_month":     run_month,
            "stage":         "stage_6_chairman",
            "agenda_item":   "ALL",
            "model":         M_HAIKU,
            "confidence":    str(conf_score),
            "full_response": ch_raw or "",
            "inserted_at":   now_str,
        })
        _wchecklist(run_month, chairman)
        mstate = dc[0].get("market_state", "SIDEWAYS") if dc else "SIDEWAYS"
        _woverride(run_month, conf_score, mstate, chairman)

    lessons_written = 0
    if not dry_run:
        lessons_written = _wlessons(run_month, chairman, now_str)

    # ── Stage 7: Telegram ─────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("STAGE 7: Telegram notification")
    if not dry_run:
        _telegram(run_month, conf_score, approved, lessons_written, chairman)
    else:
        log.info("[DRY RUN] Telegram skipped")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("COUNCIL TEST COMPLETE — %s", run_month)
    log.info("  Verdict:    %s", chairman.get("system_verdict", "?"))
    log.info("  Confidence: %d/100", conf_score)
    log.info("  Lessons:    %d written to DB", lessons_written)
    log.info("  DB tag:     %s", run_month)
    log.info("=" * 65)

    if dry_run:
        print(f"\n{'='*65}")
        print(f"[DRY RUN SUMMARY]")
        print(f"  run_month: {run_month}")
        print(f"  Data:      {cnts}")
        print(f"  Trades:    {cnts.get('trade_journal',0)} (all 4 should appear)")
        print(f"  Stack:     DeepSeek + Grok + Qwen + GPT-nano + Gemini(free) + Haiku")
        print(f"  Writes:    NONE (dry run)")
        print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Monthly Council TEST — Full 4-trade review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m analysis.monthly_council_test            # full run, writes to DB
  python -m analysis.monthly_council_test --dry-run  # no API, no DB writes
  python -m analysis.monthly_council_test --prompt   # print all prompts
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all API calls, no DB writes")
    parser.add_argument("--prompt",  action="store_true",
                        help="Print all prompts to stdout")
    parser.add_argument("--force",   action="store_true",
                        help="No-op (always runs)")
    args = parser.parse_args()

    from log_config import attach_file_handler
    attach_file_handler(__name__)

    run(dry_run=args.dry_run, print_prompts=args.prompt)


if __name__ == "__main__":
    main()