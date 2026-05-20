"""
agent/wait_evaluator.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — WAIT Condition Evaluator

Given a parsed condition JSON + fresh indicators + market state, evaluates
whether a WAIT condition is met.  All boolean arithmetic is in Python.
LLM is called only for ambiguous requirements and for final Claude escalation.

Public API
----------
    evaluate_wait(
        wait_row, parsed, indicators, market_state,
        portfolio, cycle_ts, step, shadow_mode
    ) -> dict

Return dict shape
-----------------
    {
      "outcome":   "SKIP" | "ESCALATED" | "ERROR",
      "reason":    str,
      "decision":  str | None,   # Claude decision if ESCALATED
      "symbol":    str,
    }

Decision flow
-------------
  a. Python pre-filter — evaluate "indicator" and "market" requirements.
     Any failing check → SKIP immediately.
  b. Ambiguous requirements — call ask_free() per ambiguous item.
     Any "met=false" → SKIP.  LLM failure → SKIP (safe default).
  c. Telegram flag — send admin alert once per ambiguous requirement per
     wait row (guarded by a flag in the parsed JSON).
  d. last_reviewed_date check — if reviewed today → SKIP ("reviewed_today").
  e. Escalate to Claude via agent_tools.escalate_to_claude().
     BUY       → update market_log action=BUY, clear wait fields
     NOW_AVOID → update market_log outcome=AVOIDED
     STILL_WAIT → update wait_condition (new text), clear parsed cache,
                  set last_reviewed_date = today, reset date clock

Rules
-----
- agent/agent_tools.py is NOT modified.
- All numeric evaluation in Python — LLM never does arithmetic.
- NULLIF pattern for casts already handled by get_fresh_indicators() which
  returns typed floats/ints via _safe_float/_safe_int.
- from config import NST for all timestamps.
- Fail silent on LLM errors — safe default is always SKIP.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from datetime import datetime

from config import NST

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Python pre-filter (indicator + market requirements)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and val != "" else default
    except (ValueError, TypeError):
        return default


def _evaluate_one(req: dict, indicators: dict, market_state: dict) -> tuple[bool, str]:
    """
    Evaluate a single "indicator" or "market" requirement against live data.
    Returns (passed: bool, detail: str).
    detail is used for the skip log line.
    """
    req_type = req.get("type")
    field    = req.get("field", "")
    op       = req.get("op",    "eq")
    value    = req.get("value")

    # ── Resolve actual value from data dicts ──────────────────────────────────
    if req_type == "indicator":
        actual = indicators.get(field)
    elif req_type == "market":
        actual = market_state.get(field)
    else:
        return True, ""   # unknown type — pass through, ambiguous handler will deal

    if actual is None:
        # Field missing from indicators/market_state — cannot confirm, fail safe
        return False, f"{field} not present in data"

    # ── String equality / set membership ──────────────────────────────────────
    if op == "eq":
        passed = str(actual).upper() == str(value).upper()
        return passed, f"{field}={actual} (need {op} {value})"

    if op == "neq":
        passed = str(actual).upper() != str(value).upper()
        return passed, f"{field}={actual} (need {op} {value})"

    if op == "in":
        if not isinstance(value, list):
            value = [value]
        passed = str(actual).upper() in [str(v).upper() for v in value]
        return passed, f"{field}={actual} (need in {value})"

    # ── Numeric comparisons ────────────────────────────────────────────────────
    try:
        actual_f = _safe_float(actual)
        value_f  = _safe_float(value)
    except Exception:
        return False, f"{field} numeric cast failed (actual={actual}, value={value})"

    if op == "gt":
        passed = actual_f > value_f
    elif op == "gte":
        passed = actual_f >= value_f
    elif op == "lt":
        passed = actual_f < value_f
    elif op == "lte":
        passed = actual_f <= value_f
    else:
        # Unknown op — fail safe
        log.warning("[evaluator] unknown op '%s' for field %s — failing safe", op, field)
        return False, f"unknown op {op}"

    return passed, f"{field}={actual_f:.4g} (need {op} {value_f:.4g})"


def _run_prefilter(
    symbol:       str,
    requirements: list,
    indicators:   dict,
    market_state: dict,
) -> tuple[bool, str]:
    """
    Evaluate all non-ambiguous requirements.
    Returns (all_passed: bool, fail_detail: str).
    """
    for req in requirements:
        t = req.get("type")
        if t not in ("indicator", "market"):
            continue   # ambiguous handled separately

        passed, detail = _evaluate_one(req, indicators, market_state)
        if not passed:
            log.info("[evaluator] %s SKIP — %s", symbol, detail)
            return False, detail

    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ambiguous requirement resolver (free LLM)
# ══════════════════════════════════════════════════════════════════════════════

_AMBIGUOUS_SYSTEM = """\
You are a trading condition checker. Given a condition description and the \
current market/indicator snapshot, decide if the condition is met.
Respond ONLY in valid JSON: {"met": true, "reason": "one sentence"}
"""


def _resolve_ambiguous(
    description:  str,
    indicators:   dict,
    market_state: dict,
    symbol:       str,
) -> tuple[bool, str]:
    """
    Ask ask_free() whether an ambiguous condition is met.
    Returns (met: bool, reason: str).
    On LLM failure returns (False, "llm_failed") — safe default.
    """
    from AI.openrouter import ask_free, _strip_fences

    snapshot = {
        "symbol":       symbol,
        "nepal_score":  market_state.get("nepal_score"),
        "geo_score":    market_state.get("geo_score"),
        "combined_geo": market_state.get("combined_geo"),
        "market_state": market_state.get("market_state"),
        "rsi_14":       indicators.get("rsi_14"),
        "macd_cross":   indicators.get("macd_cross"),
        "bb_signal":    indicators.get("bb_signal"),
        "tech_score":   indicators.get("tech_score"),
        "volume_ratio": indicators.get("volume_ratio"),
        "obv_trend":    indicators.get("obv_trend"),
    }

    prompt = (
        f"Condition to check: \"{description}\"\n\n"
        f"Current snapshot: {json.dumps(snapshot)}\n\n"
        f"Is the condition met? Respond only in JSON."
    )

    raw = ask_free(prompt=prompt, system=_AMBIGUOUS_SYSTEM, context="ambiguous_eval")
    if not raw:
        log.warning("[evaluator] ambiguous LLM returned None for %s — SKIP", symbol)
        return False, "llm_failed"

    try:
        result = json.loads(_strip_fences(raw))
        met    = bool(result.get("met", False))
        reason = str(result.get("reason", ""))
        log.info("[evaluator] ambiguous '%s' → met=%s reason=%s", description[:60], met, reason[:60])
        return met, reason
    except Exception as exc:
        log.warning("[evaluator] ambiguous JSON parse failed for %s: %s | raw: %.80s", symbol, exc, raw)
        return False, "parse_failed"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Telegram admin alert (once per ambiguous item per wait row)
# ══════════════════════════════════════════════════════════════════════════════

def _maybe_alert_ambiguous(
    symbol:        str,
    wait_log_id:   int,
    description:   str,
    parsed:        dict,
) -> dict:
    """
    Send a Telegram alert to admin once per (wait_log_id, description).
    Guards against repeat alerts by writing a flag into parsed["_alerted"].
    Returns the (potentially mutated) parsed dict.
    """
    alerted = parsed.setdefault("_alerted", [])
    key     = f"{wait_log_id}:{description[:60]}"
    if key in alerted:
        return parsed

    try:
        token   = os.getenv("TELEGRAM_ERROR_BOT", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            import requests
            text = (
                f"[WAIT EVALUATOR] Ambiguous condition detected\n"
                f"Symbol: {symbol} | Wait ID: {wait_log_id}\n"
                f"Condition: {description[:200]}\n"
                f"This cannot be evaluated automatically. Manual review may be needed."
            )
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
    except Exception as exc:
        log.warning("[evaluator] Telegram ambiguous alert failed: %s", exc)

    alerted.append(key)
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Claude escalation outcome handlers
# ══════════════════════════════════════════════════════════════════════════════

def _handle_still_wait(
    wait_log_id:       int,
    symbol:            str,
    new_condition:     str,
    reasoning:         str,
) -> None:
    """
    Claude said STILL_WAIT with a (possibly updated) condition.
    - Write new wait_condition text (may be same or refined).
    - NULL out wait_condition_parsed so condition_parser re-parses next cycle.
    - Set last_reviewed_date = today (NST).
    - Reset date clock: set market_log.date = today (5-day timeout restarts).
    Fails silently.
    """
    from sheets import run_raw_sql
    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    try:
        run_raw_sql(
            """
            UPDATE market_log
               SET wait_condition        = %s,
                   wait_condition_parsed = NULL,
                   last_reviewed_date    = %s,
                   date                  = %s,
                   reasoning             = %s
             WHERE id = %s
            """,
            (new_condition, today, today, reasoning[:2000], wait_log_id),
        )
        log.info("[evaluator] STILL_WAIT: updated market_log id=%d, reset date clock", wait_log_id)
    except Exception as exc:
        log.error("[evaluator] _handle_still_wait failed for id=%d: %s", wait_log_id, exc)


def _handle_now_avoid(
    wait_log_id: int,
    symbol:      str,
    reasoning:   str,
) -> None:
    """Claude said NOW_AVOID — mark outcome=AVOIDED."""
    from sheets import run_raw_sql
    try:
        run_raw_sql(
            "UPDATE market_log SET outcome = 'AVOIDED', reasoning = %s WHERE id = %s",
            (reasoning[:2000], wait_log_id),
        )
        log.info("[evaluator] NOW_AVOID: market_log id=%d outcome=AVOIDED", wait_log_id)
    except Exception as exc:
        log.error("[evaluator] _handle_now_avoid failed for id=%d: %s", wait_log_id, exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_wait(
    wait_row:     dict,
    parsed:       dict,
    indicators:   dict,
    market_state: dict,
    portfolio:    dict,
    cycle_ts:     str,
    step:         int,
    shadow_mode:  bool,
) -> dict:
    """
    Evaluate whether a WAIT condition is met and decide next action.

    Args:
        wait_row:     Row dict from get_open_waits() (id, symbol, wait_condition,
                      last_reviewed_date, reasoning, …)
        parsed:       Dict from get_parsed_condition() — never None (caller guards)
        indicators:   Dict from get_fresh_indicators()
        market_state: Dict from get_market_state()
        portfolio:    Dict from get_portfolio()
        cycle_ts:     NST timestamp string for this cycle
        step:         Loop iteration index (for agent_trace)
        shadow_mode:  If True, never call Claude

    Returns:
        {"outcome": "SKIP"|"ESCALATED"|"ERROR", "reason": str,
         "decision": str|None, "symbol": str}
    """
    symbol      = str(wait_row.get("symbol", "")).upper().strip()
    wait_log_id = int(wait_row.get("id", 0))
    today       = datetime.now(tz=NST).strftime("%Y-%m-%d")

    result: dict = {
        "outcome":  "SKIP",
        "reason":   "",
        "decision": None,
        "symbol":   symbol,
    }

    requirements = parsed.get("requirements", [])

    # ── a. Python pre-filter (indicator + market) ─────────────────────────────
    all_passed, fail_detail = _run_prefilter(symbol, requirements, indicators, market_state)
    if not all_passed:
        result["reason"] = f"prefilter_fail: {fail_detail}"
        return result

    # ── b. Ambiguous requirements ──────────────────────────────────────────────
    ambiguous = [r for r in requirements if r.get("type") == "ambiguous"]
    for amb in ambiguous:
        description = amb.get("description", "")
        if not description:
            continue

        # ── c. Telegram flag (once per requirement per wait row) ────────────
        parsed = _maybe_alert_ambiguous(symbol, wait_log_id, description, parsed)

        # Check if already resolved
        if amb.get("ambiguous_resolved"):
            log.info("[evaluator] %s ambiguous '%s...' already resolved=True — skipping re-check",
                     symbol, description[:40])
            continue

        met, amb_reason = _resolve_ambiguous(description, indicators, market_state, symbol)
        if not met:
            result["reason"] = f"ambiguous_not_met: {description[:80]} ({amb_reason})"
            return result

        # Mark resolved so we don't re-check this requirement every cycle
        amb["ambiguous_resolved"] = True
        log.info("[evaluator] %s ambiguous condition met: %s", symbol, description[:60])

    # ── d. last_reviewed_date check ────────────────────────────────────────────
    last_reviewed = str(wait_row.get("last_reviewed_date") or "")
    if last_reviewed == today:
        result["reason"] = "reviewed_today"
        log.info("[evaluator] %s already reviewed today (%s) — SKIP", symbol, today)
        return result

    # ── e. All checks passed — escalate to Claude ──────────────────────────────
    log.info("[evaluator] %s all checks passed — escalating to Claude (shadow=%s)", symbol, shadow_mode)

    try:
        from agent import agent_tools

        esc = agent_tools.escalate_to_claude(
            symbol         = symbol,
            wait_log_id    = wait_log_id,
            wait_condition = str(wait_row.get("wait_condition", "")),
            indicators     = indicators,
            market_state   = market_state,
            cycle_ts       = cycle_ts,
            step           = step,
            shadow_mode    = shadow_mode,
        )
    except Exception as exc:
        log.error("[evaluator] escalate_to_claude crashed for %s: %s", symbol, exc)
        result["outcome"] = "ERROR"
        result["reason"]  = f"escalation_exception: {exc}"
        return result

    decision  = esc.get("decision", "ERROR")
    reasoning = esc.get("reasoning", "")

    result["decision"] = decision
    result["outcome"]  = "ESCALATED"

    if shadow_mode:
        result["reason"] = "shadow_mode_escalation"
        return result

    # ── Handle Claude's verdict ────────────────────────────────────────────────
    if decision == "BUY":
        # agent_tools.escalate_to_claude already handled the DB write + Telegram
        result["reason"] = "condition_met_buy"

    elif decision == "NOW_AVOID":
        _handle_now_avoid(wait_log_id, symbol, reasoning)
        result["reason"] = "now_avoid"

    elif decision == "STILL_WAIT":
        # Claude may have updated the condition text in reasoning
        # Use the new condition from Claude response if present, else keep original
        new_condition = esc.get("new_condition") or str(wait_row.get("wait_condition", ""))
        _handle_still_wait(wait_log_id, symbol, new_condition, reasoning)
        result["reason"] = "still_wait_updated"

    else:
        # ERROR, SHADOW, or unknown
        result["reason"] = f"escalation_result: {decision}"

    log.info("[evaluator] %s outcome=%s decision=%s reason=%s",
             symbol, result["outcome"], decision, result["reason"])
    return result
