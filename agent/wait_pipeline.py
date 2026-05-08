"""
agent/wait_pipeline.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2 Plain Python WAIT Pipeline

Entry point: run_wait_pipeline()
Called by main.py when AGENT_USE_PIPELINE=true.

Flow:
  1. Read AGENTIC_WAIT_MONITOR — exit if false
  2. Read AGENT_USE_PIPELINE   — if false, delegate to legacy run_wait_monitor()
  3. get_market_state()
  4. get_open_waits()
  5. get_portfolio()
  6. No open waits → write one trace row, return
  7. For each WAIT symbol:
       a. get_fresh_indicators(symbol)
       b. ONE LLM call → {"condition_met": bool, "reason": "..."}
       c. condition_not_met  → log_skip
       d. condition_met + cap not reached + not shadow → escalate_to_claude
       e. condition_met + shadow_mode → log_skip [SHADOW]
       f. Write agent_trace for every symbol regardless

Hard rules:
  - agent/agent.py and agent/agent_tools.py NOT touched
  - All 7 tools via direct Python calls
  - Shadow mode and escalation cap enforced in Python
  - agent_trace written for every symbol — never skip
  - Boolean flags in Python — LLM reads them, never computes
  - Fail silent on LLM errors — condition_not_met, continue
  - No LangChain, raw Python only
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from datetime import datetime

from config import NST
from agent import agent_tools

log = logging.getLogger(__name__)

PIPELINE_MODEL_PRIMARY  = "openai/gpt-4o-mini:free"
PIPELINE_MODEL_FALLBACK = "openai/gpt-oss-20b:free"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LLM CONDITION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _call_condition_check(
    symbol:         str,
    wait_condition: str,
    indicators:     dict,
    market_state:   dict,
) -> dict:
    """
    Call cheap free LLM to check if WAIT condition is met.
    Tries PIPELINE_MODEL_PRIMARY first, falls back to PIPELINE_MODEL_FALLBACK.
    On any failure returns {"condition_met": False, "reason": "LLM_FAILURE"}.
    """
    from agent.prompt import build_condition_check_prompt
    from AI.openrouter import _call, _strip_fences

    prompt   = build_condition_check_prompt(symbol, wait_condition, indicators, market_state)
    messages = [{"role": "user", "content": prompt}]
    ctx      = f"pipeline_cc_{symbol}"

    for model in (PIPELINE_MODEL_PRIMARY, PIPELINE_MODEL_FALLBACK):
        raw = _call(
            model       = model,
            messages    = messages,
            max_tokens  = 80,
            temperature = 0.0,
            context     = ctx,
        )
        if raw is None:
            log.warning("[pipeline] %s returned None for %s — trying fallback", model, symbol)
            continue

        try:
            cleaned = _strip_fences(raw)
            result  = json.loads(cleaned)
            if "condition_met" in result:
                return {
                    "condition_met": bool(result["condition_met"]),
                    "reason":        str(result.get("reason", "")),
                }
        except Exception as exc:
            log.warning("[pipeline] JSON parse failed for %s (%s): %s | raw: %s",
                        symbol, model, exc, raw[:100])

    log.warning("[pipeline] All models failed for %s — treating as condition_not_met", symbol)
    return {"condition_met": False, "reason": "LLM_FAILURE"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRACE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _write_trace(
    cycle_ts:     str,
    step:         int,
    tool:         str,
    request_args: dict,
    response:     dict,
    escalated:    bool,
    decision:     str | None,
    elapsed_ms:   int,
) -> None:
    """Write one row to agent_trace. Fails silently."""
    try:
        from sheets import write_row
        write_row("agent_trace", {
            "cycle_ts":     cycle_ts,
            "step":         str(step),
            "tool":         tool,
            "request_args": json.dumps(request_args)[:2000],
            "response":     json.dumps(response)[:2000],
            "escalated":    "true" if escalated else "false",
            "decision":     decision or "",
            "elapsed_ms":   str(elapsed_ms),
            "created_at":   datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as exc:
        log.warning("[pipeline] trace write failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_wait_pipeline() -> dict:
    """
    Plain-Python WAIT pipeline entry point.

    Returns:
      {"ran": bool, "reason": str, "escalations": int, "elapsed_ms": int}

    Never raises — all errors caught and logged.
    """
    cycle_start = time.time()
    cycle_ts    = datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S")

    log.info("[pipeline] === WAIT pipeline cycle start: %s ===", cycle_ts)

    summary: dict = {
        "ran":         False,
        "reason":      "",
        "escalations": 0,
        "elapsed_ms":  0,
    }

    def _done(reason: str) -> dict:
        summary["reason"]     = reason
        summary["elapsed_ms"] = int((time.time() - cycle_start) * 1000)
        log.info("[pipeline] done: %s  elapsed=%dms", reason, summary["elapsed_ms"])
        return summary

    # ── Step 1: Master on/off ─────────────────────────────────────────────────
    try:
        from sheets import get_setting
    except Exception as exc:
        return _done(f"import_error: {exc}")

    try:
        if get_setting("AGENTIC_WAIT_MONITOR", "false").lower() != "true":
            return _done("AGENTIC_WAIT_MONITOR=false")

        # ── Step 2: Pipeline flag — guard against direct call with flag off ──
        if get_setting("AGENT_USE_PIPELINE", "false").lower() != "true":
            log.info("[pipeline] AGENT_USE_PIPELINE=false — delegating to legacy loop")
            from agent.agent import run_wait_monitor
            return run_wait_monitor()

        shadow_mode     = get_setting("AGENT_SHADOW_MODE",     "true").lower() == "true"
        max_escalations = int(get_setting("AGENT_MAX_ESCALATIONS", "2"))
        timeout_sec     = int(get_setting("AGENT_TIMEOUT_SEC",     "60"))

    except Exception as exc:
        log.error("[pipeline] settings read failed: %s", exc)
        return _done(f"settings_error: {exc}")

    log.info("[pipeline] shadow=%s max_esc=%d timeout=%ds",
             shadow_mode, max_escalations, timeout_sec)

    # ── Step 3: Market state ─────────────────────────────────────────────────
    market_state = agent_tools.get_market_state()
    if market_state.get("market_is_crisis"):
        return _done("CRISIS_market_state")
    if market_state.get("market_is_bear") and market_state.get("geo_environment_tense"):
        return _done("MARKET_ADVERSE")
    if market_state.get("trading_halted"):
        return _done("TRADING_HALTED")

    # ── Step 4: Open waits ───────────────────────────────────────────────────
    open_waits = agent_tools.get_open_waits()
    if not open_waits.get("has_waits"):
        _write_trace(cycle_ts, 0, "wait_pipeline",
                     {}, {"reason": "no_open_waits"},
                     False, None, int((time.time() - cycle_start) * 1000))
        return _done("no_open_waits")

    # ── Step 5: Portfolio ────────────────────────────────────────────────────
    portfolio = agent_tools.get_portfolio()

    # ── Step 7: Per-symbol loop ───────────────────────────────────────────────
    summary["ran"]   = True
    escalation_count = 0

    for step, wait in enumerate(open_waits["waits"], start=1):
        # Hard timeout
        if time.time() - cycle_start > timeout_sec:
            summary["reason"] = "TIMEOUT"
            break

        symbol         = wait["symbol"]
        wait_condition = wait["wait_condition"]
        wait_log_id    = wait["id"]

        log.info("[pipeline] [%d/%d] Checking %s",
                 step, open_waits["count"], symbol)

        # 7a — Fresh indicators
        indicators = agent_tools.get_fresh_indicators(symbol)

        # 7b — ONE LLM call
        t_llm    = time.time()
        llm_out  = _call_condition_check(symbol, wait_condition, indicators, market_state)
        llm_ms   = int((time.time() - t_llm) * 1000)

        condition_met = llm_out.get("condition_met", False)
        reason        = llm_out.get("reason", "")

        log.info("[pipeline] %s condition_met=%s reason=%s llm_ms=%d",
                 symbol, condition_met, reason[:60], llm_ms)

        escalated = False
        decision  = None

        if not condition_met:
            # 7c — Skip
            agent_tools.log_skip(symbol, reason or "condition_not_met", cycle_ts, step)

        elif shadow_mode:
            # 7e — Shadow skip
            agent_tools.log_skip(symbol, f"[SHADOW] would escalate: {reason}", cycle_ts, step)

        elif escalation_count >= max_escalations:
            # Cap reached
            agent_tools.log_skip(
                symbol,
                f"escalation_cap_reached ({max_escalations}): {reason}",
                cycle_ts, step,
            )

        else:
            # 7d — Escalate to Claude
            esc = agent_tools.escalate_to_claude(
                symbol         = symbol,
                wait_log_id    = wait_log_id,
                wait_condition = wait_condition,
                indicators     = indicators,
                market_state   = market_state,
                cycle_ts       = cycle_ts,
                step           = step,
                shadow_mode    = False,
            )
            escalation_count += 1
            escalated = True
            decision  = esc.get("decision")
            log.info("[pipeline] escalate_to_claude %s → %s", symbol, decision)

        # 7f — Trace for every symbol regardless
        _write_trace(
            cycle_ts     = cycle_ts,
            step         = step,
            tool         = "wait_pipeline",
            request_args = {
                "symbol":         symbol,
                "condition_met":  condition_met,
                "shadow_mode":    shadow_mode,
            },
            response     = {
                "reason":    reason,
                "escalated": escalated,
                "decision":  decision,
                "llm_ms":    llm_ms,
            },
            escalated    = escalated,
            decision     = decision,
            elapsed_ms   = int((time.time() - cycle_start) * 1000),
        )

    summary["escalations"] = escalation_count
    return _done(summary.get("reason") or "COMPLETED")
