"""
agent/agent.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 1 Agentic WAIT Monitor
DeepSeek orchestrator loop.

Entry point: run_wait_monitor()
Called by main.py every 10-minute cycle when AGENTIC_WAIT_MONITOR=true.

Flow:
  1. Python pre-checks (no LLM) — market state, open waits, portfolio
  2. DeepSeek orchestrator loop (function calling) — decides what to check
  3. Tool dispatch — calls agent_tools.py implementations
  4. Escalation cap enforced in Python (not just system prompt)
  5. 60s hard timeout — trading loop must never block
  6. JSONL trace written per cycle

Architecture rules:
  - DeepSeek V4 Pro = orchestrator ONLY. DeepSeek R1 = Kelly Criterion ONLY.
  - Max 10 iterations per cycle (runaway loop prevention)
  - Max 2 escalations per cycle (cost control)
  - 60s hard timeout
  - All failures silent — agent error never blocks main pipeline
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import requests

from config import NST
from agent.prompt import ORCHESTRATOR_SYSTEM_PROMPT
from agent.tool_schemas import TOOL_SCHEMAS
from agent import agent_tools

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
#ORCHESTRATOR_MODEL = "deepseek/deepseek-v4-pro"
ORCHESTRATOR_MODEL = "qwen/qwen3-coder:free"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MAX_ITERATIONS     = 10      # hard cap — prevents runaway loops
MAX_ESCALATIONS    = 2       # cost control — enforced in Python
TIMEOUT_SECONDS    = 60      # hard timeout for entire cycle
MAX_RETRIES        = 2       # retries on 429/503


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OPENROUTER FUNCTION CALLING
# ══════════════════════════════════════════════════════════════════════════════

def _call_orchestrator(messages: list, context: str = "agent") -> Optional[dict]:
    """
    Single call to DeepSeek orchestrator with function calling enabled.
    Returns the full response message dict (content + tool_calls) or None.

    We cannot use AI.openrouter._call() here because that function strips
    tool_use blocks and returns only text. For function calling we need
    the raw tool_calls array from the response.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        log.error("[agent] OPENROUTER_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/basbhai/nepse-ai-engine",
        "X-Title":       "NEPSE AI Engine",
    }

    payload = {
        "model":       ORCHESTRATOR_MODEL,
        "max_tokens":  1000,
        "temperature": 0.1,
        "messages":    messages,
        "tools":       TOOL_SCHEMAS,
        "tool_choice": "auto",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if resp.status_code in (429, 503) and attempt < MAX_RETRIES:
                wait = 5 * attempt
                log.warning("[agent] HTTP %d (attempt %d) — retrying in %ds",
                            resp.status_code, attempt, wait)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                log.error("[agent] OpenRouter HTTP %d: %s",
                          resp.status_code, resp.text[:200])
                return None

            data    = resp.json()
            message = data["choices"][0]["message"]
            return message

        except requests.exceptions.Timeout:
            log.warning("[agent] OpenRouter timeout (attempt %d)", attempt)
            if attempt >= MAX_RETRIES:
                return None

        except Exception as e:
            log.error("[agent] OpenRouter call failed: %s", e)
            return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOOL DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

def _dispatch_tool(
    tool_name: str,
    tool_args: dict,
    cycle_ts:  str,
    step:      int,
    shadow_mode: bool,
    escalation_count: list,  # mutable [int] for pass-by-reference
) -> dict:
    """
    Route tool calls from the orchestrator to agent_tools implementations.
    Validates tool name. Returns result dict.
    Enforces escalation_count cap before calling escalate_to_claude.
    """
    t0 = time.time()

    # Normalise
    name = str(tool_name).strip()

    if name == "get_market_state":
        return agent_tools.get_market_state()

    elif name == "get_open_waits":
        return agent_tools.get_open_waits()

    elif name == "get_portfolio":
        return agent_tools.get_portfolio()

    elif name == "get_fresh_indicators":
        symbol = str(tool_args.get("symbol", "")).upper().strip()
        return agent_tools.get_fresh_indicators(symbol)

    elif name == "check_wait_condition":
        symbol         = str(tool_args.get("symbol", "")).upper().strip()
        wait_condition = str(tool_args.get("wait_condition", ""))
        indicators     = tool_args.get("indicators", {})
        return agent_tools.check_wait_condition(symbol, wait_condition, indicators)

    elif name == "log_skip":
        symbol = str(tool_args.get("symbol", "")).upper().strip()
        reason = str(tool_args.get("reason", ""))
        return agent_tools.log_skip(symbol, reason, cycle_ts, step)

    elif name == "escalate_to_claude":
        # Python-level cap — not just system prompt
        if escalation_count[0] >= MAX_ESCALATIONS:
            log.warning("[agent] escalation cap reached (%d) — skipping %s",
                        MAX_ESCALATIONS, tool_args.get("symbol", "?"))
            return {
                "decision":  "SKIPPED_CAP",
                "reasoning": f"Escalation cap {MAX_ESCALATIONS} reached this cycle",
            }

        symbol         = str(tool_args.get("symbol", "")).upper().strip()
        wait_log_id    = int(tool_args.get("wait_log_id", 0))
        wait_condition = str(tool_args.get("wait_condition", ""))
        indicators     = tool_args.get("indicators", {})
        market_state   = tool_args.get("market_state", {})

        result = agent_tools.escalate_to_claude(
            symbol         = symbol,
            wait_log_id    = wait_log_id,
            wait_condition = wait_condition,
            indicators     = indicators,
            market_state   = market_state,
            cycle_ts       = cycle_ts,
            step           = step,
            shadow_mode    = shadow_mode,
        )
        escalation_count[0] += 1
        return result

    else:
        log.warning("[agent] Unknown tool called: %s", name)
        return {"error": f"Unknown tool: {name}"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRACE WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _write_trace(
    cycle_ts:    str,
    step:        int,
    tool:        str,
    request_args: dict,
    response:    dict,
    escalated:   bool,
    decision:    Optional[str],
    elapsed_ms:  int,
) -> None:
    """Write one row to agent_trace table. Fails silently."""
    try:
        from sheets import write_row
        write_row("agent_trace", {
            "cycle_ts":    cycle_ts,
            "step":        str(step),
            "tool":        tool,
            "request_args": json.dumps(request_args)[:2000],
            "response":    json.dumps(response)[:2000],
            "escalated":   "true" if escalated else "false",
            "decision":    decision or "",
            "elapsed_ms":  str(elapsed_ms),
            "created_at":  datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        log.warning("[agent] trace write failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PYTHON PRE-CHECKS (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _python_pre_check() -> tuple[bool, str]:
    """
    Fast Python-only checks before spending any LLM tokens.
    Returns (should_run, reason).
    Free — no API call.
    """
    try:
        from sheets import get_setting, run_raw_sql

        # Check 1: AGENTIC_WAIT_MONITOR flag
        if get_setting("AGENTIC_WAIT_MONITOR", "false").lower() != "true":
            return False, "AGENTIC_WAIT_MONITOR=false"

        # Check 2: Any WAIT rows at all?
        rows = run_raw_sql(
            """
            SELECT COUNT(*) as cnt FROM market_log
            WHERE action = 'WAIT'
              AND outcome = 'PENDING'
              AND wait_condition IS NOT NULL
              AND wait_condition != ''
            """
        )
        count = int(rows[0]["cnt"]) if rows else 0
        if count == 0:
            return False, "no_open_waits"

        # Check 3: Market state + geo — avoid full BEAR + tense geo
        market_state = get_setting("MARKET_STATE", "SIDEWAYS").upper()
        if market_state == "CRISIS":
            return False, "CRISIS_market_state"

        return True, f"ok — {count} open waits"

    except Exception as e:
        log.error("[agent] _python_pre_check failed: %s", e)
        return False, f"pre_check_error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN ORCHESTRATOR LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_wait_monitor() -> dict:
    """
    Main entry point. Called by main.py every 10-minute cycle.

    Returns summary dict:
      {
        "ran": bool,
        "reason": str,
        "iterations": int,
        "escalations": int,
        "elapsed_ms": int,
      }

    Never raises — all errors are caught and logged.
    main.py wraps this in try/except — agent failure is silent.
    """
    cycle_start = time.time()
    cycle_ts    = datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S")

    log.info("[agent] === WAIT monitor cycle start: %s ===", cycle_ts)

    summary = {
        "ran":         False,
        "reason":      "",
        "iterations":  0,
        "escalations": 0,
        "elapsed_ms":  0,
    }

    # ── Python pre-checks — free ──────────────────────────────────────────
    should_run, reason = _python_pre_check()
    if not should_run:
        log.info("[agent] pre-check blocked: %s", reason)
        summary["reason"]  = reason
        summary["elapsed_ms"] = int((time.time() - cycle_start) * 1000)
        return summary

    log.info("[agent] pre-check passed: %s", reason)

    # ── Read settings ─────────────────────────────────────────────────────
    try:
        from sheets import get_setting
        shadow_mode     = get_setting("AGENT_SHADOW_MODE", "true").lower() == "true"
        max_escalations = int(get_setting("AGENT_MAX_ESCALATIONS", str(MAX_ESCALATIONS)))
    except Exception:
        shadow_mode     = True
        max_escalations = MAX_ESCALATIONS

    log.info("[agent] shadow_mode=%s max_escalations=%d", shadow_mode, max_escalations)

    # ── Build initial messages ────────────────────────────────────────────
    messages = [
        {
            "role":    "system",
            "content": ORCHESTRATOR_SYSTEM_PROMPT,
        },
        {
            "role":    "user",
            "content": (
                f"Cycle: {cycle_ts} NST. "
                f"Shadow mode: {shadow_mode}. "
                f"Max escalations this cycle: {max_escalations}. "
                f"Check all open WAIT conditions and decide what to do."
            ),
        },
    ]

    # Mutable counter — passed to _dispatch_tool by reference
    escalation_count = [0]
    iteration        = 0
    summary["ran"]   = True

    # ── Orchestrator loop ─────────────────────────────────────────────────
    while iteration < MAX_ITERATIONS:
        # Hard timeout check
        elapsed = time.time() - cycle_start
        if elapsed > TIMEOUT_SECONDS:
            log.warning("[agent] Hard timeout (%ds) reached at iteration %d",
                        TIMEOUT_SECONDS, iteration)
            summary["reason"] = "TIMEOUT"
            break

        iteration += 1
        log.info("[agent] Iteration %d/%d", iteration, MAX_ITERATIONS)

        # Call orchestrator
        t_call = time.time()
        response_msg = _call_orchestrator(messages, context=f"agent_cycle_{iteration}")
        call_ms = int((time.time() - t_call) * 1000)

        if response_msg is None:
            log.error("[agent] Orchestrator returned None at iteration %d", iteration)
            summary["reason"] = "ORCHESTRATOR_FAILURE"
            break

        content    = response_msg.get("content") or ""
        tool_calls = response_msg.get("tool_calls") or []

        # Append assistant message to conversation
        messages.append({
            "role":       "assistant",
            "content":    content,
            "tool_calls": tool_calls,
        })

        # ── No tool calls → orchestrator is done ─────────────────────────
        if not tool_calls:
            log.info("[agent] Orchestrator done (no tool calls). Summary: %s",
                     content[:200] if content else "(no content)")
            summary["reason"] = "COMPLETED"
            break

        # ── Dispatch each tool call ───────────────────────────────────────
        tool_results_for_messages = []

        for tc in tool_calls:
            tc_id   = tc.get("id", f"call_{iteration}")
            tc_name = tc.get("function", {}).get("name", "")
            tc_args_raw = tc.get("function", {}).get("arguments", "{}")

            try:
                tc_args = json.loads(tc_args_raw) if isinstance(tc_args_raw, str) else tc_args_raw
            except json.JSONDecodeError:
                tc_args = {}
                log.warning("[agent] Could not parse tool args for %s: %s", tc_name, tc_args_raw)

            log.info("[agent] Tool call: %s(%s)", tc_name,
                     str(tc_args)[:80] if tc_args else "")

            # Dispatch
            t_tool = time.time()
            tool_result = _dispatch_tool(
                tool_name        = tc_name,
                tool_args        = tc_args,
                cycle_ts         = cycle_ts,
                step             = iteration,
                shadow_mode      = shadow_mode,
                escalation_count = escalation_count,
            )
            tool_ms = int((time.time() - t_tool) * 1000)

            # Write trace (skip log_skip — it writes its own trace)
            if tc_name not in ("log_skip",):
                is_escalation = tc_name == "escalate_to_claude"
                decision_val  = tool_result.get("decision") if is_escalation else None
                _write_trace(
                    cycle_ts    = cycle_ts,
                    step        = iteration,
                    tool        = tc_name,
                    request_args= tc_args,
                    response    = tool_result,
                    escalated   = is_escalation,
                    decision    = decision_val,
                    elapsed_ms  = tool_ms,
                )

            # Build tool result message for next iteration
            tool_results_for_messages.append({
                "role":         "tool",
                "tool_call_id": tc_id,
                "content":      json.dumps(tool_result),
            })

            log.info("[agent] Tool %s completed in %dms", tc_name, tool_ms)

        # Append all tool results to conversation
        messages.extend(tool_results_for_messages)

    # ── Final summary ─────────────────────────────────────────────────────
    total_ms = int((time.time() - cycle_start) * 1000)
    summary["iterations"]  = iteration
    summary["escalations"] = escalation_count[0]
    summary["elapsed_ms"]  = total_ms

    log.info(
        "[agent] === Cycle complete: ran=%s reason=%s iters=%d escalations=%d elapsed=%dms ===",
        summary["ran"], summary["reason"],
        summary["iterations"], summary["escalations"], summary["elapsed_ms"],
    )

    return summary
