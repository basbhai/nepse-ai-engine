"""
agent/wait_pipeline.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2 Plain Python WAIT Pipeline  (redesigned)

Entry point: run_wait_pipeline()
Called by main.py when AGENT_USE_PIPELINE=true.

Flow:
  1. Read AGENTIC_WAIT_MONITOR — exit if false.
  2. Read AGENT_USE_PIPELINE   — if false, delegate to legacy run_wait_monitor().
  3. get_market_state()  — CRISIS / BEAR+tense / HALTED block.
  4. get_open_waits()    — includes wait_condition_parsed + last_reviewed_date.
  5. get_portfolio()
  6. No open waits → write one trace row, return.
  7. For each WAIT symbol:
       a. get_fresh_indicators(symbol)
       b. condition_parser.get_parsed_condition(wait_row)
            → None: log_skip("parse_failed"), continue
       c. wait_evaluator.evaluate_wait(...)
            → SKIP / ESCALATED / ERROR
       d. _write_trace() — always, regardless of outcome.

Design decisions
----------------
- _call_condition_check() removed entirely — replaced by condition_parser +
  wait_evaluator which together eliminate the per-symbol-per-cycle LLM calls.
- get_open_waits() SQL extended to include the two new columns.
- _write_trace() is unchanged.
- All existing guards preserved: market state, timeout, escalation cap,
  shadow mode.
- agent/agent_tools.py NOT modified.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from datetime import datetime

from config import NST
from agent import agent_tools
from agent.condition_parser import get_parsed_condition
from agent.wait_evaluator   import evaluate_wait

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRACE WRITER  (unchanged from original)
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
# SECTION 2 — OPEN WAITS WITH NEW COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def _get_open_waits_extended() -> dict:
    """
    Like agent_tools.get_open_waits() but also fetches the two new columns:
      wait_condition_parsed  — cached parsed JSON (may be NULL)
      last_reviewed_date     — date of last Claude review (may be NULL)

    Returns the same shape as get_open_waits() so the rest of the pipeline
    is unaffected.
    """
    from sheets import run_raw_sql
    from agent.agent_tools import _safe_int

    result = {"waits": [], "count": 0, "has_waits": False, "elapsed_ms": 0}
    t0 = time.time()
    try:
        rows = run_raw_sql(
            """
            SELECT id, symbol, date, action, reasoning,
                   wait_condition, confidence,
                   wait_condition_parsed,
                   last_reviewed_date
            FROM market_log
            WHERE action    = 'WAIT'
              AND outcome   = 'PENDING'
              AND wait_condition IS NOT NULL
              AND wait_condition != ''
            ORDER BY id DESC
            LIMIT 20
            """
        )
        waits = []
        for r in rows:
            waits.append({
                "id":                     _safe_int(r.get("id")),
                "symbol":                 str(r.get("symbol", "")).upper().strip(),
                "signal_date":            str(r.get("date", "")),
                "wait_condition":         str(r.get("wait_condition", "")),
                "reasoning":              str(r.get("reasoning", "")),
                "confidence":             _safe_int(r.get("confidence", 0)),
                "wait_condition_parsed":  r.get("wait_condition_parsed") or "",
                "last_reviewed_date":     r.get("last_reviewed_date") or "",
            })
        result["waits"]     = waits
        result["count"]     = len(waits)
        result["has_waits"] = len(waits) > 0
        log.info("[pipeline] open waits (extended): %d rows", len(waits))
    except Exception as exc:
        log.error("[pipeline] _get_open_waits_extended failed: %s", exc)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


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

        # ── Step 2: Pipeline flag ─────────────────────────────────────────────
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

    # ── Step 4: Open waits (extended — includes parsed + reviewed columns) ────
    open_waits = _get_open_waits_extended()
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

        symbol      = wait["symbol"]
        wait_log_id = wait["id"]

        log.info("[pipeline] [%d/%d] Checking %s", step, open_waits["count"], symbol)

        # 7a — Fresh indicators
        indicators = agent_tools.get_fresh_indicators(symbol)

        # 7b — Condition parse (cache hit or free LLM parse)
        t_parse = time.time()
        parsed  = get_parsed_condition(wait)
        parse_ms = int((time.time() - t_parse) * 1000)

        if parsed is None:
            agent_tools.log_skip(symbol, "parse_failed", cycle_ts, step)
            _write_trace(
                cycle_ts     = cycle_ts,
                step         = step,
                tool         = "wait_pipeline",
                request_args = {"symbol": symbol, "stage": "parse_failed"},
                response     = {"reason": "parse_failed", "escalated": False,
                                "decision": None, "parse_ms": parse_ms},
                escalated    = False,
                decision     = "SKIPPED",
                elapsed_ms   = int((time.time() - cycle_start) * 1000),
            )
            continue

        # 7c — Evaluate (pre-filter → ambiguous → reviewed_today → Claude)
        if shadow_mode or escalation_count < max_escalations:
            t_eval = time.time()
            eval_result = evaluate_wait(
                wait_row     = wait,
                parsed       = parsed,
                indicators   = indicators,
                market_state = market_state,
                portfolio    = portfolio,
                cycle_ts     = cycle_ts,
                step         = step,
                shadow_mode  = shadow_mode,
            )
            eval_ms = int((time.time() - t_eval) * 1000)
        else:
            # Escalation cap reached — skip without calling evaluate_wait
            agent_tools.log_skip(
                symbol,
                f"escalation_cap_reached ({max_escalations})",
                cycle_ts,
                step,
            )
            _write_trace(
                cycle_ts     = cycle_ts,
                step         = step,
                tool         = "wait_pipeline",
                request_args = {"symbol": symbol, "stage": "cap_reached"},
                response     = {"reason": f"escalation_cap_reached ({max_escalations})",
                                "escalated": False, "decision": None},
                escalated    = False,
                decision     = "SKIPPED",
                elapsed_ms   = int((time.time() - cycle_start) * 1000),
            )
            continue

        outcome   = eval_result.get("outcome", "SKIP")
        reason    = eval_result.get("reason", "")
        decision  = eval_result.get("decision")
        escalated = outcome == "ESCALATED"

        if escalated and not shadow_mode:
            escalation_count += 1

        log.info("[pipeline] %s outcome=%s reason=%s decision=%s eval_ms=%d",
                 symbol, outcome, reason[:60], decision, eval_ms)

        # 7d — Trace for every symbol regardless
        _write_trace(
            cycle_ts     = cycle_ts,
            step         = step,
            tool         = "wait_pipeline",
            request_args = {
                "symbol":      symbol,
                "outcome":     outcome,
                "shadow_mode": shadow_mode,
            },
            response     = {
                "reason":    reason,
                "escalated": escalated,
                "decision":  decision,
                "parse_ms":  parse_ms,
                "eval_ms":   eval_ms,
            },
            escalated    = escalated,
            decision     = decision,
            elapsed_ms   = int((time.time() - cycle_start) * 1000),
        )

    summary["escalations"] = escalation_count
    return _done(summary.get("reason") or "COMPLETED")
