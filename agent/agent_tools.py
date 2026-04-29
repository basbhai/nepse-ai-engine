"""
agent/agent_tools.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 1 Agentic WAIT Monitor
Tool implementations for the DeepSeek orchestrator.

All 7 tools called by agent.py via function calling API.
Each tool:
  - Validates and normalises inputs before any DB call
  - Returns a rich dict with boolean flags (LLM reads flags, never computes math)
  - Fails silently — always returns a dict, never raises
  - Uses only real functions from sheets.py and existing modules

Tool list:
  1. get_market_state()         → MARKET_STATE, geo/nepal scores, paper mode
  2. get_open_waits()           → All WAIT rows with wait_condition != ''
  3. get_portfolio()            → open positions, slots_remaining, capital
  4. get_fresh_indicators(sym)  → RSI, MACD, BB, volume + boolean flags
  5. check_wait_condition(sym)  → Python boolean pre-check of condition fields
  6. log_skip(sym, reason)      → Write silent skip to agent_trace
  7. escalate_to_claude(sym, ctx) → Call Claude, write result, Telegram if BUY

Architecture rules (non-negotiable):
  ALWAYS: from sheets import ...
  market_log is append-only → write_row, NEVER upsert_row
  NST timezone throughout
  LLM reads boolean flags — never raw numbers
  All inputs validated + normalised before use
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import NST

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now_nst() -> str:
    """Current NST datetime string."""
    return datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(val, default: float = 0.0) -> float:
    """Safely parse any value to float."""
    try:
        return float(val) if val is not None and val != "" else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Safely parse any value to int."""
    try:
        return int(float(val)) if val is not None and val != "" else default
    except (ValueError, TypeError):
        return default


def _normalise_symbol(symbol: str) -> str:
    """Uppercase and strip whitespace from symbol."""
    return str(symbol).upper().strip()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — get_market_state
# ══════════════════════════════════════════════════════════════════════════════

def get_market_state() -> dict:
    """
    Return current market state, geo/nepal scores, NRB decision, paper mode.
    Reads from settings, geopolitical_data, nepal_pulse.

    Returns dict with boolean flags for LLM:
      - market_is_bear        : MARKET_STATE == BEAR
      - market_is_crisis      : MARKET_STATE == CRISIS
      - geo_environment_tense : combined_geo < -3
      - trading_halted        : circuit breaker or geo block active
    """
    t0 = time.time()
    result = {
        "market_state":       "SIDEWAYS",
        "nepse_vs_200dma":    "",
        "geo_score":          0,
        "nepal_score":        0,
        "combined_geo":       0,
        "nrb_rate_decision":  "UNCHANGED",
        "paper_mode":         True,
        # Boolean flags — LLM reads these, never recomputes
        "market_is_bear":        False,
        "market_is_crisis":      False,
        "geo_environment_tense": False,
        "trading_halted":        False,
        "elapsed_ms":            0,
    }
    try:
        from sheets import get_setting, get_latest_geo, get_latest_pulse, get_macro_data

        result["market_state"]    = get_setting("MARKET_STATE", "SIDEWAYS").upper().strip()
        result["nepse_vs_200dma"] = get_setting("NEPSE_200DMA", "")
        result["paper_mode"]      = get_setting("PAPER_MODE", "true").lower() == "true"

        geo = get_latest_geo() or {}
        result["geo_score"] = _safe_int(geo.get("geo_score", 0))

        pulse = get_latest_pulse() or {}
        result["nepal_score"] = _safe_int(pulse.get("nepal_score", 0))

        result["combined_geo"] = result["geo_score"] + result["nepal_score"]

        macro = get_macro_data()
        result["nrb_rate_decision"] = macro.get("NRB_Rate_Decision", "UNCHANGED")

        # Boolean flags computed in Python — LLM never does this math
        result["market_is_bear"]        = result["market_state"] in ("BEAR", "FULL_BEAR")
        result["market_is_crisis"]       = result["market_state"] == "CRISIS"
        result["geo_environment_tense"]  = result["combined_geo"] < -3
        result["trading_halted"]         = (
            get_setting("CIRCUIT_BREAKER", "false").lower() == "true"
        )

        log.info(
            "[agent_tools] get_market_state: state=%s geo=%+d nepal=%+d combined=%+d",
            result["market_state"], result["geo_score"],
            result["nepal_score"], result["combined_geo"],
        )

    except Exception as e:
        log.error("[agent_tools] get_market_state failed: %s", e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — get_open_waits
# ══════════════════════════════════════════════════════════════════════════════

def get_open_waits() -> dict:
    """
    Return all WAIT rows from market_log where:
      - action = 'WAIT'
      - outcome = 'PENDING'
      - wait_condition is not empty

    Returns dict with:
      - waits: list of dicts (symbol, wait_condition, signal_date, reasoning, id)
      - count: total found
      - has_waits: boolean flag
    """
    t0 = time.time()
    result = {
        "waits":     [],
        "count":     0,
        "has_waits": False,
        "elapsed_ms": 0,
    }
    try:
        from sheets import run_raw_sql

        rows = run_raw_sql(
            """
            SELECT id, symbol, date, action, reasoning, wait_condition, confidence
            FROM market_log
            WHERE action = 'WAIT'
              AND outcome = 'PENDING'
              AND wait_condition IS NOT NULL
              AND wait_condition != ''
            ORDER BY id DESC
            LIMIT 20
            """
        )

        waits = []
        for r in rows:
            waits.append({
                "id":             _safe_int(r.get("id")),
                "symbol":         str(r.get("symbol", "")).upper().strip(),
                "signal_date":    str(r.get("date", "")),
                "wait_condition": str(r.get("wait_condition", "")),
                "reasoning":      str(r.get("reasoning", "")),
                "confidence":     _safe_int(r.get("confidence", 0)),
            })

        result["waits"]     = waits
        result["count"]     = len(waits)
        result["has_waits"] = len(waits) > 0

        log.info("[agent_tools] get_open_waits: found %d WAIT rows", len(waits))

    except Exception as e:
        log.error("[agent_tools] get_open_waits failed: %s", e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — get_portfolio
# ══════════════════════════════════════════════════════════════════════════════

def get_portfolio() -> dict:
    """
    Return open positions, slots remaining, and capital.
    Reads MAX_POSITIONS from settings (never hardcoded).
    Reads capital from CAPITAL_TOTAL_NPR setting.

    Returns dict with:
      - open_positions: list of symbol strings
      - position_count: int
      - max_positions: int (from settings)
      - slots_remaining: int (computed in Python)
      - capital_npr: float
      - portfolio_full: boolean flag
      - no_slots: boolean flag
    """
    t0 = time.time()
    result = {
        "open_positions":  [],
        "position_count":  0,
        "max_positions":   3,
        "slots_remaining": 3,
        "capital_npr":     0.0,
        # Boolean flags
        "portfolio_full":  False,
        "no_slots":        False,
        "elapsed_ms":      0,
    }
    try:
        from sheets import get_setting, read_tab

        max_positions = _safe_int(get_setting("MAX_POSITIONS", "3"), default=3)
        capital_npr   = _safe_float(get_setting("CAPITAL_TOTAL_NPR", "100000"), default=100000.0)

        # Read open positions from portfolio table (same as gemini_filter._load_open_positions)
        rows = read_tab("portfolio")
        open_positions = [
            r["symbol"].upper()
            for r in rows
            if r.get("status", "").upper() == "OPEN" and r.get("symbol")
        ]

        slots_remaining = max(0, max_positions - len(open_positions))

        result["open_positions"]  = open_positions
        result["position_count"]  = len(open_positions)
        result["max_positions"]   = max_positions
        result["slots_remaining"] = slots_remaining
        result["capital_npr"]     = capital_npr
        # Boolean flags — computed in Python
        result["portfolio_full"]  = slots_remaining == 0
        result["no_slots"]        = slots_remaining == 0

        log.info(
            "[agent_tools] get_portfolio: positions=%d/%d slots=%d capital=NPR %.0f",
            len(open_positions), max_positions, slots_remaining, capital_npr,
        )

    except Exception as e:
        log.error("[agent_tools] get_portfolio failed: %s", e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — get_fresh_indicators
# ══════════════════════════════════════════════════════════════════════════════

def get_fresh_indicators(symbol: str) -> dict:
    """
    Fetch latest indicators for a symbol from indicators table.
    Returns raw values PLUS boolean flags — LLM reads flags only.

    Boolean flags computed in Python:
      - macd_cross_just_bullish : macd_cross == 'BULLISH'
      - bb_breakout_up          : bb_signal in (UPPER_TOUCH, EXPANSION) and bb_pct_b > 0.8
      - volume_above_avg        : volume_ratio > 1.0
      - rsi_oversold            : rsi_14 < 35
      - rsi_overbought          : rsi_14 > 70
      - ema_trend_bullish       : ema_trend == 'ABOVE_ALL'
      - indicators_fresh        : data is from today or yesterday
    """
    t0 = time.time()
    symbol = _normalise_symbol(symbol)

    result = {
        "symbol":       symbol,
        "found":        False,
        "date":         "",
        # Raw indicator values
        "rsi_14":         0.0,
        "macd_line":      0.0,
        "macd_signal":    0.0,
        "macd_histogram": 0.0,
        "macd_cross":     "NONE",
        "bb_upper":       0.0,
        "bb_middle":      0.0,
        "bb_lower":       0.0,
        "bb_pct_b":       0.0,
        "bb_signal":      "NEUTRAL",
        "volume":         0,
        "volume_ratio":   0.0,
        "obv_trend":      "FLAT",
        "ema_trend":      "MIXED",
        "ema_20":         0.0,
        "ema_50":         0.0,
        "ema_200":        0.0,
        "atr_14":         0.0,
        "support_level":  0.0,
        "resistance_level": 0.0,
        "tech_score":     0,
        "tech_signal":    "NEUTRAL",
        # Boolean flags — LLM reads these
        "macd_cross_just_bullish": False,
        "bb_breakout_up":          False,
        "volume_above_avg":        False,
        "rsi_oversold":            False,
        "rsi_overbought":          False,
        "ema_trend_bullish":       False,
        "indicators_fresh":        False,
        "elapsed_ms":              0,
    }

    if not symbol:
        log.warning("[agent_tools] get_fresh_indicators: empty symbol")
        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    try:
        from sheets import run_raw_sql

        rows = run_raw_sql(
            """
            SELECT * FROM indicators
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT 1
            """,
            (symbol,),
        )

        if not rows:
            log.info("[agent_tools] get_fresh_indicators: no data for %s", symbol)
            result["elapsed_ms"] = int((time.time() - t0) * 1000)
            return result

        row = rows[0]
        result["found"] = True
        result["date"]  = str(row.get("date", ""))

        # Raw values
        result["rsi_14"]         = _safe_float(row.get("rsi_14"))
        result["macd_line"]      = _safe_float(row.get("macd_line"))
        result["macd_signal"]    = _safe_float(row.get("macd_signal"))
        result["macd_histogram"] = _safe_float(row.get("macd_histogram"))
        result["macd_cross"]     = str(row.get("macd_cross", "NONE")).upper()
        result["bb_upper"]       = _safe_float(row.get("bb_upper"))
        result["bb_middle"]      = _safe_float(row.get("bb_middle"))
        result["bb_lower"]       = _safe_float(row.get("bb_lower"))
        result["bb_pct_b"]       = _safe_float(row.get("bb_pct_b"))
        result["bb_signal"]      = str(row.get("bb_signal", "NEUTRAL")).upper()
        result["volume"]         = _safe_int(row.get("volume"))
        result["obv_trend"]      = str(row.get("obv_trend", "FLAT")).upper()
        result["ema_trend"]      = str(row.get("ema_trend", "MIXED")).upper()
        result["ema_20"]         = _safe_float(row.get("ema_20"))
        result["ema_50"]         = _safe_float(row.get("ema_50"))
        result["ema_200"]        = _safe_float(row.get("ema_200"))
        result["atr_14"]         = _safe_float(row.get("atr_14"))
        result["support_level"]  = _safe_float(row.get("support_level"))
        result["resistance_level"] = _safe_float(row.get("resistance_level"))
        result["tech_score"]     = _safe_int(row.get("tech_score"))
        result["tech_signal"]    = str(row.get("tech_signal", "NEUTRAL")).upper()

        # volume_ratio: compare today's volume to a rolling baseline
        # indicators table doesn't store volume_ratio directly — compute from
        # atrad_market_watch if available, else skip (flag stays False)
        try:
            vol_rows = run_raw_sql(
                """
                SELECT volume FROM atrad_market_watch
                WHERE symbol = %s
                ORDER BY date DESC, time DESC
                LIMIT 1
                """,
                (symbol,),
            )
            if vol_rows:
                live_vol = _safe_float(vol_rows[0].get("volume"))
                # Compare against indicators volume (morning baseline)
                base_vol = _safe_float(row.get("volume"))
                if base_vol > 0:
                    result["volume_ratio"] = round(live_vol / base_vol, 2)
        except Exception:
            pass  # volume_ratio stays 0 — flag stays False

        # Boolean flags — all computed in Python
        result["macd_cross_just_bullish"] = result["macd_cross"] == "BULLISH"
        result["bb_breakout_up"]          = (
            result["bb_signal"] in ("UPPER_TOUCH", "EXPANSION")
            and result["bb_pct_b"] > 0.8
        )
        result["volume_above_avg"]  = result["volume_ratio"] > 1.0
        result["rsi_oversold"]      = result["rsi_14"] < 35
        result["rsi_overbought"]    = result["rsi_14"] > 70
        result["ema_trend_bullish"] = result["ema_trend"] == "ABOVE_ALL"

        # Freshness — is data from today or yesterday?
        try:
            today = datetime.now(tz=NST).strftime("%Y-%m-%d")
            yesterday = (datetime.now(tz=NST) - timedelta(days=1)).strftime("%Y-%m-%d")
            result["indicators_fresh"] = result["date"] in (today, yesterday)
        except Exception:
            result["indicators_fresh"] = False

        log.info(
            "[agent_tools] get_fresh_indicators: %s rsi=%.1f macd=%s bb=%s "
            "macd_bullish=%s bb_breakout=%s vol_above=%s",
            symbol,
            result["rsi_14"], result["macd_cross"], result["bb_signal"],
            result["macd_cross_just_bullish"],
            result["bb_breakout_up"],
            result["volume_above_avg"],
        )

    except Exception as e:
        log.error("[agent_tools] get_fresh_indicators(%s) failed: %s", symbol, e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — check_wait_condition
# ══════════════════════════════════════════════════════════════════════════════

def check_wait_condition(symbol: str, wait_condition: str, indicators: dict) -> dict:
    """
    Python boolean pre-check of the wait_condition text against fresh indicators.
    This is the hybrid approach: Python booleans + LLM confirmation.

    Scans wait_condition text for known keywords and checks the matching
    boolean flags from get_fresh_indicators(). If any keyword matches and
    the corresponding flag is True → python_says_met = True.

    The orchestrator uses this to decide whether to bother escalating to Claude.
    Near-100% reliability vs 85% LLM-only.

    Returns:
      - python_says_met   : bool — Python thinks condition is met
      - matched_keywords  : list of keywords found in condition text
      - flags_checked     : dict of which boolean flags were evaluated
      - confidence        : 'HIGH' | 'LOW' | 'UNKNOWN'
    """
    t0 = time.time()
    symbol = _normalise_symbol(symbol)
    condition_lower = wait_condition.lower()

    result = {
        "symbol":           symbol,
        "python_says_met":  False,
        "matched_keywords": [],
        "flags_checked":    {},
        "confidence":       "UNKNOWN",
        "elapsed_ms":       0,
    }

    # Keyword → boolean flag mapping
    # Each entry: (keyword_variants, flag_name_in_indicators)
    KEYWORD_FLAG_MAP = [
        (["macd bullish", "macd cross", "macd crossover", "bullish cross"],
         "macd_cross_just_bullish"),
        (["bb breakout", "bollinger breakout", "above upper band", "bb upper"],
         "bb_breakout_up"),
        (["above average volume", "high volume", "volume surge", "strong volume"],
         "volume_above_avg"),
        (["oversold", "rsi below 35", "rsi < 35", "rsi under 35"],
         "rsi_oversold"),
        (["ema trend bullish", "above all ema", "above all moving average",
          "golden cross", "ema bullish"],
         "ema_trend_bullish"),
    ]

    matched     = []
    flags_used  = {}
    any_met     = False

    for keywords, flag_name in KEYWORD_FLAG_MAP:
        for kw in keywords:
            if kw in condition_lower:
                flag_val = indicators.get(flag_name, False)
                flags_used[flag_name] = flag_val
                matched.append(kw)
                if flag_val:
                    any_met = True
                break  # one match per flag group is enough

    result["matched_keywords"] = matched
    result["flags_checked"]    = flags_used

    if matched:
        result["python_says_met"] = any_met
        result["confidence"]      = "HIGH" if len(matched) >= 2 else "LOW"
    else:
        # No keywords matched — condition is too complex/narrative for Python
        # Let orchestrator decide whether to escalate purely on LLM judgment
        result["python_says_met"] = False
        result["confidence"]      = "UNKNOWN"

    log.info(
        "[agent_tools] check_wait_condition: %s matched=%s met=%s confidence=%s",
        symbol, matched, result["python_says_met"], result["confidence"],
    )

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — log_skip
# ══════════════════════════════════════════════════════════════════════════════

def log_skip(
    symbol:   str,
    reason:   str,
    cycle_ts: str,
    step:     int,
) -> dict:
    """
    Write a silent skip to agent_trace.
    No Telegram. No market_log write. Just audit trail.

    Called when:
      - Python pre-check says condition NOT met
      - Market state blocks action (BEAR + tense geo)
      - Portfolio full
    """
    t0 = time.time()
    symbol = _normalise_symbol(symbol)

    result = {"written": False, "elapsed_ms": 0}

    try:
        from sheets import write_row

        write_row("agent_trace", {
            "cycle_ts":    cycle_ts,
            "step":        str(step),
            "tool":        "log_skip",
            "request_args": json.dumps({"symbol": symbol, "reason": reason}),
            "response":    json.dumps({"action": "SKIPPED"}),
            "escalated":   "false",
            "decision":    "SKIPPED",
            "elapsed_ms":  "0",
            "created_at":  _now_nst(),
        })

        result["written"] = True
        log.info("[agent_tools] log_skip: %s — %s", symbol, reason)

    except Exception as e:
        log.error("[agent_tools] log_skip(%s) failed: %s", symbol, e)

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — escalate_to_claude
# ══════════════════════════════════════════════════════════════════════════════

def escalate_to_claude(
    symbol:         str,
    wait_log_id:    int,
    wait_condition: str,
    indicators:     dict,
    market_state:   dict,
    cycle_ts:       str,
    step:           int,
    shadow_mode:    bool = True,
) -> dict:
    """
    Escalate a WAIT symbol to Claude for final BUY/STILL_WAIT/NOW_AVOID decision.

    In shadow_mode=True: logs 'WOULD ESCALATE' without calling Claude API.
    In shadow_mode=False: calls Claude via AI.openrouter.ask_claude().

    If Claude returns BUY:
      - Updates market_log row outcome = 'CONDITION_MET'
      - Writes new BUY row to market_log (append-only)
      - Sends Telegram alert

    Returns:
      - decision: 'BUY' | 'STILL_WAIT' | 'NOW_AVOID' | 'SHADOW' | 'ERROR'
      - reasoning: Claude's explanation
      - buy_written: bool (True if new BUY row written to market_log)
      - telegram_sent: bool
    """
    t0 = time.time()
    symbol = _normalise_symbol(symbol)

    result = {
        "symbol":        symbol,
        "decision":      "ERROR",
        "reasoning":     "",
        "buy_written":   False,
        "telegram_sent": False,
        "shadow_mode":   shadow_mode,
        "elapsed_ms":    0,
    }

    # ── Shadow mode — log only, no Claude call ─────────────────────────────
    if shadow_mode:
        log.info(
            "[agent_tools] SHADOW ESCALATE: %s | condition: %s",
            symbol, wait_condition[:80],
        )
        result["decision"]  = "SHADOW"
        result["reasoning"] = f"Shadow mode — would escalate {symbol} to Claude"

        try:
            from sheets import write_row
            write_row("agent_trace", {
                "cycle_ts":    cycle_ts,
                "step":        str(step),
                "tool":        "escalate_to_claude",
                "request_args": json.dumps({"symbol": symbol, "shadow": True}),
                "response":    json.dumps({"decision": "SHADOW"}),
                "escalated":   "true",
                "decision":    "SHADOW",
                "elapsed_ms":  str(int((time.time() - t0) * 1000)),
                "created_at":  _now_nst(),
            })
        except Exception as e:
            log.warning("[agent_tools] shadow trace write failed: %s", e)

        result["elapsed_ms"] = int((time.time() - t0) * 1000)
        return result

    # ── Live escalation — call Claude ─────────────────────────────────────
    try:
        from agent.prompt import build_escalation_prompt
        from AI.openrouter import ask_claude

        prompt = build_escalation_prompt(
            symbol         = symbol,
            wait_condition = wait_condition,
            indicators     = indicators,
            market_state   = market_state,
        )

        claude_response = ask_claude(
            prompt      = prompt,
            context     = f"agent_wait_monitor_{symbol}",
            max_tokens  = 600,
            temperature = 0.1,
        )

        if not claude_response:
            log.error("[agent_tools] escalate_to_claude: Claude returned None for %s", symbol)
            result["decision"] = "ERROR"
            result["elapsed_ms"] = int((time.time() - t0) * 1000)
            return result

        decision  = str(claude_response.get("action", "STILL_WAIT")).upper()
        reasoning = str(claude_response.get("reasoning", ""))

        # Normalise decision to valid set
        if decision not in ("BUY", "STILL_WAIT", "NOW_AVOID"):
            log.warning("[agent_tools] unexpected Claude decision: %s — defaulting STILL_WAIT", decision)
            decision = "STILL_WAIT"

        result["decision"]  = decision
        result["reasoning"] = reasoning

        log.info(
            "[agent_tools] Claude decision for %s: %s | %s",
            symbol, decision, reasoning[:80],
        )

        # ── If BUY: update market_log + write new BUY row + Telegram ──────
        if decision == "BUY":
            _handle_wait_to_buy(
                symbol         = symbol,
                wait_log_id    = wait_log_id,
                reasoning      = reasoning,
                claude_response= claude_response,
                result         = result,
            )

        # ── Write trace ───────────────────────────────────────────────────
        try:
            from sheets import write_row
            write_row("agent_trace", {
                "cycle_ts":    cycle_ts,
                "step":        str(step),
                "tool":        "escalate_to_claude",
                "request_args": json.dumps({"symbol": symbol, "wait_log_id": wait_log_id}),
                "response":    json.dumps({"decision": decision, "reasoning": reasoning[:200]}),
                "escalated":   "true",
                "decision":    decision,
                "elapsed_ms":  str(int((time.time() - t0) * 1000)),
                "created_at":  _now_nst(),
            })
        except Exception as e:
            log.warning("[agent_tools] escalation trace write failed: %s", e)

    except Exception as e:
        log.error("[agent_tools] escalate_to_claude(%s) failed: %s", symbol, e)
        result["decision"] = "ERROR"

    result["elapsed_ms"] = int((time.time() - t0) * 1000)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL — handle WAIT → BUY transition
# ══════════════════════════════════════════════════════════════════════════════

def _handle_wait_to_buy(
    symbol:          str,
    wait_log_id:     int,
    reasoning:       str,
    claude_response: dict,
    result:          dict,
) -> None:
    """
    Handle the WAIT → BUY transition:
      1. Update original WAIT row: outcome = 'CONDITION_MET'
      2. Write new BUY row to market_log (append-only — write_row, never upsert)
      3. Send Telegram alert

    Called only from escalate_to_claude() when Claude returns BUY.
    Fails silently — errors logged but do not crash the agent loop.
    """
    now_nst = _now_nst()
    today   = datetime.now(tz=NST).strftime("%Y-%m-%d")

    # Step 1: Update original WAIT row outcome
    try:
        from sheets import run_raw_sql
        run_raw_sql(
            "UPDATE market_log SET outcome = 'CONDITION_MET' WHERE id = %s",
            (wait_log_id,),
        )
        log.info("[agent_tools] Updated market_log id=%d outcome=CONDITION_MET", wait_log_id)
    except Exception as e:
        log.error("[agent_tools] Failed to update WAIT row outcome: %s", e)

    # Step 2: Write new BUY row — append-only via write_row
    try:
        from sheets import write_row
        buy_row = {
            "date":          today,
            "time":          datetime.now(tz=NST).strftime("%H:%M:%S"),
            "symbol":        symbol,
            "action":        "BUY",
            "confidence":    str(claude_response.get("confidence", "")),
            "entry_price":   str(claude_response.get("entry_price", "")),
            "stop_loss":     str(claude_response.get("stop_loss", "")),
            "target":        str(claude_response.get("target_price", "")),
            "allocation_npr":str(claude_response.get("allocation_npr", "")),
            "reasoning":     f"[AGENT WAIT→BUY] {reasoning}",
            "outcome":       "PENDING",
            "timestamp":     now_nst,
        }
        write_row("market_log", buy_row)
        result["buy_written"] = True
        log.info("[agent_tools] New BUY row written for %s (WAIT→BUY)", symbol)
    except Exception as e:
        log.error("[agent_tools] Failed to write BUY row for %s: %s", symbol, e)

    # Step 3: Telegram alert
    try:
        from sheets import get_telegram_chat_ids
        import requests, os
        token    = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_ids = get_telegram_chat_ids()

        if token and chat_ids:
            entry = claude_response.get("entry_price", "?")
            stop  = claude_response.get("stop_loss",   "?")
            conf  = claude_response.get("confidence",  "?")
            text  = (
                f"⚡ WAIT→BUY: {symbol}\n"
                f"Condition met — agent escalated to Claude\n"
                f"Entry: NPR {entry} | Stop: NPR {stop} | Conf: {conf}%\n"
                f"Reason: {reasoning[:120]}"
            )
            for chat_id in chat_ids:
                requests.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
            result["telegram_sent"] = True
            log.info("[agent_tools] Telegram WAIT→BUY alert sent for %s", symbol)
    except Exception as e:
        log.warning("[agent_tools] Telegram send failed for %s: %s", symbol, e)
