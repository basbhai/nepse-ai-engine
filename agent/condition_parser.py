"""
agent/condition_parser.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — WAIT Condition Parser

Parses a WAIT row's wait_condition text into structured JSON once, caches
the result in market_log.wait_condition_parsed, and audits the parse in
wait_parse_log.

Public API
----------
    get_parsed_condition(wait_row: dict) -> dict | None

Parsed JSON structure
---------------------
    {
      "requirements": [
        {"type": "indicator", "field": "macd_cross",  "op": "eq",  "value": "BULLISH"},
        {"type": "indicator", "field": "tech_score",  "op": "gte", "value": 75},
        {"type": "indicator", "field": "bb_signal",   "op": "in",  "value": ["UPPER_TOUCH","BREAKOUT"]},
        {"type": "market",    "field": "nepal_score", "op": "gte", "value": 0},
        {"type": "ambiguous", "description": "NEPSE turnover exceeds Rs 4B"}
      ],
      "logic": "ALL"
    }

Supported indicator fields  (from indicators table):
    macd_cross, bb_signal, obv_trend, ema_trend, rsi_14,
    tech_score, macd_histogram, bb_pct_b, volume_ratio

Supported market fields  (from market_state / get_market_state()):
    nepal_score, geo_score, combined_geo, market_state

Ops: eq, neq, gt, gte, lt, lte, in

Rules
-----
- Never calls schema.prisma / SQL — uses sheets.py exclusively.
- On any LLM or parse failure: log warning, return None.  Never raises.
- market_log is append-only for new signals; this module only UPDATEs the
  existing WAIT row (wait_condition_parsed) and INSERTs into wait_parse_log.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from datetime import datetime

from config import NST

log = logging.getLogger(__name__)

# ── Valid vocabulary the parser is allowed to emit ────────────────────────────

VALID_INDICATOR_FIELDS = {
    "macd_cross", "bb_signal", "obv_trend", "ema_trend", "rsi_14",
    "tech_score", "macd_histogram", "bb_pct_b", "volume_ratio",
}

VALID_MARKET_FIELDS = {
    "nepal_score", "geo_score", "combined_geo", "market_state",
}

VALID_OPS = {"eq", "neq", "gt", "gte", "lt", "lte", "in"}

# ── Free-LLM system prompt ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a structured data extractor for a stock trading system.
Respond ONLY in valid JSON. No markdown. No explanation.

You will receive a WAIT condition text that describes what needs to happen
before a stock BUY signal should be acted upon.

Extract it into this JSON structure:
{
  "requirements": [
    {
      "type": "indicator",
      "field": "<one of: macd_cross, bb_signal, obv_trend, ema_trend, rsi_14, tech_score, macd_histogram, bb_pct_b, volume_ratio>",
      "op": "<one of: eq, neq, gt, gte, lt, lte, in>",
      "value": <string, number, or array of strings>
    },
    {
      "type": "market",
      "field": "<one of: nepal_score, geo_score, combined_geo, market_state>",
      "op": "<one of: eq, neq, gt, gte, lt, lte, in>",
      "value": <string, number, or array of strings>
    },
    {
      "type": "ambiguous",
      "description": "<verbatim copy of any condition that cannot be mapped to the above fields>"
    }
  ],
  "logic": "ALL"
}

Rules:
- If a requirement clearly maps to a known field, use type indicator or market.
- If a requirement is vague, narrative, or references external data (e.g. NEPSE turnover),
  use type ambiguous and copy the original text into description.
- Do not invent fields. Only use the exact field names listed above.
- "in" op means the field value must be one of the array elements.
- Numeric values for rsi_14, tech_score, macd_histogram, bb_pct_b must be numbers (not strings).
- macd_cross values: BULLISH, BEARISH, NONE
- bb_signal values: SQUEEZE, EXPANSION, UPPER_TOUCH, LOWER_TOUCH, BREAKOUT, NEUTRAL
- obv_trend values: RISING, FALLING, FLAT
- ema_trend values: ABOVE_ALL, BELOW_ALL, MIXED
- market_state values: BULLISH, SIDEWAYS, BEAR, FULL_BEAR, CRISIS
"""


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL — call free LLM to parse condition text
# ══════════════════════════════════════════════════════════════════════════════

def _parse_via_llm(wait_condition: str) -> tuple[dict | None, str]:
    """
    Call ask_free() to parse the condition text.
    Returns (parsed_dict, model_used) or (None, "") on failure.
    """
    from AI.openrouter import ask_free, _strip_fences

    prompt = (
        f"Parse this WAIT condition into the JSON structure described in your instructions:\n\n"
        f'"{wait_condition}"'
    )

    raw = ask_free(
        prompt  = prompt,
        system  = _SYSTEM_PROMPT,
        context = "condition_parser",
    )

    if not raw:
        log.warning("[condition_parser] ask_free returned None")
        return None, ""

    try:
        cleaned = _strip_fences(raw)
        parsed  = json.loads(cleaned)
    except Exception as exc:
        log.warning("[condition_parser] JSON parse failed: %s | raw: %.120s", exc, raw)
        return None, ""

    return parsed, "free_chain"


def _validate_parsed(parsed: dict) -> bool:
    """
    Lightweight sanity check on the parsed JSON structure.
    Returns False if structure is fundamentally broken.
    """
    if not isinstance(parsed, dict):
        return False
    reqs = parsed.get("requirements")
    if not isinstance(reqs, list) or len(reqs) == 0:
        return False
    for r in reqs:
        if not isinstance(r, dict):
            return False
        t = r.get("type")
        if t not in ("indicator", "market", "ambiguous"):
            return False
        if t == "indicator" and r.get("field") not in VALID_INDICATOR_FIELDS:
            log.warning("[condition_parser] unknown indicator field: %s — treating as ambiguous", r.get("field"))
            r["type"] = "ambiguous"
            r["description"] = str(r)
        if t == "market" and r.get("field") not in VALID_MARKET_FIELDS:
            log.warning("[condition_parser] unknown market field: %s — treating as ambiguous", r.get("field"))
            r["type"] = "ambiguous"
            r["description"] = str(r)
        if t in ("indicator", "market") and r.get("op") not in VALID_OPS:
            log.warning("[condition_parser] unknown op: %s — treating as ambiguous", r.get("op"))
            r["type"] = "ambiguous"
            r["description"] = str(r)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL — persist parse result
# ══════════════════════════════════════════════════════════════════════════════

def _store_parsed(market_log_id: int, parsed: dict, raw_condition: str, model_used: str) -> None:
    """
    1. UPDATE market_log.wait_condition_parsed for this row.
    2. INSERT one row into wait_parse_log for audit.
    Fails silently.
    """
    from sheets import run_raw_sql, write_row

    now_nst   = datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S")
    json_blob = json.dumps(parsed)

    try:
        run_raw_sql(
            "UPDATE market_log SET wait_condition_parsed = %s WHERE id = %s",
            (json_blob, market_log_id),
        )
        log.info("[condition_parser] stored parsed condition for market_log id=%d", market_log_id)
    except Exception as exc:
        log.error("[condition_parser] UPDATE market_log failed (id=%d): %s", market_log_id, exc)

    try:
        write_row("wait_parse_log", {
            "market_log_id": str(market_log_id),
            "parsed_at":     now_nst,
            "raw_condition": raw_condition[:2000],
            "parsed_json":   json_blob[:4000],
            "model_used":    model_used,
        })
    except Exception as exc:
        log.error("[condition_parser] INSERT wait_parse_log failed (id=%d): %s", market_log_id, exc)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_parsed_condition(wait_row: dict) -> dict | None:
    """
    Return the parsed condition JSON for a WAIT row.

    If market_log.wait_condition_parsed is already populated, deserialise and
    return it (cache hit — zero LLM calls).

    Otherwise: call ask_free() to parse, store result to DB, return it.

    On any failure returns None — caller must treat None as "skip this symbol".

    Args:
        wait_row: dict from get_open_waits() — must contain keys:
                  id, symbol, wait_condition, (optionally) wait_condition_parsed

    Returns:
        Parsed dict with shape {"requirements": [...], "logic": "ALL"}
        or None on failure.
    """
    market_log_id  = int(wait_row.get("id", 0))
    symbol         = str(wait_row.get("symbol", "")).upper().strip()
    wait_condition = str(wait_row.get("wait_condition", "")).strip()
    cached_json    = wait_row.get("wait_condition_parsed") or ""

    if not wait_condition:
        log.warning("[condition_parser] %s has empty wait_condition — skip", symbol)
        return None

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cached_json and cached_json not in ("None", "null", ""):
        try:
            parsed = json.loads(cached_json)
            if _validate_parsed(parsed):
                log.info("[condition_parser] cache hit for %s (id=%d)", symbol, market_log_id)
                return parsed
            else:
                log.warning("[condition_parser] cached JSON failed validation for %s — re-parsing", symbol)
        except Exception as exc:
            log.warning("[condition_parser] cached JSON unreadable for %s: %s — re-parsing", symbol, exc)

    # ── Parse via free LLM ────────────────────────────────────────────────────
    log.info("[condition_parser] parsing condition for %s (id=%d): %.80s",
             symbol, market_log_id, wait_condition)

    parsed, model_used = _parse_via_llm(wait_condition)

    if parsed is None:
        log.warning("[condition_parser] LLM parse failed for %s — returning None", symbol)
        return None

    if not _validate_parsed(parsed):
        log.warning("[condition_parser] parsed JSON failed validation for %s — returning None", symbol)
        return None

    # ── Persist ───────────────────────────────────────────────────────────────
    _store_parsed(market_log_id, parsed, wait_condition, model_used)

    req_count = len(parsed.get("requirements", []))
    log.info("[condition_parser] %s parsed: %d requirements", symbol, req_count)
    return parsed
