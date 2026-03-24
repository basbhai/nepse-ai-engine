"""
gemini_filter.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 2
Purpose : Send top candidates from filter_engine.py to Gemini Flash.
          Gemini does a fast contextual screen and returns 1–5 stocks
          worth deep Claude analysis today.

Position in pipeline:
    filter_engine.py   → ranked list of 10–15 candidates (pure math)
    gemini_filter.py   → Gemini Flash contextual screen (fast, ~15 sec)
    claude_analyst.py  → Claude deep analysis per flagged stock (slow, thorough)

Why Gemini Flash here (not Claude):
    - Runs every 6 min during trading hours
    - Needs to be fast and cheap
    - Task is screening, not deep analysis
    - Claude is reserved for the one stock worth acting on

What Gemini Flash does here:
    - Reads all candidate signals + current market context
    - Applies qualitative judgment Claude would waste tokens on
      (e.g. "NABIL has MACD bullish but it's also ex-dividend tomorrow —
       price drop likely, skip")
    - Returns structured JSON: which stocks to send to Claude and why
    - Flags any urgent risks filter_engine math cannot see

What Gemini Flash does NOT do here:
    - Deep analysis (that's Claude's job)
    - Buy/Sell decisions (that's your job)
    - Position sizing (that's budget.py / DeepSeek)

Output written to:
    gemini_filter_log table in Neon (for audit + Learning Hub)
    Returns list[GeminiFlag] to trading loop

─────────────────────────────────────────────────────────────────────────────
Called by: trading.yml every 6 min
Feeds:     claude_analyst.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

NST            = timezone(timedelta(hours=5, minutes=45))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Max candidates to send to Gemini per run (token budget)
MAX_CANDIDATES_TO_GEMINI = 10

# Max stocks Gemini can flag for Claude (keep Claude calls rare and high quality)
MAX_FLAGS_FOR_CLAUDE = 3


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI FLAG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeminiFlag:
    """
    A stock flagged by Gemini Flash for Claude deep analysis.
    Contains Gemini's reasoning and urgency assessment.
    """
    symbol:          str
    sector:          str        = ""
    ltp:             float      = 0.0
    action:          str        = "ANALYZE"    # ANALYZE / SKIP / URGENT
    urgency:         str        = "NORMAL"     # NORMAL / HIGH / URGENT
    gemini_reason:   str        = ""           # why Gemini flagged this
    gemini_risk:     str        = ""           # key risk Gemini identified
    primary_signal:  str        = ""           # MACD / BB / SMA / etc.
    composite_score: float      = 0.0
    tech_score:      int        = 0
    rsi_14:          float      = 0.0
    macd_cross:      str        = ""
    bb_signal:       str        = ""
    best_candle:     str        = ""
    candle_tier:     int        = 0
    suggested_hold:  int        = 17
    geo_combined:    int        = 0
    market_state:    str        = ""

    timestamp: str = field(default_factory=lambda:
                    datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.symbol:<10} [{self.urgency}] score={self.composite_score:.1f} "
            f"signal={self.primary_signal} | {self.gemini_reason[:80]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD LEARNING HUB LESSONS
# ══════════════════════════════════════════════════════════════════════════════

def _load_relevant_lessons(symbols: list[str], limit: int = 8) -> list[str]:
    """
    Load recent Learning Hub lessons relevant to the candidate symbols.
    Gemini uses these to avoid repeating past mistakes.
    Returns list of lesson strings.
    """
    lessons = []
    try:
        from sheets import run_raw_sql

        # Get lessons for specific symbols first, then recent general lessons
        sym_list = "', '".join(symbols)
        rows = run_raw_sql(
            f"""
            SELECT symbol, pattern, lesson, outcome, win_when_applied, applied_count
            FROM learning_hub
            WHERE symbol IN ('{sym_list}') OR symbol = 'MARKET'
            ORDER BY
                CASE WHEN symbol IN ('{sym_list}') THEN 0 ELSE 1 END,
                id DESC
            LIMIT %s
            """,
            (limit,)
        )
        for r in rows:
            sym     = r.get("symbol", "?")
            pattern = r.get("pattern", "")
            lesson  = r.get("lesson", "")[:120]
            outcome = r.get("outcome", "")
            win_rate = r.get("win_when_applied", "")
            applied  = r.get("applied_count", "")
            lessons.append(
                f"{sym} [{pattern}] {outcome}: {lesson}"
                + (f" (win {win_rate}/{applied})" if applied else "")
            )

    except Exception as exc:
        logger.warning("Could not load Learning Hub lessons: %s", exc)

    return lessons


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _load_open_positions() -> list[str]:
    """
    Load currently open positions from portfolio table.
    Gemini uses this to avoid recommending stocks already held
    and to enforce max 3 simultaneous positions rule.
    """
    try:
        from sheets import read_tab
        rows = read_tab("portfolio")
        open_syms = [
            r["symbol"].upper()
            for r in rows
            if r.get("status", "").upper() == "OPEN" and r.get("symbol")
        ]
        return open_syms
    except Exception as exc:
        logger.warning("Could not load open positions: %s", exc)
        return []


def _load_total_capital() -> float:
    """Load total capital from settings."""
    try:
        from sheets import get_setting
        return float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
    except Exception:
        return 100000.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BUILD GEMINI PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    candidates:     list,
    context:        dict,
    lessons:        list[str],
    open_positions: list[str],
    total_capital:  float,
) -> str:
    """
    Build the Gemini Flash screening prompt.
    Compact but complete — Gemini needs context to make good decisions.
    """
    from filter_engine import format_candidate_for_gemini

    nst_now      = datetime.now(tz=NST)
    market_state = context.get("market_state", "SIDEWAYS")
    geo_combined = context.get("combined_geo", 0)
    bandh        = context.get("bandh_today", "NO")
    ipo_drain    = context.get("ipo_drain", "NO")
    crisis       = context.get("crisis_detected", "NO")

    # Open positions section
    positions_str = ", ".join(open_positions) if open_positions else "None"
    slots_remaining = 999 #max(0, 3 - len(open_positions)) for test

    # Candidates section
    candidates_str = "\n".join(
        f"{i+1}. {format_candidate_for_gemini(c)}"
        for i, c in enumerate(candidates[:MAX_CANDIDATES_TO_GEMINI])
    )

    # Learning Hub lessons section
    lessons_str = "\n".join(f"  - {l}" for l in lessons) if lessons else "  No lessons yet"

    prompt = f"""You are a NEPSE stock screener AI. Today is {nst_now.strftime('%Y-%m-%d %H:%M')} NST.
Your job: review these pre-filtered candidates and decide which {MAX_FLAGS_FOR_CLAUDE} (max) 
deserve deep Claude analysis today. Be selective — Claude analysis is expensive.

═══════════════════════════════════════
MARKET CONTEXT
═══════════════════════════════════════
Market State:    {market_state}
Geo Score:       {geo_combined:+d}/10
Bandh Today:     {bandh}
IPO Drain:       {ipo_drain}
Crisis:          {crisis}
Time NST:        {nst_now.strftime('%H:%M')}

═══════════════════════════════════════
PORTFOLIO STATUS
═══════════════════════════════════════
Open Positions:      {positions_str}
Slots Remaining:     {slots_remaining} 
Total Capital:       NPR {total_capital:,.0f}

═══════════════════════════════════════
LEARNING HUB — RECENT LESSONS
═══════════════════════════════════════
{lessons_str}

═══════════════════════════════════════
CANDIDATES (pre-filtered by filter_engine.py)
═══════════════════════════════════════
Field key: SYM=symbol SEC=sector LTP=price CHG=daily% VOL=volume
SCORE=composite TECH=tech_score RSI=rsi[signal] MACD=cross
BB=signal[pct_b] EMA=trend OBV=trend ATR%=volatility CONF=sharesansar
CANDLE=pattern CSTAR=C*signal HOLD=optimal_days SIG=primary_signal

{candidates_str}

═══════════════════════════════════════
SCREENING RULES (apply in order)
═══════════════════════════════════════
1. Don't skip if symbol is already in open positions
2. Skip all non equity shares like Mutual funds, debentures etc
2. SKIP if symbol already analyzed today (avoid duplicate flags)
3. SKIP if market_state is BEAR and signal is not MACD or BB
   (paper: only MACD/BB profitable in bear markets)
4. PREFER MACD cross = BULLISH as primary signal (23.64% ann. return)
5. PREFER BB = LOWER_TOUCH (PF=12.19, highest quality NEPSE signal)
6. DOWNGRADE if RSI is primary signal (lost money -4.81% standalone)
7. UPGRADE if Tier 1 candle pattern with volume confirmed
8. UPGRADE if CSTAR=Y (excess return above C*=0.129)
9. CHECK learning hub — if pattern has <40% win rate for this symbol, skip
10. FLAG max {MAX_FLAGS_FOR_CLAUDE} stocks — quality over quantity
11. If no stock is genuinely worth Claude analysis today, return empty flags
12. search internet for potiential favourable or unfavourable news/conditions (must be creditable and renouned sources)

═══════════════════════════════════════
TASK
═══════════════════════════════════════
For each candidate decide: ANALYZE (send to Claude) or SKIP.
If ANALYZE: assign urgency NORMAL or HIGH or URGENT.
  URGENT = signal may not persist past today (e.g. MACD cross + T1 candle)
  HIGH   = strong setup, act within 1-2 days
  NORMAL = good setup, time to research

Return ONLY this JSON — no markdown, no explanation, no extra text:
{{
  "run_time": "{nst_now.strftime('%Y-%m-%d %H:%M')}",
  "market_state": "{market_state}",
  "slots_remaining": {slots_remaining},
  "flags": [
    {{
      "symbol": "SYMBOL",
      "action": "ANALYZE",
      "urgency": "NORMAL or HIGH or URGENT",
      "reason": "one sentence — why this stock is worth deep analysis",
      "risk": "one sentence — key risk to watch",
      "primary_signal": "MACD or BB or SMA or OBV_MOMENTUM"
    }}
  ],
  "skipped": [
    {{
      "symbol": "SYMBOL",
      "reason": "one sentence — why skipped"
    }}
  ],
  "market_comment": "one sentence about overall market conditions today"
}}"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CALL GEMINI FLASH
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> Optional[dict]:
    """
    Send prompt to Gemini Flash. Returns parsed JSON dict or None on failure.
    Uses google.genai (new SDK — google.generativeai is deprecated).
    """
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set in .env")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        logger.info("Calling Gemini Flash (%s)...", GEMINI_MODEL)

        response = client.models.generate_content(
            model   = GEMINI_MODEL,
            contents= prompt,
            config  = types.GenerateContentConfig(
                system_instruction = (
                    "You are a Nepal stock market screening AI. "
                    "You understand NEPSE trading patterns, Nepal macro context, "
                    "and the research-backed signal weights being used. "
                    "You return only valid JSON — no markdown, no fences, no explanation."
                ),
                response_mime_type = "application/json",
                temperature        = 0.2,   # low = consistent screening decisions
            ),
        )

        raw = response.text.strip()

        # Strip code fences if Gemini adds them despite mime type
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        logger.info(
            "Gemini Flash: %d flags | %d skipped | market=%s",
            len(result.get("flags", [])),
            len(result.get("skipped", [])),
            result.get("market_state", "?"),
        )
        return result

    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.error("Gemini Flash call failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — KEYWORD FALLBACK
# If Gemini unavailable, use rule-based fallback to still produce flags.
# Maintains pipeline continuity when API is down.
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_fallback(
    candidates:     list,
    open_positions: list[str],
    slots_remaining: int,
) -> dict:
    """
    Rule-based fallback when Gemini is unavailable.
    Applies the same screening rules mechanically.
    Returns same structure as Gemini JSON response.
    """
    logger.info("Using keyword fallback for Gemini screening")

    flags   = []
    skipped = []

    for c in candidates[:MAX_CANDIDATES_TO_GEMINI]:
        sym = c.symbol

        # Rule 1: skip open positions
        if sym in open_positions:
            skipped.append({"symbol": sym, "reason": "already in open positions"})
            continue

        # Rule 2: portfolio full
        if slots_remaining == 0 and len(open_positions) >= 99:
            skipped.append({"symbol": sym, "reason": "portfolio full (3/3 positions)"})
            continue

        # Rule 3: prefer MACD/BB as primary signal
        if c.primary_signal == "RSI":
            skipped.append({"symbol": sym, "reason": "RSI as primary signal — lost money standalone in NEPSE"})
            continue

        # Build flag
        if len(flags) >= MAX_FLAGS_FOR_CLAUDE:
            skipped.append({"symbol": sym, "reason": "max flags reached for this run"})
            continue

        # Urgency
        if c.macd_cross == "BULLISH" and c.candle_tier == 1:
            urgency = "URGENT"
        elif c.macd_cross == "BULLISH" or c.bb_signal == "LOWER_TOUCH":
            urgency = "HIGH"
        else:
            urgency = "NORMAL"

        reason = (
            f"{c.primary_signal} signal with score {c.composite_score:.1f}, "
            f"tech={c.tech_score}, conf={c.conf_score:.0f}"
        )
        if c.best_candle:
            reason += f", {c.best_candle}(T{c.candle_tier})"

        flags.append({
            "symbol":         sym,
            "action":         "ANALYZE",
            "urgency":        urgency,
            "reason":         reason,
            "risk":           "verify with Claude — no AI context available (Gemini fallback)",
            "primary_signal": c.primary_signal,
        })

    return {
        "run_time":       datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M"),
        "market_state":   candidates[0].market_state if candidates else "UNKNOWN",
        "slots_remaining": slots_remaining,
        "flags":          flags,
        "skipped":        skipped,
        "market_comment": "Gemini unavailable — keyword fallback used",
        "fallback_used":  True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ASSEMBLE GEMINI FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_flags(
    gemini_result: dict,
    candidates:    list,
) -> list[GeminiFlag]:
    """
    Convert Gemini JSON response into GeminiFlag objects.
    Enriches with full candidate data for claude_analyst.py.
    """
    # Build candidate lookup
    cand_map = {c.symbol: c for c in candidates}

    flags = []
    for f in gemini_result.get("flags", []):
        sym = str(f.get("symbol", "")).upper()
        if not sym:
            continue

        action = str(f.get("action", "ANALYZE")).upper()
        if action != "ANALYZE":
            continue

        c = cand_map.get(sym)
        if not c:
            logger.warning("Gemini flagged %s but not in candidates — skipping", sym)
            continue

        flags.append(GeminiFlag(
            symbol          = sym,
            sector          = c.sector,
            ltp             = c.ltp,
            action          = action,
            urgency         = str(f.get("urgency", "NORMAL")).upper(),
            gemini_reason   = str(f.get("reason", ""))[:200],
            gemini_risk     = str(f.get("risk", ""))[:200],
            primary_signal  = str(f.get("primary_signal", c.primary_signal)),
            composite_score = c.composite_score,
            tech_score      = c.tech_score,
            rsi_14          = c.rsi_14,
            macd_cross      = c.macd_cross,
            bb_signal       = c.bb_signal,
            best_candle     = c.best_candle,
            candle_tier     = c.candle_tier,
            suggested_hold  = c.suggested_hold,
            geo_combined    = c.combined_geo,
            market_state    = c.market_state,
        ))

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def _write_log(
    gemini_result: dict,
    flags:         list[GeminiFlag],
    candidates:    list,
) -> None:
    """
    Write Gemini screening run to market_log table for audit trail.
    One row per flagged stock.
    """
    try:
        from sheets import write_row
        nst_now = datetime.now(tz=NST)

        for flag in flags:
            write_row("market_log", {
                "date":       nst_now.strftime("%Y-%m-%d"),
                "time":       nst_now.strftime("%H:%M:%S"),
                "symbol":     flag.symbol,
                "sector":     flag.sector,
                "action":     f"GEMINI_FLAG_{flag.urgency}",
                "confidence": str(int(flag.composite_score)),
                "entry_price": str(flag.ltp),
                "reasoning":  (
                    f"[Gemini] {flag.gemini_reason} | "
                    f"Risk: {flag.gemini_risk} | "
                    f"Signal: {flag.primary_signal}"
                ),
                "outcome":    "PENDING_CLAUDE",
                "geo_score":  str(flag.geo_combined),
                "rsi_14":     str(flag.rsi_14),
                "candle_pattern": flag.best_candle,
                "timestamp":  flag.timestamp,
            })

        # Log skipped stocks at DEBUG level only (no DB write needed)
        skipped = gemini_result.get("skipped", [])
        if skipped:
            logger.debug("Gemini skipped: %s",
                         ", ".join(s["symbol"] for s in skipped if s.get("symbol")))

        market_comment = gemini_result.get("market_comment", "")
        if market_comment:
            logger.info("Gemini market comment: %s", market_comment)

    except Exception as exc:
        logger.warning("_write_log failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_gemini_filter(
    candidates: list   = None,
    market_data: dict  = None,
    date:        str   = None,
) -> list[GeminiFlag]:
    """
    Main entry point. Called every 6 min by trading.yml
    after filter_engine.run_filter().

    Args:
        candidates:  list[FilterCandidate] from filter_engine.run_filter()
                     If None, runs filter_engine automatically.
        market_data: dict[symbol, PriceRow] — passed to filter_engine if
                     candidates is None.
        date:        Override date YYYY-MM-DD (default: today NST).

    Returns:
        list[GeminiFlag] — stocks to send to claude_analyst.py
        Empty list if nothing worth analyzing today.
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("gemini_filter.run_gemini_filter() — %s", date)

    # ── Get candidates from filter_engine if not provided ────────────────────
    if candidates is None:
        try:
            from filter_engine import run_filter, get_filter_context
            candidates = run_filter(market_data=market_data, top_n=MAX_CANDIDATES_TO_GEMINI, date=date)
        except Exception as exc:
            logger.error("filter_engine failed: %s", exc)
            return []

    if not candidates:
        logger.info("No candidates from filter_engine — nothing to screen")
        return []

    logger.info("Screening %d candidates with Gemini Flash", len(candidates))

    # ── Context ───────────────────────────────────────────────────────────────
    try:
        from filter_engine import get_filter_context
        context = get_filter_context()
    except Exception:
        context = {}

    # ── Portfolio + Learning Hub ──────────────────────────────────────────────
    open_positions  = _load_open_positions()
    slots_remaining =  99#max(0, 3 - len(open_positions))
    total_capital   = _load_total_capital()
    symbols         = [c.symbol for c in candidates]
    lessons         = _load_relevant_lessons(symbols)

    logger.info(
        "Portfolio: %d open | %d slots | capital NPR %.0f | %d lessons loaded",
        len(open_positions), slots_remaining, total_capital, len(lessons),
    )

    # ── Portfolio full check ──────────────────────────────────────────────────
    if slots_remaining == 0 and len(open_positions) >= 99:  #temp disable
        logger.info("Portfolio full (3/3 positions) — no new signals needed")
        return []

    # ── Build prompt and call Gemini ──────────────────────────────────────────
    prompt = _build_prompt(
        candidates     = candidates,
        context        = context,
        lessons        = lessons,
        open_positions = open_positions,
        total_capital  = total_capital,
    )

    gemini_result = _call_gemini(prompt)

    # ── Fallback if Gemini unavailable ────────────────────────────────────────
    if gemini_result is None:
        logger.warning("Gemini unavailable — using keyword fallback")
        gemini_result = _keyword_fallback(candidates, open_positions, slots_remaining)

    # ── Assemble flags ────────────────────────────────────────────────────────
    flags = _assemble_flags(gemini_result, candidates)

    # ── Write audit log ───────────────────────────────────────────────────────
    if flags:
        _write_log(gemini_result, flags, candidates)

    # ── Summary ───────────────────────────────────────────────────────────────
    fallback_note = " [FALLBACK]" if gemini_result.get("fallback_used") else ""
    logger.info(
        "gemini_filter done%s: %d flagged for Claude | %d skipped | comment: %s",
        fallback_note,
        len(flags),
        len(gemini_result.get("skipped", [])),
        gemini_result.get("market_comment", "—")[:80],
    )

    for f in flags:
        logger.info("  FLAG: %s", f.summary())

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HELPER FOR claude_analyst.py
# ══════════════════════════════════════════════════════════════════════════════

def format_flag_for_claude(flag: GeminiFlag) -> str:
    """
    Compact summary of a GeminiFlag for inclusion in Claude analyst prompt.
    Claude gets this as pre-context before doing deep analysis.
    """
    candle = f"{flag.best_candle}(T{flag.candle_tier})" if flag.best_candle else "none"
    return (
        f"SYMBOL:{flag.symbol} SECTOR:{flag.sector} LTP:{flag.ltp:.2f} "
        f"SIGNAL:{flag.primary_signal} URGENCY:{flag.urgency} "
        f"TECH:{flag.tech_score} RSI:{flag.rsi_14:.1f} MACD:{flag.macd_cross} "
        f"BB:{flag.bb_signal} CANDLE:{candle} GEO:{flag.geo_combined:+d} "
        f"REASON:{flag.gemini_reason} RISK:{flag.gemini_risk}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python gemini_filter.py              → full pipeline run
#   python gemini_filter.py --dry-run    → use cached data, no DB write
#   python gemini_filter.py --no-gemini  → force keyword fallback
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GEMINI_FILTER] %(levelname)s: %(message)s",
    )

    args      = sys.argv[1:]
    dry_run   = "--dry-run"   in args
    no_gemini = "--no-gemini" in args

    print("\n" + "=" * 70)
    print("  NEPSE AI — gemini_filter.py")
    print("=" * 70)

    # ── Get candidates ────────────────────────────────────────────────────────
    print("\n[1/3] Running filter_engine...")
    try:
        from filter_engine import run_filter, get_filter_context

        if dry_run:
            from modules.indicators import HistoryCache, DEFAULT_LOAD_PERIODS
            from modules.scraper import PriceRow as PR
            cache = HistoryCache()
            cache.load(periods=DEFAULT_LOAD_PERIODS)
            md = {
                s: PR(
                    symbol=s, ltp=c[-1],
                    open_price=c[-2] if len(c) > 1 else c[-1],
                    close=c[-1],
                    high=cache.get_highs(s)[-1] if cache.get_highs(s) else c[-1],
                    low=cache.get_lows(s)[-1]   if cache.get_lows(s)  else c[-1],
                    prev_close=c[-2] if len(c) > 1 else c[-1],
                    volume=int(cache.get_volumes(s)[-1]) if cache.get_volumes(s) else 10000,
                    conf_score=55.0, conf_signal="BULLISH", change_pct=0.5,
                )
                for s, c in cache.closes.items() if c
            }
            candidates = run_filter(market_data=md, top_n=MAX_CANDIDATES_TO_GEMINI)
        else:
            from modules.scraper import get_all_market_data
            md = get_all_market_data(write_breadth=False)
            candidates = run_filter(market_data=md, top_n=MAX_CANDIDATES_TO_GEMINI)

        print(f"  ✅ {len(candidates)} candidates from filter_engine")
        if candidates:
            for c in candidates[:5]:
                print(f"     {c.summary()}")

    except Exception as e:
        print(f"  ❌ filter_engine failed: {e}")
        sys.exit(1)

    if not candidates:
        print("\n  No candidates — nothing to screen today")
        sys.exit(0)

    # ── Load context ──────────────────────────────────────────────────────────
    print("\n[2/3] Loading context...")
    context         = get_filter_context()
    open_positions  = _load_open_positions()
    slots_remaining =  999            #max(0, 3 - len(open_positions)) for testing
    total_capital   = _load_total_capital()
    lessons         = _load_relevant_lessons([c.symbol for c in candidates])

    print(f"  Open positions: {open_positions or 'None'}")
    print(f"  Slots remaining: {slots_remaining}")
    print(f"  Capital: NPR {total_capital:,.0f}")
    print(f"  Lessons loaded: {len(lessons)}")

    if slots_remaining == 0:
        print("\n  Portfolio full — no new signals needed")
        sys.exit(0)

    # ── Gemini screening ──────────────────────────────────────────────────────
    print(f"\n[3/3] {'Keyword fallback (--no-gemini)' if no_gemini else 'Gemini Flash screening'}...")

    if no_gemini:
        gemini_result = _keyword_fallback(candidates, open_positions, slots_remaining)
    else:
        prompt        = _build_prompt(candidates, context, lessons, open_positions, total_capital)
        gemini_result = _call_gemini(prompt)
        if gemini_result is None:
            print("  ⚠️  Gemini unavailable — keyword fallback")
            gemini_result = _keyword_fallback(candidates, open_positions, slots_remaining)

    flags = _assemble_flags(gemini_result, candidates)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n  Market comment: {gemini_result.get('market_comment', '—')}")
    print()

    if flags:
        print(f"  ✅ {len(flags)} stock(s) flagged for Claude analysis:\n")
        for f in flags:
            print(f"  {'🚨' if f.urgency == 'URGENT' else '⚡' if f.urgency == 'HIGH' else '📊'} "
                  f"{f.symbol} [{f.urgency}]")
            print(f"     Signal:  {f.primary_signal} | Score: {f.composite_score:.1f} | "
                  f"Tech: {f.tech_score} | RSI: {f.rsi_14:.1f}")
            print(f"     Reason:  {f.gemini_reason}")
            print(f"     Risk:    {f.gemini_risk}")
            print(f"     Hold:    ~{f.suggested_hold} days")
            print()

        print(f"  Claude-ready format:")
        print("  " + "─" * 60)
        for f in flags:
            print(f"  {format_flag_for_claude(f)}")
    else:
        print("  No stocks flagged for Claude today.")
        print("  Skipped:")
        for s in gemini_result.get("skipped", [])[:5]:
            print(f"    {s.get('symbol','?')}: {s.get('reason','')}")

    if not dry_run and flags:
        _write_log(gemini_result, flags, candidates)
        print(f"\n  ✅ Audit log written to market_log table")

    print("\n" + "=" * 70 + "\n")
