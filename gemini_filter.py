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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from config import NST, GEMINI_API_KEY, GEMINI_MODEL
from filter_engine import run_filter, get_filter_context, get_last_near_misses
from AI import ask_gemini_json, ask_gemini_text

logger = logging.getLogger(__name__)

# Max candidates to send to Gemini per run (token budget)
MAX_CANDIDATES_TO_GEMINI = 10

# Max stocks Gemini can flag for Claude (keep Claude calls rare and high quality)
MAX_FLAGS_FOR_CLAUDE = 3


# ══════════════════════════════════════════════════════════════════════════════
# NEAR-MISS FLUSH  (Fix 1)
# ══════════════════════════════════════════════════════════════════════════════

def flush_near_misses_to_db() -> None:
    """
    Write NearMiss objects captured by the last run_filter() call to gate_misses.
    Upsert on (symbol, date) — re-running same day overwrites, never duplicates.
    Called automatically after every run_filter() in this module.
    """
    from sheets import upsert_row
    misses = get_last_near_misses()
    if not misses:
        return
    written = 0
    for m in misses:
        try:
            upsert_row(
                "gate_misses",
                {
                    "symbol":                   m.symbol,
                    "sector":                   m.sector,
                    "date":                     m.date,
                    "gate_reason":              m.gate_reason,
                    "gate_category":            m.gate_category,
                    "price_at_block":           str(m.price_at_block) if m.price_at_block else None,
                    "market_state":             m.market_state,
                    "tech_score":               str(m.tech_score),
                    "conf_score":               str(m.conf_score),
                    "composite_score_would_be": str(m.composite_score_would_be),
                    "volume_os_ratio":          str(m.volume_os_ratio) if hasattr(m, "volume_os_ratio") else "0",
                    "outcome":                  None,
                    "tracking_days":            "0",
                },
                conflict_columns=["symbol", "date"],
            )
            written += 1
        except Exception as exc:
            logger.warning("flush_near_misses_to_db: failed for %s — %s", m.symbol, exc)
    logger.info(
        "flush_near_misses_to_db: wrote %d/%d near-misses to gate_misses",
        written, len(misses),
    )


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
    obv_trend:       str        = ""
    ema_trend:       str        = "" 
    best_candle:     str        = ""
    candle_tier:     int        = 0
    suggested_hold:  int        = 17
    geo_combined:    int        = 0
    market_state:    str        = ""
    support_level:   float      = 0.0
    resistance_level:float      = 0.0
    market_log_id:   int        = None

    # ── Full FilterCandidate passthrough ──────────────────────────────────────
    change_pct:       float      = 0.0
    volume:           int        = 0
    rsi_signal:       str        = ""
    ema_20_50_cross:  str        = ""
    ema_50_200_cross: str        = ""
    macd_histogram:   float      = 0.0
    macd_line:        float      = 0.0   # MACD line value (from FilterCandidate)
    macd_signal_line: float      = 0.0   # MACD signal line value
    bb_pct_b:         float      = 0.5
    bb_upper:         float      = 0.0
    bb_lower:         float      = 0.0
    atr_pct:          float      = 0.0
    tech_signal:      str        = ""
    conf_score:       float      = 0.0
    candle_conf:      int        = 0
    geo_score:        int        = 0
    nepal_score:      int        = 0
    bandh_today:      str        = "NO"
    crisis_detected:  str        = "NO"
    ipo_drain:        str        = "NO"
    sector_mult:      float      = 1.0
    cstar_signal:     bool       = False
    fundamental_adj:  float      = 0.0
    fundamental_reason: str      = ""

    vwap_dev:         float      = 0.0
    bid_ask_ratio:    float      = 0.0
    dpr_proximity:    float      = 0.0
    volume_os_ratio:  float      = 0.0

    timestamp: str = field(default_factory=lambda:
                    datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.symbol:<10} [{self.urgency}] score={self.composite_score:.1f} "
            f"signal={self.primary_signal} | {self.gemini_reason[:750]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD LEARNING HUB LESSONS
# ══════════════════════════════════════════════════════════════════════════════

def _load_relevant_lessons(symbols: list[str], limit: int = 8) -> list[str]:
    """
    Load recent Learning Hub lessons relevant to the candidate symbols.
    Gemini uses these to avoid repeating past mistakes.
    """
    lessons = []
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT condition, finding, action, confidence_level
            FROM learning_hub
            WHERE active = 'true'
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )
        for r in (rows or []):
            condition  = r.get("condition", "")
            finding    = (r.get("finding") or "")[:500]
            action     = r.get("action", "")
            confidence = r.get("confidence_level", "")
            lessons.append(
                f"[{action}] If {condition}: {finding}"
                + (f" (confidence: {confidence})" if confidence else "")
            )
    except Exception as exc:
        logger.warning("Could not load Learning Hub lessons: %s", exc)
    return lessons
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD OPEN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _load_open_positions() -> list[str]:
    """Load currently open positions from portfolio table."""
    try:
        from sheets import read_tab
        rows = read_tab("portfolio")
        return [
            r["symbol"].upper()
            for r in rows
            if r.get("status", "").upper() == "OPEN" and r.get("symbol")
        ]
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
    from filter_engine import format_candidate_for_gemini

    nst_now      = datetime.now(tz=NST)
    market_state = context.get("market_state", "SIDEWAYS")
    geo_combined = context.get("combined_geo", 0)
    bandh        = context.get("bandh_today", "NO")
    ipo_drain    = context.get("ipo_drain", "NO")
    crisis       = context.get("crisis_detected", "NO")

    positions_str   = ", ".join(open_positions) if open_positions else "None"
    slots_remaining = 999  # max(0, 3 - len(open_positions)) for test

    candidates_str = "\n".join(
        f"{i+1}. {format_candidate_for_gemini(c)} "
        f"VWAPD={getattr(c, 'vwap_dev', 0.0):+.1f}% "
        f"BAR={getattr(c, 'bid_ask_ratio', 0.0):.2f} "
        f"DPRP={getattr(c, 'dpr_proximity', 0.0):.2f} "
        f"VOS={getattr(c, 'volume_os_ratio', 0.0):.2f}%OS"
        for i, c in enumerate(candidates[:MAX_CANDIDATES_TO_GEMINI])
    )

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
VWAPD=vwap_dev% BAR=bid_ask_ratio DPRP=dpr_proximity VOS=vol_pct_of_outstanding_shares

{candidates_str}

═══════════════════════════════════════
SCREENING RULES (apply in order)
═══════════════════════════════════════
1. Don't skip if symbol is already in open positions (evaluate EXIT/RE-ENTRY)
2. Filter out if the stock's sector belongs to mutual fund or debentures
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
13. APPLY LAGGARD LOGIC:
    - Identify the 'Sector Leader' (highest % CHG and VOL in sector).
    - If a candidate is in the same sector, has a lower RSI, but VOL > 1.5x avg,
      UPGRADE to 'ANALYZE'. This is a 'Catch-up Play'.
14. UPGRADE to URGENT if VOS > 1.0% AND primary_signal is LAGGARD_PLAY or VOLUME_BREAKOUT
    (smart money / operator accumulation detected)

═══════════════════════════════════════
TASK
═══════════════════════════════════════
1. Search the internet for latest news/conditions on current candidates.
2. For each candidate decide: ANALYZE or SKIP based on rules above.
3. If ANALYZE: assign urgency NORMAL or HIGH or URGENT.

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
      "reason": "3 sentence why it is worth watching citing specific volume (search web for history txn  if required)  or laggard setup. also take account of added technical details",
      "risk": "Key risk or support level (e.g., NHPC support at 300)",
      "primary_signal": "VOLUME_BREAKOUT or LAGGARD_PLAY"
    }}
  ],
  "skipped": [
    {{
      "symbol": "SYMBOL",
      "reason": "One sentence — why skipped"
    }}
  ],
  "market_comment": "One sentence summary of NEPSE's momentum today."
}}
"""

    return prompt



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — KEYWORD FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_fallback(
    candidates:      list,
    open_positions:  list[str],
    slots_remaining: int,
) -> dict:
    """Rule-based fallback when Gemini is unavailable."""
    logger.info("Using keyword fallback for Gemini screening")

    flags   = []
    skipped = []

    for c in candidates[:MAX_CANDIDATES_TO_GEMINI]:
        sym = c.symbol

        if sym in open_positions:
            skipped.append({"symbol": sym, "reason": "already in open positions"})
            continue

        if slots_remaining == 0 and len(open_positions) >= 99:
            skipped.append({"symbol": sym, "reason": "portfolio full (3/3 positions)"})
            continue

        if c.primary_signal == "RSI":
            skipped.append({"symbol": sym, "reason": "RSI as primary signal — lost money standalone in NEPSE"})
            continue

        if len(flags) >= MAX_FLAGS_FOR_CLAUDE:
            skipped.append({"symbol": sym, "reason": "max flags reached for this run"})
            continue

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
        "run_time":        datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M"),
        "market_state":    candidates[0].market_state if candidates else "UNKNOWN",
        "slots_remaining": slots_remaining,
        "flags":           flags,
        "skipped":         skipped,
        "market_comment":  "Gemini unavailable — keyword fallback used",
        "fallback_used":   True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ASSEMBLE GEMINI FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_flags(
    gemini_result: dict,
    candidates:    list,
) -> list[GeminiFlag]:
    """Convert Gemini JSON response into GeminiFlag objects."""
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
            symbol           = sym,
            sector           = c.sector,
            ltp              = c.ltp,
            action           = action,
            urgency          = str(f.get("urgency", "NORMAL")).upper(),
            gemini_reason    = str(f.get("reason", ""))[:200],
            gemini_risk      = str(f.get("risk", ""))[:200],
            primary_signal   = str(f.get("primary_signal", c.primary_signal)),
            composite_score  = c.composite_score,
            tech_score       = c.tech_score,
            obv_trend        = c.obv_trend,
            ema_trend        = c.ema_trend,
            rsi_14           = c.rsi_14,
            macd_cross       = c.macd_cross,
            bb_signal        = c.bb_signal,
            best_candle      = c.best_candle,
            candle_tier      = c.candle_tier,
            suggested_hold   = c.suggested_hold,
            geo_combined     = c.combined_geo,
            market_state     = c.market_state,
            support_level    = c.support_level,
            resistance_level = c.resistance_level,
            change_pct       = c.change_pct,
            volume           = c.volume,
            rsi_signal       = c.rsi_signal,
            ema_20_50_cross  = c.ema_20_50_cross,
            ema_50_200_cross = c.ema_50_200_cross,
            macd_histogram   = c.macd_histogram,
            bb_pct_b         = c.bb_pct_b,
            bb_upper         = float(getattr(c, "bb_upper",  0.0) or 0.0),
            bb_lower         = float(getattr(c, "bb_lower",  0.0) or 0.0),
            macd_line        = float(getattr(c, "macd_line",        0.0) or 0.0),
            macd_signal_line = float(getattr(c, "macd_signal_line", 0.0) or 0.0),
            atr_pct          = c.atr_pct,
            tech_signal      = c.tech_signal,
            conf_score       = c.conf_score,
            candle_conf      = c.candle_conf,
            geo_score        = c.geo_score,
            nepal_score      = c.nepal_score,
            bandh_today      = c.bandh_today,
            crisis_detected  = c.crisis_detected,
            ipo_drain        = c.ipo_drain,
            sector_mult      = c.sector_mult,
            cstar_signal     = c.cstar_signal,
            fundamental_adj  = c.fundamental_adj,
            fundamental_reason = c.fundamental_reason,

            vwap_dev        = float(getattr(c, "vwap_dev",        0.0) or 0.0),
            bid_ask_ratio   = float(getattr(c, "bid_ask_ratio",   0.0) or 0.0),
            dpr_proximity   = float(getattr(c, "dpr_proximity",   0.0) or 0.0),
            volume_os_ratio = float(getattr(c, "volume_os_ratio", 0.0) or 0.0),
        ))

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6— WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def _write_log(
    gemini_result: dict,
    flags:         list[GeminiFlag],
    candidates:    list,
) -> None:
    """
    Write Gemini screening run to market_log table for audit trail.
    Conflict key: action LIKE 'GEMINI_FLAG%' + symbol + date — never
    overwrites Claude BUY/WAIT/AVOID rows written earlier today.
    """
    try:
        from sheets import write_row, run_raw_sql
        nst_now = datetime.now(tz=NST)
        today   = nst_now.strftime("%Y-%m-%d")

        for flag in flags:
            existing = run_raw_sql(
                """
                SELECT id FROM market_log
                WHERE symbol = %s
                  AND date   = %s
                  AND action LIKE 'GEMINI_FLAG%%'
                ORDER BY id DESC
                LIMIT 1
                """,
                (flag.symbol, today)
            )

            if existing:
                flag.market_log_id = existing[0]["id"]
                logger.debug(
                    "%s: GEMINI_FLAG row already exists today (id=%s) — reusing",
                    flag.symbol, flag.market_log_id,
                )
                continue

            write_row("market_log", {
                "date":           today,
                "time":           nst_now.strftime("%H:%M:%S"),
                "symbol":         flag.symbol,
                "sector":         flag.sector,
                "action":         f"GEMINI_FLAG_{flag.urgency}",
                "confidence":     str(int(flag.composite_score)),
                "entry_price":    str(flag.ltp),
                "reasoning":      (
                    f"[Gemini] {flag.gemini_reason} | "
                    f"Risk: {flag.gemini_risk} | "
                    f"Signal: {flag.primary_signal}"
                ),
                "outcome":        "PENDING_CLAUDE",
                "geo_score":      str(flag.geo_combined),
                "rsi_14":         str(flag.rsi_14),
                "candle_pattern": flag.best_candle,
                "timestamp":      flag.timestamp,
                "vwap_dev":        str(flag.vwap_dev),
                "bid_ask_ratio":   str(flag.bid_ask_ratio),
                "dpr_proximity":   str(flag.dpr_proximity),
                "volume_os_ratio": str(flag.volume_os_ratio),
            })

            rows = run_raw_sql(
                """
                SELECT id FROM market_log
                WHERE symbol = %s
                  AND date   = %s
                  AND action LIKE 'GEMINI_FLAG%%'
                ORDER BY id DESC
                LIMIT 1
                """,
                (flag.symbol, today)
            )
            flag.market_log_id = rows[0]["id"] if rows else None
            logger.debug("%s: GEMINI_FLAG row written (id=%s)", flag.symbol, flag.market_log_id)

        market_comment = gemini_result.get("market_comment", "")
        if market_comment:
            logger.info("Gemini market comment: %s", market_comment)

    except Exception as exc:
        logger.warning("_write_log failed: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_gemini_filter(
    candidates:  list  = None,
    market_data: dict  = None,
    date:        str   = None,
) -> list[GeminiFlag]:
    """
    Main entry point. Called every 6 min by trading.yml
    after filter_engine.run_filter().
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("gemini_filter.run_gemini_filter() — %s", date)

    # ── Get candidates from filter_engine if not provided ────────────────────
    if candidates is None:
        try:
            candidates = run_filter(
                market_data=market_data,
                top_n=MAX_CANDIDATES_TO_GEMINI,
                date=date,
            )
            flush_near_misses_to_db()   # Fix 1 — always flush after run_filter()
        except Exception as exc:
            logger.error("filter_engine failed: %s", exc)
            return []
    else:
        # candidates were provided externally — still flush whatever run_filter
        # captured in its last call (trading loop may have called it separately)
        flush_near_misses_to_db()       # Fix 1

    if not candidates:
        logger.info("No candidates from filter_engine — nothing to screen")
        return []

    logger.info("Screening %d candidates with Gemini Flash", len(candidates))

    # ── Context ───────────────────────────────────────────────────────────────
    try:
        context = get_filter_context()
    except Exception:
        context = {}

    # ── Portfolio ─────────────────────────────────────────────────────────────
    open_positions  = _load_open_positions()
    slots_remaining = 999  # max(0, 3 - len(open_positions)) for test
    total_capital   = _load_total_capital()
    symbols         = [c.symbol for c in candidates]
    lessons         = _load_relevant_lessons(symbols)

    logger.info(
        "Portfolio: %d open | %d slots | capital NPR %.0f | %d lessons loaded",
        len(open_positions), slots_remaining, total_capital, len(lessons),
    )

    if slots_remaining == 0 and len(open_positions) >= 99:
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

    gemini_result = ask_gemini_json(
        prompt,
        system  = (
            "You are a Nepal stock market screening AI. "
            "You understand NEPSE trading patterns, Nepal macro context, "
            "and the research-backed signal weights being used. "
            "You return only valid JSON — no markdown, no fences, no explanation."
        ),
        context    = "gemini_filter",
        use_search = True,
    )

    if gemini_result is None:
        logger.warning("Gemini unavailable — skipping Claude this cycle, no fallback")
        return []

    # ── Assemble + log ────────────────────────────────────────────────────────
    flags = _assemble_flags(gemini_result, candidates)

    if flags:
        _write_log(gemini_result, flags, candidates)

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
# SECTION 8 — HELPER FOR claude_analyst.py
# ══════════════════════════════════════════════════════════════════════════════

def format_flag_for_claude(flag: GeminiFlag) -> str:
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
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    args      = sys.argv[1:]
    dry_run   = "--dry-run"   in args
    no_gemini = "--no-gemini" in args

    print("\n" + "=" * 70)
    print("  NEPSE AI — gemini_filter.py")
    print("=" * 70)

    print("\n[1/3] Running filter_engine...")
    try:
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

        flush_near_misses_to_db()  # Fix 1
        print(f"  ✅ {len(candidates)} candidates from filter_engine")
        for c in candidates[:5]:
            print(f"     {c.summary()}")

    except Exception as e:
        print(f"  ❌ filter_engine failed: {e}")
        sys.exit(1)

    if not candidates:
        print("\n  No candidates — nothing to screen today")
        sys.exit(0)

    print("\n[2/3] Loading context...")
    context         = get_filter_context()
    open_positions  = _load_open_positions()
    slots_remaining = 999
    total_capital   = _load_total_capital()
    lessons         = _load_relevant_lessons([c.symbol for c in candidates])

    print(f"  Open positions:  {open_positions or 'None'}")
    print(f"  Slots remaining: {slots_remaining}")
    print(f"  Capital:         NPR {total_capital:,.0f}")
    print(f"  Lessons loaded:  {len(lessons)}")

    if slots_remaining == 0:
        print("\n  Portfolio full — no new signals needed")
        sys.exit(0)

    print(f"\n[3/3] {'Keyword fallback (--no-gemini)' if no_gemini else 'Gemini Flash screening'}...")

    if no_gemini:
        gemini_result = _keyword_fallback(candidates, open_positions, slots_remaining)
    else:
        prompt        = _build_prompt(candidates, context, lessons, open_positions, total_capital)
        gemini_result = ask_gemini_json(
            prompt,
            system  = (
                "You are a Nepal stock market screening AI. "
                "You understand NEPSE trading patterns, Nepal macro context, "
                "and the research-backed signal weights being used. "
                "You return only valid JSON — no markdown, no fences, no explanation."
            ),
            context    = "gemini_filter",
            use_search = True,
        )
        if gemini_result is None:
            print("  ⚠️  Gemini unavailable — no fallback, exiting")
            sys.exit(0)

    flags = _assemble_flags(gemini_result, candidates)

    print(f"\n  Market comment: {gemini_result.get('market_comment', '—')}")
    print()

    if flags:
        print(f"  ✅ {len(flags)} stock(s) flagged for Claude analysis:\n")
        for f in flags:
            print(
                f"  {'🚨' if f.urgency == 'URGENT' else '⚡' if f.urgency == 'HIGH' else '📊'} "
                f"{f.symbol} [{f.urgency}]"
            )
            print(f"     Signal:  {f.primary_signal} | Score: {f.composite_score:.1f} | "
                  f"Tech: {f.tech_score} | RSI: {f.rsi_14:.1f}")
            print(f"     Reason:  {f.gemini_reason}")
            print(f"     Risk:    {f.gemini_risk}")
            print(f"     Hold:    ~{f.suggested_hold} days")
            print()
        print("  Claude-ready format:")
        print("  " + "─" * 60)
        for f in flags:
            print(f"  {format_flag_for_claude(f)}")
    else:
        print("  No stocks flagged for Claude today.")
        for s in gemini_result.get("skipped", [])[:5]:
            print(f"    {s.get('symbol','?')}: {s.get('reason','')}")

    if not dry_run and flags:
        _write_log(gemini_result, flags, candidates)
        print(f"\n  ✅ Audit log written to market_log table")

    print("\n" + "=" * 70 + "\n")