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

# Fallback defaults — overridden at runtime by settings table via get_filter_context()
# Update live via: update_setting("GEMINI_MAX_CANDIDATES", "12") etc.
MAX_CANDIDATES_TO_GEMINI = 10   # → setting key: GEMINI_MAX_CANDIDATES
MAX_FLAGS_FOR_CLAUDE     = 3    # → setting key: GEMINI_MAX_FLAGS


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

    NON_TRACKABLE = {"MUTUAL_FUND", "NON_EQUITY", "HISTORY", "NO_LTP"}

    misses = get_last_near_misses()
    if not misses:
        return

    written  = 0
    skipped  = 0
    for m in misses:
        # Skip categories that can never be meaningfully evaluated
        if m.gate_category in NON_TRACKABLE:
            skipped += 1
            continue
        # Skip rows with no valid price — can never compute return
        if not m.price_at_block or m.price_at_block == 0.0:
            skipped += 1
            continue
        try:
            upsert_row(
                "gate_misses",
                {
                    "symbol":                   m.symbol,
                    "sector":                   m.sector,
                    "date":                     m.date,
                    "gate_reason":              m.gate_reason,
                    "gate_category":            m.gate_category,
                    "price_at_block":           str(m.price_at_block),
                    "market_state":             m.market_state,
                    "tech_score":               str(m.tech_score),
                    "conf_score":               str(m.conf_score),
                    "composite_score_would_be": str(m.composite_score_would_be),
                    "volume_os_ratio":          str(m.volume_os_ratio) if hasattr(m, "volume_os_ratio") else "0",
                    "vwap_dev":                 str(m.vwap_dev),
                    "bid_ask_ratio":            str(m.bid_ask_ratio),
                    "dpr_proximity":            str(m.dpr_proximity),
                    "outcome":                  None,
                    "tracking_days":            "0",
                    "engine_source":            getattr(m, "engine_source", "shared") or "shared",
                },
                conflict_columns=["symbol", "date"],
            )
            written += 1
        except Exception as exc:
            logger.warning("flush_near_misses_to_db: failed for %s — %s", m.symbol, exc)

    logger.info(
        "flush_near_misses_to_db: wrote %d/%d near-misses to gate_misses (skipped %d non-trackable/no-price)",
        written, len(misses), skipped,
    )

# ══════════════════════════════════════════════════════════════════════════════
# GEMINI SKIP FLUSH
# ══════════════════════════════════════════════════════════════════════════════

def flush_gemini_skips_to_db(
    gemini_result: dict,
    candidates: list,
    date: str = None,
) -> None:
    """
    Write Gemini-skipped symbols from the `skipped` array in gemini_result to gate_misses.
    Upsert on (symbol, date) — same conflict key as existing near-miss rows.
    Fail silently — never blocks the pipeline.
    """
    from sheets import upsert_row

    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    skipped_list = gemini_result.get("skipped", [])
    if not skipped_list:
        return

    cand_map = {c.symbol: c for c in candidates}
    written  = 0
    skipped  = 0

    for item in skipped_list:
        sym    = str(item.get("symbol", "")).upper()
        reason = str(item.get("reason", ""))[:500]
        if not sym:
            skipped += 1
            continue

        c = cand_map.get(sym)
        if not c:
            skipped += 1
            continue

        price = float(getattr(c, "ltp", 0) or 0)
        if not price:
            skipped += 1
            continue

        try:
            upsert_row(
                "gate_misses",
                {
                    "symbol":                   sym,
                    "date":                     date,
                    "sector":                   getattr(c, "sector", "") or "",
                    "gate_reason":              reason,
                    "gate_category":            "GEMINI_SKIP",
                    "decision":                 "GEMINI_SKIP",
                    "price_at_block":           str(price),
                    "market_state":             getattr(c, "market_state", "") or "",
                    "tech_score":               str(getattr(c, "tech_score", "") or ""),
                    "conf_score":               str(getattr(c, "conf_score", "") or ""),
                    "composite_score_would_be": str(getattr(c, "composite_score", "") or ""),
                    "volume_os_ratio":          str(getattr(c, "volume_os_ratio", 0) or 0),
                    "vwap_dev":                 str(getattr(c, "vwap_dev",        0) or 0),
                    "bid_ask_ratio":            str(getattr(c, "bid_ask_ratio",   0) or 0),
                    "dpr_proximity":            str(getattr(c, "dpr_proximity",   0) or 0),
                    "outcome":                  None,
                    "tracking_days":            "0",
                },
                conflict_columns=["symbol", "date"],
            )
            written += 1
        except Exception as exc:
            logger.warning("flush_gemini_skips_to_db: failed for %s — %s", sym, exc)
            skipped += 1

    logger.info(
        "flush_gemini_skips_to_db: wrote %d/%d Gemini SKIPs to gate_misses (skipped %d no-candidate/no-price)",
        written, len(skipped_list), skipped,
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
    pivot_r1:        float      = 0.0
    pivot_r2:        float      = 0.0
    pivot_r3:        float      = 0.0
    pivot_s1:        float      = 0.0
    pivot_s2:        float      = 0.0
    pivot_s3:        float      = 0.0
    market_log_id:   int        = None
    intraday_trend:  str        = ""  # Gemini breadth-based trend: ACCUMULATING|DISTRIBUTING|RECOVERING|FADING|CHOPPY|EARLY_SESSION
    sector_momentum: str        = ""  # compact sector context line for this flag's sector
    news_catalyst:   str        = ""  # from nepal_pulse stock_specific_catalysts

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

    engine_source:      str      = "v1"
    composite_score_v2: float    = 0.0
    primary_signal_v2:  str      = ""
    co_flagged_by:      str      = ""

    vwap_dev:         float      = 0.0
    bid_ask_ratio:    float      = 0.0
    dpr_proximity:    float      = 0.0
    volume_os_ratio:  float      = 0.0

    # ── Momentum passthrough from FilterCandidate ────────────────────────────
    momentum_status:  str   = "NEUTRAL"
    rsi_slope_3d:     float = 0.0
    macd_hist_slope:  float = 0.0
    bb_pct_b_slope:   float = 0.0
    bounce_failed:    bool  = False
    reversal_days:    int   = 0

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
              AND (consumer = 'ALL' OR consumer = 'gemini_only')
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


def _compute_sector_momentum(
    market_data: dict,
    candidate_sectors: set,
    sector_map: dict,
) -> str:
    """
    Compute sector momentum context block from full market_data.
    Only computes for sectors present in shortlisted candidates.
    Returns compact pipe-delimited string for prompt injection.
    No DB hits — pure in-memory computation from market_data.
    """
    try:
        if not market_data or not candidate_sectors:
            return "none — no candidate sectors identified"

        nst_now = datetime.now(tz=NST)
        total_min = nst_now.hour * 60 + nst_now.minute
        if total_min < 11 * 60 + 30:
            session_phase = "OPEN"
        elif total_min <= 13 * 60 + 30:
            session_phase = "MID"
        else:
            session_phase = "CLOSE"

        sector_symbols: dict = {}
        for sym, row in market_data.items():
            sec = sector_map.get(sym.upper(), "others")
            if sec not in sector_symbols:
                sector_symbols[sec] = []
            sector_symbols[sec].append((sym.upper(), row))

        candidate_sectors_lower = {s.lower() for s in candidate_sectors if s}

        lines = []
        for sector in sorted(candidate_sectors_lower):
            members = sector_symbols.get(sector, [])
            if not members:
                continue
            total = len(members)
            pos = sum(1 for _, r in members if (getattr(r, "change_pct", None) or 0.0) > 0)
            avg_chg = (
                sum((getattr(r, "change_pct", None) or 0.0) for _, r in members) / total
            )
            leader_sym = ""
            l_chg = 0.0
            best_comp = float("-inf")
            for sym, r in members:
                chg = getattr(r, "change_pct", None) or 0.0
                vol = getattr(r, "volume", None) or 0
                comp = chg * vol
                if comp > best_comp:
                    best_comp = comp
                    leader_sym = sym
                    l_chg = chg
            leader_gap = l_chg - avg_chg
            circuit_locked = "YES" if l_chg >= 9.5 else "NO"
            lines.append(
                f"sector={sector} | pos={pos}/{total} | avg_chg={avg_chg:+.2f}% | "
                f"leader={leader_sym} | l_chg={l_chg:+.2f}% | "
                f"leader_gap={leader_gap:+.2f}% | circuit_locked={circuit_locked} | "
                f"session_phase={session_phase}"
            )

        if not lines:
            return "none — no candidate sectors identified"
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("_compute_sector_momentum error (non-fatal): %s", exc)
        return ""


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
    candidates:           list,
    context:              dict,
    lessons:              list[str],
    open_positions:       list[str],
    total_capital:        float,
    max_candidates:       int  = MAX_CANDIDATES_TO_GEMINI,
    max_flags:            int  = MAX_FLAGS_FOR_CLAUDE,
    market_data:          dict = None,
    sector_momentum_map:  dict = None,
) -> str:
    from filter_engine import format_candidate_for_gemini

    nst_now      = datetime.now(tz=NST)
    market_state = context.get("market_state", "SIDEWAYS")
    geo_combined = context.get("combined_geo", 0)
    bandh        = context.get("bandh_today", "NO")
    ipo_drain    = context.get("ipo_drain", "NO")
    crisis       = context.get("crisis_detected", "NO")
    today_date   = nst_now.strftime("%Y-%m-%d")

    # ── Intraday breadth timeline ─────────────────────────────────────────────
    try:
        from sheets import get_intraday_breadth
        breadth_rows = get_intraday_breadth(today_date)
    except Exception:
        breadth_rows = []

    if breadth_rows:
        breadth_lines = []
        for r in breadth_rows:
            ts    = str(r.get("timestamp", ""))[11:16]   # HH:MM
            adv   = r.get("advancing",     "?")
            dec   = r.get("declining",     "?")
            score = r.get("breadth_score", "?")
            sig   = r.get("market_signal", "?")
            nepse = r.get("nepse_index",   "?")
            breadth_lines.append(
                f"  {ts}  adv={adv:<4} dec={dec:<4} score={str(score):>7}  {sig:<18}  NEPSE={nepse}"
            )
        breadth_block = (
            "INTRADAY BREADTH TIMELINE (oldest → newest):\n"
            + "\n".join(breadth_lines)
        )
    else:
        breadth_block = "INTRADAY BREADTH TIMELINE: no snapshots yet this session"

    # ── Sector momentum context ───────────────────────────────────────────────
    sector_momentum_block = ""
    if sector_momentum_map:
        lines = list(sector_momentum_map.values())
        if lines:
            sector_momentum_block = "\n".join(lines)
    elif market_data:
        try:
            from sheets import read_tab
            share_sectors = read_tab("share_sectors")
            sector_map = {
                r["symbol"].upper(): (r.get("sectorname") or "others").lower()
                for r in share_sectors if r.get("symbol")
            }
            candidate_sectors = {c.sector for c in candidates[:max_candidates] if c.sector}
            sector_momentum_block = _compute_sector_momentum(market_data, candidate_sectors, sector_map)
        except Exception as exc:
            logger.warning("sector_momentum failed (non-fatal): %s", exc)

    positions_str   = ", ".join(open_positions) if open_positions else "None"
    max_positions   = context.get("max_positions", 3)
    slots_remaining = max(0, max_positions - len(open_positions))

    candidates_str = "\n".join(
        f"{i+1}. {format_candidate_for_gemini(c)} "
        f"VWAPD={getattr(c, 'vwap_dev', 0.0):+.1f}% "
        f"BAR={getattr(c, 'bid_ask_ratio', 0.0):.2f} "
        f"DPRP={getattr(c, 'dpr_proximity', 0.0):.2f} "
        f"VOS={getattr(c, 'volume_os_ratio', 0.0):.2f}%OS"
        for i, c in enumerate(candidates[:max_candidates])
    )

    lessons_str = "\n".join(f"  - {l}" for l in lessons) if lessons else "  No lessons yet"

    prompt = f"""You are a NEPSE stock screener AI. Today is {nst_now.strftime('%Y-%m-%d %H:%M')} NST.
Your job: review these pre-filtered candidates and decide which {max_flags} (max)
deserve deep Claude analysis today. Be selective — Claude analysis is expensive.

═══════════════════════════════════════
DUAL-ENGINE SCORING (see ENGINE: tag on each candidate line)
═══════════════════════════════════════
Candidates are scored by two independent engines and merged into one list:
  ENGINE:v1   = flagged by snapshot scoring only (T0 indicator state)
  ENGINE:v2   = flagged by 6-day progression scoring only — v1's gate would
                have blocked this symbol at T0, but its 6-day trend improved
  ENGINE:BOTH = flagged independently by both engines — stronger signal,
                weigh this more favorably than a single-engine flag
A bracketed [co_flagged_by ...] note, when present, shows the other engine's
opinion (score/signal) even when it didn't independently make that engine's
own top list.

═══════════════════════════════════════
MARKET CONTEXT
═══════════════════════════════════════
Market State:    {market_state}
Geo Score:       {geo_combined:+d}/10
Bandh Today:     {bandh}
IPO Drain:       {ipo_drain}
Crisis:          {crisis}
Time NST:        {nst_now.strftime('%H:%M')}

{breadth_block}

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
SECTOR MOMENTUM  (context for your judgment — not a signal trigger)
Each line = sector of one or more candidates above.
Use this to strengthen or weaken conviction: if a candidate's sector
is broadly distributing, be more conservative even if technicals look good.
═══════════════════════════════════════
{sector_momentum_block or "no sector data available this cycle"}

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
═══════════════════════════════════════
SCREENING RULES (apply in order)
═══════════════════════════════════════

1. Don't skip if symbol is already in open positions
   (evaluate EXIT / RE-ENTRY context if relevant).

2. Filter out if the stock's sector belongs to mutual fund or debentures.

3. SKIP if symbol already analyzed today
   (avoid duplicate Claude reviews).

4. PREFER MACD cross = BULLISH as a primary signal
   (23.64% annualized return in historical NEPSE research).
   This is a reference, not a hard rule.

5. PREFER BB = LOWER_TOUCH
   (PF = 12.19, historically one of the highest-quality NEPSE signals).
   This is a reference, not a hard rule.

6. DOWNGRADE if RSI is the primary signal AND no reversal evidence exists
   (standalone RSI lost money historically at -4.81%).
   Do not automatically downgrade if there is clear reversal evidence,
   improving momentum, or multi-signal confirmation.

7. UPGRADE if a Tier-1 candle pattern is present and volume confirms it.

8. UPGRADE if CSTAR = Y
   (excess return above C* = 0.129).
   UPGRADE to URGENT if:
   - VOS > 1.0%
   - AND primary_signal is VOLUME_BREAKOUT

9. CHECK Learning Hub lessons.
   If a high-confidence lesson or a pattern with win rate < 40%
   directly contradicts the setup, skip it.

10. FLAG a maximum of {max_flags} stocks.
    Quality is more important than quantity.

11. Use both market context and stock-specific context.
    Consider:
    - market_state
    - breadth trend
    - sector momentum
    - engine source (v1 = snapshot, v2 = progression/unvalidated, BOTH = strongest confirmation)
    - momentum status
    - reversal evidence
    - learning hub lessons
    Do NOT reject a candidate solely because the overall market is bearish.
12. V2 EXEMPTION: Any active Learning Hub lesson above whose condition is a
    tech_score threshold (pattern: "tech_score < N") does NOT apply to
    candidates marked ENGINE:v2 (engine_source="v2", not "BOTH"). Such
    lessons are derived from v1-scored evidence — v2 candidates are surfaced
    via momentum/reversal signals independent of tech_score by design, so
    tech_score lagging is expected for this population, not disqualifying.
    Evaluate ENGINE:v2 candidates on v2's own signal strength
    (momentum_status, rsi_slope_3d, bb_pct_b_slope, composite_score_v2)
    instead. Candidates marked ENGINE:BOTH still require the tech_score
    threshold as normal, since v1 independently confirmed them.
13. Explain clearly WHY a stock deserves Claude review.
    Reference:
    - primary signal
    - momentum characteristics
    - reversal evidence
    - breadth or sector behavior
    - key risk

    If no candidate genuinely deserves Claude analysis after applying all
    rules above, return an empty flag list.

═══════════════════════════════════════
TASK
═══════════════════════════════════════

1. For each candidate decide: ANALYZE or SKIP based on rules above.
2. If ANALYZE: assign urgency NORMAL or HIGH or URGENT.

- Study the INTRADAY BREADTH TIMELINE. Identify whether breadth_score is trending up, down, or oscillating.
  Set intraday_trend accordingly. Breadth affects URGENCY ONLY — NEVER skip a candidate solely due to breadth.
  DISTRIBUTING/FADING → downgrade urgency to NORMAL only.
  ACCUMULATING/RECOVERING → urgency can be elevated if signals are strong.

Return ONLY this JSON — no markdown, no explanation, no extra text:
{{
  "run_time": "{nst_now.strftime('%Y-%m-%d %H:%M')}",
  "market_state": "{market_state}",
  "slots_remaining": {slots_remaining},
  "intraday_trend": "Classify breadth pattern as one of: ACCUMULATING (breadth consistently improving across snapshots) | DISTRIBUTING (consistently worsening) | RECOVERING (started weak, now strengthening) | FADING (started strong, now weakening) | CHOPPY (no clear direction, oscillating) | EARLY_SESSION (fewer than 3 snapshots, cannot classify). A stock flagged during ACCUMULATING or RECOVERING carries lower timing risk than one flagged during FADING or DISTRIBUTING — factor this into urgency decisions.",
  "flags": [
    {{
      "symbol": "SYMBOL",
      "action": "ANALYZE",
      "urgency": "NORMAL or HIGH or URGENT",
      "reason": "4 sentence why it is worth watching citing specific volume  or laggard setup. also take account of added technical details",
      "risk": "Key risk or support level (e.g., NHPC support at 300)",
      "primary_signal": "MACD or BB or SMA or RSI or OBV_MOMENTUM or VOLUME_BREAKOUT"
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
    max_candidates:  int = MAX_CANDIDATES_TO_GEMINI,
    max_flags:       int = MAX_FLAGS_FOR_CLAUDE,
    max_positions:   int = 3,
) -> dict:
    """Rule-based fallback when Gemini is unavailable."""
    logger.info("Using keyword fallback for Gemini screening")

    flags   = []
    skipped = []

    for c in candidates[:max_candidates]:
        sym = c.symbol

        if sym in open_positions:
            skipped.append({"symbol": sym, "reason": "already in open positions"})
            continue

        if slots_remaining == 0 and len(open_positions) >= max_positions:
            skipped.append({"symbol": sym, "reason": f"portfolio full ({max_positions}/{max_positions} positions)"})
            continue

        if c.primary_signal == "RSI":
            skipped.append({"symbol": sym, "reason": "RSI as primary signal — lost money standalone in NEPSE"})
            continue

        if len(flags) >= max_flags:
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
        "intraday_trend":  "EARLY_SESSION",
        "flags":           flags,
        "skipped":         skipped,
        "market_comment":  "Gemini unavailable — keyword fallback used",
        "fallback_used":   True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ASSEMBLE GEMINI FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_flags(
    gemini_result:        dict,
    candidates:           list,
    sector_momentum_map:  dict = None,
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
            gemini_reason    = str(f.get("reason", "")),
            gemini_risk      = str(f.get("risk",   "")),
            primary_signal   = str(f.get("primary_signal", c.primary_signal)),
            intraday_trend   = str(gemini_result.get("intraday_trend", "")),
            sector_momentum  = (sector_momentum_map or {}).get(
                                    c.sector.lower() if c.sector else "", ""),
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
            pivot_r1         = float(getattr(c, "pivot_r1", 0.0) or 0.0),
            pivot_r2         = float(getattr(c, "pivot_r2", 0.0) or 0.0),
            pivot_r3         = float(getattr(c, "pivot_r3", 0.0) or 0.0),
            pivot_s1         = float(getattr(c, "pivot_s1", 0.0) or 0.0),
            pivot_s2         = float(getattr(c, "pivot_s2", 0.0) or 0.0),
            pivot_s3         = float(getattr(c, "pivot_s3", 0.0) or 0.0),
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

            engine_source      = getattr(c, "engine_source",      "v1") or "v1",
            composite_score_v2 = float(getattr(c, "composite_score_v2", 0.0) or 0.0),
            primary_signal_v2  = getattr(c, "primary_signal_v2",  "") or "",
            co_flagged_by      = getattr(c, "co_flagged_by",      "") or "",

            vwap_dev        = float(getattr(c, "vwap_dev",        0.0) or 0.0),
            bid_ask_ratio   = float(getattr(c, "bid_ask_ratio",   0.0) or 0.0),
            dpr_proximity   = float(getattr(c, "dpr_proximity",   0.0) or 0.0),
            volume_os_ratio = float(getattr(c, "volume_os_ratio", 0.0) or 0.0),

            momentum_status  = getattr(c, "momentum_status",  "NEUTRAL") or "NEUTRAL",
            rsi_slope_3d     = float(getattr(c, "rsi_slope_3d",   0.0) or 0.0),
            macd_hist_slope  = float(getattr(c, "macd_hist_slope", 0.0) or 0.0),
            bb_pct_b_slope   = float(getattr(c, "bb_pct_b_slope",  0.0) or 0.0),
            bounce_failed    = bool(getattr(c, "bounce_failed",  False)),
            reversal_days    = int(getattr(c, "reversal_days",   0)   or 0),
            news_catalyst    = getattr(c, "news_catalyst", ""),
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
                    f"Signal: {flag.primary_signal} | "
                    f"Breadth: {flag.intraday_trend}"
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
                "gemini_reason":   flag.gemini_reason,
                "gemini_risk":     flag.gemini_risk,
                "primary_signal":  flag.primary_signal,
                "engine_source":  flag.engine_source,
                "co_flagged_by":  flag.co_flagged_by,
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
    from sheets import run_raw_sql
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("gemini_filter.run_gemini_filter() — %s", date)

    # ── Context (load first so max_candidates is available for run_filter) ────
    try:
        context = get_filter_context()
    except Exception:
        context = {}

    max_positions  = context.get("max_positions",         3)
    max_candidates = context.get("gemini_max_candidates", MAX_CANDIDATES_TO_GEMINI)
    max_flags      = context.get("gemini_max_flags",      MAX_FLAGS_FOR_CLAUDE)

    # ── Get candidates from filter_engine if not provided ────────────────────
    if candidates is None:
        try:
            candidates = run_filter(
                market_data=market_data,
                top_n=max_candidates,
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

    # ── Portfolio ─────────────────────────────────────────────────────────────
    open_positions  = _load_open_positions()
    slots_remaining = max(0, max_positions - len(open_positions))
    total_capital   = _load_total_capital()
    symbols         = [c.symbol for c in candidates]
    lessons         = _load_relevant_lessons(symbols)
    # ── Hard dedup: skip symbols Claude already decided on today ─────────────
    try:
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        done_rows = run_raw_sql(
            """
            SELECT DISTINCT symbol FROM market_log
            WHERE date = %s AND action IN ('BUY', 'AVOID')
            """,
            (today,),
        )
        already_done = {r["symbol"] for r in (done_rows or [])}
        if already_done:
            before = len(candidates)
            candidates = [c for c in candidates if c.symbol not in already_done]
            logger.info(
                "Dedup: removed %d already-decided symbols %s",
                before - len(candidates), already_done,
            )
    except Exception as exc:
        logger.warning("Dedup check failed — proceeding without filter: %s", exc)

    # Sector concentration gate — belt-and-suspenders for externally provided candidates
    _sector_counts  = context.get("sector_position_counts", {})
    _max_per_sector = context.get("max_positions_per_sector", 2)
    if _sector_counts:
        _before = len(candidates)
        _filtered: list = []
        for _c in candidates:
            _sec = (_c.sector or "").lower().strip()
            _cnt = _sector_counts.get(_sec, 0) if _sec else 0
            if _sec and _cnt >= _max_per_sector:
                logger.info(
                    "SECTOR_CONCENTRATION: %s sector=%s count=%d>=%d — excluded pre-Gemini",
                    _c.symbol, _sec, _cnt, _max_per_sector,
                )
                try:
                    from sheets import upsert_row as _upsert_row
                    _today_sc = datetime.now(tz=NST).strftime("%Y-%m-%d")
                    _upsert_row(
                        "gate_misses",
                        {
                            "symbol":                   _c.symbol,
                            "sector":                   _c.sector,
                            "date":                     _today_sc,
                            "gate_reason":              f"SECTOR_LIMIT={_sec}:{_cnt}>={_max_per_sector}",
                            "gate_category":            "SECTOR_CONCENTRATION",
                            "price_at_block":           str(_c.ltp),
                            "market_state":             _c.market_state,
                            "tech_score":               str(_c.tech_score),
                            "conf_score":               str(_c.conf_score),
                            "composite_score_would_be": str(_c.composite_score),
                            "volume_os_ratio":          str(getattr(_c, "volume_os_ratio", 0.0)),
                            "vwap_dev":                 str(getattr(_c, "vwap_dev",        0.0)),
                            "bid_ask_ratio":            str(getattr(_c, "bid_ask_ratio",   0.0)),
                            "dpr_proximity":            str(getattr(_c, "dpr_proximity",   0.0)),
                            "outcome":                  None,
                            "tracking_days":            "0",
                        },
                        conflict_columns=["symbol", "date"],
                    )
                except Exception as _exc:
                    logger.warning("SECTOR_CONCENTRATION gate_miss write failed: %s", _exc)
            else:
                _filtered.append(_c)
        if len(_filtered) < _before:
            logger.info(
                "Sector concentration gate: removed %d candidates pre-Gemini",
                _before - len(_filtered),
            )
        candidates = _filtered

    logger.info(
        "Portfolio: %d open | %d slots | capital NPR %.0f | %d lessons loaded",
        len(open_positions), slots_remaining, total_capital, len(lessons),
    )

    if slots_remaining == 0 and len(open_positions) >= max_positions:
        logger.info("Portfolio full (%d/%d positions) — no new signals needed", max_positions, max_positions)
        return []

    # ── Sector momentum map (computed once, reused for prompt + flags) ────────
    sector_momentum_map: dict = {}
    if market_data:
        try:
            from sheets import read_tab as _read_tab
            _ss = _read_tab("share_sectors")
            _smap = {
                r["symbol"].upper(): (r.get("sectorname") or "others").lower()
                for r in _ss if r.get("symbol")
            }
            _block = _compute_sector_momentum(
                market_data,
                {c.sector for c in candidates[:max_candidates] if c.sector},
                _smap,
            )
            for _line in _block.splitlines():
                if _line.startswith("sector=") and " | " in _line:
                    _sec = _line.split(" | ")[0].replace("sector=", "").strip()
                    sector_momentum_map[_sec] = _line
        except Exception as exc:
            logger.warning("sector_momentum_map build failed (non-fatal): %s", exc)

    # ── Build prompt and call Gemini ──────────────────────────────────────────
    prompt = _build_prompt(
        candidates          = candidates,
        context             = context,
        lessons             = lessons,
        open_positions      = open_positions,
        total_capital       = total_capital,
        max_candidates      = max_candidates,
        max_flags           = max_flags,
        market_data         = market_data,
        sector_momentum_map = sector_momentum_map,
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
        use_search = False,
    )

    if gemini_result is None:
        logger.warning("Gemini unavailable — skipping Claude this cycle, no fallback")
        return []

    # ── Assemble + log ────────────────────────────────────────────────────────
    flags = _assemble_flags(gemini_result, candidates, sector_momentum_map)

    if flags:
        _write_log(gemini_result, flags, candidates)

    # Flush Gemini SKIPs to gate_misses for forensic tracking
    try:
        flush_gemini_skips_to_db(gemini_result, candidates, date=date)
    except Exception as _skip_exc:
        logger.warning("flush_gemini_skips_to_db failed (non-fatal): %s", _skip_exc)

    fallback_note = " [FALLBACK]" if gemini_result.get("fallback_used") else ""
    logger.info(
        "gemini_filter done%s: %d flagged for Claude | %d skipped | comment: %s",
        fallback_note,
        len(flags),
        len(gemini_result.get("skipped", [])),
        gemini_result.get("market_comment", "—"),
    )
    for f in flags:
        logger.info("  FLAG: %s", f.summary())

    return flags


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HELPER FOR claude_analyst.py
# ══════════════════════════════════════════════════════════════════════════════

def format_flag_for_claude(flag: GeminiFlag) -> str:
    candle = f"{flag.best_candle}(T{flag.candle_tier})" if flag.best_candle else "none"
    catalyst = f" NEWS_CATALYST:{flag.news_catalyst}" if getattr(flag, "news_catalyst", "") else ""
    engine_str = f" ENGINE:{flag.engine_source}"
    if flag.co_flagged_by:
        engine_str += f" [{flag.co_flagged_by}]"
    return (
        f"SYMBOL:{flag.symbol} SECTOR:{flag.sector} LTP:{flag.ltp:.2f} "
        f"SIGNAL:{flag.primary_signal} URGENCY:{flag.urgency} "
        f"TECH:{flag.tech_score} RSI:{flag.rsi_14:.1f} MACD:{flag.macd_cross} "
        f"BB:{flag.bb_signal} CANDLE:{candle} GEO:{flag.geo_combined:+d} "
        f"REASON:{flag.gemini_reason} RISK:{flag.gemini_risk}"
        f" BREADTH_TREND:{flag.intraday_trend}"
        + (f" SECTOR_MOMENTUM:{flag.sector_momentum}" if flag.sector_momentum else "")
        + catalyst
        + engine_str
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
        # Load context first so max_candidates is available for run_filter
        _cli_ctx           = get_filter_context()
        _cli_max_cands     = _cli_ctx.get("gemini_max_candidates", MAX_CANDIDATES_TO_GEMINI)

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
            candidates = run_filter(market_data=md, top_n=_cli_max_cands)
        else:
            from modules.scraper import get_all_market_data
            md = get_all_market_data(write_breadth=False)
            candidates = run_filter(market_data=md, top_n=_cli_max_cands)

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
    context        = get_filter_context()
    open_positions = _load_open_positions()
    max_positions  = context.get("max_positions",         3)
    max_candidates = context.get("gemini_max_candidates", MAX_CANDIDATES_TO_GEMINI)
    max_flags      = context.get("gemini_max_flags",      MAX_FLAGS_FOR_CLAUDE)
    slots_remaining = max(0, max_positions - len(open_positions))
    total_capital  = _load_total_capital()
    lessons        = _load_relevant_lessons([c.symbol for c in candidates])

    print(f"  Open positions:  {open_positions or 'None'}")
    print(f"  Slots remaining: {slots_remaining}")
    print(f"  Max positions:   {max_positions}")
    print(f"  Max candidates:  {max_candidates}")
    print(f"  Max flags:       {max_flags}")
    print(f"  Capital:         NPR {total_capital:,.0f}")
    print(f"  Lessons loaded:  {len(lessons)}")

    if slots_remaining == 0:
        print("\n  Portfolio full — no new signals needed")
        sys.exit(0)

    print(f"\n[3/3] {'Keyword fallback (--no-gemini)' if no_gemini else 'Gemini Flash screening'}...")

    if no_gemini:
        gemini_result = _keyword_fallback(candidates, open_positions, slots_remaining,
                                          max_candidates=max_candidates, max_flags=max_flags,
                                          max_positions=max_positions)
    else:
        prompt        = _build_prompt(candidates, context, lessons, open_positions, total_capital,
                                      max_candidates=max_candidates, max_flags=max_flags)
        gemini_result = ask_gemini_json(
            prompt,
            system  = (
                "You are a Nepal stock market screening AI. "
                "You understand NEPSE trading patterns, Nepal macro context, "
                "and the research-backed signal weights being used. "
                "You return only valid JSON — no markdown, no fences, no explanation."
            ),
            context    = "gemini_filter",
            use_search = False,
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