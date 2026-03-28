"""
claude_analyst.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 3
Purpose : Deep Claude analysis for each stock flagged by gemini_filter.py.
          Produces BUY / WAIT / AVOID with specific entry, stop, target,
          NPR allocation, and full reasoning.

Position in pipeline:
    filter_engine.py   → ranked candidates (pure math)
    gemini_filter.py   → Gemini Flash contextual screen (fast)
    claude_analyst.py  → Claude deep analysis (slow, thorough, one per flag)

What Claude does here:
    - Reads ALL available context (geo, macro, nepal, portfolio, lessons)
    - Applies research-backed rules (MACD holds 17d, BB holds 130d, etc.)
    - Checks herding bubble / capitulation conditions
    - Applies insurance political sensitivity rule
    - Checks trading day effect (buy Sun/Mon, exit Wed/Thu)
    - Produces structured JSON with specific NPR amounts
    - Writes recommendation to market_log table

What Claude does NOT do here:
    - Position sizing math (that's budget.py / DeepSeek Kelly Criterion)
    - Portfolio execution (always user's decision)
    - Macro extraction (that's NotebookLM monthly workflow)

Evidence base for rules injected into Claude prompt:
    MACD 12/26/9       → 23.64% ann. return, hold 17 days (Karki 2023)
    Bollinger Band     → PF=12.19, hold 130 days (Karki 2023)
    SMA crossover      → 21.33% ann. return, hold 33 days (Karki 2023)
    RSI standalone     → -4.81% ann. return — NEVER sole trigger (Karki 2023)
    Non-Life Insurance → best risk-adj sector β=0.034 (Khadka 2023)
    Insurance politics → 4.6x more sensitive, 3-5 day recovery (political paper)
    Herding            → β=0.351-0.428, amplifies bubbles (herd paper)
    Trading days       → Buy Sun/Mon, exit Wed/Thu (trading days paper)
    Dividend leakage   → starts 6 days pre-announcement (dividend paper)
    Dashain rally      → +3.5% pre-festival, -2.5% post (seasonal paper)

─────────────────────────────────────────────────────────────────────────────
Called by: trading.yml every 6 min (only when gemini_filter flags stocks)
Input:     list[GeminiFlag] from gemini_filter.py
Output:    list[AnalystResult] + rows written to market_log table
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

NST = timezone(timedelta(hours=5, minutes=45))

# # ── Anthropic native API config ───────────────────────────────────────────────
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20251022")

# ── Nepal fee constants (for breakeven calculation) ───────────────────────────
BROKERAGE_PCT = 0.40
SEBON_PCT     = 0.015
DP_CHARGE_NPR = 25.0
CGT_PCT       = 7.5


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalystResult:
    """
    Claude's recommendation for one stock.
    Written to market_log table.
    """
    symbol:           str
    action:           str        = "WAIT"     # BUY / WAIT / AVOID
    confidence:       int        = 0          # 0–100
    entry_price:      float      = 0.0
    stop_loss:        float      = 0.0
    target:           float      = 0.0
    allocation_npr:   float      = 0.0
    shares:           int        = 0
    breakeven:        float      = 0.0
    risk_reward:      float      = 0.0
    suggested_hold:   int        = 17         # days
    reasoning:        str        = ""
    lesson_applied:   str        = ""
    primary_signal:   str        = ""
    sector:           str        = ""
    geo_score:        int        = 0
    rsi_14:           float      = 0.0
    candle_pattern:   str        = ""
    urgency:          str        = "NORMAL"   # from GeminiFlag
    gemini_reason:    str        = ""
    market_log_id:    int       = None

    timestamp: str = field(default_factory=lambda:
                    datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.action == "BUY":
            return (
                f"✅ BUY {self.symbol} | conf={self.confidence}% | "
                f"entry={self.entry_price:.0f} stop={self.stop_loss:.0f} "
                f"target={self.target:.0f} | NPR {self.allocation_npr:,.0f} | "
                f"hold ~{self.suggested_hold}d | {self.reasoning[:60]}"
            )
        return (
            f"{'⏸' if self.action == 'WAIT' else '🚫'} {self.action} {self.symbol} | "
            f"{self.reasoning[:80]}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONTEXT LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_portfolio() -> dict:
    """Load open positions and capital from Neon."""
    try:
        from sheets import read_tab, get_setting

        rows = read_tab("portfolio")
        open_pos = [r for r in rows if r.get("status", "").upper() == "OPEN"]

        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
        invested      = sum(float(r.get("total_cost", 0) or 0) for r in open_pos)
        liquid        = max(0.0, total_capital - invested)

        return {
            "total_capital_npr": total_capital,
            "liquid_npr":        liquid,
            "invested_npr":      invested,
            "open_positions":    len(open_pos),
            "max_positions":     99,
            "slots_remaining":    99,#max(0, 3 - len(open_pos)), 
            "holdings": [
                {
                    "symbol":    r.get("symbol", ""),
                    "shares":    r.get("shares", ""),
                    "wacc":      r.get("wacc", ""),
                    "pnl_pct":   r.get("pnl_pct", ""),
                    "pnl_npr":   r.get("pnl_npr", ""),
                }
                for r in open_pos
            ],
        }
    except Exception as exc:
        logger.warning("_load_portfolio failed: %s", exc)
        return {
            "total_capital_npr": 100000,
            "liquid_npr":        100000,
            "invested_npr":      0,
            "open_positions":    0,
            "max_positions":     3,
            "slots_remaining":   3,
            "holdings":          [],
        }


def _load_geo_context() -> dict:
    """Load latest geo + nepal scores."""
    try:
        from sheets import get_latest_geo, get_latest_pulse

        geo   = get_latest_geo()   or {}
        pulse = get_latest_pulse() or {}

        return {
            "geo_score":      int(geo.get("geo_score",     0) or 0),
            "nepal_score":    int(pulse.get("nepal_score", 0) or 0),
            "combined":       int(geo.get("geo_score", 0) or 0) + int(pulse.get("nepal_score", 0) or 0),
            "geo_status":     geo.get("status",        "NEUTRAL"),
            "nepal_status":   pulse.get("nepal_status", "NEUTRAL"),
            "vix":            geo.get("vix",           "?"),
            "vix_level":      geo.get("vix_level",     "?"),
            "nifty_chg":      geo.get("nifty_change_pct", "?"),
            "crude":          geo.get("crude_price",   "?"),
            "bandh":          pulse.get("bandh_today", "NO"),
            "ipo_drain":      pulse.get("ipo_fpo_active", "NO"),
            "crisis":         pulse.get("crisis_detected", "NO"),
            "key_geo_event":  geo.get("key_event",    ""),
            "key_nepal_event":pulse.get("key_event",   ""),
        }
    except Exception as exc:
        logger.warning("_load_geo_context failed: %s", exc)
        return {"geo_score": 0, "nepal_score": 0, "combined": 0}


def _load_macro_context() -> dict:
    """Load macro indicators from Neon macro_data table."""
    try:
        from sheets import read_tab, get_setting
        rows = read_tab("nrb_monthly", limit=1)
        nrb = rows[0] if rows else {}
        fd_rate = get_setting("FD_RATE_PCT", "8.5")
        return {
                "policy_rate":      nrb.get("policy_rate",              "?"),
                "bank_rate":        nrb.get("bank_rate",                "?"),
                "cpi_inflation":    nrb.get("cpi_inflation",            "?"),
                "credit_growth":    nrb.get("credit_growth_rate",       "?"),
                "remittance_yoy":   nrb.get("remittance_yoy_change_pct","?"),
                "forex_reserve":    nrb.get("fx_reserve_months",        "?"),
                "bop_balance":      nrb.get("bop_overall_balance_usd_m","?"),
                "bop_status":       nrb.get("bop_status",               "?"),
                "bop_trend":        nrb.get("bop_trend",                "?"),
                "liquidity":        nrb.get("liquidity_injected_billion","?"),
                "sentiment":        nrb.get("overall_sentiment",        "?"),
                "key_risks":        nrb.get("key_risks",                "?"),
                "period":           nrb.get("period",                   "?"),
                "fd_rate":          fd_rate,
                "fd_signal":        get_setting("FD_SCORE_SIGNAL",      "NEUTRAL"),
            }
    except Exception as exc:
        logger.warning("_load_macro_context failed: %s", exc)
        return {}


def _load_lessons(symbol: str, sector: str, limit: int = 6) -> list[str]:
    """
    Load relevant Learning Hub lessons.
    Prioritises: this symbol > this sector > MARKET-level lessons.
    Uses new learning_hub schema: condition, finding, action, confidence_level.
    """
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT symbol, sector, lesson_type, condition, finding, action,
                   confidence_level, win_rate, trade_count
            FROM learning_hub
            WHERE active = 'true'
              AND (symbol = %s
               OR sector = %s
               OR symbol = 'MARKET'
               OR applies_to = 'ALL')
            ORDER BY
                CASE WHEN symbol = %s    THEN 0
                     WHEN sector = %s    THEN 1
                     ELSE 2 END,
                CASE WHEN confidence_level = 'HIGH'   THEN 0
                     WHEN confidence_level = 'MEDIUM' THEN 1
                     ELSE 2 END,
                id DESC
            LIMIT %s
            """,
            (symbol.upper(), sector.lower(),
             symbol.upper(), sector.lower(), limit)
        )
        lessons = []
        for r in rows:
            sym    = r.get("symbol", "?")
            ltype  = r.get("lesson_type", "")
            cond   = r.get("condition",   "")[:80]
            find   = r.get("finding",     "")[:120]
            act    = r.get("action",      "")
            conf   = r.get("confidence_level", "LOW")
            wr     = r.get("win_rate",    "")
            n      = r.get("trade_count", "")
            stat   = f" (win_rate={wr}, n={n})" if n else ""
            lessons.append(f"[{sym}|{ltype}|{conf}] IF {cond} → {act}: {find}{stat}")
        return lessons
    except Exception as exc:
        logger.warning("_load_lessons failed: %s", exc)
        return []


def _load_market_state() -> str:
    """Read current market state from settings."""
    try:
        from sheets import get_setting
        return get_setting("MARKET_STATE", "SIDEWAYS").upper().strip()
    except Exception:
        return "SIDEWAYS"


def _load_loss_streak() -> int:
    """Read current loss streak from financials table."""
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            "SELECT current_value FROM financials WHERE kpi_name = 'current_loss_streak'"
        )
        return int(float(rows[0]["current_value"] or 0)) if rows else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _calc_breakeven(entry_price: float, shares: int = 1) -> float:
    """
    True breakeven price after Nepal transaction costs.
    Must exceed this price to avoid loss.
    """
    if entry_price <= 0:
        return entry_price
    trade_value   = entry_price * shares
    buy_brokerage = trade_value * BROKERAGE_PCT / 100
    buy_sebon     = trade_value * SEBON_PCT / 100
    buy_dp        = DP_CHARGE_NPR
    # At breakeven, sell costs are symmetric — add sell side too
    # breakeven_price × shares = trade_value + buy_costs + sell_costs + CGT
    # Approximated as: entry × (1 + 2×(brokerage+sebon)/100) + 2×DP / shares
    total_fee_rate = 2 * (BROKERAGE_PCT + SEBON_PCT) / 100
    breakeven      = entry_price * (1 + total_fee_rate) + (2 * DP_CHARGE_NPR / max(shares, 1))
    return round(breakeven, 2)


def _calc_true_profit(entry: float, exit_price: float, shares: int) -> float:
    """Calculate actual NPR profit after ALL Nepal transaction costs."""
    buy_val    = entry      * shares
    sell_val   = exit_price * shares
    gross      = sell_val - buy_val

    buy_cost   = buy_val  * (BROKERAGE_PCT + SEBON_PCT) / 100 + DP_CHARGE_NPR
    sell_cost  = sell_val * (BROKERAGE_PCT + SEBON_PCT) / 100 + DP_CHARGE_NPR
    cgt        = max(0, gross) * CGT_PCT / 100

    return round(gross - buy_cost - sell_cost - cgt, 2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRADING DAY CONTEXT
# ══════════════════════════════════════════════════════════════════════════════

def _trading_day_context() -> str:
    """
    Returns a one-liner about today's weekday for Claude to factor in.
    Research: Buy Sun/Mon (-0.04% to -0.02% mean), Exit Wed/Thu (+0.13%)
    """
    nst_now  = datetime.now(tz=NST)
    weekday  = nst_now.weekday()  # Mon=0 ... Sun=6
    day_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][weekday]

    if weekday == 6:   # Sunday
        return f"Today is {day_name}: historically lowest return day (-0.04% mean). Good entry day — buy dips."
    elif weekday == 0: # Monday
        return f"Today is {day_name}: second lowest return day. Still good for entries."
    elif weekday == 1: # Tuesday
        return f"Today is {day_name}: returns positive (+0.07% mean). Neutral for entry/exit."
    elif weekday == 2: # Wednesday
        return f"Today is {day_name}: historically best return day (+0.13% mean). Prefer exit over entry."
    elif weekday == 3: # Thursday
        return f"Today is {day_name}: second best return day, lowest volatility. Good exit day."
    else:
        return f"Today is {day_name}: non-trading day."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HERDING / BUBBLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _herding_context(flag, market_state: str) -> str:
    """
    Detect potential herding bubble or capitulation conditions.
    Research: herding β=0.351-0.428, strongest in bull markets and crises.
    When herding is active, prices deviate from true value — increases risk.
    """
    warnings = []

    # High RSI + bull market = potential bubble
    rsi = float(flag.rsi_14 or 0)
    if rsi > 72 and market_state in ("FULL_BULL", "CAUTIOUS_BULL"):
        warnings.append(
            f"RSI={rsi:.1f} in {market_state} — herding bubble risk. "
            "Retail herd may be chasing. Wait for pullback."
        )

    # Very low RSI + BEAR = potential capitulation (contrarian opportunity)
    if rsi < 25 and market_state == "BEAR":
        warnings.append(
            f"RSI={rsi:.1f} in BEAR — possible capitulation. "
            "Herding-driven oversell. Watch for reversal signal."
        )

    # Conf score very high = crowd already bought in
    conf = float(getattr(flag, "composite_score", 0) or 0)
    if conf > 90:
        warnings.append(
            "Very high composite score — most alpha may already be priced in by crowd."
        )

    # High geo combined + bull market = euphoria risk
    geo = int(getattr(flag, "geo_combined", 0) or 0)
    if geo >= 4 and market_state == "FULL_BULL":
        warnings.append(
            "High geo score in FULL_BULL — market euphoria conditions. "
            "Herding amplifies both gains and reversals. Tighten stops."
        )

    if not warnings:
        return "No herding bubble or capitulation signals detected."
    return "HERDING ALERT: " + " | ".join(warnings)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BUILD CLAUDE PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    flag:         object,  # GeminiFlag
    portfolio:    dict,
    geo:          dict,
    macro:        dict,
    lessons:      list[str],
    market_state: str,
    loss_streak:  int,
) -> str:
    """
    Build the full deep-analysis prompt for Claude.
    Injects all research-backed rules explicitly.
    """
    nst_now = datetime.now(tz=NST)

    lessons_str = "\n".join(f"  - {l}" for l in lessons) if lessons else "  No lessons yet."
    holdings_str = "\n".join(
        f"  {h['symbol']}: {h['shares']} shares @ WACC {h['wacc']} | P&L {h['pnl_pct']}% (NPR {h['pnl_npr']})"
        for h in portfolio.get("holdings", [])
    ) or "  No open positions."

    day_context   = _trading_day_context()
    herding_alert = _herding_context(flag, market_state)

    # Candle info
    candle_str = f"{flag.best_candle} (Tier {flag.candle_tier})" if flag.best_candle else "None"

    # Suggested hold from Gemini
    hold_days = getattr(flag, "suggested_hold", 17)

    prompt = f"""You are a senior NEPSE quantitative analyst with deep knowledge of Nepal market research.
Analyze this specific stock and produce a precise trading recommendation.

TODAY: {nst_now.strftime('%Y-%m-%d %H:%M')} NST
MARKET STATE: {market_state}
LOSS STREAK: {loss_streak} consecutive losses (circuit breaker at 8)

═══════════════════════════════════════════════
STOCK UNDER ANALYSIS
═══════════════════════════════════════════════
Symbol:          {flag.symbol}
Sector:          {flag.sector}
LTP:             NPR {flag.ltp:.2f}
Daily Change:    {getattr(flag, 'change_pct', 0):+.2f}%
Urgency:         {flag.urgency}  (from Gemini screener)
Gemini Reason:   {flag.gemini_reason}
Gemini Risk:     {flag.gemini_risk}

TECHNICAL INDICATORS (frozen at 10:30 AM NST, computed from historical data):
  RSI 14:          {flag.rsi_14:.1f}  [{flag.rsi_signal if hasattr(flag, 'rsi_signal') else ''}]
  MACD Cross:      {flag.macd_cross}
  BB Signal:       {flag.bb_signal}
  OBV Trend:       {getattr(flag, 'obv_trend', '?')}
  EMA Trend:       {getattr(flag, 'ema_trend', '?')}
  Tech Score:      {flag.tech_score}/100
  Candle Pattern:  {candle_str}
  C* Signal:       {'YES — excess return above C*=0.129 (SIM paper)' if getattr(flag, 'cstar_signal', False) else 'NO'}

Primary Signal:  {flag.primary_signal}
Composite Score: {flag.composite_score:.1f}
Suggested Hold:  ~{hold_days} days (research-based)
Geo Score:       {flag.geo_combined:+d}/10

═══════════════════════════════════════════════
MARKET CONTEXT
═══════════════════════════════════════════════
Geo Score:       {geo.get('geo_score', 0):+d}/5  ({geo.get('geo_status', '?')})
Nepal Score:     {geo.get('nepal_score', 0):+d}/5  ({geo.get('nepal_status', '?')})
Combined:        {geo.get('combined', 0):+d}/10
VIX:             {geo.get('vix', '?')} [{geo.get('vix_level', '?')}]
Nifty Change:    {geo.get('nifty_chg', '?')}%
Crude:           ${geo.get('crude', '?')}
Bandh Today:     {geo.get('bandh', 'NO')}
IPO Drain:       {geo.get('ipo_drain', 'NO')}
Key Geo Event:   {geo.get('key_geo_event', 'None')}
Key Nepal Event: {geo.get('key_nepal_event', 'None')}

MACRO (NRB Latest — updated monthly):
 MACRO (NRB {macro.get('period', '?')} — updated monthly):
  Policy Rate:     {macro.get('policy_rate', '?')}%
  CPI Inflation:   {macro.get('cpi_inflation', '?')}%
  Credit Growth:   {macro.get('credit_growth', '?')}%
  Remittance YoY:  {macro.get('remittance_yoy', '?')}%
  FX Reserve:      {macro.get('forex_reserve', '?')} months
  BOP Balance:     USD {macro.get('bop_balance', '?')}M ({macro.get('bop_status', '?')} — {macro.get('bop_trend', '?')})
  Liquidity:       {macro.get('liquidity', '?')}
  NRB Sentiment:   {macro.get('sentiment', '?')}
  Key Risks:       {macro.get('key_risks', '?')}
  FD Rate (1yr):   {macro.get('fd_rate', '?')}%  [{macro.get('fd_signal', '?')}]


═══════════════════════════════════════════════
YOUR PORTFOLIO
═══════════════════════════════════════════════
Total Capital:   NPR {portfolio.get('total_capital_npr', 0):,.0f}
Available Cash:  NPR {portfolio.get('liquid_npr', 0):,.0f}
Invested:        NPR {portfolio.get('invested_npr', 0):,.0f}
Open Positions:  {portfolio.get('open_positions', 0)}/1000 max
Slots Left:      {portfolio.get('slots_remaining', 0)}

Current Holdings:
{holdings_str}

═══════════════════════════════════════════════
TRADING DAY INTELLIGENCE
═══════════════════════════════════════════════
{day_context}
Research: Buy Sunday/Monday (lowest return days = better entry prices).
          Exit Wednesday/Thursday (highest return days = better exits).
          (Trading Day Effect paper, n=4,504 days, F=3.217, p=0.012)

═══════════════════════════════════════════════
HERDING / BUBBLE DETECTION
═══════════════════════════════════════════════
{herding_alert}
Research: Herding β=0.351-0.428 in NEPSE (p<0.052). Strongest in bull markets
          and global crises. Herding amplifies bubbles AND crashes.
          When herd is active: tighten stops, reduce allocation by 10-20%.

═══════════════════════════════════════════════
LEARNING HUB — RELEVANT LESSONS
═══════════════════════════════════════════════
{lessons_str}

═══════════════════════════════════════════════
RESEARCH-BACKED RULES (APPLY THESE)
═══════════════════════════════════════════════


NEPAL FEES (always include):
  Brokerage: 
  | Transaction Amount       | Commission Rate |
|--------------------------|-----------------|
| ≤ Rs. 2,500              | Flat Rs. 10     |
| Rs. 2,501 – 50,000       | 0.36%           |
| Rs. 50,001 – 500,000     | 0.33%           |
| Rs. 500,001 – 2,000,000  | 0.31%           |
| Rs. 2,000,001 – 10,000,000 | 0.27%         |
| Above Rs. 10,000,000     | 0.24%           |

  SEBON:     0.015% buy + 0.015% sell
  DP charge: NPR 25 flat per transaction
  CGT:       7.5% on profit
  Breakeven = Entry Price × (1 + 0.75%) + (Rs. 50 / Number of Shares)

═══════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════
Analyze {flag.symbol} and decide: BUY / WAIT / AVOID

BUY if: confidence ≥ 70%, setup is clean, risk is defined, allocation makes sense
WAIT if: setup looks good but timing is off, or need confirmation
AVOID if: risk too high, wrong sector, Learning Hub warns against it, or contra-indicators present

For BUY: calculate exact NPR amounts. Use max 20% of total capital per position.
For WAIT/AVOID: give specific conditions that would change your answer.
Include only ordinary shares, exclude mutual funds, debentures, promotor shares etc

Respond ONLY with this JSON — no markdown, no explanation outside JSON:
{{
  "action": "BUY or WAIT or AVOID",
  "confidence": 0-100,
  "entry_price": number,
  "stop_loss": number,
  "target": number,
  "allocation_npr": number,
  "shares": number,
  "breakeven": number,
  "risk_reward": number,
  "suggested_hold_days": number,
  "primary_signal": "MACD or BB or SMA or OBV_MOMENTUM or RSI",
  "reasoning": "3-4 sentences covering: why this signal, what risks, what the Learning Hub says, sector context",
  "lesson_applied": "which lesson from Learning Hub was most relevant, or NONE",
  "wait_condition": "if WAIT/AVOID: what specific condition would make this a BUY",
  "herding_note": "one sentence on herding/bubble risk or NONE"
}}"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6.a — CALL CLAUDE
# ══════════════════════════════════════════════════════════════════════════════
def ask_claude(prompt: str) -> Optional[dict]:
    from typing import Optional
    from anthropic import Anthropic
    # The SDK automatically uses os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", # Latest Sonnet 3.5 identifier
            max_tokens=1200,
            # Native SDK moves system instruction to its own argument
            system=(
                "You are a senior NEPSE quantitative analyst. "
                "You respond ONLY in valid JSON. No markdown fences. "
                "You never recommend a trade without a clear stop loss."
            ),
            messages=[
                {"role": "user", "content": prompt},
                # Tip: You can "prefill" the assistant response with "{" 
                # to further force JSON, but requires a different structure.
            ],
        )

        # Content is a list of blocks; get the text from the first one
        raw = response.content[0].text.strip()

        # Robust cleaning in case Claude still includes markdown fences
        if raw.startswith("```"):
            # Split by backticks and find the segment that looks like JSON
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        
        logger.info("Claude: %s %s | conf=%s",
                    result.get("action", "?"),
                    result.get("primary_signal", "?"),
                    result.get("confidence", "?"))
        return result

    except json.JSONDecodeError as exc:
        logger.error("Claude returned invalid JSON: %s | Raw: %s", exc, raw)
        return None
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return None
    
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6.b — CALL CLAUDE VIA NATIVE ANTHROPIC API
# ══════════════════════════════════════════════════════════════════════════════
def _call_claude(prompt: str) -> Optional[dict]:
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.6" ,#"anthropic/claude-sonnet-4-5",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior NEPSE quantitative analyst. "
                        "You respond ONLY in valid JSON. No markdown fences. "
                        "You never recommend a trade without a clear stop loss."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        logger.info("Claude: %s %s | conf=%s",
                    result.get("action", "?"),
                    result.get("primary_signal", "?"),
                    result.get("confidence", "?"))
        return result

    except json.JSONDecodeError as exc:
        logger.error("Claude returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ASSEMBLE RESULT
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_result(claude_json: dict, flag) -> AnalystResult:
    """Convert Claude JSON response into AnalystResult dataclass."""
    action     = str(claude_json.get("action", "WAIT")).upper()
    entry      = float(claude_json.get("entry_price", flag.ltp) or flag.ltp)
    shares     = int(claude_json.get("shares", 0) or 0)
    breakeven  = float(claude_json.get("breakeven", 0) or 0)

    # If Claude didn't calculate breakeven, compute it ourselves
    if breakeven <= 0 and entry > 0:
        breakeven = _calc_breakeven(entry, max(shares, 1))

    return AnalystResult(
        symbol         = flag.symbol,
        action         = action,
        confidence     = int(claude_json.get("confidence", 0) or 0),
        entry_price    = entry,
        stop_loss      = float(claude_json.get("stop_loss", 0)      or 0),
        target         = float(claude_json.get("target", 0)         or 0),
        allocation_npr = float(claude_json.get("allocation_npr", 0) or 0),
        shares         = shares,
        breakeven      = breakeven,
        risk_reward    = float(claude_json.get("risk_reward", 0)    or 0),
        suggested_hold = int(claude_json.get("suggested_hold_days",
                              getattr(flag, "suggested_hold", 17)) or 17),
        reasoning      = str(claude_json.get("reasoning",     "")[:500]),
        lesson_applied = str(claude_json.get("lesson_applied","")[:200]),
        primary_signal = str(claude_json.get("primary_signal",
                              getattr(flag, "primary_signal", ""))[:20]),
        sector         = getattr(flag, "sector",        ""),
        geo_score      = getattr(flag, "geo_combined",  0),
        rsi_14         = float(getattr(flag, "rsi_14",  0) or 0),
        candle_pattern = getattr(flag, "best_candle",   ""),
        urgency        = getattr(flag, "urgency",       "NORMAL"),
        gemini_reason  = getattr(flag, "gemini_reason", "")[:200],
        market_log_id =     getattr(flag, 'market_log_id', None),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def _write_to_db(result: AnalystResult) -> bool:
    """Write analyst result to market_log table, replacing Gemini row."""
    try:
        from sheets import upsert_row
        from db.connection import _db

        nst_now = datetime.now(tz=NST)

        row = {
            "date":             nst_now.strftime("%Y-%m-%d"),
            "time":             nst_now.strftime("%H:%M:%S"),
            "symbol":           result.symbol,
            "sector":           result.sector,
            "action":           result.action,
            "confidence":       str(result.confidence),
            "entry_price":      str(result.entry_price),
            "stop_loss":        str(result.stop_loss),
            "target":           str(result.target),
            "allocation_npr":   str(result.allocation_npr),
            "shares":           str(result.shares),
            "breakeven":        str(result.breakeven),
            "risk_reward":      str(result.risk_reward),
            "rsi_14":           str(result.rsi_14),
            "candle_pattern":   result.candle_pattern,
            "conf_score":       str(result.confidence),
            "geo_score":        str(result.geo_score),
            "macd_line":        str(getattr(result, "macd_line",         "")),
            "macd_signal":      str(getattr(result, "macd_signal",       "")),
            "obv_trend":        str(getattr(result, "obv_trend",         "")),
            "volume_ratio":     str(getattr(result, "volume_ratio",      "")),
            "fundamental_score":str(getattr(result, "fundamental_score", "")),
            "macro_score":      str(result.geo_score),
            "reasoning":        (
                f"[Claude|{result.primary_signal}|{result.urgency}] "
                f"{result.reasoning}"
                + (f" | Gemini: {result.gemini_reason}" if result.gemini_reason else "")
                + (f" | Lesson: {result.lesson_applied}" if result.lesson_applied and result.lesson_applied != "NONE" else "")
            )[:800],
            "outcome":          "PENDING",
            "timestamp":        result.timestamp,
        }

        # Update existing Gemini row if we have its id
        if result.market_log_id:
            row["id"] = result.market_log_id
            ok = upsert_row("market_log", row, conflict_columns=["id"])
        else:
            from sheets import write_row
            ok = write_row("market_log", row)

        if ok:
            logger.info("market_log written: %s %s conf=%d%%",
                        result.action, result.symbol, result.confidence)

            # Delete the original Gemini row if we wrote a new one (no id case)
            # If we upserted by id, the row is already updated in place — no delete needed

        return ok

    except Exception as exc:
        logger.error("DB error: %s", exc)
        return False

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis(flags: list) -> list[AnalystResult]:
    """
    Main entry point. Called by trading.yml after gemini_filter.

    Args:
        flags: list[GeminiFlag] from gemini_filter.run_gemini_filter()

    Returns:
        list[AnalystResult] — all results including WAIT/AVOID
        BUY results are written to market_log.
    """
    if not flags:
        logger.info("claude_analyst: no flags to analyze")
        return []

    logger.info("=" * 60)
    logger.info("claude_analyst.run_analysis() — %d flags", len(flags))

    # ── Load shared context once ──────────────────────────────────────────────
    portfolio    = _load_portfolio()
    geo          = _load_geo_context()
    macro        = _load_macro_context()
    market_state = _load_market_state()
    loss_streak  = _load_loss_streak()

    logger.info(
        "Context: market=%s | liquid=NPR %.0f | slots=%d | loss_streak=%d",
        market_state,
        portfolio.get("liquid_npr", 0),
        portfolio.get("slots_remaining", 0),
        loss_streak,
    )

    # ── Portfolio full check ──────────────────────────────────────────────────
    if portfolio.get("slots_remaining", 0) <= 0:
        logger.info("Portfolio full (3/3 positions open) — no analysis needed")
        return []

    # ── Circuit breaker ───────────────────────────────────────────────────────
    if loss_streak >= 8:
        logger.warning("Circuit breaker: loss_streak=%d — skipping all analysis", loss_streak)
        return []

    results: list[AnalystResult] = []

    for flag in flags:
        sym = flag.symbol
        logger.info("─" * 40)
        logger.info("Analyzing %s [%s] urgency=%s", sym, flag.sector, flag.urgency)

        # Skip if portfolio now full (from earlier BUY in this run)
        buy_count = sum(1 for r in results if r.action == "BUY")
        if portfolio.get("open_positions", 0) + buy_count >= 100:
            logger.info("%s: portfolio full after earlier BUYs — skipping", sym)
            continue

        # Load per-symbol lessons
        lessons = _load_lessons(sym, getattr(flag, "sector", ""))

        # Build prompt and call Claude
        prompt = _build_prompt(
            flag, portfolio, geo, macro, lessons, market_state, loss_streak
        )
        claude_json = _call_claude(prompt)#ask_claude(prompt)               #

        if claude_json is None:
            logger.warning("%s: Claude returned no result — skipping", sym)
            continue

        result = _assemble_result(claude_json, flag)
        results.append(result)

        # Write ALL results to DB (BUY, WAIT, AVOID all get logged)
        _write_to_db(result)

        logger.info("Result: %s", result.summary())

    # ── Summary ───────────────────────────────────────────────────────────────
    buys   = [r for r in results if r.action == "BUY"]
    waits  = [r for r in results if r.action == "WAIT"]
    avoids = [r for r in results if r.action == "AVOID"]

    logger.info(
        "claude_analyst done: %d analyzed | %d BUY | %d WAIT | %d AVOID",
        len(results), len(buys), len(waits), len(avoids),
    )
    for r in buys:
        logger.info("  BUY: %s", r.summary())

    return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FOR notifier.py
# ══════════════════════════════════════════════════════════════════════════════

def format_buy_signal(result: AnalystResult) -> str:
    """
    Format a BUY result for Telegram notification.
    Called by notifier.py.
    """
    true_profit = _calc_true_profit(result.entry_price, result.target, result.shares)

    lines = [
        f"📈 *BUY SIGNAL — {result.symbol}*",
        f"Sector:      {result.sector}",
        f"Confidence:  {result.confidence}%  |  Signal: {result.primary_signal}",
        f"Entry:       NPR {result.entry_price:,.0f}",
        f"Stop Loss:   NPR {result.stop_loss:,.0f}",
        f"Target:      NPR {result.target:,.0f}",
        f"Breakeven:   NPR {result.breakeven:,.0f}",
        f"R/R Ratio:   {result.risk_reward:.1f}x",
        f"Shares:      {result.shares}",
        f"Allocation:  NPR {result.allocation_npr:,.0f}",
        f"True Profit: NPR {true_profit:+,.0f} (after all Nepal fees)",
        f"Hold:        ~{result.suggested_hold} days",
        f"",
        f"Reasoning: {result.reasoning[:200]}",
    ]

    if result.lesson_applied and result.lesson_applied != "NONE":
        lines.append(f"Lesson Applied: {result.lesson_applied[:100]}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python claude_analyst.py              → run with live gemini_filter data
#   python claude_analyst.py --dry-run    → use synthetic flag, no DB write
#   python claude_analyst.py NABIL        → analyze specific symbol directly
#   python claude_analyst.py --print-prompt => print prompt 
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [CLAUDE_ANALYST] %(levelname)s: %(message)s",
    )

    args         = sys.argv[1:]
    dry_run      = "--dry-run"      in args
    print_prompt = "--print-prompt" in args
    sym_args     = [a.upper() for a in args if not a.startswith("--")]

    print("\n" + "=" * 70)
    print("  NEPSE AI — claude_analyst.py")
    print("=" * 70)

    # ── Print-prompt mode: full pipeline run but NO Claude API call ───────────
    if print_prompt:
        print("\n[PRINT-PROMPT MODE] Running gemini_filter → building Claude prompt(s) — NO API call\n")

        # Step 1: run gemini_filter exactly as live
        print("[1/3] Running gemini_filter to get real flags...")
        try:
            from gemini_filter import run_gemini_filter
            flags = run_gemini_filter()
            if not flags:
                print("  No flags from Gemini — nothing to print")
                sys.exit(0)
            print(f"  {len(flags)} flag(s) received from Gemini\n")
        except Exception as e:
            print(f"  gemini_filter failed: {e}")
            sys.exit(1)

        # Filter to specific symbol if passed on CLI
        if sym_args:
            flags = [f for f in flags if f.symbol in sym_args]
            if not flags:
                print(f"  Symbol(s) {sym_args} not in Gemini flag list")
                sys.exit(0)

        # Step 2: load shared context exactly as live
        print("[2/3] Loading context (portfolio, geo, macro, market state)...")
        portfolio    = _load_portfolio()
        geo          = _load_geo_context()
        macro        = _load_macro_context()
        market_state = _load_market_state()
        loss_streak  = _load_loss_streak()
        print(f"  market={market_state} | liquid=NPR {portfolio.get('liquid_npr',0):,.0f} | "
              f"slots={portfolio.get('slots_remaining',0)} | loss_streak={loss_streak}\n")

        # Step 3: build and print prompt for each flag — NO Claude call
        print(f"[3/3] Building prompt(s) for {len(flags)} flag(s)...\n")
        for i, flag in enumerate(flags, 1):
            lessons = _load_lessons(flag.symbol, getattr(flag, "sector", ""))
            prompt  = _build_prompt(flag, portfolio, geo, macro, lessons, market_state, loss_streak)

            char_count = len(prompt)
            token_est  = char_count // 4

            print("=" * 70)
            print(f"  PROMPT {i}/{len(flags)} — {flag.symbol} [{getattr(flag,'sector','')}]")
            print(f"  Chars: {char_count} | ~Tokens: {token_est} | "
                  f"~Cost: ${token_est * 0.000003:.4f}")
            print("=" * 70)
            print(prompt)
            print()

        total_tokens = sum(len(_build_prompt(
            f, portfolio, geo, macro,
            _load_lessons(f.symbol, getattr(f, "sector", "")),
            market_state, loss_streak
        )) // 4 for f in flags)
        print("─" * 70)
        print(f"TOTAL: {len(flags)} prompt(s) | ~{total_tokens} tokens | "
              f"~${total_tokens * 0.000003:.4f} input cost if sent to Claude")
        print("[DONE] No API call made. Cost: $0.00")
        print("=" * 70 + "\n")
        sys.exit(0)

    # ── Dry run with synthetic flag ───────────────────────────────────────────
    if dry_run:
        print("\n[DRY RUN] Creating synthetic GeminiFlag for NABIL...\n")

        from dataclasses import dataclass as dc, field as f

        @dc
        class SyntheticFlag:
            symbol: str = "NABIL"
            sector: str = "commercial bank"
            ltp: float = 1240.0
            change_pct: float = 0.8
            urgency: str = "NORMAL"
            gemini_reason: str = "MACD bullish cross with volume surge"
            gemini_risk: str = "Banking sector excluded from optimal portfolio"
            primary_signal: str = "MACD"
            composite_score: float = 68.5
            tech_score: int = 65
            rsi_14: float = 42.3
            rsi_signal: str = "NEUTRAL"
            macd_cross: str = "BULLISH"
            bb_signal: str = "NEUTRAL"
            bb_pct_b: float = 0.42
            obv_trend: str = "RISING"
            ema_trend: str = "ABOVE_ALL"
            best_candle: str = "Hammer"
            candle_tier: int = 1
            candle_conf: int = 72
            cstar_signal: bool = False
            suggested_hold: int = 17
            geo_combined: int = 2

        flags = [SyntheticFlag()]
        print("  Synthetic flag created")

    # ── Live run from gemini_filter ───────────────────────────────────────────
    else:
        print("\n[1/2] Running gemini_filter to get flags...")
        try:
            from gemini_filter import run_gemini_filter
            flags = run_gemini_filter()
            if not flags:
                print("  No flags from Gemini — nothing to analyze")
                sys.exit(0)
            print(f"  {len(flags)} flags received")
        except Exception as e:
            print(f"  Failed to get flags: {e}")
            sys.exit(1)

        # Filter to specific symbol if passed
        if sym_args:
            flags = [f for f in flags if f.symbol in sym_args]
            if not flags:
                print(f"  Symbol(s) {sym_args} not in flag list")
                sys.exit(0)

    # ── Run analysis ──────────────────────────────────────────────────────────
    print(f"\n[2/2] Running Claude analysis on {len(flags)} flag(s)...")
    if dry_run:
        print("  [DRY RUN] DB writes disabled — results printed only\n")

    results = run_analysis(flags) if not dry_run else []

    # Dry run: manually call single analysis
    if dry_run and flags:
        portfolio    = _load_portfolio()
        geo          = _load_geo_context()
        macro        = _load_macro_context()
        market_state = _load_market_state()
        loss_streak  = _load_loss_streak()
        lessons      = _load_lessons(flags[0].symbol, flags[0].sector)

        prompt = _build_prompt(
            flags[0], portfolio, geo, macro, lessons, market_state, loss_streak
        )

        print("  Calling Claude...\n")
        claude_json = _call_claude(prompt)

        if claude_json:
            result = _assemble_result(claude_json, flags[0])
            results = [result]

    # ── Print results ─────────────────────────────────────────────────────────
    if results:
        print(f"\n  {'='*60}")
        print(f"  ANALYSIS RESULTS")
        print(f"  {'='*60}")
        for r in results:
            action_emoji = "✅" if r.action == "BUY" else ("⏸" if r.action == "WAIT" else "🚫")
            print(f"\n  {action_emoji} {r.action} — {r.symbol} [{r.sector}]")
            print(f"     Confidence:  {r.confidence}%")
            print(f"     Signal:      {r.primary_signal}")
            print(f"     Urgency:     {r.urgency}")
            if r.action == "BUY":
                print(f"     Entry:       NPR {r.entry_price:,.0f}")
                print(f"     Stop:        NPR {r.stop_loss:,.0f}")
                print(f"     Target:      NPR {r.target:,.0f}")
                print(f"     Breakeven:   NPR {r.breakeven:,.0f}")
                print(f"     R/R:         {r.risk_reward:.1f}x")
                print(f"     Shares:      {r.shares}")
                print(f"     Allocation:  NPR {r.allocation_npr:,.0f}")
                print(f"     Hold:        ~{r.suggested_hold} days")
                print()
                print("  ── TELEGRAM FORMAT ──────────────────────────────")
                print(format_buy_signal(r))
            print(f"\n     Reasoning:   {r.reasoning[:200]}")
            if r.lesson_applied and r.lesson_applied != "NONE":
                print(f"     Lesson:      {r.lesson_applied[:100]}")
    else:
        print("\n  No analysis results.")

    if not dry_run and results:
        buy_count = sum(1 for r in results if r.action == "BUY")
        print(f"\n  ✅ {buy_count} BUY signal(s) written to market_log table")

    print("\n" + "=" * 70 + "\n")