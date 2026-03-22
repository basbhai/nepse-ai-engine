"""
capital_allocator.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Full wealth management module.
          The "financial advisor" brain of the system.

What this does (plain English):
  1. Reads current market state (BULL/CAUTIOUS/BEAR/SIDEWAYS/CRISIS)
  2. Reads your live portfolio from Neon (synced by meroshare.py)
  3. Reads geo_score + nepal_score + NEPSE vs 200 DMA
  4. Reads your win rate and system confidence from financials table
  5. Uses Claude (via OpenRouter) to produce a full wealth management report
  6. Writes recommendation to capital_allocation + financial_advisor tables
  7. Returns structured advice for briefing.py to send via Telegram

What it recommends:
  STOCK ROLLING:
    "HBL is up 18% — trailing stop protecting gains.
     Consider rolling 50% profit into NABIL at RSI 36.
     Same sector. Net exposure unchanged. Lock partial gains."

  STOCK CLEARANCE:
    "Market turned BEAR. NEPSE crossed below 200 DMA.
     Recommend exiting ALL positions today.
     Move NPR 87,000 to FD at 8.5% for 6 months."

  ADDITIONAL ACQUIRING:
    "NABIL pulled back 8% on no bad news.
     Your WACC is 1,180. Current 1,090.
     Recommend buying 50 more units.
     Average down WACC to 1,140."

  FD RECOMMENDATION:
    "Market BEARISH. Geo score -3.
     Recommend: 60% → 6-month FD at 8.5%
                20% → savings account
                20% → dry powder for market recovery"

  3-MONTH FORECAST:
    "NEPSE outlook: POSITIVE. Confidence: 68%.
     Key catalyst: Budget season May-June.
     Recommendation: Hold current stocks."

─────────────────────────────────────────────────────────────────────────────

SOP — STANDARD OPERATING PROCEDURE
───────────────────────────────────
WHAT IT DOES:
  Reads all system data and produces a full wealth management recommendation
  using Claude as the reasoning engine.

HOW TO TEST:
  python capital_allocator.py        → full analysis, print report
  python capital_allocator.py latest → print last recommendation from Neon

WHEN IT RUNS:
  Called by morning_brief.yml (10:30 AM NST) — before market opens
  Called by weekly_review.yml (Sunday 6 PM) — weekly deep review
  Can be called anytime manually for on-demand advice

COMMON ERRORS AND FIXES:
  "No portfolio data" → Run meroshare.py first to sync holdings
  "No geo data"       → Run geo_sentiment.py first
  "Claude API error"  → Check OPENROUTER_API_KEY in .env
  DB write fails      → Logged, recommendation still printed to console

─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from AI.gemini import ask_ai
from dotenv import load_dotenv

from sheets import (
    get_setting,
    write_row,
    read_tab,
    get_latest_geo,
    get_macro_data,
    get_latest_pulse,
)
from modules.meroshare import get_portfolio_summary, PortfolioSummary

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ALLOCATOR] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST = timezone(timedelta(hours=5, minutes=45))



# NEPSE transaction costs (from handoff Section 8.1)
BROKERAGE_PCT  = 0.40   # % on buy and sell
SEBON_PCT      = 0.015  # % Securities Board fee
DP_CHARGE_NPR  = 25.0   # flat per transaction
CGT_PCT        = 5.0    # % capital gains tax on profit

# Allocation targets by market state (from handoff Section 3.2)
ALLOCATION_TARGETS = {
    "FULL_BULL": {
        "stocks_pct": 80, "savings_pct": 20, "fd_pct": 0, "od_pct": 0,
        "description": "Full trading mode — 80% stocks, 20% cash",
    },
    "CAUTIOUS_BULL": {
        "stocks_pct": 60, "savings_pct": 40, "fd_pct": 0, "od_pct": 0,
        "description": "Selective trading — 60% stocks, 40% savings",
    },
    "SIDEWAYS": {
        "stocks_pct": 30, "savings_pct": 30, "fd_pct": 40, "od_pct": 0,
        "description": "No new buys — 30% stocks, 40% FD, 30% savings",
    },
    "BEAR": {
        "stocks_pct": 0, "savings_pct": 30, "fd_pct": 60, "od_pct": 10,
        "description": "Sell all — 60% FD, 30% savings, 10% OD",
    },
    "CRISIS": {
        "stocks_pct": 0, "savings_pct": 20, "fd_pct": 80, "od_pct": 0,
        "description": "Emergency exit — 100% FD or savings",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — GATHER ALL CONTEXT
# ──────────────────────────────────────────────────────────────────────────────
# Pull everything Claude needs from Neon before making the API call.
# ══════════════════════════════════════════════════════════════════════════════

def _get_market_state() -> str:
    """Read current market state from settings. Defaults to SIDEWAYS."""
    return get_setting("MARKET_STATE", default="SIDEWAYS").upper().strip()


def _get_nepse_vs_200dma() -> dict:
    """
    Read NEPSE index and 200 DMA from Neon.
    Returns dict with index, dma, position (ABOVE/BELOW/AT).
    """
    try:
        # Read latest market breadth for NEPSE index value
        rows = read_tab("market_breadth", limit=1)
        nepse_index = float(rows[0].get("nepse_index", 0)) if rows else 0.0

        # Read 200 DMA from settings (updated by morning_brief or manually)
        dma_200 = float(get_setting("NEPSE_200DMA", default="2400"))

        if nepse_index <= 0:
            return {"index": 0, "dma_200": dma_200, "position": "UNKNOWN", "pct_from_dma": 0}

        pct_from_dma = ((nepse_index - dma_200) / dma_200 * 100)
        if nepse_index > dma_200 * 1.02:
            position = "ABOVE"
        elif nepse_index < dma_200 * 0.98:
            position = "BELOW"
        else:
            position = "AT"

        return {
            "index":       round(nepse_index, 1),
            "dma_200":     round(dma_200, 1),
            "position":    position,
            "pct_from_dma": round(pct_from_dma, 1),
        }

    except Exception as exc:
        log.warning("Could not read NEPSE vs 200DMA: %s", exc)
        return {"index": 0, "dma_200": 0, "position": "UNKNOWN", "pct_from_dma": 0}


def _get_system_confidence() -> dict:
    """
    Read win rate and system confidence from financials table.
    Returns dict with win_rate_30d, confidence_pct, loss_streak.
    """
    try:
        rows = read_tab("financials")
        kpi_map = {r.get("kpi_name", ""): r.get("current_value", "") for r in rows}
        return {
            "win_rate_30d":  float(kpi_map.get("Win_Rate_30d", 0) or 0),
            "profit_factor": float(kpi_map.get("Profit_Factor", 0) or 0),
            "max_drawdown":  float(kpi_map.get("Max_Drawdown_Pct", 0) or 0),
            "loss_streak":   int(float(kpi_map.get("Current_Loss_Streak", 0) or 0)),
            "alpha_vs_nepse": float(kpi_map.get("Alpha_vs_NEPSE", 0) or 0),
        }
    except Exception as exc:
        log.warning("Could not read system confidence: %s", exc)
        return {"win_rate_30d": 0, "profit_factor": 0, "max_drawdown": 0, "loss_streak": 0, "alpha_vs_nepse": 0}


def _get_recent_lessons(n: int = 5) -> list[str]:
    """Read last N lessons from Learning Hub."""
    try:
        rows = read_tab("learning_hub", limit=n)
        return [
            f"{r.get('symbol','?')} — {r.get('lesson','')[:100]}"
            for r in rows if r.get("lesson")
        ]
    except Exception:
        return []


def _calculate_fee(shares: int, price: float) -> dict:
    """
    Calculate full NEPSE transaction cost for selling a position.
    Returns breakdown of all fees.
    """
    trade_value  = shares * price
    brokerage    = trade_value * BROKERAGE_PCT / 100
    sebon        = trade_value * SEBON_PCT / 100
    total_cost   = brokerage + sebon + DP_CHARGE_NPR

    return {
        "trade_value": round(trade_value, 2),
        "brokerage":   round(brokerage, 2),
        "sebon":       round(sebon, 2),
        "dp_charge":   DP_CHARGE_NPR,
        "total_cost":  round(total_cost, 2),
        "cost_pct":    round(total_cost / trade_value * 100, 3) if trade_value > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD CLAUDE PROMPT
# ──────────────────────────────────────────────────────────────────────────────
# All context assembled into one comprehensive prompt for Claude.
# Claude is the reasoning engine — numbers come from Python, judgment from Claude.
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    portfolio:     PortfolioSummary,
    market_state:  str,
    nepse:         dict,
    geo:           dict,
    nepal:         dict,
    macro:         dict,
    confidence:    dict,
    lessons:       list[str],
    total_capital: float,
    fd_rate:       float,
) -> str:
    """
    Build the full wealth management prompt for Claude.
    """
    # Portfolio details
    portfolio_text = ""
    if portfolio and portfolio.holdings:
        lines = []
        for h in portfolio.holdings:
            fee_info = _calculate_fee(h.shares, h.current_price)
            lines.append(
                f"  {h.symbol}: {h.shares} shares | WACC {h.wacc:.2f} | "
                f"LTP {h.current_price:.2f} | P&L {h.pnl_pct:+.1f}% "
                f"(NPR {h.pnl_npr:+,.0f}) | Sell cost: NPR {fee_info['total_cost']:.0f}"
            )
        portfolio_text = "\n".join(lines)
        invested       = portfolio.total_cost_npr
        liquid         = max(0, total_capital - invested)
    else:
        portfolio_text = "  No open positions"
        invested       = 0
        liquid         = total_capital

    # Target allocation for current state
    target = ALLOCATION_TARGETS.get(market_state, ALLOCATION_TARGETS["SIDEWAYS"])

    prompt = f"""You are a Nepal stock market wealth manager and financial advisor.
Analyze the complete situation below and provide specific actionable advice.

═══════════════════════════════════════════════
MARKET CONTEXT
═══════════════════════════════════════════════
Market State:      {market_state}
NEPSE Index:       {nepse['index']:,.0f}  ({nepse['position']} 200DMA by {nepse['pct_from_dma']:+.1f}%)
200 DMA:           {nepse['dma_200']:,.0f}

Geo Score:         {geo.get('geo_score', 0)} / 5  ({geo.get('status', 'UNKNOWN')})
Nepal Score:       {nepal.get('nepal_score', 0)} / 5  ({nepal.get('nepal_status', 'UNKNOWN')})
Combined Geo:      {int(geo.get('geo_score', 0) or 0) + int(nepal.get('nepal_score', 0) or 0)} / 10

Key Global Event:  {geo.get('key_event', 'None')}
Key Nepal Event:   {nepal.get('key_event', 'None')}

═══════════════════════════════════════════════
MACRO DATA (Nepal — NRB Latest)
═══════════════════════════════════════════════
Policy Rate:       {macro.get('Policy_Rate', 'Unknown')}%
Inflation:         {macro.get('Inflation_Pct', 'Unknown')}%
Remittance YoY:    {macro.get('Remittance_YoY_Pct', 'Unknown')}%
Forex Reserve:     {macro.get('Forex_Reserve_Months', 'Unknown')} months import cover
FD Rate (1yr):     {fd_rate}%

═══════════════════════════════════════════════
YOUR PORTFOLIO
═══════════════════════════════════════════════
Total Capital:     NPR {total_capital:,.0f}
Invested:          NPR {invested:,.0f}
Liquid/Cash:       NPR {liquid:,.0f}
Total P&L:         NPR {portfolio.total_pnl_npr if portfolio else 0:+,.0f} ({portfolio.total_pnl_pct if portfolio else 0:+.1f}%)

Positions:
{portfolio_text}

═══════════════════════════════════════════════
SYSTEM PERFORMANCE
═══════════════════════════════════════════════
Win Rate (30d):    {confidence['win_rate_30d']:.0f}%
Profit Factor:     {confidence['profit_factor']:.1f}
Max Drawdown:      {confidence['max_drawdown']:.1f}%
Current Loss Run:  {confidence['loss_streak']} consecutive
Alpha vs NEPSE:    {confidence['alpha_vs_nepse']:+.1f}%

═══════════════════════════════════════════════
RECENT LEARNING HUB LESSONS
═══════════════════════════════════════════════
{chr(10).join(lessons) if lessons else "No lessons recorded yet"}

═══════════════════════════════════════════════
RECOMMENDED TARGET ALLOCATION FOR {market_state}
═══════════════════════════════════════════════
Stocks:   {target['stocks_pct']}% = NPR {total_capital * target['stocks_pct'] / 100:,.0f}
FD:       {target['fd_pct']}% = NPR {total_capital * target['fd_pct'] / 100:,.0f}
Savings:  {target['savings_pct']}% = NPR {total_capital * target['savings_pct'] / 100:,.0f}
OD:       {target['od_pct']}% (use only if good opportunity with low rate)

═══════════════════════════════════════════════
TASK
═══════════════════════════════════════════════
Provide a complete wealth management recommendation covering:

1. CAPITAL ALLOCATION — Should allocation change? What exact amounts go where?

2. EXISTING POSITIONS — For each open position:
   - HOLD (reason)?
   - SELL (partial or full, reason, urgency)?
   - AVERAGE DOWN (buy more, reason, how much)?
   - Roll profits into another stock (which one, why)?

3. FD RECOMMENDATION — If capital should go to FD:
   - Exact amount
   - Recommended duration (3/6/12 months)
   - Why now vs waiting

4. NEW OPPORTUNITIES — Given current market state, is this a good time to buy?
   What sectors look interesting?

5. THREE-MONTH OUTLOOK — Where do you see NEPSE in 3 months?
   Key catalysts and risks. Confidence level (%).

6. URGENT ACTIONS — Any actions needed TODAY before market opens?

Be specific with NPR amounts. Reference the Learning Hub lessons where relevant.
If market is BEAR/CRISIS, be direct about capital preservation over profit.

Return ONLY this JSON with no other text:
{{
  "capital_allocation": {{
    "stocks_pct": number,
    "fd_pct": number,
    "savings_pct": number,
    "od_pct": number,
    "stocks_npr": number,
    "fd_npr": number,
    "savings_npr": number,
    "reasoning": "one sentence"
  }},
  "position_actions": [
    {{
      "symbol": "symbol",
      "action": "HOLD or SELL or AVERAGE_DOWN or ROLL",
      "urgency": "TODAY or THIS_WEEK or OPTIONAL",
      "shares_to_act": number,
      "reasoning": "specific reason",
      "roll_into": "symbol or empty string"
    }}
  ],
  "fd_recommendation": {{
    "amount_npr": number,
    "duration_months": number,
    "reasoning": "specific reason or empty if no FD needed"
  }},
  "new_opportunity": {{
    "good_time_to_buy": true_or_false,
    "sectors_to_watch": ["sector1", "sector2"],
    "reasoning": "one sentence"
  }},
  "three_month_outlook": {{
    "direction": "BULLISH or NEUTRAL or BEARISH",
    "confidence_pct": number,
    "nepse_range_low": number,
    "nepse_range_high": number,
    "key_catalyst": "main positive catalyst",
    "key_risk": "main risk to watch",
    "summary": "two sentence outlook"
  }},
  "urgent_actions": "specific actions needed today, or 'No urgent actions'",
  "overall_advice": "two sentence summary of the full recommendation"
}}"""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ASK CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

def _ask_claude(prompt: str) -> Optional[dict]:
     return ask_ai(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WRITE TO NEON
# ══════════════════════════════════════════════════════════════════════════════

def _write_recommendation(result: dict, market_state: str, nepse: dict) -> bool:
    """Write full recommendation to capital_allocation and financial_advisor tables."""
    nst_now = datetime.now(tz=NST)

    alloc  = result.get("capital_allocation", {})
    outlook = result.get("three_month_outlook", {})

    # capital_allocation table — current snapshot
    cap_row = {
        "date":           nst_now.strftime("%Y-%m-%d"),
        "market_state":   market_state,
        "nepse_vs_200dma": str(nepse.get("pct_from_dma", 0)),
        "stocks_pct":     str(alloc.get("stocks_pct", "")),
        "fd_pct":         str(alloc.get("fd_pct", "")),
        "savings_pct":    str(alloc.get("savings_pct", "")),
        "od_pct":         str(alloc.get("od_pct", "")),
        "fd_rate_used":   get_setting("FD_RATE_PCT", "8.5"),
        "expected_return": str(outlook.get("confidence_pct", "")),
        "reasoning":      result.get("overall_advice", ""),
        "review_date":    nst_now.strftime("%Y-%m-%d"),
        "status":         "ACTIVE",
    }

    # financial_advisor table — 3-month forecast
    adv_row = {
        "date":                   nst_now.strftime("%Y-%m-%d"),
        "recommendation_type":    "WEALTH_MANAGEMENT",
        "market_phase":           market_state,
        "confidence_pct":         str(outlook.get("confidence_pct", "")),
        "capital_in_stocks_pct":  str(alloc.get("stocks_pct", "")),
        "capital_in_fd_pct":      str(alloc.get("fd_pct", "")),
        "capital_in_savings_pct": str(alloc.get("savings_pct", "")),
        "capital_in_od_pct":      str(alloc.get("od_pct", "")),
        "three_month_outlook":    outlook.get("summary", ""),
        "expected_return_pct":    str(outlook.get("confidence_pct", "")),
        "fd_rate_used":           get_setting("FD_RATE_PCT", "8.5"),
        "trigger_to_change":      outlook.get("key_risk", ""),
        "review_date":            nst_now.strftime("%Y-%m-%d"),
        "actual_outcome":         "",
        "was_forecast_correct":   "",
    }

    ok1 = write_row("capital_allocation", cap_row)
    ok2 = write_row("financial_advisor",  adv_row)

    if ok1 and ok2:
        log.info("✅ Recommendation written to Neon")
    else:
        log.warning("Partial write to Neon — cap_allocation: %s, financial_advisor: %s", ok1, ok2)

    return ok1 and ok2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run() -> Optional[dict]:
    """
    Full capital allocation analysis.

    Flow:
      1. Gather all context (portfolio, market, geo, macro, confidence)
      2. Build Claude prompt
      3. Ask Claude for wealth management recommendation
      4. Write to Neon
      5. Return structured result for briefing.py

    Returns full result dict or None on failure.
    """
    nst_now = datetime.now(tz=NST)
    log.info("=" * 60)
    log.info("CAPITAL ALLOCATOR starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # ── Gather context ────────────────────────────────────────────────────────
    portfolio    = get_portfolio_summary()
    market_state = _get_market_state()
    nepse        = _get_nepse_vs_200dma()
    geo          = get_latest_geo()      or {}
    nepal        = get_latest_pulse()    or {}
    macro        = get_macro_data()      or {}
    confidence   = _get_system_confidence()
    lessons      = _get_recent_lessons(5)
    total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
    fd_rate       = float(get_setting("FD_RATE_PCT", "8.5"))

    log.info(
        "Context: Market=%s | NEPSE=%s 200DMA | Geo=%s | Nepal=%s | WinRate=%.0f%%",
        market_state,
        nepse.get("position", "?"),
        geo.get("geo_score", "?"),
        nepal.get("nepal_score", "?"),
        confidence.get("win_rate_30d", 0),
    )

    # ── Build prompt and ask Claude ───────────────────────────────────────────
    prompt = _build_prompt(
        portfolio, market_state, nepse, geo, nepal,
        macro, confidence, lessons, total_capital, fd_rate
    )

    result = _ask_claude(prompt)

    if not result:
        log.error("❌ No recommendation received from Claude")
        return None

    # ── Write to Neon ─────────────────────────────────────────────────────────
    _write_recommendation(result, market_state, nepse)

    # ── Log summary ───────────────────────────────────────────────────────────
    alloc   = result.get("capital_allocation", {})
    outlook = result.get("three_month_outlook", {})
    urgent  = result.get("urgent_actions", "")

    log.info("✅ Capital Allocation Complete")
    log.info("   Stocks: %s%%  FD: %s%%  Savings: %s%%",
             alloc.get("stocks_pct"), alloc.get("fd_pct"), alloc.get("savings_pct"))
    log.info("   Outlook: %s (%s%% confidence)",
             outlook.get("direction"), outlook.get("confidence_pct"))
    if urgent and urgent != "No urgent actions":
        log.warning("   URGENT: %s", urgent)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def _print_result(result: dict) -> None:
    """Pretty print the full recommendation to console."""
    if not result:
        print("\n  No recommendation available\n")
        return

    alloc    = result.get("capital_allocation", {})
    outlook  = result.get("three_month_outlook", {})
    actions  = result.get("position_actions", [])
    fd_rec   = result.get("fd_recommendation", {})
    opp      = result.get("new_opportunity", {})
    urgent   = result.get("urgent_actions", "")
    advice   = result.get("overall_advice", "")

    print(f"\n{'='*60}")
    print(f"  WEALTH MANAGEMENT REPORT")
    print(f"{'='*60}")
    print(f"\n  OVERALL: {advice}")

    print(f"\n  CAPITAL ALLOCATION:")
    print(f"    Stocks:  {alloc.get('stocks_pct')}%  (NPR {alloc.get('stocks_npr', 0):,.0f})")
    print(f"    FD:      {alloc.get('fd_pct')}%  (NPR {alloc.get('fd_npr', 0):,.0f})")
    print(f"    Savings: {alloc.get('savings_pct')}%  (NPR {alloc.get('savings_npr', 0):,.0f})")

    if actions:
        print(f"\n  POSITION ACTIONS:")
        for a in actions:
            urgency = f"[{a.get('urgency','?')}]" if a.get('urgency') != "OPTIONAL" else ""
            roll    = f" → roll into {a.get('roll_into')}" if a.get("roll_into") else ""
            print(f"    {a.get('symbol')}: {a.get('action')}{roll} {urgency}")
            print(f"      {a.get('reasoning','')}")

    if fd_rec.get("amount_npr", 0) > 0:
        print(f"\n  FD RECOMMENDATION:")
        print(f"    Amount:   NPR {fd_rec.get('amount_npr', 0):,.0f}")
        print(f"    Duration: {fd_rec.get('duration_months')} months")
        print(f"    Reason:   {fd_rec.get('reasoning','')}")

    print(f"\n  3-MONTH OUTLOOK: {outlook.get('direction')} ({outlook.get('confidence_pct')}% confidence)")
    print(f"    NEPSE range:  {outlook.get('nepse_range_low', '?')} – {outlook.get('nepse_range_high', '?')}")
    print(f"    Catalyst:     {outlook.get('key_catalyst','')}")
    print(f"    Risk:         {outlook.get('key_risk','')}")
    print(f"    Summary:      {outlook.get('summary','')}")

    if opp.get("good_time_to_buy"):
        print(f"\n  NEW OPPORTUNITIES: {', '.join(opp.get('sectors_to_watch', []))}")

    if urgent and urgent != "No urgent actions":
        print(f"\n  ⚠️  URGENT: {urgent}")

    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python capital_allocator.py        → full analysis
#   python capital_allocator.py latest → print last rec from Neon
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ALLOCATOR] %(levelname)s: %(message)s",
    )

    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "latest":
        rows = read_tab("financial_advisor", limit=1)
        if rows:
            row = rows[0]
            print(f"\n  Latest recommendation: {row.get('date')}")
            print(f"  Market phase:    {row.get('market_phase')}")
            print(f"  Stocks:          {row.get('capital_in_stocks_pct')}%")
            print(f"  FD:              {row.get('capital_in_fd_pct')}%")
            print(f"  Outlook:         {row.get('three_month_outlook')}")
            print(f"  Confidence:      {row.get('confidence_pct')}%\n")
        else:
            print("  No recommendations in database yet\n")
        sys.exit(0)

    result = run()
    _print_result(result)
    sys.exit(0 if result else 1)
