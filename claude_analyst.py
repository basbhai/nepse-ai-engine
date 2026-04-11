# -*- coding: utf-8 -*-
"""
claude_analyst.py
-----------------------------------------------------------------------------
NEPSE AI Engine - Phase 3, Module 3

Key architectural rule:
    gemini_filter._write_log() writes a GEMINI_FLAG_* row to market_log
    and stores the row id on flag.market_log_id.
    claude_analyst._write_to_db() UPDATES that same row in place --
    one row per signal, never two.
    Falls back to write_row() (insert) only if market_log_id is missing.

CLI:
    python claude_analyst.py              -> live run
    python claude_analyst.py --dry-run    -> synthetic flag, no DB write
    python claude_analyst.py --print-prompt -> print prompt, no API call
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from config import NST

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class AnalystResult:
    symbol:             str
    action:             str   = "WAIT"
    confidence:         int   = 0
    entry_price:        float = 0.0
    stop_loss:          float = 0.0
    target:             float = 0.0
    allocation_npr:     float = 0.0
    shares:             int   = 0
    breakeven:          float = 0.0
    risk_reward:        float = 0.0
    suggested_hold:     int   = 17
    reasoning:          str   = ""
    lesson_applied:     str   = ""
    primary_signal:     str   = ""
    sector:             str   = ""
    geo_score:          int   = 0
    rsi_14:             float = 0.0
    candle_pattern:     str   = ""
    urgency:            str   = "NORMAL"
    gemini_reason:      str   = ""
    market_log_id:      int   = None
    support_level:      float = 0.0
    resistance_level:   float = 0.0
    headlines_politics: str   = ""
    headlines_economy:  str   = ""
    headlines_stock:    str   = ""

    timestamp: str = field(default_factory=lambda:
                   datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.action == "BUY":
            return (
                f"BUY {self.symbol} | conf={self.confidence}% | "
                f"entry={self.entry_price:.0f} stop={self.stop_loss:.0f} "
                f"target={self.target:.0f} | NPR {self.allocation_npr:,.0f} | "
                f"hold ~{self.suggested_hold}d | {self.reasoning[:60]}"
            )
        return (
            f"{'WAIT' if self.action == 'WAIT' else 'AVOID'} {self.symbol} | "
            f"{self.reasoning[:80]}"
        )


# =============================================================================
# SECTION 1 - CONTEXT LOADERS
# =============================================================================

def _load_portfolio() -> dict:
    try:
        from sheets import get_setting, read_tab
        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
        rows          = read_tab("portfolio")
        open_rows     = [r for r in rows if r.get("status", "").upper() == "OPEN"]
        invested      = sum(float(r.get("total_cost", 0) or 0) for r in open_rows)
        liquid        = max(0.0, total_capital - invested)
        holdings      = [
            {
                "symbol":  r.get("symbol", "?"),
                "shares":  r.get("shares", 0),
                "wacc":    r.get("wacc", 0),
                "pnl_pct": r.get("pnl_pct", 0),
                "pnl_npr": r.get("pnl_npr", 0),
            }
            for r in open_rows
        ]
        return {
            "total_capital_npr": total_capital,
            "liquid_npr":        liquid,
            "invested_npr":      invested,
            "open_positions":    len(open_rows),
            "slots_remaining":   max(0, 3 - len(open_rows)),
            "holdings":          holdings,
        }
    except Exception as exc:
        logger.warning("_load_portfolio failed: %s", exc)
        return {
            "total_capital_npr": 100000,
            "liquid_npr":        100000,
            "invested_npr":      0,
            "open_positions":    0,
            "slots_remaining":   3,
            "holdings":          [],
        }


def _load_geo_context() -> dict:
    """
    Read latest geo + nepal scores + headlines from their respective tables.

    geo_score comes from geopolitical_data (via get_latest_geo).
    nepal_score + headlines come from nepal_pulse (via get_latest_pulse).
    Headlines are stored as separate columns in nepal_pulse:
        headlines_politics, headlines_economy, headlines_stock
    """
    try:
        from sheets import get_latest_geo, get_latest_pulse
        geo   = get_latest_geo()   or {}
        pulse = get_latest_pulse() or {}

        # -- headlines: explicit key lookup with empty-string fallback
        # Column names in nepal_pulse table must match exactly.
        # If your table uses a different naming convention, adjust here.
        headlines_politics = (
            pulse.get("headlines_politics")
            or pulse.get("headlines_political")
            or pulse.get("political_headlines")
            or ""
        )
        headlines_economy = (
            pulse.get("headlines_economy")
            or pulse.get("headlines_economic")
            or pulse.get("economic_headlines")
            or ""
        )
        headlines_stock = (
            pulse.get("headlines_stock")
            or pulse.get("headlines_stocks")
            or pulse.get("stock_headlines")
            or ""
        )

        return {
            "geo_score":          int(geo.get("geo_score",    0) or 0),
            "nepal_score":        int(pulse.get("nepal_score",0) or 0),
            "combined":           int(geo.get("geo_score", 0) or 0) + int(pulse.get("nepal_score", 0) or 0),
            "geo_status":         geo.get("geo_status",      "NEUTRAL"),
            "nepal_status":       pulse.get("status",        "NEUTRAL"),
            "bandh":              pulse.get("bandh_today",   "NO"),
            "ipo_drain":          pulse.get("ipo_fpo_active","NO"),
            "crisis_detected":    pulse.get("crisis_detected","NO"),
            "key_geo_event":      geo.get("key_event",       "None"),
            "key_nepal_event":    pulse.get("key_event",     "None"),
            "dxy":                geo.get("dxy",              "99"),
            "headlines_politics": str(headlines_politics),
            "headlines_economy":  str(headlines_economy),
            "headlines_stock":    str(headlines_stock),
        }
    except Exception as exc:
        logger.warning("_load_geo_context failed: %s", exc)
        return {
            "geo_score": 0, "nepal_score": 0, "combined": 0,
            "geo_status": "NEUTRAL", "nepal_status": "NEUTRAL",
            "bandh": "NO", "ipo_drain": "NO", "crisis_detected": "NO",
            "key_geo_event": "None", "key_nepal_event": "None", "dxy": "99",
            "headlines_politics": "", "headlines_economy": "", "headlines_stock": "",
        }


def _load_macro_context() -> dict:
    try:
        from sheets import get_setting, run_raw_sql
        rows = run_raw_sql("SELECT * FROM nrb_monthly ORDER BY id DESC LIMIT 1", ())
        row  = rows[0] if rows else {}
        _fwd = (row.get("forward_guidance") or "").upper()
        nrb_decision = (
            "HIKE"     if _fwd == "TIGHT"
            else "CUT" if _fwd == "LOOSE"
            else "UNCHANGED"
        )
        return {
            "policy_rate":          row.get("policy_rate",               "?"),
            "nrb_rate_decision":    nrb_decision,
            "inflation_pct":        row.get("cpi_inflation",             "?"),
            "remittance_yoy_pct":   row.get("remittance_yoy_change_pct", "?"),
            "forex_reserve_months": row.get("fx_reserve_months",         "?"),
            "lending_rate":         row.get("bank_rate",                 "?"),
            "period":               row.get("period",                    "?"),
            "fd_rate":              get_setting("FD_RATE_PCT",            "8.5"),
            "fd_signal":            get_setting("FD_SCORE_SIGNAL",        "NEUTRAL"),
        }
    except Exception as exc:
        logger.warning("_load_macro_context failed: %s", exc)
        return {}


def _load_lessons(symbol: str, sector: str, limit: int = 6) -> list[str]:
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT symbol, sector, lesson_type, condition, finding, action,
                   confidence_level, win_rate, trade_count
            FROM learning_hub
            WHERE active = 'true'
              AND (symbol = %s OR sector = %s OR symbol = 'MARKET' OR applies_to = 'ALL')
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
            (symbol.upper(), sector.lower(), symbol.upper(), sector.lower(), limit)
        )
        lessons = []
        for r in rows:
            sym  = r.get("symbol", "?")
            ltype= r.get("lesson_type", "")
            cond = r.get("condition",   "")[:80]
            find = r.get("finding",     "")[:120]
            act  = r.get("action",      "")
            conf = r.get("confidence_level", "LOW")
            wr   = r.get("win_rate",    "")
            n    = r.get("trade_count", "")
            stat = f" (win_rate={wr}, n={n})" if n else ""
            lessons.append(f"[{sym}|{ltype}|{conf}] IF {cond} -> {act}: {find}{stat}")
        return lessons
    except Exception as exc:
        logger.warning("_load_lessons failed: %s", exc)
        return []


def _load_market_state() -> str:
    try:
        from sheets import get_setting
        return get_setting("MARKET_STATE", "SIDEWAYS").upper().strip()
    except Exception:
        return "SIDEWAYS"


def _load_loss_streak() -> int:
    try:
        from sheets import read_tab
        rows    = read_tab("financials")
        kpi_map = {r.get("kpi_name", ""): r.get("current_value", "") for r in rows}
        return int(float(kpi_map.get("Current_Loss_Streak", 0) or 0))
    except Exception:
        return 0


# =============================================================================
# FIX 2 + FIX 3: Skip helpers — checked before every Claude call
# =============================================================================

def _already_reviewed_today(symbol: str) -> bool:
    """
    Returns True if market_log already has ANY row for this symbol
    with today's date (regardless of action or outcome).
    Prevents re-analyzing the same stock every 6 minutes.
    """
    try:
        from sheets import run_raw_sql
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        rows = run_raw_sql(
            """
            SELECT id FROM market_log
            WHERE symbol = %s
              AND date = %s
              AND action IN ('BUY', 'WAIT', 'AVOID')
            LIMIT 1
            """,
            (symbol.upper(), today),
        )
        return len(rows) > 0
    except Exception as exc:
        logger.warning("_already_reviewed_today(%s) failed: %s", symbol, exc)
        return False  # fail open — let Claude analyze if check fails


def _has_open_buy(symbol: str) -> bool:
    """
    Returns True if market_log already has a BUY row for this symbol
    with outcome = PENDING (active position being tracked).
    """
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT id FROM market_log
            WHERE symbol = %s
              AND action = 'BUY'
              AND outcome = 'PENDING'
            LIMIT 1
            """,
            (symbol.upper(),),
        )
        return len(rows) > 0
    except Exception as exc:
        logger.warning("_has_open_buy(%s) failed: %s", symbol, exc)
        return False  # fail open


# =============================================================================
# SECTION 2 - FUNDAMENTAL CONTEXT
# =============================================================================

def _load_fundamentals_context(symbol: str, sector: str) -> dict:
    result = {"found": False}
    try:
        from sheets import run_raw_sql

        rows = run_raw_sql(
            """
            SELECT symbol, fiscal_year, quarter,
                   npl, capital_fund_to_rwa, cd_ratio,
                   roa, roe, eps, dps,
                   pe_ratio, peg_value, net_profit,
                   prev_quarter_profit, growth_rate,
                   interest_spread, cost_of_fund, base_rate,
                   net_worth, promoter_shares, net_interest_income
            FROM fundamentals
            WHERE symbol = %s
            ORDER BY fiscal_year DESC, quarter DESC
            LIMIT 2
            """,
            (symbol.upper(),)
        )
        if not rows:
            return result

        latest = dict(rows[0])
        prev   = dict(rows[1]) if len(rows) > 1 else {}

        beta_rows = run_raw_sql(
            "SELECT beta, market_corr_p, n_months FROM fundamental_beta WHERE symbol = %s",
            (symbol.upper(),)
        )
        beta = None
        beta_source = "not_available"
        if beta_rows:
            b_val = float(beta_rows[0].get("beta", 0))
            b_p   = float(beta_rows[0].get("market_corr_p", 1))
            b_n   = int(beta_rows[0].get("n_months", 0))
            if b_p < 0.05 and b_n >= 12 and -2.0 <= b_val <= 3.0:
                beta        = round(b_val, 3)
                beta_source = f"empirical ({b_n}m, p={b_p:.3f})"
            else:
                beta_source = f"empirical_rejected (p={b_p:.3f}, n={b_n}m)"

        trend_notes = []
        sector_key  = sector.lower()

        def _f(d, key):
            try:
                v = d.get(key)
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        if any(s in sector_key for s in ["bank", "finance", "microfinance"]):
            npl_new = _f(latest, "npl")
            npl_old = _f(prev,   "npl")
            if npl_new is not None:
                if npl_new > 5.0:
                    trend_notes.append(f"NPL={npl_new:.1f}% above 5% (elevated credit risk)")
                elif npl_new < 2.0:
                    trend_notes.append(f"NPL={npl_new:.1f}% very clean loan book")
                if npl_old is not None:
                    if npl_new > npl_old:
                        trend_notes.append(f"NPL rising: {npl_old:.1f}% -> {npl_new:.1f}%")
                    elif npl_new < npl_old:
                        trend_notes.append(f"NPL improving: {npl_old:.1f}% -> {npl_new:.1f}%")
            cfrwa = _f(latest, "capital_fund_to_rwa")
            if cfrwa is not None:
                if cfrwa < 11.0:
                    trend_notes.append(f"capital_fund_to_rwa={cfrwa:.1f}% below NRB minimum 11%")
                elif cfrwa > 13.0:
                    trend_notes.append(f"capital_fund_to_rwa={cfrwa:.1f}% well capitalised")
            spread = _f(latest, "interest_spread")
            if spread is not None:
                if spread > 4.5:
                    trend_notes.append(f"interest_spread={spread:.2f}% healthy margin")
                elif spread < 3.0:
                    trend_notes.append(f"interest_spread={spread:.2f}% compressed margin")

        if "hydro" in sector_key:
            roa = _f(latest, "roa")
            roe = _f(latest, "roe")
            dps = _f(latest, "dps")
            if roa is not None and roa > 5.0:
                trend_notes.append(f"ROA={roa:.1f}% strong (Hydro signal rho=+0.19)")
            if roe is not None and roe > 10.0:
                trend_notes.append(f"ROE={roe:.1f}% good equity return")
            if dps is not None and dps > 0:
                trend_notes.append(f"DPS={dps:.0f} dividend paying (Hydro rho=+0.14)")

        gr = _f(latest, "growth_rate")
        if gr is not None:
            if gr < -10:
                trend_notes.append(f"growth_rate={gr:.1f}% significant earnings decline")
            elif gr > 20:
                trend_notes.append(f"growth_rate={gr:.1f}% strong earnings growth")

        pe = _f(latest, "pe_ratio")
        if pe is not None and pe > 0:
            if pe > 40:
                trend_notes.append(f"PE={pe:.0f}x premium valuation")
            elif pe < 10:
                trend_notes.append(f"PE={pe:.0f}x low valuation, potential value opportunity")

        prom = _f(latest, "promoter_shares")
        if prom is not None and prom > 70:
            trend_notes.append(
                f"promoter_shares={prom:.0f}% high concentration (validated neg signal rho=-0.12)"
            )

        return {
            "found":       True,
            "period":      f"{latest.get('fiscal_year','?')} Q{latest.get('quarter','?')}",
            "latest":      latest,
            "beta":        beta,
            "beta_source": beta_source,
            "trend_notes": trend_notes,
        }

    except Exception as exc:
        logger.warning("_load_fundamentals_context(%s) failed: %s", symbol, exc)
        return {"found": False}


def _format_fundamental_section(fund_ctx: dict) -> str:
    if not fund_ctx or not fund_ctx.get("found"):
        return ""

    latest = fund_ctx.get("latest", {})
    beta   = fund_ctx.get("beta")
    notes  = fund_ctx.get("trend_notes", [])
    period = fund_ctx.get("period", "?")

    def _fmt(key, fmt=".2f", suffix=""):
        val = latest.get(key)
        try:
            return f"{float(val):{fmt}}{suffix}" if val is not None else "N/A"
        except (TypeError, ValueError):
            return "N/A"

    beta_str = (
        f"{beta:.3f} ({fund_ctx.get('beta_source', '')})"
        if beta is not None
        else f"N/A ({fund_ctx.get('beta_source', '')})"
    )

    lines = [
        "",
        "=" * 47,
        f"FUNDAMENTAL DATA  ({period})",
        "=" * 47,
        f"Beta vs NEPSE:     {beta_str}",
        f"EPS:               {_fmt('eps')}",
        f"DPS:               {_fmt('dps')}",
        f"PE Ratio:          {_fmt('pe_ratio')}",
        f"ROA:               {_fmt('roa', '.2f', '%')}",
        f"ROE:               {_fmt('roe', '.2f', '%')}",
        f"NPL:               {_fmt('npl', '.2f', '%')}",
        f"Capital/RWA:       {_fmt('capital_fund_to_rwa', '.2f', '%')}",
        f"CD Ratio:          {_fmt('cd_ratio', '.2f', '%')}",
        f"Interest Spread:   {_fmt('interest_spread', '.2f', '%')}",
        f"Growth Rate:       {_fmt('growth_rate', '.1f', '%')}",
        f"Net Profit:        NPR {_fmt('net_profit', ',.0f')}",
        f"Promoter Shares:   {_fmt('promoter_shares', '.1f', '%')}",
    ]
    if notes:
        lines.append("")
        lines.append("KEY FUNDAMENTAL SIGNALS (research-validated, lag=1Q):")
        for note in notes:
            lines.append(f"  {note}")
    lines.append("=" * 47)
    lines.append(
        "NOTE: Fundamentals are supporting context only. "
        "Primary signal must still be technical (MACD/BB/SMA). "
        "Data is 1 quarter lagged."
    )
    return "\n".join(lines)


# =============================================================================
# SECTION 3 - HELPER CONTEXTS
# =============================================================================

def _trading_day_context() -> str:
    day = datetime.now(tz=NST).weekday()
    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                 3: "Thursday", 4: "Friday", 6: "Sunday"}
    today = day_names.get(day, "Unknown")
    if day in (6, 0):
        return f"TODAY IS {today} -- best entry days (Sun/Mon effect). Favor BUY."
    elif day in (2, 3):
        return f"TODAY IS {today} -- mid-week. Consider exits. New entries need strong signal."
    return f"TODAY IS {today} -- neutral day."


def _herding_context(flag, market_state: str) -> str:
    warnings = []
    rsi  = float(flag.rsi_14 or 0)
    conf = float(getattr(flag, "composite_score", 0) or 0)
    geo  = int(getattr(flag, "geo_combined", 0) or 0)
    if rsi > 72 and market_state in ("FULL_BULL", "CAUTIOUS_BULL"):
        warnings.append(
            f"RSI={rsi:.1f} in {market_state} -- herding bubble risk. "
            "Retail herd may be chasing. Wait for pullback."
        )
    if rsi < 25 and market_state == "BEAR":
        warnings.append(
            f"RSI={rsi:.1f} in BEAR -- possible capitulation. "
            "Herding-driven oversell. Watch for reversal signal."
        )
    if conf > 90:
        warnings.append("Very high composite score -- most alpha may already be priced in.")
    if geo >= 4 and market_state == "FULL_BULL":
        warnings.append(
            "High geo score in FULL_BULL -- market euphoria. "
            "Herding amplifies reversals. Tighten stops."
        )
    if not warnings:
        return "No herding bubble or capitulation signals detected."
    return "HERDING ALERT: " + " | ".join(warnings)


def _calc_breakeven(entry_price: float, shares: int) -> float:
    if shares <= 0:
        return entry_price
    total_buy_cost      = entry_price * shares * (1 + (0.40 + 0.015) / 100) + 25
    breakeven_per_share = (total_buy_cost + entry_price * shares * (0.40 + 0.015) / 100 + 25) / shares
    return round(breakeven_per_share, 2)


# =============================================================================
# SECTION 4 - BUILD CLAUDE PROMPT
# =============================================================================

def _build_prompt(
    flag,
    portfolio:    dict,
    geo:          dict,
    macro:        dict,
    lessons:      list[str],
    market_state: str,
    loss_streak:  int,
    fund_ctx:     dict = None,
) -> str:
    nst_now = datetime.now(tz=NST)

    lessons_str  = "\n".join(f"  - {l}" for l in lessons) if lessons else "  No lessons yet."
    holdings_str = "\n".join(
        f"  {h['symbol']}: {h['shares']} shares @ WACC {h['wacc']} | "
        f"P&L {h['pnl_pct']}% (NPR {h['pnl_npr']})"
        for h in portfolio.get("holdings", [])
    ) or "  No open positions."

    day_context      = _trading_day_context()
    herding_alert    = _herding_context(flag, market_state)
    candle_str       = f"{flag.best_candle} (Tier {flag.candle_tier})" if flag.best_candle else "None"
    hold_days        = getattr(flag, "suggested_hold", 17)
    fund_section_str = _format_fundamental_section(fund_ctx) if fund_ctx else ""

    return f"""You are a senior NEPSE quantitative analyst with deep knowledge of Nepal market research.
Analyze this specific stock and produce a precise trading recommendation.

TODAY: {nst_now.strftime('%Y-%m-%d %H:%M')} NST
MARKET STATE: {market_state}
LOSS STREAK: {loss_streak} consecutive losses (circuit breaker at 8)

==============================================
STOCK UNDER ANALYSIS
==============================================
Symbol:          {flag.symbol}
Sector:          {flag.sector}
LTP:             NPR {flag.ltp:.2f}
Daily Change:    {getattr(flag, 'change_pct', 0):+.2f}%
Urgency:         {flag.urgency}  (from Gemini screener)
Gemini Reason:   {flag.gemini_reason}
Gemini Risk:     {flag.gemini_risk}

TECHNICAL INDICATORS (frozen at 10:30 AM NST):
  RSI 14:          {flag.rsi_14:.1f}  [{getattr(flag, 'rsi_signal', '')}]
  MACD Cross:      {flag.macd_cross}
  BB Signal:       {flag.bb_signal}
  OBV Trend:       {getattr(flag, 'obv_trend', '?')}
  EMA Trend:       {getattr(flag, 'ema_trend', '?')}
  Tech Score:      {flag.tech_score}/100
  Candle Pattern:  {candle_str}
  C* Signal:       {'YES -- excess return above C*=0.129 (SIM paper)' if getattr(flag, 'cstar_signal', False) else 'NO'}
  Fundamental Adj: {getattr(flag, 'fundamental_adj', 0.0):+.2f} pts [{getattr(flag, 'fundamental_reason', 'n/a')}]

Primary Signal:  {flag.primary_signal}
Composite Score: {flag.composite_score:.1f}
Suggested Hold:  ~{hold_days} days (research-based)
Geo Score:       {flag.geo_combined:+d}/10

PRICE LEVELS (20-day range):
  Support:         NPR {flag.support_level:,.2f}
  Resistance:      NPR {flag.resistance_level:,.2f}
  LTP vs Support:  {((flag.ltp - flag.support_level) / flag.support_level * 100) if flag.support_level else 0:+.1f}%
  LTP vs Resist:   {((flag.resistance_level - flag.ltp) / flag.ltp * 100) if flag.resistance_level else 0:+.1f}%
{fund_section_str}
==============================================
MARKET CONTEXT
==============================================
Geo Score:       {geo.get('geo_score', 0):+d}/5  ({geo.get('geo_status', '?')})
Nepal Score:     {geo.get('nepal_score', 0):+d}/5  ({geo.get('nepal_status', '?')})
Combined:        {geo.get('combined', 0):+d}/10
Bandh Today:     {geo.get('bandh', 'NO')}
IPO Drain:       {geo.get('ipo_drain', 'NO')}
Key Geo Event:   {geo.get('key_geo_event', 'None')}
Key Nepal Event: {geo.get('key_nepal_event', 'None')}

MACRO (NRB {macro.get('period', '?')} -- updated monthly):
  Policy Rate:     {macro.get('policy_rate', '?')}%
  NRB Decision:    {macro.get('nrb_rate_decision', '?')}
  Inflation:       {macro.get('inflation_pct', '?')}%
  Remittance YoY:  {macro.get('remittance_yoy_pct', '?')}%
  Forex Reserve:   {macro.get('forex_reserve_months', '?')} months
  Lending Rate:    {macro.get('lending_rate', '?')}%
  FD Rate (1yr):   {macro.get('fd_rate', '?')}%  [{macro.get('fd_signal', 'NEUTRAL')}]

==============================================
YOUR PORTFOLIO
==============================================
Total Capital:   NPR {portfolio.get('total_capital_npr', 0):,.0f}
Liquid Cash:     NPR {portfolio.get('liquid_npr', 0):,.0f}
Open Positions:  {portfolio.get('open_positions', 0)}/3
Slots Left:      {portfolio.get('slots_remaining', 0)}

Holdings:
{holdings_str}

==============================================
SITUATIONAL CONTEXT
==============================================
Trading Day:     {day_context}
Herding Check:   {herding_alert}

==============================================
LEARNING HUB LESSONS (most relevant first)
==============================================
{lessons_str}

==============================================
NEPAL FEE MATH (use for all price calculations)
==============================================
  Buy cost:    trade_value x 1.00415 + NPR 25
  Sell cost:   trade_value x 1.00415 + NPR 25
  Breakeven:   entry x (1 + 0.0083) + NPR 50/shares
  CGT:         5% on net profit only (individuals)
  Max position: 10% of total capital = NPR {portfolio.get('total_capital_npr', 0) * 0.10:,.0f}
  Max positions: 3 simultaneous

==============================================
TASK
==============================================
Produce a precise BUY / WAIT / AVOID recommendation.
- BUY: only if primary signal is MACD/BB/SMA (RSI alone is never enough)
- Stop loss: always 3% below entry (hard rule)
- Target: use resistance level as reference, must exceed breakeven by >1%
- Hold: use suggested hold from research ({hold_days} days for {flag.primary_signal})
- Use max 10% of total capital per position
- For WAIT/AVOID: give specific conditions that would change your answer
- Include only ordinary shares, exclude mutual funds, debentures, promoter shares
- Consider fundamental signals as supporting context, not primary trigger

Respond ONLY with this JSON -- no markdown, no explanation outside JSON:
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
  "reasoning": "5-6 sentences covering: why this signal, what risks, what the Learning Hub says, sector context, and any fundamental quality flags",
  "lesson_applied": "which lesson from Learning Hub was most relevant, or NONE",
  "wait_condition": "if WAIT/AVOID: what specific condition would make this a BUY",
  "herding_note": "one sentence on herding/bubble risk or NONE"
}}"""


# =============================================================================
# SECTION 5 - CALL CLAUDE
# =============================================================================

def _call_claude(prompt: str) -> Optional[dict]:
    import urllib.request
    try:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set in environment")
            return None

        payload = json.dumps({
            "model": "anthropic/claude-sonnet-4-6",
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior NEPSE quantitative analyst. "
                        "You respond ONLY in valid JSON. No markdown fences. "
                        "You never recommend a trade without a clear stop loss."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }).encode()

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())

        raw = data["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else parts[0]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        logger.info(
            "Claude: %s %s | conf=%s",
            result.get("action", "?"),
            result.get("primary_signal", "?"),
            result.get("confidence", "?"),
        )
        return result
    except json.JSONDecodeError as exc:
        logger.warning("Claude returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return None


# =============================================================================
# SECTION 6 - ASSEMBLE RESULT + WRITE TO DB
# =============================================================================

def _assemble_result(claude_json: dict, flag, geo: dict) -> AnalystResult:
    return AnalystResult(
        symbol             = flag.symbol,
        action             = claude_json.get("action",           "WAIT"),
        confidence         = int(claude_json.get("confidence",   0)),
        entry_price        = float(claude_json.get("entry_price",0)),
        stop_loss          = float(claude_json.get("stop_loss",  0)),
        target             = float(claude_json.get("target",     0)),
        allocation_npr     = float(claude_json.get("allocation_npr", 0)),
        shares             = int(claude_json.get("shares",       0)),
        breakeven          = float(claude_json.get("breakeven",  0)),
        risk_reward        = float(claude_json.get("risk_reward",0)),
        suggested_hold     = int(claude_json.get("suggested_hold_days", 17)),
        reasoning          = claude_json.get("reasoning",        ""),
        lesson_applied     = claude_json.get("lesson_applied",   "NONE"),
        primary_signal     = claude_json.get("primary_signal",   ""),
        sector             = flag.sector,
        geo_score          = geo.get("combined", 0),
        rsi_14             = float(flag.rsi_14 or 0),
        candle_pattern     = flag.best_candle or "",
        urgency            = flag.urgency,
        gemini_reason      = flag.gemini_reason,
        support_level      = float(flag.support_level    or 0),
        resistance_level   = float(flag.resistance_level or 0),
        # FIX 1: headlines now carried from geo context (loaded from nepal_pulse)
        headlines_politics = geo.get("headlines_politics", ""),
        headlines_economy  = geo.get("headlines_economy",  ""),
        headlines_stock    = geo.get("headlines_stock",    ""),
    )


def _load_fundamentals_for_log(symbol: str) -> dict:
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT pe_ratio, eps, roe, npl
            FROM fundamentals
            WHERE symbol = %s
            ORDER BY fiscal_year DESC, quarter DESC
            LIMIT 1
            """,
            (symbol.upper(),),
        )
        return dict(rows[0]) if rows else {}
    except Exception as exc:
        logger.debug("_load_fundamentals_for_log(%s) failed: %s", symbol, exc)
        return {}


def _write_to_db(result: AnalystResult, flag=None) -> None:
    """
    Update the existing GEMINI_FLAG row with Claude's analysis.
    One row per signal -- Gemini wrote it, Claude completes it.
    Falls back to insert if market_log_id is not available.
    """
    def _s(val) -> str:
        return "" if val is None else str(val)

    try:
        from sheets import write_row, update_row

        today = datetime.now(tz=NST).strftime("%Y-%m-%d")
        now   = datetime.now(tz=NST).strftime("%H:%M:%S")

        fund = _load_fundamentals_for_log(result.symbol)

        if flag:
            rsi_14          = _s(flag.rsi_14)
            macd_cross      = _s(flag.macd_cross)
            macd_histogram  = _s(flag.macd_histogram)
            ema_trend       = _s(flag.ema_trend)
            ema_20_50_cross = _s(flag.ema_20_50_cross)
            ema_50_200_cross= _s(flag.ema_50_200_cross)
            bb_signal       = _s(flag.bb_signal)
            bb_pct_b        = _s(flag.bb_pct_b)
            bb_upper        = _s(flag.bb_upper) if float(getattr(flag, "bb_upper", 0) or 0) != 0 else ""
            bb_lower        = _s(flag.bb_lower) if float(getattr(flag, "bb_lower", 0) or 0) != 0 else ""
            obv_trend       = _s(flag.obv_trend)
            atr_pct         = _s(flag.atr_pct)
            tech_score      = _s(flag.tech_score)
            conf_score      = _s(flag.conf_score)
            candle_pat      = _s(flag.best_candle)
            support         = _s(flag.support_level)
            resistance      = _s(flag.resistance_level)
            volume          = _s(flag.volume) if flag.volume else ""
            change_pct      = _s(flag.change_pct) if flag.change_pct else "0"
            fund_adj        = _s(flag.fundamental_adj)
            geo_score       = _s(flag.geo_score)
            nepal_score     = _s(flag.nepal_score)
        else:
            rsi_14 = _s(result.rsi_14)
            macd_cross = macd_histogram = ema_trend = ""
            ema_20_50_cross = ema_50_200_cross = ""
            bb_signal = bb_pct_b = bb_upper = bb_lower = ""
            obv_trend = atr_pct = tech_score = conf_score = ""
            candle_pat = ""
            support    = _s(result.support_level)
            resistance = _s(result.resistance_level)
            volume = change_pct = fund_adj = ""
            geo_score = nepal_score = ""

        columns = {
            # Identity
            "date":              today,
            "time":              now,
            "symbol":            _s(result.symbol),
            "sector":            _s(result.sector),

            # Claude decision
            "action":            _s(result.action),
            "confidence":        _s(result.confidence),
            "entry_price":       _s(result.entry_price),
            "stop_loss":         _s(result.stop_loss),
            "target":            _s(result.target),
            "allocation_npr":    _s(result.allocation_npr),
            "shares":            _s(result.shares),
            "breakeven":         _s(result.breakeven),
            "risk_reward":       _s(result.risk_reward),
            "reasoning":         _s(result.reasoning),
            "outcome":           "PENDING",
            "timestamp":         _s(result.timestamp),

            # Technical indicators
            # Note: column names are legacy from original schema --
            # macd_line stores cross string, macd_signal stores histogram value
            "rsi_14":            rsi_14,
            "macd_line":         macd_cross,
            "macd_signal":       macd_histogram,
            "obv_trend":         obv_trend,
            "ema_200":           ema_trend,
            "atr_14":            atr_pct,
            "bollinger_upper":   bb_upper,
            "bollinger_lower":   bb_lower,
            "candle_pattern":    candle_pat,
            "conf_score":        conf_score,
            "support_level":     support,
            "resistance_level":  resistance,
            "volume":            volume,
            "volume_ratio":      change_pct,

            # Fundamentals
            "fundamental_score": fund_adj,
            "pe_ratio":          _s(fund.get("pe_ratio")),
            "eps":               _s(fund.get("eps")),
            "roe":               _s(fund.get("roe")),
            "npl_pct":           _s(fund.get("npl")),

            # Geo / macro scores
            "geo_score":         geo_score,
            "macro_score":       nepal_score,

            # FIX 1: headlines written from result (which got them from geo context)
            "headlines_politics": _s(result.headlines_politics),
            "headlines_economy":  _s(result.headlines_economy),
            "headlines_stock":    _s(result.headlines_stock),

            # Eval fields -- empty now, recommendation_tracker fills later
            "eval_date":             "",
            "eval_geo_score":        "",
            "eval_nepal_score":      "",
            "eval_nepse_index":      "",
            "eval_market_state":     "",
            "eval_policy_rate":      "",
            "eval_fd_rate_pct":      "",
            "eval_geo_delta":        "",
            "eval_nepal_delta":      "",
            "eval_key_news":         "",
            "eval_price_change_pct": "",
            "eval_nepse_change_pct": "",
            "eval_alpha":            "",
        }

        # UPDATE existing Gemini row -- one row per signal, not two
        market_log_id = getattr(flag, "market_log_id", None) if flag else None

        if market_log_id:
            update_row("market_log", columns, {"id": str(market_log_id)})
            logger.info(
                "Updated market_log id=%s: %s %s",
                market_log_id, result.action, result.symbol,
            )
        else:
            # Fallback insert -- no Gemini row to update
            write_row("market_log", columns)
            logger.info(
                "Inserted new market_log row: %s %s (no market_log_id)",
                result.action, result.symbol,
            )

    except Exception as exc:
        logger.error("_write_to_db failed for %s: %s", result.symbol, exc)


# =============================================================================
# SECTION 7 - MAIN ENTRY POINT
# =============================================================================

def run_analysis(flags: list) -> list[AnalystResult]:
    if not flags:
        logger.info("claude_analyst: no flags to analyze")
        return []

    logger.info("=" * 60)
    logger.info("claude_analyst.run_analysis() -- %d flags", len(flags))

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

    if portfolio.get("slots_remaining", 0) <= 0:
        logger.info("Portfolio full (3/3 positions open) -- no analysis needed")
        return []

    if loss_streak >= 8:
        logger.warning("Circuit breaker: loss_streak=%d -- skipping all analysis", loss_streak)
        return []

    results: list[AnalystResult] = []

    for flag in flags:
        sym = flag.symbol
        logger.info("-" * 40)
        logger.info("Analyzing %s [%s] urgency=%s", sym, flag.sector, flag.urgency)

        # FIX 3: skip if already reviewed today (any action)
        if _already_reviewed_today(sym):
            logger.info("%s: already reviewed today -- skipping", sym)
            continue

        # FIX 2: skip if open BUY still pending
        if _has_open_buy(sym):
            logger.info("%s: open BUY pending in market_log -- skipping", sym)
            continue

        buy_count = sum(1 for r in results if r.action == "BUY")
        if portfolio.get("open_positions", 0) + buy_count >= 3:
            logger.info("%s: portfolio full after earlier BUYs -- skipping", sym)
            continue

        lessons  = _load_lessons(sym, getattr(flag, "sector", ""))
        fund_ctx = _load_fundamentals_context(sym, getattr(flag, "sector", ""))

        logger.info(
            "Fundamentals for %s: found=%s | beta=%s | notes=%d",
            sym,
            fund_ctx.get("found", False),
            fund_ctx.get("beta", "N/A"),
            len(fund_ctx.get("trend_notes", [])),
        )

        prompt      = _build_prompt(
            flag, portfolio, geo, macro, lessons, market_state, loss_streak,
            fund_ctx=fund_ctx,
        )
        claude_json = _call_claude(prompt)

        if claude_json is None:
            logger.warning("%s: Claude returned no result -- skipping", sym)
            continue

        result = _assemble_result(claude_json, flag, geo)
        results.append(result)
        _write_to_db(result, flag=flag)
        logger.info("Result: %s", result.summary())

        # Stamp AVOID closed immediately -- no hold period needed
        if result.action == "AVOID":
            mid = getattr(flag, "market_log_id", None)
            if mid:
                try:
                    from analysis.recommendation_tracker import stamp_avoid_closed
                    stamp_avoid_closed(mid, result.symbol)
                except Exception as exc:
                    logger.warning("stamp_avoid_closed failed: %s", exc)

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


# =============================================================================
# SECTION 8 - FORMAT FOR NOTIFIER
# =============================================================================

def format_buy_signal(result: AnalystResult) -> str:
    lines = [
        f"BUY SIGNAL -- {result.symbol}",
        f"Sector:      {result.sector}",
        f"Confidence:  {result.confidence}%  |  Signal: {result.primary_signal}",
        f"Entry:       NPR {result.entry_price:,.0f}",
        f"Stop Loss:   NPR {result.stop_loss:,.0f}",
        f"Target:      NPR {result.target:,.0f}",
        f"Breakeven:   NPR {result.breakeven:,.0f}",
        f"R/R Ratio:   {result.risk_reward:.1f}x",
        f"Shares:      {result.shares}",
        f"Allocation:  NPR {result.allocation_npr:,.0f}",
        f"Hold:        ~{result.suggested_hold} days",
        f"",
        f"Reasoning: {result.reasoning[:200]}",
    ]
    if result.lesson_applied and result.lesson_applied != "NONE":
        lines.append(f"Lesson Applied: {result.lesson_applied[:100]}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [CLAUDE_ANALYST] %(levelname)s: %(message)s",
    )

    args         = sys.argv[1:]
    dry_run      = "--dry-run"      in args
    print_prompt = "--print-prompt" in args
    sym_args     = [a.upper() for a in args if not a.startswith("--")]

    print("\n" + "=" * 70)
    print("  NEPSE AI -- claude_analyst.py")
    print("=" * 70)

    if print_prompt:
        try:
            from gemini_filter import run_gemini_filter
            flags = run_gemini_filter()
            if not flags:
                print("  No flags from Gemini -- nothing to print")
                sys.exit(0)
        except Exception as e:
            print(f"  gemini_filter failed: {e}")
            sys.exit(1)

        if sym_args:
            flags = [f for f in flags if f.symbol in sym_args]

        portfolio    = _load_portfolio()
        geo          = _load_geo_context()
        macro        = _load_macro_context()
        market_state = _load_market_state()
        loss_streak  = _load_loss_streak()

        for i, flag in enumerate(flags, 1):
            lessons  = _load_lessons(flag.symbol, getattr(flag, "sector", ""))
            fund_ctx = _load_fundamentals_context(flag.symbol, getattr(flag, "sector", ""))
            prompt   = _build_prompt(
                flag, portfolio, geo, macro, lessons, market_state, loss_streak,
                fund_ctx=fund_ctx,
            )
            char_count = len(prompt)
            token_est  = char_count // 4
            print("=" * 70)
            print(f"  PROMPT {i}/{len(flags)} -- {flag.symbol} [{getattr(flag,'sector','')}]")
            print(f"  Chars: {char_count} | ~Tokens: {token_est} | ~Cost: ${token_est * 0.000003:.4f}")
            print("=" * 70)
            print(prompt)
            print()
        sys.exit(0)

    print("\n[1/2] Running gemini_filter...")
    try:
        from gemini_filter import run_gemini_filter
        flags = run_gemini_filter()
        if not flags:
            print("  No flags from Gemini -- nothing to analyze")
            sys.exit(0)
        print(f"  {len(flags)} flag(s) to analyze\n")
    except Exception as e:
        print(f"  gemini_filter failed: {e}")
        sys.exit(1)

    if sym_args:
        flags = [f for f in flags if f.symbol in sym_args]

    print("[2/2] Running Claude analysis...")
    results = run_analysis(flags)
    try:
        from helper.notifier import send_buy_signal, send_wait_signal

        for r in results:
            if r.action == "BUY":
                send_buy_signal(r)
            elif r.action == "WAIT":
                send_wait_signal(r)
        logger.info('Telegram message sent')

    except Exception as e:
        from helper.notifier import send_error_alert
        send_error_alert('claude_analyst', str(e))
    
    print(f"\n  {len(results)} result(s):")
    for r in results:
        print(f"  {r.summary()}")
    print()