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
from AI import ask_claude
from config import NST
from modules.atrad_scraper import fetch_order_book

logger = logging.getLogger(__name__)

# ── Political event context (optional — fails gracefully if tables not ready) ─
try:
    import modules.event_detector as _event_detector_ca
    _CA_ED_AVAILABLE = True
except Exception:
    _event_detector_ca = None  # type: ignore[assignment]
    _CA_ED_AVAILABLE = False

# Module-level buy-block state set once per run_analysis() call
_BUY_BLOCKED:     bool = False
_BUY_BLOCK_SCORE: int  = 0


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
    wait_condition:     str   = ""
    herding_note:       str   = ""
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
        max_positions = int(get_setting("MAX_POSITIONS", "3"))
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
            "slots_remaining":   max(0, max_positions - len(open_rows)),
            "max_positions":     max_positions,
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
            "max_positions":     3,
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

        _nepal_score_int = int(pulse.get("nepal_score", 0) or 0)
        if _nepal_score_int <= -3:
            _nepal_status = "BEARISH"
        elif _nepal_score_int >= 3:
            _nepal_status = "BULLISH"
        else:
            _nepal_status = "NEUTRAL"

        return {
            "geo_score":          int(geo.get("geo_score",    0) or 0),
            "nepal_score":        _nepal_score_int,
            "combined":           int(geo.get("geo_score", 0) or 0) + _nepal_score_int,
            "geo_status":         geo.get("geo_status",      "NEUTRAL"),
            "nepal_status":       _nepal_status,
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
        def _nrb(key, fallback="N/A"):
            v = row.get(key)
            return v if (v is not None and v != "") else fallback

        return {
            "policy_rate":          _nrb("policy_rate") or _nrb("policy_rate_pct") or _nrb("bank_rate"),
            "nrb_rate_decision":    nrb_decision,
            "inflation_pct":        _nrb("cpi_inflation"),
            "remittance_yoy_pct":   _nrb("remittance_yoy_change_pct"),
            "forex_reserve_months": _nrb("fx_reserve_months"),
            "lending_rate":         _nrb("lending_rate_pct") or _nrb("lending_rate") or _nrb("bank_rate") or _nrb("base_rate"),
            "period":               _nrb("period"),
            "fd_rate":              get_setting("FD_RATE_PCT",     "8.5"),
            "fd_signal":            get_setting("FD_SCORE_SIGNAL", "NEUTRAL"),
            "deposit_rate_pct":                     _nrb("deposit_rate_pct"),
            "interbank_rate_pct":                   _nrb("interbank_rate_pct"),
            "tbill_91d_rate_pct":                   _nrb("tbill_91d_rate_pct"),
            "npl_ratio_pct":                        _nrb("npl_ratio_pct"),
            "m2_growth_yoy_pct":                    _nrb("m2_growth_yoy_pct"),
            "private_sector_credit_growth_yoy_pct": _nrb("private_sector_credit_growth_yoy_pct"),
            "bop_impact_on_nepse":                  _nrb("bop_impact_on_nepse"),
            "usd_npr_rate":                         _nrb("usd_npr_rate"),
            "remittance_total_billion_npr":         _nrb("remittance_total_billion_npr"),
        }
    except Exception as exc:
        logger.warning("_load_macro_context failed: %s", exc)
        return {
            "policy_rate": "", "nrb_rate_decision": "", "inflation_pct": "",
            "remittance_yoy_pct": "", "forex_reserve_months": "", "lending_rate": "",
            "period": "", "fd_rate": "", "fd_signal": "",
            "deposit_rate_pct": "", "interbank_rate_pct": "", "tbill_91d_rate_pct": "",
            "npl_ratio_pct": "", "m2_growth_yoy_pct": "", "private_sector_credit_growth_yoy_pct": "",
            "bop_impact_on_nepse": "", "usd_npr_rate": "", "remittance_total_billion_npr": "",
        }


def _load_lessons(symbol: str, sector: str, limit: int = 6) -> list[str]:
    try:
        from sheets import run_raw_sql, get_setting

        # Load weights from settings — used only for fallback when source_weight is NULL
        w_council = float(get_setting("LESSON_WEIGHT_COUNCIL",    "1.5"))
        w_gpt     = float(get_setting("LESSON_WEIGHT_GPT_WEEKLY", "1.0"))
        w_seeder  = float(get_setting("LESSON_WEIGHT_SEEDER",     "0.8"))
        _source_defaults = {
            "monthly_council": w_council,
            "gpt_weekly":      w_gpt,
            "learning_seeder": w_seeder,
        }
        _conf_mult = {"HIGH": 1.2, "MEDIUM": 1.0, "LOW": 0.8}

        rows = run_raw_sql(
            """
            SELECT symbol, sector, lesson_type, condition, finding, action,
                   confidence_level, win_rate, trade_count, source, source_weight
            FROM learning_hub
            WHERE active = 'true'
              AND (symbol = %s OR sector = %s OR symbol = 'MARKET' OR applies_to = 'ALL')
              AND (consumer = 'ALL' OR consumer = 'claude_only')
            """,
            (symbol.upper(), sector.lower()),
        )

        def _eff_weight(r: dict) -> float:
            # Prefer stored source_weight; fall back to settings default for that source
            sw_raw = r.get("source_weight")
            if sw_raw is not None and sw_raw != "":
                sw = float(sw_raw)
            else:
                sw = _source_defaults.get(r.get("source", ""), 1.0)
            cm = _conf_mult.get(r.get("confidence_level", "LOW"), 1.0)
            return sw * cm

        sorted_rows = sorted(rows or [], key=_eff_weight, reverse=True)[:limit]

        lessons = []
        for r in sorted_rows:
            sym   = r.get("symbol", "?")
            ltype = r.get("lesson_type", "")
            cond  = r.get("condition",   "")
            find  = r.get("finding",     "")
            act   = r.get("action",      "")
            conf  = r.get("confidence_level", "LOW")
            wr    = r.get("win_rate",    "")
            n     = r.get("trade_count", "")
            stat  = f" (win_rate={wr}, n={n})" if n else ""
            lessons.append(f"[{sym}|{ltype}|{conf}] IF {cond} -> {act}: {find}{stat}")
        return lessons
    except Exception as exc:
        logger.warning("_load_lessons failed: %s", exc)
        return []


def _load_monthly_override() -> None:
    """
    Read monthly_override for the current month, set module-level _BUY_BLOCKED flag,
    and stamp last_read_at.  Always runs — DB reads are allowed in all modes.
    """
    global _BUY_BLOCKED, _BUY_BLOCK_SCORE
    from sheets import run_raw_sql, upsert_row

    now           = datetime.now(tz=NST)
    current_month = now.strftime("%Y-%m")

    try:
        rows = run_raw_sql(
            "SELECT buy_blocked, confidence_score, stop_trigger "
            "FROM monthly_override WHERE run_month = %s",
            (current_month,),
        )
        if rows:
            row              = rows[0]
            _BUY_BLOCKED     = (row.get("buy_blocked", "false") == "true")
            _BUY_BLOCK_SCORE = int(row.get("confidence_score") or 0)
            if _BUY_BLOCKED:
                logger.warning(
                    "[monthly_override] BUY signals BLOCKED — "
                    "council confidence=%d. All BUY → WAIT.",
                    _BUY_BLOCK_SCORE,
                )
            # Stamp last_read_at — DB write always allowed
            try:
                upsert_row(
                    "monthly_override",
                    {
                        "run_month":   current_month,
                        "last_read_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    conflict_columns=["run_month"],
                )
            except Exception as e:
                logger.warning("[monthly_override] last_read_at stamp failed: %s", e)
        else:
            _BUY_BLOCKED     = False
            _BUY_BLOCK_SCORE = 0
    except Exception as exc:
        logger.warning("_load_monthly_override failed: %s", exc)
        _BUY_BLOCKED     = False
        _BUY_BLOCK_SCORE = 0


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
        return int(float(kpi_map.get("current_loss_streak", 0) or 0))
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


def _load_broker_flow_context(symbol: str) -> dict:
    """
    Query broker_flow and broker_holdings for symbol on today's date.
    Returns dict with 'flow' and 'holdings' keys, or empty dict on any failure.
    Never raises.
    """
    try:
        from sheets import run_raw_sql
        today = datetime.now(tz=NST).strftime("%Y-%m-%d")

        flow_rows = run_raw_sql(
            "SELECT * FROM broker_flow WHERE symbol = %s AND date = %s LIMIT 1",
            (symbol.upper(), today),
        )
        holdings_rows = run_raw_sql(
            "SELECT * FROM broker_holdings WHERE symbol = %s AND date = %s LIMIT 1",
            (symbol.upper(), today),
        )

        flow     = dict(flow_rows[0])     if flow_rows     else {}
        holdings = dict(holdings_rows[0]) if holdings_rows else {}

        if not flow and not holdings:
            return {}
        return {"flow": flow, "holdings": holdings}
    except Exception as exc:
        logger.debug("_load_broker_flow_context(%s) failed: %s", symbol, exc)
        return {}


def _format_broker_flow_section(broker_ctx: dict) -> str:
    """
    Format broker flow data for Claude prompt.
    Returns empty string if no data — never shows noise when table is empty.
    """
    if not broker_ctx:
        return ""

    flow     = broker_ctx.get("flow", {})
    holdings = broker_ctx.get("holdings", {})

    if not flow and not holdings:
        return ""

    def _cr(amount: float) -> str:
        if amount >= 1_00_00_000:
            return f"{amount / 1_00_00_000:.1f}Cr"
        elif amount >= 1_00_000:
            return f"{amount / 1_00_000:.1f}L"
        return f"{amount:,.0f}"

    def _fv(row: dict, key: str) -> float:
        try:
            v = row.get(key)
            return float(v) if v not in (None, "", "None") else 0.0
        except (TypeError, ValueError):
            return 0.0

    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    sep   = "═" * 47
    lines = [sep, f"BROKER FLOW — {today}", sep]

    if flow:
        acc_count  = int(_fv(flow, "acc_broker_count_1d"))
        dist_count = int(_fv(flow, "dist_broker_count_1d"))
        acc_amt    = _fv(flow, "acc_amount_1d")
        dist_amt   = _fv(flow, "dist_amount_1d")
        net_amt    = acc_amt - dist_amt
        net_sign   = "+" if net_amt >= 0 else "-"

        acc_qty    = _fv(flow, "acc_qty_1d")
        dist_qty   = _fv(flow, "dist_qty_1d")
        acc_score  = (acc_amt * acc_qty)  / max(acc_count, 1)
        dist_score = (dist_amt * dist_qty) / max(dist_count, 1)

        if acc_score > dist_score:
            flow_class = "NET_ACCUMULATION"
        elif dist_score > acc_score:
            flow_class = "NET_DISTRIBUTION"
        else:
            flow_class = "NEUTRAL"

        lines.append(f"Daily Flow:    {flow_class}")
        lines.append(f"  Buying:      {acc_count} brokers | NPR {_cr(acc_amt)}")
        lines.append(f"  Selling:     {dist_count} brokers  | NPR {_cr(dist_amt)}")
        lines.append(f"  Net:         {net_sign}NPR {_cr(abs(net_amt))}")

        acc_1w_count = int(_fv(flow, "acc_broker_count_1w"))
        acc_1w_amt   = _fv(flow, "acc_amount_1w")
        if acc_1w_count > 0:
            lines.append(
                f"Weekly:        {acc_1w_count} brokers accumulated | NPR {_cr(acc_1w_amt)} this week"
            )

    if holdings:
        stealth    = _fv(holdings, "stealth_score")
        top3_pct   = _fv(holdings, "top3_holding_pct")
        public_pct = _fv(holdings, "public_trade_pct")

        lines.append("")
        lines.append(
            f"Stealth Score: {stealth:.1f} "
            f"(top 3 hold {top3_pct:.1f}% | only {public_pct:.1f}% public leakage)"
        )

        b1_name = holdings.get("top_broker_1_name", "")
        b1_pct  = _fv(holdings, "top_broker_1_pct")
        b2_name = holdings.get("top_broker_2_name", "")
        b2_pct  = _fv(holdings, "top_broker_2_pct")
        b3_name = holdings.get("top_broker_3_name", "")
        b3_pct  = _fv(holdings, "top_broker_3_pct")

        if b1_name:
            lines.append(f"  #1 {b1_name}: {b1_pct:.1f}%")
        if b2_name:
            lines.append(f"  #2 {b2_name}: {b2_pct:.1f}%")
        if b3_name:
            lines.append(f"  #3 {b3_name}: {b3_pct:.1f}%")

    lines.append(sep)
    lines.append(
        "NOTE: Broker flow is supporting context. Primary signal must be technical."
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

def _audit_prompt_fields(flag, geo: dict, macro: dict, portfolio: dict) -> None:
    """Print a pre-send field audit so missing values are caught before the LLM call."""
    fields = {
        "symbol":           flag.symbol,
        "sector":           flag.sector,
        "ltp":              flag.ltp,
        "rsi_14":           flag.rsi_14,
        "macd_cross":       flag.macd_cross,
        "bb_signal":        flag.bb_signal,
        "obv_trend":        getattr(flag, "obv_trend",       None),
        "ema_trend":        getattr(flag, "ema_trend",        None),
        "tech_score":       flag.tech_score,
        "composite_score":  flag.composite_score,
        "support_level":    flag.support_level,
        "resistance_level": flag.resistance_level,
        "vwap_dev":         getattr(flag, "vwap_dev",         None),
        "bid_ask_ratio":    getattr(flag, "bid_ask_ratio",    None),
        "dpr_proximity":    getattr(flag, "dpr_proximity",    None),
        "volume_os_ratio":  getattr(flag, "volume_os_ratio",  None),
        "geo_score":        geo.get("geo_score"),
        "nepal_score":      geo.get("nepal_score"),
        "combined":         geo.get("combined"),
        "policy_rate":      macro.get("policy_rate"),
        "lending_rate":     macro.get("lending_rate"),
        "fd_rate":          macro.get("fd_rate"),
        "inflation_pct":    macro.get("inflation_pct"),
        "total_capital":    portfolio.get("total_capital_npr"),
        "liquid_npr":       portfolio.get("liquid_npr"),
    }
    ZERO_OK = {"bid_ask_ratio", "dpr_proximity", "volume_os_ratio", "vwap_dev", "combined"}
    warnings = []
    for k, v in fields.items():
        str_v = str(v) if v is not None else ""
        if v is None or str_v in ("", "?", "None", "N/A"):
            warnings.append(f"  MISSING   {k} = {repr(v)}")
        elif isinstance(v, float) and v == 0.0 and k not in ZERO_OK:
            warnings.append(f"  ZERO?     {k} = {v}  (check if intentional)")
    if warnings:
        print(f"\n⚠️  PROMPT FIELD AUDIT — {flag.symbol}")
        print("\n".join(warnings))
    else:
        print(f"✅  All audit fields populated for {flag.symbol}")


def _build_prompt(
    flag,
    portfolio:    dict,
    geo:          dict,
    macro:        dict,
    lessons:      list[str],
    market_state: str,
    loss_streak:  int,
    fund_ctx:     dict = None,
    broker_ctx:   dict = None,
) -> str:
    nst_now = datetime.now(tz=NST)
    LESSON_ACTION_GLOSSARY = """LESSON ACTION CODES — apply these exactly when a lesson fires:
    BLOCK_ENTRY              → Hard block. Do not issue BUY regardless of other signals. Only valid with 25+ supporting trades.
    SOFT_BLOCK               → Strong negative signal. Weight heavily against entry but does NOT unilaterally reject — can be outweighed by sufficiently strong positive signals.
    ADD_TO_REASONING         → Factor this into your reasoning. Weight it alongside indicators. This is NEVER a hard block — even HIGH-confidence ADD_TO_REASONING can be outweighed by strong positive signals. Do not treat it as a veto.
    INCREASE_CONFIDENCE_BY_N → Add N points to your confidence score for this stock.
    REDUCE_CONFIDENCE_BY_N   → Subtract N points from your confidence score.
    INCREASE_ALLOCATION_BY_N → Note in reasoning: increase suggested allocation by N%.
    REDUCE_ALLOCATION_BY_N   → Note in reasoning: reduce suggested allocation by N%.
    A lesson fires when its IF condition matches the current stock's attributes.
    IMPORTANT: confidence_level (HIGH/MEDIUM/LOW) reflects evidence strength, not action strength.
    HIGH confidence ADD_TO_REASONING is a strong soft signal — it cannot veto a BUY by itself.
    Only BLOCK_ENTRY is a hard veto. All other action codes are soft inputs to your reasoning."""

    _engine_source = getattr(flag, "engine_source", "v1") or "v1"
    LESSON_APPLICABILITY_BY_ENGINE = f"""LESSON APPLICABILITY BY ENGINE SOURCE: This candidate's engine_source is {_engine_source}.
    If engine_source is "v2" (not "BOTH"), do NOT apply any active lesson whose condition is a
    tech_score threshold ("tech_score < N") when reasoning — such lessons' evidence was drawn from
    v1-scored candidates and tech_score is not v2's basis for selection. All other active lessons
    still apply normally regardless of engine_source. If co_flagged_by shows v1 explicitly disagreed
    with this candidate's direction, factor that into your reasoning as you would any other
    conflicting signal — but it does not reinstate the tech_score gate."""

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
    fund_section_str         = _format_fundamental_section(fund_ctx)   if fund_ctx   else ""
    broker_flow_section_str  = _format_broker_flow_section(broker_ctx) if broker_ctx else ""

    # Political event context block (empty string if nothing active)
    _event_ctx = ""
    if _CA_ED_AVAILABLE:
        try:
            _event_ctx = _event_detector_ca.build_claude_context() or ""
        except Exception:
            _event_ctx = ""
    event_ctx_section = (
        f"\n{_event_ctx}\n" if _event_ctx else ""
    )

    # Build momentum block for prompt
    _momentum_status  = str(getattr(flag, "momentum_status",  "NEUTRAL"))
    _rsi_slope_3d     = float(getattr(flag, "rsi_slope_3d",   0.0) or 0.0)
    _macd_hist_slope  = float(getattr(flag, "macd_hist_slope", 0.0) or 0.0)
    _bb_pct_b_slope   = float(getattr(flag, "bb_pct_b_slope",  0.0) or 0.0)
    _bounce_failed    = bool(getattr(flag, "bounce_failed",    False))
    _reversal_days    = int(getattr(flag, "reversal_days",     0)   or 0)
    _momentum_dir     = "↑ IMPROVING" if _rsi_slope_3d > 0 else "↓ DECLINING"
    momentum_block = (
        "\n═══════════════════════════════════════════════\n"
        "7-DAY MOMENTUM TREND (pre-computed by Python)\n"
        "═══════════════════════════════════════════════\n"
        f"Momentum Status:   {_momentum_status}\n"
        f"RSI Direction:     {_momentum_dir}  (avg slope={_rsi_slope_3d:+.2f}/day)\n"
        f"MACD Hist Slope:   {_macd_hist_slope:+.6f}  ({'improving' if _macd_hist_slope > 0 else 'worsening'})\n"
        f"BB %B Slope:       {_bb_pct_b_slope:+.4f}  ({'recovering' if _bb_pct_b_slope > 0 else 'pressing lower band'})\n"
        f"Bounce Failed:     {'YES — dead-cat trap detected' if _bounce_failed else 'No'}\n"
        f"Reversal Days:     {_reversal_days} consecutive days RSI improving (with noise tolerance)\n"
        "\nMOMENTUM RULES — apply strictly:\n"
        "  FALLING_KNIFE     → Strong signal to AVOID. Price still declining despite oversold reading.\n"
        "  OVERSOLD_WATCH    → WAIT only. RSI barely improving — no confirmation yet.\n"
        "  EARLY_REVERSAL    → WAIT is valid. BUY only if MACD cross also confirms.\n"
        "  CONFIRMED_REVERSAL → BUY candidate. Multi-day improvement with volume.\n"
        "  NEUTRAL           → Standard scoring applies. Momentum not a factor.\n"
        "  bounce_failed=YES → AVOID regardless of other indicators. Dead-cat trap.\n"
    )

    # ── Intraday breadth context ──────────────────────────────────────────────
    try:
        from sheets import get_intraday_breadth
        breadth_rows = get_intraday_breadth()
    except Exception:
        breadth_rows = []

    if breadth_rows:
        breadth_lines = []
        for r in breadth_rows:
            ts    = str(r.get("timestamp", ""))[11:16]
            adv   = r.get("advancing",     "?")
            dec   = r.get("declining",     "?")
            score = r.get("breadth_score", "?")
            sig   = r.get("market_signal", "?")
            breadth_lines.append(f"  {ts}  adv={adv:<4} dec={dec:<4} score={str(score):>7}  {sig}")
        breadth_section = (
            "INTRADAY BREADTH TIMELINE:\n"
            + "\n".join(breadth_lines)
            + f"\nGemini breadth classification: {getattr(flag, 'intraday_trend', 'UNKNOWN')}"
        )
    else:
        breadth_section = (
            f"INTRADAY BREADTH: no snapshots yet\n"
            f"Gemini breadth classification: {getattr(flag, 'intraday_trend', 'UNKNOWN')}"
        )

    # ── Sector momentum context ───────────────────────────────────────────────
    sector_momentum = getattr(flag, "sector_momentum", "") or ""
    sector_momentum_section = f"""==============================================
SECTOR MOMENTUM (context only — not a signal trigger)
==============================================
{sector_momentum if sector_momentum else "no sector data this cycle"}

Use this to inform conviction:
- Broad green sector (high pos, high vol_breadth, positive leader_gap) → supports entry
- Sector distributing or leader_gap flat → raise caution even if stock looks strong
- circuit_locked=YES on sector leader → capital trapped in leader, rotation likely
"""

    _hold_signal_str = f" for {flag.primary_signal}" if flag.primary_signal not in ("LAGGARD_PLAY", "") else ""
    
    
    # ── News catalyst block ───────────────────────────────────────────────────
    news_catalyst = getattr(flag, "news_catalyst", "") or ""
    catalyst_block = ""
    if news_catalyst:
        catalyst_block = (
            "\n═══════════════════════════════════════════════\n"
            "NEWS CATALYST (from today's Nepal news)\n"
            "═══════════════════════════════════════════════\n"
            f"{news_catalyst}\n"
            "NOTE: This is a policy/regulatory/financial event directly named in today's news\n"
            "for this symbol. Factor this into your conviction alongside technicals.\n"
            "═══════════════════════════════════════════════"
        )

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
{catalyst_block}

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
  VWAP Dev:        {getattr(flag, 'vwap_dev', 0.0):+.2f}%  (+ = above fair value, its live data)
  Bid/Ask Ratio:   {getattr(flag, 'bid_ask_ratio', 0.0):.2f}  (full depth order book; >0.5 = buy pressure)
  DPR Proximity:   {getattr(flag, 'dpr_proximity', 0.0):.2f}  (0=near low circuit, 1=near high circuit and it is live data)
  Volume/OS Ratio: {getattr(flag, 'volume_os_ratio', 0.0):.2f}%  (>1% = smart money signal, >3% = operator/institutional)

Primary Signal:  {flag.primary_signal}
Composite Score: {flag.composite_score:.1f}
Suggested Hold:  ~{hold_days} days (research-based)
Engine:          {getattr(flag, 'engine_source', 'v1')}{f" [{flag.co_flagged_by}]" if getattr(flag, 'co_flagged_by', '') else ""}

PRICE LEVELS (20-day range):
  Support:         NPR {flag.support_level:,.2f}
  Resistance:      NPR {flag.resistance_level:,.2f}
  LTP vs Support:  {((flag.ltp - flag.support_level) / flag.support_level * 100) if flag.support_level else 0:+.1f}%
  LTP vs Resist:   {((flag.resistance_level - flag.ltp) / flag.ltp * 100) if flag.resistance_level else 0:+.1f}%
{fund_section_str}
{broker_flow_section_str}
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

{breadth_section}

TODAY'S NEPAL HEADLINES (from nepal_pulse — untruncated):
  Politics: {geo.get('headlines_politics', 'None') or 'None'}
  Economy:  {geo.get('headlines_economy',  'None') or 'None'}
  Market:   {geo.get('headlines_stock',    'None') or 'None'}

MACRO (NRB {macro.get('period') or 'N/A'} -- updated monthly):
  Policy Rate:     {macro.get('policy_rate') or 'N/A'}%
  NRB Decision:    {macro.get('nrb_rate_decision') or 'N/A'}
  Inflation:       {macro.get('inflation_pct') or 'N/A'}%
  Remittance YoY:  {macro.get('remittance_yoy_pct') or 'N/A'}%
  Forex Reserve:   {macro.get('forex_reserve_months') or 'N/A'} months
  Lending Rate:    {macro.get('lending_rate') or 'N/A'}%
  FD Rate (1yr):   {macro.get('fd_rate') or 'N/A'}%  [{macro.get('fd_signal') or 'NEUTRAL'}]

{sector_momentum_section}
==============================================
YOUR PORTFOLIO
==============================================
Total Capital:   NPR {portfolio.get('total_capital_npr', 0):,.0f}
Liquid Cash:     NPR {portfolio.get('liquid_npr', 0):,.0f}
Open Positions:  {portfolio.get('open_positions', 0)}/{portfolio.get('max_positions', 3)}
Slots Left:      {portfolio.get('slots_remaining', 0)}

Holdings:
{holdings_str}

==============================================
SITUATIONAL CONTEXT
==============================================
Trading Day:     {day_context}
Herding Check:   {herding_alert}
{event_ctx_section}
{momentum_block}
==============================================
LEARNING HUB LESSONS (most relevant first)
==============================================
{LESSON_ACTION_GLOSSARY}

{LESSON_APPLICABILITY_BY_ENGINE}

{lessons_str}

==============================================
NEPAL FEE MATH (use for all price calculations)
==============================================
  Buy cost:    trade_value x 1.00415 + NPR 25
  Sell cost:   trade_value x 1.00415 + NPR 25
  Breakeven:   entry x (1 + 0.0083) + NPR 50/shares
  CGT:         5% on net profit only (individuals)
  Max position: 10% of total capital = NPR {portfolio.get('total_capital_npr', 0) * 0.10:,.0f}
  Max positions: {portfolio.get('max_positions', 3)} simultaneous

==============================================
HOW TO WEIGH THE EVIDENCE (research-based)
==============================================
- NEPSE alpha is regime-driven, not setup-driven. Weight sector momentum, intraday breadth and the
  Nepal score heavily: a clean single-stock setup inside a deteriorating regime is usually a WAIT.
- Highest-conviction technical setup is BB_LOWER_TOUCH + OBV rising -- treat that combination as a
  genuine edge.
- MACD is the weakest signal historically, worst in Development Banks and Non-Life Insurance. Do NOT
  issue a MACD-primary BUY unless it is corroborated by volume, OBV or broker-flow accumulation.
- C* and candle patterns are weak/unvalidated priors -- supporting context only, never a trigger.
- Do not add a position that would put more than 2 holdings in one sector (infer current exposure
  from Holdings above).
- Engine v1 = today's frozen snapshot indicators. Engine v2 = 6-day trend/slope-based (experimental,
  not yet win-rate validated). BOTH = both engines independently flagged this stock -- treat as mild
  corroboration, not proof.

==============================================
TASK
==============================================
Produce a precise BUY / WAIT / AVOID recommendation.
- Consider INTRADAY BREADTH TIMELINE: if breadth is FADING or DISTRIBUTING at time of analysis,
  raise your confidence threshold by 5 points before issuing BUY. If ACCUMULATING or RECOVERING,
  breadth supports the entry — note this in reasoning.
- BUY: use your best judgment across all available signals. State your primary signal clearly in the JSON.
- WAIT condition: max 2 conditions, stock-specific price/indicator only -- never require tech_score >X,
  confidence >X, breadth state, or nepal_score.
- Stop loss: anchor it just below the 20-day support level (structure-based). The stop distance should
  generally land in the 8-15% band that research supports -- if support sits tighter than ~8%, widen
  toward 8% so NEPSE intraday noise doesn't shake you out; if it sits wider than ~15%, the setup is too
  risky, lean WAIT/AVOID. Do NOT use a fixed 3% stop -- it is empirically invalidated.
- Target: reference the resistance level; it must exceed breakeven by >1%.
- Risk/reward: compute risk_reward honestly as (target - entry) / (entry - stop_loss). Prefer setups with
  risk_reward >= 1.2. If risk_reward < 1.0, default to WAIT or AVOID unless a high-conviction validated
  edge (BB_LOWER_TOUCH + OBV rising, or confirmed broker accumulation) justifies the entry.
- Hold: use suggested hold from research ({hold_days} days{_hold_signal_str}).
- Use max 10% of total capital per position.
- Max {portfolio.get('max_positions', 3)} simultaneous positions -- slots remaining: {portfolio.get('slots_remaining', 0)}.
- Include only ordinary shares, exclude mutual funds, debentures, promoter shares.
- Consider fundamental signals as supporting context, not primary trigger.

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
  "primary_signal": "MACD or BB or SMA or OBV_MOMENTUM or RSI or VOLUME_BREAKOUT",
  "reasoning": "5-6 sentences covering: why this signal (name the validated edge if any), the risk_reward and key risks, what the Learning Hub says, sector/regime context, and any fundamental quality flags",
  "lesson_applied": "which lesson from Learning Hub was most relevant, or NONE",
  "wait_condition": "if WAIT: max 2 stock-specific conditions only (e.g. price holds above X support, RSI rises above 45). NO tech_score thresholds, NO confidence thresholds, NO breadth conditions, NO nepal_score conditions. Keep it simple and triggerable.",
  "herding_note": "one sentence on herding/bubble risk or NONE"
}}"""


# =============================================================================
# SECTION 5- ASSEMBLE RESULT + WRITE TO DB
# =============================================================================

def _assemble_result(claude_json: dict, flag, geo: dict) -> AnalystResult:
    primary_signal = claude_json.get("primary_signal", "")
    # Sanitize — LAGGARD_PLAY is no longer a valid signal
    VALID_PRIMARY_SIGNALS = {"MACD", "BB", "SMA", "OBV_MOMENTUM", "RSI", "VOLUME_BREAKOUT"}
    if False:  # signal whitelist removed � Claude decides freely
        logger.warning(
            "%s: invalid primary_signal '%s' from Claude — defaulting to MACD",
            flag.symbol, primary_signal
        )
        primary_signal = "MACD"

    return AnalystResult(
        symbol             = flag.symbol,
        action             = claude_json.get("action",           "WAIT"),
        confidence         = int(claude_json.get("confidence",   0)),
        entry_price        = float(claude_json.get("entry_price") or 0),
        stop_loss          = float(claude_json.get("stop_loss")  or 0),
        target             = float(claude_json.get("target")     or 0),
        allocation_npr     = float(claude_json.get("allocation_npr", 0)),
        shares             = int(claude_json.get("shares",       0)),
        breakeven          = float(claude_json.get("breakeven")  or 0),
        risk_reward        = float(claude_json.get("risk_reward") or 0),
        suggested_hold     = int(claude_json.get("suggested_hold_days", 17)),
        reasoning          = claude_json.get("reasoning",        ""),
        lesson_applied     = claude_json.get("lesson_applied",   "NONE"),
        wait_condition     = claude_json.get("wait_condition",   ""),
        herding_note       = claude_json.get("herding_note",     "NONE"),
        primary_signal     = primary_signal,
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
            momentum_status = str(getattr(flag, "momentum_status",  "NEUTRAL") or "NEUTRAL")
            rsi_slope_3d    = str(getattr(flag, "rsi_slope_3d",     0.0)       or "0")
            macd_hist_slope = str(getattr(flag, "macd_hist_slope",  0.0)       or "0")
            bb_pct_b_slope  = str(getattr(flag, "bb_pct_b_slope",   0.0)       or "0")
            bounce_failed   = str(getattr(flag, "bounce_failed",     False)     or "false").lower()
            reversal_days   = str(getattr(flag, "reversal_days",     0)         or "0")
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
            momentum_status = "NEUTRAL"
            rsi_slope_3d = macd_hist_slope = bb_pct_b_slope = "0"
            bounce_failed = "false"
            reversal_days = "0"

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
            "wait_condition":    _s(result.wait_condition),
            "herding_note":      _s(result.herding_note),
            "lesson_applied":    _s(result.lesson_applied),
            "primary_signal":    _s(result.primary_signal),
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

            # ATrad intraday signals
            "vwap_dev":        _s(getattr(flag, "vwap_dev",        0.0)) if flag else "",
            "bid_ask_ratio":   _s(getattr(flag, "bid_ask_ratio",   0.0)) if flag else "",
            "dpr_proximity":   _s(getattr(flag, "dpr_proximity",   0.0)) if flag else "",
            "volume_os_ratio": _s(getattr(flag, "volume_os_ratio", 0.0)) if flag else "",

            # FIX 1: headlines written from result (which got them from geo context)
            "headlines_politics": _s(result.headlines_politics),
            "headlines_economy":  _s(result.headlines_economy),
            "headlines_stock":    _s(result.headlines_stock),

            # Momentum direction (pre-computed by filter_engine)
            "momentum_status":  momentum_status,
            "rsi_slope_3d":     rsi_slope_3d,
            "macd_hist_slope":  macd_hist_slope,
            "bb_pct_b_slope":   bb_pct_b_slope,
            "bounce_failed":    bounce_failed,
            "reversal_days":    reversal_days,

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
# SECTION 6 - MAIN ENTRY POINT
# =============================================================================

def run_analysis(flags: list) -> list[AnalystResult]:
    if not flags:
        logger.info("claude_analyst: no flags to analyze")
        return []

    logger.info("=" * 60)
    logger.info("claude_analyst.run_analysis() -- %d flags", len(flags))

    # Read monthly_override before any analysis (DB read — always allowed)
    _load_monthly_override()

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
        logger.info("Portfolio full (%d/%d positions open) -- no analysis needed",
                    portfolio.get("open_positions", 0), portfolio.get("max_positions", 3))
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
        if portfolio.get("open_positions", 0) + buy_count >= portfolio.get("max_positions", 3):
            logger.info("%s: portfolio full after earlier BUYs -- skipping", sym)
            continue

        lessons     = _load_lessons(sym, getattr(flag, "sector", ""))
        fund_ctx    = _load_fundamentals_context(sym, getattr(flag, "sector", ""))
        broker_ctx  = _load_broker_flow_context(sym)

        logger.info(
            "Fundamentals for %s: found=%s | beta=%s | notes=%d",
            sym,
            fund_ctx.get("found", False),
            fund_ctx.get("beta", "N/A"),
            len(fund_ctx.get("trend_notes", [])),
        )
        logger.info(
            "Broker flow for %s: flow=%s holdings=%s",
            sym,
            "yes" if broker_ctx.get("flow") else "no",
            "yes" if broker_ctx.get("holdings") else "no",
        )

        try:
            _ob = fetch_order_book(flag.symbol)
            print(f"DEBUG ORDER_BOOK {flag.symbol}: {_ob}")
            if _ob and "imbalance" in _ob:
                _new_imb = float(_ob.get("imbalance", flag.bid_ask_ratio))
                flag.bid_ask_ratio = _new_imb
                print(f"DEBUG OVERRIDDEN to {_new_imb}")
        except Exception as e:
            print(f"DEBUG ORDER_BOOK EXCEPTION: {e}")

        _audit_prompt_fields(flag, geo, macro, portfolio)
        prompt      = _build_prompt(
            flag, portfolio, geo, macro, lessons, market_state, loss_streak,
            fund_ctx=fund_ctx,
            broker_ctx=broker_ctx,
        )
        claude_json = ask_claude(prompt, context="claude_analyst")

        if claude_json is None:
            logger.warning("%s: Claude returned no result -- skipping", sym)
            continue

        result = _assemble_result(claude_json, flag, geo)

        # 2b: monthly_override buy block
        if _BUY_BLOCKED and result.action == "BUY":
            logger.warning(
                "[monthly_override] BUY blocked for %s → WAIT (reason=monthly_override_block)",
                sym,
            )
            result.action    = "WAIT"
            result.reasoning = (
                f"[monthly_override_block] Council confidence={_BUY_BLOCK_SCORE} ≤ 20 "
                f"in BEAR/CRISIS regime. BUY overridden to WAIT. "
                f"Original: {result.reasoning}"
            )

        results.append(result)
        _write_to_db(result, flag=flag)
        logger.info("Result: %s", result.summary())

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
# SECTION 7 - FORMAT FOR NOTIFIER
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
        lines.append(f"Lesson Applied: {result.lesson_applied}")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [CLAUDE_ANALYST] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

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
            from modules.scraper import get_all_market_data as _get_md
            _md = _get_md(write_breadth=False)
            flags = run_gemini_filter(market_data=_md)
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
            lessons    = _load_lessons(flag.symbol, getattr(flag, "sector", ""))
            fund_ctx   = _load_fundamentals_context(flag.symbol, getattr(flag, "sector", ""))
            broker_ctx = _load_broker_flow_context(flag.symbol)
            try:
                _ob = fetch_order_book(flag.symbol)
                print(f"DEBUG ORDER_BOOK {flag.symbol}: imbalance={_ob.get('imbalance')}")
                if _ob and "imbalance" in _ob:
                    flag.bid_ask_ratio = float(_ob["imbalance"])
                    print(f"DEBUG OVERRIDDEN bid_ask_ratio={flag.bid_ask_ratio:.4f}")
            except Exception as e:
                print(f"DEBUG ORDER_BOOK EXCEPTION: {e}")
            _audit_prompt_fields(flag, geo, macro, portfolio)
            prompt     = _build_prompt(
                flag, portfolio, geo, macro, lessons, market_state, loss_streak,
                fund_ctx=fund_ctx,
                broker_ctx=broker_ctx,
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
        from modules.scraper import get_all_market_data as _get_md
        _md = _get_md(write_breadth=False)
        flags = run_gemini_filter(market_data=_md)
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
