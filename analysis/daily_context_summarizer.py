"""
daily_context_summarizer.py — NEPSE AI Engine
==============================================
Runs nightly at ~9 PM NST after market close and auditor.py.
Collapses all intraday data (geopolitical_data, nepal_pulse, market_breadth,
market_log) into ONE clean summary row in daily_context_log per trading day.

GPT Sunday reviewer reads daily_context_log instead of raw intraday tables —
one structured row per day instead of 40+ intraday snapshots.

Run modes:
    python -m analysis.daily_context_summarizer            # summarize today
    python -m analysis.daily_context_summarizer --date 2026-04-01  # specific date
    python -m analysis.daily_context_summarizer --dry-run  # compute, do not write
    python -m analysis.daily_context_summarizer --prompt   # print Gemini prompt only

Called by:
    nightly_summary.yml (GitHub Actions, ~9 PM NST Sun-Thu)
"""

from curses import raw
from curses import raw
import os
import sys
import logging
import argparse
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from AI import ask_gemini_text


from sheets import run_raw_sql, upsert_row, get_setting

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")






    
# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHERS — one function per source table
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_geo_rows(target_date: str) -> list[dict]:
    """All geopolitical_data rows for target_date, ordered by id."""
    try:
        return run_raw_sql(
            "SELECT * FROM geopolitical_data WHERE date = %s ORDER BY id ASC",
            (target_date,),
        ) or []
    except Exception as e:
        log.warning("geo fetch failed for %s: %s", target_date, e)
        return []


def _fetch_pulse_rows(target_date: str) -> list[dict]:
    """All nepal_pulse rows for target_date, ordered by id."""
    try:
        return run_raw_sql(
            "SELECT * FROM nepal_pulse WHERE date = %s ORDER BY id ASC",
            (target_date,),
        ) or []
    except Exception as e:
        log.warning("pulse fetch failed for %s: %s", target_date, e)
        return []


def _fetch_breadth_row(target_date: str) -> dict | None:
    """Latest market_breadth row for target_date."""
    try:
        rows = run_raw_sql(
            "SELECT * FROM market_breadth WHERE date = %s ORDER BY id DESC LIMIT 1",
            (target_date,),
        )
        return rows[0] if rows else None
    except Exception as e:
        log.warning("breadth fetch failed for %s: %s", target_date, e)
        return None


def _fetch_market_log_rows(target_date: str) -> list[dict]:
    """All market_log BUY/WAIT/AVOID rows for target_date."""
    try:
        return run_raw_sql(
            """
            SELECT symbol, sector, action, confidence, reasoning,
                   entry_price, stop_loss, target, geo_score, macro_score
            FROM market_log
            WHERE date = %s AND action IN ('BUY','WAIT','AVOID')
            ORDER BY action, confidence DESC
            """,
            (target_date,),
        ) or []
    except Exception as e:
        log.warning("market_log fetch failed for %s: %s", target_date, e)
        return []

def _fetch_gate_miss_day(target_date: str) -> dict:
    """
    Fetch gate_misses stats for target_date.
    Returns count, top category, and rolling false_block_rate.
    """
    try:
        # Count today's gate_misses
        count_rows = run_raw_sql(
            "SELECT COUNT(*) as cnt, gate_category FROM gate_misses WHERE date = %s GROUP BY gate_category ORDER BY cnt DESC",
            (target_date,)
        )
        if not count_rows:
            return {"gate_miss_count": 0, "gate_top_category": "", "gate_false_block_pct": ""}

        total = sum(int(r["cnt"]) for r in count_rows)
        top_cat = count_rows[0]["gate_category"] if count_rows else ""

        # Rolling 30-day false block rate (all categories)
        rate_rows = run_raw_sql(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'FALSE_BLOCK' THEN 1 ELSE 0 END) as false_blocks
            FROM gate_misses
            WHERE date::date >= (%s::date - INTERVAL '30 days')
            AND outcome IS NOT NULL
            """,
            (target_date,)
        )
        false_block_pct = ""
        if rate_rows and int(rate_rows[0]["total"] or 0) > 0:
            pct = int(rate_rows[0]["false_blocks"] or 0) / int(rate_rows[0]["total"]) * 100
            false_block_pct = str(round(pct, 1))

        return {
            "gate_miss_count":     str(total),
            "gate_top_category":   top_cat,
            "gate_false_block_pct": false_block_pct,
        }
    except Exception as e:
        log.warning("_fetch_gate_miss_day failed: %s", e)
        return {"gate_miss_count": "", "gate_top_category": "", "gate_false_block_pct": ""}

def _fetch_signals_confidence(target_date: str) -> str:
    """Average Claude confidence across all signals today."""
    try:
        rows = run_raw_sql(
            """
            SELECT AVG(confidence::float) as avg_conf
            FROM market_log
            WHERE date = %s AND action IN ('BUY','WAIT','AVOID')
              AND confidence IS NOT NULL AND confidence != ''
            """,
            (target_date,)
        )
        if rows and rows[0]["avg_conf"]:
            return str(round(float(rows[0]["avg_conf"]), 1))
        return ""
    except Exception:
        return ""

def _fetch_nrb_latest() -> dict | None:
    """Most recent nrb_monthly row."""
    try:
        rows = run_raw_sql(
            """
            SELECT policy_rate, bank_rate, credit_growth_rate,
                   bop_status, bop_overall_balance_usd_m,
                   overall_sentiment, forward_guidance, key_risks,
                   remittance_yoy_change_pct, cpi_inflation,
                   liquidity_injected_billion, fx_reserve_months
            FROM nrb_monthly
            ORDER BY fiscal_year DESC, month_number DESC
            LIMIT 1
            """
        )
        return rows[0] if rows else None
    except Exception as e:
        log.warning("nrb_monthly fetch failed: %s", e)
        return None


def _fetch_nepse_index(target_date: str) -> dict | None:
    """NEPSE composite index for target_date or closest prior day."""
    try:
        rows = run_raw_sql(
            """
            SELECT current_value, change_pct
            FROM nepse_indices
            WHERE index_id = '58' AND date <= %s AND current_value IS NOT NULL
            ORDER BY date DESC LIMIT 1
            """,
            (target_date,),
        )
        return rows[0] if rows else None
    except Exception as e:
        log.warning("nepse_indices fetch failed: %s", e)
        return None


def _fetch_prev_nepse_index(target_date: str) -> float | None:
    """NEPSE index value for the day BEFORE target_date."""
    try:
        rows = run_raw_sql(
            """
            SELECT current_value FROM nepse_indices
            WHERE index_id = '58' AND date < %s AND current_value IS NOT NULL
            ORDER BY date DESC LIMIT 1
            """,
            (target_date,),
        )
        if rows:
            return float(rows[0]["current_value"])
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# DATA ASSEMBLERS
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(val) -> str | None:
    """Convert to string safely. Returns None for None — never 'None'."""
    if val is None:
        return None
    return str(val)


def _assemble_geo_data(geo_rows: list[dict]) -> dict:
    """Extract end-of-day geo values from intraday rows."""
    if not geo_rows:
        return {}
    last = geo_rows[-1]
    return {
        "geo_score_eod": last.get("geo_score"),
        "dxy_value":     last.get("dxy"),
        "dxy_raw_rows":  geo_rows,
    }


def _assemble_pulse_data(pulse_rows: list[dict]) -> dict:
    """Extract end-of-day nepal score + key events."""
    if not pulse_rows:
        return {}
    last = pulse_rows[-1]

    # Collect unique key_event lines across all intraday rows
    all_events = []
    seen = set()
    for row in pulse_rows:
        evt = (row.get("key_event") or "").strip()
        if evt and evt not in seen:
            seen.add(evt)
            all_events.append(evt)

    highlights = " | ".join(all_events[:3]) if all_events else ""

    return {
        "nepal_score_eod":       last.get("nepal_score"),
        "nepal_pulse_highlights": highlights,
        "pulse_raw_rows":        pulse_rows,
        "bandh_today":           last.get("bandh_today"),
        "govt_stability":        last.get("govt_stability"),
        "ipo_fpo_active":        last.get("ipo_fpo_active"),
        "circuit_breaker":       last.get("circuit_breaker"),
        "inflation_pct":         last.get("inflation_pct"),
        "forex_reserve_months":  last.get("forex_reserve_months"),
    }


def _assemble_signals_summary(market_log_rows: list[dict]) -> dict:
    """Build human-readable signals summary from market_log."""
    buys   = [r for r in market_log_rows if r.get("action") == "BUY"]
    waits  = [r for r in market_log_rows if r.get("action") == "WAIT"]
    avoids = [r for r in market_log_rows if r.get("action") == "AVOID"]

    def fmt(rows):
        parts = []
        for r in rows:
            sym  = r.get("symbol", "?")
            conf = r.get("confidence", "?")
            parts.append(f"{sym}({conf}%)")
        return ", ".join(parts) if parts else "none"

    parts = []
    if buys:
        parts.append(f"BUY: {fmt(buys)}")
    if waits:
        parts.append(f"WAIT: {fmt(waits)}")
    if avoids:
        parts.append(f"AVOID: {fmt(avoids)}")

    summary = " | ".join(parts) if parts else "No signals today"

    return {
        "signals_summary": summary,
        "buy_count":       str(len(buys)),
        "wait_count":      str(len(waits)),
        "avoid_count":     str(len(avoids)),
        "signals_raw":     market_log_rows,
    }

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI NARRATIVE BUILDER — token-optimized prompt
# ─────────────────────────────────────────────────────────────────────────────

# Fields to keep per source table — everything else stripped
_GEO_KEEP   = {"geo_score", "dxy", "status", "key_event", "vix_level"}
_PULSE_KEEP = {"nepal_score", "nepal_status", "key_event", "bandh_today",
               "govt_stability", "ipo_fpo_active", "circuit_breaker"}


def _compact_rows(rows: list[dict], keep: set[str], limit: int = 6) -> str:
    """Serialize rows keeping only relevant fields. Token-efficient."""
    if not rows:
        return "none"
    subset = rows[-limit:]
    lines = []
    for r in subset:
        filtered = {k: v for k, v in r.items()
                    if k in keep and v is not None and str(v).strip()}
        if filtered:
            lines.append(json.dumps(filtered, ensure_ascii=False))
    return "\n".join(lines) if lines else "none"


def _build_gemini_prompt(
    target_date: str,
    geo_data: dict,
    pulse_data: dict,
    breadth: dict | None,
    nrb: dict | None,
    signals: dict,
    nepse_index: dict | None,
    gate_data: dict | None = None,
    avg_conf: str | None = None,
) -> str:
    """Build token-optimized prompt for Gemini to generate narrative summaries."""

    geo_compact   = _compact_rows(geo_data.get("dxy_raw_rows", []), _GEO_KEEP)
    pulse_compact = _compact_rows(pulse_data.get("pulse_raw_rows", []), _PULSE_KEEP)

    breadth_str = ""
    if breadth:
        breadth_str = (
            f"Adv: {breadth.get('advancing')} | Dec: {breadth.get('declining')} | "
            f"Breadth: {breadth.get('breadth_score')} | Turnover: {breadth.get('total_turnover_npr')} NPR"
        )

    nrb_str = ""
    if nrb:
        nrb_keep = {k: v for k, v in nrb.items()
                    if v is not None and k in (
                        "policy_rate", "credit_growth_rate", "bop_status",
                        "bop_overall_balance_usd_m", "overall_sentiment",
                        "forward_guidance", "cpi_inflation", "fx_reserve_months")}
        nrb_str = json.dumps(nrb_keep, ensure_ascii=False)

    nepse_str = ""
    if nepse_index:
        nepse_str = f"NEPSE: {nepse_index.get('current_value')} ({nepse_index.get('change_pct')}%)"

    signals_str = signals.get("signals_summary", "No signals today")

    # Compact signals — just symbol, action, confidence
    sig_detail = ""
    sig_raw = signals.get("signals_raw", [])
    if sig_raw:
        sig_lines = []
        for r in sig_raw[:8]:
            sig_lines.append(json.dumps({
                "symbol": r.get("symbol"), "action": r.get("action"),
                "confidence": r.get("confidence"), "sector": r.get("sector"),
            }, ensure_ascii=False))
        sig_detail = "\n".join(sig_lines)

    return f"""You are the daily market summarizer for the NEPSE AI Engine.
Date: {target_date}

Read the raw data below and produce THREE narrative summaries + headlines.
Be factual, specific, NEPSE-focused. No filler. Output stored in DB for GPT-4o weekly review.

━━━ DATA ━━━


GEO/INTERNATIONAL (last intraday rows):
{geo_compact}

NEPAL PULSE (last intraday rows):
{pulse_compact}

BREADTH: {breadth_str if breadth_str else "no data"}
{nepse_str if nepse_str else ""}

f"\nGATE BLOCKS TODAY: {gate_data.get('gate_miss_count','?')} blocked | top reason: {gate_data.get('gate_top_category','?')} | 30d false block rate: {gate_data.get('gate_false_block_pct','?')}%"
f"\nSIGNAL CONFIDENCE AVG: {avg_conf or 'N/A'}"

NRB MACRO (latest monthly):
{nrb_str if nrb_str else "no data"}

SIGNALS: {signals_str}
{sig_detail if sig_detail else ""}

━━━ OUTPUT — JSON only, no markdown ━━━

{{
  "key_events_summary": "<4-6 sentences: what happened today in Nepal markets, politics, economy, international signals for NEPSE. Be specific.>",
  "geo_summary": "<2-3 sentences: DXY movement + implications for NEPSE remittance flows. State DXY value and direction.>",
  "nrb_macro_summary": "<2 sentences: current NRB macro backdrop — policy rate, credit growth, BOP, liquidity.>",
  "headlines_political": ["<political headline 1>", "<political headline 2 or empty string>"],
  "headlines_economy": ["<economic headline 1>", "<economic headline 2 or empty string>"],
  "headlines_nepse": ["<NEPSE headline 1>", "<NEPSE headline 2 or empty string>"]
}}

Headlines: concise (one sentence each), factual, NEPSE-relevant. Use "" for missing slots."""


def _parse_gemini_response(raw: str) -> dict:
    """Parse Gemini JSON response with robust fallback."""
    if not raw:
        return {}
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                log.warning("Gemini JSON parse failed: %s | raw: %s", e, raw[:200])
        else:
            log.warning("Gemini response has no JSON object: %s", raw[:200])
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# CORE: BUILD ONE DAILY_CONTEXT_LOG ROW
# ─────────────────────────────────────────────────────────────────────────────


def build_daily_context(target_date: str, dry_run: bool = False) -> dict | None:
    """
    Fetch all intraday data for target_date, call Gemini for narratives,
    assemble and upsert one row into daily_context_log.
    Returns the assembled row dict (or None on failure).
    """
    log.info("Building daily context for %s", target_date)

    # ── Fetch all sources
    geo_rows   = _fetch_geo_rows(target_date)
    pulse_rows = _fetch_pulse_rows(target_date)
    breadth    = _fetch_breadth_row(target_date)
    log_rows   = _fetch_market_log_rows(target_date)
    nrb        = _fetch_nrb_latest()
    nepse_idx  = _fetch_nepse_index(target_date)
    prev_nepse = _fetch_prev_nepse_index(target_date)
    gate_data   = _fetch_gate_miss_day(target_date)
    avg_conf    = _fetch_signals_confidence(target_date)

    # ── Check if any data exists at all for this date
    if not geo_rows and not pulse_rows and not breadth and not log_rows:
        log.info("No data found for %s — skipping (likely non-trading day)", target_date)
        return None

    # ── Assemble structured data
    geo_data  = _assemble_geo_data(geo_rows)
    pulse_data = _assemble_pulse_data(pulse_rows)
    signals   = _assemble_signals_summary(log_rows)

    # ── Compute combined score
    geo_eod   = _safe_float(geo_data.get("geo_score_eod"))
    nepal_eod = _safe_float(pulse_data.get("nepal_score_eod"))
    combined  = None
    if geo_eod is not None and nepal_eod is not None:
        combined = geo_eod + nepal_eod

    # ── NEPSE change pct
    nepse_val = None
    nepse_chg = None
    if nepse_idx:
        nepse_val = nepse_idx.get("current_value")
        nepse_chg = nepse_idx.get("change_pct")
        if nepse_chg is None and prev_nepse and nepse_val:
            try:
                curr = float(nepse_val)
                chg  = ((curr - prev_nepse) / prev_nepse) * 100
                nepse_chg = str(round(chg, 4))
            except (ValueError, ZeroDivisionError):
                pass

    # ── DXY change pct (first vs last intraday row)
    dxy_chg = None
    if len(geo_rows) >= 2:
        try:
            first_dxy = _safe_float(geo_rows[0].get("dxy"))
            last_dxy  = _safe_float(geo_rows[-1].get("dxy"))
            if first_dxy and last_dxy and first_dxy != 0:
                dxy_chg = str(round(((last_dxy - first_dxy) / first_dxy) * 100, 4))
        except Exception:
            pass

    # ── Market state from settings
    try:
        market_state = get_setting("MARKET_STATE") or "SIDEWAYS"
    except Exception:
        market_state = "SIDEWAYS"

    # ── FD rate from settings
    try:
        fd_rate = get_setting("FD_RATE_PCT") or "8.5"
    except Exception:
        fd_rate = "8.5"

    # ── NRB fields
    policy_rate   = nrb.get("policy_rate")          if nrb else None
    lending_rate  = nrb.get("bank_rate")            if nrb else None
    bop_status    = nrb.get("bop_status")           if nrb else None
    nrb_sentiment = nrb.get("overall_sentiment")    if nrb else None

    # ── Call Gemini for narratives
    prompt = _build_gemini_prompt(
        target_date, geo_data, pulse_data, breadth, nrb, signals, nepse_idx,
        gate_data=gate_data, avg_conf=avg_conf
    )
    raw        = ask_gemini_text(prompt, context="daily_summarizer")
    narratives = _parse_gemini_response(raw) if raw else {}

    if not narratives:
        log.warning("Gemini narrative failed for %s — using fallback", target_date)
        narratives = {
            "key_events_summary": pulse_data.get("nepal_pulse_highlights") or "No summary available",
            "geo_summary":        f"DXY: {geo_data.get('dxy_value', 'N/A')}",
            "nrb_macro_summary":  f"Policy rate: {policy_rate or 'N/A'} | BOP: {bop_status or 'N/A'}",
            "headlines_political": ["", ""],
            "headlines_economy":   ["", ""],
            "headlines_nepse":     ["", ""],
        }

    # ── Build headlines pipe-separated strings
    def _join_headlines(key: str) -> str | None:
        items = narratives.get(key, ["", ""])
        joined = " | ".join(h for h in items if h)
        return joined if joined else None

    # ── Assemble final row — all values through _safe_str to prevent 'None' strings
    now_nst = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "date":                   target_date,

        # Scores
        "geo_score_eod":          _safe_str(geo_eod),
        "nepal_score_eod":        _safe_str(nepal_eod),
        "combined_score_eod":     _safe_str(combined),
        "nepse_index_eod":        _safe_str(nepse_val),
        "nepse_change_pct":       _safe_str(nepse_chg),

        # International
        "dxy_value":              _safe_str(geo_data.get("dxy_value")),
        "dxy_change_pct":         _safe_str(dxy_chg),

        # Market breadth
        "market_state":           market_state,
        "advancing":              _safe_str(breadth.get("advancing"))         if breadth else None,
        "declining":              _safe_str(breadth.get("declining"))         if breadth else None,
        "breadth_score":          _safe_str(breadth.get("breadth_score"))     if breadth else None,
        "total_turnover_npr":     _safe_str(breadth.get("total_turnover_npr")) if breadth else None,

        # Macro
        "policy_rate":            _safe_str(policy_rate),
        "fd_rate_pct":            fd_rate,
        "lending_rate":           _safe_str(lending_rate),
        "bop_status":             bop_status,
        "overall_sentiment":      nrb_sentiment,

        # Gemini narratives
        "key_events_summary":     narratives.get("key_events_summary"),
        "nepal_pulse_highlights": pulse_data.get("nepal_pulse_highlights") or None,
        "geo_summary":            narratives.get("geo_summary"),
        "nrb_macro_summary":      narratives.get("nrb_macro_summary"),
        "headlines_political":    _join_headlines("headlines_political"),
        "headlines_economy":      _join_headlines("headlines_economy"),
        "headlines_nepse":        _join_headlines("headlines_nepse"),

        # Signals
        "signals_summary":        signals.get("signals_summary"),
        "buy_count":              signals.get("buy_count", "0"),
        "wait_count":             signals.get("wait_count", "0"),
        "avoid_count":            signals.get("avoid_count", "0"),


        "gate_miss_count":        gate_data.get("gate_miss_count"),
        "gate_top_category":      gate_data.get("gate_top_category"),
        "gate_false_block_pct":   gate_data.get("gate_false_block_pct"),
        "signals_avg_confidence": avg_conf or None,

        # Metadata
        "source":                 "gemini_nightly",
        "backfilled":             "false",
        "created_at":             now_nst,
    }

    # Note: headlines_political / headlines_economy / headlines_nepse are NOT
    # in the daily_context_log schema. They are embedded in key_events_summary.
    # Gemini still generates them for structured extraction if schema is extended.

    if dry_run:
        log.info("[DRY-RUN] Would write daily_context_log for %s", target_date)
        log.info("  geo=%.1f nepal=%.1f combined=%s nepse=%s signals=%s",
                 geo_eod or 0, nepal_eod or 0,
                 combined or "N/A", nepse_val or "N/A",
                 signals.get("signals_summary"))
        return row

    try:
        upsert_row("daily_context_log", row, conflict_columns=["date"])
        log.info("Wrote daily_context_log for %s", target_date)
    except Exception as e:
        log.error("Failed to write daily_context_log for %s: %s", target_date, e)
        return None

    return row


def get_prompt_for_date(target_date: str) -> str:
    """
    Build and return the Gemini prompt for a date without calling the API.
    Used by prompt_viewer CLI.
    """
    geo_rows   = _fetch_geo_rows(target_date)
    pulse_rows = _fetch_pulse_rows(target_date)
    breadth    = _fetch_breadth_row(target_date)
    log_rows   = _fetch_market_log_rows(target_date)
    nrb        = _fetch_nrb_latest()
    nepse_idx  = _fetch_nepse_index(target_date)

    geo_data   = _assemble_geo_data(geo_rows)
    pulse_data = _assemble_pulse_data(pulse_rows)
    signals    = _assemble_signals_summary(log_rows)

    return _build_gemini_prompt(
        target_date, geo_data, pulse_data, breadth, nrb, signals, nepse_idx
    )

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def run() -> None:
    """Entry point called by eod_workflow.py. Summarizes today's date."""
    target = datetime.now(NST).strftime("%Y-%m-%d")
    log.info("Summarizing %s", target)
    result = build_daily_context(target, dry_run=False)
    if result:
        log.info(
            "Done. geo=%.1f nepal=%.1f combined=%s",
            _safe_float(result.get("geo_score_eod"), 0),
            _safe_float(result.get("nepal_score_eod"), 0),
            result.get("combined_score_eod", "N/A"),
        )
    else:
        log.warning("build_daily_context returned None for %s", target)


def main():
    parser = argparse.ArgumentParser(description="NEPSE Daily Context Summarizer")
    parser.add_argument("--date", type=str, default=None,
                        help="Summarize a specific date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and log but do not write to DB")
    parser.add_argument("--prompt", action="store_true",
                        help="Print the Gemini prompt and exit (no API call)")
    args = parser.parse_args()

    if args.date:
        target = args.date
    else:
        target = datetime.now(NST).strftime("%Y-%m-%d")

    if args.prompt:
        prompt = get_prompt_for_date(target)
        print(prompt)
        print(f"\n--- Estimated tokens: ~{len(prompt) // 4} ---")
        return

    log.info("Summarizing %s (dry_run=%s)", target, args.dry_run)
    result = build_daily_context(target, dry_run=args.dry_run)

    if result:
        log.info("Done. geo=%.1f nepal=%.1f combined=%s",
                 _safe_float(result.get("geo_score_eod"), 0),
                 _safe_float(result.get("nepal_score_eod"), 0),
                 result.get("combined_score_eod", "N/A"))
    else:
        log.warning("No output produced for %s", target)

    if args.dry_run:
        log.info("[DRY-RUN] No writes performed.")


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    main()