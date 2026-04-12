# -*- coding: utf-8 -*-
"""
learning_hub.py  -  NEPSE AI Engine
===================================
GPT-5o weekly learning review. Runs every Sunday at ~5:45 PM NST.

Reads:
  - trade_journal          -  all BUY trade outcomes with full causal attribution
  - market_log             -  all evaluated WAIT/AVOID outcomes from recommendation_tracker
  - daily_context_log      -  one clean row per trading day (Gemini nightly summary)
  - learning_hub           -  all currently active lessons (for GPT to decide what to supersede)

Writes (Option B  -  supersede, never overwrite):
  - New lessons:     insert new row, active=true
  - Updated lessons: set old row active=false + superseded_by=new_id,
                     insert new row with supersedes_lesson_id=old_id
  - No lesson is ever deleted  -  full audit trail preserved

Anti-overfitting gates enforced in prompt:
  - BLOCK_ENTRY only after 25+ trades supporting the pattern
  - Per-sector blocks need sector-level trade count
  - Per-signal blocks need signal-level trade count
  - LOW confidence → inform only, never block
  - HIGH confidence → can block, must have 25+ trades

Run modes:
    python -m analysis.learning_hub            # full Sunday review
    python -m analysis.learning_hub --dry-run  # compute, print, do not write
    python -m analysis.learning_hub --status   # show current lessons summary
    python -m analysis.learning_hub --prompt   # print GPT prompt only (no API call)

Called by:
    weekly_review.yml (GitHub Actions, Sunday ~5:45 PM NST)
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from sheets import run_raw_sql, upsert_row, write_row, get_setting

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
# TOKEN BUDGET  -  controls how much data we send to GPT
# ─────────────────────────────────────────────────────────────────────────────

MAX_TRADES        = 80     # most recent trades (GPT sees aggregate stats for older)
MAX_WAIT_AVOID    = 80     # most recent evaluated WAIT/AVOID signals
MAX_DAILY_CONTEXT = 90     # ~3 months of trading days
MAX_GPT_TOKENS    = 8000   # max_tokens for GPT response

# ─────────────────────────────────────────────────────────────────────────────
# GPT SETUP (OpenRouter)
# ─────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GPT_MODEL          = os.environ.get("GPT_MODEL", "openai/gpt-5o")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"


def _call_gpt(system_prompt: str, user_prompt: str, max_tokens: int = MAX_GPT_TOKENS) -> str:
    """Call GPT via OpenRouter. Returns text response."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/nepse-ai-engine",
        "X-Title":       "NEPSE AI Engine  -  Learning Hub",
    }
    payload = {
        "model":      GPT_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("GPT call failed: %s", e)
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  -  with token-aware limits
# ─────────────────────────────────────────────────────────────────────────────


def _load_trade_journal() -> tuple[list[dict], dict]:
    """
    Load trade_journal rows. Returns (recent_rows, aggregate_stats).
    Recent rows are the last MAX_TRADES trades for GPT detail.
    Aggregate stats cover ALL trades for orientation.
    """
    try:
        # Aggregate stats  -  all time
        agg_rows = run_raw_sql(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CAST(NULLIF(return_pct, '') AS FLOAT)) as avg_return,
                AVG(CAST(NULLIF(alpha_vs_nepse, '') AS FLOAT)) as avg_alpha
            FROM trade_journal
            """
        ) or [{}]
        agg = agg_rows[0] if agg_rows else {}

        # Recent trades for detail
        rows = run_raw_sql(
            f"""
            SELECT * FROM trade_journal
            ORDER BY entry_date DESC
            LIMIT {MAX_TRADES}
            """
        ) or []
        # Reverse to chronological order for GPT
        rows.reverse()
        log.info("Loaded %d trade_journal rows (total in DB: %s)", len(rows), agg.get("total", "?"))
        return rows, agg
    except Exception as e:
        log.error("trade_journal load failed: %s", e)
        return [], {}


def _load_wait_avoid_outcomes() -> list[dict]:
    """
    All WAIT/AVOID/BUY rows from market_log for GPT review.
    Includes PENDING rows -- GPT needs the full picture, not just evaluated ones.
    AVOID rows are CLOSED immediately on write (no hold period).
    WAIT rows expire after 5 calendar days.
    BUY rows are stamped by evaluate_buy_signals() after hold period.
    """
    try:
        rows = run_raw_sql(
            f"""
            SELECT id, date, symbol, sector, action, confidence, reasoning,
                   geo_score, macro_score, candle_pattern,
                   outcome, exit_date, exit_price,
                   eval_date, eval_geo_score, eval_nepal_score,
                   eval_nepse_index, eval_market_state, eval_policy_rate,
                   eval_fd_rate_pct, eval_geo_delta, eval_nepal_delta,
                   eval_price_change_pct, eval_nepse_change_pct, eval_alpha,
                   eval_key_news,
                   entry_price, stop_loss, target, primary_signal
            FROM market_log
            WHERE action IN ('WAIT', 'AVOID', 'BUY')
            ORDER BY date DESC
            LIMIT {MAX_WAIT_AVOID}
            """
        ) or []
        rows.reverse()
        log.info("Loaded %d market_log rows (all actions, all outcomes) for GPT", len(rows))
        return rows
    except Exception as e:
        log.error("market_log load failed: %s", e)
        return []


def _load_buy_decisions() -> list[dict]:
    """
    Load BUY rows from market_log with full signal context.
    Cross-references trade_journal to attach final outcome where available.
    """
    try:
        rows = run_raw_sql(
            f"""
            SELECT
                ml.id,
                ml.date,
                ml.symbol,
                ml.sector,
                ml.action,
                ml.confidence,
                ml.reasoning,
                ml.entry_price,
                ml.stop_loss,
                ml.target,
                ml.allocation_npr,
                ml.shares,
                ml.primary_signal,
                ml.rsi_14,
                ml.macd_line,
                ml.macd_signal,
                ml.macd_histogram,
                ml.bb_pct_b,
                ml.bb_upper,
                ml.bb_lower,
                ml.ema_20_50_cross,
                ml.ema_50_200_cross,
                ml.atr_14,
                ml.volume,
                ml.conf_score,
                ml.geo_score,
                ml.macro_score,
                ml.fundamental_score,
                ml.pe_ratio,
                ml.eps,
                ml.roe,
                ml.npl_pct,
                ml.sector_mult,
                ml.cstar_signal,
                ml.candle_pattern,
                ml.market_state,
                ml.gemini_reason,
                ml.gemini_risk,
                ml.outcome,
                -- cross-reference trade_journal for final outcome
                tj.result          AS tj_result,
                tj.return_pct      AS tj_return_pct,
                tj.pnl_npr         AS tj_pnl_npr,
                tj.hold_days_actual AS tj_hold_days_actual,
                tj.exit_date       AS tj_exit_date,
                tj.exit_reason     AS tj_exit_reason,
                tj.loss_cause      AS tj_loss_cause,
                tj.alpha_vs_nepse  AS tj_alpha
            FROM market_log ml
            LEFT JOIN trade_journal tj
                ON tj.symbol = ml.symbol
               AND tj.entry_date::date = ml.date::date
            WHERE ml.action = 'BUY'
            ORDER BY ml.date DESC
            LIMIT 80
            """
        ) or []
        rows.reverse()   # chronological for GPT
        log.info("Loaded %d BUY decision rows from market_log", len(rows))
        return rows
    except Exception as e:
        log.error("market_log BUY load failed: %s", e)
        return []


def _load_daily_context() -> list[dict]:
    """Load last MAX_DAILY_CONTEXT days of daily_context_log."""
    try:
        cutoff = (datetime.now(NST) - timedelta(days=MAX_DAILY_CONTEXT)).strftime("%Y-%m-%d")
        rows = run_raw_sql(
            """
            SELECT date, geo_score_eod, nepal_score_eod, combined_score_eod,
                   nepse_index_eod, nepse_change_pct, dxy_value, dxy_change_pct,
                   market_state, advancing, declining, breadth_score,
                   policy_rate, fd_rate_pct, lending_rate, bop_status,
                   overall_sentiment, key_events_summary, nepal_pulse_highlights,
                   geo_summary, nrb_macro_summary, signals_summary,
                   buy_count, wait_count, avoid_count
            FROM daily_context_log
            WHERE date >= %s
            ORDER BY date ASC
            """,
            (cutoff,),
        ) or []
        log.info("Loaded %d daily_context_log rows (last %d days)", len(rows), MAX_DAILY_CONTEXT)
        return rows
    except Exception as e:
        log.error("daily_context_log load failed: %s", e)
        return []


def _load_active_lessons() -> list[dict]:
    """All currently active lessons from learning_hub."""
    try:
        rows = run_raw_sql(
            """
            SELECT * FROM learning_hub
            WHERE active = 'true'
            ORDER BY created_at ASC
            """
        ) or []
        log.info("Loaded %d active lessons", len(rows))
        return rows
    except Exception as e:
        log.error("learning_hub load failed: %s", e)
        return []


def _load_nrb_macro() -> dict | None:
    """Latest nrb_monthly row for macro context."""
    try:
        rows = run_raw_sql(
            """
            SELECT * FROM nrb_monthly
            ORDER BY fiscal_year DESC, month_number DESC
            LIMIT 1
            """
        )
        return rows[0] if rows else None
    except Exception as e:
        log.warning("nrb_monthly load failed: %s", e)
        return None


def _load_gate_miss_summary() -> dict:
    """Gate miss outcomes summary for GPT. Calls gate_miss_tracker."""
    try:
        from analysis.gate_miss_tracker import get_summary_for_gpt
        return get_summary_for_gpt(days=90)
    except Exception as e:
        log.warning("gate_miss_summary load failed: %s", e)
        return {}


def _load_macro_trend() -> list[dict]:
    """Last 6 months of NRB data  -  trend matters more than snapshot."""
    try:
        rows = run_raw_sql(
            """
            SELECT period, fiscal_year, month_number, policy_rate, bank_rate,
                   cpi_inflation, credit_growth_rate, remittance_yoy_change_pct,
                   fx_reserve_months, bop_overall_balance_usd_m, bop_status,
                   bop_trend, overall_sentiment, forward_guidance, key_risks
            FROM nrb_monthly
            WHERE is_annual = 'false'
            ORDER BY fiscal_year DESC, month_number::int DESC
            LIMIT 6
            """
        ) or []
        rows.reverse()   # chronological order for GPT
        return rows
    except Exception as e:
        log.warning("macro_trend load failed: %s", e)
        return []


def _load_fd_trend() -> list[dict]:
    """Last 6 months of FD rate summaries."""
    try:
        rows = run_raw_sql(
            """
            SELECT fetch_date, avg_rate_pct, benchmark_rate_pct,
                   rate_direction, fd_score_signal, best_bank_name, best_bank_rate
            FROM fd_rate_summary
            ORDER BY fetch_date DESC
            LIMIT 6
            """
        ) or []
        rows.reverse()
        return rows
    except Exception as e:
        log.warning("fd_trend load failed: %s", e)
        return []


def _load_nepse_trend() -> list[dict]:
    """Last 30 trading days of NEPSE composite index."""
    try:
        rows = run_raw_sql(
            """
            SELECT date, current_value, change_pct
            FROM nepse_indices
            WHERE index_id = '58' AND current_value IS NOT NULL
            ORDER BY date DESC
            LIMIT 30
            """
        ) or []
        rows.reverse()
        return rows
    except Exception as e:
        log.warning("nepse_trend load failed: %s", e)
        return []


def _load_backtest_results() -> list[dict]:
    """Signal performance from backtester  -  confirmed edges."""
    try:
        return run_raw_sql(
            """
            SELECT test_name, sim_mode, win_rate_pct, profit_factor,
                   annual_ret_pct, total_trades, signal_breakdown,
                   period_start, period_end, notes
            FROM backtest_results
            ORDER BY date_run DESC
            LIMIT 10
            """
        ) or []
    except Exception as e:
        log.warning("backtest_results load failed: %s", e)
        return []


def _load_claude_audit_history() -> list[dict]:
    """
    Load last 12 weeks of claude_audit rows so GPT can track accuracy trends
    over time (e.g. declining avoid_accuracy, improving buy_win_rate).
    Ordered oldest-first so GPT reads trend direction naturally.
    """
    try:
        rows = run_raw_sql(
            """
            SELECT review_week, buy_count, buy_win_rate, buy_avg_return,
                   wait_count, wait_accuracy, avoid_count, avoid_accuracy,
                   false_avoid_rate, missed_entry_rate, overall_accuracy,
                   macro_accuracy, audit_summary
            FROM claude_audit
            ORDER BY review_week DESC
            LIMIT 12
            """
        ) or []
        rows.reverse()   # oldest first → GPT reads trend direction naturally
        return rows
    except Exception as e:
        log.warning("claude_audit_history load failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE REVIEW GUARD
# ─────────────────────────────────────────────────────────────────────────────


def _check_review_already_done(review_week: str) -> bool:
    """Check if a review for this week already produced lessons."""
    try:
        rows = run_raw_sql(
            """
            SELECT COUNT(*) as cnt FROM learning_hub
            WHERE review_week = %s AND source = 'gpt_weekly'
            """,
            (review_week,),
        )
        cnt = int(rows[0].get("cnt", 0)) if rows else 0
        return cnt > 0
    except Exception:
        return False  # fail open  -  better to potentially duplicate than skip


# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZERS  -  token-optimized JSON for GPT
# ─────────────────────────────────────────────────────────────────────────────

_TRADE_KEYS = (
    "symbol", "sector", "entry_date", "exit_date", "result", "return_pct",
    "pnl_npr", "hold_days_actual", "primary_signal", "confidence_at_entry",
    "exit_reason", "loss_cause", "alpha_vs_nepse", "geo_delta", "nepal_delta",
    "geo_score_entry", "nepal_score_entry", "rsi_entry", "bb_signal_entry",
    "macd_hist_entry", "ema_trend_entry", "obv_trend_entry",
    "market_state_entry", "market_state_exit", "volume_ratio_entry", "paper_mode",
)

_WAIT_AVOID_KEYS = (
    "date", "symbol", "sector", "action", "confidence", "outcome",
    "eval_price_change_pct", "eval_nepse_change_pct", "eval_alpha",
    "geo_score", "macro_score", "eval_geo_delta", "eval_nepal_delta",
    "eval_market_state", "eval_key_news", "reasoning",
)

_DAILY_CONTEXT_KEYS = (
    "date", "geo_score_eod", "nepal_score_eod", "combined_score_eod",
    "nepse_index_eod", "nepse_change_pct", "dxy_value", "dxy_change_pct",
    "market_state", "advancing", "declining", "fd_rate_pct", "policy_rate",
    "bop_status", "overall_sentiment", "signals_summary", "key_events_summary",
    "geo_summary","gate_miss_count", "gate_top_category", "gate_false_block_pct",
"signals_avg_confidence",
)

_LESSON_KEYS = (
    "id", "lesson_type", "source", "symbol", "sector", "applies_to",
    "condition", "finding", "action", "confidence_level",
    "trade_count", "win_rate", "last_validated",
)

_GATE_MISS_KEYS = (
    "date", "gate_category", "gate_reason", "symbol", "sector",
    "market_state", "outcome", "outcome_return_pct", "tracking_days",
)

_MACRO_TREND_KEYS = (
    "period", "policy_rate", "bank_rate", "cpi_inflation",
    "credit_growth_rate", "bop_status", "bop_trend",
    "overall_sentiment", "forward_guidance",
)

_FD_TREND_KEYS = (
    "fetch_date", "benchmark_rate_pct", "rate_direction", "fd_score_signal",
)

_NEPSE_TREND_KEYS = (
    "date", "current_value", "change_pct",
)

_CLAUDE_AUDIT_KEYS = (
    "review_week", "buy_count", "buy_win_rate", "buy_avg_return",
    "wait_count", "wait_accuracy", "avoid_count", "avoid_accuracy",
    "false_avoid_rate", "missed_entry_rate", "overall_accuracy",
    "macro_accuracy", "audit_summary",
)

_BUY_DECISION_KEYS = (
    "id", "date", "symbol", "sector", "confidence", "reasoning",
    "entry_price", "stop_loss", "target", "allocation_npr", "shares",
    "primary_signal", "rsi_14", "macd_histogram", "bb_pct_b",
    "ema_20_50_cross", "conf_score", "geo_score", "macro_score",
    "fundamental_score", "pe_ratio", "eps", "roe", "npl_pct",
    "sector_mult", "cstar_signal", "candle_pattern", "market_state",
    "gemini_reason", "gemini_risk", "outcome",
    "tj_result", "tj_return_pct", "tj_pnl_npr", "tj_hold_days_actual",
    "tj_exit_date", "tj_exit_reason", "tj_loss_cause", "tj_alpha",
)


def _serialize_compact(rows: list[dict], keys: tuple) -> str:
    """Compact JSON rows keeping only specified keys. Strips nulls."""
    if not rows:
        return "No data yet."
    lines = []
    for r in rows:
        compact = {k: r.get(k) for k in keys if r.get(k) is not None}
        if compact:
            lines.append(json.dumps(compact, ensure_ascii=False))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS (unchanged from here onwards)
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """You are the NEPSE AI Engine's weekly learning coach  -  GPT-5o.

Your job: review trading evidence and update the learning_hub  -  a database of lessons
that Claude Sonnet reads before every BUY/WAIT/AVOID decision.

NEPSE context:
- Nepal Stock Exchange trades Sun-Thu (NST). Very illiquid vs global markets.
- Fees ~1.24% round-trip. Only high profit-factor signals survive.
- BB_LOWER_TOUCH: only confirmed edge (WR=55%, PF=1.66). MACD PF=0.88 in backtest.
- RSI: context only  -  never standalone trigger (-4.81% annualized standalone).
- DXY: only validated international signal (Spearman ρ=-0.1708, 7-day lag via remittance).
- Herding threshold: RSI > 72 (not 65  -  no evidence for 65).
- Max 3 positions. Circuit breaker after 7-loss streak.
- Geo combined < -3 = auto block.

ANTI-OVERFITTING RULES  -  ENFORCE STRICTLY:
1. BLOCK_ENTRY requires 25+ trades supporting the pattern. Below 25: REDUCE_CONFIDENCE only.
2. Sector-level blocks need sector-level trade count.
3. Signal-level blocks need signal-level trade count.
4. LOW confidence = inform only. Never block on LOW.
5. Single trade cannot flip HIGH confidence lesson. Need 5+ contradictions.
6. EXPIRED outcomes = weak evidence  -  weight 20% of CORRECT/FALSE outcomes.
7. Research paper seeds have HIGH base confidence  -  need strong contradicting live evidence.
GATE PROPOSAL RULES:
- Only propose threshold changes if gate_misses sample >= 20 for that category
- Only propose if false_block_rate > 40% (filter too aggressive)
- Proposals must be conservative: ±5 on score thresholds maximum
- Never propose removing a gate entirely  -  only loosening
- If false_block_rate < 20%: gate is working well, do not touch
- Macro proposals: only if 3+ consecutive months of consistent directional change
- For SIDEWAYS vs BULL market: thresholds may legitimately differ  -  note in reasoning

LESSON TYPES:
SIGNAL_FILTER | SECTOR_FILTER | MACRO_FILTER | ENTRY_TIMING | STOP_CALC | PORTFOLIO_RULE | DIVIDEND_PATTERN | CALENDAR_EFFECT | FAILURE_MODE

ACTIONS (escalating strength):
MONITOR | REDUCE_CONFIDENCE_BY_15 | REDUCE_CONFIDENCE_BY_25 | REQUIRE_VOLUME_CONFIRM | REQUIRE_MACRO_STABLE | WAIT_FOR_CONFIRMATION | TIGHTEN_STOP | BLOCK_ENTRY

LESSON SCHEMA (each lesson you write):
{
  "lesson_type": "<type>",
  "source": "gpt_weekly",
  "symbol": "<symbol or MARKET>",
  "sector": "<sector or ALL>",
  "applies_to": "<ALL | sector-specific | symbol-specific>",
  "condition": "<machine-readable trigger>",
  "finding": "<human-readable observation>",
  "action": "<from actions list>",
  "trade_count": "<number of trades>",
  "win_count": "<wins>",
  "loss_count": "<losses>",
  "win_rate": "<percentage>",
  "avg_return_pct": "<average return>",
  "confidence_level": "<LOW | MEDIUM | HIGH>",
  "loss_cause_primary": "<if applicable>",
  "geo_delta_avg": "<avg geo delta>",
  "nepal_delta_avg": "<avg nepal delta>",
  "alpha_vs_nepse_avg": "<avg alpha>",
  "gpt_reasoning": "<your explanation>",
  "supersedes_lesson_id": "<old lesson id or null>",
  "trade_journal_ids": "<comma-separated ids>",
  "market_log_ids": "<comma-separated ids>"
}

SUPERSEDE LOGIC:
- Do NOT modify existing lessons. Instead:
  1. Mark old lesson id in lessons_to_deactivate.
  2. Write NEW lesson with supersedes_lesson_id = old id.
- If lesson is still valid  -  leave it alone.
- If evidence too thin  -  write nothing.

OUTPUT FORMAT  -  EXACTLY this JSON. No markdown, no extra text:
{
  "review_summary": "<2-3 sentence overview>",
  "lessons_to_deactivate": [<list of integer ids>],
  "lessons_to_write": [<list of lesson objects>]
  "gate_proposals": [
    {
      "proposal_number": 1,
      "parameter_name": "MIN_CONF_SCORE or TECH_SCORE_THRESHOLDS.SIDEWAYS etc",
      "current_value": "50",
      "proposed_value": "45",
      "reasoning": "evidence-based explanation",
      "false_block_rate": 0.43,
      "sample_size": 31
    }
  ],
  "claude_audit": {
    "buy_count": number,
    "buy_win_rate": number (0-1),
    "buy_avg_return": number,
    "wait_count": number,
    "wait_accuracy": number (0-1),
    "avoid_count": number,
    "avoid_accuracy": number (0-1),
    "false_avoid_rate": number (0-1),
    "missed_entry_rate": number (0-1),
    "overall_accuracy": number (0-1),
    "macro_accuracy": "qualitative: did macro trend calls match outcomes",
    "audit_summary": "2-3 sentence assessment of Claude decision quality"
  }
}
"""

def _build_user_prompt(
    review_week: str,
    trades: list[dict],
    trade_agg: dict,
    wait_avoid: list[dict],
    buy_decisions: list[dict],
    daily_context: list[dict],
    active_lessons: list[dict],
    nrb: dict | None,
    gate_summary: dict,
    macro_trend: list[dict],
    fd_trend: list[dict],
    nepse_trend: list[dict],
    backtest: list[dict],
    claude_audit_history: list[dict],
) -> str:

    nrb_str = json.dumps(
        {k: v for k, v in nrb.items() if v is not None},
        ensure_ascii=False,
    ) if nrb else "No NRB data available"

    # Aggregate stats  -  all time (not just windowed)
    total_all  = int(trade_agg.get("total", 0) or 0)
    wins_all   = int(trade_agg.get("wins", 0) or 0)
    losses_all = int(trade_agg.get("losses", 0) or 0)

    # Windowed stats
    wins_w   = sum(1 for t in trades if t.get("result") == "WIN")
    losses_w = sum(1 for t in trades if t.get("result") == "LOSS")

    correct_avoids = sum(1 for r in wait_avoid if r.get("outcome") == "CORRECT_AVOID")
    false_avoids   = sum(1 for r in wait_avoid if r.get("outcome") == "FALSE_AVOID")
    missed_entries = sum(1 for r in wait_avoid if r.get("outcome") == "MISSED_ENTRY")
    correct_waits  = sum(1 for r in wait_avoid if r.get("outcome") == "CORRECT_WAIT")

    # BUY decision stats
    buys_with_outcome  = sum(1 for b in buy_decisions if b.get("tj_result"))
    buys_open          = sum(1 for b in buy_decisions if not b.get("tj_result") and b.get("outcome") not in ("WIN", "LOSS"))
    buys_not_traded    = sum(1 for b in buy_decisions if not b.get("tj_result") and b.get("outcome") in (None, "", "PENDING"))

    return f"""WEEKLY LEARNING REVIEW  -  Week {review_week}
Review date: {datetime.now(NST).strftime("%Y-%m-%d %H:%M NST")}

=== EVIDENCE SUMMARY ===
ALL-TIME trades: {total_all} ({wins_all} W, {losses_all} L)
Recent trades shown below: {len(trades)} ({wins_w} W, {losses_w} L)
BUY decisions shown: {len(buy_decisions)} ({buys_with_outcome} closed, {buys_open} open, {buys_not_traded} not traded)
Evaluated WAIT/AVOID shown: {len(wait_avoid)}
  CORRECT_AVOID: {correct_avoids} | FALSE_AVOID: {false_avoids}
  MISSED_ENTRY: {missed_entries} | CORRECT_WAIT: {correct_waits}
Daily context rows: {len(daily_context)} trading days
Active lessons: {len(active_lessons)}

=== NRB MACRO CONTEXT ===
{nrb_str}

=== MACRO TREND (last 6 NRB months  -  direction matters) ===
{_serialize_compact(macro_trend, _MACRO_TREND_KEYS)}

=== FD RATE TREND (last 6 months) ===
{_serialize_compact(fd_trend, _FD_TREND_KEYS)}

=== NEPSE INDEX TREND (last 30 trading days) ===
{_serialize_compact(nepse_trend, _NEPSE_TREND_KEYS)}

=== BACKTEST SIGNAL PERFORMANCE ===
{_serialize_compact(backtest, ('test_name','sim_mode','win_rate_pct','profit_factor','signal_breakdown'))}

=== GATE MISS ANALYSIS (last 90 days) ===
Total misses tracked: {gate_summary.get('total_misses', 'N/A')}
Total stamped: {gate_summary.get('total_stamped', 'N/A')}

By category:
{json.dumps(gate_summary.get('by_category', {}), ensure_ascii=False, indent=2)}

By market state:
{json.dumps(gate_summary.get('by_market_state', {}), ensure_ascii=False, indent=2)}

Worst category: {gate_summary.get('worst_category', 'N/A')} (false block rate: {gate_summary.get('worst_false_block_rate', 'N/A')})

=== DAILY CONTEXT LOG (last {MAX_DAILY_CONTEXT} days  -  geo, nepal, NEPSE, signals, events) ===
{_serialize_compact(daily_context, _DAILY_CONTEXT_KEYS)}

=== RECENT TRADES (trade_journal  -  BUY outcomes with causal attribution) ===
{_serialize_compact(trades, _TRADE_KEYS)}

=== BUY DECISIONS (market_log  -  full signal context + trade_journal cross-ref) ===
Rows with tj_result = WIN/LOSS are closed trades. Rows with tj_result empty are
either open positions or signals the user chose not to trade. Use these to review
Claude's full reasoning, confidence calibration, and whether signals matched outcomes.
{_serialize_compact(buy_decisions, _BUY_DECISION_KEYS)}

=== EVALUATED WAIT/AVOID OUTCOMES (recommendation_tracker stamped) ===
{_serialize_compact(wait_avoid, _WAIT_AVOID_KEYS)}

=== ACTIVE LESSONS (what Claude reads before every decision) ===
Review each: still valid? needs strengthening/weakening? supersede?
{_serialize_compact(active_lessons, _LESSON_KEYS)}

=== CLAUDE ACCURACY AUDIT  -  LAST {len(claude_audit_history)} WEEKS ===
Use this to spot trends: is BUY accuracy improving or degrading? Is false_avoid_rate rising?
Are macro calls consistently off? Trend direction matters more than any single week.
{_serialize_compact(claude_audit_history, _CLAUDE_AUDIT_KEYS) if claude_audit_history else "No prior audit history yet  -  this is the first review."}

=== YOUR TASK ===
1. Analyse trade + WAIT/AVOID outcomes + daily context.
2. Find patterns: which signals work, which fail, in which conditions/sectors.
3. Cross-reference: did macro explain the outcome? Alpha near 0 = macro caused it.
4. Review each active lesson  -  still valid, strengthen, or supersede?
5. Write new lessons only with sufficient evidence (anti-overfitting rules).
6. Output JSON as specified. No markdown."""

# ─────────────────────────────────────────────────────────────────────────────
# GPT OUTPUT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_LESSON_FIELDS = {"lesson_type", "condition", "finding", "action", "confidence_level"}


def _validate_lesson(lesson: dict, index: int) -> bool:
    """Validate a single lesson object from GPT output. Returns True if valid."""
    missing = _REQUIRED_LESSON_FIELDS - set(lesson.keys())
    if missing:
        log.warning("Lesson #%d missing required fields: %s  -  skipping", index, missing)
        return False

    # Validate confidence_level
    if lesson.get("confidence_level") not in ("LOW", "MEDIUM", "HIGH"):
        log.warning("Lesson #%d has invalid confidence_level: %s  -  skipping",
                     index, lesson.get("confidence_level"))
        return False

    # Validate action
    valid_actions = {
        "MONITOR", "REDUCE_CONFIDENCE_BY_15", "REDUCE_CONFIDENCE_BY_25",
        "REQUIRE_VOLUME_CONFIRM", "REQUIRE_MACRO_STABLE",
        "WAIT_FOR_CONFIRMATION", "TIGHTEN_STOP", "BLOCK_ENTRY",
    }
    if lesson.get("action") not in valid_actions:
        log.warning("Lesson #%d has invalid action: %s  -  skipping",
                     index, lesson.get("action"))
        return False

    # Anti-overfitting: BLOCK_ENTRY needs 25+ trades
    if lesson.get("action") == "BLOCK_ENTRY":
        try:
            tc = int(lesson.get("trade_count", 0) or 0)
            if tc < 25:
                log.warning("Lesson #%d: BLOCK_ENTRY with only %d trades  -  "
                           "downgrading to REDUCE_CONFIDENCE_BY_25", index, tc)
                lesson["action"] = "REDUCE_CONFIDENCE_BY_25"
                lesson["gpt_reasoning"] = (
                    (lesson.get("gpt_reasoning") or "") +
                    f" [AUTO-DOWNGRADED: BLOCK_ENTRY requires 25+ trades, had {tc}]"
                )
        except (ValueError, TypeError):
            pass

    return True


# ─────────────────────────────────────────────────────────────────────────────
# LESSON WRITER  -  Option B: supersede, never overwrite
# ─────────────────────────────────────────────────────────────────────────────


def _write_lessons(
    lessons_to_write: list[dict],
    lessons_to_deactivate: list[int],
    review_week: str,
    daily_context_days: int,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Deactivate old lessons, insert new ones with backward + forward pointers.
    Uses run_raw_sql with RETURNING id for reliable forward pointers.
    Returns (written_count, deactivated_count).
    """
    now_nst     = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    written     = 0
    deactivated = 0

    # Step 1: Deactivate old lessons
    for old_id in lessons_to_deactivate:
        if dry_run:
            log.info("[DRY-RUN] Would deactivate lesson id=%s", old_id)
            deactivated += 1
            continue
        try:
            upsert_row(
                "learning_hub",
                {"id": str(old_id), "active": "false"},
                conflict_columns=["id"],
            )
            log.info("Deactivated lesson id=%s", old_id)
            deactivated += 1
        except Exception as e:
            log.error("Failed to deactivate lesson id=%s: %s", old_id, e)

    # Step 2: Insert new lessons with validation
    for i, lesson in enumerate(lessons_to_write):
        if not _validate_lesson(lesson, i):
            continue

        # Build the column values
        columns = {
            "created_at":           now_nst,
            "lesson_type":          lesson.get("lesson_type"),
            "source":               lesson.get("source", "gpt_weekly"),
            "symbol":               lesson.get("symbol", "MARKET"),
            "sector":               lesson.get("sector", "ALL"),
            "applies_to":           lesson.get("applies_to", "ALL"),
            "condition":            lesson.get("condition"),
            "finding":              lesson.get("finding"),
            "action":               lesson.get("action"),
            "trade_count":          str(lesson.get("trade_count", "")),
            "win_count":            str(lesson.get("win_count", "")),
            "loss_count":           str(lesson.get("loss_count", "")),
            "win_rate":             str(lesson.get("win_rate", "")),
            "avg_return_pct":       str(lesson.get("avg_return_pct", "")),
            "confidence_level":     lesson.get("confidence_level", "LOW"),
            "loss_cause_primary":   lesson.get("loss_cause_primary"),
            "geo_delta_avg":        str(lesson.get("geo_delta_avg", "")),
            "nepal_delta_avg":      str(lesson.get("nepal_delta_avg", "")),
            "alpha_vs_nepse_avg":   str(lesson.get("alpha_vs_nepse_avg", "")),
            "active":               "true",
            "superseded_by":        None,
            "supersedes_lesson_id": str(lesson["supersedes_lesson_id"])
                                     if lesson.get("supersedes_lesson_id") else None,
            "review_week":          review_week,
            "evidence_window_days": str(daily_context_days),
            "market_log_ids":       lesson.get("market_log_ids"),
            "gpt_reasoning":        lesson.get("gpt_reasoning"),
            "last_validated":       now_nst[:10],
            "validation_count":     "1",
            "trade_journal_ids":    lesson.get("trade_journal_ids"),
        }

        if dry_run:
            log.info("[DRY-RUN] Would write lesson: %s | %s | %s | %s",
                     columns.get("lesson_type"), columns.get("sector"),
                     columns.get("action"), columns.get("confidence_level"))
            log.info("  condition: %s", columns.get("condition"))
            log.info("  finding:   %s", columns.get("finding"))
            written += 1
            continue

        try:
            # Use run_raw_sql with RETURNING id for reliable forward pointer
            valid_cols = {k: v for k, v in columns.items() if v is not None}
            col_names  = list(valid_cols.keys())
            col_sql    = ", ".join(f'"{c}"' for c in col_names)
            val_sql    = ", ".join("%s" for _ in col_names)
            values     = [str(v) if v is not None else None for v in valid_cols.values()]

            new_rows = run_raw_sql(
                f'INSERT INTO learning_hub ({col_sql}) VALUES ({val_sql}) RETURNING id',
                tuple(values),
            )

            new_id = new_rows[0]["id"] if new_rows else None

            if new_id:
                written += 1

                # Set forward pointer on old lesson if superseding
                old_id = lesson.get("supersedes_lesson_id")
                if old_id:
                    try:
                        upsert_row(
                            "learning_hub",
                            {"id": str(old_id), "superseded_by": str(new_id)},
                            conflict_columns=["id"],
                        )
                        log.info("Linked: old lesson %s → superseded_by %s", old_id, new_id)
                    except Exception as e:
                        log.warning("Failed to link superseded_by: %s", e)

                log.info("Wrote lesson id=%s: %s | %s | %s | %s",
                         new_id, columns.get("lesson_type"), columns.get("sector"),
                         columns.get("action"), columns.get("confidence_level"))
            else:
                log.error("INSERT returned no id for lesson: %s", columns.get("finding"))

        except Exception as e:
            log.error("Failed to write lesson: %s | error: %s", columns.get("finding"), e)

    return written, deactivated


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────


def _send_telegram_summary(
    review_week: str,
    review_summary: str,
    written: int,
    deactivated: int,
    total_trades: int,
    total_wait_avoid: int,
    gate_proposals: list = None,
) -> None:
    """Send weekly review summary to Telegram."""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id   = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        log.warning("Telegram not configured  -  skipping notification")
        return

    gate_line = (
        f"🔧 Gate proposals: {len(gate_proposals)} pending  -  /gate_review to see them\n"
        if gate_proposals else ""
    )

    msg = (
        f"🧠 *Learning Hub  -  Week {review_week}*\n\n"
        f"{review_summary}\n\n"
        f"📊 Evidence reviewed:\n"
        f"  • Completed trades: {total_trades}\n"
        f"  • WAIT/AVOID outcomes: {total_wait_avoid}\n\n"
        f"✏️ Lessons written: {written}\n"
        f"🔄 Lessons superseded: {deactivated}\n\n"
        f"{gate_line}"
        f"_Claude will apply updated lessons from next signal cycle._"
    )

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={
            "chat_id":    chat_id,
            "text":       msg,
            "parse_mode": "Markdown",
        }, timeout=10)
        log.info("Telegram summary sent")
    except Exception as e:
        log.warning("Telegram send failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS REPORTER
# ─────────────────────────────────────────────────────────────────────────────


def print_status():
    """Show current learning_hub summary."""
    try:
        all_lessons = run_raw_sql(
            "SELECT * FROM learning_hub ORDER BY active DESC, created_at DESC"
        ) or []
    except Exception as e:
        log.error("Status query failed: %s", e)
        return

    active   = [l for l in all_lessons if l.get("active") == "true"]
    inactive = [l for l in all_lessons if l.get("active") != "true"]

    print(f"\n{'=' * 70}")
    print(f"LEARNING HUB STATUS  -  {datetime.now(NST).strftime('%Y-%m-%d %H:%M NST')}")
    print(f"{'=' * 70}")
    print(f"\nActive lessons: {len(active)} | Superseded: {len(inactive)}")

    if active:
        print(f"\n  {'Type':<20} {'Sector':<20} {'Action':<30} {'Conf':<8} Source")
        print(f"  {'-' * 85}")
        for l in active:
            print(
                f"  {(l.get('lesson_type') or ''):.<20} "
                f"{(l.get('sector') or ''):.<20} "
                f"{(l.get('action') or ''):.<30} "
                f"{(l.get('confidence_level') or ''):.<8} "
                f"{l.get('source') or ''}"
            )

    from collections import Counter
    type_counts = Counter(l.get("lesson_type") for l in active)
    print(f"\n  By type:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:<30} {c}")

    src_counts = Counter(l.get("source") for l in active)
    print(f"\n  By source:")
    for s, c in sorted(src_counts.items()):
        print(f"    {s:<30} {c}")

    # Show recent reviews
    try:
        recent_weeks = run_raw_sql(
            """
            SELECT review_week, COUNT(*) as cnt
            FROM learning_hub
            WHERE source = 'gpt_weekly' AND review_week IS NOT NULL
            GROUP BY review_week
            ORDER BY review_week DESC
            LIMIT 5
            """
        ) or []
        if recent_weeks:
            print(f"\n  Recent reviews:")
            for rw in recent_weeks:
                print(f"    {rw.get('review_week', '?')}: {rw.get('cnt', 0)} lessons")
    except Exception:
        pass

    print()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT VIEWER (for --prompt mode and prompt_viewer CLI)
# ─────────────────────────────────────────────────────────────────────────────


def get_review_prompts() -> tuple[str, str]:
    """
    Build and return (system_prompt, user_prompt) without calling GPT.
    Used by --prompt mode and prompt_viewer CLI.
    """
    now     = datetime.now(NST)
    iso_cal = now.isocalendar()
    review_week = f"{iso_cal.year}-W{iso_cal.week:02d}"

    trades, trade_agg = _load_trade_journal()
    wait_avoid    = _load_wait_avoid_outcomes()
    buy_decisions = _load_buy_decisions()
    daily_context = _load_daily_context()
    active_lessons = _load_active_lessons()
    nrb           = _load_nrb_macro()
    gate_summary  = _load_gate_miss_summary()
    macro_trend   = _load_macro_trend()
    fd_trend      = _load_fd_trend()
    nepse_trend   = _load_nepse_trend()
    backtest      = _load_backtest_results()
    claude_audit_history = _load_claude_audit_history()

    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(
        review_week, trades, trade_agg, wait_avoid, buy_decisions, daily_context,
        active_lessons, nrb, gate_summary, macro_trend, fd_trend, nepse_trend,
        backtest, claude_audit_history,
    )

    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REVIEW RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _write_gate_proposals(proposals: list[dict], review_week: str) -> int:
    """Write GPT gate proposals to gate_proposals table."""
    from datetime import datetime
    now = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    written = 0
    for p in proposals:
        try:
            write_row("gate_proposals", {
                "review_week":     review_week,
                "proposal_number": str(p.get("proposal_number", "")),
                "parameter_name":  p.get("parameter_name", ""),
                "current_value":   str(p.get("current_value", "")),
                "proposed_value":  str(p.get("proposed_value", "")),
                "reasoning":       p.get("reasoning", ""),
                "false_block_rate":str(p.get("false_block_rate", "")),
                "sample_size":     str(p.get("sample_size", "")),
                "status":          "PENDING",
                "inserted_at":     now,
            })
            written += 1
        except Exception as e:
            log.error("Failed to write gate proposal: %s", e)
    log.info("Gate proposals written: %d", written)
    return written


def _write_claude_audit(audit: dict, review_week: str) -> bool:
    """Write Claude accuracy audit to claude_audit table."""
    from datetime import datetime
    now = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")
    try:
        upsert_row("claude_audit", {
            "review_week":       review_week,
            "buy_count":         str(audit.get("buy_count", "")),
            "buy_win_rate":      str(audit.get("buy_win_rate", "")),
            "buy_avg_return":    str(audit.get("buy_avg_return", "")),
            "wait_count":        str(audit.get("wait_count", "")),
            "wait_accuracy":     str(audit.get("wait_accuracy", "")),
            "avoid_count":       str(audit.get("avoid_count", "")),
            "avoid_accuracy":    str(audit.get("avoid_accuracy", "")),
            "false_avoid_rate":  str(audit.get("false_avoid_rate", "")),
            "missed_entry_rate": str(audit.get("missed_entry_rate", "")),
            "overall_accuracy":  str(audit.get("overall_accuracy", "")),
            "macro_accuracy":    audit.get("macro_accuracy", ""),
            "audit_summary":     audit.get("audit_summary", ""),
            "inserted_at":       now,
        }, conflict_columns=["review_week"])
        return True
    except Exception as e:
        log.error("Failed to write claude_audit: %s", e)
        return False


def run_weekly_review(dry_run: bool = False):
    """Full GPT Sunday review cycle."""

    now     = datetime.now(NST)
    iso_cal = now.isocalendar()
    review_week = f"{iso_cal.year}-W{iso_cal.week:02d}"

    log.info("Starting weekly learning review  -  %s (dry_run=%s)", review_week, dry_run)

    # ── Duplicate guard
    if not dry_run and _check_review_already_done(review_week):
        log.warning("Review for %s already exists in learning_hub  -  aborting to prevent duplicates",
                     review_week)
        log.warning("Use --dry-run to preview, or manually delete existing rows to re-run")
        return None

    # -- Load all data
    trades, trade_agg = _load_trade_journal()
    wait_avoid    = _load_wait_avoid_outcomes()
    buy_decisions = _load_buy_decisions()
    daily_context = _load_daily_context()
    active_lessons = _load_active_lessons()
    nrb           = _load_nrb_macro()
    gate_summary  = _load_gate_miss_summary()
    macro_trend   = _load_macro_trend()
    fd_trend      = _load_fd_trend()
    nepse_trend   = _load_nepse_trend()
    backtest      = _load_backtest_results()
    claude_audit_history = _load_claude_audit_history()

    # -- Build prompts
    system_prompt = _build_system_prompt()
    user_prompt   = _build_user_prompt(
        review_week, trades, trade_agg, wait_avoid, buy_decisions, daily_context,
        active_lessons, nrb, gate_summary, macro_trend, fd_trend, nepse_trend,
        backtest, claude_audit_history,
    )

    log.info("Calling GPT-5o for weekly review (prompt ~%d tokens)...",
             (len(system_prompt) + len(user_prompt)) // 4)
    raw_response = _call_gpt(system_prompt, user_prompt, max_tokens=MAX_GPT_TOKENS)

    if not raw_response:
        log.error("GPT returned empty response  -  aborting review")
        return None

    # ── Parse GPT response
    try:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])
        gpt_output = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error("GPT JSON parse failed: %s", e)
        log.error("Raw (first 500 chars): %s", raw_response[:500])
        return None

    review_summary   = gpt_output.get("review_summary", "")
    lessons_to_write = gpt_output.get("lessons_to_write", [])
    lessons_to_deact = gpt_output.get("lessons_to_deactivate", [])

    log.info("GPT review complete:")
    log.info("  Summary: %s", review_summary)
    log.info("  Lessons to write: %d", len(lessons_to_write))
    log.info("  Lessons to deactivate: %d", len(lessons_to_deact))

    # ── Write to DB
    written, deactivated = _write_lessons(
        lessons_to_write,
        lessons_to_deact,
        review_week,
        len(daily_context),
        dry_run=dry_run,
    )

    # Write gate proposals
    gate_proposals = gpt_output.get("gate_proposals", [])
    if gate_proposals and not dry_run:
        _write_gate_proposals(gate_proposals, review_week)

    # Write claude audit
    claude_audit = gpt_output.get("claude_audit", {})
    if claude_audit and not dry_run:
        _write_claude_audit(claude_audit, review_week)

    log.info("Review complete: %d lessons written, %d deactivated", written, deactivated)

    # ── Telegram notification
    if not dry_run:
        _send_telegram_summary(
            review_week, review_summary,
            written, deactivated,
            int(trade_agg.get("total", 0) or 0), len(wait_avoid),
            gate_proposals=gate_proposals,
        )

    return {
        "review_week":         review_week,
        "review_summary":      review_summary,
        "written":             written,
        "deactivated":         deactivated,
        "trades_reviewed":     len(trades),
        "wait_avoid_reviewed": len(wait_avoid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NEPSE Learning Hub  -  GPT Weekly Review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print lessons but do not write to DB")
    parser.add_argument("--status", action="store_true",
                        help="Show current learning_hub summary")
    parser.add_argument("--prompt", action="store_true",
                        help="Print the GPT prompt and exit (no API call)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.prompt:
        system_prompt, user_prompt = get_review_prompts()
        print("=" * 70)
        print("SYSTEM PROMPT")
        print("=" * 70)
        print(system_prompt)
        print(f"\n--- System tokens: ~{len(system_prompt) // 4} ---\n")
        print("=" * 70)
        print("USER PROMPT")
        print("=" * 70)
        print(user_prompt)
        print(f"\n--- User tokens: ~{len(user_prompt) // 4} ---")
        print(f"--- Total tokens: ~{(len(system_prompt) + len(user_prompt)) // 4} ---")
        return

    result = run_weekly_review(dry_run=args.dry_run)

    if result:
        log.info(
            "Done. Week %s | %d written | %d deactivated | %d trades | %d wait/avoid",
            result["review_week"], result["written"], result["deactivated"],
            result["trades_reviewed"], result["wait_avoid_reviewed"],
        )

    if args.dry_run:
        log.info("[DRY-RUN] No writes performed.")


if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    main()