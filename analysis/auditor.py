"""
auditor.py — NEPSE AI Engine
EOD WIN/LOSS marking + causal attribution engine.

Runs at 3:15 PM NST (via eod.yml GitHub Actions).

Responsibilities:
1. Scan portfolio table for OPEN positions with exit signals
2. Compute all delta fields (geo_delta, nepal_delta, nepse_return_pct, alpha_vs_nepse)
3. Call _attribute_loss() for every closed trade → stamp loss_cause
4. Write complete trade record to trade_journal (immutable — never updated)
5. Update financials KPI table (win_rate, avg_return, streak counters)
6. Update market_log outcomes
7. Send EOD Telegram summary

Paper trading mode (--paper):
  Reads paper_portfolio WHERE status='CLOSED' AND audited IS NULL
  Runs same causal attribution pipeline
  Writes to trade_journal with paper_mode='true'
  Stamps audited='true' on paper_portfolio row
  Links to market_log outcome by symbol + date range

Import rule: from sheets import ... — NEVER from db import ...

CLI:
  python -m analysis.auditor              → live EOD run
  python -m analysis.auditor --paper      → paper trading audit
  python -m analysis.auditor --dry-run    → compute everything, write nothing
  python -m analysis.auditor --paper --dry-run
  python -m analysis.auditor --kpis
  python -m analysis.auditor --status
  python -m analysis.auditor --market-state
  python -m analysis.auditor --attribute -2.5 -1.0 -3.0 -5.0 STOP_LOSS HYDRO
"""

import logging
import sys
from datetime import datetime, date
from typing import Optional

from sheets import get_setting, write_row, read_tab, upsert_row, update_setting, update_row

try:
    from modules.geo_sentiment import get_latest_geo_score
except ImportError:
    from modules.geo_sentiment import get_latest_geo_score

try:
    from modules.nepal_pulse import get_latest_nepal_score
except ImportError:
    from modules.nepal_pulse import get_latest_nepal_score

try:
    from helper.notifier import send_telegram
except ImportError:
    def send_telegram(msg: str):
        log.warning("Telegram notifier not available. Message: %s", msg)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("auditor")

# ── Constants ────────────────────────────────────────────────────────────────
GEO_DELTA_MACRO_THRESHOLD   = -2.0
NEPAL_DELTA_MACRO_THRESHOLD = -2.0
NEPSE_SELLOFF_THRESHOLD     = -2.0
ALPHA_FAILURE_THRESHOLD     = -3.0
STOP_RECOVERY_DAYS          =  3
ALPHA_NEAR_ZERO_BAND        =  1.5

# ── Fee constants — must match telegram_bot.py exactly ───────────────────────
SEBON_PCT   = 0.00015   # 0.015%
DP_CHARGE   = 25.0      # flat per trade side
CGT_RATE    = 0.075     # 7.5% on gross profit only (FIX 3: was 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 0. FEE HELPERS  (tiered brokerage — matches telegram_bot._brokerage exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _brokerage(trade_value: float) -> float:
    """
    NEPSE tiered brokerage. Mirrors telegram_bot.py _brokerage() exactly.
    FIX 3: replaces old flat 0.00415 rate.
    """
    if trade_value <= 2500:
        return 10.0
    elif trade_value <= 50_000:
        rate = 0.0036
    elif trade_value <= 500_000:
        rate = 0.0033
    elif trade_value <= 2_000_000:
        rate = 0.0031
    elif trade_value <= 10_000_000:
        rate = 0.0027
    else:
        rate = 0.0024
    return round(trade_value * rate, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS CAUSE ATTRIBUTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _attribute_loss(
    geo_delta: float,
    nepal_delta: float,
    nepse_return_pct: float,
    alpha_vs_nepse: float,
    exit_reason: str,
    sector: str,
    nepal_score_exit: float,
    hold_days_actual: int,
) -> str:
    """
    Core causal attribution. Returns one of:
        MACRO_DETERIORATION | MARKET_WIDE_SELLOFF | SIGNAL_FAILURE |
        ENTRY_TIMING | STOP_TOO_TIGHT | SECTOR_EVENT | NULL
    """
    # 1. STOP_TOO_TIGHT
    if exit_reason == "STOP_LOSS" and hold_days_actual <= STOP_RECOVERY_DAYS:
        return "STOP_TOO_TIGHT"

    # 2. SECTOR_EVENT
    if nepal_delta <= -3.0 and hold_days_actual <= 5:
        return "SECTOR_EVENT"

    # 3. MACRO_DETERIORATION
    macro_deteriorated = (geo_delta <= GEO_DELTA_MACRO_THRESHOLD or
                          nepal_delta <= NEPAL_DELTA_MACRO_THRESHOLD)
    alpha_near_zero = abs(alpha_vs_nepse) <= ALPHA_NEAR_ZERO_BAND
    if macro_deteriorated and alpha_near_zero:
        return "MACRO_DETERIORATION"

    # 4. MARKET_WIDE_SELLOFF
    if nepse_return_pct <= NEPSE_SELLOFF_THRESHOLD and alpha_near_zero:
        return "MARKET_WIDE_SELLOFF"

    # 5. SIGNAL_FAILURE
    macro_stable  = (geo_delta > GEO_DELTA_MACRO_THRESHOLD and
                     nepal_delta > NEPAL_DELTA_MACRO_THRESHOLD)
    nepse_stable  = nepse_return_pct > NEPSE_SELLOFF_THRESHOLD
    alpha_bad     = alpha_vs_nepse <= ALPHA_FAILURE_THRESHOLD
    if macro_stable and nepse_stable and alpha_bad:
        return "SIGNAL_FAILURE"

    # 6. ENTRY_TIMING
    if macro_stable and nepse_stable:
        return "ENTRY_TIMING"

    return "MACRO_DETERIORATION"  # fallback — mixed signals


# ─────────────────────────────────────────────────────────────────────────────
# 2. OUTCOME CLASSIFIER & PNL
#    FIX 3: tiered brokerage + 7.5% CGT (was flat 0.00415 + 5% CGT)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_result(return_pct: float, pnl_npr: float = 0) -> str:
    if return_pct > 0.5:  return "WIN"
    if return_pct < -0.5: return "LOSS"
    return "BREAKEVEN"


def _compute_pnl_npr(entry_price: float, exit_price: float, shares: float) -> float:
    """
    Compute net P&L after all Nepal fees and CGT.
    FIX 3: tiered brokerage + 7.5% CGT — now matches telegram_bot.py calc_sell_fees() exactly.
    """
    buy_value  = entry_price * shares
    sell_value = exit_price  * shares
    gross      = sell_value - buy_value

    buy_cost   = _brokerage(buy_value)  + (buy_value  * SEBON_PCT) + DP_CHARGE
    sell_cost  = _brokerage(sell_value) + (sell_value * SEBON_PCT) + DP_CHARGE

    # CGT: 7.5% on gross profit only (never on a loss)
    cgt = max(0.0, gross * CGT_RATE)

    return round(gross - buy_cost - sell_cost - cgt, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DELTA FIELD CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deltas(position: dict, current_price: float,
                    geo_score: float, nepal_score: float) -> dict:
    """Compute all delta fields needed for trade_journal + causal attribution."""
    entry_price = float(position.get("entry_price") or 0)
    shares      = float(position.get("shares") or 0)

    geo_score_entry   = float(position.get("geo_score_entry")   or 0)
    nepal_score_entry = float(position.get("nepal_score_entry") or 0)
    nepse_entry       = float(position.get("nepse_index_entry") or 0)

    return_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0.0

    # NEPSE current value
    nepse_exit = 0.0
    try:
        rows = sorted(
            [r for r in read_tab("nepse_indices")
             if r.get("index_id") == "58" and r.get("current_value")],
            key=lambda x: x.get("date", "")
        )
        if rows:
            nepse_exit = float(str(rows[-1]["current_value"]).replace(",", ""))
    except Exception as e:
        log.warning("NEPSE index fetch failed: %s", e)

    nepse_return_pct = ((nepse_exit - nepse_entry) / nepse_entry * 100) if nepse_entry else 0.0
    alpha_vs_nepse   = return_pct - nepse_return_pct

    return {
        "return_pct":       round(return_pct, 4),
        "geo_score_exit":   geo_score,
        "nepal_score_exit": nepal_score,
        "geo_delta":        round(geo_score   - geo_score_entry,   2),
        "nepal_delta":      round(nepal_score - nepal_score_entry, 2),
        "combined_geo_delta": round(
            (geo_score + nepal_score) - (geo_score_entry + nepal_score_entry), 2
        ),
        "nepse_return_pct": round(nepse_return_pct, 4),
        "alpha_vs_nepse":   round(alpha_vs_nepse,   4),
        "nepse_index_exit": nepse_exit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRADE JOURNAL WRITER (IMMUTABLE)
# ─────────────────────────────────────────────────────────────────────────────

def _write_trade_journal(position: dict, deltas: dict,
                         exit_context: dict, paper: bool = False) -> Optional[int]:
    try:
        symbol      = position.get("symbol", "")
        entry_price = float(position.get("entry_price") or 0)
        exit_price  = float(exit_context.get("exit_price") or 0)
        shares      = float(position.get("shares") or 0)

        return_pct  = deltas.get("return_pct", 0.0)
        pnl_npr     = _compute_pnl_npr(entry_price, exit_price, shares)
        result      = _classify_result(return_pct, pnl_npr)

        loss_cause = "NULL"
        if result == "LOSS":
            hold_days  = int(position.get("hold_days_actual") or 1)
            loss_cause = _attribute_loss(
                geo_delta        = deltas["geo_delta"],
                nepal_delta      = deltas["nepal_delta"],
                nepse_return_pct = deltas["nepse_return_pct"],
                alpha_vs_nepse   = deltas["alpha_vs_nepse"],
                exit_reason      = exit_context.get("exit_reason", ""),
                sector           = position.get("sector", ""),
                nepal_score_exit = deltas["nepal_score_exit"],
                hold_days_actual = hold_days,
            )
            log.info("Loss cause for %s: %s (alpha=%.2f%%)", symbol, loss_cause, deltas["alpha_vs_nepse"])

        row = {
            "created_at":           _now_nst(),
            "symbol":               symbol,
            "sector":               position.get("sector", ""),
            "paper_mode":           "true" if paper else get_setting("PAPER_MODE", "true"),
            "entry_date":           position.get("entry_date", ""),
            "entry_price":          str(entry_price),
            "shares":               str(shares),
            "allocation_npr":       position.get("allocation_npr", ""),
            "primary_signal":       position.get("primary_signal", ""),
            "secondary_signal":     position.get("secondary_signal", "NONE"),
            "candle_pattern":       position.get("candle_pattern", "NONE"),
            "confidence_at_entry":  position.get("confidence_at_entry", ""),
            "rsi_entry":            position.get("rsi_entry", ""),
            "macd_hist_entry":      position.get("macd_hist_entry", ""),
            "bb_signal_entry":      position.get("bb_signal_entry", ""),
            "ema_trend_entry":      position.get("ema_trend_entry", ""),
            "obv_trend_entry":      position.get("obv_trend_entry", ""),
            "conf_score_entry":     position.get("conf_score_entry", ""),
            "volume_ratio_entry":   position.get("volume_ratio_entry", ""),
            "atr_pct_entry":        position.get("atr_pct_entry", ""),
            "market_state_entry":   position.get("market_state_entry", ""),
            "geo_score_entry":      position.get("geo_score_entry", ""),
            "nepal_score_entry":    position.get("nepal_score_entry", ""),
            "combined_geo_entry":   position.get("combined_geo_entry", ""),
            "nepse_index_entry":    position.get("nepse_index_entry", ""),
            "stop_loss_planned":    position.get("stop_loss_planned", ""),
            "target_planned":       position.get("target_planned", ""),
            "hold_days_planned":    position.get("hold_days_planned", ""),
            "exit_date":            exit_context.get("exit_date", ""),
            "exit_price":           str(exit_price),
            "exit_reason":          exit_context.get("exit_reason", ""),
            "market_state_exit":    exit_context.get("market_state_exit",
                                        get_setting("MARKET_STATE", "SIDEWAYS")),
            "geo_score_exit":       str(deltas["geo_score_exit"]),
            "nepal_score_exit":     str(deltas["nepal_score_exit"]),
            "combined_geo_exit":    str(deltas["geo_score_exit"] + deltas["nepal_score_exit"]),
            "nepse_index_exit":     str(deltas["nepse_index_exit"]),
            "hold_days_actual":     position.get("hold_days_actual", ""),
            "return_pct":           str(return_pct),
            "pnl_npr":              str(pnl_npr),
            "result":               result,
            "geo_delta":            str(deltas["geo_delta"]),
            "nepal_delta":          str(deltas["nepal_delta"]),
            "combined_geo_delta":   str(deltas["combined_geo_delta"]),
            "nepse_return_pct":     str(deltas["nepse_return_pct"]),
            "alpha_vs_nepse":       str(deltas["alpha_vs_nepse"]),
            "loss_cause":           loss_cause,
            "lesson_ids":           "",
        }

        trade_id = write_row("trade_journal", row)
        log.info("trade_journal written — %s | %s | return=%.2f%% | NPR %.0f | cause=%s | paper=%s",
                 symbol, result, return_pct, pnl_npr, loss_cause, paper)
        return trade_id

    except Exception as e:
        log.error("Failed to write trade_journal for %s: %s", position.get("symbol", "?"), e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. STOP LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _check_stop_conditions(position: dict, current_price: float,
                            geo_score: float) -> Optional[dict]:
    """Returns exit_context dict if should close, None to hold."""
    symbol      = position.get("symbol", "")
    entry_price = float(position.get("entry_price") or 0)
    stop_level  = float(position.get("stop_level") or (entry_price * 0.97))
    target      = float(position.get("target_planned") or 0)

    if not entry_price:
        return None

    # 1. Geo block — hard rule, no exceptions
    # FIX 2: CURRENT_NEPAL_SCORE is now always written by run_eod_audit() /
    #         run_paper_audit() before the position loop, so this reads a real value.
    nepal_score  = float(get_setting("CURRENT_NEPAL_SCORE", "0") or 0)
    combined_geo = geo_score + nepal_score
    if combined_geo < -3:
        log.warning("%s: Geo block — combined_geo=%.1f. Forced exit.", symbol, combined_geo)
        return {"exit_reason": "GEO_BLOCK", "exit_price": current_price, "exit_date": _today()}

    # 2. Hard stop loss
    if current_price <= stop_level:
        log.info("%s: Hard stop — %.2f <= %.2f", symbol, current_price, stop_level)
        return {"exit_reason": "STOP_LOSS", "exit_price": current_price, "exit_date": _today()}

    # 3. Trailing stop
    if position.get("trail_active") == "true":
        trail_stop = float(position.get("trail_stop") or 0)
        if trail_stop and current_price <= trail_stop:
            log.info("%s: Trail stop — %.2f <= %.2f", symbol, current_price, trail_stop)
            return {"exit_reason": "TRAILING_STOP", "exit_price": current_price, "exit_date": _today()}

    # 4. Target hit
    if target and current_price >= target:
        log.info("%s: Target hit — %.2f >= %.2f", symbol, current_price, target)
        return {"exit_reason": "TARGET_HIT", "exit_price": current_price, "exit_date": _today()}

    return None


def _update_trailing_stop(position: dict, current_price: float) -> Optional[dict]:
    """Activate trail at +5%, trail 3% below peak. Returns fields to update or None."""
    entry_price = float(position.get("entry_price") or 0)
    if not entry_price:
        return None

    return_pct = ((current_price - entry_price) / entry_price) * 100
    peak_price = float(position.get("peak_price") or entry_price)
    updates    = {}

    if current_price > peak_price:
        peak_price            = current_price
        updates["peak_price"] = str(peak_price)

    trail_active = position.get("trail_active", "false") == "true"
    if return_pct >= 5.0 and not trail_active:
        updates["trail_active"] = "true"
        updates["stop_type"]    = "TRAILING"
        log.info("%s: Trail activated at +%.1f%%", position.get("symbol"), return_pct)

    if trail_active or return_pct >= 5.0:
        new_trail             = round(peak_price * 0.97, 2)
        updates["trail_stop"] = str(new_trail)
        updates["stop_level"] = str(new_trail)

    return updates if updates else None


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY CONTEXT ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_position_with_entry_context(position: dict) -> dict:
    """Pull full entry snapshot from market_log BUY record."""
    symbol = position.get("symbol", "")
    try:
        rows = [r for r in read_tab("market_log")
                if r.get("symbol") == symbol and r.get("action") == "BUY"]
        rows = sorted(rows, key=lambda x: (x.get("date", ""), x.get("time", "")),
                      reverse=True)[:1]
        if rows:
            ml        = rows[0]
            field_map = {
                "rsi_14":       "rsi_entry",
                "macd_line":    "macd_hist_entry",
                "bb_signal":    "bb_signal_entry",
                "ema_trend":    "ema_trend_entry",
                "obv_trend":    "obv_trend_entry",
                "conf_score":   "conf_score_entry",
                "volume_ratio": "volume_ratio_entry",
                "atr_14":       "atr_pct_entry",
                "geo_score":    "geo_score_entry",
                "stop_loss":    "stop_loss_planned",
                "target":       "target_planned",
            }
            for src, dst in field_map.items():
                if not position.get(dst):
                    position[dst] = ml.get(src, "")
            for f in ["primary_signal", "secondary_signal", "candle_pattern"]:
                if not position.get(f):
                    position[f] = ml.get(f, "NONE")
            if not position.get("confidence_at_entry"):
                position["confidence_at_entry"] = ml.get("conf_score", "")
    except Exception as e:
        log.warning("Entry context enrichment failed for %s: %s", symbol, e)
    return position


# ─────────────────────────────────────────────────────────────────────────────
# 7. KPI UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def _update_financials_kpis():
    try:
        trades = read_tab("trade_journal")
        if not trades:
            log.info("No trades yet — skipping KPI update")
            return

        total    = len(trades)
        wins     = sum(1 for t in trades if t.get("result") == "WIN")
        losses   = sum(1 for t in trades if t.get("result") == "LOSS")
        win_rate = round((wins / total) * 100, 1) if total else 0
        returns  = [float(t.get("return_pct") or 0) for t in trades]
        pnls     = [float(t.get("pnl_npr")    or 0) for t in trades]
        avg_ret  = round(sum(returns) / len(returns), 2) if returns else 0
        avg_pnl  = round(sum(pnls)    / len(pnls),    2) if pnls    else 0
        total_pnl = round(sum(pnls), 2)

        sorted_trades = sorted(trades, key=lambda x: x.get("exit_date") or "")
        win_streak = loss_streak = 0
        for t in reversed(sorted_trades):
            r = t.get("result", "")
            if r == "WIN":
                if loss_streak > 0: break
                win_streak += 1
            elif r == "LOSS":
                if win_streak > 0: break
                loss_streak += 1
            else:
                break

        now  = _today()
        kpis = [
            ("overall_win_rate_pct", str(win_rate),   "65.0", "55.0",
             "ALERT" if win_rate < 55 else "ACTIVE"),
            ("total_trades",         str(total),       "30",   None,   "ACTIVE"),
            ("wins_total",           str(wins),        None,   None,   "ACTIVE"),
            ("losses_total",         str(losses),      None,   None,   "ACTIVE"),
            ("current_win_streak",   str(win_streak),  None,   None,   "ACTIVE"),
            ("current_loss_streak",  str(loss_streak), "7",    "7",
             "ALERT" if loss_streak >= 7 else "ACTIVE"),
            ("avg_return_pct",       str(avg_ret),     None,   None,   "ACTIVE"),
            ("avg_pnl_npr",          str(avg_pnl),     None,   None,   "ACTIVE"),
            ("total_pnl_npr",        str(total_pnl),   None,   None,   "ACTIVE"),
        ]

        for kpi_name, current_value, target_value, alert_level, status in kpis:
            upsert_row("financials", {
                "kpi_name":      kpi_name,
                "current_value": current_value,
                "target_value":  target_value or "",
                "alert_level":   alert_level  or "",
                "status":        status,
                "last_updated":  now,
                "notes":         f"Auto-updated by auditor.py on {now}",
            }, ["kpi_name"])

        log.info("KPIs — trades=%d | win_rate=%.1f%% | pnl=NPR %.0f | loss_streak=%d",
                 total, win_rate, total_pnl, loss_streak)

        if loss_streak >= 7:
            log.critical("CIRCUIT BREAKER — %d consecutive losses.", loss_streak)
            upsert_row("settings", {
                "key":          "CIRCUIT_BREAKER",
                "value":        "true",
                "description":  f"Auto-triggered by auditor.py — {loss_streak} consecutive losses",
                "last_updated": now,
                "set_by":       "auditor.py",
            }, ["key"])
            send_telegram(
                f"🚨 CIRCUIT BREAKER TRIGGERED\n"
                f"{loss_streak} consecutive losses.\n"
                f"ALL SIGNALS PAUSED. Manual review required."
            )

    except Exception as e:
        log.error("KPI update failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# 7b. MARKET STATE AUTO-UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def _update_market_state() -> str:
    """
    Auto-calculate MARKET_STATE from NEPSE SMA200 + market breadth.
    Falls back to previous day breadth if today's is missing.
    Writes result to settings table.
    """
    try:
        rows = sorted(
            [r for r in read_tab("nepse_indices")
             if r.get("current_value") and r.get("index_id") == "58"],
            key=lambda x: x.get("date", "")
        )
        if len(rows) < 30:
            log.warning("MARKET_STATE: Not enough NEPSE history (%d rows). Skipping.", len(rows))
            return get_setting("MARKET_STATE", "SIDEWAYS")

        closes = []
        for r in rows:
            val = r.get("current_value")
            if val is None:
                continue
            val_str = str(val).replace(",", "").strip()
            try:
                closes.append(float(val_str))
            except ValueError:
                log.warning("Skipping non-numeric NEPSE value: %s", val_str)
                continue

        if not closes:
            log.error("No valid NEPSE values after cleaning.")
            return get_setting("MARKET_STATE", "SIDEWAYS")

        nepse_today = closes[-1]
        sma_period  = min(200, len(closes))
        sma200      = sum(closes[-sma_period:]) / sma_period
        pct_from_sma = ((nepse_today - sma200) / sma200) * 100

        # Market breadth
        adv_ratio = 0.5
        try:
            breadth_rows = read_tab("market_breadth", limit=2)
            if breadth_rows:
                br = breadth_rows[0]
                adv = float(br.get("advancing", 0) or 0)
                dec = float(br.get("declining", 0) or 0)
                if adv + dec > 0:
                    adv_ratio = adv / (adv + dec)
        except Exception:
            pass

        if pct_from_sma >= 5 and adv_ratio >= 0.55:
            state = "FULL_BULL"
        elif pct_from_sma >= 0 and adv_ratio >= 0.5:
            state = "CAUTIOUS_BULL"
        elif pct_from_sma >= -5 and adv_ratio >= 0.4:
            state = "SIDEWAYS"
        elif pct_from_sma >= -15:
            state = "BEAR"
        else:
            state = "CRISIS"

        upsert_row("settings", {
            "key":          "MARKET_STATE",
            "value":        state,
            "description":  (f"Auto: NEPSE={nepse_today:.1f} SMA{sma_period}={sma200:.1f} "
                             f"pct={pct_from_sma:+.2f}% adv={adv_ratio:.2f}"),
            "last_updated": _today(),
            "set_by":       "auditor.py",
        }, ["key"])

        upsert_row("settings", {
            "key":          "NEPSE_200DMA",
            "value":        str(round(sma200, 2)),
            "last_updated": _today(),
            "set_by":       "auditor.py",
        }, ["key"])

        return state

    except Exception as e:
        log.exception("_update_market_state failed: %s", e)
        return get_setting("MARKET_STATE", "SIDEWAYS")


# ─────────────────────────────────────────────────────────────────────────────
# 8. EOD TELEGRAM SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _build_eod_summary(closed_trades: list, open_positions: list,
                       paper: bool = False) -> str:
    today  = _today()
    prefix = "📄 [PAPER] " if paper else ""

    if not closed_trades and not open_positions:
        return f"{prefix}📊 EOD Summary — {today}\nNo activity today."

    lines = [f"{prefix}📊 *EOD Summary — {today}*\n"]

    if closed_trades:
        lines.append(f"*Closed: {len(closed_trades)} trade(s)*")
        for t in closed_trades:
            emoji = "✅" if t["result"] == "WIN" else ("❌" if t["result"] == "LOSS" else "➖")
            cause = f" [{t['loss_cause']}]" if t["result"] == "LOSS" else ""
            lines.append(
                f"  {emoji} {t['symbol']}: {t['result']} | "
                f"{t['return_pct']:+.1f}% | NPR {t['pnl_npr']:+.0f}{cause}"
            )
        lines.append("")

    if open_positions:
        lines.append(f"*Open: {len(open_positions)} position(s)*")
        for p in open_positions:
            trail = " 🔒" if p.get("trail_active") == "true" else ""
            pnl   = float(p.get("pnl_pct") or 0)
            lines.append(f"  📈 {p.get('symbol')}: {pnl:+.1f}%{trail}")
        lines.append("")

    try:
        fin = [r for r in read_tab("financials") if r.get("kpi_name") == "overall_win_rate_pct"]
        tot = [r for r in read_tab("financials") if r.get("kpi_name") == "total_trades"]
        if fin and tot:
            lines.append(f"*Win rate: {fin[0].get('current_value')}% "
                         f"({tot[0].get('current_value')} trades)*")
    except Exception:
        pass

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 9. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hold_days(entry_date: str, exit_date: str) -> int:
    try:
        fmt   = "%Y-%m-%d"
        entry = datetime.strptime(entry_date, fmt).date()
        exit_ = datetime.strptime(exit_date,  fmt).date()
        return max(1, (exit_ - entry).days)
    except Exception:
        return 1


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


def _now_nst() -> str:
    from datetime import timezone, timedelta
    nst = timezone(timedelta(hours=5, minutes=45))
    return datetime.now(nst).strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN EOD RUN — LIVE TRADING
# ─────────────────────────────────────────────────────────────────────────────

def run_eod_audit(dry_run: bool = False) -> dict:
    log.info("=" * 60)
    log.info("AUDITOR.PY — EOD Run (LIVE) | dry_run=%s", dry_run)
    log.info("=" * 60)

    today   = _today()
    summary = {"closed": 0, "held": 0, "errors": 0, "dry_run": dry_run}
    closed_trades = []

    # ── Fetch macro scores ────────────────────────────────────────────────────
    try:
        geo_score = get_latest_geo_score()
        log.info("Geo score: %.2f", geo_score)
    except Exception as e:
        log.error("geo_score fetch failed: %s", e)
        geo_score = 0.0

    try:
        np_data     = get_latest_nepal_score()
        nepal_score = float(np_data.get("nepal_score", 0)
                            if isinstance(np_data, dict) else np_data or 0)
        log.info("Nepal score: %.2f", nepal_score)

        # FIX 2: write CURRENT_NEPAL_SCORE so _check_stop_conditions reads real value
        if not dry_run:
            update_setting("CURRENT_NEPAL_SCORE", str(nepal_score), set_by="auditor.py")
            log.info("CURRENT_NEPAL_SCORE written: %s", nepal_score)

    except Exception as e:
        log.error("nepal_score fetch failed: %s", e)
        nepal_score = 0.0

    # ── Load open positions ───────────────────────────────────────────────────
    try:
        open_positions = [p for p in read_tab("portfolio") if p.get("status") == "OPEN"]
        log.info("Open positions: %d", len(open_positions))
    except Exception as e:
        log.error("Portfolio load failed: %s", e)
        return summary

    if not open_positions:
        log.info("No open positions — running KPI update only.")
        _update_financials_kpis()
        _update_market_state()
        return summary

    still_open = []

    for position in open_positions:
        symbol = position.get("symbol", "?")
        try:
            position = _enrich_position_with_entry_context(position)

            # Get current price from price_history
            price_rows = [r for r in read_tab("price_history")
                          if r.get("symbol") == symbol]
            price_rows = sorted(price_rows, key=lambda x: x.get("date", ""), reverse=True)
            if not price_rows:
                log.warning("%s: No price data — skipping", symbol)
                still_open.append(position)
                summary["held"] += 1
                continue

            current_price = float(price_rows[0].get("close") or
                                  price_rows[0].get("ltp") or 0)
            if not current_price:
                log.warning("%s: Zero price — skipping", symbol)
                still_open.append(position)
                summary["held"] += 1
                continue

            # Update trailing stop
            trail_updates = _update_trailing_stop(position, current_price)
            if trail_updates and not dry_run:
                update_row("portfolio", trail_updates, where={"id": position["id"]})
                position.update(trail_updates)

            # Check exit conditions
            exit_context = _check_stop_conditions(position, current_price, geo_score)

            if exit_context:
                entry_date = position.get("entry_date", today)
                exit_date  = exit_context.get("exit_date", today)
                hold_days  = _compute_hold_days(entry_date, exit_date)
                position["hold_days_actual"] = str(hold_days)

                deltas = _compute_deltas(position, current_price, geo_score, nepal_score)

                if not dry_run:
                    trade_id = _write_trade_journal(position, deltas, exit_context, paper=False)

                    final_pnl = _compute_pnl_npr(
                        float(position.get("entry_price") or 0),
                        current_price,
                        float(position.get("shares") or 0),
                    )
                    update_row("portfolio", {
                        "status":      "CLOSED",
                        "exit_date":   today,
                        "exit_price":  str(current_price),
                        "exit_reason": exit_context["exit_reason"],
                        "pnl_npr":     str(final_pnl),
                        "pnl_pct":     str(deltas["return_pct"]),
                    }, where={"id": position["id"]})

                    try:
                        result = _classify_result(deltas["return_pct"])
                        update_row("market_log", {
                            "outcome":     result,
                            "exit_date":   today,
                            "exit_price":  str(current_price),
                            "exit_reason": exit_context["exit_reason"],
                            "actual_pnl":  str(deltas["return_pct"]),
                        }, where={"symbol": symbol, "action": "BUY", "outcome": "PENDING"})
                    except Exception as e:
                        log.warning("market_log update failed for %s: %s", symbol, e)

                closed_trades.append({
                    "symbol":     symbol,
                    "result":     _classify_result(deltas["return_pct"]),
                    "return_pct": deltas["return_pct"],
                    "pnl_npr":    _compute_pnl_npr(
                        float(position.get("entry_price") or 0),
                        current_price,
                        float(position.get("shares") or 0),
                    ),
                    "loss_cause": (
                        _attribute_loss(
                            deltas["geo_delta"], deltas["nepal_delta"],
                            deltas["nepse_return_pct"], deltas["alpha_vs_nepse"],
                            exit_context["exit_reason"], position.get("sector", ""),
                            deltas["nepal_score_exit"], int(position.get("hold_days_actual", 1)),
                        ) if _classify_result(deltas["return_pct"]) == "LOSS" else "NULL"
                    ),
                })
                summary["closed"] += 1

            else:
                # Still open — refresh unrealised P&L
                if not dry_run:
                    entry_price = float(position.get("entry_price") or 0)
                    if entry_price:
                        unreal_pct = round((current_price - entry_price) / entry_price * 100, 2)
                        unreal_npr = _compute_pnl_npr(entry_price, current_price,
                                                       float(position.get("shares") or 0))
                        update_row("portfolio", {
                            "current_price": str(current_price),
                            "pnl_pct":       str(unreal_pct),
                            "pnl_npr":       str(unreal_npr),
                        }, where={"id": position["id"]})
                still_open.append(position)
                summary["held"] += 1

        except Exception as e:
            log.error("Error processing %s: %s", symbol, e)
            summary["errors"] += 1
            still_open.append(position)

    if not dry_run:
        _update_financials_kpis()
        _update_market_state()

    msg = _build_eod_summary(closed_trades, still_open, paper=False)
    if not dry_run:
        send_telegram(msg)
    else:
        log.info("DRY RUN — would send:\n%s", msg)

    log.info("EOD complete — closed=%d held=%d errors=%d",
             summary["closed"], summary["held"], summary["errors"])
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 11. PAPER TRADING AUDIT  (FIX 1 — new function)
# ─────────────────────────────────────────────────────────────────────────────

def run_paper_audit(dry_run: bool = False) -> dict:
    """
    Paper trading equivalent of run_eod_audit().

    Reads paper_portfolio WHERE status='CLOSED' AND audited IS NULL.
    These are trades the Telegram bot has already closed via /sell.
    Runs the full causal attribution pipeline and writes to trade_journal.
    Stamps audited='true' on the paper_portfolio row when done.
    Links to market_log by symbol + date range to update outcome.

    Run manually at EOD during paper trading:
        python -m analysis.auditor --paper
        python -m analysis.auditor --paper --dry-run
    """
    log.info("=" * 60)
    log.info("AUDITOR.PY — Paper Trading Audit | dry_run=%s", dry_run)
    log.info("=" * 60)

    today   = _today()
    summary = {"processed": 0, "skipped": 0, "errors": 0, "dry_run": dry_run}

    # ── Fetch macro scores ────────────────────────────────────────────────────
    try:
        geo_score = get_latest_geo_score()
        log.info("Geo score: %.2f", geo_score)
    except Exception as e:
        log.error("geo_score fetch failed: %s", e)
        geo_score = 0.0

    try:
        np_data     = get_latest_nepal_score()
        nepal_score = float(np_data.get("nepal_score", 0)
                            if isinstance(np_data, dict) else np_data or 0)
        log.info("Nepal score: %.2f", nepal_score)

        # FIX 2: write CURRENT_NEPAL_SCORE (paper audit path also needs this)
        if not dry_run:
            update_setting("CURRENT_NEPAL_SCORE", str(nepal_score), set_by="auditor.py --paper")
            log.info("CURRENT_NEPAL_SCORE written: %s", nepal_score)

    except Exception as e:
        log.error("nepal_score fetch failed: %s", e)
        nepal_score = 0.0

    # ── Load closed, unaudited paper trades ───────────────────────────────────
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("""
                SELECT * FROM paper_portfolio
                WHERE status = 'CLOSED'
                AND (audited IS NULL OR audited = 'false')
                ORDER BY exit_date ASC
            """)
            closed_rows = cur.fetchall()
        log.info("Unaudited closed paper trades: %d", len(closed_rows))
    except Exception as e:
        log.error("paper_portfolio load failed: %s", e)
        return summary

    if not closed_rows:
        log.info("No unaudited paper trades — running KPI update only.")
        _update_financials_kpis()
        _update_market_state()
        return summary

    processed_trades = []

    for row in closed_rows:
        symbol = row.get("symbol", "?")
        try:
            # ── Map paper_portfolio fields to what _write_trade_journal expects ──
            # paper_portfolio has: total_shares, wacc (effective entry price),
            # total_cost, first_buy_date, exit_date, exit_price, net_pnl, result
            # Indicator context (rsi_entry etc.) will be empty — enriched from
            # market_log if available; gets richer once recommendation_tracker built.

            entry_price = float(row.get("wacc") or 0)
            exit_price  = float(row.get("exit_price") or 0)
            shares      = float(row.get("total_shares") or 0)
            entry_date  = row.get("first_buy_date", "")
            exit_date   = row.get("exit_date", today)

            if not entry_price or not exit_price or not shares:
                log.warning("%s (paper): Missing price/shares data — skipping", symbol)
                summary["skipped"] += 1
                continue

            hold_days = _compute_hold_days(entry_date, exit_date)

            # Build a position dict in the same shape _write_trade_journal expects
            position = {
                "symbol":           symbol,
                "sector":           row.get("sector", ""),
                "entry_date":       entry_date,
                "entry_price":      str(entry_price),
                "shares":           str(shares),
                "allocation_npr":   row.get("total_cost", ""),
                "primary_signal":   "",
                "secondary_signal": "NONE",
                "candle_pattern":   "NONE",
                "confidence_at_entry": "",
                "rsi_entry":        "",
                "macd_hist_entry":  "",
                "bb_signal_entry":  "",
                "ema_trend_entry":  "",
                "obv_trend_entry":  "",
                "conf_score_entry": "",
                "volume_ratio_entry": "",
                "atr_pct_entry":    "",
                "market_state_entry": get_setting("MARKET_STATE", "SIDEWAYS"),
                "geo_score_entry":  "",
                "nepal_score_entry": "",
                "combined_geo_entry": "",
                "nepse_index_entry": "",
                "stop_loss_planned": "",
                "target_planned":   "",
                "hold_days_planned": "",
                "hold_days_actual": str(hold_days),
                "telegram_id":      str(row.get("telegram_id", "")),
            }

            # Try to enrich with market_log context (signal info at entry time)
            position = _enrich_position_with_entry_context(position)

            exit_context = {
                "exit_price":       exit_price,
                "exit_date":        exit_date,
                "exit_reason":      row.get("result", "MANUAL"),  # bot closes are always manual
                "market_state_exit": get_setting("MARKET_STATE", "SIDEWAYS"),
            }

            # Compute deltas — use current macro for exit context
            # (paper trades closed by bot during day; best approximation available)
            deltas = _compute_deltas(position, exit_price, geo_score, nepal_score)

            return_pct = deltas["return_pct"]
            pnl_npr    = _compute_pnl_npr(entry_price, exit_price, shares)
            result     = _classify_result(return_pct, pnl_npr)

            log.info(
                "Paper trade: %s | %s | entry=%.2f exit=%.2f | return=%.2f%% | NPR %.0f | user=%s",
                symbol, result, entry_price, exit_price, return_pct, pnl_npr,
                row.get("telegram_id", "?"),
            )

            if not dry_run:
                # Write immutable trade_journal record
                trade_id = _write_trade_journal(position, deltas, exit_context, paper=True)

                # Link to market_log — match by symbol + date range
                try:
                    market_rows = [
                        r for r in read_tab("market_log")
                        if r.get("symbol") == symbol
                        and r.get("action") == "BUY"
                        and r.get("outcome") in ("PENDING", None, "")
                        and r.get("date", "") <= exit_date
                        and r.get("date", "") >= entry_date
                    ]
                    if market_rows:
                        target_row = sorted(market_rows,
                                            key=lambda x: x.get("date", ""),
                                            reverse=True)[0]
                        update_row("market_log", {
                            "outcome":     result,
                            "exit_date":   exit_date,
                            "exit_price":  str(exit_price),
                            "exit_reason": exit_context["exit_reason"],
                            "actual_pnl":  str(return_pct),
                        }, where={"id": target_row["id"]})
                        log.info("market_log updated for %s → %s", symbol, result)
                    else:
                        log.info("No matching market_log BUY row for %s in date range", symbol)
                except Exception as e:
                    log.warning("market_log link failed for %s: %s", symbol, e)

                # Stamp audited='true' on paper_portfolio row
                try:
                    from db.connection import _db
                    with _db() as cur:
                        cur.execute(
                            "UPDATE paper_portfolio SET audited = 'true' WHERE id = %s",
                            (row["id"],)
                        )
                    log.info("%s (paper): stamped audited=true (id=%s)", symbol, row["id"])
                except Exception as e:
                    log.error("Failed to stamp audited for %s id=%s: %s",
                              symbol, row["id"], e)

            processed_trades.append({
                "symbol":     symbol,
                "result":     result,
                "return_pct": return_pct,
                "pnl_npr":    pnl_npr,
                "telegram_id": row.get("telegram_id", ""),
                "loss_cause": (
                    _attribute_loss(
                        deltas["geo_delta"], deltas["nepal_delta"],
                        deltas["nepse_return_pct"], deltas["alpha_vs_nepse"],
                        exit_context["exit_reason"], position.get("sector", ""),
                        deltas["nepal_score_exit"], hold_days,
                    ) if result == "LOSS" else "NULL"
                ),
            })
            summary["processed"] += 1

        except Exception as e:
            log.error("Error processing paper trade %s: %s", symbol, e)
            summary["errors"] += 1

    # ── KPI update ────────────────────────────────────────────────────────────
    if not dry_run and processed_trades:
        _update_financials_kpis()

    # ── Summary log ──────────────────────────────────────────────────────────
    if processed_trades:
        log.info("Paper audit complete — processed=%d skipped=%d errors=%d",
                 summary["processed"], summary["skipped"], summary["errors"])
        for t in processed_trades:
            emoji = "✅" if t["result"] == "WIN" else ("❌" if t["result"] == "LOSS" else "➖")
            log.info("  %s %s: %s | %.2f%% | NPR %.0f | user=%s",
                     emoji, t["symbol"], t["result"], t["return_pct"],
                     t["pnl_npr"], t["telegram_id"])
    else:
        log.info("Paper audit complete — nothing to process.")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    import argparse

    parser = argparse.ArgumentParser(
        description="NEPSE Auditor — EOD trade closer + causal attribution"
    )
    parser.add_argument("--paper",      action="store_true",
                        help="Run paper trading audit (reads paper_portfolio)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Compute everything, write nothing")
    parser.add_argument("--kpis",       action="store_true",
                        help="Recompute KPIs only")
    parser.add_argument("--status",     action="store_true",
                        help="Show open positions + recent trades")
    parser.add_argument("--market-state", action="store_true",
                        help="Recompute MARKET_STATE only")
    parser.add_argument(
        "--attribute",
        nargs=6,
        metavar=("GEO_DELTA", "NEPAL_DELTA", "NEPSE_RETURN", "ALPHA",
                 "EXIT_REASON", "SECTOR"),
        help="Test _attribute_loss() directly",
    )
    args = parser.parse_args()

    if args.attribute:
        geo_d, nepal_d, nepse_r, alpha, reason, sector = args.attribute
        cause = _attribute_loss(float(geo_d), float(nepal_d), float(nepse_r),
                                float(alpha), reason, sector, 0.0, 5)
        print(f"\nLoss cause: {cause}\n")

    elif args.kpis:
        _update_financials_kpis()

    elif args.market_state:
        state = _update_market_state()
        print(f"\nMARKET_STATE → {state}\n")

    elif args.status:
        try:
            paper_mode = get_setting("PAPER_MODE", "true").lower() == "true"
            if paper_mode:
                from db.connection import _db
                with _db() as cur:
                    cur.execute("""
                        SELECT pp.*, pu.username
                        FROM paper_portfolio pp
                        LEFT JOIN paper_users pu ON pp.telegram_id = pu.telegram_id
                        WHERE pp.status = 'OPEN'
                        ORDER BY pp.telegram_id, pp.symbol
                    """)
                    open_pos = cur.fetchall()
                print(f"\nOpen paper positions: {len(open_pos)}")
                for p in open_pos:
                    print(f"  [{p.get('username') or p.get('telegram_id')}] "
                          f"{p.get('symbol')} | shares={p.get('total_shares')} "
                          f"| wacc={p.get('wacc')}")
            else:
                open_pos = [p for p in read_tab("portfolio") if p.get("status") == "OPEN"]
                print(f"\nOpen positions: {len(open_pos)}")
                for p in open_pos:
                    print(f"  {p.get('symbol')} | entry={p.get('entry_price')} "
                          f"| trail={p.get('trail_active')}")

            recent = read_tab("trade_journal", limit=5)
            print(f"\nLast 5 trades (trade_journal):")
            for t in recent:
                print(f"  {t.get('symbol')} | {t.get('result')} | "
                      f"{t.get('return_pct')}% | cause={t.get('loss_cause')} "
                      f"| paper={t.get('paper_mode')}")
        except Exception as e:
            print(f"Status check failed: {e}")

    elif args.paper:
        result = run_paper_audit(dry_run=args.dry_run)
        print(f"\nPaper audit result: {result}")

    else:
        result = run_eod_audit(dry_run=args.dry_run)
        print(f"\nLive audit result: {result}")