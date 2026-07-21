"""
filter_engine.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 1
Purpose : Combine frozen daily indicators + live prices + geo/nepal scores
          + candlestick signals into a ranked list of trade candidates.
          Feeds gemini_filter.py every 6 minutes during the trading loop.

─────────────────────────────────────────────────────────────────────────────
PHASE 1 SPLIT (2026-07-03): this file is now a thin orchestrator.
    filter_common.py  — gates, loaders, dataclasses, composite score
                         (shared by v1 and v2, unchanged logic)
    filter_v1.py       — snapshot-state indicator scorers (this file's
                         original behavior, moved verbatim)
    filter_v2.py       — NEW progression/slope scorers (6-day T-5→T0
                         window). NOT called by run_filter() yet — it's
                         built and standalone-testable (python filter_v2.py)
                         but does not affect any live decision until the
                         Phase 0 questions (circuit breaker / capital pool
                         split — still open) are resolved and Phase 3
                         wiring happens deliberately.

run_filter()'s behavior below is UNCHANGED from pre-split filter_engine.py.
Verify with: python filter_engine.py --dry-run (diff top-10 vs pre-split,
filter_engine.py.pre_phase1_backup kept alongside for that comparison).
─────────────────────────────────────────────────────────────────────────────

Evidence base for every weight and threshold in this file:
─────────────────────────────────────────────────────────────────────────────

INDICATOR WEIGHTS (Karki et al. 2023 — 10yr NEPSE backtest, n=2294)
    — now in filter_v1.INDICATOR_WEIGHTS. See filter_v2.INDICATOR_WEIGHTS_V2
    for the experimental progression-based weights and their provenance.

SECTOR MULTIPLIERS (Khadka & Rajopadhyaya 2023 — Single Index Model):
    Non-Life Insurance  1.25  → best risk-adj return (2.732), β=0.034
    Finance             1.15  → selected, excess return 0.270
    Micro Finance       1.10  → selected, excess return 0.165
    Hydro Power         1.10  → selected (1.737 risk-adj), β=0.042
    Development Bank    1.08  → selected, excess return 0.151
    Life Insurance      1.05  → excluded from optimal (0.106 < C*=0.129)
    Others              1.00  → baseline
    Banking             0.90  → excluded from optimal (0.051 < C*=0.129)
    Manufacturing       0.75  → worst risk-adj return (-0.044)

    NOTE on Hydro: SIM paper shows strong risk-adj (1.737) but technical
    paper shows RSI loses most (-6.49%) in Hydro. Resolution: full sector
    multiplier applies, but RSI contribution is halved for hydro symbols.

INSURANCE CONDITIONAL MULTIPLIER (Political events paper):
    Insurance 4.6x more sensitive to political events than NEPSE.
    Pre-event leakage window: -10 to -1 days (Insurance AAR=0.272).
    crisis_detected=YES  → insurance multiplier drops to 0.85
    nepal_score >= 1     → insurance multiplier gets +0.10 boost

HARD GATES (system never signals past these — hard rules from handoff):
    combined_geo <= -3      → full block (capital preservation)
    bandh_today == YES      → full block (zero liquidity)
    market_state == CRISIS  → full block
    loss_streak > 7         → circuit breaker
    ltp <= 0                → no live price, skip
    history_days < 20       → insufficient history for indicators
    rsi > 75                → overbought, blocked
    rsi < 30 + no MACD/BB   → RSI oversold alone is not a signal (paper)

TECH SCORE THRESHOLDS by market state:
    FULL_BULL       50   (catch more candidates)
    CAUTIOUS_BULL   58   (BULL threshold from indicators.py)
    SIDEWAYS        65
    BEAR            72   (very selective)
    CRISIS          999  (blocks everything)

C* LIVE RANKING (Khadka & Rajopadhyaya 2023 SIM paper):
    (daily_change_pct/100 - Rf_daily) / sector_beta > 0.129 → +5 bonus
    Rf_daily = annual Rf / 252

─────────────────────────────────────────────────────────────────────────────
Called by: trading.yml every 6 min, 10:45 AM – 3:00 PM NST
Next:      gemini_filter.py reads output of this module
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime
from typing import Optional
from config import NST

import filter_common
import filter_v2
from filter_common import (
    FilterCandidate,
    NearMiss,
    _categorize_gate_reason,
    _load_context,
    _load_recent_indicators,
    _compute_momentum,
    _check_hard_gates,
    _check_symbol_gates,
    _get_sector_multiplier,
    _check_cstar_signal,
    _load_candle_signals,
    _candle_bonus,
    _load_fundamental_data,
    _is_non_equity_by_beta,
    _get_fundamental_adj,
    _load_broker_flow_cache,
    _compute_broker_flow_adj,
    _compute_vos_adj,
    _compute_live_adj,
    _compute_composite_score,
    MIN_CONF_SCORE,
    MIN_VOS_PCT,
    RF_ANNUAL_PCT,
)
from filter_v1 import _compute_indicator_score

logger = logging.getLogger(__name__)

# Gate categories that block v1 but are soft enough for v2 to score anyway —
# they reflect T0-snapshot weakness (tech score, RSI state, broker flow, DPR),
# not identity/data-quality problems. v2 is a progression engine and may see
# a symbol improving even though its T0 snapshot tripped one of these.
_V2_RESCUE_ELIGIBLE_CATEGORIES = {
    "TECH_SCORE", "RSI_OVERBOUGHT", "RSI_NO_CONFIRM", "DPR_UPPER",
    "BROKER_FLOW", "CONF_SCORE",
}


def _try_v2_rescue(
    sym:             str,
    ind:             dict,
    price_row,
    ctx:             dict,
    momentum:        dict,
    recent_map:      dict,
    sector_map:      dict,
    candle_map:      dict,
    fund_map:        dict,
    beta_map:        dict,
    flow_cache:      dict,
    holdings_cache:  dict,
    rf_rate:         float,
    volume_os_ratio: float,
    date:            str,
) -> Optional["FilterCandidate"]:
    """
    Score a v1-gate-blocked symbol with the v2 (progression) engine, using
    the same shared filter_common helpers v1 would have used had it passed.

    Only called for symbols blocked by a soft/rescue-eligible gate category
    (see _V2_RESCUE_ELIGIBLE_CATEGORIES) — structural gates (NO_LTP,
    MUTUAL_FUND, NON_EQUITY, HISTORY, ILLIQUID/VOS) never reach here.

    Fails silently: returns None on any error so the pipeline continues
    (per spec — v2 rescue errors must not block the run).
    """
    try:
        sector    = sector_map.get(sym) or str(ind.get("sector", "others") or "others")
        sect_mult = _get_sector_multiplier(sector, ctx)

        ltp        = float(getattr(price_row, "ltp", 0)        or getattr(price_row, "close", 0) or 0)
        change_pct = float(getattr(price_row, "change_pct", 0) or 0)
        cstar      = _check_cstar_signal(change_pct, sector, rf_rate)

        patterns                        = candle_map.get(sym, [])
        c_bonus, c_name, c_tier, c_conf = _candle_bonus(patterns)

        fund_adj, fund_reason = _get_fundamental_adj(sym, sector, fund_map, beta_map)
        vos_adj               = _compute_vos_adj(volume_os_ratio)
        live_adj              = _compute_live_adj(price_row)
        broker_flow_adj       = _compute_broker_flow_adj(sym, flow_cache, holdings_cache)

        ind_score_v2, primary_v2, hold_days_v2 = filter_v2.compute_indicator_score_v2(
            momentum, recent_map.get(sym, []), sector,
        )
        composite_v2 = _compute_composite_score(
            indicator_score=ind_score_v2, sector_mult=sect_mult,
            candle_bonus=c_bonus, cstar_signal=cstar,
            conf_score=float(getattr(price_row, "conf_score", 0) or 0),
            geo_combined=ctx["combined_geo"], ipo_drain=ctx["ipo_drain"],
            fundamental_adj=fund_adj, vos_adj=vos_adj,
            broker_flow_adj=broker_flow_adj, live_adj=live_adj,
            min_conf_score=ctx.get("min_conf_score", MIN_CONF_SCORE),
        )

        candidate_news_catalyst = ctx.get("_sym_catalysts", {}).get(sym, "")

        return FilterCandidate(
            symbol           = sym,
            sector           = sector,
            ltp              = ltp,
            change_pct       = change_pct,
            volume           = int(getattr(price_row, "volume", 0) or 0),

            rsi_14           = float(ind.get("rsi_14",           0)   or 0),
            rsi_signal       = str(ind.get("rsi_signal",         "")  or ""),
            ema_trend        = str(ind.get("ema_trend",          "")  or ""),
            ema_20_50_cross  = str(ind.get("ema_20_50_cross",    "")  or ""),
            ema_50_200_cross = str(ind.get("ema_50_200_cross",   "")  or ""),
            macd_cross       = str(ind.get("macd_cross",     "NONE") or "NONE"),
            macd_histogram   = float(ind.get("macd_histogram",   0)   or 0),
            macd_line        = float(ind.get("macd_line",         0)   or 0),
            macd_signal_line = float(ind.get("macd_signal",       0)   or 0),
            bb_signal        = str(ind.get("bb_signal",  "NEUTRAL")   or "NEUTRAL"),
            bb_pct_b         = float(ind.get("bb_pct_b",        0.5)  or 0.5),
            bb_upper         = float(ind.get("bb_upper",          0)   or 0),
            bb_lower         = float(ind.get("bb_lower",          0)   or 0),

            obv_trend        = str(ind.get("obv_trend",    "FLAT")    or "FLAT"),
            atr_pct          = float(ind.get("atr_pct",          0)   or 0),
            tech_score       = int(ind.get("tech_score",         0)   or 0),
            tech_signal      = str(ind.get("tech_signal",        "")  or ""),
            history_days     = int(ind.get("history_days",       0)   or 0),

            support_level    = float(ind.get("support_level",    0)   or 0),
            resistance_level = float(ind.get("resistance_level", 0)   or 0),
            pivot_r1         = float(ind.get("pivot_r1",         0)   or 0),
            pivot_r2         = float(ind.get("pivot_r2",         0)   or 0),
            pivot_r3         = float(ind.get("pivot_r3",         0)   or 0),
            pivot_s1         = float(ind.get("pivot_s1",         0)   or 0),
            pivot_s2         = float(ind.get("pivot_s2",         0)   or 0),
            pivot_s3         = float(ind.get("pivot_s3",         0)   or 0),

            conf_score       = float(getattr(price_row, "conf_score",  0) or 0),
            conf_signal      = str(getattr(price_row,  "conf_signal", "") or ""),

            candle_patterns  = patterns,
            best_candle      = c_name,
            candle_tier      = c_tier,
            candle_conf      = c_conf,

            geo_score        = ctx["geo_score"],
            nepal_score      = ctx["nepal_score"],
            combined_geo     = ctx["combined_geo"],
            bandh_today      = ctx["bandh_today"],
            crisis_detected  = ctx["crisis_detected"],
            ipo_drain        = ctx["ipo_drain"],
            market_state     = ctx["market_state"],

            # v1 never scored this symbol — indicator_score/composite_score
            # stay at their dataclass defaults (0.0) until merge assigns v2's
            # values for the "v2"-tagged output candidate.
            sector_mult      = sect_mult,
            cstar_signal     = cstar,
            engine_source    = "v2_rescue",

            fundamental_adj    = fund_adj,
            fundamental_reason = fund_reason,
            broker_flow_adj    = broker_flow_adj,

            indicator_score_v2 = ind_score_v2,
            composite_score_v2 = composite_v2,
            primary_signal_v2  = primary_v2,
            suggested_hold_v2  = hold_days_v2,

            vwap_dev        = float(getattr(price_row, "vwap_dev",       0) or 0),
            bid_ask_ratio   = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
            dpr_proximity   = float(getattr(price_row, "dpr_proximity",  0) or 0),
            volume_os_ratio = volume_os_ratio,

            momentum_status  = momentum["momentum_status"],
            rsi_slope_3d     = momentum["rsi_slope_3d"],
            macd_hist_slope  = momentum["macd_hist_slope"],
            bb_pct_b_slope   = momentum["bb_pct_b_slope"],
            bounce_failed    = momentum["bounce_failed"],
            reversal_days    = momentum["reversal_days"],

            news_catalyst    = candidate_news_catalyst,
        )
    except Exception as exc:
        logger.warning("v2 gate-rescue scoring failed for %s (%s) — skipped", sym, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN FILTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_filter(
    market_data: dict   = None,
    top_n:       int    = 10,
    date:        str    = None,
) -> list[FilterCandidate]:
    """
    Main entry point. Called every 6 min by trading.yml.

    Args:
        market_data: dict[symbol, PriceRow] from scraper.get_all_market_data()
                     If None, fetches live data automatically.
        top_n:       Number of top candidates to return (default 10).
        date:        Override date string YYYY-MM-DD (default: today NST).

    Returns:
        list[FilterCandidate] ranked by composite_score descending.
        Empty list if hard gates block the run.

    NOTE: v1-only, unchanged from pre-Phase-1-split behavior. v2 scoring
    (filter_v2.py) is not called here yet — see module docstring.
    """
    if date is None:
        date = datetime.now(tz=NST).strftime("%Y-%m-%d")

    filter_common._last_near_misses = []

    logger.info("=" * 60)
    logger.info("filter_engine.run_filter() — %s", date)

    # ── Context ───────────────────────────────────────────────────────────────
    ctx = _load_context()

    from sheets import get_setting
    v2_enabled      = str(get_setting("FILTER_V2_ENABLED", "false")).strip().lower() == "true"
    v2_top_n        = int(get_setting("FILTER_V2_TOP_N", "3"))
    v2_gate_rescue  = str(get_setting("FILTER_V2_GATE_RESCUE", "true")).strip().lower() == "true"

    # ── System hard gates ─────────────────────────────────────────────────────
    gates_ok, gate_reason = _check_hard_gates(ctx)
    if not gates_ok:
        logger.warning("HARD GATE BLOCKED: %s", gate_reason)
        return []

    # ── Live prices ───────────────────────────────────────────────────────────
    if market_data is None:
        try:
            from modules.scraper import get_all_market_data
            market_data = get_all_market_data(write_breadth=False)
        except Exception as exc:
            logger.error("Market data fetch failed: %s", exc)
            return []

    if not market_data:
        logger.warning("No market data available")
        return []

    # ── Frozen indicators ─────────────────────────────────────────────────────
    try:
        from sheets import read_today_indicators
        indicators_map = read_today_indicators(date)
    except Exception as exc:
        logger.error("Could not load indicators: %s", exc)
        return []

    if not indicators_map:
        logger.warning("No indicators for %s — run indicators.py first", date)
        return []

    # ── Momentum history (7-day batch load) ───────────────────────────────────
    valid_syms = [
        s.upper() for s in market_data
        if str(s).replace("-", "").replace("_", "").isalpha()
        and indicators_map.get(s.upper())
    ]
    recent_map = _load_recent_indicators(valid_syms, date, lookback=7)
    logger.info("Momentum history loaded: %d symbols", len(recent_map))

    # ── Candle signals ────────────────────────────────────────────────────────
    valid_symbols = [
        s.upper() for s in market_data
        if str(s).replace("-", "").replace("_", "").isalpha()
    ]
    candle_map = _load_candle_signals(valid_symbols, date)

    # ── Fundamental data (loaded ONCE for all symbols) ────────────────────────
    fund_map, beta_map = _load_fundamental_data()

    # ── Broker flow cache (loaded ONCE for all symbols) ───────────────────────
    flow_cache, holdings_cache = _load_broker_flow_cache(date)

    # ── Sector info from watchlist ────────────────────────────────────────────
    sector_map: dict[str, str] = {}
    try:
        from sheets import read_tab
        share_sectors = read_tab("share_sectors")
        sector_map = {
            r["symbol"].upper(): (r.get("sectorname") or "others").lower()
            for r in share_sectors if r.get("symbol")
        }
    except Exception as exc:
        logger.warning("Could not load share_sectors: %s", exc)

    # ── Score every symbol ────────────────────────────────────────────────────
    candidates:            list[FilterCandidate] = []
    v2_rescue_candidates:  list[FilterCandidate] = []
    skipped_gate   = 0
    skipped_no_ind = 0
    processed      = 0
    rf_rate        = ctx.get("rf_rate_annual", RF_ANNUAL_PCT)

    for symbol, price_row in market_data.items():
        sym = str(symbol).upper()
        if not sym.replace("-", "").replace("_", "").isalpha():
            continue

        ind = indicators_map.get(sym)
        if not ind:
            skipped_no_ind += 1
            continue

        processed += 1

        # ── VOS computation — before sym_ok so it's available for NearMiss ──
        fund_data  = fund_map.get(sym, {})
        paidup     = float(fund_data.get("paidup_capital") or 0)
        os_shares  = paidup / 100.0 if paidup > 0 else 0.0
        volume_val = float(getattr(price_row, "volume", 0) or 0)
        volume_os_ratio = round((volume_val / os_shares * 100), 4) if os_shares > 0 else 0.0
        logger.debug("VOS: %s vol=%.0f os=%.0f ratio=%.3f%%", sym, volume_val, os_shares, volume_os_ratio)

        # Compute momentum for this symbol and inject into ctx temporarily
        momentum = _compute_momentum(recent_map.get(sym, []))
        ctx["_sym_momentum"] = momentum   # consumed by _check_symbol_gates

        sym_ok, sym_reason = _check_symbol_gates(sym, ind, price_row, ctx, flow_cache=flow_cache)
        if not sym_ok:
            skipped_gate += 1
            logger.debug("GATE: %s — %s", sym, sym_reason)
            gate_category = _categorize_gate_reason(sym_reason)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)

            near_miss_v2_score = 0.0
            near_miss_engine   = "shared"

            # ── v2 gate-rescue: soft (non-structural) gates don't block v2 ──────
            if (
                v2_enabled and v2_gate_rescue
                and gate_category in _V2_RESCUE_ELIGIBLE_CATEGORIES
            ):
                rescue_candidate = _try_v2_rescue(
                    sym=sym, ind=ind, price_row=price_row, ctx=ctx,
                    momentum=momentum, recent_map=recent_map,
                    sector_map=sector_map, candle_map=candle_map,
                    fund_map=fund_map, beta_map=beta_map,
                    flow_cache=flow_cache, holdings_cache=holdings_cache,
                    rf_rate=rf_rate, volume_os_ratio=volume_os_ratio,
                    date=date,
                )
                if rescue_candidate is not None:
                    v2_rescue_candidates.append(rescue_candidate)
                    near_miss_v2_score = rescue_candidate.composite_score_v2
                    near_miss_engine   = "v2_rescue"

            filter_common._last_near_misses.append(NearMiss(
                symbol                   = sym,
                sector                   = sector_map.get(sym, "others"),
                date                     = date,
                gate_reason              = sym_reason,
                gate_category            = gate_category,
                price_at_block           = ltp,
                market_state             = ctx["market_state"],
                tech_score               = int(ind.get("tech_score", 0) or 0),
                conf_score               = float(getattr(price_row, "conf_score", 0) or 0),
                composite_score_would_be = 0.0,
                volume_os_ratio          = volume_os_ratio,
                vwap_dev                 = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio            = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity            = float(getattr(price_row, "dpr_proximity",  0) or 0),
                engine_source            = near_miss_engine,
                composite_score_v2       = near_miss_v2_score,
            ))
            continue

        # ── Non-equity exclusion via beta (debentures/bonds) ─────────────────
        if _is_non_equity_by_beta(sym, beta_map):
            skipped_gate += 1
            logger.debug("GATE: %s — NON_EQUITY_BY_BETA", sym)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)
            filter_common._last_near_misses.append(NearMiss(
                symbol          = sym,
                sector          = sector_map.get(sym, "others"),
                date            = date,
                gate_reason     = "NON_EQUITY_BY_BETA",
                gate_category   = "NON_EQUITY",
                price_at_block  = ltp,
                market_state    = ctx["market_state"],
                tech_score      = int(ind.get("tech_score", 0) or 0),
                conf_score      = float(getattr(price_row, "conf_score", 0) or 0),
                volume_os_ratio = volume_os_ratio,
                vwap_dev        = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio   = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity   = float(getattr(price_row, "dpr_proximity",  0) or 0),
            ))
            continue

        # ── VOS soft gate: truly dead stock ──────────────────────────────────
        if volume_os_ratio < MIN_VOS_PCT and os_shares > 0:
            skipped_gate += 1
            vos_gate_reason = f"VOS={volume_os_ratio:.3f}%<{MIN_VOS_PCT}%"
            logger.debug("GATE: %s — %s", sym, vos_gate_reason)
            ltp = float(getattr(price_row, "ltp", 0) or getattr(price_row, "close", 0) or 0)
            filter_common._last_near_misses.append(NearMiss(
                symbol                   = sym,
                sector                   = sector_map.get(sym, "others"),
                date                     = date,
                gate_reason              = vos_gate_reason,
                gate_category            = "ILLIQUID",
                price_at_block           = ltp,
                market_state             = ctx["market_state"],
                tech_score               = int(ind.get("tech_score", 0) or 0),
                conf_score               = float(getattr(price_row, "conf_score", 0) or 0),
                composite_score_would_be = 0.0,
                volume_os_ratio          = volume_os_ratio,
                vwap_dev                 = float(getattr(price_row, "vwap_dev",       0) or 0),
                bid_ask_ratio            = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                dpr_proximity            = float(getattr(price_row, "dpr_proximity",  0) or 0),
            ))
            continue

        sector    = sector_map.get(sym) or str(ind.get("sector", "others") or "others")
        ind_score, primary, hold_days = _compute_indicator_score(
            ind, sector,
            momentum["momentum_status"],
            momentum.get("momentum_score", 50.0),
        )
        sect_mult = _get_sector_multiplier(sector, ctx)

        ltp        = float(getattr(price_row, "ltp", 0)        or getattr(price_row, "close", 0) or 0)
        change_pct = float(getattr(price_row, "change_pct", 0) or 0)
        cstar      = _check_cstar_signal(change_pct, sector, rf_rate)

        patterns                        = candle_map.get(sym, [])
        c_bonus, c_name, c_tier, c_conf = _candle_bonus(patterns)

        # ── Fundamental adjustment ────────────────────────────────────────────
        fund_adj, fund_reason = _get_fundamental_adj(sym, sector, fund_map, beta_map)
        logger.debug("FUND: %s adj=%.2f [%s]", sym, fund_adj, fund_reason)

        # ── VOS scoring adjustment ────────────────────────────────────────────
        vos_adj = _compute_vos_adj(volume_os_ratio)
        logger.debug("VOS_ADJ: %s vos=%.3f%% adj=%+.1f", sym, volume_os_ratio, vos_adj)

        # ── Live intraday adjustment ──────────────────────────────────────────────
        live_adj = _compute_live_adj(price_row)
        logger.debug("LIVE_ADJ: %s vwap=%.2f%% bar=%.2f dpr=%.2f adj=%+.2f",
                     sym,
                     float(getattr(price_row, "vwap_dev",       0) or 0),
                     float(getattr(price_row, "bid_ask_ratio",  0) or 0),
                     float(getattr(price_row, "dpr_proximity",  0) or 0),
                     live_adj)

        # ── Broker flow scoring adjustment ────────────────────────────────────
        broker_flow_adj = _compute_broker_flow_adj(sym, flow_cache, holdings_cache)
        logger.debug("BROKER_FLOW_ADJ: %s adj=%+.1f", sym, broker_flow_adj)

        # ── News catalyst (from nepal_pulse STOCK_CATALYSTS_TODAY) ────────────
        candidate_news_catalyst = ctx.get("_sym_catalysts", {}).get(sym, "")


        composite = _compute_composite_score(
            indicator_score = ind_score,
            sector_mult     = sect_mult,
            candle_bonus    = c_bonus,
            cstar_signal    = cstar,
            conf_score      = float(getattr(price_row, "conf_score", 0) or 0),
            geo_combined    = ctx["combined_geo"],
            ipo_drain       = ctx["ipo_drain"],
            fundamental_adj = fund_adj,
            vos_adj         = vos_adj,
            broker_flow_adj = broker_flow_adj,
            live_adj        = live_adj,
            min_conf_score  = ctx.get("min_conf_score", MIN_CONF_SCORE),
        )
        logger.debug("VOS_ADJ: %s composite=%.1f", sym, composite)

        ind_score_v2  = 0.0
        composite_v2  = 0.0
        primary_v2    = ""
        hold_days_v2  = 0
        if v2_enabled:
            try:
                ind_score_v2, primary_v2, hold_days_v2 = filter_v2.compute_indicator_score_v2(
                    momentum, recent_map.get(sym, []), sector,
                )
                composite_v2 = _compute_composite_score(
                    indicator_score=ind_score_v2, sector_mult=sect_mult,
                    candle_bonus=c_bonus, cstar_signal=cstar,
                    conf_score=float(getattr(price_row, "conf_score", 0) or 0),
                    geo_combined=ctx["combined_geo"], ipo_drain=ctx["ipo_drain"],
                    fundamental_adj=fund_adj, vos_adj=vos_adj,
                    broker_flow_adj=broker_flow_adj, live_adj=live_adj,
                    min_conf_score=ctx.get("min_conf_score", MIN_CONF_SCORE),
                )
            except Exception as exc:
                logger.warning("v2 scoring failed for %s (%s) — v1-only for this symbol", sym, exc)

        candidates.append(FilterCandidate(
            symbol           = sym,
            sector           = sector,
            ltp              = ltp,
            change_pct       = change_pct,
            volume           = int(getattr(price_row, "volume", 0) or 0),

            rsi_14           = float(ind.get("rsi_14",           0)   or 0),
            rsi_signal       = str(ind.get("rsi_signal",         "")  or ""),
            ema_trend        = str(ind.get("ema_trend",          "")  or ""),
            ema_20_50_cross  = str(ind.get("ema_20_50_cross",    "")  or ""),
            ema_50_200_cross = str(ind.get("ema_50_200_cross",   "")  or ""),
            macd_cross       = str(ind.get("macd_cross",     "NONE") or "NONE"),
            macd_histogram   = float(ind.get("macd_histogram",   0)   or 0),
            macd_line        = float(ind.get("macd_line",         0)   or 0),
            macd_signal_line = float(ind.get("macd_signal",       0)   or 0),
            bb_signal        = str(ind.get("bb_signal",  "NEUTRAL")   or "NEUTRAL"),
            bb_pct_b         = float(ind.get("bb_pct_b",        0.5)  or 0.5),
            bb_upper         = float(ind.get("bb_upper",          0)   or 0),
            bb_lower         = float(ind.get("bb_lower",          0)   or 0),
            
            obv_trend        = str(ind.get("obv_trend",    "FLAT")    or "FLAT"),
            atr_pct          = float(ind.get("atr_pct",          0)   or 0),
            tech_score       = int(ind.get("tech_score",         0)   or 0),
            tech_signal      = str(ind.get("tech_signal",        "")  or ""),
            history_days     = int(ind.get("history_days",       0)   or 0),

            support_level    = float(ind.get("support_level",    0)   or 0),
            resistance_level = float(ind.get("resistance_level", 0)   or 0),
            pivot_r1         = float(ind.get("pivot_r1",         0)   or 0),
            pivot_r2         = float(ind.get("pivot_r2",         0)   or 0),
            pivot_r3         = float(ind.get("pivot_r3",         0)   or 0),
            pivot_s1         = float(ind.get("pivot_s1",         0)   or 0),
            pivot_s2         = float(ind.get("pivot_s2",         0)   or 0),
            pivot_s3         = float(ind.get("pivot_s3",         0)   or 0),

            conf_score       = float(getattr(price_row, "conf_score",  0) or 0),
            conf_signal      = str(getattr(price_row,  "conf_signal", "") or ""),

            candle_patterns  = patterns,
            best_candle      = c_name,
            candle_tier      = c_tier,
            candle_conf      = c_conf,

            geo_score        = ctx["geo_score"],
            nepal_score      = ctx["nepal_score"],
            combined_geo     = ctx["combined_geo"],
            bandh_today      = ctx["bandh_today"],
            crisis_detected  = ctx["crisis_detected"],
            ipo_drain        = ctx["ipo_drain"],
            market_state     = ctx["market_state"],

            indicator_score  = ind_score,
            sector_mult      = sect_mult,
            cstar_signal     = cstar,
            composite_score  = composite,
            primary_signal   = primary,
            suggested_hold   = hold_days,

            fundamental_adj    = fund_adj,
            fundamental_reason = fund_reason,
            broker_flow_adj    = broker_flow_adj,

            indicator_score_v2 = ind_score_v2,
            composite_score_v2 = composite_v2,
            primary_signal_v2  = primary_v2,
            suggested_hold_v2  = hold_days_v2,

            vwap_dev        = float(getattr(price_row, "vwap_dev",       0) or 0),
            bid_ask_ratio   = float(getattr(price_row, "bid_ask_ratio",  0) or 0),
            dpr_proximity   = float(getattr(price_row, "dpr_proximity",  0) or 0),
            volume_os_ratio = volume_os_ratio,

            momentum_status  = momentum["momentum_status"],
            rsi_slope_3d     = momentum["rsi_slope_3d"],
            macd_hist_slope  = momentum["macd_hist_slope"],
            bb_pct_b_slope   = momentum["bb_pct_b_slope"],
            bounce_failed    = momentum["bounce_failed"],
            reversal_days    = momentum["reversal_days"],

            news_catalyst    = candidate_news_catalyst,
        ))

    ctx.pop("_sym_momentum", None)

    # ── Rank and trim ─────────────────────────────────────────────────────────
    if not v2_enabled:
        candidates.sort(key=lambda c: c.composite_score, reverse=True)
        top = candidates[:top_n]
    else:
        # v2_gate_rescue=true (default): v2 scores ALL symbols that passed
        # structural gates, including ones v1's soft gates blocked — those
        # live in v2_rescue_candidates, never in v1's own candidate pool.
        # v2_gate_rescue=false: v2 only re-ranks v1 survivors (old behavior).
        v2_pool = candidates + v2_rescue_candidates if v2_gate_rescue else candidates

        v1_top = sorted(candidates, key=lambda c: c.composite_score,    reverse=True)[:v2_top_n]
        v2_top = sorted(v2_pool,    key=lambda c: c.composite_score_v2, reverse=True)[:v2_top_n]

        merged: dict = {}
        for c in v1_top:
            c.engine_source = "v1"
            merged[c.symbol] = c
        for c in v2_top:
            if c.composite_score_v2 <= 0.0:
                continue
            if c.symbol in merged:
                existing = merged[c.symbol]
                existing.engine_source = "BOTH"
                existing.co_flagged_by = (
                    f"v2 also flagged: score={c.composite_score_v2:.1f} "
                    f"signal={c.primary_signal_v2} hold={c.suggested_hold_v2}d"
                )
            else:
                if c.engine_source == "v2_rescue":
                    c.co_flagged_by = "v1 gate blocked this symbol — v2 rescued it"
                else:
                    c.co_flagged_by = f"v1 opinion: score={c.composite_score:.1f} signal={c.primary_signal}"
                c.composite_score = c.composite_score_v2
                c.primary_signal  = c.primary_signal_v2
                c.suggested_hold  = c.suggested_hold_v2
                c.engine_source   = "v2"
                merged[c.symbol]  = c

        top = sorted(merged.values(), key=lambda c: c.composite_score, reverse=True)
        logger.info("DUAL-ENGINE: v1_top=%s v2_top=%s",
                     [c.symbol for c in v1_top], [c.symbol for c in v2_top])

    logger.info(
        "run_filter done: %d processed | %d passed | %d gate-skipped | %d no-indicator",
        processed, len(candidates), skipped_gate, skipped_no_ind,
    )
    for c in top:
        logger.info("  %s", c.summary())

    _log_filter_candidates(top, date)

    return top

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7.5 — FILTER CANDIDATE LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _log_filter_candidates(candidates: list, date: str) -> None:
    """
    Log all filter candidates passed to Gemini each cycle.
    Upserts on (symbol, date) — increments pass_count on repeat appearances.
    Uses execute_dml from sheets (never from db directly).
    Fails silently — logging must never break the trading pipeline.
    """
    if not candidates:
        return
    try:
        from sheets import execute_dml
        for c in candidates:
            execute_dml(
                """
                INSERT INTO filter_candidates_log
                    (date, symbol, sector, composite_score, primary_signal,
                     tech_score, macro_score, market_state, engine_source, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (symbol, date) DO UPDATE SET
                    pass_count      = filter_candidates_log.pass_count + 1,
                    last_seen       = now(),
                    composite_score = EXCLUDED.composite_score,
                    tech_score      = EXCLUDED.tech_score,
                    macro_score     = EXCLUDED.macro_score,
                    engine_source   = EXCLUDED.engine_source
                """,
                (
                    date,
                    c.symbol,
                    c.sector,
                    c.composite_score,
                    c.primary_signal,
                    float(c.tech_score),
                    float(c.nepal_score),
                    c.market_state,
                    c.engine_source,
                ),
            )
        logger.info("filter_candidates_log: wrote %d candidates", len(candidates))
    except Exception as e:
        logger.warning("filter_candidates_log write failed: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HELPERS FOR gemini_filter.py
# ══════════════════════════════════════════════════════════════════════════════

def get_filter_context() -> dict:
    """Current context dict without running the full filter."""
    return _load_context()

def get_last_near_misses() -> list[NearMiss]:
    """Return near-misses captured during the last run_filter() call."""
    return list(filter_common._last_near_misses)

def format_candidate_for_gemini(c: FilterCandidate) -> str:
    """
    Compact single-line string for Gemini Flash prompt.
    Keeps token count low while preserving all signal information.
    """
    candle = f"{c.best_candle}(T{c.candle_tier},{c.candle_conf}%)" if c.best_candle else "none"
    cstar  = "Y" if c.cstar_signal else "N"
    momentum_str = f" | momentum={c.momentum_status} rsi_d={c.rsi_slope_3d:+.2f}"
    if c.bounce_failed:
        momentum_str += " BOUNCE_FAILED"
    elif c.reversal_days > 0:
        momentum_str += f" rev_days={c.reversal_days}"
    engine_str = f" | ENGINE:{c.engine_source}"
    if c.co_flagged_by:
        engine_str += f" [{c.co_flagged_by}]"
    return (
        f"SYM:{c.symbol} SEC:{c.sector} LTP:{c.ltp:.2f} CHG:{c.change_pct:+.2f}% "
        f"VOL:{c.volume:,} SCORE:{c.composite_score:.1f} TECH:{c.tech_score} "
        f"RSI:{c.rsi_14:.1f}[{c.rsi_signal}] MACD:{c.macd_cross} "
        f"BB:{c.bb_signal}[{c.bb_pct_b:.2f}] EMA:{c.ema_trend} "
        f"OBV:{c.obv_trend} ATR%:{c.atr_pct:.1f} CONF:{c.conf_score:.0f} "
        f"CANDLE:{candle} CSTAR:{cstar} HOLD:{c.suggested_hold}d SIG:{c.primary_signal}"
    ) + momentum_str + engine_str


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python filter_engine.py              → full run, top 10
#   python filter_engine.py --dry-run    → synthetic prices from cache
#   python filter_engine.py NABIL HBL   → specific symbols only
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [FILTER] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    args     = sys.argv[1:]
    dry_run  = "--dry-run" in args
    sym_args = [a.upper() for a in args if not a.startswith("--")]

    print("\n" + "=" * 70)
    print("  NEPSE AI — filter_engine.py")
    print("=" * 70)

    # ── Market data ───────────────────────────────────────────────────────────
    if dry_run:
        print("\n[DRY RUN] Synthetic prices from HistoryCache...")
        try:
            from modules.indicators import HistoryCache, DEFAULT_LOAD_PERIODS
            from modules.scraper import PriceRow
            cache = HistoryCache()
            if not cache.load(periods=DEFAULT_LOAD_PERIODS):
                print("  ❌ Cache empty"); sys.exit(1)
            market_data = {
                s: PriceRow(
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
            print(f"  ✅ {len(market_data)} symbols")
        except Exception as e:
            print(f"  ❌ {e}"); sys.exit(1)
    else:
        print("\n[1/2] Fetching live prices...")
        try:
            from modules.scraper import get_all_market_data
            market_data = get_all_market_data(write_breadth=False)
            if not market_data:
                print("  ⚠️  No data (market closed?)"); sys.exit(0)
            print(f"  ✅ {len(market_data)} symbols")
        except Exception as e:
            print(f"  ❌ {e}"); sys.exit(1)

    if sym_args:
        market_data = {k: v for k, v in market_data.items() if k in sym_args}
        print(f"  Filtered to: {list(market_data.keys())}")

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n[2/2] Running filter engine...")
    results = run_filter(market_data=market_data, top_n=15)

    if not results:
        print("\n  No candidates passed all gates.")
        print("  Check: bandh, geo score, market state, tech/conf thresholds.\n")
    else:
        print(f"\n  {len(results)} candidates:\n")
        print(f"  {'#':<3} {'Symbol':<10} {'Score':>6} {'Tech':>5} {'RSI':>5} "
              f"{'MACD':<10} {'BB':<14} {'Geo':>4} {'C*':>3} "
              f"{'x':>4} {'Candle':<22} Signal")
        print("  " + "─" * 100)
        for i, c in enumerate(results, 1):
            candle = f"{c.best_candle[:16]}(T{c.candle_tier})" if c.best_candle else "—"
            print(
                f"  {i:<3} {c.symbol:<10} {c.composite_score:>6.1f} "
                f"{c.tech_score:>5} {c.rsi_14:>5.1f} "
                f"{c.macd_cross:<10} {c.bb_signal:<14} "
                f"{c.combined_geo:>+4} {'✓' if c.cstar_signal else ' ':>3} "
                f"{c.sector_mult:>4.2f} {candle:<22} {c.primary_signal}"
            )

        c0 = results[0]
        print(f"\n  Market: {c0.market_state} | Geo: {c0.combined_geo:+d} | "
              f"Bandh: {c0.bandh_today} | IPO: {c0.ipo_drain}")

        print(f"\n  Gemini-ready format (top 3):")
        print("  " + "─" * 70)
        for c in results[:3]:
            print(f"  {format_candidate_for_gemini(c)}")

    print("\n" + "=" * 70 + "\n")
