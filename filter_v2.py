"""
filter_v2.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — filter_v1 / filter_v2 split, Phase 1

v2 = PROGRESSION scoring engine. Every function here reads the last 6
trading days (T-5 → T0) and scores the DIRECTION OF CHANGE, not the T0
snapshot. Same 5 indicator categories as v1 (macd/bollinger/sma/stochastic/
rsi) — only what each one measures changes, per project instruction.

STATUS: EXPERIMENTAL. Not yet wired into run_filter(). Built and ready to
validate standalone (see __main__ below), but NOT driving any live paper
trade decision until Phase 3 wiring + the Phase 0 decisions (circuit
breaker / capital pool split) are settled.

─────────────────────────────────────────────────────────────────────────────
WEIGHT PROVENANCE — read before touching these numbers
─────────────────────────────────────────────────────────────────────────────
Source: Mann-Whitney U / Cliff's delta analysis, 17,767 events, 6-day
backward window (T-5→T0), winners vs combined losers (LOSE_10+LOSE_5).

Headline finding from that analysis, repeated here on purpose:
    EVERY feature tested falls below the conventional "negligible effect"
    threshold (|δ| = 0.147). Best performer (OBV) is δ=+0.061 — a 53.1%
    probability of superiority. This is barely better than a coin flip.
    Only OBV survives Bonferroni correction across 60 tests.

These weights are RELATIVE RANKINGS AMONG WEAK SIGNALS, not a validated
edge. Treat v2's composite score as "which weak signal is currently least
weak," not as a trading edge in its own right. Walk-forward validation
(fit on 2021-2024, score 2025-2026, symbol-blocked resampling) has NOT
been done. Do not raise these weights' influence in _compute_composite_score
or gates until that validation exists.

Weight derivation — this is a judgment call, not a verbatim copy of either
table the study proposed, for two specific reasons:

  1. The study's "MACD line/signal" weight (0.22 in its strict table) was
     built on spread-AT-T0 — a snapshot/state feature, not a slope. That's
     the wrong kind of feature for an engine whose entire premise is
     progression-over-state. The only genuine MACD *slope* feature the
     study has is histogram net change, which is also the feature that
     collapsed from δ=+0.058 (2021-23) to δ=+0.001 (2024-26) — real
     signal decay, not noise. Kept, but downweighted hard rather than
     zeroed, since it's the only valid MACD trend feature available.
  2. EMA/SMA crossover (cross_net) was tested and is NOT significant
     (95% CI crosses zero — it "dies" under the study's own strict
     criterion). Kept at a token weight only to preserve the 5-category
     structure v1 has (per project instruction), not because the data
     supports it. Flagged as the weakest link in this table.

    INDICATOR_WEIGHTS_V2 = {
        "stochastic": 0.35,   # OBV 6-day progression — only Bonferroni survivor
        "bollinger":  0.25,   # BB %B slope — δ=+0.046 combined, stable across eras
        "rsi":        0.25,   # RSI slope — δ=+0.044 combined, stable across eras
        "macd":       0.10,   # MACD histogram slope — real signal, era-unstable → downweighted
        "sma":        0.05,   # EMA crossover net — NOT significant, token weight only
    }

Revisit after: (a) a dedicated EMA-slope study (none exists yet — cross_net
was measured but never the focus), (b) walk-forward validation, (c) enough
v2-tagged paper trades to apply the 30-trade anti-overfitting gate.
─────────────────────────────────────────────────────────────────────────────
Window: 6 days (T-5→T0), matching the empirical study exactly. NOT 7 days —
flagged and confirmed with Hembro 2026-07-03. If you need a 7-day window,
that requires a fresh raw_data_dump.py run first; nothing below is validated
on a 7-day window.
─────────────────────────────────────────────────────────────────────────────
"""

# Empirically-informed weights — see provenance block above. Sums to 1.00.
INDICATOR_WEIGHTS_V2 = {
    "stochastic": 0.35,   # OBV 6-day progression
    "bollinger":  0.25,   # BB %B slope
    "rsi":        0.25,   # RSI slope
    "macd":       0.10,   # MACD histogram slope (downweighted for era instability)
    "sma":        0.05,   # EMA crossover net (not significant — token weight)
}

# Same hold-day mapping as v1 — hold period is a property of the signal type
# (Karki 2023), not of how we detect it. No evidence yet to justify different
# hold periods for slope-triggered vs state-triggered entries.
OPTIMAL_HOLD_DAYS_V2 = {
    "macd":       17,
    "sma":        33,
    "bollinger":  130,
    "stochastic": 4,
}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR SCORING (progression / slope-based, 6-day window T-5→T0)
# ══════════════════════════════════════════════════════════════════════════════

def _score_macd_v2(momentum: dict) -> float:
    """
    Weight 0.10 (downweighted — see module docstring for why).
    Uses macd_hist_slope from filter_common._compute_momentum() —
    already computed on the shared 6-day window, no new loader needed.

    Positive slope = histogram improving = momentum building.
    Scale is provisional (not calibrated against the study's raw
    distribution) — revisit once walk-forward validation exists.
    """
    slope = float(momentum.get("macd_hist_slope", 0.0) or 0.0)

    if slope > 0.015:
        return 90.0
    elif slope > 0.003:
        return 70.0
    elif slope > -0.003:
        return 50.0      # flat — genuinely uninformative, matches the study's weak effect size
    elif slope > -0.015:
        return 30.0
    else:
        return 10.0


def _score_bollinger_v2(momentum: dict) -> float:
    """
    Weight 0.25. Uses bb_pct_b_slope (positive = moving away from lower
    band = recovering). δ=+0.046 combined in the study, one of the two
    most stable indicators across the 2021-23 / 2024-26 era split.
    """
    slope = float(momentum.get("bb_pct_b_slope", 0.0) or 0.0)

    if slope > 0.03:
        return 85.0
    elif slope > 0.01:
        return 65.0
    elif slope > -0.01:
        return 50.0
    elif slope > -0.03:
        return 30.0
    else:
        return 10.0


def _score_rsi_v2(momentum: dict) -> float:
    """
    Weight 0.25. Uses rsi_slope_3d directly — NOT gated by RSI level like
    v1's _score_rsi(). v1 already penalizes RSI-alone entries at the hard
    gate; v2's job here is purely "is RSI improving," independent of where
    it started. δ=+0.044 combined, stable across eras.
    """
    slope = float(momentum.get("rsi_slope_3d", 0.0) or 0.0)

    if slope > 1.5:
        return 85.0
    elif slope > 0.5:
        return 65.0
    elif slope > -0.5:
        return 50.0
    elif slope > -1.5:
        return 30.0
    else:
        return 10.0


def _compute_obv_progression(recent_rows: list[dict]) -> tuple[float, int]:
    """
    OBV 6-day progression from obv_trend history (already loaded by
    filter_common._load_recent_indicators — no new query needed).

    This is a proxy for the study's obv_dir/obv_ols_norm features (which
    were computed from raw signed volume, not available here without a
    separate raw-data pull). What we have instead: how many of the last
    5 available snapshots had obv_trend == RISING. Weaker than the
    study's actual feature, but directionally consistent with it and
    buildable from existing indicators-table history.

    Returns (score_0_100, rising_day_count).
    """
    if not recent_rows:
        return 40.0, 0

    window = recent_rows[:5]
    rising = sum(1 for r in window if str(r.get("obv_trend", "")).upper() == "RISING")
    falling = sum(1 for r in window if str(r.get("obv_trend", "")).upper() == "FALLING")
    n = len(window)
    if n == 0:
        return 40.0, 0

    ratio = rising / n
    if ratio >= 0.8:
        score = 90.0
    elif ratio >= 0.6:
        score = 75.0
    elif rising > falling:
        score = 60.0
    elif rising == falling:
        score = 45.0
    else:
        score = 20.0

    return score, rising


def _score_stochastic_proxy_v2(recent_rows: list[dict]) -> float:
    """
    Weight 0.35 — highest, only Bonferroni-surviving signal in the study.
    OBV progression over the 6-day window (see _compute_obv_progression).
    """
    score, _ = _compute_obv_progression(recent_rows)
    return score


def _compute_ema_progression(recent_rows: list[dict]) -> float:
    """
    EMA trend persistence over the window — proxy for the study's cross_net
    feature (which was NOT significant; see module docstring). How many of
    the last 5 snapshots had ema_trend == ABOVE_ALL, i.e. sustained uptrend
    rather than a single-day cross.

    Requires ema_trend in recent_rows (added to _load_recent_indicators's
    SELECT in filter_common.py as part of this Phase 1 change).
    """
    if not recent_rows:
        return 40.0

    window = recent_rows[:5]
    above_all = sum(1 for r in window if str(r.get("ema_trend", "")).upper() == "ABOVE_ALL")
    below_all = sum(1 for r in window if str(r.get("ema_trend", "")).upper() == "BELOW_ALL")
    n = len(window)
    if n == 0:
        return 40.0

    if above_all >= n - 1:
        return 70.0
    elif above_all > below_all:
        return 55.0
    elif above_all == below_all:
        return 40.0
    else:
        return 20.0


def _score_sma_v2(recent_rows: list[dict]) -> float:
    """
    Weight 0.05 — token weight only. cross_net was tested and found
    non-significant (CI crosses zero). This function exists to preserve
    the 5-category structure, not because the data supports acting on it.
    """
    return _compute_ema_progression(recent_rows)


def compute_indicator_score_v2(
    momentum:     dict,
    recent_rows:  list[dict],
    sector:       str,
) -> tuple[float, str, int]:
    """
    Weighted composite progression score (0–100). Mirrors v1's
    _compute_indicator_score() signature and return shape, so the
    orchestrator can call either engine interchangeably.

    Args:
        momentum:    output of filter_common._compute_momentum() — already
                     computed once per symbol per cycle, shared with v1's
                     gates. No duplicate computation.
        recent_rows: raw 6-7 day indicator history from
                     filter_common._load_recent_indicators() — needed here
                     for OBV and EMA progression (momentum dict doesn't
                     carry those).
        sector:      kept in the signature to match v1 and for a future
                     sector-conditional adjustment, but NOT used yet — no
                     empirical basis for a v2 sector adjustment exists.
                     (v1's hydro RSI halving does NOT carry over here; that
                     was validated on state RSI, not slope RSI. Revisit once
                     there's era/sector-split data on RSI slope specifically.)

    Returns (score, primary_signal, suggested_hold_days).
    """
    scores = {
        "macd":       _score_macd_v2(momentum)              * INDICATOR_WEIGHTS_V2["macd"],
        "bollinger":  _score_bollinger_v2(momentum)          * INDICATOR_WEIGHTS_V2["bollinger"],
        "sma":        _score_sma_v2(recent_rows)             * INDICATOR_WEIGHTS_V2["sma"],
        "stochastic": _score_stochastic_proxy_v2(recent_rows) * INDICATOR_WEIGHTS_V2["stochastic"],
        "rsi":        _score_rsi_v2(momentum)                * INDICATOR_WEIGHTS_V2["rsi"],
    }

    total   = sum(scores.values())
    primary = max(scores, key=lambda k: scores[k])

    signal_labels = {
        "macd":       "MACD_V2",
        "bollinger":  "BB_V2",
        "sma":        "SMA_V2",
        "stochastic": "OBV_MOMENTUM_V2",
        "rsi":        "RSI_V2",
    }

    return round(total, 2), signal_labels[primary], OPTIMAL_HOLD_DAYS_V2.get(primary, 17)


# ══════════════════════════════════════════════════════════════════════════════
# CLI — standalone validation, does NOT touch run_filter() or live trading
#   python filter_v2.py NABIL         → shows v1 vs v2 score for one symbol
#   python filter_v2.py --sample 20   → runs v2 over 20 random gate-passed
#                                        symbols from today's actual filter run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [FILTER_V2] %(message)s")

    print("\n" + "=" * 70)
    print("  filter_v2.py — standalone progression-score check")
    print("  NOT wired into run_filter() — this only validates scoring logic")
    print("=" * 70)

    args = sys.argv[1:]

    try:
        from filter_common import _load_recent_indicators, _compute_momentum
        from filter_v1 import _compute_indicator_score as _v1_score
        from sheets import read_today_indicators
        from datetime import datetime
        from config import NST
    except Exception as e:
        print(f"  Could not import pipeline modules: {e}")
        print("  (expected if run outside the actual repo / without DB access)")
        sys.exit(1)

    date = datetime.now(tz=NST).strftime("%Y-%m-%d")
    indicators_map = read_today_indicators(date) or {}

    if not indicators_map:
        print(f"  No indicators for {date} — run indicators.py first")
        sys.exit(1)

    if args and not args[0].startswith("--"):
        symbols = [a.upper() for a in args]
    else:
        symbols = list(indicators_map.keys())[:20]

    recent_map = _load_recent_indicators(symbols, date, lookback=7)

    print(f"\n  {'Symbol':<10} {'V1 Score':>9} {'V1 Sig':<14} {'V2 Score':>9} {'V2 Sig':<18}")
    print("  " + "─" * 70)
    for sym in symbols:
        ind = indicators_map.get(sym)
        if not ind:
            continue
        recent_rows = recent_map.get(sym, [])
        momentum = _compute_momentum(recent_rows)
        sector = str(ind.get("sector", "others") or "others")

        v1_score, v1_sig, _ = _v1_score(ind, sector, momentum["momentum_status"], momentum["momentum_score"])
        v2_score, v2_sig, _ = compute_indicator_score_v2(momentum, recent_rows, sector)

        print(f"  {sym:<10} {v1_score:>9.1f} {v1_sig:<14} {v2_score:>9.1f} {v2_sig:<18}")

    print("\n" + "=" * 70 + "\n")
