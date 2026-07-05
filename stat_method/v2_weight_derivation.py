# -*- coding: utf-8 -*-
"""
v2_weight_derivation.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Research Tool (filter_engine_v2 empirical weight derivation)

Non-overlapping event scan across price_history. For every symbol, walks
trading days in fixed FORWARD_DAYS-day tiles. Each tile anchor (T0) is
labeled by its forward return, and scored by the indicator SLOPE over the
SIGNAL_DAYS days immediately preceding T0 (T-5 -> T0).

This intentionally mirrors modules/spring_loading_screener.py's non-overlapping
tiling (NOT a sliding window). A sliding day-by-day scan is what produced the
24.7x sample inflation already diagnosed in winner_loser_events.csv. Advancing
by the full FORWARD_DAYS block per tile keeps each labeled event independent.

Labels (forward FORWARD_DAYS-day return from T0):
    WINNER     — forward_return_pct >= WINNER_THRESHOLD    (+10%)
    LOSER      — forward_return_pct <= LOSER_THRESHOLD     (-5%)
    BIG_LOSER  — forward_return_pct <= BIG_LOSER_THRESHOLD (-10%, subset of LOSER)
    NEUTRAL    — everything else (kept in raw CSV for completeness, excluded
                 from the winner/loser statistical comparison)

Corporate-action guard: |forward_return_pct| > CORP_ACTION_ABS_PCT discarded
entirely (split/bonus/rights artifact, not a real price move) — same rule
already applied when building winner_loser_events.csv.

Score features — slope over the SIGNAL_DAYS days ending at T0, using the
SAME average-of-differences formula as filter_engine._compute_momentum_status()
(rsi_slope_3d / macd_hist_slope / bb_pct_b_slope), so v2 scorers can reuse the
identical convention once wired into the live pipeline:
    rsi_slope_5d
    macd_hist_slope_5d
    bb_pctb_slope_5d
    obv_slope_5d        (normalized: (end-start)/max(abs(start),1) — same
                          convention as modules/spring_loading_screener.py)
    ema_spread_slope_5d (NEW — EMA20-EMA50 spread as % of price, then sloped.
                          Not in _compute_momentum_status() today; candidate
                          feature for v2 only, flagged as new.)

Output — four CSVs under ~/nepse-engine/output/v2_weight_derivation/:
    events_<timestamp>.csv       — every labeled event, slope summary + label
    summary_<timestamp>.csv      — per-indicator Mann-Whitney U + Cliff's delta,
                                    WINNER vs (LOSER+BIG_LOSER combined) as the
                                    primary comparison, WINNER vs BIG_LOSER-only
                                    as a secondary tail-effect check
    winners_raw_<timestamp>.csv  — WINNER events only, wide-format day-by-day
                                    values (close/RSI/MACD-hist/BB%B/OBV/EMA-
                                    spread) for T-4..T0 — the actual sequence
                                    each slope in events_*.csv was computed from
    losers_raw_<timestamp>.csv   — same, for LOSER+BIG_LOSER events (filter by
                                    the label column to isolate BIG_LOSER only)

This script is READ-ONLY. No DB writes. Nothing here is wired into the live
pipeline — output is a CANDIDATE INDICATOR_WEIGHTS_V2 table for manual
review, not an auto-applied config. Purged/time-based holdout validation is
a separate follow-up step, not run here.

Run:
    python -m research.v2_weight_derivation
    python -m research.v2_weight_derivation --forward-days=20 --signal-days=5
    python -m research.v2_weight_derivation --dry-run   (first 15 symbols only)

Import rule: from sheets import ... — NEVER from db import ...
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import os
import csv
from datetime import datetime

import pandas as pd
import pandas_ta as ta

from config import NST
from sheets import run_raw_sql

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE          = "2021-01-01"
FORWARD_DAYS        = 20     # label window (forward return horizon from T0)
SIGNAL_DAYS         = 5      # backward scoring window (T-5 -> T0)
MIN_HISTORY_DAYS    = 60     # min prior bars needed before T0 for stable indicators
WINNER_THRESHOLD    = 10.0
LOSER_THRESHOLD     = -5.0
BIG_LOSER_THRESHOLD = -10.0
CORP_ACTION_ABS_PCT = 100.0  # |return| beyond this = corporate action artifact, discard

DRY_RUN = "--dry-run" in sys.argv

for _a in sys.argv[1:]:
    if _a.startswith("--forward-days="):
        FORWARD_DAYS = int(_a.split("=")[1])
    elif _a.startswith("--signal-days="):
        SIGNAL_DAYS = int(_a.split("=")[1])

OUTPUT_DIR = os.path.expanduser("~/nepse-engine/output/v2_weight_derivation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PRICE HISTORY  (same pattern as spring_loading_precision.py)
# ══════════════════════════════════════════════════════════════════════════════

def _load_all_price_history() -> dict[str, pd.DataFrame]:
    """Load all price history since START_DATE. Returns {symbol: DataFrame}."""
    log.info("Loading price_history since %s...", START_DATE)
    rows = run_raw_sql(
        """
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s
        ORDER BY symbol, date ASC
        """,
        (START_DATE,),
    )
    log.info("Loaded %d rows", len(rows))

    def _f(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    sym_data: dict[str, list] = {}
    for r in rows:
        sym   = str(r["symbol"]).upper()
        close = _f(r["close"]) or _f(r["ltp"])
        if close is None:
            continue
        sym_data.setdefault(sym, []).append({
            "date":   r["date"],
            "open":   _f(r["open"]) or close,
            "high":   _f(r["high"]) or close,
            "low":    _f(r["low"])  or close,
            "close":  close,
            "volume": _f(r["volume"]) or 0.0,
        })

    result = {}
    for sym, records in sym_data.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_HISTORY_DAYS + SIGNAL_DAYS + FORWARD_DAYS:
            result[sym] = df

    log.info("Symbols with enough history: %d", len(result))
    if DRY_RUN:
        result = dict(list(result.items())[:15])
        log.info("[DRY RUN] Trimmed to %d symbols", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SLOPE FEATURES ON A SLICE ENDING AT T0
# ══════════════════════════════════════════════════════════════════════════════

def _avg_slope(vals: list) -> float:
    """
    Average-of-differences slope, ascending chronological order.
    Same formula as filter_engine._compute_momentum_status() so v2 can reuse
    the identical convention once wired live.
    """
    if len(vals) < 2:
        return 0.0
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    return sum(diffs) / len(diffs)


def _compute_features_at_t0(hist_df: pd.DataFrame):
    """
    hist_df: full price history UP TO AND INCLUDING T0 (so indicators are
    computed correctly), sorted ascending by date.
    Returns dict of slope + raw-value features over the last SIGNAL_DAYS
    bars, or None if indicators aren't computable yet.
    """
    if len(hist_df) < MIN_HISTORY_DAYS:
        return None

    close  = hist_df["close"]
    high   = hist_df["high"]
    low    = hist_df["low"]
    volume = hist_df["volume"]

    try:
        rsi   = ta.rsi(close, length=14)
        macd  = ta.macd(close, fast=12, slow=26, signal=9)
        bb    = ta.bbands(close, length=20, std=2)
        obv   = ta.obv(close, volume)
        ema20 = ta.ema(close, length=20)
        ema50 = ta.ema(close, length=50)
    except Exception as exc:
        log.debug("_compute_features_at_t0: indicator calc failed: %s", exc)
        return None

    if rsi is None or macd is None or bb is None or obv is None or ema20 is None or ema50 is None:
        return None

    hist_col = macd.get("MACDh_12_26_9")
    pctb_col = bb.get("BBP_20_2.0")
    if hist_col is None or pctb_col is None:
        return None

    close_vals = close.iloc[-SIGNAL_DAYS:].tolist()
    rsi_vals   = rsi.dropna().iloc[-SIGNAL_DAYS:].tolist()
    hist_vals  = hist_col.dropna().iloc[-SIGNAL_DAYS:].tolist()
    pctb_vals  = pctb_col.dropna().iloc[-SIGNAL_DAYS:].tolist()
    obv_vals   = obv.dropna().iloc[-SIGNAL_DAYS:].tolist()

    ema_spread      = (ema20 - ema50) / close * 100.0
    ema_spread_vals = ema_spread.dropna().iloc[-SIGNAL_DAYS:].tolist()

    lengths = [len(close_vals), len(rsi_vals), len(hist_vals), len(pctb_vals),
               len(obv_vals), len(ema_spread_vals)]
    if min(lengths) < SIGNAL_DAYS:
        return None

    obv_start = obv_vals[0]
    obv_end   = obv_vals[-1]
    obv_slope = (obv_end - obv_start) / max(abs(obv_start), 1)

    summary = {
        "rsi_at_t0":           round(rsi_vals[-1], 2),
        "rsi_slope_5d":        round(_avg_slope(rsi_vals), 4),
        "macd_hist_at_t0":     round(hist_vals[-1], 4),
        "macd_hist_slope_5d":  round(_avg_slope(hist_vals), 6),
        "bb_pctb_at_t0":       round(pctb_vals[-1], 4),
        "bb_pctb_slope_5d":    round(_avg_slope(pctb_vals), 5),
        "obv_slope_5d":        round(obv_slope, 6),
        "ema_spread_at_t0":    round(ema_spread_vals[-1], 3),
        "ema_spread_slope_5d": round(_avg_slope(ema_spread_vals), 5),
    }

    # Raw day-by-day values, oldest -> newest: t_minus{N-1} .. t0.
    # Feeds the separate winners_raw_*.csv / losers_raw_*.csv dumps — the
    # actual sequence each slope above was computed from, not just the slope.
    day_labels = [f"t_minus{SIGNAL_DAYS - 1 - i}" for i in range(SIGNAL_DAYS - 1)] + ["t0"]
    raw = {}
    for i, day in enumerate(day_labels):
        raw[f"close_{day}"]      = round(close_vals[i], 3)
        raw[f"rsi_{day}"]        = round(rsi_vals[i], 3)
        raw[f"macd_hist_{day}"]  = round(hist_vals[i], 5)
        raw[f"bb_pctb_{day}"]    = round(pctb_vals[i], 4)
        raw[f"obv_{day}"]        = round(obv_vals[i], 2)
        raw[f"ema_spread_{day}"] = round(ema_spread_vals[i], 4)

    return summary, raw


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NON-OVERLAPPING EVENT SCAN PER SYMBOL
# ══════════════════════════════════════════════════════════════════════════════

def _classify_label(forward_return_pct: float):
    if abs(forward_return_pct) > CORP_ACTION_ABS_PCT:
        return None  # corporate action artifact — discard entirely
    if forward_return_pct >= WINNER_THRESHOLD:
        return "WINNER"
    if forward_return_pct <= BIG_LOSER_THRESHOLD:
        return "BIG_LOSER"
    if forward_return_pct <= LOSER_THRESHOLD:
        return "LOSER"
    return "NEUTRAL"


def _scan_symbol(sym: str, df: pd.DataFrame) -> tuple:
    """
    Non-overlapping tiling: T0 anchors advance by FORWARD_DAYS each step.
    Requires MIN_HISTORY_DAYS of prior bars before T0 and FORWARD_DAYS bars after.

    Returns (summary_events, raw_events) — parallel lists, one entry per T0.
    summary_events feed events_*.csv (slope-only, unchanged shape).
    raw_events feed winners_raw_*.csv / losers_raw_*.csv (day-by-day values).
    """
    summary_events = []
    raw_events = []
    n = len(df)
    t0_idx = MIN_HISTORY_DAYS

    while t0_idx + FORWARD_DAYS < n:
        t0_close  = float(df.iloc[t0_idx]["close"])
        fwd_close = float(df.iloc[t0_idx + FORWARD_DAYS]["close"])

        if t0_close <= 0:
            t0_idx += FORWARD_DAYS
            continue

        forward_return_pct = (fwd_close - t0_close) / t0_close * 100.0
        label = _classify_label(forward_return_pct)

        if label is not None:
            hist_slice = df.iloc[: t0_idx + 1]  # up to & including T0
            computed = _compute_features_at_t0(hist_slice)
            if computed:
                summary_feats, raw_feats = computed
                base = {
                    "symbol":             sym,
                    "t0_date":            df.iloc[t0_idx]["date"].strftime("%Y-%m-%d"),
                    "forward_return_pct": round(forward_return_pct, 2),
                    "label":              label,
                }
                summary_events.append({**base, **summary_feats})
                raw_events.append({**base, **raw_feats})

        t0_idx += FORWARD_DAYS  # non-overlapping jump — avoids the sample-inflation bug

    return summary_events, raw_events


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLIFF'S DELTA + MANN-WHITNEY U  (no hard scipy dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _cliffs_delta(a: list, b: list) -> float:
    """
    Cliff's delta effect size. +1 = a entirely > b, -1 = a entirely < b, 0 =
    no separation. O(n*m) — fine at these sample sizes.
    """
    if not a or not b:
        return 0.0
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (more - less) / (len(a) * len(b))


def _mann_whitney_u(a: list, b: list):
    """
    Returns (U, p) or None. Cliff's delta above doesn't need scipy and is the
    primary ranking metric — this is a supporting significance check only.
    """
    if not a or not b:
        return None
    try:
        from scipy import stats as _stats
        u, p = _stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ImportError:
        log.warning("scipy not available — reporting Cliff's delta only, no p-value")
        return None
    except Exception as exc:
        log.debug("mannwhitneyu failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    "rsi_slope_5d", "macd_hist_slope_5d", "bb_pctb_slope_5d",
    "obv_slope_5d", "ema_spread_slope_5d",
]


def run() -> dict:
    ts = datetime.now(tz=NST).strftime("%Y%m%d_%H%M%S")
    log.info("=" * 70)
    log.info("v2_weight_derivation — forward=%dd, signal=%dd%s",
              FORWARD_DAYS, SIGNAL_DAYS, " [DRY RUN]" if DRY_RUN else "")

    sym_data = _load_all_price_history()
    if not sym_data:
        log.error("No price data — aborting")
        return {}

    all_events: list = []
    all_raw: list = []
    for sym, df in sym_data.items():
        summary_evts, raw_evts = _scan_symbol(sym, df)
        all_events.extend(summary_evts)
        all_raw.extend(raw_evts)

    log.info("Total labeled events: %d", len(all_events))
    if not all_events:
        log.warning("No events produced — check thresholds / history depth")
        return {}

    events_path = os.path.join(OUTPUT_DIR, f"events_{ts}.csv")
    with open(events_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_events[0].keys()))
        writer.writeheader()
        writer.writerows(all_events)
    log.info("Wrote %d events to %s", len(all_events), events_path)

    # ── Raw day-by-day dumps, split winner vs loser ─────────────────────────
    winners_raw_path = losers_raw_path = None
    winners_raw = [e for e in all_raw if e["label"] == "WINNER"]
    losers_raw  = [e for e in all_raw if e["label"] in ("LOSER", "BIG_LOSER")]

    if winners_raw:
        winners_raw_path = os.path.join(OUTPUT_DIR, f"winners_raw_{ts}.csv")
        with open(winners_raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(winners_raw[0].keys()))
            writer.writeheader()
            writer.writerows(winners_raw)
        log.info("Wrote %d winner raw events to %s", len(winners_raw), winners_raw_path)

    if losers_raw:
        losers_raw_path = os.path.join(OUTPUT_DIR, f"losers_raw_{ts}.csv")
        with open(losers_raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(losers_raw[0].keys()))
            writer.writeheader()
            writer.writerows(losers_raw)
        log.info("Wrote %d loser raw events (LOSER+BIG_LOSER) to %s",
                  len(losers_raw), losers_raw_path)

    winners    = [e for e in all_events if e["label"] == "WINNER"]
    losers_all = [e for e in all_events if e["label"] in ("LOSER", "BIG_LOSER")]
    big_losers = [e for e in all_events if e["label"] == "BIG_LOSER"]

    log.info("WINNER=%d  LOSER+BIG_LOSER=%d  (BIG_LOSER only=%d)",
              len(winners), len(losers_all), len(big_losers))

    summary_rows = []
    for feat in FEATURE_COLUMNS:
        w_vals  = [e[feat] for e in winners]
        l_vals  = [e[feat] for e in losers_all]
        bl_vals = [e[feat] for e in big_losers]

        delta_primary   = _cliffs_delta(w_vals, l_vals)
        mw_primary      = _mann_whitney_u(w_vals, l_vals)
        delta_secondary = _cliffs_delta(w_vals, bl_vals) if bl_vals else 0.0

        summary_rows.append({
            "feature":                    feat,
            "n_winner":                   len(w_vals),
            "n_loser_combined":           len(l_vals),
            "n_big_loser":                len(bl_vals),
            "cliffs_delta_primary":       round(delta_primary, 4),
            "mannwhitney_u":              mw_primary[0] if mw_primary else "",
            "mannwhitney_p":              mw_primary[1] if mw_primary else "",
            "cliffs_delta_vs_big_loser":  round(delta_secondary, 4),
        })

    summary_rows.sort(key=lambda r: abs(r["cliffs_delta_primary"]), reverse=True)

    log.info("── Ranked by |Cliff's delta| (WINNER vs LOSER+BIG_LOSER) ──")
    for r in summary_rows:
        log.info("  %-20s delta=%+.4f  p=%s  n=%d/%d",
                  r["feature"], r["cliffs_delta_primary"],
                  r["mannwhitney_p"], r["n_winner"], r["n_loser_combined"])

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{ts}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info("Wrote summary to %s", summary_path)

    # Suggested v2 weights — normalized |delta|. NOT auto-applied anywhere;
    # this is a candidate table for manual review before it goes anywhere
    # near filter_v2.py / INDICATOR_WEIGHTS_V2.
    total_abs_delta = sum(abs(r["cliffs_delta_primary"]) for r in summary_rows) or 1.0
    suggested_weights = {
        r["feature"]: round(abs(r["cliffs_delta_primary"]) / total_abs_delta, 3)
        for r in summary_rows
    }
    log.info("── Suggested INDICATOR_WEIGHTS_V2 (normalized |delta|, for review only) ──")
    for k, v in suggested_weights.items():
        log.info("  %-20s %.3f", k, v)

    return {
        "events_path":       events_path,
        "summary_path":      summary_path,
        "winners_raw_path":  winners_raw_path,
        "losers_raw_path":   losers_raw_path,
        "n_events":          len(all_events),
        "n_winner":          len(winners),
        "n_loser":           len(losers_all),
        "suggested_weights": suggested_weights,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
