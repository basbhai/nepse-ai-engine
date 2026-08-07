"""
opus_pattern_finder.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Opus Pattern Finder
Reads momentum_enriched.json, builds a comprehensive structured prompt,
sends to Claude Opus, saves the analysis.

Output: stat_method/output/opus_pattern_analysis.md

Usage:
    cd ~/nepse-engine
    python stat_method/opus_pattern_finder.py
    python stat_method/opus_pattern_finder.py --dry-run   # print prompt only
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [OPUS] %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA SUMMARISATION — compress enriched data into Opus-digestible stats
# ══════════════════════════════════════════════════════════════════════════════

def _pct(val, total):
    return round(val / total * 100, 1) if total > 0 else 0.0


def _avg(vals):
    valid = [v for v in vals if v is not None]
    return round(sum(valid) / len(valid), 2) if valid else None


def _median(vals):
    valid = sorted(v for v in vals if v is not None)
    if not valid:
        return None
    n = len(valid)
    if n % 2 == 0:
        return round((valid[n//2 - 1] + valid[n//2]) / 2, 2)
    return round(valid[n//2], 2)


def _buckets(vals, thresholds, labels):
    """Count values falling in each bucket."""
    counts = defaultdict(int)
    for v in vals:
        if v is None:
            counts["unknown"] += 1
            continue
        placed = False
        for threshold, label in zip(thresholds, labels):
            if v < threshold:
                counts[label] += 1
                placed = True
                break
        if not placed:
            counts[labels[-1]] += 1
    return dict(counts)


def summarise(hits: list[dict]) -> dict:
    """Build a statistical summary of all enriched hits."""
    n = len(hits)

    # ── Forward returns outcome classification
    fwd_5  = [h.get("fwd_5d_pct")  for h in hits]
    fwd_10 = [h.get("fwd_10d_pct") for h in hits]
    fwd_20 = [h.get("fwd_20d_pct") for h in hits]

    has_fwd_5  = [v for v in fwd_5  if v is not None]
    has_fwd_10 = [v for v in fwd_10 if v is not None]
    has_fwd_20 = [v for v in fwd_20 if v is not None]

    # ── Sector breakdown
    sectors = Counter(h.get("sector") or "Unknown" for h in hits)

    # ── Gain distribution
    gains = [h["gain_15d_pct"] for h in hits]

    # ── Volume ratio on trigger day
    vol_ratios = [h.get("vol_ratio") for h in hits]

    # ── Technical state on trigger day
    ema_trends    = Counter(h.get("ema_trend")       or "UNKNOWN" for h in hits)
    rsi_signals   = Counter(h.get("rsi_signal")      or "UNKNOWN" for h in hits)
    macd_crosses  = Counter(h.get("macd_cross")      or "UNKNOWN" for h in hits)
    bb_signals    = Counter(h.get("bb_signal")       or "UNKNOWN" for h in hits)
    obv_trends    = Counter(h.get("obv_trend")       or "UNKNOWN" for h in hits)
    ema_20_50     = Counter(h.get("ema_20_50_cross") or "NONE"    for h in hits)
    ema_50_200    = Counter(h.get("ema_50_200_cross")or "NONE"    for h in hits)

    # RSI distribution
    rsi_vals = [h.get("rsi_14") for h in hits]

    # bb_pct_b distribution
    bb_pct_b_vals = [h.get("bb_pct_b") for h in hits]

    # ── Floorsheet signals
    instit_flags = Counter(h.get("fs_institutional_flag") or "no_data" for h in hits)
    buyer_press  = [h.get("fs_buyer_pressure")      for h in hits]
    broker_conc  = [h.get("fs_broker_concentration") for h in hits]
    pre_acc_bp   = [h.get("pre_move_avg_buyer_pressure") for h in hits]
    pre_inst_days= [h.get("pre_move_institutional_days") for h in hits]

    # ── 52-week range position
    pos_52w = [h.get("pct_52w_range") for h in hits]

    # ── Fundamentals
    roe_vals = [h.get("fund_roe")      for h in hits]
    eps_vals = [h.get("fund_eps")      for h in hits]
    pe_vals  = [h.get("fund_pe_ratio") for h in hits]
    npl_vals = [h.get("fund_npl")      for h in hits]

    # ── Month distribution
    by_month = Counter(h["trigger_date"][:7] for h in hits)

    # ── Forward return: continue vs reverse
    def outcome_split(fwd_list, threshold=0):
        cont = sum(1 for v in fwd_list if v is not None and v > threshold)
        rev  = sum(1 for v in fwd_list if v is not None and v <= threshold)
        unk  = sum(1 for v in fwd_list if v is None)
        tot  = cont + rev
        return {
            "continue": cont,
            "reverse":  rev,
            "unknown":  unk,
            "continue_pct": _pct(cont, tot),
            "reverse_pct":  _pct(rev,  tot),
        }

    return {
        "total_hits": n,
        "unique_symbols": len(set(h["symbol"] for h in hits)),
        "date_range": {
            "first": min(h["trigger_date"] for h in hits),
            "last":  max(h["trigger_date"] for h in hits),
        },
        "gain_distribution": {
            "avg_gain_pct":    _avg(gains),
            "median_gain_pct": _median(gains),
            "15_20pct":  sum(1 for g in gains if 15 <= g < 20),
            "20_30pct":  sum(1 for g in gains if 20 <= g < 30),
            "30_50pct":  sum(1 for g in gains if 30 <= g < 50),
            "50pct_plus":sum(1 for g in gains if g >= 50),
        },
        "forward_returns": {
            "fwd_5d":  {
                "avg":    _avg(has_fwd_5),
                "median": _median(has_fwd_5),
                "n":      len(has_fwd_5),
                **outcome_split(has_fwd_5),
            },
            "fwd_10d": {
                "avg":    _avg(has_fwd_10),
                "median": _median(has_fwd_10),
                "n":      len(has_fwd_10),
                **outcome_split(has_fwd_10),
            },
            "fwd_20d": {
                "avg":    _avg(has_fwd_20),
                "median": _median(has_fwd_20),
                "n":      len(has_fwd_20),
                **outcome_split(has_fwd_20),
            },
        },
        "sector_breakdown": dict(sectors.most_common()),
        "hits_by_month":    dict(sorted(by_month.items())),
        "volume_ratio": {
            "avg":    _avg(vol_ratios),
            "median": _median(vol_ratios),
            "above_2x": sum(1 for v in vol_ratios if v is not None and v >= 2.0),
            "above_3x": sum(1 for v in vol_ratios if v is not None and v >= 3.0),
        },
        "technical_state": {
            "ema_trend":      dict(ema_trends),
            "rsi_signal":     dict(rsi_signals),
            "rsi_avg":        _avg(rsi_vals),
            "rsi_median":     _median(rsi_vals),
            "rsi_above_60":   sum(1 for v in rsi_vals if v is not None and v > 60),
            "rsi_below_40":   sum(1 for v in rsi_vals if v is not None and v < 40),
            "macd_cross":     dict(macd_crosses),
            "bb_signal":      dict(bb_signals),
            "bb_pct_b_avg":   _avg(bb_pct_b_vals),
            "bb_pct_b_above_1": sum(1 for v in bb_pct_b_vals if v is not None and v > 1.0),
            "obv_trend":      dict(obv_trends),
            "ema_20_50_cross":dict(ema_20_50),
            "ema_50_200_cross":dict(ema_50_200),
        },
        "floorsheet": {
            "institutional_flag": dict(instit_flags),
            "buyer_pressure_avg":          _avg(buyer_press),
            "broker_concentration_avg":    _avg(broker_conc),
            "pre_move_buyer_pressure_avg": _avg(pre_acc_bp),
            "pre_move_institutional_days_avg": _avg(pre_inst_days),
            "high_concentration_hits": sum(
                1 for v in broker_conc if v is not None and v > 0.4
            ),
        },
        "52w_range_position": {
            "avg_pct":    _avg(pos_52w),
            "median_pct": _median(pos_52w),
            "near_52w_high_above_80pct": sum(1 for v in pos_52w if v is not None and v > 80),
            "mid_range_40_80pct":        sum(1 for v in pos_52w if v is not None and 40 <= v <= 80),
            "lower_range_below_40pct":   sum(1 for v in pos_52w if v is not None and v < 40),
        },
        "fundamentals": {
            "symbols_with_data": sum(1 for h in hits if h.get("fund_eps") is not None),
            "roe_avg":    _avg(roe_vals),
            "eps_avg":    _avg(eps_vals),
            "pe_avg":     _avg(pe_vals),
            "npl_avg":    _avg(npl_vals),
            "high_roe_above_15pct": sum(1 for v in roe_vals if v is not None and v > 15),
            "negative_eps":         sum(1 for v in eps_vals if v is not None and v < 0),
        },
    }


def sample_hits(hits: list[dict], n: int = 30) -> list[dict]:
    """
    Pick a representative sample: top gainers, mid gainers, examples
    with good floorsheet data, examples with institutional flags.
    Strip bulk OHLCV columns to keep prompt compact.
    """
    KEEP_COLS = [
        "symbol", "trigger_date", "sector",
        "close", "close_15d_ago", "gain_15d_pct",
        "vol_ratio", "conf_score", "pct_52w_range",
        "rsi_14", "rsi_signal", "ema_trend",
        "ema_20", "ema_50", "ema_200",
        "ema_20_50_cross", "ema_50_200_cross",
        "macd_line", "macd_signal", "macd_histogram", "macd_cross",
        "bb_pct_b", "bb_signal", "atr_pct",
        "obv_trend",
        "trajectory",
        "fs_buyer_pressure", "fs_seller_pressure",
        "fs_broker_concentration", "fs_institutional_flag",
        "fs_large_order_pct",
        "pre_move_avg_buyer_pressure", "pre_move_institutional_days",
        "pre_move_floorsheet_days_avail",
        "fund_eps", "fund_roe", "fund_pe_ratio", "fund_npl",
        "fwd_5d_pct", "fwd_10d_pct", "fwd_20d_pct",
    ]

    def slim(h):
        return {k: h.get(k) for k in KEEP_COLS}

    # Top 10 by gain
    top_gain = sorted(hits, key=lambda h: h["gain_15d_pct"], reverse=True)[:10]

    # 10 with institutional flag = true
    inst = [h for h in hits if h.get("fs_institutional_flag") == "true"][:10]

    # 10 random from the rest
    import random
    rest = [h for h in hits if h not in top_gain and h not in inst]
    random.seed(42)
    sample_rest = random.sample(rest, min(10, len(rest)))

    combined = {h["symbol"] + h["trigger_date"]: h
                for h in (top_gain + inst + sample_rest)}
    return [slim(h) for h in list(combined.values())[:n]]


def build_prompt(summary: dict, sample: list[dict], meta: dict) -> str:
    """Build the full structured Opus prompt."""

    summary_json = json.dumps(summary, indent=2)
    sample_json  = json.dumps(sample,  indent=2)

    prompt = f"""You are a quantitative analyst specialising in the Nepal Stock Exchange (NEPSE).

I have run a momentum screener on NEPSE price data from {meta['start_date']} to present.
The screener flagged every (symbol, date) pair where the closing price was ≥{meta['min_gain_pct']}% above
its closing price exactly {meta['lookback_days']} trading days earlier.

For each flagged event I have enriched the data with:
- Full technical indicators at trigger date computed from raw OHLCV (RSI-14, EMA 20/50/200, MACD 12/26/9, Bollinger Bands, ATR-14, OBV trend)
- Indicator trajectory: snapshots at day −15, −12, −9, −6, −3, 0 — each with close, volume, RSI, MACD histogram, BB %B, OBV trend — showing how the move built up
- Floorsheet accumulation signals (buyer pressure, broker concentration, institutional flag, large order %)
- Pre-move accumulation: avg buyer pressure and institutional trading days in the 15 trading days BEFORE the trigger
- Fundamentals (EPS, ROE, PE ratio, NPL) with at least 1-quarter lag
- Sector classification
- Forward returns at +5, +10, +20 trading days from the trigger date

DATASET SUMMARY (all {summary['total_hits']} hits, {summary['unique_symbols']} unique symbols):
{summary_json}

REPRESENTATIVE SAMPLE ({len(sample)} hits — top gainers, institutional-flagged, and random):
{sample_json}

────────────────────────────────────────────────────────────────
YOUR TASK: Find every meaningful pattern in this data. Be specific and quantitative.

Address ALL of the following:

1. CONTINUATION vs REVERSAL PATTERN
   - What % of these momentum moves continue vs reverse at D+5, D+10, D+20?
   - Are there measurable technical conditions at the trigger date that predict continuation?
   - What is the median forward return, and is there a fat tail on either side?

2. TECHNICAL SETUP AT TRIGGER
   - What EMA configuration is most common at trigger? (ABOVE_ALL vs MIXED vs BELOW_ALL)
   - RSI: are these stocks overbought at trigger, or mid-range? Does RSI level predict continuation?
   - MACD: does a BULLISH cross at trigger correlate with better forward returns vs no cross?
   - Bollinger: are most triggers happening at or above the upper band (breakout exhaustion) or within bands?
   - OBV: does RISING OBV at trigger predict better outcomes?

3. TRAJECTORY ANALYSIS (day −15 to day 0 snapshots)
   - What is the typical RSI path? Does it rise steadily, spike late, or show divergence?
   - MACD histogram trajectory: expanding (accelerating momentum) or contracting (fading)?
   - BB %B trajectory: gradual push to upper band or sudden breakout in final 3 days?
   - OBV trajectory: does it turn RISING before or after the price move starts?
   - Are there trajectory patterns that predict better continuation at D+20?
     (e.g. RSI rising through the full 15 days vs RSI spiking only in final 3 days)

4. VOLUME SIGNATURE
   - What is the typical volume ratio (trigger day vs 15 days ago)?
   - Is there a volume threshold above which continuation is more likely?
   - How does pre-move floorsheet buyer pressure correlate with forward returns?

5. FLOORSHEET / ACCUMULATION SIGNALS
   - What % of triggers have an institutional_flag = true on the trigger day?
   - What is the typical broker_concentration at trigger? High concentration = operator-driven moves?
   - Do stocks with high pre_move_institutional_days (smart money buying before the move) show better continuation?
   - Large order % on trigger day: does this predict anything?

6. SECTOR PATTERNS
   - Which sectors produce the most hits? Is this random (sector size) or disproportionate?
   - Which sectors show better post-trigger continuation?
   - Are there sectors to avoid (high reversal rate)?

7. 52-WEEK RANGE POSITION
   - Are these moves happening near 52-week highs (breakouts) or from mid-range (recoveries)?
   - Does the position in the 52-week range predict forward return direction?

8. FUNDAMENTALS OVERLAY
   - Do stocks with positive EPS or high ROE show different continuation rates?
   - Is PE ratio relevant — are these expensive stocks or value stocks making these moves?
   - For banking stocks: does NPL level matter?

9. TIMING / SEASONALITY
   - Are there months with higher hit frequency? Does hit frequency correlate with market regime?
   - Are there months with systematically better forward returns?

10. ACTIONABLE SIGNAL CRITERIA
    Based on everything above, define the BEST-CASE profile for a stock that has just triggered
    this momentum screen — the combination of technical + trajectory + floorsheet + fundamental
    conditions that historically predicted the best D+20 outcome. Be specific with thresholds.

    Also define the WARNING PROFILE — conditions at trigger that predict reversal.

11. WHAT YOUR SYSTEM SHOULD DO
    Given these findings, what specific changes (if any) should the NEPSE AI Engine make?
    Consider:
    - Should 15-day +15% momentum be added as a positive or negative signal in the scoring?
    - At what stage (filter_engine, gemini_filter, claude_analyst) should it be applied?
    - Should there be a sector-specific version of this signal?
    - Any floorsheet conditions that should gate the signal?

Be direct. Use numbers. Where patterns are weak or sample size is too small, say so.
Do not speculate beyond what the data shows. Flag any confounding factors.
"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# OPUS API CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_opus(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    log.info("Sending to Opus API (~%d chars in prompt)...", len(prompt))
    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt only, do not call Opus")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="Number of example hits to include in prompt")
    args = parser.parse_args()

    in_path = Path(__file__).parent / "output" / "momentum_enriched.json"
    if not in_path.exists():
        log.error("momentum_enriched.json not found. Run momentum_enricher.py first.")
        sys.exit(1)

    with open(in_path) as f:
        data = json.load(f)

    hits = data["hits"]
    meta = data["meta"]
    log.info("Loaded %d enriched hits", len(hits))

    summary = summarise(hits)
    sample  = sample_hits(hits, n=args.sample_size)
    prompt  = build_prompt(summary, sample, meta)

    log.info("Prompt length: %d chars", len(prompt))

    if args.dry_run:
        print("\n" + "=" * 70)
        print("PROMPT PREVIEW (first 3000 chars):")
        print("=" * 70)
        print(prompt[:3000])
        print("\n... [truncated] ...")

        # Print summary stats
        print("\n" + "=" * 70)
        print("SUMMARY STATS:")
        print("=" * 70)
        print(json.dumps(summary, indent=2))
        return

    response = call_opus(prompt)

    out_dir  = Path(__file__).parent / "output"
    out_path = out_dir / "opus_pattern_analysis.md"

    with open(out_path, "w") as f:
        f.write(f"# NEPSE Momentum Pattern Analysis — Opus\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {meta['total_hits']} hits, {meta['unique_symbols']} symbols, ")
        f.write(f"from {meta['start_date']}, min gain {meta['min_gain_pct']}%\n\n")
        f.write("---\n\n")
        f.write(response)

    # Also save the summary JSON for reference
    summary_path = out_dir / "momentum_summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Analysis written to %s", out_path)
    log.info("Summary stats written to %s", summary_path)


if __name__ == "__main__":
    main()
