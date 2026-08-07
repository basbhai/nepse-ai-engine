"""
validation_v2_merge.py
─────────────────────────────────────────────────────────────────────────────
Diagnostic validator for the dual-engine (v1/v2) merge logic in
filter_engine.run_filter(). Re-run this after any future change to
filter_v2.py or run_filter()'s rank-and-trim / merge section.

Does NOT touch FILTER_V2_ENABLED or any pipeline file — purely reads
current settings + historical indicator/price data and asserts the merge
invariants documented in run_filter():
  - v2-only candidates carry v1's preserved opinion in co_flagged_by and
    have v2's signal as the operative primary_signal.
  - BOTH candidates carry v2's opinion in co_flagged_by.

Usage: python validation_v2_merge.py
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from filter_engine import run_filter
from sheets import read_today_indicators

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TEST_DATES = ["2026-06-24", "2026-06-19"]  # fallback date tried if first is inconclusive


def _build_market_data():
    """Same synthetic-price construction as filter_engine.py's --dry-run CLI path."""
    from modules.indicators import HistoryCache, DEFAULT_LOAD_PERIODS
    from modules.scraper import PriceRow

    cache = HistoryCache()
    if not cache.load(periods=DEFAULT_LOAD_PERIODS):
        return {}
    return {
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


def run_for_date(test_date: str, market_data: dict):
    indicators = read_today_indicators(test_date)
    if not indicators:
        print(f"No indicators for {test_date} — pick a different real trading date")
        return None

    results = run_filter(market_data=market_data, top_n=10, date=test_date)

    print(f"\n{len(results)} candidates returned for {test_date}\n")
    for c in results:
        print(f"{c.symbol:<10} engine={c.engine_source:<5} "
              f"v1_score={c.composite_score:>6.1f} v1_sig={c.primary_signal:<14} "
              f"v2_score={c.composite_score_v2:>6.1f} v2_sig={c.primary_signal_v2:<18} "
              f"co_flag={c.co_flagged_by}")

    both_count = sum(1 for c in results if c.engine_source == "BOTH")
    v2_only    = [c for c in results if c.engine_source == "v2"]
    v1_only    = [c for c in results if c.engine_source == "v1"]
    print(f"\nBOTH={both_count}  v2-only={len(v2_only)}  v1-only={len(v1_only)}")

    for c in v2_only:
        assert c.co_flagged_by.startswith("v1 opinion:"), \
            f"{c.symbol}: v2-only candidate missing v1's preserved opinion"
        assert c.primary_signal == c.primary_signal_v2, \
            f"{c.symbol}: v2-only candidate should have v2's signal as operative"

    for c in results:
        if c.engine_source == "BOTH":
            assert c.co_flagged_by.startswith("v2 also flagged:"), \
                f"{c.symbol}: BOTH candidate missing v2's preserved opinion"

    print("\nAll assertions passed" if results else "No candidates to assert on")
    return both_count, len(v2_only), len(v1_only)


def main():
    market_data = _build_market_data()
    if not market_data:
        print("HistoryCache empty — cannot build market_data")
        return

    for test_date in TEST_DATES:
        outcome = run_for_date(test_date, market_data)
        if outcome is None:
            continue
        both_count, v2_only_count, _ = outcome
        if both_count > 0 and v2_only_count > 0:
            print(f"\n{test_date}: exercised BOTH and v2-only branches — stopping here.")
            return
        print(f"\n{test_date}: inconclusive (BOTH={both_count}, v2-only={v2_only_count})"
              f" — trying next date if available.")

    print("\nNo test date exercised both the BOTH and v2-only merge branches — "
          "merge logic remains partially unverified on real data.")


if __name__ == "__main__":
    main()
