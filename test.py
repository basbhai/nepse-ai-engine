"""
Direct test of the RSI/RSI_V2 fallback gap — no DB, no API calls.
Constructs synthetic candidates and calls the real _keyword_fallback()
function to see exactly what it does with each signal name.
"""
from dataclasses import dataclass

@dataclass
class FakeCandidate:
    symbol: str
    primary_signal: str
    composite_score: float = 80.0
    tech_score: int = 65
    conf_score: float = 50.0
    macd_cross: str = "NONE"
    bb_signal: str = "NEUTRAL"
    candle_tier: int = 0
    best_candle: str = ""
    market_state: str = "SIDEWAYS"

candidates = [
    FakeCandidate("TEST_RSI_V1",  primary_signal="RSI"),      # should be SKIPPED
    FakeCandidate("TEST_RSI_V2",  primary_signal="RSI_V2"),   # the gap — will it be skipped?
    FakeCandidate("TEST_MACD",    primary_signal="MACD"),     # control — should pass through
]

from gemini_filter import _keyword_fallback

result = _keyword_fallback(
    candidates=candidates,
    open_positions=[],
    slots_remaining=3,
    max_candidates=10,
    max_flags=3,
    max_positions=3,
)

flagged_symbols = {f["symbol"] for f in result["flags"]}
skipped_symbols = {s["symbol"]: s["reason"] for s in result["skipped"]}

print("FLAGGED (passed through):", flagged_symbols)
print("SKIPPED:", skipped_symbols)
print()
print("TEST_RSI_V1 skipped as expected:", "TEST_RSI_V1" in skipped_symbols)
print("TEST_RSI_V2 skipped (should also skip if gap is fixed):", "TEST_RSI_V2" in skipped_symbols)
print("TEST_MACD flagged as expected:", "TEST_MACD" in flagged_symbols)

if "TEST_RSI_V2" in flagged_symbols:
    print("\n⚠️  GAP CONFIRMED: RSI_V2 slipped through a filter meant to block standalone RSI signals.")
else:
    print("\n✅ No gap — RSI_V2 is being caught (was this already fixed, or was my read of the code wrong?)")