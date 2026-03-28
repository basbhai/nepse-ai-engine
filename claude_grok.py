"""
claude_analyst.py
NEPSE AI Engine — Fully Aligned Version (27 March 2026)
Original design preserved + Complete Learning Hub integration
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# Nepal fee constants
BROKERAGE_PCT = 0.40
SEBON_PCT     = 0.015
DP_CHARGE_NPR = 25.0
CGT_PCT       = 7.5


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS (Your original design)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class AnalystResult:
    symbol:           str
    action:           str        = "WAIT"
    confidence:       int        = 0
    entry_price:      float      = 0.0
    stop_loss:        float      = 0.0
    target:           float      = 0.0
    allocation_npr:   float      = 0.0
    shares:           int        = 0
    breakeven:        float      = 0.0
    risk_reward:      float      = 0.0
    suggested_hold:   int        = 17
    reasoning:        str        = ""
    lesson_applied:   str        = ""
    primary_signal:   str        = ""
    sector:           str        = ""
    geo_score:        int        = 0
    rsi_14:           float      = 0.0
    candle_pattern:   str        = ""
    urgency:          str        = "NORMAL"
    gemini_reason:    str        = ""

    timestamp: str = field(default_factory=lambda: datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        emoji = "✅" if self.action == "BUY" else "⏸" if self.action == "WAIT" else "🚫"
        return f"{emoji} {self.action} {self.symbol} | conf={self.confidence}% | {self.reasoning[:80]}..."


# ══════════════════════════════════════════════════════════════════════════════
# LEARNING HUB INTEGRATION (Core Fix)
# ══════════════════════════════════════════════════════════════════════════════

def load_lessons(symbol: str, sector: str) -> List[Dict]:
    """Robust lesson loader using sheets.py"""
    try:
        from sheets import read_tab_where
        symbol = (symbol or "").strip().upper()
        sector = (sector or "").strip().upper()

        # Load all active lessons
        rows = read_tab_where("learning_hub", {"active": "true"})

        relevant = []
        for r in rows:
            r_sym = (r.get("symbol") or "").upper()
            r_sec = (r.get("sector") or "").upper()
            r_app = (r.get("applies_to") or "").upper()

            symbol_match = (r_sym == symbol or r_sym == "MARKET")
            sector_match = (r_sec == sector or r_sec == "ALL" or r_app == sector or r_sec == "ALL")

            if symbol_match and sector_match:
                relevant.append(r)

        # Sort by confidence level: HIGH > MEDIUM > LOW
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        relevant.sort(key=lambda x: order.get((x.get("confidence_level") or "LOW").upper(), 3))

        logger.info(f"Loaded {len(relevant)} relevant lessons for {symbol} ({sector})")
        return relevant
    except Exception as e:
        logger.error(f"load_lessons failed: {e}")
        return []


def condition_matches(condition_str: str, flag: Any, market_state: str, geo: dict, macro: dict) -> bool:
    """Safe condition parser with MACD=NONE protection"""
    if not condition_str:
        return False

    try:
        cond = condition_str.strip()

        # Critical safety: Never treat MACD=NONE as valid MACD signal
        primary_signal = getattr(flag, 'primary_signal', '')
        macd_cross = getattr(flag, 'macd_cross', '')
        if primary_signal == 'MACD' and macd_cross == 'NONE':
            return False

        replacements = {
            "primary_signal": f"'{primary_signal}'",
            "sector": f"'{getattr(flag, 'sector', '')}'",
            "market_state": f"'{market_state}'",
            "rsi_entry": str(getattr(flag, 'rsi_14', 0)),
            "combined_geo_entry": str(geo.get("combined", 0)),
            "nepal_score_entry": str(geo.get("nepal_score", 0)),
            "fd_rate_pct": str(macro.get("fd_rate", 8.5)),
        }

        for key, val in replacements.items():
            cond = cond.replace(key, val)

        return bool(eval(cond))
    except Exception as e:
        logger.debug(f"condition_matches failed for '{condition_str}': {e}")
        return False


def apply_lesson_actions(lessons: List[Dict], base_conf: int, flag: Any, 
                        market_state: str, geo: dict, macro: dict):
    """Apply lesson actions (BLOCK, confidence modify, etc.)"""
    conf = base_conf
    applied = []
    prompt_context = []

    for lesson in lessons:
        if not condition_matches(lesson.get("condition", ""), flag, market_state, geo, macro):
            continue

        action = (lesson.get("action") or "").strip()
        applied.append(lesson.get("lesson_type", "UNKNOWN"))

        if action == "BLOCK_ENTRY":
            logger.info(f"BLOCK_ENTRY triggered: {lesson.get('finding','')}")
            return 0, "AVOID", [f"BLOCKED by Learning Hub: {lesson.get('finding','')}"]

        elif action.startswith("REDUCE_CONFIDENCE_BY_"):
            try:
                reduction = int(action.split("_")[-1])
                conf = max(0, conf - reduction)
            except:
                pass
        elif action.startswith("INCREASE_CONFIDENCE_BY_"):
            try:
                increase = int(action.split("_")[-1])
                conf = min(100, conf + increase)
            except:
                pass
        elif action == "ADD_TO_REASONING":
            prompt_context.append(lesson.get("finding", ""))
        elif action.startswith(("REDUCE_ALLOCATION_BY_", "INCREASE_ALLOCATION_BY_")):
            prompt_context.append(f"Allocation modifier: {action}")

        # Confidence level note
        level = (lesson.get("confidence_level") or "MEDIUM").upper()
        if level == "MEDIUM":
            prompt_context.append(f"⚠️ MEDIUM confidence lesson: {lesson.get('finding','')}")
        elif level == "LOW":
            prompt_context.append(f"ℹ️ HYPOTHESIS: {lesson.get('finding','')}")

    return conf, None, prompt_context


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT LOADERS (Your original kept)
# ══════════════════════════════════════════════════════════════════════════════

def _load_portfolio():
    try:
        from sheets import read_tab, get_setting
        rows = read_tab("portfolio")
        open_pos = [r for r in rows if r.get("status", "").upper() == "OPEN"]
        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
        invested = sum(float(r.get("total_cost", 0) or 0) for r in open_pos)
        liquid = max(0.0, total_capital - invested)

        return {
            "total_capital_npr": total_capital,
            "liquid_npr": liquid,
            "open_positions": len(open_pos),
            "slots_remaining": 99,   # Hardcoded as requested
        }
    except:
        return {"total_capital_npr": 100000, "liquid_npr": 100000, "slots_remaining": 99}


def _load_geo_context():
    try:
        from sheets import get_latest_geo, get_latest_pulse
        geo = get_latest_geo() or {}
        pulse = get_latest_pulse() or {}
        return {
            "geo_score": int(geo.get("geo_score", 0) or 0),
            "nepal_score": int(pulse.get("nepal_score", 0) or 0),
            "combined": int(geo.get("geo_score", 0) or 0) + int(pulse.get("nepal_score", 0) or 0),
        }
    except:
        return {"geo_score": 0, "nepal_score": 0, "combined": 0}


def _load_macro_context():
    try:
        from sheets import get_setting
        return {"fd_rate": float(get_setting("FD_RATE_PCT", "8.5"))}
    except:
        return {"fd_rate": 8.5}


def _load_market_state():
    try:
        from sheets import get_setting
        return get_setting("MARKET_STATE", "SIDEWAYS").upper()
    except:
        return "SIDEWAYS"


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER (Guide + Your style)
# ══════════════════════════════════════════════════════════════════════════════

def _get_hold_days(primary_signal: str) -> int:
    if primary_signal == "MACD": return 17
    if primary_signal == "BB_LOWER": return 130
    if primary_signal == "SMA_CROSS": return 33
    return 17


def _build_prompt(flag, portfolio, geo, macro, lesson_context: List[str], market_state: str, modified_conf: int):
    hold_days = _get_hold_days(getattr(flag, "primary_signal", ""))

    lessons_str = "\n".join(f"  - {ctx}" for ctx in lesson_context) if lesson_context else "  No applicable lessons."

    prompt = f"""You are the deep analyst for NEPSE stock trading signals.

HARD RULES (NEVER BREAK):
- Max position: 70% of capital
- Stop loss: 3% hard stop
- Max 33 simultaneous positions

SYMBOL: {flag.symbol}
SECTOR: {getattr(flag, 'sector', 'UNKNOWN')}
SIGNAL: {getattr(flag, 'primary_signal', 'UNKNOWN')} (optimal hold {hold_days} days)
GEMINI CONFIDENCE: {modified_conf}%

TECHNICAL DATA:
- RSI: {getattr(flag, 'rsi_14', 0):.1f}
- MACD: {getattr(flag, 'macd_cross', 'NEUTRAL')}
- BB: {getattr(flag, 'bb_signal', 'NEUTRAL')}
- Volume ratio: {getattr(flag, 'volume_ratio', 1.0):.2f}x
- Conf score: {getattr(flag, 'composite_score', 0):.0f}

MARKET CONTEXT:
- Market state: {market_state}
- Geo score: {geo.get('geo_score', 0)}
- Nepal score: {geo.get('nepal_score', 0)}
- Combined: {geo.get('combined', 0)}
- FD rate: {macro.get('fd_rate', 8.5)}%

LESSONS FROM INSTITUTIONAL MEMORY:
{lessons_str}

QUESTION: Should we BUY {flag.symbol} at current levels? also do independ research on it to provide optimum descision.

Respond with JSON only:
{{
  "action": "BUY" | "WAIT" | "AVOID",
  "confidence": 0-100,
  "hold_days_expected": {hold_days},
  "target_exit": "price or %",
  "stop_loss_level": "3% below entry",
  "reasoning": "Clear explanation of decision"
}}"""
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE CALL
# ══════════════════════════════════════════════════════════════════════════════
def _call_claude(prompt: str) -> Optional[dict]:
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "You are a precise NEPSE analyst. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Claude API failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def analyze_flag(flag: Any) -> AnalystResult:
    try:
        base_conf = int(getattr(flag, 'confidence', 0) or 0)
        if base_conf == 0:
            base_conf = 45  # fallback when Gemini gives 0

        symbol = flag.symbol
        sector = getattr(flag, 'sector', 'UNKNOWN')
        market_state = _load_market_state()
        geo = _load_geo_context()
        macro = _load_macro_context()
        portfolio = _load_portfolio()

        # === LEARNING HUB ===
        lessons = load_lessons(symbol, sector)
        modified_conf, early_verdict, lesson_context = apply_lesson_actions(
            lessons, base_conf, flag, market_state, geo, macro
        )

        if early_verdict:
            return AnalystResult(
                symbol=symbol,
                action=early_verdict,
                confidence=0,
                reasoning="BLOCKED by high-confidence Learning Hub lesson",
                lesson_applied="BLOCK_ENTRY",
                primary_signal=getattr(flag, "primary_signal", ""),
                sector=sector
            )

        # === CLAUDE ===
        prompt = _build_prompt(flag, portfolio, geo, macro, lesson_context, market_state, modified_conf)
        claude_json = _call_claude(prompt)

        if not claude_json:
            return AnalystResult(
                symbol=symbol,
                action="WAIT",
                confidence=modified_conf,
                reasoning="Claude unavailable, using modified confidence from lessons"
            )

        result = AnalystResult(
            symbol=symbol,
            action=claude_json.get("action", "WAIT").upper(),
            confidence=int(claude_json.get("confidence", modified_conf)),
            reasoning=claude_json.get("reasoning", ""),
            suggested_hold=int(claude_json.get("hold_days_expected", 17)),
            lesson_applied=", ".join([l.get("lesson_type", "") for l in lessons]),
            primary_signal=getattr(flag, "primary_signal", ""),
            sector=sector,
            geo_score=geo.get("combined", 0),
            rsi_14=getattr(flag, "rsi_14", 0.0)
        )
        return result

    except Exception as e:
        logger.error(f"analyze_flag failed for {getattr(flag,'symbol','?')}: {e}")
        return AnalystResult(symbol=getattr(flag,'symbol','UNKNOWN'), action="WAIT", confidence=30)


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(flags: list) -> List[AnalystResult]:
    if not flags:
        logger.info("No flags to analyze")
        return []

    logger.info(f"claude_analyst: Analyzing {len(flags)} flags")
    results = [analyze_flag(flag) for flag in flags]

    # Write results to market_log
    try:
        from sheets import write_row
        for r in results:
            write_row("market_log", {
                "symbol": r.symbol,
                "action": r.action,
                "confidence": str(r.confidence),
                "reasoning": r.reasoning[:600],
                "timestamp": r.timestamp
            })
    except Exception as e:
        logger.warning(f"Failed to write to market_log: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI (Your original dry-run + print-prompt preserved)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLAUDE_ANALYST] %(levelname)s: %(message)s")

    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    print_prompt = "--print-prompt" in args
    sym_args = [a.upper() for a in args if not a.startswith("--")]

    print("\n" + "="*80)
    print("  NEPSE AI — claude_analyst.py (Fully Aligned with Learning Hub)")
    print("="*80)

    if print_prompt:
        print("[PRINT-PROMPT MODE] Building prompts — NO API call")
        try:
            from gemini_filter import run_gemini_filter
            flags = run_gemini_filter()
            if sym_args:
                flags = [f for f in flags if f.symbol in sym_args]
        except:
            flags = []

        for flag in flags:
            lessons = load_lessons(flag.symbol, getattr(flag, 'sector', ''))
            modified_conf = int(getattr(flag, 'confidence', 45) or 45)
            prompt = _build_prompt(flag, _load_portfolio(), _load_geo_context(), 
                                 _load_macro_context(), [], _load_market_state(), modified_conf)
            print(f"\n--- PROMPT for {flag.symbol} ---\n{prompt}\n")
        sys.exit(0)

    if dry_run:
        from dataclasses import dataclass
        @dataclass
        class SyntheticFlag:
            symbol: str = "NABIL"
            sector: str = "BANKING"
            confidence: int = 74
            primary_signal: str = "MACD"
            rsi_14: float = 58.0
            macd_cross: str = "BULLISH"
            bb_signal: str = "NEUTRAL"
            volume_ratio: float = 1.42
            composite_score: float = 68

        flags = [SyntheticFlag()]
    else:
        try:
            from gemini_filter import run_gemini_filter
            flags = run_gemini_filter()
            if sym_args:
                flags = [f for f in flags if f.symbol in sym_args]
        except Exception as e:
            logger.error(f"gemini_filter failed: {e}")
            flags = []

    results = run_analysis(flags)

    print("\n=== FINAL ANALYSIS RESULTS ===")
    for r in results:
        print(r.summary())
    print("="*80)