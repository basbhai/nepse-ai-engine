"""
budget.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 3, Module 4
Purpose : Position sizing, NEPSE fee calculations, and Kelly Criterion
          via DeepSeek R1 (best for math reasoning).

Responsibilities:
  1. Calculate exact Nepal transaction costs (buy + sell)
  2. Calculate breakeven price (true, after all fees)
  3. Calculate true profit after all Nepal fees + CGT
  4. Kelly Criterion sizing via DeepSeek R1 math reasoning
  5. Apply confidence-based allocation modifier
  6. Enforce hard rules (max 10% per position, max 3 positions)

Called by: claude_analyst.py before generating BUY recommendation
           briefing.py for portfolio P&L display

Evidence base:
  Nepal NEPSE fee structure (official):
    Brokerage:   0.40% on trade value (both buy and sell)
    SEBON levy:  0.015% on trade value (both buy and sell)
    DP charge:   NPR 25 flat per transaction
    CGT:         5% on individuals, 7.5% on institutions
                 (applied to net profit only)
  Kelly Criterion:
    f = Win_Rate - ((1 - Win_Rate) / Avg_Win_Loss_Ratio)
    Position = f × total_capital × confidence_modifier
    (DeepSeek R1 handles the Kelly math — best for precise reasoning)

─────────────────────────────────────────────────────────────────────────────
CLI:
  python budget.py                       → show current capital status
  python budget.py NABIL 1200 1300 1140  → calc for symbol entry target stop
  python budget.py kelly                 → show kelly output from learning hub
  python budget.py --prompt              → show DeepSeek prompt for manual testing
  python budget.py fees 1200 1350 100    → fee breakdown
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from AI import ask_deepseek
from config import NST

logger = logging.getLogger(__name__)

# ── Fee constants (official NEPSE rates) ──────────────────────────────────────
BROKERAGE_PCT   = 0.40    # % both buy and sell
SEBON_PCT       = 0.015   # % both buy and sell
DP_CHARGE_NPR   = 25.0    # NPR flat per transaction (buy = 25, sell = 25)
CGT_INDIVIDUAL  = 5.0     # % capital gains tax — individuals
CGT_INSTITUTION = 7.5     # % — institutions (use individual for personal account)
CGT_PCT         = CGT_INDIVIDUAL

# ── Risk limits (from handoff hard rules) ─────────────────────────────────────
MAX_POSITION_PCT  = 10    # max % of total capital per trade
MAX_POSITIONS     = 3     # max simultaneous open positions
MIN_CONFIDENCE    = 70    # min Claude confidence % to size a position
DEFAULT_STOP_PCT  = 3.0   # default stop loss % below entry

# ── DeepSeek model ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DEEPSEEK_MODEL     = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1")


# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING RESULT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    """
    Complete position sizing recommendation for one trade.
    All NPR values are exact after full Nepal fee calculation.
    """
    symbol:              str
    entry_price:         float
    stop_loss:           float
    target:              float

    # Shares and allocation
    shares:              int   = 0
    allocation_npr:      float = 0.0    # amount to deploy (shares × entry_price)
    allocation_pct:      float = 0.0    # % of total capital

    # Fee breakdown (buy side)
    buy_brokerage:       float = 0.0
    buy_sebon:           float = 0.0
    buy_dp:              float = DP_CHARGE_NPR
    total_buy_cost:      float = 0.0    # all-in cost to buy

    # Fee breakdown (sell side at target)
    sell_brokerage:      float = 0.0
    sell_sebon:          float = 0.0
    sell_dp:             float = DP_CHARGE_NPR
    total_sell_cost:     float = 0.0

    # Profit metrics
    breakeven_price:     float = 0.0    # must exceed this to profit
    gross_profit:        float = 0.0    # target - entry (per share × shares)
    net_profit_npr:      float = 0.0    # after all fees + CGT
    net_profit_pct:      float = 0.0    # net profit as % of allocation

    # Risk metrics
    risk_per_share:      float = 0.0    # entry - stop_loss
    risk_total_npr:      float = 0.0    # total loss if stop hit
    reward_risk_ratio:   float = 0.0    # reward / risk

    # Kelly output
    kelly_fraction:      float = 0.0    # raw Kelly output
    kelly_confidence:    str   = "LOW"  # LOW / MEDIUM / HIGH

    # Notes
    notes:               str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.shares} shares @ NPR {self.entry_price:.0f} | "
            f"Alloc NPR {self.allocation_npr:,.0f} ({self.allocation_pct:.1f}%) | "
            f"BE={self.breakeven_price:.0f} | Net profit NPR {self.net_profit_npr:,.0f} | "
            f"R/R={self.reward_risk_ratio:.1f}x | Kelly={self.kelly_fraction:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CAPITAL STATUS
# ══════════════════════════════════════════════════════════════════════════════

def get_capital_status() -> dict:
    """
    Read current capital state from Neon.
    Returns dict with total, invested, liquid, slots.
    """
    try:
        from sheets import get_setting, read_tab

        total_capital = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))

        rows = read_tab("portfolio")
        open_rows = [r for r in rows if r.get("status", "").upper() == "OPEN"]
        invested  = sum(float(r.get("total_cost", 0) or 0) for r in open_rows)
        liquid    = max(0.0, total_capital - invested)
        slots     = max(0, MAX_POSITIONS - len(open_rows))

        return {
            "total_capital_npr":  total_capital,
            "invested_npr":       invested,
            "liquid_npr":         liquid,
            "open_positions":     len(open_rows),
            "slots_remaining":    slots,
            "max_per_trade_npr":  total_capital * MAX_POSITION_PCT / 100,
            "deployable_npr":     min(liquid, total_capital * MAX_POSITION_PCT / 100),
        }
    except Exception as exc:
        logger.warning("get_capital_status failed: %s", exc)
        return {
            "total_capital_npr":  100000,
            "liquid_npr":         100000,
            "invested_npr":       0,
            "open_positions":     0,
            "slots_remaining":    MAX_POSITIONS,
            "max_per_trade_npr":  10000,
            "deployable_npr":     10000,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEE CALCULATIONS
# Pure math — no IO, no exceptions possible.
# ══════════════════════════════════════════════════════════════════════════════

def calc_buy_fees(trade_value: float) -> dict:
    """
    Calculate all-in buy-side fees for a given trade value (shares × entry_price).

    Returns:
        brokerage, sebon, dp, total_cost
    """
    brokerage  = trade_value * BROKERAGE_PCT / 100
    sebon      = trade_value * SEBON_PCT / 100
    dp         = DP_CHARGE_NPR
    total      = brokerage + sebon + dp
    return {
        "brokerage":  round(brokerage, 2),
        "sebon":      round(sebon, 2),
        "dp":         dp,
        "total_cost": round(total, 2),
    }


def calc_sell_fees(trade_value: float) -> dict:
    """Sell-side fees (same structure as buy)."""
    brokerage  = trade_value * BROKERAGE_PCT / 100
    sebon      = trade_value * SEBON_PCT / 100
    dp         = DP_CHARGE_NPR
    total      = brokerage + sebon + dp
    return {
        "brokerage":  round(brokerage, 2),
        "sebon":      round(sebon, 2),
        "dp":         dp,
        "total_cost": round(total, 2),
    }


def calc_breakeven(entry_price: float, shares: int) -> float:
    """
    True breakeven price — must exceed this to avoid net loss.

    Formula derivation:
      buy_cost  = entry × shares × (brokerage + sebon) / 100 + DP
      sell_cost = exit × shares × (brokerage + sebon) / 100 + DP
      At breakeven: exit × shares - buy_cost - sell_cost = 0
      Solving for exit:
        exit = (entry × shares × (1 + fee_rate) + 2 × DP) / (shares × (1 - fee_rate))
      Approximation (fee_rate << 1):
        exit ≈ entry × (1 + 2 × fee_rate) + 2 × DP / shares
    """
    if entry_price <= 0 or shares <= 0:
        return entry_price
    fee_rate  = (BROKERAGE_PCT + SEBON_PCT) / 100
    breakeven = entry_price * (1 + 2 * fee_rate) + 2 * DP_CHARGE_NPR / shares
    return round(breakeven, 2)


def calc_true_profit(
    entry_price: float,
    exit_price:  float,
    shares:      int,
) -> dict:
    """
    Calculate true net profit/loss after ALL Nepal fees and CGT.

    Args:
        entry_price: Buy price per share
        exit_price:  Sell price per share (use target for expected profit)
        shares:      Number of shares

    Returns:
        dict with gross, fees, cgt, net profit, net_pct
    """
    buy_val   = entry_price * shares
    sell_val  = exit_price  * shares
    gross     = sell_val - buy_val

    buy_fees  = calc_buy_fees(buy_val)
    sell_fees = calc_sell_fees(sell_val)
    total_fees = buy_fees["total_cost"] + sell_fees["total_cost"]

    # CGT only on positive profit
    taxable = max(0, gross - total_fees)
    cgt     = round(taxable * CGT_PCT / 100, 2)

    net_profit  = round(gross - total_fees - cgt, 2)
    net_pct     = round(net_profit / buy_val * 100, 2) if buy_val > 0 else 0.0

    return {
        "buy_value":   round(buy_val, 2),
        "sell_value":  round(sell_val, 2),
        "gross_profit":round(gross, 2),
        "buy_fees":    buy_fees["total_cost"],
        "sell_fees":   sell_fees["total_cost"],
        "total_fees":  round(total_fees, 2),
        "cgt":         cgt,
        "net_profit":  net_profit,
        "net_pct":     net_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — KELLY CRITERION VIA DEEPSEEK R1
# DeepSeek R1 is best for math reasoning — used here instead of Claude.
# ══════════════════════════════════════════════════════════════════════════════

def _load_win_stats() -> dict:
    """
    Load win rate and average win/loss ratio from Learning Hub.
    Returns stats needed for Kelly Criterion calculation.
    """
    try:
        from sheets import run_raw_sql

        rows = run_raw_sql("""
            SELECT
                COUNT(*) FILTER (WHERE outcome = 'WIN')  AS wins,
                COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses,
                AVG(CASE WHEN outcome = 'WIN'  AND pnl_npr ~ '^-?[0-9.]+$'
                         THEN pnl_npr::float END) AS avg_win_npr,
                AVG(CASE WHEN outcome = 'LOSS' AND pnl_npr ~ '^-?[0-9.]+$'
                         THEN ABS(pnl_npr::float) END) AS avg_loss_npr
            FROM learning_hub
            WHERE outcome IN ('WIN', 'LOSS')
              AND pnl_npr IS NOT NULL
              AND pnl_npr != ''
        """)

        if not rows or not rows[0]:
            return _default_win_stats()

        r         = rows[0]
        wins      = int(r.get("wins", 0)  or 0)
        losses    = int(r.get("losses", 0) or 0)
        total     = wins + losses

        if total < 10:
            logger.info("Insufficient trade history for Kelly (%d trades) — using conservative defaults", total)
            return _default_win_stats()

        win_rate   = wins / total
        avg_win    = float(r.get("avg_win_npr",  0) or 0)
        avg_loss   = float(r.get("avg_loss_npr", 0) or 0)

        if avg_loss <= 0:
            return _default_win_stats()

        return {
            "win_rate":   round(win_rate, 4),
            "avg_win":    round(avg_win, 2),
            "avg_loss":   round(avg_loss, 2),
            "wins":       wins,
            "losses":     losses,
            "total":      total,
        }

    except Exception as exc:
        logger.warning("_load_win_stats failed: %s", exc)
        return _default_win_stats()


def _default_win_stats() -> dict:
    """
    Conservative defaults when insufficient history.
    55% win rate, 1:1.5 win/loss ratio → slightly positive Kelly.
    """
    return {
        "win_rate":  0.55,
        "avg_win":   3000.0,
        "avg_loss":  2000.0,
        "wins":      0,
        "losses":    0,
        "total":     0,
        "note":      "default_no_history",
    }

def _kelly_via_deepseek(
    win_rate:      float,
    avg_win:       float,
    avg_loss:      float,
    total_capital: float,
    max_pct:       float = MAX_POSITION_PCT,
) -> dict:
    """
    Calculate Kelly Criterion fraction via DeepSeek R1.
    DeepSeek R1 handles math reasoning better than Claude.
    Returns dict with fraction, recommended_npr, confidence_note.
    Falls back to local calculation if DeepSeek unavailable.
    """
    prompt = f"""Calculate the Kelly Criterion position size for a NEPSE stock trade.

INPUTS:
  Win Rate (p):              {win_rate:.4f}  ({win_rate*100:.1f}%)
  Average Win (b × stake):   NPR {avg_win:.2f}
  Average Loss (a × stake):  NPR {avg_loss:.2f}
  Total Capital:             NPR {total_capital:.2f}
  Maximum Position:          {max_pct}% of capital

FORMULA:
  Kelly fraction = p - ((1 - p) / (avg_win / avg_loss))
  This is the optimal fraction of capital to risk.

NEPAL-SPECIFIC ADJUSTMENTS (apply after Kelly):
  1. Use Half-Kelly (multiply by 0.5) — reduces variance, essential for retail traders
  2. Cap at {max_pct}% maximum regardless of Kelly output
  3. If Kelly fraction is negative → position size = 0 (do not trade)
  4. Round down to nearest share

CALCULATE:
  1. Raw Kelly fraction (show working)
  2. Half-Kelly fraction
  3. Recommended NPR allocation (capped at {max_pct}% = NPR {total_capital * max_pct / 100:.0f})
  4. Kelly confidence level (LOW if Kelly < 0.05, MEDIUM if 0.05-0.15, HIGH if > 0.15)

Return ONLY this JSON with no explanation, no markdown:
{{
  "raw_kelly_fraction": float,
  "half_kelly_fraction": float,
  "recommended_pct": float,
  "recommended_npr": float,
  "confidence": "LOW or MEDIUM or HIGH",
  "calculation_note": "one line showing the key calculation"
}}"""

    result = ask_deepseek(prompt, context="budget_kelly")

    if result is None:
        logger.warning("DeepSeek Kelly failed — local fallback")
        return _kelly_local(win_rate, avg_win, avg_loss, total_capital, max_pct)

    logger.info(
        "DeepSeek Kelly: raw=%.4f half=%.4f pct=%.1f%% NPR=%.0f conf=%s",
        result.get("raw_kelly_fraction", 0),
        result.get("half_kelly_fraction", 0),
        result.get("recommended_pct", 0),
        result.get("recommended_npr", 0),
        result.get("confidence", "?"),
    )
    return result


def _kelly_local(
    win_rate:  float,
    avg_win:   float,
    avg_loss:  float,
    total_capital: float,
    max_pct:   float = MAX_POSITION_PCT,
) -> dict:
    """
    Local Kelly calculation — used as fallback when DeepSeek unavailable.
    Half-Kelly, capped at max_pct.
    """
    if avg_loss <= 0:
        return {"raw_kelly_fraction": 0, "half_kelly_fraction": 0,
                "recommended_pct": 5, "recommended_npr": total_capital * 0.05,
                "confidence": "LOW", "calculation_note": "avg_loss=0 — default 5%"}

    # Standard Kelly: f* = p - (1-p)/b  where b = avg_win/avg_loss
    b = avg_win / avg_loss
    raw_kelly = win_rate - (1 - win_rate) / b
    half_kelly = raw_kelly * 0.5

    # Cap at max
    recommended_pct = min(max(0, half_kelly * 100), max_pct)
    recommended_npr = total_capital * recommended_pct / 100

    confidence = (
        "HIGH"   if raw_kelly > 0.15 else
        "MEDIUM" if raw_kelly > 0.05 else
        "LOW"
    )

    note = f"f*={raw_kelly:.4f} half={half_kelly:.4f} b={b:.2f} p={win_rate:.3f}"

    return {
        "raw_kelly_fraction":  round(raw_kelly, 4),
        "half_kelly_fraction": round(half_kelly, 4),
        "recommended_pct":     round(recommended_pct, 2),
        "recommended_npr":     round(recommended_npr, 2),
        "confidence":          confidence,
        "calculation_note":    note,
    }

# ══════════════════════════════════════════════════════════════════════════════
# get symbol betas
# ══════════════════════════════════════════════════════════════════════════════
def _get_symbol_beta(symbol: str, sector: str = "others") -> tuple[float, str]:
    """
    Get empirical beta for a symbol from fundamental_beta table.
    Falls back to sector-level beta from SIM paper if not found or insignificant.
 
    Args:
        symbol: Stock ticker (e.g. "NABIL")
        sector: Sector name for fallback lookup
 
    Returns:
        (beta_value, source_string)
        source: "empirical" | "sector_fallback" | "default_fallback"
 
    Rules:
        - Use empirical beta only if: market_corr_p < 0.05 AND n_months >= 12
        - If beta > 3.0 or beta < -2.0 → treat as outlier, use sector fallback
        - Outlier symbols (HATHY 2.26, PLI 3.21, GMLI -1.82): sector fallback safer
    """
    from filter_engine import SECTOR_BETAS, DEFAULT_SECTOR_BETA
    try:
        from db import get_db_connection  # adjust to your actual DB import
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT beta, market_corr_p, n_months
                    FROM fundamental_beta
                    WHERE symbol = %s
                    """,
                    (symbol.upper(),)
                )
                row = cur.fetchone()
 
        if row:
            beta_val  = float(row["beta"]          if hasattr(row, "__getitem__") else row[0])
            corr_p    = float(row["market_corr_p"] if hasattr(row, "__getitem__") else row[1])
            n_months  = int(row["n_months"]        if hasattr(row, "__getitem__") else row[2])
 
            # Quality checks before accepting empirical beta
            if corr_p < 0.05 and n_months >= 12 and -2.0 <= beta_val <= 3.0:
                logger.debug(
                    "_get_symbol_beta(%s): empirical beta=%.4f (p=%.4f, n=%d months)",
                    symbol, beta_val, corr_p, n_months,
                )
                return round(beta_val, 4), "empirical"
 
            logger.debug(
                "_get_symbol_beta(%s): empirical beta rejected "
                "(p=%.4f, n=%d, beta=%.4f) — using sector fallback",
                symbol, corr_p, n_months, beta_val,
            )
 
    except Exception as exc:
        logger.debug("_get_symbol_beta(%s) DB lookup failed: %s", symbol, exc)
 
    # Sector fallback
    sector_key = sector.lower().strip()
    sector_beta = SECTOR_BETAS.get(sector_key)
    if sector_beta is None:
        for key, val in SECTOR_BETAS.items():
            if key in sector_key or sector_key in key:
                sector_beta = val
                break
 
    if sector_beta is not None:
        return sector_beta, "sector_fallback"
 
    return DEFAULT_SECTOR_BETA, "default_fallback"
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — POSITION SIZING
# ══════════════════════════════════════════════════════════════════════════════

def size_position(
    symbol:      str,
    entry_price: float,
    stop_loss:   float,
    target:      float,
    confidence:  int   = 75,   # Claude confidence %
    sector:      str   = "",   # for beta lookup
) -> Optional[PositionSize]:
    """
    Calculate optimal position size for a trade.

    Uses Kelly Criterion (via DeepSeek) for base sizing,
    then applies confidence modifier, beta adjustment, and hard caps.

    Args:
        symbol:      Stock ticker
        entry_price: Planned entry price
        stop_loss:   Stop loss price (must be below entry)
        target:      Price target
        confidence:  Claude confidence % (0-100)
        sector:      Sector name for beta fallback lookup

    Returns:
        PositionSize dataclass or None if position should not be taken.
    """
    if entry_price <= 0 or stop_loss <= 0 or target <= 0:
        logger.warning("size_position(%s): invalid prices", symbol)
        return None

    if stop_loss >= entry_price:
        logger.warning("size_position(%s): stop_loss >= entry_price", symbol)
        return None

    if target <= entry_price:
        logger.warning("size_position(%s): target <= entry_price", symbol)
        return None

    if confidence < MIN_CONFIDENCE:
        logger.info("size_position(%s): confidence %d < %d minimum", symbol, confidence, MIN_CONFIDENCE)
        return None

    # ── Capital check ─────────────────────────────────────────────────────────
    capital = get_capital_status()
    if capital["slots_remaining"] == 0:
        logger.info("size_position(%s): portfolio full", symbol)
        return None

    total_capital = capital["total_capital_npr"]
    liquid        = capital["liquid_npr"]

    if liquid <= 0:
        logger.info("size_position(%s): no liquid cash", symbol)
        return None

    # ── Kelly Criterion ────────────────────────────────────────────────────────
    win_stats = _load_win_stats()
    kelly     = _kelly_via_deepseek(
        win_rate      = win_stats["win_rate"],
        avg_win       = win_stats["avg_win"],
        avg_loss      = win_stats["avg_loss"],
        total_capital = total_capital,
        max_pct       = MAX_POSITION_PCT,
    )

    # ── Confidence modifier ────────────────────────────────────────────────────
    # Scale Kelly by (confidence / 100)² to penalise low confidence trades
    conf_modifier = (confidence / 100) ** 2
    base_npr      = kelly["recommended_npr"]
    adjusted_npr  = base_npr * conf_modifier

    # Hard cap: max MAX_POSITION_PCT of total capital, and max liquid
    max_npr    = min(total_capital * MAX_POSITION_PCT / 100, liquid)
    allocation = min(adjusted_npr, max_npr)

    # ── Shares (floor — never exceed allocation) ───────────────────────────────
    # Account for buy fees when calculating shares
    # effective_per_share = entry_price × (1 + (brokerage+sebon)/100) + DP/shares
    # Approximate: shares = allocation / (entry × (1 + fee_rate) + DP_est)
    fee_rate     = (BROKERAGE_PCT + SEBON_PCT) / 100
    dp_per_share = DP_CHARGE_NPR / max(allocation / entry_price, 1)
    effective_pp = entry_price * (1 + fee_rate) + dp_per_share
    shares       = int(allocation / effective_pp)

    if shares <= 0:
        logger.info("size_position(%s): allocation too small for even 1 share at %.0f", symbol, entry_price)
        return None

    # ── Notes (collect all warnings before beta adjustment) ───────────────────
    risk_per_share   = entry_price - stop_loss
    reward_per_share = target - entry_price
    rr = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

    notes = []
    if rr < 2:
        notes.append(f"Low R/R={rr:.1f} — prefer >2")

    # Pre-check profit warning using preliminary shares
    _prelim_profit = calc_true_profit(entry_price, target, shares)
    if _prelim_profit["net_profit"] < 0:
        notes.append("Target does not cover fees — adjust target up")

    if kelly["raw_kelly_fraction"] <= 0:
        notes.append("Negative Kelly — poor expected value, reduce size")

    # ── Beta adjustment ────────────────────────────────────────────────────────
    shares, symbol_beta, beta_source = _apply_beta_to_sizing(
        symbol, sector, shares, entry_price, notes
    )
    if shares <= 0:
        logger.info("size_position(%s): shares=0 after beta adjustment", symbol)
        return None

    # ── Recalculate with exact shares ─────────────────────────────────────────
    actual_allocation = shares * entry_price

    buy_fees  = calc_buy_fees(actual_allocation)
    sell_fees = calc_sell_fees(shares * target)

    breakeven = calc_breakeven(entry_price, shares)
    profit    = calc_true_profit(entry_price, target, shares)
    risk_calc = calc_true_profit(entry_price, stop_loss, shares)

    result = PositionSize(
        symbol            = symbol,
        entry_price       = entry_price,
        stop_loss         = stop_loss,
        target            = target,
        shares            = shares,
        allocation_npr    = round(actual_allocation, 2),
        allocation_pct    = round(actual_allocation / total_capital * 100, 2),

        buy_brokerage     = buy_fees["brokerage"],
        buy_sebon         = buy_fees["sebon"],
        buy_dp            = DP_CHARGE_NPR,
        total_buy_cost    = buy_fees["total_cost"],

        sell_brokerage    = sell_fees["brokerage"],
        sell_sebon        = sell_fees["sebon"],
        sell_dp           = DP_CHARGE_NPR,
        total_sell_cost   = sell_fees["total_cost"],

        breakeven_price   = breakeven,
        gross_profit      = profit["gross_profit"],
        net_profit_npr    = profit["net_profit"],
        net_profit_pct    = profit["net_pct"],

        risk_per_share    = round(risk_per_share, 2),
        risk_total_npr    = abs(round(risk_calc["net_profit"], 2)),
        reward_risk_ratio = rr,

        kelly_fraction    = kelly["raw_kelly_fraction"],
        kelly_confidence  = kelly.get("confidence", "LOW"),

        symbol_beta       = symbol_beta,
        beta_source       = beta_source,

        notes             = " | ".join(notes) if notes else "Clean setup",
    )

    logger.info("size_position(%s): %s", symbol, result.summary())
    return result

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PORTFOLIO FEE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_fee_summary() -> dict:
    """
    Calculate total fees paid and pending on current portfolio.
    Used by briefing.py and auditor.py.
    """
    try:
        from sheets import read_tab

        rows     = read_tab("portfolio")
        open_pos = [r for r in rows if r.get("status", "").upper() == "OPEN"]

        total_cost        = 0.0
        total_value       = 0.0
        total_fees_paid   = 0.0   # buy side fees already paid
        total_fees_pending = 0.0  # sell side fees when we exit

        for r in open_pos:
            cost        = float(r.get("total_cost",    0) or 0)
            value       = float(r.get("current_value", 0) or 0)
            shares      = int(float(r.get("shares", 0) or 0))

            total_cost  += cost
            total_value += value

            # Buy fees already paid
            bf = calc_buy_fees(cost)
            total_fees_paid += bf["total_cost"]

            # Sell fees when we exit at current value
            sf = calc_sell_fees(value)
            total_fees_pending += sf["total_cost"]

        return {
            "total_cost_npr":         round(total_cost, 2),
            "total_value_npr":        round(total_value, 2),
            "unrealised_pnl_npr":     round(total_value - total_cost, 2),
            "fees_paid_buy_npr":      round(total_fees_paid, 2),
            "fees_pending_sell_npr":  round(total_fees_pending, 2),
            "total_fee_drag_npr":     round(total_fees_paid + total_fees_pending, 2),
        }

    except Exception as exc:
        logger.error("portfolio_fee_summary failed: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUICK CALC (no Kelly, just fees)
# Used by claude_analyst.py when budget.py not fully wired yet
# ══════════════════════════════════════════════════════════════════════════════

def quick_size(
    entry_price:   float,
    stop_loss:     float,
    target:        float,
    total_capital: float,
    liquid_npr:    float,
    confidence:    int = 75,
) -> dict:
    """
    Fast position sizing without DeepSeek call.
    Uses fixed 5-10% allocation based on confidence.
    For use when Kelly not needed or DeepSeek unavailable.
    """
    # Confidence-based base allocation
    if confidence >= 85:
        base_pct = MAX_POSITION_PCT        # 10%
    elif confidence >= 78:
        base_pct = MAX_POSITION_PCT * 0.75 # 7.5%
    else:
        base_pct = MAX_POSITION_PCT * 0.5  # 5%

    max_npr    = min(total_capital * base_pct / 100, liquid_npr)
    shares     = int(max_npr / entry_price) if entry_price > 0 else 0
    allocation = shares * entry_price if shares > 0 else 0.0
    breakeven  = calc_breakeven(entry_price, max(shares, 1))

    if shares > 0 and target > entry_price:
        profit_info = calc_true_profit(entry_price, target, shares)
        net_profit  = profit_info["net_profit"]
    else:
        net_profit = 0.0

    rr = (target - entry_price) / (entry_price - stop_loss) if (entry_price - stop_loss) > 0 else 0

    return {
        "shares":          shares,
        "allocation_npr":  round(allocation, 2),
        "allocation_pct":  round(allocation / total_capital * 100, 2) if total_capital > 0 else 0,
        "breakeven_price": breakeven,
        "net_profit_npr":  round(net_profit, 2),
        "reward_risk":     round(rr, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m helper.budget                           → capital status
#   python -m helper.budget NABIL 1200 1300 1140      → size position
#   python -m helper.budget kelly                     → show kelly output
#   python -m helper.budget --prompt                  → show DeepSeek prompt
#   python -m helper.budget fees 1200 1350 100        → fee breakdown
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [BUDGET] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    args = sys.argv[1:]

    # ── Help / Prompt ─────────────────────────────────────────────────────────
    if "--prompt" in args:
        stats = _load_win_stats()
        cap = get_capital_status()
        total_capital = cap["total_capital_npr"]
        max_pct = MAX_POSITION_PCT
        prompt = f"""Calculate the Kelly Criterion position size for a NEPSE stock trade.

INPUTS:
  Win Rate (p):              {stats['win_rate']:.4f}  ({stats['win_rate']*100:.1f}%)
  Average Win (b × stake):   NPR {stats['avg_win']:.2f}
  Average Loss (a × stake):  NPR {stats['avg_loss']:.2f}
  Total Capital:             NPR {total_capital:.2f}
  Maximum Position:          {max_pct}% of capital

FORMULA:
  Kelly fraction = p - ((1 - p) / (avg_win / avg_loss))
  This is the optimal fraction of capital to risk.

NEPAL-SPECIFIC ADJUSTMENTS (apply after Kelly):
  1. Use Half-Kelly (multiply by 0.5) — reduces variance, essential for retail traders
  2. Cap at {max_pct}% maximum regardless of Kelly output
  3. If Kelly fraction is negative → position size = 0 (do not trade)
  4. Round down to nearest share

CALCULATE:
  1. Raw Kelly fraction (show working)
  2. Half-Kelly fraction
  3. Recommended NPR allocation (capped at {max_pct}% = NPR {total_capital * max_pct / 100:.0f})
  4. Kelly confidence level (LOW if Kelly < 0.05, MEDIUM if 0.05-0.15, HIGH if > 0.15)

Return ONLY this JSON with no explanation, no markdown:
{{
  "raw_kelly_fraction": float,
  "half_kelly_fraction": float,
  "recommended_pct": float,
  "recommended_npr": float,
  "confidence": "LOW or MEDIUM or HIGH",
  "calculation_note": "one line showing the key calculation"
}}"""
        print("\n" + "="*70)
        print("  DEEPSEEK KELLY PROMPT (copy this to test manually)")
        print("="*70)
        print(prompt)
        print("\n" + "="*70 + "\n")
        sys.exit(0)

    # ── Capital status ────────────────────────────────────────────────────────
    if not args:
        cap = get_capital_status()
        print(f"\n{'='*55}")
        print(f"  CAPITAL STATUS")
        print(f"{'='*55}")
        print(f"  Total capital:    NPR {cap['total_capital_npr']:>12,.0f}")
        print(f"  Invested:         NPR {cap['invested_npr']:>12,.0f}")
        print(f"  Liquid:           NPR {cap['liquid_npr']:>12,.0f}")
        print(f"  Open positions:   {cap['open_positions']}/{MAX_POSITIONS}")
        print(f"  Slots remaining:  {cap['slots_remaining']}")
        print(f"  Max per trade:    NPR {cap['max_per_trade_npr']:>12,.0f}  ({MAX_POSITION_PCT}%)")
        print(f"  Deployable now:   NPR {cap['deployable_npr']:>12,.0f}")
        print(f"{'='*55}\n")
        sys.exit(0)

    # ── Kelly stats ───────────────────────────────────────────────────────────
    if args[0].lower() == "kelly":
        stats = _load_win_stats()
        cap   = get_capital_status()
        kelly = _kelly_via_deepseek(
            stats["win_rate"], stats["avg_win"], stats["avg_loss"],
            cap["total_capital_npr"]
        )
        print(f"\n{'='*55}")
        print(f"  KELLY CRITERION")
        print(f"{'='*55}")
        print(f"  History:     {stats.get('total', 0)} trades  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)")
        print(f"  Win rate:    {stats['win_rate']*100:.1f}%")
        print(f"  Avg win:     NPR {stats['avg_win']:,.0f}")
        print(f"  Avg loss:    NPR {stats['avg_loss']:,.0f}")
        print(f"  Raw Kelly:   {kelly['raw_kelly_fraction']:.4f}  ({kelly['raw_kelly_fraction']*100:.1f}%)")
        print(f"  Half-Kelly:  {kelly['half_kelly_fraction']:.4f}  ({kelly['half_kelly_fraction']*100:.1f}%)")
        print(f"  Recommended: {kelly['recommended_pct']:.1f}%  =  NPR {kelly['recommended_npr']:,.0f}")
        print(f"  Confidence:  {kelly['confidence']}")
        if kelly.get("calculation_note"):
            print(f"  Note:        {kelly['calculation_note']}")
        print(f"{'='*55}\n")
        sys.exit(0)

    # ── Fee breakdown ─────────────────────────────────────────────────────────
    if args[0].lower() == "fees":
        try:
            entry, target, shares = float(args[1]), float(args[2]), int(args[3])
            buy_val  = entry  * shares
            sell_val = target * shares
            bf = calc_buy_fees(buy_val)
            sf = calc_sell_fees(sell_val)
            be = calc_breakeven(entry, shares)
            p  = calc_true_profit(entry, target, shares)
            print(f"\n  FEE BREAKDOWN: {shares} shares @ NPR {entry:.0f} → {target:.0f}")
            print(f"  {'─'*45}")
            print(f"  Buy trade value:   NPR {buy_val:>10,.0f}")
            print(f"  Buy brokerage:     NPR {bf['brokerage']:>10,.2f}")
            print(f"  Buy SEBON:         NPR {bf['sebon']:>10,.2f}")
            print(f"  Buy DP charge:     NPR {bf['dp']:>10.0f}")
            print(f"  Total buy cost:    NPR {bf['total_cost']:>10,.2f}")
            print(f"  {'─'*45}")
            print(f"  Sell trade value:  NPR {sell_val:>10,.0f}")
            print(f"  Sell brokerage:    NPR {sf['brokerage']:>10,.2f}")
            print(f"  Sell SEBON:        NPR {sf['sebon']:>10,.2f}")
            print(f"  Sell DP charge:    NPR {sf['dp']:>10.0f}")
            print(f"  Total sell cost:   NPR {sf['total_cost']:>10,.2f}")
            print(f"  {'─'*45}")
            print(f"  Gross profit:      NPR {p['gross_profit']:>+10,.0f}")
            print(f"  CGT ({CGT_PCT}%):      NPR {p['cgt']:>+10,.2f}")
            print(f"  Net profit:        NPR {p['net_profit']:>+10,.0f}  ({p['net_pct']:+.2f}%)")
            print(f"  Breakeven price:   NPR {be:>10.2f}\n")
        except (IndexError, ValueError):
            print("  Usage: python budget.py fees <entry> <target> <shares>")
        sys.exit(0)

    # ── Position sizing ───────────────────────────────────────────────────────
    try:
        symbol = args[0].upper()
        entry  = float(args[1])
        target = float(args[2])
        stop   = float(args[3])
        conf   = int(args[4]) if len(args) > 4 else 75

        print(f"\n  Sizing position for {symbol}...")
        pos = size_position(symbol, entry, stop, target, conf)

        if not pos:
            print(f"  ❌ Cannot size position — check capital, confidence, or prices\n")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  POSITION SIZE: {symbol}")
        print(f"{'='*60}")
        print(f"  Shares:        {pos.shares}")
        print(f"  Entry:         NPR {pos.entry_price:,.2f}")
        print(f"  Stop Loss:     NPR {pos.stop_loss:,.2f}")
        print(f"  Target:        NPR {pos.target:,.2f}")
        print(f"  Breakeven:     NPR {pos.breakeven_price:,.2f}")
        print(f"  Allocation:    NPR {pos.allocation_npr:,.0f}  ({pos.allocation_pct:.1f}%)")
        print(f"  Net profit:    NPR {pos.net_profit_npr:,.0f}  ({pos.net_profit_pct:+.2f}%)")
        print(f"  Risk (if stop): NPR {pos.risk_total_npr:,.0f}")
        print(f"  R/R Ratio:     {pos.reward_risk_ratio:.1f}x")
        print(f"  Kelly:         {pos.kelly_fraction:.4f}  [{pos.kelly_confidence}]")
        if pos.notes != "Clean setup":
            print(f"  ⚠️  Notes:       {pos.notes}")
        print(f"{'='*60}\n")

    except (IndexError, ValueError):
        print("\n  Usage: python budget.py SYMBOL entry target stop [confidence]")
        print("  Example: python budget.py NABIL 1200 1350 1164 78\n")
        sys.exit(1)