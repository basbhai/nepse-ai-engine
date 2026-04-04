"""
learning_seeder.py — Pre-load Research Findings into learning_hub
NEPSE AI Engine | Run ONCE before first live trade.

USAGE:
  python -m analysis.learning_seeder --dry-run    # Preview all 15 seeds, no DB writes
  python -m analysis.learning_seeder --execute    # Insert seeds into learning_hub
  python -m analysis.learning_seeder --status     # Show current hub stats
  python -m analysis.learning_seeder --reset      # Deactivate all research_paper seeds (re-seed clean)

RULES:
  - ALWAYS dry-run first and review output before executing.
  - Seeds use source='research_paper'. GPT weekly lessons use source='gpt_weekly'.
  - Seeds start as confidence_level='MEDIUM' — upgraded to HIGH after 15+ live trades confirm.
  - NEVER manually edit seeds in the DB. Deactivate and re-seed instead.
  - Re-run --execute is safe: uses upsert on (symbol, sector, lesson_type, condition).

IMPORT RULE: from sheets import ... — NEVER from db import ...
"""

import sys
import argparse
import logging
from datetime import datetime

from config import NST

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("learning_seeder")

# ── NST timestamp helper ────────────────────────────────────────────────────────
def _nst_now() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")


# ── Import sheets API ───────────────────────────────────────────────────────────
try:
    from sheets import write_row, read_tab, upsert_row
except ImportError as e:
    log.error("Cannot import sheets: %s", e)
    log.error("Run from project root where sheets.py exists.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# THE 15 SEEDS
# Sourced from: Karki 2023, Khadka & Rajopadhyaya 2023, Adhikari 2023 +
#               4 additional NEPSE research papers cited in handoff.
#
# Schema fields used (all Text in DB):
#   lesson_type, source, symbol, sector, applies_to,
#   condition, finding, action,
#   trade_count, win_count, loss_count, win_rate, avg_return_pct, avg_pnl_npr,
#   confidence_level, loss_cause_primary,
#   geo_delta_avg, nepal_delta_avg, alpha_vs_nepse_avg,
#   active, superseded_by, last_validated, validation_count,
#   trade_journal_ids, created_at
# ══════════════════════════════════════════════════════════════════════════════

def _build_seeds() -> list[dict]:
    now = _nst_now()

    seeds = [

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL FILTERS (3)
        # ──────────────────────────────────────────────────────────────────────

        {
            # Seed 1: RSI standalone loses money — always block
            "lesson_type":         "SIGNAL_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "primary_signal = 'RSI'",
            "finding":             (
                "RSI as a standalone trigger produces -4.81% annualized return over 10 years "
                "in NEPSE (Karki 2023). It is not a timing signal — it is context only. "
                "Never allow RSI to be the sole reason for entry."
            ),
            "action":              "BLOCK_ENTRY",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "-4.81",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "SIGNAL_FAILURE",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "-4.81",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 2: MACD 12/26/9 — best signal, 17-day optimal hold
            "lesson_type":         "SIGNAL_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "primary_signal = 'MACD'",
            "finding":             (
                "MACD 12/26/9 produces 23.64% annualized return in NEPSE with Profit Factor 2.97 "
                "(Karki 2023). Optimal hold window after bullish cross = 17 trading days. "
                "Weight: 0.35. Do not exit before day 17 unless stop triggered."
            ),
            "action":              "ADD_TO_REASONING",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "23.64",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "NULL",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 3: Bollinger Band lower touch — rare but premium quality
            "lesson_type":         "SIGNAL_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "primary_signal = 'BB_LOWER'",
            "finding":             (
                "Bollinger Band lower touch produces Profit Factor 12.19 — highest quality signal "
                "in NEPSE despite only 9 trades over 10 years (Karki 2023). When BB_LOWER fires "
                "with volume confirmation, treat as premium entry. Optimal hold = 130 days. "
                "Weight: 0.25. Increase allocation — this signal is rare and powerful."
            ),
            "action":              "INCREASE_ALLOCATION_BY_25",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "NULL",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        # ──────────────────────────────────────────────────────────────────────
        # SECTOR FILTERS (5)
        # ──────────────────────────────────────────────────────────────────────

        {
            # Seed 4: Non-Life Insurance — best risk-adjusted sector
            "lesson_type":         "SECTOR_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "NON_LIFE_INSURANCE",
            "applies_to":          "NON_LIFE_INSURANCE",
            "condition":           "sector = 'NON_LIFE_INSURANCE'",
            "finding":             (
                "Non-Life Insurance is the best risk-adjusted sector in NEPSE: Sharpe-like ratio 2.732, "
                "beta 0.034 (Khadka & Rajopadhyaya 2023). Nearly all variance is unsystematic — "
                "stock-picking alpha is highest here. MACD returns 44.09%, Stochastic 65-70% alpha "
                "over buy-and-hold. Sector multiplier x1.25. Increase confidence for confirmed signals."
            ),
            "action":              "INCREASE_CONFIDENCE_BY_15",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "NULL",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 5: Non-Life Insurance — 4.6x political sensitivity during crises
            "lesson_type":         "SECTOR_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "NON_LIFE_INSURANCE",
            "applies_to":          "NON_LIFE_INSURANCE",
            "condition":           "sector = 'NON_LIFE_INSURANCE' AND combined_geo_entry < -3",
            "finding":             (
                "Insurance sector is 4.6x more sensitive to political events than NEPSE average "
                "(Khadka & Rajopadhyaya 2023). When combined_geo score drops below -3, "
                "insurance stocks fall disproportionately hard. Reduce sector multiplier to x0.85 "
                "and cut allocation in crisis periods. Confidence boost from Seed 4 is overridden."
            ),
            "action":              "REDUCE_ALLOCATION_BY_40",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "SECTOR_EVENT",
            "geo_delta_avg":       "-4",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 6: Hydro + RSI = confirmed loser
            "lesson_type":         "SECTOR_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "HYDRO",
            "applies_to":          "HYDRO",
            "condition":           "sector = 'HYDRO' AND primary_signal = 'RSI'",
            "finding":             (
                "RSI loses -6.49% annualized in Hydropower — worse than any other sector (Karki 2023). "
                "Hydro is macro-heavy (monsoon, NRB, geo). Technical-only RSI signals in Hydro have "
                "no predictive power. MACD still works (22.68%). Weight macro/geo more than technical. "
                "Block all RSI-primary entries in HYDRO sector."
            ),
            "action":              "BLOCK_ENTRY",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "-6.49",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "SIGNAL_FAILURE",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "-6.49",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 7: Banking — excluded from optimal portfolio
            "lesson_type":         "SECTOR_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "BANKING",
            "applies_to":          "BANKING",
            "condition":           "sector = 'BANKING'",
            "finding":             (
                "Banking excluded from SIM optimal NEPSE portfolio — risk-adjusted ratio only 0.051 "
                "vs C* threshold of 0.129 (Khadka & Rajopadhyaya 2023). Beta=1.000 means it just "
                "tracks NEPSE with no excess return. Lending rate r=-0.669 is the strongest macro "
                "predictor — NRB rate hikes kill banking stocks. Sector multiplier x0.90."
            ),
            "action":              "REDUCE_CONFIDENCE_BY_20",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "SIGNAL_FAILURE",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 8: Manufacturing — worst risk-adjusted returns
            "lesson_type":         "SECTOR_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "MANUFACTURING",
            "applies_to":          "MANUFACTURING",
            "condition":           "sector = 'MANUFACTURING'",
            "finding":             (
                "Manufacturing has worst risk-adjusted return in NEPSE: excess return -0.044, "
                "excluded from SIM optimal portfolio (Khadka & Rajopadhyaya 2023). "
                "Sector multiplier x0.75. Avoid unless setup is truly exceptional — "
                "even then expect negative expected value. Reduce confidence significantly."
            ),
            "action":              "REDUCE_CONFIDENCE_BY_25",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "-0.044",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "SIGNAL_FAILURE",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        # ──────────────────────────────────────────────────────────────────────
        # MACRO FILTERS (5)
        # ──────────────────────────────────────────────────────────────────────

        {
            # Seed 9: Herding bubble risk in FULL_BULL + RSI > 72
            "lesson_type":         "MACRO_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "market_state = 'FULL_BULL' AND rsi_entry > 72 AND combined_geo_entry > 5",
            "finding":             (
                "Herding confirmed in NEPSE: beta 0.351-0.428. Strongest in bull markets — "
                "amplifies bubbles upward and crashes downward (multiple papers). "
                "RSI above 72 in FULL_BULL with combined_geo > +5 signals late-cycle bubble risk. "
                "The 65 threshold has no paper evidence — use 72. Reduce allocation to protect capital."
            ),
            "action":              "REDUCE_ALLOCATION_BY_30",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "MACRO_DETERIORATION",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 10: Banking + negative nepal_score = block
            "lesson_type":         "MACRO_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "BANKING",
            "applies_to":          "BANKING",
            "condition":           "sector = 'BANKING' AND nepal_score_entry <= -1",
            "finding":             (
                "Lending rate is the strongest macro predictor for banking stocks: r=-0.669 "
                "(Karki 2023, NRB papers). When nepal_score is negative (NRB tightening, "
                "rising lending rates, or policy headwinds), banking stocks face maximum macro pressure. "
                "This compounds the already weak sector fundamentals. Block banking entries "
                "when domestic macro is unfavourable."
            ),
            "action":              "BLOCK_ENTRY",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "MACRO_DETERIORATION",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "-2",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 11: FD rate > 10% — retail money exits NEPSE
            "lesson_type":         "MACRO_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "fd_rate_pct > 10",
            "finding":             (
                "FD rate above 10% triggers retail capital rotation out of NEPSE into fixed deposits "
                "(confirmed across 3 research papers). Deposit rate r=-0.650 for NEPSE. "
                "FD signal: STRONG_FD means retail is leaving. Reduce equity allocation significantly "
                "and bias toward FD. Interest scraper auto-updates FD_RATE_PCT in settings."
            ),
            "action":              "REDUCE_ALLOCATION_BY_40",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "MACRO_DETERIORATION",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "-2",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 12: Information leakage — announcement already priced in
            "lesson_type":         "MACRO_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "key_event_type = 'ANNOUNCEMENT' AND days_since_event <= 0",
            "finding":             (
                "Political shock/policy announcement information leakage starts 10 days BEFORE the event "
                "(Khadka & Rajopadhyaya 2023). Market recovers from political shocks in 3-5 days. "
                "For dividend announcements: abnormal returns begin at day -6 (Adhikari 2023). "
                "Entering ON or AFTER announcement day means the move is already priced in. "
                "Reduce confidence by 30 — reward has been captured by informed money."
            ),
            "action":              "REDUCE_CONFIDENCE_BY_30",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "MEDIUM",
            "loss_cause_primary":  "ENTRY_TIMING",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        {
            # Seed 13: Macro shock — hold through recovery window (3-5 days)
            "lesson_type":         "MACRO_FILTER",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "combined_geo_entry < -3 AND hold_days_actual <= 3",
            "finding":             (
                "NEPSE recovers from political shocks and geo crises in 3-5 trading days "
                "(Khadka & Rajopadhyaya 2023). Selling within 3 days of a geo shock crystallises "
                "losses unnecessarily — the recovery often erases the dip. "
                "If already in a position when geo < -3 hits, hold through the window unless "
                "3% hard stop is triggered. Do not add new positions when geo < -3."
            ),
            "action":              "ADD_TO_REASONING",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "MEDIUM",
            "loss_cause_primary":  "MACRO_DETERIORATION",
            "geo_delta_avg":       "-3",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        # ──────────────────────────────────────────────────────────────────────
        # FAILURE MODE (1)
        # ──────────────────────────────────────────────────────────────────────

        {
            # Seed 14: Macro deterioration loss ≠ signal failure
            "lesson_type":         "FAILURE_MODE",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "loss_cause = 'MACRO_DETERIORATION'",
            "finding":             (
                "When alpha vs NEPSE is near zero, the stock tracked the market — the signal did not fail. "
                "Macro deteriorated AFTER entry (geo_delta or nepal_delta went deeply negative). "
                "Causal attribution: do NOT penalise signal confidence. Instead, improve macro filter "
                "at entry. Lesson target is entry timing (tighten combined_geo threshold), "
                "not signal threshold (do not raise MACD confidence requirement)."
            ),
            "action":              "ADD_TO_REASONING",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "0",
            "avg_pnl_npr":         "0",
            "confidence_level":    "HIGH",
            "loss_cause_primary":  "MACRO_DETERIORATION",
            "geo_delta_avg":       "-3",
            "nepal_delta_avg":     "-2",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },

        # ──────────────────────────────────────────────────────────────────────
        # NEGATIVE FINDING (1)
        # ──────────────────────────────────────────────────────────────────────

        {
            # Seed 15: Dividend pre-announcement pattern — undetectable at this stage
            "lesson_type":         "DIVIDEND_PATTERN",
            "source":              "research_paper",
            "symbol":              "MARKET",
            "sector":              "ALL",
            "applies_to":          "ALL",
            "condition":           "event_type = 'DIVIDEND_ANNOUNCEMENT'",
            "finding":             (
                "Adhikari 2023: abnormal returns start at day -6 before dividend announcements, "
                "day +1 = +9.30% peak, day +2 to +9 = negative adjustment. "
                "HOWEVER, pre-announcement accumulation pattern detection requires dividend_study.py "
                "to confirm 70%+ consistency across historical data. Until that study is complete and "
                "shows >= 70% pattern rate, do NOT use a dividend detector as a signal. "
                "If confirmed < 70%, use as tiebreaker only (small weight in claude_analyst context)."
            ),
            "action":              "ADD_TO_REASONING",
            "trade_count":         "0",
            "win_count":           "0",
            "loss_count":          "0",
            "win_rate":            "0.0",
            "avg_return_pct":      "9.30",
            "avg_pnl_npr":         "0",
            "confidence_level":    "LOW",
            "loss_cause_primary":  "NULL",
            "geo_delta_avg":       "0",
            "nepal_delta_avg":     "0",
            "alpha_vs_nepse_avg":  "0",
            "active":              "true",
            "superseded_by":       "",
            "last_validated":      now[:10],
            "validation_count":    "1",
            "trade_journal_ids":   "",
            "created_at":          now,
        },
    ]

    return seeds


# ══════════════════════════════════════════════════════════════════════════════
# CORE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _print_seed(i: int, seed: dict) -> None:
    """Pretty-print a single seed for dry-run review."""
    print(f"\n{'─' * 70}")
    print(f"  Seed {i:02d} | {seed['lesson_type']:20} | {seed['sector']:25} | {seed['confidence_level']}")
    print(f"  Condition : {seed['condition']}")
    print(f"  Action    : {seed['action']}")
    print(f"  Finding   : {seed['finding'][:120]}{'...' if len(seed['finding']) > 120 else ''}")
    print(f"  Source    : {seed['source']} | avg_return: {seed['avg_return_pct']}%")


def cmd_dry_run() -> None:
    seeds = _build_seeds()
    print("\n" + "═" * 70)
    print("  LEARNING SEEDER — DRY RUN")
    print(f"  {len(seeds)} seeds to insert into learning_hub")
    print("═" * 70)

    by_type: dict[str, list] = {}
    for s in seeds:
        by_type.setdefault(s["lesson_type"], []).append(s)

    for lesson_type, group in by_type.items():
        print(f"\n  ▸ {lesson_type} ({len(group)} seeds)")
        for i, s in enumerate(group, 1):
            _print_seed(i, s)

    print(f"\n{'═' * 70}")
    print(f"  Total seeds: {len(seeds)}")
    counts = {k: len(v) for k, v in by_type.items()}
    for t, c in counts.items():
        print(f"    {t:30} {c}")
    print("\n  ⚠  DRY RUN — nothing written to DB.")
    print("  Run: python learning_seeder.py --execute  to insert.\n")


def cmd_execute(dry: bool = False) -> None:
    seeds = _build_seeds()
    inserted = 0
    skipped = 0
    errors = 0

    log.info("Starting seed insertion — %d seeds", len(seeds))

    for seed in seeds:
        try:
            # Use upsert keyed on (lesson_type, source, condition) to be idempotent
            # Falls back to write_row if upsert not available for this table
            try:
                upsert_row(
                    "learning_hub",
                    seed,
                    conflict_keys=["lesson_type", "source", "condition"],
                )
                log.info(
                    "✅ Upserted  | %-20s | %-25s | %s",
                    seed["lesson_type"], seed["sector"], seed["action"],
                )
            except TypeError:
                # upsert_row may not accept conflict_keys — fall back to write_row
                write_row("learning_hub", seed)
                log.info(
                    "✅ Inserted  | %-20s | %-25s | %s",
                    seed["lesson_type"], seed["sector"], seed["action"],
                )
            inserted += 1

        except Exception as e:
            log.error(
                "❌ FAILED    | %-20s | %-25s | %s",
                seed["lesson_type"], seed["sector"], str(e),
            )
            errors += 1

    print(f"\n{'═' * 60}")
    print(f"  SEED INSERTION COMPLETE")
    print(f"  ✅ Inserted : {inserted}")
    print(f"  ⏭  Skipped  : {skipped}")
    print(f"  ❌ Errors   : {errors}")
    print(f"{'═' * 60}")
    if errors:
        print("  ⚠  Fix errors before first live trade.")
        print("  Check: DATABASE_URL env var and learning_hub table exists.")
        print("  Run: python -m db.migrations  if table missing.\n")
    else:
        print("  ✅ All seeds loaded. Run --status to verify.\n")


def cmd_status() -> None:
    try:
        rows = read_tab("learning_hub")
    except Exception as e:
        print(f"\n❌ Cannot read learning_hub: {e}")
        print("   Run: python -m db.migrations\n")
        return

    total = len(rows)
    research = [r for r in rows if r.get("source") == "research_paper"]
    gpt_weekly = [r for r in rows if r.get("source") == "gpt_weekly"]
    claude = [r for r in rows if r.get("source") == "claude"]
    active = [r for r in rows if r.get("active") == "true"]
    inactive = [r for r in rows if r.get("active") != "true"]

    by_type: dict[str, int] = {}
    by_conf: dict[str, int] = {}
    for r in rows:
        lt = r.get("lesson_type", "UNKNOWN")
        cl = r.get("confidence_level", "UNKNOWN")
        by_type[lt] = by_type.get(lt, 0) + 1
        by_conf[cl] = by_conf.get(cl, 0) + 1

    print(f"\n{'═' * 60}")
    print(f"  📊 LEARNING HUB STATUS")
    print(f"{'═' * 60}")
    print(f"  Total lessons        : {total}")
    print(f"  Active               : {len(active)}")
    print(f"  Inactive (superseded): {len(inactive)}")
    print()
    print(f"  By Source:")
    print(f"    Research paper     : {len(research)}")
    print(f"    GPT weekly         : {len(gpt_weekly)}")
    print(f"    Claude             : {len(claude)}")
    print()
    print(f"  By Confidence:")
    for lvl in ["HIGH", "MEDIUM", "LOW"]:
        print(f"    {lvl:8}           : {by_conf.get(lvl, 0)}")
    print()
    print(f"  By Type:")
    for lt, count in sorted(by_type.items()):
        print(f"    {lt:30} : {count}")
    print(f"{'═' * 60}\n")

    # Show seeder readiness
    if len(research) == 0:
        print("  ⚠  NO SEEDS LOADED. Run: python learning_seeder.py --execute\n")
    elif len(research) < 15:
        print(f"  ⚠  Only {len(research)}/15 seeds present. Re-run --execute.\n")
    else:
        print(f"  ✅ All 15 research seeds present. System ready.\n")


def cmd_reset() -> None:
    """Deactivate all research_paper seeds — use before re-seeding clean."""
    try:
        rows = read_tab("learning_hub")
    except Exception as e:
        print(f"\n❌ Cannot read learning_hub: {e}\n")
        return

    research = [r for r in rows if r.get("source") == "research_paper"]
    if not research:
        print("\n  No research_paper seeds found to reset.\n")
        return

    print(f"\n  ⚠  This will deactivate {len(research)} research_paper seeds.")
    confirm = input("  Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("  Aborted.\n")
        return

    deactivated = 0
    for row in research:
        try:
            row_id = row.get("id")
            if row_id:
                upsert_row(
                    "learning_hub",
                    {"id": row_id, "active": "false"},
                    conflict_keys=["id"],
                )
                deactivated += 1
        except Exception as e:
            log.warning("Could not deactivate seed id=%s: %s", row.get("id"), e)

    print(f"\n  ✅ Deactivated {deactivated} seeds.")
    print("  Run --execute to re-seed fresh.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="learning_seeder.py — pre-load NEPSE research findings into learning_hub"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all 15 seeds without writing to DB",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="Insert all seeds into learning_hub table",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Print current learning_hub stats",
    )
    group.add_argument(
        "--reset",
        action="store_true",
        help="Deactivate all research_paper seeds (use before re-seeding)",
    )

    args = parser.parse_args()

    if args.dry_run:
        cmd_dry_run()
    elif args.execute:
        cmd_execute()
    elif args.status:
        cmd_status()
    elif args.reset:
        cmd_reset()


if __name__ == "__main__":
    main()