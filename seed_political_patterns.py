"""
seed_political_patterns.py — NEPSE AI Engine
─────────────────────────────────────────────
One-time idempotent script that seeds validated political event patterns into
news_effect_patterns and writes default settings for the political event system.

Safe to re-run — checks active='true' rows by event_type before inserting.

Usage:
  python seed_political_patterns.py              # live run
  python seed_political_patterns.py --dry-run    # print only, no DB writes
  python seed_political_patterns.py --force      # re-insert even if rows exist
"""

import argparse
import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PATTERN CATALOGUE
# Evidence quality: STRONG / MODERATE / WEAK
# Status:          ACTIVE (affects score) | MONITOR_ONLY | DISABLED
# Magnitude:       signed float string, e.g. "-3.5"
# lag_start/end:   business days after event date
# ══════════════════════════════════════════════════════════════════════════════

PATTERNS: list[dict] = [
    # ── ACTIVE patterns ───────────────────────────────────────────────────────
    {
        "event_type":       "GOVT_FORMATION",
        "event_category":   "POLITICAL",
        "lag_start":        "2",
        "lag_end":          "5",
        "magnitude":        "3.5",
        "market_regime":    "ALL",
        "status":           "ACTIVE",
        "evidence_quality": "STRONG",
        "confidence_basis": "4-model consensus: new stable government relieves policy uncertainty; "
                            "NEPSE rallied avg +3.5 pct in 2-5 business days across 6 confirmed formations",
        "occurrence_count": "6",
        "weighted_accuracy": "0.83",
        "notes":            "Effect stronger in BEAR regime (relief rally); weaker if market already priced in coalition deal",
    },
    {
        "event_type":       "CORRUPTION_PROBE",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "3",
        "magnitude":        "-2.0",
        "market_regime":    "ALL",
        "status":           "ACTIVE",
        "evidence_quality": "MODERATE",
        "confidence_basis": "3-model consensus: high-profile CIB/CIAA investigations trigger FII pullback "
                            "and sentiment sell-off in banking/finance sector; avg -2 pct within 3 days",
        "occurrence_count": "5",
        "weighted_accuracy": "0.70",
        "notes":            "Impact highest when probe targets SOE-linked entities; weaker for minor officials",
    },
    {
        "event_type":       "CIVIL_UNREST",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "2",
        "magnitude":        "-1.5",
        "market_regime":    "ALL",
        "status":           "ACTIVE",
        "evidence_quality": "MODERATE",
        "confidence_basis": "3-model consensus: bandh/strike reduces broker activity on same/next trading day; "
                            "avg -1.5 pct move; partial day bandh shows ~-0.8 pct",
        "occurrence_count": "9",
        "weighted_accuracy": "0.67",
        "notes":            "Bandh effect well-documented but short-lived (1-2 days); note: bandh weight in "
                            "nepal_pulse is separately -1 on top of this lagged signal",
    },
    {
        "event_type":       "ADMIN",
        "event_category":   "REGULATORY",
        "lag_start":        "1",
        "lag_end":          "4",
        "magnitude":        "-1.0",
        "market_regime":    "ALL",
        "status":           "ACTIVE",
        "evidence_quality": "MODERATE",
        "confidence_basis": "3-model consensus: SEBON margin/collateral rule changes create short-term liquidity "
                            "pressure; forced selling lasts 1-4 business days; avg impact -1 pct",
        "occurrence_count": "4",
        "weighted_accuracy": "0.75",
        "notes":            "Positive regulatory changes (fee reduction, market access) flip sign to +1.0",
    },

    # ── MONITOR_ONLY patterns ─────────────────────────────────────────────────
    {
        "event_type":       "SC_RULING",
        "event_category":   "JUDICIAL",
        "lag_start":        "1",
        "lag_end":          "3",
        "magnitude":        "-2.5",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "MODERATE",
        "confidence_basis": "2-model consensus: Supreme Court orders on company listings or political stability "
                            "have shown market impact but sample size only 3; promoting to MONITOR_ONLY pending more data",
        "occurrence_count": "3",
        "weighted_accuracy": "0.60",
        "notes":            "Direction varies by ruling type; negative magnitude assumes adverse ruling default",
    },
    {
        "event_type":       "HOUSE_DISSOLUTION",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "5",
        "magnitude":        "-4.0",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "MODERATE",
        "confidence_basis": "3-model consensus: parliamentary dissolution creates acute policy vacuum; "
                            "strong sell signal historically but only 2 confirmed instances since 2017",
        "occurrence_count": "2",
        "weighted_accuracy": "0.75",
        "notes":            "Largest magnitude pattern; needs ≥5 occurrences before ACTIVE promotion",
    },
    {
        "event_type":       "COALITION_CHANGE",
        "event_category":   "POLITICAL",
        "lag_start":        "2",
        "lag_end":          "4",
        "magnitude":        "-1.5",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: coalition reshuffles cause short-term uncertainty; "
                            "effect often absorbed within GOVT_FORMATION signal; tracking separately",
        "occurrence_count": "4",
        "weighted_accuracy": "0.50",
        "notes":            "Co-occurs frequently with GOVT_FORMATION; magnitude may be double-counted",
    },
    {
        "event_type":       "CHINA_RELATIONS",
        "event_category":   "GEOPOLITICAL",
        "lag_start":        "2",
        "lag_end":          "5",
        "magnitude":        "1.5",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: positive China engagement (BRI, aid, connectivity) shows mild "
                            "positive sentiment; border/encroachment news is negative; default magnitude is positive",
        "occurrence_count": "4",
        "weighted_accuracy": "0.50",
        "notes":            "Direction is sensitive to news framing; keyword matching may misclassify; keep MONITOR_ONLY",
    },
    {
        "event_type":       "CORRUPTION_ARREST",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "3",
        "magnitude":        "-1.0",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: arrests of high-profile figures create uncertainty; "
                            "weaker and less consistent than CORRUPTION_PROBE; tracking separately",
        "occurrence_count": "3",
        "weighted_accuracy": "0.50",
        "notes":            "May be subsumed under CORRUPTION_PROBE; keeping separate to track independently",
    },
    {
        "event_type":       "TRADE_DEAL",
        "event_category":   "ECONOMIC",
        "lag_start":        "2",
        "lag_end":          "5",
        "magnitude":        "2.0",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: trade MOU/deal announcements lift export-linked stocks; "
                            "limited NEPSE-wide data; only 2 clean instances identified",
        "occurrence_count": "2",
        "weighted_accuracy": "0.50",
        "notes":            "Sector-specific (hydropower, manufacturing); index-level effect unclear",
    },
    {
        "event_type":       "FOREIGN_AID",
        "event_category":   "ECONOMIC",
        "lag_start":        "2",
        "lag_end":          "5",
        "magnitude":        "1.0",
        "market_regime":    "BEAR",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: large aid/grant announcements (MCC, World Bank) can lift sentiment "
                            "but effect is regime-dependent; only significant in BEAR markets",
        "occurrence_count": "3",
        "weighted_accuracy": "0.50",
        "notes":            "MCC announcement 2022 is the cleanest data point; BULL regime shows no measurable effect",
    },
    {
        "event_type":       "INDIA_RELATIONS",
        "event_category":   "GEOPOLITICAL",
        "lag_start":        "1",
        "lag_end":          "4",
        "magnitude":        "-2.0",
        "market_regime":    "ALL",
        "status":           "MONITOR_ONLY",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: India-Nepal friction (blockade, trade restriction) causes sharp "
                            "negative sentiment given trade dependency; positive engagement shows weaker reverse effect",
        "occurrence_count": "3",
        "weighted_accuracy": "0.55",
        "notes":            "Negative magnitude default (friction more common than positive news); "
                            "keyword 'india' alone is noisy — needs additional negative context words",
    },

    # ── DISABLED patterns ─────────────────────────────────────────────────────
    {
        "event_type":       "PARTY_EVENT",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "3",
        "magnitude":        "-0.5",
        "market_regime":    "ALL",
        "status":           "DISABLED",
        "evidence_quality": "WEAK",
        "confidence_basis": "1-model only: intra-party events (conventions, leadership contests) show no "
                            "consistent NEPSE impact; pattern disabled pending better keyword taxonomy",
        "occurrence_count": "5",
        "weighted_accuracy": "0.30",
        "notes":            "Signal too noisy; re-enable only if accuracy improves above 0.50 with ≥8 occurrences",
    },
    {
        "event_type":       "CONSTITUTIONAL_CHANGE",
        "event_category":   "POLITICAL",
        "lag_start":        "2",
        "lag_end":          "7",
        "magnitude":        "2.0",
        "market_regime":    "ALL",
        "status":           "DISABLED",
        "evidence_quality": "WEAK",
        "confidence_basis": "2-model consensus: constitutional amendments are rare one-off events; "
                            "2015 constitution promulgation data point is not generalizable; disabling",
        "occurrence_count": "1",
        "weighted_accuracy": "0.0",
        "notes":            "Single data point from 2015; effect confounded by earthquake recovery; keep disabled",
    },
    {
        "event_type":       "PRESIDENTIAL",
        "event_category":   "POLITICAL",
        "lag_start":        "1",
        "lag_end":          "3",
        "magnitude":        "0.5",
        "market_regime":    "ALL",
        "status":           "DISABLED",
        "evidence_quality": "WEAK",
        "confidence_basis": "1-model only: Presidential/Vice-Presidential elections are largely ceremonial "
                            "in Nepal's system; no consistent NEPSE impact detected across 3 events",
        "occurrence_count": "3",
        "weighted_accuracy": "0.33",
        "notes":            "Keep disabled; promote only if a Presidential action creates genuine policy change",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS DEFAULTS
# These go into the `settings` table as key/value rows.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS: dict[str, str] = {
    "political_event_detection_enabled":   "true",
    "political_event_min_score_threshold": "0.0",   # minimum nepal_pulse score to write a prediction
    "political_pattern_weekly_scoring":    "true",   # learning_hub auto-scores pending predictions
    "political_pattern_quarterly_review":  "true",   # monthly_council runs quarterly pattern review
    "political_lagged_adj_max":            "5.0",    # cap on lagged score adjustment (positive)
    "political_lagged_adj_min":            "-5.0",   # cap on lagged score adjustment (negative)
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _existing_event_types() -> set[str]:
    """Return event_type values already present with active='true'."""
    from sheets import run_raw_sql
    rows = run_raw_sql(
        "SELECT event_type FROM news_effect_patterns WHERE active = 'true'"
    ) or []
    return {r["event_type"] for r in rows}


def _existing_setting_keys() -> set[str]:
    """Return keys already present in the settings table."""
    from sheets import run_raw_sql
    rows = run_raw_sql("SELECT key FROM settings") or []
    return {r["key"] for r in rows}


def _insert_pattern(p: dict, dry_run: bool) -> None:
    from sheets import write_row
    row = {
        "event_type":        p["event_type"],
        "event_category":    p["event_category"],
        "lag_start":         p["lag_start"],
        "lag_end":           p["lag_end"],
        "magnitude":         p["magnitude"],
        "market_regime":     p["market_regime"],
        "status":            p["status"],
        "confidence_basis":  p["confidence_basis"],
        "evidence_quality":  p["evidence_quality"],
        "occurrence_count":  p["occurrence_count"],
        "weighted_accuracy": p["weighted_accuracy"],
        "source":            "research_validation_2026",
        "active":            "true",
        "notes":             p.get("notes", ""),
    }
    if dry_run:
        log.info("[DRY-RUN] Would insert pattern: %s (%s)", p["event_type"], p["status"])
        return
    write_row("news_effect_patterns", row)
    log.info("Inserted pattern: %s (%s)", p["event_type"], p["status"])


def _upsert_setting(key: str, value: str, existing: set[str], dry_run: bool) -> None:
    if dry_run:
        action = "update" if key in existing else "insert"
        log.info("[DRY-RUN] Would %s setting: %s = %s", action, key, value)
        return
    from sheets import upsert_row
    upsert_row("settings", {"key": key}, {"key": key, "value": value})
    log.info("Upserted setting: %s = %s", key, value)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(dry_run: bool = False, force: bool = False) -> None:
    log.info("═══ seed_political_patterns ════════════════════════════════")
    log.info("dry_run=%s  force=%s  date=%s", dry_run, force, datetime.utcnow().date())

    # ── Patterns ──────────────────────────────────────────────────────────────
    existing = _existing_event_types() if not force else set()
    inserted = skipped = 0

    for p in PATTERNS:
        et = p["event_type"]
        if et in existing:
            log.info("SKIP (already exists): %s", et)
            skipped += 1
            continue
        _insert_pattern(p, dry_run)
        inserted += 1

    log.info("Patterns — inserted: %d  skipped: %d  total: %d", inserted, skipped, len(PATTERNS))

    # ── Settings ──────────────────────────────────────────────────────────────
    existing_keys = _existing_setting_keys() if not dry_run else set()
    for key, value in DEFAULTS.items():
        _upsert_setting(key, value, existing_keys, dry_run)

    log.info("Settings — written: %d", len(DEFAULTS))
    log.info("═══ seed complete ═══════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed political event patterns into NEPSE DB")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen; no DB writes")
    parser.add_argument("--force",   action="store_true", help="Re-insert even if rows already exist")
    args = parser.parse_args()

    if args.force and args.dry_run:
        log.error("--force and --dry-run are mutually exclusive")
        sys.exit(1)

    main(dry_run=args.dry_run, force=args.force)
