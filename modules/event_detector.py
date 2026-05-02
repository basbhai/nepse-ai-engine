"""
modules/event_detector.py — NEPSE AI Engine
────────────────────────────────────────────
Detects political event types from daily_context_log headlines.
Makes lag predictions and tracks their accuracy.
Builds pre-digested context block for claude_analyst.

Called by:
  modules/nepal_pulse.py   — detect + predict + score (every 30 min)
  claude_analyst.py        — build context block (every trading cycle)
  analysis/learning_hub.py — score pending predictions (weekly)
"""

import logging
import uuid
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD MAP
# ══════════════════════════════════════════════════════════════════════════════

_POLITICAL_KEYWORDS: list[tuple[str, str]] = [
    ("bandh|strike|chakka jam",                   "CIVIL_UNREST"),
    ("resign|impeach|dissolved|dissolution",       "HOUSE_DISSOLUTION"),
    ("supreme court|sc ruling|court order",        "SC_RULING"),
    ("prime minister|pm sworn|new government|cabinet formed", "GOVT_FORMATION"),
    ("coalition|alliance shifted|new majority",    "COALITION_CHANGE"),
    ("cib|probe|investigation|inquiry|fraud",      "CORRUPTION_PROBE"),
    ("arrested|detained|held",                     "CORRUPTION_ARREST"),
    ("china|border encroach|boundary",             "CHINA_RELATIONS"),
    ("india|blockade|trade restrict",              "INDIA_RELATIONS"),
    ("protest|unrest|demonstration|uprising",      "CIVIL_UNREST"),
    ("election|voting|polling",                    "ELECTION"),
    ("president|vice president",                   "PRESIDENTIAL"),
    ("anti-corruption|acc|property audit",         "ANTI_CORRUPTION"),
    ("pledge fee|collateral|margin rule",          "ADMIN"),
]

_ECONOMY_KEYWORDS: list[tuple[str, str]] = [
    ("trade deal|export agreement|mou",                   "TRADE_DEAL"),
    ("foreign aid|grant|mcc|millenium",                   "FOREIGN_AID"),
    ("budget|fiscal|salary reform",                       "FISCAL_POLICY"),
]

# NEPSE headlines handled by existing nepal_pulse scoring — no event classification


def _skip_weekends(date: datetime, days: int) -> datetime:
    """Add `days` business days to date, skipping Sat/Sun."""
    result = date
    added  = 0
    while added < days:
        result += timedelta(days=1)
        if result.weekday() < 5:  # Mon-Fri
            added += 1
    return result


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME
# ══════════════════════════════════════════════════════════════════════════════

def _get_market_regime() -> str:
    """
    Returns BULL / BEAR / SIDEWAYS based on NEPSE 20-day trend.
    Reads last 25 rows from nepse_indices (index_id='58').
    Fails silently — returns 'UNKNOWN' on error.
    """
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT current_value FROM nepse_indices
            WHERE index_id = '58' AND current_value IS NOT NULL
            ORDER BY date DESC LIMIT 25
            """,
        ) or []
        if len(rows) < 2:
            return "UNKNOWN"
        values = [float(r["current_value"]) for r in rows if r.get("current_value")]
        if not values:
            return "UNKNOWN"
        current = values[0]
        avg_20  = sum(values[:20]) / min(len(values), 20)
        if current > avg_20 * 1.02:
            return "BULL"
        elif current < avg_20 * 0.98:
            return "BEAR"
        return "SIDEWAYS"
    except Exception as e:
        log.warning("_get_market_regime failed: %s", e)
        return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# PATTERN LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_active_patterns() -> list[dict]:
    """
    Load ACTIVE patterns from news_effect_patterns.
    MONITOR_ONLY patterns are NOT returned — they don't affect nepal_score.
    Fails silently — returns [] on error.
    """
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT id, event_type, event_category, lag_start, lag_end,
                   magnitude, market_regime, status, confidence_basis,
                   evidence_quality, occurrence_count, weighted_accuracy,
                   source, notes
            FROM news_effect_patterns
            WHERE status = 'ACTIVE' AND active = 'true'
            ORDER BY id
            """,
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("_load_active_patterns failed: %s", e)
        return []


def _load_all_patterns() -> list[dict]:
    """Load all non-DISABLED patterns (ACTIVE + MONITOR_ONLY) for context building."""
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT id, event_type, event_category, lag_start, lag_end,
                   magnitude, market_regime, status, confidence_basis,
                   evidence_quality, occurrence_count, weighted_accuracy, notes
            FROM news_effect_patterns
            WHERE status != 'DISABLED' AND active = 'true'
            ORDER BY id
            """,
        ) or []
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("_load_all_patterns failed: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# EVENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_event_types(
    headlines_political: str,
    headlines_economy: str,
    headlines_nepse: str,
) -> list[dict]:
    """
    Classify headlines into event types using keyword matching.
    Returns list of detected events. Returns [] on any error. Never raises.
    """
    try:
        detected_types: list[tuple[str, str, str]] = []  # (event_type, category, sample)
        text_pol  = (headlines_political or "").lower()
        text_eco  = (headlines_economy  or "").lower()

        for pattern, event_type in _POLITICAL_KEYWORDS:
            keywords = [k.strip() for k in pattern.split("|")]
            for kw in keywords:
                if kw and kw in text_pol:
                    sample = _extract_sample(text_pol, kw)
                    detected_types.append((event_type, "POLITICAL", sample))
                    break

        for pattern, event_type in _ECONOMY_KEYWORDS:
            keywords = [k.strip() for k in pattern.split("|")]
            for kw in keywords:
                if kw and kw in text_eco:
                    sample = _extract_sample(text_eco, kw)
                    detected_types.append((event_type, "ECONOMY", sample))
                    break

        if not detected_types:
            return []

        # Deduplicate by event_type (keep first occurrence)
        seen: set[str] = set()
        unique: list[tuple[str, str, str]] = []
        for et, cat, sample in detected_types:
            if et not in seen:
                seen.add(et)
                unique.append((et, cat, sample))

        # Determine primary event: highest abs magnitude from patterns
        try:
            patterns = _load_all_patterns()
            pattern_map = {p["event_type"]: float(p.get("magnitude") or 0) for p in patterns}
        except Exception:
            pattern_map = {}

        def _abs_mag(et: str) -> float:
            return abs(pattern_map.get(et, 0.0))

        primary_et = max(unique, key=lambda x: _abs_mag(x[0]))[0]

        cluster_id = str(uuid.uuid4())
        co_count   = len(unique)

        result = []
        for et, cat, sample in unique:
            result.append({
                "event_type":         et,
                "event_category":     cat,
                "headline_sample":    sample,
                "event_cluster_id":   cluster_id,
                "co_occurrence_count": co_count,
                "primary":            (et == primary_et),
            })
        return result

    except Exception as e:
        log.warning("detect_event_types failed: %s", e)
        return []


def _extract_sample(text: str, keyword: str) -> str:
    """Extract a short snippet around the matched keyword."""
    idx = text.find(keyword)
    if idx == -1:
        return text[:80]
    start = max(0, idx - 20)
    end   = min(len(text), idx + 60)
    return text[start:end].strip()


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_prediction(
    event: dict,
    pattern: dict,
    nepal_crisis_score: int,
    event_date_str: str,
) -> bool:
    """
    Write one prediction row to pattern_validation_log.
    Only writes ACTIVE patterns, primary events, no duplicate (event_type, event_date).
    Returns True on success, False on failure. Never raises.
    """
    try:
        if pattern.get("status") != "ACTIVE":
            return False
        if not event.get("primary"):
            return False

        from sheets import run_raw_sql, write_row

        # Dedup check
        existing = run_raw_sql(
            """
            SELECT id FROM pattern_validation_log
            WHERE event_type = %s AND event_date = %s AND outcome = 'PENDING'
            LIMIT 1
            """,
            (event["event_type"], event_date_str),
        ) or []
        if existing:
            log.debug("Prediction already exists for %s %s — skipping", event["event_type"], event_date_str)
            return False

        # Compute prediction window (skip weekends)
        from zoneinfo import ZoneInfo
        NST = ZoneInfo("Asia/Kathmandu")
        event_dt   = datetime.strptime(event_date_str, "%Y-%m-%d").replace(tzinfo=NST)
        lag_start  = int(pattern.get("lag_start") or 0)
        lag_end    = int(pattern.get("lag_end")   or 0)

        start_dt = _skip_weekends(event_dt, lag_start)
        end_dt   = _skip_weekends(event_dt, lag_end)

        market_regime = _get_market_regime()

        write_row("pattern_validation_log", {
            "event_type":                     event["event_type"],
            "event_date":                     event_date_str,
            "event_cluster_id":               event.get("event_cluster_id", ""),
            "pattern_id":                     str(pattern.get("id", "")),
            "lag_start":                      str(pattern.get("lag_start", "")),
            "lag_end":                        str(pattern.get("lag_end", "")),
            "magnitude_applied":              str(pattern.get("magnitude", "")),
            "market_regime":                  market_regime,
            "nepal_crisis_score_at_detection": str(nepal_crisis_score),
            "predicted_date_start":           start_dt.strftime("%Y-%m-%d"),
            "predicted_date_end":             end_dt.strftime("%Y-%m-%d"),
            "co_occurrence_count":            str(event.get("co_occurrence_count", 1)),
            "outcome":                        "PENDING",
            "scored_by":                      "pending",
        })
        return True

    except Exception as e:
        log.warning("write_prediction failed: %s", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PENDING PREDICTION SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_pending_predictions(dry_run: bool = False) -> dict:
    """
    Auto-score PENDING predictions whose window has passed.
    Returns {"scored": N, "correct": N, "wrong_direction": N, "wrong_timing": N}.
    Never raises. Fails silently per row.
    """
    result = {"scored": 0, "correct": 0, "wrong_direction": 0, "wrong_timing": 0}
    try:
        from sheets import run_raw_sql
        from zoneinfo import ZoneInfo
        NST   = ZoneInfo("Asia/Kathmandu")
        today = datetime.now(NST).strftime("%Y-%m-%d")

        pending = run_raw_sql(
            """
            SELECT id, event_type, event_date, magnitude_applied,
                   predicted_date_start, predicted_date_end
            FROM pattern_validation_log
            WHERE outcome = 'PENDING' AND predicted_date_end < %s
            ORDER BY predicted_date_end ASC
            """,
            (today,),
        ) or []

        for row in pending:
            try:
                _score_one_prediction(row, dry_run, result)
            except Exception as e:
                log.warning("score_pending_predictions: error on id=%s: %s", row.get("id"), e)

    except Exception as e:
        log.warning("score_pending_predictions failed: %s", e)

    return result


def _score_one_prediction(row: dict, dry_run: bool, tally: dict) -> None:
    from sheets import run_raw_sql
    from zoneinfo import ZoneInfo
    NST = ZoneInfo("Asia/Kathmandu")

    start_str = row["predicted_date_start"]
    end_str   = row["predicted_date_end"]
    mag_pred  = float(row.get("magnitude_applied") or 0)
    pred_id   = row["id"]

    # Fetch NEPSE change_pct values within window
    nepse_rows = run_raw_sql(
        """
        SELECT date, change_pct FROM nepse_indices
        WHERE index_id = '58'
          AND date >= %s AND date <= %s
          AND change_pct IS NOT NULL
        ORDER BY date ASC
        """,
        (start_str, end_str),
    ) or []

    if not nepse_rows:
        return  # No NEPSE data for this window — skip

    # Find strongest absolute move in window
    changes = []
    for r in nepse_rows:
        try:
            changes.append(float(r["change_pct"]))
        except (TypeError, ValueError):
            pass

    if not changes:
        return

    actual_peak = max(changes, key=abs)

    # Score outcome
    if mag_pred >= 0:
        if actual_peak >= 0:
            outcome = "CORRECT"
        else:
            outcome = "WRONG_DIRECTION"
    else:
        if actual_peak < 0:
            outcome = "CORRECT"
        else:
            outcome = "WRONG_DIRECTION"

    mag_error = round(actual_peak - mag_pred, 3)
    scored_at = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

    tally["scored"] += 1
    tally[outcome.lower() if outcome in ("CORRECT", "WRONG_DIRECTION") else "wrong_timing"] += 1
    # Map outcome to tally key
    if outcome == "CORRECT":
        tally["correct"] += 1
    elif outcome == "WRONG_DIRECTION":
        tally["wrong_direction"] += 1

    if dry_run:
        log.info("[DRY RUN] Would score id=%s → %s (actual=%.2f, pred=%.2f)",
                 pred_id, outcome, actual_peak, mag_pred)
        return

    run_raw_sql(
        """
        UPDATE pattern_validation_log
        SET actual_nepse_pct = %s,
            magnitude_error  = %s,
            outcome          = %s,
            scored_by        = 'auto',
            scored_at        = %s
        WHERE id = %s
        """,
        (str(actual_peak), str(mag_error), outcome, scored_at, pred_id),
    )
    log.info("Scored prediction id=%s event=%s → %s (actual=%.2f pred=%.2f)",
             pred_id, row["event_type"], outcome, actual_peak, mag_pred)


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_claude_context() -> str:
    """
    Build pre-digested event context for claude_analyst.
    Returns formatted string with active/upcoming impact windows and recent outcomes.
    Returns "" if no active windows. Always fails silently — never raises.
    """
    try:
        from sheets import run_raw_sql
        from zoneinfo import ZoneInfo
        NST   = ZoneInfo("Asia/Kathmandu")
        today = datetime.now(NST).strftime("%Y-%m-%d")

        # ── Active impact windows (today is inside them) ──────────────────────
        active_rows = run_raw_sql(
            """
            SELECT pvl.id, pvl.event_type, pvl.event_date,
                   pvl.predicted_date_start, pvl.predicted_date_end,
                   pvl.magnitude_applied, pvl.market_regime,
                   nep.evidence_quality, nep.weighted_accuracy, nep.occurrence_count
            FROM pattern_validation_log pvl
            LEFT JOIN news_effect_patterns nep
                ON nep.event_type = pvl.event_type AND nep.active = 'true'
            WHERE pvl.outcome = 'PENDING'
              AND pvl.predicted_date_start <= %s
              AND pvl.predicted_date_end   >= %s
            ORDER BY ABS(pvl.magnitude_applied::float) DESC
            """,
            (today, today),
        ) or []

        # ── Upcoming windows (peaks within 3 days) ────────────────────────────
        future_cutoff = (datetime.now(NST) + timedelta(days=3)).strftime("%Y-%m-%d")
        upcoming_rows = run_raw_sql(
            """
            SELECT pvl.event_type, pvl.event_date,
                   pvl.predicted_date_start, pvl.predicted_date_end,
                   pvl.magnitude_applied
            FROM pattern_validation_log pvl
            WHERE pvl.outcome = 'PENDING'
              AND pvl.predicted_date_start > %s
              AND pvl.predicted_date_start <= %s
            ORDER BY pvl.predicted_date_start ASC
            """,
            (today, future_cutoff),
        ) or []

        # ── Recent outcomes (last 14 days) ────────────────────────────────────
        recent_cutoff = (datetime.now(NST) - timedelta(days=14)).strftime("%Y-%m-%d")
        recent_rows = run_raw_sql(
            """
            SELECT event_type, magnitude_applied, actual_nepse_pct, outcome
            FROM pattern_validation_log
            WHERE outcome IN ('CORRECT', 'WRONG_DIRECTION', 'WRONG_TIMING')
              AND scored_at >= %s
            ORDER BY scored_at DESC LIMIT 10
            """,
            (recent_cutoff,),
        ) or []

        if not active_rows and not upcoming_rows and not recent_rows:
            return ""

        lines = ["═══ POLITICAL EVENT CONTEXT ═══"]

        # Active windows
        if active_rows:
            lines.append("")
            lines.append("[ACTIVE IMPACT WINDOWS — today is inside these]")
            for r in active_rows:
                mag   = float(r.get("magnitude_applied") or 0)
                wa    = r.get("weighted_accuracy") or "0"
                n     = r.get("occurrence_count")  or "0"
                eq    = r.get("evidence_quality")  or "?"
                regime = r.get("market_regime", "")
                lines.append(
                    f"{r['event_type']} detected {r['event_date']}"
                )
                lines.append(
                    f"  Impact window: {r['predicted_date_start']} – {r['predicted_date_end']} "
                    f"| Expected: {mag:+.1f}% | Market regime: {r['market_regime']}"
                )
                if wa and float(wa) > 0:
                    lines.append(
                        f"  Pattern accuracy: {float(wa)*100:.0f}% ({n} events, weighted) "
                        f"| Evidence: {eq}"
                    )
                if regime in ("BEAR", "CRISIS") and mag < 0:
                    lines.append("  → Regime amplifies negative signal")

        # Upcoming windows
        if upcoming_rows:
            lines.append("")
            lines.append("[UPCOMING IMPACT WINDOWS — peaks within 3 days]")
            for r in upcoming_rows:
                mag = float(r.get("magnitude_applied") or 0)
                lines.append(
                    f"{r['event_type']} detected {r['event_date']}"
                )
                lines.append(
                    f"  Impact window: {r['predicted_date_start']} – {r['predicted_date_end']} "
                    f"| Expected: {mag:+.1f}%"
                )

        # Cluster warning
        if active_rows and len(active_rows) >= 3:
            earliest = min(r["event_date"] for r in active_rows)
            lines.append("")
            lines.append(
                f"[CLUSTER WARNING] {len(active_rows)} co-occurring events detected {earliest} "
                "→ signal reliability reduced"
            )

        # Recent outcomes
        if recent_rows:
            lines.append("")
            lines.append("[RECENT OUTCOMES — last 14 days]")
            for r in recent_rows:
                mag    = r.get("magnitude_applied", "?")
                actual = r.get("actual_nepse_pct", "?")
                lines.append(
                    f"{r['event_type']}: {r['outcome']} "
                    f"(predicted {mag}%, actual {actual}%)"
                )

        # Action bias
        if active_rows:
            lines.append("")
            total_pressure = sum(float(r.get("magnitude_applied") or 0) for r in active_rows)
            total_pressure = max(-5.0, min(5.0, total_pressure))
            et_list = " and ".join(r["event_type"] for r in active_rows[:2])
            lines.append("TODAY ACTION BIAS:")
            lines.append(f"→ Inside {et_list} window{'s' if len(active_rows) > 1 else ''} simultaneously")
            if total_pressure < -0.5:
                lines.append("→ Avoid aggressive BUY. Prefer WAIT unless strong technical confirmation.")
            elif total_pressure > 0.5:
                lines.append("→ Political tailwind. Consider BUY if technicals confirm.")
            lines.append(f"→ Combined expected pressure: {total_pressure:+.1f}%")

        return "\n".join(lines)

    except Exception as e:
        log.warning("build_claude_context failed: %s", e)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def get_accuracy_summary() -> dict:
    """
    Return accuracy stats per event_type for weekly GPT and monthly council.
    Fails silently — returns {} on error.
    """
    try:
        from sheets import run_raw_sql
        rows = run_raw_sql(
            """
            SELECT event_type, outcome, magnitude_applied, actual_nepse_pct
            FROM pattern_validation_log
            WHERE outcome != 'PENDING'
            ORDER BY event_type, id DESC
            """,
        ) or []

        # Group by event_type
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            et = r["event_type"]
            grouped.setdefault(et, []).append(r)

        summary = {}
        for et, entries in grouped.items():
            total          = len(entries)
            correct        = sum(1 for e in entries if e["outcome"] == "CORRECT")
            wrong_dir      = sum(1 for e in entries if e["outcome"] == "WRONG_DIRECTION")
            wrong_timing   = sum(1 for e in entries if e["outcome"] == "WRONG_TIMING")
            last_3         = [e["outcome"] for e in entries[:3]]

            # Weighted accuracy: CORRECT=1.0, WRONG_DIRECTION=0.0, WRONG_TIMING=0.5
            weights = {"CORRECT": 1.0, "WRONG_DIRECTION": 0.0, "WRONG_TIMING": 0.5}
            wa = sum(weights.get(e["outcome"], 0) for e in entries) / total if total else 0.0

            summary[et] = {
                "total":            total,
                "correct":          correct,
                "wrong_direction":  wrong_dir,
                "wrong_timing":     wrong_timing,
                "weighted_accuracy": round(wa, 3),
                "last_3_outcomes":  last_3,
            }

        return summary

    except Exception as e:
        log.warning("get_accuracy_summary failed: %s", e)
        return {}
