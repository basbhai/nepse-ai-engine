"""
nepal_pulse.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Nepal-specific domestic market context score.
          Runs every 30 minutes during market hours.

Score meaning:
  +5  Ideal conditions  — stable politics, low rates, strong BOP
   0  Neutral           — mixed signals
  -5  Crisis conditions — bandh + political chaos + high FD rates

How it fits the system:
  geo_sentiment.py  → geo_score   (global: Crude, VIX, Nifty, DXY, Gold)
  nepal_pulse.py    → nepal_score (domestic: politics, rates, bandh)
  combined_geo = geo_score + nepal_score  (range: -10 to +10)

HEADLINE DEDUPLICATION (added April 2026):
  Headlines are hashed (MD5) and stored in settings key
  SEEN_HEADLINE_HASHES as JSON: {"hash": "YYYY-MM-DD HH:MM"}.
  Each run filters out already-seen headlines before sending to Deepseek.
  TTL = 48 hours — hashes older than 48h are dropped so truly recurring
  stories after 2 days get picked up again.
  This prevents week-old resignations / events from re-triggering
  crisis_detected=YES on every 30-min cycle.

CHANGES FROM PREVIOUS VERSION:
  - Removed duplicate nrb_rate_decision key from SCORE_MAP
  - Removed china_nepal and remittance from scoring (no paper evidence)
  - Renamed rbi → nrb_rate_decision throughout
  - Wired in fd_rate_level (reads from FD_RATE_PCT setting)
  - Wired in lending_rate_level (reads from macro_data table)
  - Fixed inflation_pct typo ("ict") in _read_nrb_macro()
  - Fixed log typos in _read_nrb_macro()
  - Moved google.genai import inside _ask_gemini() — no top-level crash
─────────────────────────────────────────────────────────────────────────────
"""

import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from sheets import (
    get_setting,
    update_setting,
    write_nepal_pulse,
    get_latest_pulse,
    get_macro_data,
)
from calendar_guard import flag_adhoc_closure, today_nst
from AI import ask_deepseek_text
from config import NST

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [NEPAL_PULSE] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NEWS_SOURCES = [
    "https://onlinekhabar.com/feed",
    "https://www.sharesansar.com/category/latest",
    "https://www.sharesansar.com/announcement",
    "https://ekantipur.com/business",
    "https://ekantipur.com/politics",
    "https://eng.merolagani.com/",
]

NEWS_TIMEOUT = 10

# Headline cache settings key and TTL
HEADLINE_CACHE_KEY = "SEEN_HEADLINE_HASHES"
HEADLINE_CACHE_TTL_HOURS = 48

# ── Nepal location anchors ────────────────────────────────────────────────────
NEPAL_ANCHORS = [
    "nepal", "नेपाल", "kathmandu", "काठमाडौं",
    "pokhara", "biratnagar", "birgunj",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — HEADLINE DEDUPLICATION CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _hash_headline(headline: str) -> str:
    """MD5 hash of lowercased, stripped headline."""
    return hashlib.md5(headline.lower().strip().encode()).hexdigest()


def _load_seen_hashes() -> dict[str, str]:
    """
    Load seen headline hashes from settings.
    Returns dict: {md5_hash: "YYYY-MM-DD HH:MM"} for hashes within TTL.
    Silently returns empty dict on any failure.
    """
    try:
        raw = get_setting(HEADLINE_CACHE_KEY, default="{}")
        cache: dict = json.loads(raw) if raw else {}

        # Drop expired entries (older than TTL)
        cutoff = datetime.now(NST) - timedelta(hours=HEADLINE_CACHE_TTL_HOURS)
        active = {}
        for h, ts_str in cache.items():
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M").replace(
                    tzinfo=NST.utcoffset(datetime.now())
                    if hasattr(NST, "utcoffset") else None
                )
                # Simple string comparison works since format is ISO-like
                if ts_str >= cutoff.strftime("%Y-%m-%d %H:%M"):
                    active[h] = ts_str
            except Exception:
                pass  # drop malformed entries

        expired = len(cache) - len(active)
        if expired:
            log.info("Headline cache: dropped %d expired hashes (>%dh old)",
                     expired, HEADLINE_CACHE_TTL_HOURS)
        log.info("Headline cache: %d active hashes loaded", len(active))
        return active

    except Exception as e:
        log.warning("_load_seen_hashes failed: %s — proceeding without cache", e)
        return {}


def _save_seen_hashes(cache: dict[str, str]) -> None:
    """
    Persist updated hash cache to settings.
    Silently fails — never blocks the main pipeline.
    """
    try:
        # Cap at 2000 entries to keep settings value size reasonable
        # Keep newest entries if over limit (sort by timestamp desc)
        if len(cache) > 2000:
            sorted_items = sorted(cache.items(), key=lambda x: x[1], reverse=True)
            cache = dict(sorted_items[:2000])
            log.info("Headline cache: capped at 2000 entries")

        update_setting(
            HEADLINE_CACHE_KEY,
            json.dumps(cache, ensure_ascii=False),
            set_by="nepal_pulse",
        )
        log.info("Headline cache: saved %d hashes to settings", len(cache))
    except Exception as e:
        log.warning("_save_seen_hashes failed: %s", e)


def _filter_new_headlines(df: pd.DataFrame, seen: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Filter DataFrame to only new (unseen) headlines.
    Also adds hashes of new headlines to the seen dict.

    Returns:
        (new_df, updated_seen_dict)
    """
    if df.empty:
        return df, seen

    now_str = datetime.now(NST).strftime("%Y-%m-%d %H:%M")
    new_rows = []
    skipped  = 0
    updated  = dict(seen)  # copy — don't mutate caller's dict

    for _, row in df.iterrows():
        h = _hash_headline(str(row.get("headline", "")))
        if h in seen:
            skipped += 1
        else:
            new_rows.append(row)
            updated[h] = now_str

    new_df = pd.DataFrame(new_rows, columns=df.columns) if new_rows else pd.DataFrame(columns=df.columns)

    log.info(
        "Headline dedup: %d total | %d new | %d already seen (filtered out)",
        len(df), len(new_rows), skipped,
    )
    return new_df, updated


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — READ MANUAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def _read_manual_settings() -> dict:
    """
    Read manually-set Neon settings.
    GULF_STABILITY and REMITTANCE_RISK are auto-updated by Deepseek.
    Returns dict with all fields, defaulting to neutral if not set.
    """
    log.info("Reading manual settings from Neon...")

    settings = {
        "india_nepal_relations": get_setting(
            "INDIA_NEPAL_RELATIONS", default="STABLE"
        ).upper().strip(),

        "gulf_stability": get_setting(
            "GULF_STABILITY", default="STABLE"
        ).upper().strip(),

        "nrb_rate_decision": get_setting(
            "NRB_RATE_DECISION", default="UNCHANGED"
        ).upper().strip(),
    }

    log.info(
        "Manual settings: India-Nepal=%s | Gulf=%s | NRB=%s",
        settings["india_nepal_relations"],
        settings["gulf_stability"],
        settings["nrb_rate_decision"],
    )

    return settings


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCRAPE NEPAL NEWS INTO DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_headlines(url: str) -> list[dict]:
    """
    Fetch headlines from one source — RSS or HTML auto-detected.
    Returns list of {source, headline} dicts.
    Returns empty list on any failure — never raises.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        }
        resp = requests.get(url, timeout=NEWS_TIMEOUT, headers=headers)
        resp.raise_for_status()

        source_name  = url.split("/")[2].replace("www.", "")
        headlines    = []
        content_type = resp.headers.get("content-type", "")

        if "xml" in content_type or "rss" in content_type or url.endswith("/feed"):
            soup = BeautifulSoup(resp.content, "xml")
            for item in soup.find_all("item"):
                title = item.find("title")
                desc  = item.find("description")
                if title and title.get_text(strip=True):
                    headlines.append({
                        "source":   source_name,
                        "headline": title.get_text(strip=True).lower(),
                    })
                if desc and desc.get_text(strip=True):
                    headlines.append({
                        "source":   source_name,
                        "headline": desc.get_text(strip=True)[:150].lower(),
                    })
        else:
            soup = BeautifulSoup(resp.content, "html.parser")

            if "sharesansar" in url:
                master_selector = (
                    "h4.featured-news-title, "
                    ".news-title, "
                    "ul.news-list li, "
                    ".announcement-list .featured-news-list div:nth-of-type(2) h4"
                )
                tags = soup.select(master_selector)
            else:
                tags = soup.find_all(["h1", "h2", "h3", "h4"])

            for tag in tags:
                text = tag.get_text(strip=True)
                if len(text) > 20:
                    headlines.append({
                        "source":   source_name,
                        "headline": text.lower(),
                    })

        log.info("News %s: fetched %d headlines", source_name, len(headlines))
        return headlines

    except requests.exceptions.Timeout:
        log.warning("News %s: timed out — skipping", url.split("/")[2])
        return []
    except Exception as exc:
        log.warning("News %s: failed (%s) — skipping", url.split("/")[2], exc)
        return []


def _build_headlines_df() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Scrape all sources, deduplicate against seen cache.

    Returns:
        (new_headlines_df, updated_cache)
        new_headlines_df — only headlines NOT seen in the last 48h
        updated_cache    — cache dict to save after analysis completes
    """
    all_rows: list[dict] = []
    for url in NEWS_SOURCES:
        all_rows.extend(_fetch_headlines(url))

    log.info("Total headlines collected (before dedup): %d", len(all_rows))

    if not all_rows:
        return pd.DataFrame(columns=["source", "headline"]), {}

    raw_df = pd.DataFrame(all_rows)

    # Load cache and filter
    seen          = _load_seen_hashes()
    new_df, updated_cache = _filter_new_headlines(raw_df, seen)

    return new_df, updated_cache


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DEEPSEEK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _build_gemini_prompt(df: pd.DataFrame) -> str:
    data_str = df.to_string(index=False)
    return f"""You are a Nepal financial market analyst.
Analyze these Nepal news headlines scraped today.
Detect signals relevant to Nepal stock market (NEPSE).

1. Is there a bandh, strike, chakka jam, or transport/business shutdown IN NEPAL today?
   Only count events physically happening inside Nepal — ignore global news.
2. Is any IPO, FPO, or right share application open on NEPSE today? must be general public not right, reserved issue.
3. Is there a political crisis , PM resignation, or government instability IN NEPAL, only central government related?
4. What is the Gulf/Middle East stability? (affects Nepal remittance workers abroad)
5. What is the remittance risk level based on Gulf/foreign employment news?
6. What is the overall Nepal market sentiment today?
7. Crisis details should be only that effect share market.
8 india_nepal_relations: "STABLE" | "TENSE" | "HOSTILE"
  (based on any India-Nepal border, trade, treaty, political news recent only)
9 nrb_rate_decision: "CUT" | "RAISED" | "UNCHANGED"
  (based on any NRB monetary policy announcement)


Headlines:
{data_str}

Return ONLY this JSON object with no other text, no markdown, no explanation:
{{
  "bandh_today": "YES or NO",
  "bandh_detail": "who called it and where in Nepal, or empty string",
  "ipo_fpo_active": "YES or NO",
  "ipo_fpo_detail": "company name and issue type, or empty string",
  "crisis_detected": "YES or NO",
  "crisis_detail": "what happened in Nepal, or empty string",
  "gulf_signal": "STABLE or TENSE or CRISIS",
  "gulf_detail": "reason from headlines, or empty string",
  "remittance_signal": "LOW or MEDIUM or HIGH",
  "remittance_detail": "reason from headlines, or empty string",
  "overall_sentiment": "POSITIVE or NEUTRAL or NEGATIVE",
  "key_event": "single most important news that will directly effect NEPSE",
  "headlines_politics": "title 1: lorem ipsum |title: 2: lorem_ipum[only related to Nepal]",
  "headlines_economy":  "title 1: lorem ipsum |title: 2: lorem_ipum [only related to Nepal]",
  "headlines_stock":    "title 1: lorem ipsum |title: 2: lorem_ipum [only related to NEPSE]"
}}"""


def _is_nepal_source(source: str) -> bool:
    nepal_sources = {
        "onlinekhabar.com", "ekantipur.com", "sharesansar.com",
        "eng.merolagani.com", "merolagani.com", "thehimalayantimes.com",
        "kathmandupost.com", "myrepublica.nagariknetwork.com",
        "ratopati.com", "setopati.com",
    }
    return any(ns in source.lower() for ns in nepal_sources)


def _has_nepal_anchor(headline: str) -> bool:
    return any(anchor in headline for anchor in NEPAL_ANCHORS)


def _keyword_detect(df: pd.DataFrame) -> dict:
    """
    Pure keyword fallback — no AI dependency.
    Used when Deepseek is unavailable or key not set.
    """
    bandh_keywords  = ["bandh", "strike", "chakka jam", "shutdown", "closure"]
    crisis_keywords = ["resign", "dismiss", "impeach", "dissolution", "protest", "crisis"]

    headlines = df["headline"].str.lower().tolist() if not df.empty else []

    bandh_today  = any(k in h for h in headlines for k in bandh_keywords)
    crisis_today = any(k in h for h in headlines for k in crisis_keywords)

    bandh_detail  = next((h for h in headlines for k in bandh_keywords if k in h), "")
    crisis_detail = next((h for h in headlines for k in crisis_keywords if k in h), "")

    return {
        "bandh_today":        "YES" if bandh_today  else "NO",
        "bandh_detail":       bandh_detail,
        "ipo_fpo_active":     "NO",
        "ipo_fpo_detail":     "",
        "crisis_detected":    "YES" if crisis_today else "NO",
        "crisis_detail":      crisis_detail,
        "gulf_signal":        "",
        "gulf_detail":        "",
        "remittance_signal":  "",
        "remittance_detail":  "",
        "overall_sentiment":  "NEUTRAL",
        "key_event":          "",
        "headlines_politics": "",
        "headlines_economy":  "",
        "headlines_stock":    "",
    }


def _empty_result() -> dict:
    return {
        "bandh_today":        "NO",
        "bandh_detail":       "",
        "ipo_fpo_active":     "NO",
        "ipo_fpo_detail":     "",
        "crisis_detected":    "NO",
        "crisis_detail":      "",
        "gulf_signal":        "",
        "gulf_detail":        "",
        "remittance_signal":  "",
        "remittance_detail":  "",
        "overall_sentiment":  "NEUTRAL",
        "key_event":          "No new headlines",
        "headlines_politics": "",
        "headlines_economy":  "",
        "headlines_stock":    "",
    }


def _scrape_and_analyze(force_keywords: bool = False) -> dict:
    """
    Full news pipeline:
      1. Scrape all sources → DataFrame (deduped against 48h cache)
      2. Deepseek analysis (or keyword fallback)
      3. Save updated hash cache to settings
      4. Auto-update GULF_STABILITY / INDIA_NEPAL_RELATIONS / NRB in Neon
      5. Flag bandh to calendar_guard if detected
    """
    new_df, updated_cache = _build_headlines_df()

    if new_df.empty:
        log.info("No new headlines after dedup — skipping AI call, returning neutral")
        # Still save cache (may have pruned expired entries)
        _save_seen_hashes(updated_cache)
        return _empty_result()

    result = {}
    if not force_keywords:
        result = ask_deepseek_text(
            prompt  = _build_gemini_prompt(new_df),
            context = "nepal_pulse",
        )
        if result is None:
            log.warning("Deepseek failed — keyword fallback")
            result = _keyword_detect(new_df)
        else:
            log.info("Deepseek analysis used on %d new headlines", len(new_df))
    else:
        log.info("Keyword fallback forced")
        result = _keyword_detect(new_df)

    # Save cache AFTER successful analysis
    _save_seen_hashes(updated_cache)

    result["headlines_checked"] = len(new_df)

    # Auto-update settings
    gulf_ai = result.get("gulf_signal", "").upper().strip()
    if gulf_ai in ("STABLE", "TENSE", "CRISIS"):
        update_setting("GULF_STABILITY", gulf_ai, set_by="nepal_pulse")
        log.info("Auto-updated GULF_STABILITY → %s", gulf_ai)

    india_ai = result.get("india_nepal_relations", "").upper().strip()
    if india_ai in ("STABLE", "TENSE", "HOSTILE"):
        update_setting("INDIA_NEPAL_RELATIONS", india_ai, set_by="nepal_pulse")

    nrb_ai = result.get("nrb_rate_decision", "").upper().strip()
    if nrb_ai in ("CUT", "RAISED", "UNCHANGED"):
        update_setting("NRB_RATE_DECISION", nrb_ai, set_by="nepal_pulse")

    # Flag bandh to calendar_guard
    if result.get("bandh_today") == "YES":
        flag_adhoc_closure(
            closure_date=today_nst(),
            reason=f"Bandh: {result.get('bandh_detail','')[:60]}",
            source="nepal_pulse",
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — READ NRB MACRO DATA
# ══════════════════════════════════════════════════════════════════════════════

def _read_nrb_macro() -> dict:
    """
    Read latest NRB macro fields from nrb_monthly table.
    Entered manually monthly.
    """
    try:
        macro = get_macro_data()
        if not macro:
            log.warning("No NRB macro data found")
            return {}
        log.info("NRB macro loaded: policy_rate=%s", macro.get("policy_rate", "?"))
        return macro
    except Exception as e:
        log.error("_read_nrb_macro failed: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORE COMPUTATION (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def _is_budget_season(now: datetime) -> str:
    """Nepal budget is typically announced in May/June."""
    return "YES" if now.month in (5, 6) else "NO"


def _compute_nepal_score(manual: dict, scraped: dict, macro: dict) -> tuple[int, str, str]:
    """
    Compute nepal_score (-5 to +5), status label, and key event string.
    Purely additive scoring — no single factor dominates.
    """
    score = 0
    events = []

    # Bandh — hard negative
    if scraped.get("bandh_today") == "YES":
        score -= 2
        events.append(f"BANDH: {scraped.get('bandh_detail','')[:50]}")

    # Crisis
    if scraped.get("crisis_detected") == "YES":
        score -= 2
        events.append(f"CRISIS: {scraped.get('crisis_detail','')[:50]}")

    # IPO drain (liquidity leaves market)
    if scraped.get("ipo_fpo_active") == "YES":
        score -= 1
        events.append(f"IPO/FPO: {scraped.get('ipo_fpo_detail','')[:40]}")

    # Gulf stability
    gulf = scraped.get("gulf_signal", manual.get("gulf_stability", "STABLE")).upper()
    if gulf == "CRISIS":
        score -= 1
        events.append("Gulf: CRISIS")
    elif gulf == "TENSE":
        score -= 0  # neutral — watch only
        events.append("Gulf: TENSE")

    # India-Nepal relations
    india = manual.get("india_nepal_relations", "STABLE").upper()
    if india == "HOSTILE":
        score -= 1
        events.append("India-Nepal: HOSTILE")
    elif india == "TENSE":
        score -= 0

    # NRB rate decision
    nrb = scraped.get("nrb_rate_decision", manual.get("nrb_rate_decision", "UNCHANGED")).upper()
    if nrb == "CUT":
        score += 1
        events.append("NRB: rate CUT")
    elif nrb == "RAISED":
        score -= 1
        events.append("NRB: rate RAISED")

    # Sentiment bonus
    sentiment = scraped.get("overall_sentiment", "NEUTRAL").upper()
    if sentiment == "POSITIVE":
        score += 1
    elif sentiment == "NEGATIVE":
        score -= 1

    # Clamp to range
    score = max(-5, min(5, score))

    if score >= 3:
        status = "BULLISH"
    elif score >= 1:
        status = "POSITIVE"
    elif score >= -1:
        status = "NEUTRAL"
    elif score >= -3:
        status = "NEGATIVE"
    else:
        status = "BEARISH"

    key_event = scraped.get("key_event", "") or (" | ".join(events[:2]) if events else "No significant events")

    return score, status, key_event


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(force_keywords: bool = False) -> bool:
    """
    Main entry point. Called by morning_workflow, main.py (30-min throttle),
    and summary_workflow (9 PM fresh run).

    Flow:
      1. Read manual settings from Neon
      2. Scrape news + Deepseek (deduplicated — only new headlines sent to AI)
      3. Read NRB macro from Neon
      4. Compute nepal_score
      5. Write snapshot to nepal_pulse table

    Returns True on success, False on failure.
    """
    nst_now = datetime.now(tz=NST)
    log.info("=" * 60)
    log.info("NEPAL PULSE starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    manual  = _read_manual_settings()
    scraped = _scrape_and_analyze(force_keywords=force_keywords)
    macro   = _read_nrb_macro()

    nepal_score, nepal_status, key_event = _compute_nepal_score(
        manual, scraped, macro
    )

    pulse = {
        "date": nst_now.strftime("%Y-%m-%d"),
        "time": nst_now.strftime("%H:%M"),

        # NRB / monetary
        "policy_rate":        macro.get("policy_rate", ""),
        "policy_rate_change": macro.get("nrb_rate_decision", ""),
        "ccd_ratio":          macro.get("ccd_ratio", ""),
        "crr":                macro.get("crr", ""),
        "slr":                macro.get("slr", ""),
        "nrb_event":          macro.get("nrb_event", ""),

        # SEBON / market structure
        "sebon_event":     get_setting("SEBON_EVENT",     default=""),
        "circuit_breaker": get_setting("CIRCUIT_BREAKER", default=""),
        "ipo_fpo_active":  scraped.get("ipo_fpo_active", "NO"),
        "ipo_fpo_detail":  scraped.get("ipo_fpo_detail", ""),

        # Government / political
        "govt_stability":  get_setting("GOVT_STABILITY", default="STABLE"),
        "political_event": scraped.get("crisis_detail", ""),
        "budget_season":   _is_budget_season(nst_now),
        "budget_event":    get_setting("BUDGET_EVENT", default=""),

        # Economic indicators
        "remittance_yoy_pct":   macro.get("remittance_yoy_pct", ""),
        "inflation_pct":        macro.get("inflation_pct", ""),
        "forex_reserve_months": macro.get("forex_reserve_months", ""),
        "trade_deficit_npr":    macro.get("trade_deficit_npr", ""),

        # Domestic disruption
        "bandh_today":       scraped.get("bandh_today", "NO"),
        "bandh_detail":      scraped.get("bandh_detail", ""),
        "load_shedding_hrs": get_setting("LOAD_SHEDDING_HRS", default="0"),

        # Composite
        "nepal_score":  str(nepal_score),
        "nepal_status": nepal_status,
        "key_event":    key_event,
        "source":       "nepal_pulse.py",
        "timestamp":    nst_now.strftime("%Y-%m-%d %H:%M:%S"),

        # Headlines — written by Deepseek, empty string on keyword fallback
        "headlines_politics": scraped.get("headlines_politics", ""),
        "headlines_economy":  scraped.get("headlines_economy",  ""),
        "headlines_stock":    scraped.get("headlines_stock",    ""),
    }

    success = write_nepal_pulse(pulse)
    if success:
        log.info(
            "Nepal pulse written — score=%s status=%s new_headlines=%s",
            nepal_score, nepal_status, scraped.get("headlines_checked", 0),
        )
    else:
        log.error("Failed to write nepal_pulse row")

    return success


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nepal Pulse — news scraper + scorer")
    parser.add_argument("--keywords", action="store_true",
                        help="Force keyword fallback (no AI)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the seen headline hash cache before running")
    args = parser.parse_args()

    if args.clear_cache:
        update_setting(HEADLINE_CACHE_KEY, "{}", set_by="cli")
        print("Headline cache cleared.")

    success = run(force_keywords=args.keywords)
    sys.exit(0 if success else 1)