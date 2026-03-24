"""
nepal_pulse.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Nepal-specific domestic market context score.
          Runs every 30 minutes during market hours via GitHub Actions.

Score meaning:
  +5  Ideal conditions  — stable politics, low rates, strong BOP
   0  Neutral           — mixed signals
  -5  Crisis conditions — bandh + political chaos + high FD rates

How it fits the system:
  geo_sentiment.py  → geo_score   (global: Crude, VIX, Nifty, DXY, Gold)
  nepal_pulse.py    → nepal_score (domestic: politics, rates, bandh)
  combined_geo = geo_score + nepal_score  (range: -10 to +10)

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

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from sheets import (
    get_setting,
    update_setting,
    write_nepal_pulse,
    get_latest_pulse,
    get_macro_data,
)
from calendar_guard import flag_adhoc_closure, today_nst

load_dotenv()

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

NST = timezone(timedelta(hours=5, minutes=45))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

NEWS_SOURCES = [
    "https://onlinekhabar.com/feed",
    "https://www.sharesansar.com/category/latest",
    "https://www.sharesansar.com/announcement",
    "https://ekantipur.com/business",
    "https://ekantipur.com/politics",
    "https://eng.merolagani.com/",
]

NEWS_TIMEOUT = 10

# ── Nepal location anchors ────────────────────────────────────────────────────
NEPAL_ANCHORS = [
    "nepal", "नेपाल", "kathmandu", "काठमाडौं",
    "pokhara", "biratnagar", "birgunj", "lalitpur",
    "nepse", "नेप्से",
]

# ── Keywords ──────────────────────────────────────────────────────────────────
BANDH_KEYWORDS = [
    "बन्द", "चक्काजाम", "आम हड्ताल",
    "nepal bandh", "nepal strike", "chakka jam",
    "nepal shutdown", "nepal transport strike",
    "nepse closed", "nepse closure",
]

IPO_KEYWORDS = [
    "ipo open", "fpo open", "ipo application", "fpo application",
    "share application", "public issue open", "आईपीओ",
    "right share open", "debenture open", "issues ipo",
    "ipo for general public", "public issue opens", "nepse ipo",
]

CRISIS_KEYWORDS = [
    "राजीनामा", "संसद विघटन", "अविश्वास प्रस्ताव", "सरकार ढल्यो",
    "nepal prime minister resign", "nepal government collapse",
    "nepal parliament dissolve", "nepal political crisis",
    "nepal no confidence", "pm oli resign", "pm dahal resign",
    "pm prachanda resign",
]

# ══════════════════════════════════════════════════════════════════════════════
# SCORE MAP
# Evidence-based weights — see research paper synthesis in session notes.
#
# REMOVED from previous version:
#   - china_nepal     (no paper evidence for weight)
#   - remittance      (ρ=+0.277 too weak, directionally captured by gulf)
#   - bank_rate       (noise in all papers — S5 explicitly says no impact)
#   - rbi             (renamed to nrb_rate_decision — NRB matters, not RBI)
#
# ADDED:
#   - fd_rate_level   (S7 deposit rate r=-0.650, strongest non-lending signal)
#   - lending_rate_level (S7 lending rate r=-0.669, strongest overall)
# ══════════════════════════════════════════════════════════════════════════════

SCORE_MAP = {

    # NRB RATE DECISION — S5, S7 confirm direction matters
    # Lending rate r=-0.669, T-bill r=-0.423 (Neupane 2018)
    "nrb_rate_decision": {
        "CUT":       +2,
        "UNCHANGED":  0,
        "RAISED":    -2,
    },

    # CPI INFLATION — S4, S6 confirm negative (Fama proxy hypothesis)
    # Spearman ρ=-0.30 from our macro_correlation.py analysis
    # Not wired into scoring yet — needs macro_data entry: "CPI_Inflation"
    "cpi_level": {
        "BELOW_4":   +1,
        "4_TO_7":     0,
        "ABOVE_7":   -2,
        "ABOVE_10":  -3,
    },

    # BOP OVERALL BALANCE — strongest macro signal (ρ=+0.495, p=0.0005)
    # Not wired into scoring yet — needs macro_data entry: "BOP_Trend"
    "bop_trend": {
        "SURPLUS_GROWING":   +2,
        "SURPLUS_STABLE":    +1,
        "SURPLUS_SHRINKING":  0,
        "DEFICIT":           -2,
    },

    # FX RESERVE — Spearman ρ=+0.325 concurrent signal
    # Not wired into scoring yet — needs macro_data entry: "FX_Reserve_Months"
    "fx_reserve_months": {
        "ABOVE_15":  +2,
        "ABOVE_12":  +1,
        "8_TO_12":    0,
        "BELOW_8":   -2,
    },

    # INDIA-NEPAL RELATIONS — S4 political events paper confirms
    # non-economic factors significantly affect NEPSE
    "india_nepal": {
        "STABLE":  +1,
        "TENSE":   -1,
        "HOSTILE": -2,
    },

    # GULF STABILITY — remittance channel (ρ=+0.277 marginal, directional)
    "gulf": {
        "STABLE":  +1,
        "TENSE":   -1,
        "CRISIS":  -2,
    },

    # FD RATE LEVEL — S7 deposit rate r=-0.650 significant negative
    # FD > 9-10% = retail money leaves NEPSE (confirmed across macro papers)
    # Fed by interest_scraper.py → FD_RATE_PCT setting
    "fd_rate_level": {
        "BELOW_7":   +2,
        "7_TO_8":    +1,
        "8_TO_9":     0,
        "9_TO_10":   -1,
        "ABOVE_10":  -2,
    },

    # LENDING RATE LEVEL — S7 lending rate r=-0.669, strongest predictor
    # Fed by macro_data table → "Lending_Rate" indicator (NRB monthly paste)
    "lending_rate_level": {
        "BELOW_9":   +1,
        "9_TO_11":    0,
        "ABOVE_11":  -2,
        "ABOVE_13":  -3,
    },

    # BANDH — S4 event study confirms political/social disruption matters
    "bandh": {
        "YES": -3,
        "NO":   0,
    },

    # IPO DRAIN — liquidity withdrawal from market
    "ipo_drain": {
        "YES": -1,
        "NO":   0,
    },

    # POLITICAL CRISIS — S4 confirms non-economic factors significant
    "crisis": {
        "YES": -2,
        "NO":   0,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — READ MANUAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

def _read_manual_settings() -> dict:
    """
    Read manual geo fields from settings table.
    GULF_STABILITY and REMITTANCE_RISK are auto-updated by Gemini Flash.
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

        # KEY CHANGE: was RBI_LAST_DECISION — now NRB_RATE_DECISION
        # NRB is Nepal's central bank; RBI is India's
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


def _build_headlines_df() -> pd.DataFrame:
    """Scrape all sources and return combined DataFrame (source, headline)."""
    all_rows: list[dict] = []
    for url in NEWS_SOURCES:
        all_rows.extend(_fetch_headlines(url))

    log.info("Total headlines collected: %d", len(all_rows))

    if not all_rows:
        return pd.DataFrame(columns=["source", "headline"])

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GEMINI FLASH ANALYSIS
# google.genai import is INSIDE this function — no top-level crash if
# package not installed or API key missing. Keyword fallback handles it.
# ══════════════════════════════════════════════════════════════════════════════

def _ask_gemini(df: pd.DataFrame) -> dict:
    """
    Send headlines DataFrame to Gemini Flash for context-aware analysis.
    Uses google.genai (new SDK — google.generativeai is deprecated).
    Raises exception on failure — caller handles keyword fallback.
    """
    # Import inside function — prevents crash if package missing
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env — using keyword fallback")

    client = genai.Client(api_key=GEMINI_API_KEY)
    data_str = df.to_string(index=False)

    prompt = f"""You are a Nepal financial market analyst.
Analyze these Nepal news headlines scraped today.
Detect signals relevant to Nepal stock market (NEPSE).

1. Is there a bandh, strike, chakka jam, or transport/business shutdown IN NEPAL today?
   Only count events physically happening inside Nepal — ignore global news.
2. Is any IPO, FPO, or right share application open on NEPSE today?
3. Is there a political crisis, PM resignation, or government instability IN NEPAL?
4. What is the Gulf/Middle East stability? (affects Nepal remittance workers abroad)
5. What is the remittance risk level based on Gulf/foreign employment news?
6. What is the overall Nepal market sentiment today?

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
  "key_event": "single most important Nepal market event in one sentence, or empty string"
}}"""

    log.info("Sending %d headlines to Gemini Flash...", len(df))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    raw    = response.text.strip()
    result = json.loads(raw)

    log.info(
        "Gemini Flash: bandh=%s | IPO=%s | crisis=%s | gulf=%s",
        result.get("bandh_today"),
        result.get("ipo_fpo_active"),
        result.get("crisis_detected"),
        result.get("gulf_signal"),
    )

    return result


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
    Used when Gemini is unavailable or key not set.
    """
    log.info("Using keyword fallback for signal detection...")

    if df.empty:
        return _empty_result()

    rows = list(zip(
        df["headline"].tolist(),
        df["source"].tolist() if "source" in df.columns else [""] * len(df),
    ))

    # ── Bandh detection ───────────────────────────────────────────────────────
    bandh_today  = "NO"
    bandh_detail = ""

    for h, source in rows:
        for kw in BANDH_KEYWORDS:
            if kw not in h:
                continue
            is_nepali = any(ord(c) > 127 for c in kw)
            if is_nepali:
                bandh_today  = "YES"
                bandh_detail = h[:120]
                log.warning("BANDH (nepali keyword): %s", h[:120])
                break
            if _is_nepal_source(source) or _has_nepal_anchor(h):
                bandh_today  = "YES"
                bandh_detail = h[:120]
                log.warning("BANDH (english keyword, nepal confirmed): %s", h[:120])
                break
        if bandh_today == "YES":
            break

    # ── IPO detection ─────────────────────────────────────────────────────────
    ipo_active = "NO"
    ipo_detail = ""

    for h, source in rows:
        for kw in IPO_KEYWORDS:
            if kw in h:
                ipo_active = "YES"
                ipo_detail = h[:120]
                log.info("IPO (keyword): %s", h[:120])
                break
        if ipo_active == "YES":
            break

    # ── Crisis detection ──────────────────────────────────────────────────────
    crisis_detected = "NO"
    crisis_detail   = ""

    for h, source in rows:
        for kw in CRISIS_KEYWORDS:
            if kw not in h:
                continue
            is_nepali = any(ord(c) > 127 for c in kw)
            if is_nepali or _is_nepal_source(source) or _has_nepal_anchor(h):
                crisis_detected = "YES"
                crisis_detail   = h[:120]
                log.warning("CRISIS (keyword): %s", h[:120])
                break
        if crisis_detected == "YES":
            break

    return {
        "bandh_today":       bandh_today,
        "bandh_detail":      bandh_detail,
        "ipo_fpo_active":    ipo_active,
        "ipo_fpo_detail":    ipo_detail,
        "crisis_detected":   crisis_detected,
        "crisis_detail":     crisis_detail,
        "gulf_signal":       "",
        "gulf_detail":       "",
        "remittance_signal": "",
        "remittance_detail": "",
        "overall_sentiment": "NEUTRAL",
        "key_event":         "",
    }


def _empty_result() -> dict:
    return {
        "bandh_today":       "NO",
        "bandh_detail":      "",
        "ipo_fpo_active":    "NO",
        "ipo_fpo_detail":    "",
        "crisis_detected":   "NO",
        "crisis_detail":     "",
        "gulf_signal":       "",
        "gulf_detail":       "",
        "remittance_signal": "",
        "remittance_detail": "",
        "overall_sentiment": "NEUTRAL",
        "key_event":         "No headlines available",
    }


def _scrape_and_analyze(force_keywords: bool = False) -> dict:
    """
    Full news pipeline:
      1. Scrape all sources → DataFrame
      2. Gemini Flash analysis (or keyword fallback)
      3. Auto-update GULF_STABILITY in Neon settings
      4. Flag bandh to calendar_guard if detected
    """
    df = _build_headlines_df()

    if df.empty:
        log.warning("No headlines fetched — returning neutral defaults")
        return _empty_result()

    result = {}
    if not force_keywords:
        try:
            result = _ask_gemini(df)
            log.info("Gemini Flash analysis used")
        except Exception as exc:
            log.warning("Gemini Flash failed (%s) — keyword fallback", exc)
            result = _keyword_detect(df)
    else:
        log.info("Keyword fallback forced")
        result = _keyword_detect(df)

    result["headlines_checked"] = len(df)

    # Auto-update Gulf in Neon
    gulf_ai = result.get("gulf_signal", "").upper().strip()
    if gulf_ai in ("STABLE", "TENSE", "CRISIS"):
        update_setting("GULF_STABILITY", gulf_ai, set_by="gemini_flash")
        log.info("Auto-updated GULF_STABILITY → %s", gulf_ai)

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
    Read latest NRB macro fields from macro_data table.
    You paste this monthly via NotebookLM.
    """
    log.info("Reading NRB macro data from Neon...")

    try:
        macro = get_macro_data()
    except Exception as exc:
        log.warning("Failed to read macro_data: %s — using defaults", exc)
        return {}

    fields = {
        "policy_rate":          macro.get("Policy_Rate", ""),
        "nrb_rate_decision":    macro.get("NRB_Rate_Decision", "UNCHANGED").upper(),
        "ccd_ratio":            macro.get("CCD_Ratio", ""),
        "crr":                  macro.get("CRR", ""),
        "slr":                  macro.get("SLR", ""),
        # FIX: was "ict" — correct key is "inflation_pct"
        "inflation_pct":        macro.get("Inflation_Pct", ""),
        "remittance_yoy_pct":   macro.get("Remittance_YoY_Pct", ""),
        "forex_reserve_months": macro.get("Forex_Reserve_Months", ""),
        "trade_deficit_npr":    macro.get("Trade_Deficit_NPR", ""),
        "nrb_event":            macro.get("NRB_Event", ""),
        "lending_rate":         macro.get("Lending_Rate", ""),
    }

    # FIX: was referencing fields["inflation_pct"] with typo "fornflation_pex_reserve_months"
    log.info(
        "NRB macro: Policy Rate=%s | NRB Decision=%s | Inflation=%s%% | Forex=%s months | Lending=%s%%",
        fields["policy_rate"],
        fields["nrb_rate_decision"],
        fields["inflation_pct"],
        fields["forex_reserve_months"],
        fields["lending_rate"],
    )

    return fields


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COMPUTE NEPAL SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_nepal_score(
    manual:  dict,
    scraped: dict,
    macro:   dict,
) -> tuple[int, str, str]:
    """
    Compute nepal_score from all inputs.
    Range: -5 to +5 (clamped).

    Evidence-based score breakdown:
      NRB rate decision   : -2 to +2  (S5, S7 confirmed — lending r=-0.669)
      India-Nepal         : -2 to +1  (S4 political events paper)
      Gulf                : -2 to +1  (remittance channel ρ=+0.277)
      FD rate level       : -2 to +2  (S7 deposit rate r=-0.650)
      Lending rate level  : -3 to +1  (S7 lending rate r=-0.669, strongest)
      Bandh today         : -3 to  0  (S4 event study)
      IPO/FPO drain       : -1 to  0  (liquidity withdrawal)
      Crisis detected     : -2 to  0  (S4 non-economic factors)

    REMOVED vs previous version:
      china_nepal    — no paper evidence for any weight
      remittance     — ρ=+0.277 too weak, captured by gulf directionally
      rbi_decision   — renamed to nrb_rate_decision (NRB matters, not RBI)
    """
    score   = 0
    factors = []

    # ── NRB rate decision ─────────────────────────────────────────────────────
    # Read from manual settings (you update when NRB changes rate)
    # Falls back to macro_data table if available
    nrb_decision = (
        macro.get("nrb_rate_decision", "").upper().strip()
        or manual.get("nrb_rate_decision", "UNCHANGED").upper().strip()
    )
    if nrb_decision not in ("CUT", "UNCHANGED", "RAISED"):
        nrb_decision = "UNCHANGED"

    score += SCORE_MAP["nrb_rate_decision"].get(nrb_decision, 0)
    factors.append(f"NRB:{nrb_decision}")

    # ── India-Nepal relations ─────────────────────────────────────────────────
    india_rel = manual.get("india_nepal_relations", "STABLE").upper()
    if india_rel not in ("STABLE", "TENSE", "HOSTILE"):
        india_rel = "STABLE"

    score += SCORE_MAP["india_nepal"].get(india_rel, 0)
    factors.append(f"India-Nepal:{india_rel}")

    # ── Gulf stability ────────────────────────────────────────────────────────
    # Prefer Gemini Flash result, fall back to manual
    gulf = (
        scraped.get("gulf_signal", "").upper().strip()
        or manual.get("gulf_stability", "STABLE").upper().strip()
    )
    if gulf not in ("STABLE", "TENSE", "CRISIS"):
        gulf = "STABLE"

    score += SCORE_MAP["gulf"].get(gulf, 0)
    factors.append(f"Gulf:{gulf}")

    # ── FD rate level ─────────────────────────────────────────────────────────
    # Reads from FD_RATE_PCT setting — fed by interest_scraper.py monthly
    # S7: deposit rate r=-0.650 significant negative for banking stocks
    try:
        fd_rate = float(get_setting("FD_RATE_PCT", "8.5"))
        if fd_rate < 7.0:
            fd_bucket = "BELOW_7"
        elif fd_rate < 8.0:
            fd_bucket = "7_TO_8"
        elif fd_rate < 9.0:
            fd_bucket = "8_TO_9"
        elif fd_rate < 10.0:
            fd_bucket = "9_TO_10"
        else:
            fd_bucket = "ABOVE_10"
        score += SCORE_MAP["fd_rate_level"].get(fd_bucket, 0)
        factors.append(f"FD:{fd_rate:.1f}%[{fd_bucket}]")
    except Exception as exc:
        log.debug("FD rate read failed: %s — skipping", exc)

    # ── Lending rate level ────────────────────────────────────────────────────
    # Reads from macro_data table — you paste from NRB monthly
    # S7: lending rate r=-0.669, strongest predictor of banking stock prices
    try:
        lending_rate_str = macro.get("lending_rate", "")
        if lending_rate_str:
            lr = float(str(lending_rate_str).replace("%", "").strip())
            if lr < 9.0:
                lr_bucket = "BELOW_9"
            elif lr < 11.0:
                lr_bucket = "9_TO_11"
            elif lr < 13.0:
                lr_bucket = "ABOVE_11"
            else:
                lr_bucket = "ABOVE_13"
            score += SCORE_MAP["lending_rate_level"].get(lr_bucket, 0)
            factors.append(f"Lending:{lr:.1f}%[{lr_bucket}]")
    except Exception as exc:
        log.debug("Lending rate read failed: %s — skipping", exc)

    # ── AI / scraped signals ──────────────────────────────────────────────────
    bandh  = scraped.get("bandh_today",     "NO").upper()
    ipo    = scraped.get("ipo_fpo_active",  "NO").upper()
    crisis = scraped.get("crisis_detected", "NO").upper()

    score += SCORE_MAP["bandh"].get(bandh, 0)
    score += SCORE_MAP["ipo_drain"].get(ipo, 0)
    score += SCORE_MAP["crisis"].get(crisis, 0)

    if bandh  == "YES": factors.append("BANDH_TODAY")
    if ipo    == "YES": factors.append("IPO_DRAIN")
    if crisis == "YES": factors.append("CRISIS_DETECTED")

    # ── Clamp to -5 .. +5 ────────────────────────────────────────────────────
    score = max(-5, min(5, score))

    # ── Status label ──────────────────────────────────────────────────────────
    if bandh == "YES" or crisis == "YES":
        status = "CRISIS"
    elif score >= 3:
        status = "BULLISH"
    elif score >= 1:
        status = "POSITIVE"
    elif score >= -1:
        status = "NEUTRAL"
    elif score >= -3:
        status = "BEARISH"
    else:
        status = "CRISIS"

    # ── Key event ─────────────────────────────────────────────────────────────
    gemini_key = scraped.get("key_event", "").strip()

    if bandh == "YES":
        key_event = f"BANDH: {scraped.get('bandh_detail', 'Nepal bandh detected')[:80]}"
    elif crisis == "YES":
        key_event = f"CRISIS: {scraped.get('crisis_detail', 'Political crisis')[:80]}"
    elif ipo == "YES":
        key_event = f"IPO_DRAIN: {scraped.get('ipo_fpo_detail', 'IPO/FPO open today')[:80]}"
    elif gemini_key:
        key_event = gemini_key[:120]
    elif gulf != "STABLE":
        key_event = f"Gulf {gulf}: {scraped.get('gulf_detail', '')[:60]}"
    elif india_rel != "STABLE":
        key_event = f"India-Nepal relations: {india_rel}"
    elif nrb_decision == "CUT":
        key_event = "NRB rate cut — positive for NEPSE"
    elif nrb_decision == "RAISED":
        key_event = "NRB rate raised — negative for NEPSE"
    else:
        key_event = "No major event — routine conditions"

    log.info("Nepal score: %+d | Status: %s | Key event: %s", score, status, key_event)
    log.info("Score factors: %s", " | ".join(factors))

    return score, status, key_event


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(force_keywords: bool = False) -> bool:
    """
    Main entry point. Called by GitHub Actions every 30 minutes.

    Flow:
      1. Read manual settings from Neon
      2. Scrape news + Gemini Flash (or keyword fallback)
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
        "sebon_event":     get_setting("SEBON_EVENT", default=""),
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
    }

    success = write_nepal_pulse(pulse)

    if success:
        log.info("✅ Nepal pulse written successfully")
        log.info("   Score:     %+d", nepal_score)
        log.info("   Status:    %s",  nepal_status)
        log.info("   Event:     %s",  key_event)
        log.info("   Bandh:     %s",  scraped.get("bandh_today", "NO"))
        log.info("   IPO:       %s",  scraped.get("ipo_fpo_active", "NO"))
        log.info("   Gulf:      %s",  scraped.get("gulf_signal", "manual"))
        log.info("   Headlines: %d",  scraped.get("headlines_checked", 0))
    else:
        log.error("❌ Failed to write nepal pulse to Neon")

    return success


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — called by other modules
# ══════════════════════════════════════════════════════════════════════════════

def _is_budget_season(dt: datetime) -> str:
    """YES if May or June — Nepal budget season."""
    return "YES" if dt.month in (5, 6) else "NO"


def get_latest_nepal_score() -> int:
    """
    Returns latest nepal_score as integer.
    Called by filter_engine.py and claude_analyst.py.
    Returns 0 (neutral) if no data available.
    """
    try:
        latest = get_latest_pulse()
        if latest and latest.get("nepal_score"):
            return int(latest["nepal_score"])
    except Exception as exc:
        log.warning("get_latest_nepal_score failed: %s", exc)
    return 0


def get_combined_geo_score() -> int:
    """
    Returns geo_score + nepal_score combined (-10 to +10).
    Called by filter_engine.py and gemini_filter.py.
    """
    from sheets import get_latest_geo

    geo_score   = 0
    nepal_score = get_latest_nepal_score()

    try:
        latest_geo = get_latest_geo()
        if latest_geo and latest_geo.get("geo_score"):
            geo_score = int(latest_geo["geo_score"])
    except Exception as exc:
        log.warning("Could not read geo_score: %s", exc)

    combined = geo_score + nepal_score
    log.info(
        "Combined geo: %+d (geo=%+d + nepal=%+d)",
        combined, geo_score, nepal_score,
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# CLI
#   python -m modules.nepal_pulse           → full run with Gemini Flash
#   python -m modules.nepal_pulse score     → print latest scores only
#   python -m modules.nepal_pulse keywords  → full run, force keyword fallback
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [NEPAL_PULSE] %(levelname)s: %(message)s",
    )

    arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if arg == "score":
        nepal    = get_latest_nepal_score()
        combined = get_combined_geo_score()
        print(f"\n{'='*50}")
        print(f"  Nepal Score:    {nepal:+d}")
        print(f"  Combined Geo:   {combined:+d}")
        print(f"{'='*50}\n")
        sys.exit(0)

    force_kw = (arg == "keywords")
    if force_kw:
        log.info("Keyword fallback mode forced")

    success = run(force_keywords=force_kw)

    if success:
        latest = get_latest_pulse()
        if latest:
            score_val = int(latest.get("nepal_score", 0))
            print(f"\n{'='*50}")
            print(f"  NEPAL PULSE SUMMARY")
            print(f"{'='*50}")
            print(f"  Date:         {latest.get('date')} {latest.get('time')}")
            print(f"  Nepal Score:  {score_val:>+3}")
            print(f"  Status:       {latest.get('nepal_status')}")
            print(f"  Key Event:    {latest.get('key_event')}")
            print(f"  Bandh Today:  {latest.get('bandh_today')}")
            print(f"  IPO Drain:    {latest.get('ipo_fpo_active')}")
            print(f"  Gulf:         {latest.get('gulf_signal', 'see settings')}")
            print(f"{'='*50}\n")
        sys.exit(0)
    else:
        sys.exit(1)