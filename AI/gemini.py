"""
ai/gemini.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine
Purpose : Central AI interface using Gemini.
          Drop-in replacement for OpenRouter/Claude calls.
          Used by capital_allocator.py, briefing.py, and any future modules.

Usage:
  from ai.gemini import ask_ai

CREDENTIALS NEEDED IN .env:
  GEMINI_API_KEY=your_key
  GEMINI_MODEL=gemini-2.5-flash  (optional)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a Nepal stock market financial advisor and wealth manager. "
    "You give specific, actionable advice based on NEPSE market conditions. "
    "You always return valid JSON only — no markdown, no explanation, no code fences. "
    "You prioritize capital preservation over speculation. "
    "You understand Nepal macro context: remittance economy, NRB policy, seasonal patterns."
)


def ask_ai(prompt: str, system: str = None) -> Optional[dict]:
    """
    Send prompt to Gemini, return parsed JSON dict.
    Drop-in replacement for _ask_claude() in capital_allocator.py.

    Args:
        prompt: Full user prompt
        system: Optional system prompt override (uses default if None)

    Returns:
        Parsed dict from Gemini JSON response, or None on failure.
    """
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set in .env")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = system or SYSTEM_PROMPT

    log.info("Sending prompt to Gemini (%s)...", GEMINI_MODEL)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.3,   # lower = more consistent financial advice
            ),
        )

        raw = response.text.strip()

        # Strip code fences if Gemini adds them despite mime type
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        log.info("✅ Gemini response received")
        return result

    except json.JSONDecodeError as e:
        log.error("Gemini returned invalid JSON: %s", e)
        log.error("Raw response: %s", raw[:300] if 'raw' in locals() else "N/A")
        return None
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return None


def ask_ai_text(prompt: str, system: str = None) -> Optional[str]:
    """
    Same as ask_ai but returns raw text instead of parsed JSON.
    Used for free-form responses like briefing summaries.
    """
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set in .env")
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    system_instruction = system or SYSTEM_PROMPT

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
            ),
        )
        return response.text.strip()

    except Exception as e:
        log.error("Gemini text API error: %s", e)
        return None