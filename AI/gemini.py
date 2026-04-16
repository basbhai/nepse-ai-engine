"""
AI/gemini.py
============
Central Gemini Flash gateway for the NEPSE AI Engine.

All modules must import from here — no inline Gemini calls anywhere else.

    from AI.gemini import ask_gemini_json, ask_gemini_text

Retry policy: 5 attempts, random jitter 4-15s between attempts.
Retries on: 503, 429, timeout, connection error.
Telegram alert sent to admin when ALL retries are exhausted.
"""

import json
import logging
import os
import random
import time
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

MAX_RETRIES  = 5
JITTER_MIN   = 4    # seconds
JITTER_MAX   = 15   # seconds

# Errors that warrant a retry (transient)
_RETRYABLE = (
    "503",
    "429",
    "UNAVAILABLE",
    "quota",
    "rate",
    "timeout",
    "timed out",
    "connection",
    "ServiceUnavailable",
    "ResourceExhausted",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _RETRYABLE)


def _alert_admin(context: str, last_error: str) -> None:
    """Send Telegram alert to admin when all retries are exhausted."""
    try:
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        text = (
            f"🔴 *Gemini Flash — ALL RETRIES EXHAUSTED*\n"
            f"Context: `{context}`\n"
            f"Last error: `{last_error[:200]}`\n"
            f"Model: `{GEMINI_MODEL}`"
        )
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.error("Failed to send Telegram alert: %s", e)


def _raw_gemini_call(
    prompt: str,
    system: Optional[str],
    response_mime_type: str,
    temperature: float,
) -> str:
    """
    Single raw call to Gemini. Returns response text.
    Raises on any error — caller handles retry.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    config_kwargs = dict(
        temperature=temperature,
    )
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if system:
        config_kwargs["system_instruction"] = system

    response = client.models.generate_content(
        model   = GEMINI_MODEL,
        contents= prompt,
        config  = types.GenerateContentConfig(**config_kwargs),
    )
    return response.text.strip()


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences Gemini sometimes adds despite mime type."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _gemini_with_retry(
    prompt: str,
    system: Optional[str],
    response_mime_type: str,
    temperature: float,
    context: str,          # human-readable caller name for logs/alerts
) -> Optional[str]:
    """
    Call Gemini with retry logic. Returns raw text or None on total failure.
    """
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY not set in .env")
        return None

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] Gemini call attempt %d/%d", context, attempt, MAX_RETRIES)
            raw = _raw_gemini_call(prompt, system, response_mime_type, temperature)
            log.info("[%s] Gemini responded on attempt %d", context, attempt)
            return raw

        except Exception as exc:
            last_error = str(exc)
            if _is_retryable(exc):
                if attempt < MAX_RETRIES:
                    wait = random.uniform(JITTER_MIN, JITTER_MAX)
                    log.warning(
                        "[%s] Gemini transient error (attempt %d/%d): %s — retrying in %.1fs",
                        context, attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "[%s] Gemini failed after %d attempts: %s",
                        context, MAX_RETRIES, exc,
                    )
                    _alert_admin(context, last_error)
            else:
                # Non-retryable error — fail immediately
                log.error("[%s] Gemini non-retryable error: %s", context, exc)
                return None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_gemini_json(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    context: str = "gemini_json",
) -> Optional[dict]:
    """
    Send prompt to Gemini Flash. Returns parsed JSON dict or None on failure.

    Args:
        prompt:      Full user prompt.
        system:      Optional system instruction override.
        temperature: Sampling temperature (default 0.2 for structured output).
        context:     Caller name — shown in logs and Telegram alerts.

    Returns:
        Parsed dict, or None if all retries fail or JSON is invalid.

    Usage:
        from AI.gemini import ask_gemini_json
        result = ask_gemini_json(prompt, system="You are ...", context="gemini_filter")
    """
    raw = _gemini_with_retry(
        prompt             = prompt,
        system             = system,
        response_mime_type = "application/json",
        temperature        = temperature,
        context            = context,
    )
    if raw is None:
        return None

    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("[%s] Gemini returned invalid JSON: %s | raw: %s", context, exc, raw[:300])
        return None


def ask_gemini_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    context: str = "gemini_text",
) -> Optional[str]:
    """
    Send prompt to Gemini Flash. Returns raw text string or None on failure.

    Args:
        prompt:      Full user prompt.
        system:      Optional system instruction override.
        temperature: Sampling temperature (default 0.4 for free-form text).
        context:     Caller name — shown in logs and Telegram alerts.

    Returns:
        Text string, or None if all retries fail.

    Usage:
        from AI.gemini import ask_gemini_text
        summary = ask_gemini_text(prompt, context="daily_context_summarizer")
    """
    return _gemini_with_retry(
        prompt             = prompt,
        system             = system,
        response_mime_type = "",          # no mime type = free text
        temperature        = temperature,
        context            = context,
    )