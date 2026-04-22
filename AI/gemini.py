"""
AI/gemini.py
============
Central Gemini gateway for the NEPSE AI Engine.

Strategy:
  1. Try Google SDK with 3 rotating API keys (free quota)
     - Retry 5 times rotating keys: key1 → key2 → key3 → key1 → key2
     - Jitter 4-15s between retries
  2. If all 5 fail → fallback to OpenRouter paid Gemini
  3. Telegram alert to admin only if OpenRouter also fails

All modules import from here:
    from AI.gemini import ask_gemini_json, ask_gemini_text
Or via __init__.py:
    from AI import ask_gemini_json, ask_gemini_text
"""

import json
import logging
import os
import random
import time
from typing import Optional

log = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")

MAX_RETRIES = 5
JITTER_MIN  = 4
JITTER_MAX  = 15

# 3 rotating API keys — key1 used as fallback for GEMINI_API_KEY backwards compat
_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1") ,
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
]

# Retries rotate across keys: attempt 0→key0, 1→key1, 2→key2, 3→key0, 4→key1
def _get_key(attempt: int) -> str:
    """Rotate across available keys. Skip empty keys."""
    available = [k for k in _GEMINI_KEYS if k.strip()]
    if not available:
        return ""
    return available[attempt % len(available)]


# ---------------------------------------------------------------------------
# Retryable error detection
# ---------------------------------------------------------------------------
_RETRYABLE = (
    "503", "429", "UNAVAILABLE", "quota", "rate",
    "timeout", "timed out", "connection",
    "ServiceUnavailable", "ResourceExhausted",
    "overloaded", "try again",
)

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _RETRYABLE)


# ---------------------------------------------------------------------------
# Telegram admin alert
# ---------------------------------------------------------------------------
def _alert_admin(context: str, last_error: str) -> None:
    """Alert admin only when BOTH Gemini SDK and OpenRouter fallback fail."""
    try:
        import requests
        token   = os.getenv("TELEGRAM_ERROR_BOT", "") 
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        text = (
            f"🔴 *Gemini — ALL KEYS + OPENROUTER FAILED*\n"
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


# ---------------------------------------------------------------------------
# Google SDK raw call
# ---------------------------------------------------------------------------
def _sdk_call(
    prompt: str,
    system: Optional[str],
    response_mime_type: str,
    temperature: float,
    api_key: str,
    use_search: bool = False,
) -> str:
    """
    Single raw call via Google SDK.
    Raises on any error — caller handles retry.
    When use_search=True, adds Google Search grounding tool and
    skips response_mime_type (incompatible with search grounding).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config_kwargs = dict(temperature=temperature)
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    elif response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if system:
        config_kwargs["system_instruction"] = system

    response = client.models.generate_content(
        model   = GEMINI_MODEL,
        contents= prompt,
        config  = types.GenerateContentConfig(**config_kwargs),
    )
    if response.text is None:
        raise ValueError("Gemini returned None response.text — likely safety filter or empty response")
    return response.text.strip()


# ---------------------------------------------------------------------------
# OpenRouter paid fallback
# ---------------------------------------------------------------------------
def _openrouter_fallback(
    prompt: str,
    system: Optional[str],
    response_mime_type: str,
    temperature: float,
    context: str,
) -> Optional[str]:
    """
    Fallback to OpenRouter paid Gemini when all SDK keys fail.
    Uses same _call() from openrouter.py — retry logic already inside.
    """
    try:
        from AI.openrouter import _call
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        log.warning("[%s] All Gemini SDK keys failed — falling back to OpenRouter paid", context)

        return _call(
            model       = OPENROUTER_GEMINI_MODEL,
            messages    = messages,
            max_tokens  = 10000,
            temperature = temperature,
            context     = f"{context}_openrouter_fallback",
            use_search = True,  # OpenRouter fallback does NOT support search grounding
        )
    except Exception as e:
        log.error("[%s] OpenRouter fallback also failed: %s", context, e)
        return None


# ---------------------------------------------------------------------------
# Strip markdown fences
# ---------------------------------------------------------------------------
def _strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


# ---------------------------------------------------------------------------
# Core retry loop
# ---------------------------------------------------------------------------
def _gemini_with_retry(
    prompt: str,
    system: Optional[str],
    response_mime_type: str,
    temperature: float,
    context: str,
    use_search: bool = False,
) -> Optional[str]:
    """
    Try Google SDK 5 times rotating across 3 keys.
    On total failure → OpenRouter paid fallback.
    On OpenRouter failure → Telegram alert + return None.

    Key rotation:
        attempt 0 → GEMINI_API_KEY_1
        attempt 1 → GEMINI_API_KEY_2
        attempt 2 → GEMINI_API_KEY_3
        attempt 3 → GEMINI_API_KEY_1
        attempt 4 → GEMINI_API_KEY_2
    """
    available_keys = [k for k in _GEMINI_KEYS if k.strip()]
    if not available_keys:
        log.error("[%s] No Gemini API keys configured", context)
        return _openrouter_fallback(
            prompt, system, response_mime_type, temperature, context
        )

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        api_key  = _get_key(attempt - 1)
        key_num  = (attempt - 1) % len(available_keys) + 1

        try:
            log.info(
                "[%s] Gemini SDK attempt %d/%d (key_%d)",
                context, attempt, MAX_RETRIES, key_num,
            )
            raw = _sdk_call(prompt, system, response_mime_type, temperature, api_key, use_search)
            log.info("[%s] Gemini responded on attempt %d (key_%d)", context, attempt, key_num)
            return raw

        except Exception as exc:
            last_error = str(exc)
            if _is_retryable(exc):
                if attempt < MAX_RETRIES:
                    wait = random.uniform(JITTER_MIN, JITTER_MAX)
                    log.warning(
                        "[%s] Gemini transient error key_%d (attempt %d/%d): %s — retrying in %.1fs",
                        context, key_num, attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "[%s] All %d Gemini SDK attempts failed. Last: %s",
                        context, MAX_RETRIES, exc,
                    )
            else:
                # Non-retryable — skip remaining retries for this key
                # but still try next key
                log.warning(
                    "[%s] Gemini non-retryable error key_%d: %s — trying next key",
                    context, key_num, exc,
                )
                if attempt < MAX_RETRIES:
                    continue

    # All SDK attempts failed
    if use_search:
        # Google Search grounding is incompatible with OpenRouter — alert and bail
        log.error("[%s] All Gemini SDK attempts failed (use_search=True) — no OpenRouter fallback", context)
        _alert_admin(context, last_error)
        return None

    raw = _openrouter_fallback(
        prompt, system, response_mime_type, temperature, context
    )

    if raw is None:
        # Both SDK and OpenRouter failed — alert admin
        _alert_admin(context, last_error)

    return raw


# ---------------------------------------------------------------------------
# Public API — signatures identical to before, callers unchanged
# ---------------------------------------------------------------------------

def ask_gemini_json(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    context: str = "gemini_json",
    use_search: bool = False,
) -> Optional[dict]:
    """
    Send prompt to Gemini Flash. Returns parsed JSON dict or None.

    Internally:
      1. Tries Google SDK × 5 (rotating 3 keys)
      2. Falls back to OpenRouter paid Gemini (skipped when use_search=True)
      3. Alerts admin only if both fail

    Usage:
        from AI import ask_gemini_json
        result = ask_gemini_json(prompt, system="You are...", context="gemini_filter")
    """
    raw = _gemini_with_retry(
        prompt             = prompt,
        system             = system,
        response_mime_type = "" if use_search else "application/json",
        temperature        = temperature,
        context            = context,
        use_search         = use_search,
    )
    if raw is None:
        return None

    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error(
            "[%s] Gemini returned invalid JSON: %s | raw: %s",
            context, exc, raw[:300],
        )
        return None


def ask_gemini_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    context: str = "gemini_text",
    use_search: bool = False,
) -> Optional[str]:
    """
    Send prompt to Gemini Flash. Returns raw text or None.

    Internally:
      1. Tries Google SDK × 5 (rotating 3 keys)
      2. Falls back to OpenRouter paid Gemini (skipped when use_search=True)
      3. Alerts admin only if both fail

    Usage:
        from AI import ask_gemini_text
        result = ask_gemini_text(prompt, context="daily_summarizer")
    """
    return _gemini_with_retry(
        prompt             = prompt,
        system             = system,
        response_mime_type = "",   # no mime = free text
        temperature        = temperature,
        context            = context,
        use_search         = use_search,
    )