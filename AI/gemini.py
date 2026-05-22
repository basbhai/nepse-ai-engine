"""
AI/gemini.py
============
Central Gemini gateway for the NEPSE AI Engine.

Strategy:
  1. Try Google SDK with 3 rotating API keys (free quota)
     - Retry 5 times rotating keys: key1 → key2 → key3 → key1 → key2
     - Jitter 4-15s between retries
  2. If all 5 fail → fallback to Playwright DeepSeek (free, no API cost)
  3. Telegram alert to admin only if Playwright also fails

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
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")

MAX_RETRIES = 5
JITTER_MIN  = 4
JITTER_MAX  = 15

# 3 rotating API keys — key1 used as fallback for GEMINI_API_KEY backwards compat
_GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
]

# Retries rotate across keys: attempt 0→key0, 1→key1, 2→key2, 3→key0, 4→key1
def _get_key(attempt: int) -> str:
    """Rotate across available keys. Skip empty keys."""
    available = [k for k in _GEMINI_KEYS if k and k.strip()]
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
    """Alert admin only when BOTH Gemini SDK and Playwright fallback fail."""
    try:
        import requests
        token   = os.getenv("TELEGRAM_ERROR_BOT", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        text = (
            f"🔴 *Gemini — ALL KEYS + PLAYWRIGHT FALLBACK FAILED*\n"
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
# Playwright DeepSeek fallback
# ---------------------------------------------------------------------------
def _playwright_fallback(
    prompt: str,
    system: Optional[str],
    context: str,
) -> Optional[str]:
    """
    Fallback to Playwright DeepSeek browser automation when all Gemini SDK keys fail.
    Free — no API cost. Returns raw text or None.

    Note: ask_deepseek_text() ignores temperature (browser UI has no temp control).
    System prompt is passed through and prepended by the Playwright driver.

    ask_deepseek_text() returns Optional[dict] (already parsed internally).
    We re-serialize to str so _gemini_with_retry() stays str-typed throughout,
    and ask_gemini_json()'s existing _strip_fences() + json.loads() chain works unchanged.
    ask_gemini_text() callers get a JSON string — acceptable for text use cases.
    """
    try:
        from AI.deepseek import ask_deepseek_text
        log.warning(
            "[%s] All Gemini SDK keys failed — falling back to Playwright DeepSeek",
            context,
        )
        result = ask_deepseek_text(prompt, system=system, context=context)
        if result is None:
            return None
        # result is a dict — serialize back to string for uniform pipeline
        return json.dumps(result) if isinstance(result, dict) else str(result)
    except Exception as e:
        log.error("[%s] Playwright DeepSeek fallback failed: %s", context, e)
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
    On total failure → Playwright DeepSeek fallback (free, no API cost).
    On Playwright failure → Telegram alert + return None.

    Key rotation:
        attempt 0 → GEMINI_API_KEY_1
        attempt 1 → GEMINI_API_KEY_2
        attempt 2 → GEMINI_API_KEY_3
        attempt 3 → GEMINI_API_KEY_1
        attempt 4 → GEMINI_API_KEY_2
    """
    available_keys = [k for k in _GEMINI_KEYS if k and k.strip()]
    if not available_keys:
        log.error("[%s] No Gemini API keys configured", context)
        raw = _playwright_fallback(prompt, system, context)
        if raw is None:
            _alert_admin(context, "No Gemini API keys configured")
        return raw

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        api_key = _get_key(attempt - 1)
        key_num = (attempt - 1) % len(available_keys) + 1

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

    # All SDK attempts failed — try Playwright DeepSeek (free fallback)
    raw = _playwright_fallback(prompt, system, context)

    if raw is None:
        # Both SDK and Playwright failed — alert admin
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
      2. Falls back to Playwright DeepSeek (free — no API cost)
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
            "[%s] JSON parse failed: %s | raw: %s",
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
      2. Falls back to Playwright DeepSeek (free — no API cost)
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


def ask_gemini_json_with_key(
    prompt: str,
    key_index: int = 0,
    system: Optional[str] = None,
    temperature: float = 0.2,
    context: str = "gemini_forced",
):
    """
    Single-attempt Gemini call forcing a specific API key index (0-based).
    Returns parsed JSON (list or dict) or None. No retry — caller handles failures.
    Intended for quota-isolated callers (e.g. agenda enricher on key index 2).

    If key_index exceeds the number of available keys, falls back to the last key.
    """
    available = [k for k in _GEMINI_KEYS if k and k.strip()]
    if not available:
        log.warning("[%s] No Gemini API keys configured", context)
        return None
    actual_index = min(key_index, len(available) - 1)
    if actual_index != key_index:
        log.warning(
            "[%s] Forced key index %d unavailable (%d keys), using index %d",
            context, key_index, len(available), actual_index,
        )
    api_key = available[actual_index]
    try:
        raw = _sdk_call(prompt, system, "application/json", temperature, api_key)
        raw = _strip_fences(raw)
        return json.loads(raw)
    except Exception as exc:
        log.warning(
            "[%s] ask_gemini_json_with_key(key_index=%d) failed: %s",
            context, key_index, exc,
        )
        return None