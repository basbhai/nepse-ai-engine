"""
AI/openrouter.py
================
Central OpenRouter gateway for the NEPSE AI Engine.
Handles Claude (Sonnet), GPT-4o, and DeepSeek R1 — all via OpenRouter.

All modules must import from here — no inline OpenRouter calls anywhere else.

    from AI.openrouter import ask_claude, ask_gpt, ask_deepseek

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
from dotenv import load_dotenv

import requests

log = logging.getLogger(__name__)
load_dotenv()  # Load environment variables from .env file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

CLAUDE_MODEL   = os.getenv("CLAUDE_MODEL",   "anthropic/claude-sonnet-4-6")
GPT_MODEL      = os.getenv("GPT_MODEL",      "openai/gpt-4o")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1")

MAX_RETRIES = 5
JITTER_MIN  = 4    # seconds
JITTER_MAX  = 15   # seconds

# HTTP status codes that warrant a retry
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Error message substrings that warrant a retry
_RETRYABLE_MSG = (
    "timeout",
    "timed out",
    "connection",
    "unavailable",
    "rate limit",
    "overloaded",
    "try again",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_retryable_exc(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE_MSG)


def _alert_admin(context: str, last_error: str) -> None:
    """Send Telegram alert to admin when all retries are exhausted."""
    try:
        token   = os.getenv("TELEGRAM_ERROR_BOT", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        text = (
            f"🔴 *OpenRouter — ALL RETRIES EXHAUSTED*\n"
            f"Context: `{context}`\n"
            f"Last error: `{last_error[:200]}`"
        )
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.error("Failed to send Telegram alert: %s", e)


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences models sometimes add."""
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _call(
    model:      str,
    messages:   list,
    max_tokens: int,
    temperature: float,
    context:    str,
    extra_body: Optional[dict] = None,
) -> Optional[str]:
    """
    Core HTTP call to OpenRouter with retry logic.
    Returns raw response text or None on total failure.
    """
    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not set in .env")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/basbhai/nepse-ai-engine",
        "X-Title":       "NEPSE AI Engine",
    }

    payload = {
        "model":       model,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "messages":    messages,
    }
    if extra_body:
        payload.update(extra_body)

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] OpenRouter call attempt %d/%d (model=%s)", context, attempt, MAX_RETRIES, model)

            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            # Check for retryable HTTP status
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt < MAX_RETRIES:
                    wait = random.uniform(JITTER_MIN, JITTER_MAX)
                    log.warning(
                        "[%s] OpenRouter HTTP %d (attempt %d/%d) — retrying in %.1fs",
                        context, resp.status_code, attempt, MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    log.error("[%s] OpenRouter failed after %d attempts: %s", context, MAX_RETRIES, last_error)
                    _alert_admin(context, last_error)
                    return None

            # Non-retryable HTTP error
            if resp.status_code != 200:
                log.error("[%s] OpenRouter non-retryable HTTP %d: %s", context, resp.status_code, resp.text[:300])
                return None

            data = resp.json()
            raw  = data["choices"][0]["message"]["content"].strip()
            log.info("[%s] OpenRouter responded on attempt %d", context, attempt)
            return raw

        except requests.exceptions.Timeout as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = random.uniform(JITTER_MIN, JITTER_MAX)
                log.warning(
                    "[%s] OpenRouter timeout (attempt %d/%d) — retrying in %.1fs",
                    context, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = random.uniform(JITTER_MIN, JITTER_MAX)
                log.warning(
                    "[%s] OpenRouter connection error (attempt %d/%d) — retrying in %.1fs",
                    context, attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)

        except Exception as exc:
            last_error = str(exc)
            if _is_retryable_exc(exc) and attempt < MAX_RETRIES:
                wait = random.uniform(JITTER_MIN, JITTER_MAX)
                log.warning(
                    "[%s] OpenRouter transient error (attempt %d/%d): %s — retrying in %.1fs",
                    context, attempt, MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                log.error("[%s] OpenRouter non-retryable error: %s", context, exc)
                return None

    # All retries exhausted
    log.error("[%s] OpenRouter failed after %d attempts", context, MAX_RETRIES)
    _alert_admin(context, last_error)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_claude(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    context: str = "claude",
) -> Optional[dict]:
    """
    Call Claude Sonnet via OpenRouter. Returns parsed JSON dict or None.

    Claude is always instructed to return valid JSON — the system prompt
    enforces this. Caller does NOT need to parse JSON.

    Args:
        prompt:      Full user prompt.
        system:      System prompt. Defaults to NEPSE analyst instruction.
        max_tokens:  Max tokens (default 1200 — sufficient for signal JSON).
        temperature: Sampling temperature (default 0.2).
        context:     Caller name — shown in logs and Telegram alerts.

    Returns:
        Parsed dict, or None if all retries fail or JSON is invalid.

    Usage:
        from AI.openrouter import ask_claude
        result = ask_claude(prompt, system="You are ...", context="claude_analyst")
    """
    system_prompt = system or (
        "You are a senior NEPSE quantitative analyst. "
        "You respond ONLY in valid JSON. No markdown fences. "
        "You never recommend a trade without a clear stop loss."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    raw = _call(
        model       = CLAUDE_MODEL,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        context     = context,
    )
    if raw is None:
        return None

    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("[%s] Claude returned invalid JSON: %s | raw: %s", context, exc, raw[:300])
        return None


def ask_gpt(
    system: str,
    prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    context: str = "gpt",
) -> Optional[str]:
    """
    Call GPT-4o via OpenRouter. Returns raw text string or None.

    GPT (learning hub) returns free-form structured text, not JSON.
    Caller is responsible for any parsing needed.

    Args:
        system:      System prompt.
        prompt:      User prompt.
        max_tokens:  Max tokens (default 4000 — learning hub needs long output).
        temperature: Sampling temperature (default 0.2).
        context:     Caller name — shown in logs and Telegram alerts.

    Returns:
        Raw text string, or None if all retries fail.

    Usage:
        from AI.openrouter import ask_gpt
        text = ask_gpt(system_prompt, user_prompt, context="learning_hub")
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    return _call(
        model       = GPT_MODEL,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        context     = context,
    )


def ask_deepseek(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    context: str = "deepseek",
) -> Optional[dict]:
    """
    Call DeepSeek R1 via OpenRouter. Returns parsed JSON dict or None.

    DeepSeek is used for Kelly Criterion math reasoning.
    Temperature is low (0.1) — math needs determinism.
    Reasoning mode is enabled via extra_body.

    Args:
        prompt:      Full prompt (no system prompt — DeepSeek R1 is user-turn only).
        max_tokens:  Max tokens (default 1000).
        temperature: Sampling temperature (default 0.1).
        context:     Caller name — shown in logs and Telegram alerts.

    Returns:
        Parsed dict, or None if all retries fail or JSON is invalid.

    Usage:
        from AI.openrouter import ask_deepseek
        result = ask_deepseek(prompt, context="budget_kelly")
    """
    messages = [
        {"role": "user", "content": prompt},
    ]

    raw = _call(
        model       = DEEPSEEK_MODEL,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        context     = context,
        extra_body  = {"reasoning": {"enabled": True}},
    )
    if raw is None:
        return None

    raw = _strip_fences(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("[%s] DeepSeek returned invalid JSON: %s | raw: %s", context, exc, raw[:300])
        return None


def ask_free(
    prompt: str,
    system: Optional[str] = None,
    context: str = "free",
) -> Optional[str]:
    """
    Call a free OpenRouter model for lightweight NLP tasks.
    No retry — free tier is flaky, fail fast and let caller handle it.
    Returns raw text or None on failure.

    Usage:
        from AI.openrouter import ask_free
        result = ask_free(prompt, system="You are a parser...", context="telegram_nlp")
    """
    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not set in .env")
        return None

    # Gemma free models don't support system role — merge into user message
    combined = f"{system}\n\n{prompt}" if system else prompt
    messages = [{"role": "user", "content": combined}]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/basbhai/nepse-ai-engine",
        "X-Title":       "NEPSE AI Engine",
    }
    payload = {
        "model":       "google/gemma-3n-e2b-it:free",
        "max_tokens":  200,
        "temperature": 0.1,
        "messages":    messages,
    }

    try:
        log.info("[%s] Free model call...", context)
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )
        if resp.status_code != 200:
            log.warning(
                "[%s] Free model HTTP %d: %s",
                context, resp.status_code, resp.text[:200],
            )
            return None
        content = resp.json()["choices"][0]["message"].get("content") or ""
        if not content:
            log.warning("[%s] Free model returned empty content", context)
            return None
        log.info("[%s] Free model responded", context)
        return content.strip()
    except Exception as exc:
        log.warning("[%s] Free model failed: %s", context, exc)
        return None
    

def ask_gemini_lite(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,
    context: str = "gemini_lite",
) -> Optional[str]:
    """
    Call Gemini 2.5 Flash-Lite via OpenRouter — cheap nightly summarization.
    Returns raw text or None on failure.

    Usage:
        from AI import ask_gemini_lite
        result = ask_gemini_lite(prompt, context="daily_summarizer")
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _call(
        model       = "google/gemini-2.5-flash-lite",
        messages    = messages,
        max_tokens  = 1000,
        temperature = temperature,
        context     = context,
    )