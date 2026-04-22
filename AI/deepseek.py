"""
AI/deepseek.py
==============
DeepSeek gateway using Playwright browser automation.

Exposes ask_deepseek_text() with the same signature as ask_gemini_text(),
but drives the DeepSeek chat UI via Playwright and returns parsed JSON.

Usage:
    from AI.deepseek import ask_deepseek_text
    result = ask_deepseek_text(prompt, system="You are...", context="nepse_analysis")
"""

import json
import logging
import os
import random
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEEPSEEK_SIGN_IN = "https://chat.deepseek.com/sign_in"
DEEPSEEK_HOME    = "https://chat.deepseek.com"

MAX_RETRIES      = 5
JITTER_MIN       = 4
JITTER_MAX       = 15

STABLE_CHECKS    = 4      # consecutive unchanged polls → generation done
POLL_INTERVAL_MS = 500    # ms between polls
RESPONSE_TIMEOUT = 90_000 # ms — max wait for <pre> to appear

SEL_EMAIL    = 'input[placeholder="Phone number / email address"]'
SEL_PASSWORD = 'input[placeholder="Password"]'
SEL_TEXTAREA = 'textarea[placeholder="Message DeepSeek"]'

_EMAIL    = os.getenv("DEEPSEEK_EMAIL", "")
_PASSWORD = os.getenv("DEEPSEEK_PASSWORD", "")

# ---------------------------------------------------------------------------
# Module-level singleton session
# ---------------------------------------------------------------------------
_playwright_instance = None
_browser             = None
_page                = None
_logged_in           = False


# ---------------------------------------------------------------------------
# Retryable error detection
# ---------------------------------------------------------------------------
_RETRYABLE = (
    "timeout", "timed out", "crash", "disconnected",
    "net::", "connection", "target closed", "page closed",
)

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k.lower() in msg for k in _RETRYABLE)


# ---------------------------------------------------------------------------
# Telegram admin alert  (mirrors gemini.py)
# ---------------------------------------------------------------------------
def _alert_admin(context: str, last_error: str) -> None:
    try:
        import requests
        token   = os.getenv("TELEGRAM_ERROR_BOT", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        text = (
            f"🔴 *DeepSeek — ALL RETRIES FAILED*\n"
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


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def _launch_and_login() -> None:
    """Launch browser and log in to DeepSeek. Stores state in module globals."""
    global _playwright_instance, _browser, _page, _logged_in

    from playwright.sync_api import sync_playwright

    if _playwright_instance is None:
        log.info("[deepseek] Launching Playwright browser")
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=False)

    _page = _browser.new_page()

    log.info("[deepseek] Logging in")
    _page.goto(DEEPSEEK_SIGN_IN)
    _page.wait_for_selector(SEL_EMAIL, state="visible", timeout=15_000)
    _page.fill(SEL_EMAIL, _EMAIL)
    _page.fill(SEL_PASSWORD, _PASSWORD)
    _page.press(SEL_PASSWORD, "Enter")
    _page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=30_000)

    _logged_in = True
    log.info("[deepseek] Login successful")


def _ensure_session() -> None:
    """Ensure a live logged-in session; re-launch/re-login if stale."""
    global _logged_in
    try:
        if _page is None or not _logged_in:
            raise RuntimeError("no session")
        _page.evaluate("() => document.title")  # liveness check
    except Exception:
        log.warning("[deepseek] Session stale — reinitialising")
        _logged_in = False
        _launch_and_login()


def close_session() -> None:
    """Cleanly close the browser. Call at app shutdown."""
    global _playwright_instance, _browser, _page, _logged_in
    try:
        if _browser:
            _browser.close()
        if _playwright_instance:
            _playwright_instance.stop()
    except Exception as e:
        log.warning("[deepseek] Error closing session: %s", e)
    finally:
        _playwright_instance = _browser = _page = None
        _logged_in = False
    log.info("[deepseek] Browser session closed")


# ---------------------------------------------------------------------------
# Core Playwright call
# ---------------------------------------------------------------------------
def _playwright_call(prompt: str, system: Optional[str]) -> str:
    """
    Send a prompt to the DeepSeek chat UI and return the raw <pre> text.

    System prompt is prepended to the user prompt since the chat UI
    has no separate system prompt field.
    Raises on timeout or page errors — caller handles retry.
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    _ensure_session()

    # Navigate to a fresh chat
    _page.goto(DEEPSEEK_HOME)
    _page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=20_000)

    _page.locator(SEL_TEXTAREA).fill(full_prompt)
    _page.press(SEL_TEXTAREA, "Enter")

    # Wait for response block to appear
    _page.wait_for_selector("pre", state="visible", timeout=RESPONSE_TIMEOUT)

    # Poll until text stops changing (generation complete)
    pre    = _page.locator("pre").last
    last   = ""
    stable = 0

    while stable < STABLE_CHECKS:
        _page.wait_for_timeout(POLL_INTERVAL_MS)
        current = pre.inner_text()
        if current and current == last:
            stable += 1
        else:
            stable = 0
            last   = current

    if not last:
        raise ValueError("DeepSeek returned empty response")

    return last


# ---------------------------------------------------------------------------
# Strip markdown fences  (mirrors gemini.py)
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
# Public API
# ---------------------------------------------------------------------------
def ask_deepseek_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,       # accepted for signature compatibility; not used by UI
    context: str = "deepseek_text",
    use_search: bool = False,        # accepted for signature compatibility; not used here
) -> Optional[dict]:
    """
    Send prompt to DeepSeek via Playwright browser automation.
    Returns parsed JSON dict, or None on failure.

    Signature is intentionally identical to ask_gemini_text() so callers
    can swap between the two without any changes.

    Internally:
      1. Launches a singleton browser session (login once, reuse)
      2. Sends the prompt via the DeepSeek chat UI
      3. Waits for generation to finish (stable <pre> text)
      4. Strips markdown fences and parses JSON
      5. Retries up to MAX_RETRIES times with jitter on transient errors
      6. Alerts admin via Telegram if all retries fail

    Usage:
        from AI.deepseek import ask_deepseek_text
        result = ask_deepseek_text(
            prompt  = "Analyse NEPSE top gainers today",
            system  = "You are a financial analyst. Respond in JSON only.",
            context = "nepse_gainers",
        )
    """
    last_error = ""
    raw        = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] DeepSeek attempt %d/%d", context, attempt, MAX_RETRIES)

            raw    = _playwright_call(prompt, system)
            raw    = _strip_fences(raw)
            result = json.loads(raw)

            log.info("[%s] DeepSeek responded on attempt %d", context, attempt)
            return result

        except json.JSONDecodeError as exc:
            # Got a response but it wasn't valid JSON — log and retry
            last_error = f"JSONDecodeError: {exc} | raw: {raw[:200]}"
            log.error("[%s] Invalid JSON (attempt %d): %s", context, attempt, last_error)

        except Exception as exc:
            last_error = str(exc)
            if _is_retryable(exc):
                if attempt < MAX_RETRIES:
                    wait = random.uniform(JITTER_MIN, JITTER_MAX)
                    log.warning(
                        "[%s] Transient error (attempt %d/%d): %s — retrying in %.1fs",
                        context, attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                    continue
            else:
                log.warning(
                    "[%s] Non-retryable error (attempt %d): %s",
                    context, attempt, exc,
                )

    log.error("[%s] All %d attempts failed. Last: %s", context, MAX_RETRIES, last_error)
    _alert_admin(context, last_error)
    return None