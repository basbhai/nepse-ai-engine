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
import re
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

STABLE_CHECKS    = 4        # consecutive unchanged polls → generation done
POLL_INTERVAL_MS = 500      # ms between polls
RESPONSE_TIMEOUT = 1200_000  # ms — max wait for markdown div to appear (20 min for 55k token prompts)

SEL_EMAIL    = 'input[placeholder="Phone number / email address"]'
SEL_PASSWORD = 'input[placeholder="Password"]'
SEL_TEXTAREA = 'textarea[placeholder="Message DeepSeek"]'
SEL_THINKING  = '.ds-think-content'
SEL_RESPONSE  = '.ds-assistant-message-main-content'
SEL_MARKDOWN = '.ds-assistant-message-main-content'  # response div selector

DEEP_THINK_ENABLED = True  # enable Deep Thinking mode before sending
DEEP_EXPERT =False  # if True, also enable "Deep Expert" mode (if available) for more complex reasoning

# Optional manual override: set DEEPSEEK_DEEP_THINK_SELECTOR to a stable CSS selector
DEEP_THINK_SELECTOR = os.getenv("DEEPSEEK_DEEP_THINK_SELECTOR", "")


_EMAIL    = os.getenv("DEEPSEEK_EMAIL", "")
_PASSWORD = os.getenv("DEEPSEEK_PASSWORD", "")

_HEADLESS = False  # Headless on Linux (production), visible on Windows (development)

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
# Telegram admin alert
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
    """Launch browser with stealth and log in to DeepSeek."""
    global _playwright_instance, _browser, _page, _logged_in

    from playwright.sync_api import sync_playwright

    try:
        from playwright_stealth import stealth_sync
        _stealth_available = True
    except ImportError:
        log.warning("[deepseek] playwright-stealth not installed — captcha risk higher")
        _stealth_available = False

    if _playwright_instance is None:
        log.info("[deepseek] Launching Playwright browser (headless=%s)", _HEADLESS)
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(
            headless=_HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

    _page = _browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
    ).new_page()

    if _stealth_available:
        stealth_sync(_page)
        log.info("[deepseek] Stealth applied")

    log.info("[deepseek] Logging in")
    _page.goto(DEEPSEEK_SIGN_IN)
    _page.wait_for_selector(SEL_EMAIL, state="visible", timeout=15_000)
    _page.fill(SEL_EMAIL, _EMAIL)
    _page.fill(SEL_PASSWORD, _PASSWORD)
    _page.press(SEL_PASSWORD, "Enter")
    _page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=30_000)

    _logged_in = True
    log.info("[deepseek] Login successful")
    # Deep Thinking toggle is *not* enabled here – we do it per chat.


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


def _logout() -> None:
    """Log out via UI dropdown after each response."""
    global _logged_in
    try:
        _page.click('._2afd28d')
        _page.wait_for_timeout(500)
        _page.locator('.ds-dropdown-menu-option', has_text='Log out').click()
        _page.wait_for_timeout(1000)
        _logged_in = False
        log.info("[deepseek] Logged out successfully")
    except Exception as e:
        log.warning("[deepseek] Logout failed: %s", e)
        _logged_in = False


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
# Deep Thinking toggle
# ---------------------------------------------------------------------------
def _enable_deep_expert() -> None:
    """Enable Expert mode (radio button with data-model-type='expert')."""
    try:
        btn = _page.locator('div[data-model-type="expert"]')
        if btn.count() == 0:
            log.warning("[deepseek] Expert mode button not found — skipping")
            return
        state = btn.get_attribute("aria-checked")
        if state == "true":
            log.info("[deepseek] Expert mode already ON")
            return
        btn.click(timeout=5000)
        # Wait for aria-checked to become true
        _page.wait_for_function(
            "() => document.querySelector('div[data-model-type=\"expert\"]')"
            "?.getAttribute('aria-checked') === 'true'",
            timeout=5000,
        )
        log.info("[deepseek] Expert mode enabled")
    except Exception as e:
        log.warning("[deepseek] Expert mode toggle failed: %s", e)

# ---------------------------------------------------------------------------
# Deep Thinking toggle
# ---------------------------------------------------------------------------
def _enable_deep_thinking() -> None:
    """
    Enable Deep Thinking mode if not already active.
    Tries multiple known selectors and falls back gracefully.
    Saves a debug screenshot on failure to help you update the selector.
    """
    # If user provided a custom selector, use it directly
    if DEEP_THINK_SELECTOR:
        try:
            btn = _page.locator(DEEP_THINK_SELECTOR)
            if btn.count() > 0 and btn.is_visible():
                state = btn.get_attribute("aria-pressed")
                if state != "true":
                    btn.click(timeout=5000)
                    btn.wait_for_function(
                        "el => el.getAttribute('aria-pressed') === 'true'",
                        timeout=5000,
                    )
                    log.info("[deepseek] Deep thinking enabled (custom selector)")
                else:
                    log.info("[deepseek] Deep thinking already ON (custom selector)")
                return
            else:
                log.warning("[deepseek] Custom selector not found/visible: %s", DEEP_THINK_SELECTOR)
        except Exception as e:
            log.warning("[deepseek] Custom selector click failed: %s", e)

    # Default automatic detection – try a list of plausible selectors
    selectors_to_try = [
        # Most likely: a button/div with aria-pressed containing the text "Deep"
        '*[aria-pressed]:has-text("Deep")',
        # Alternative: a toggle switch with label "Deep thinking"
        '[role="switch"]:has-text("Deep")',
        # Fallback: any element that contains "Deep thinking" exactly
        ':has-text("Deep thinking")',
    ]

    for sel in selectors_to_try:
        try:
            elem = _page.locator(sel).first
            if elem.count() == 0:
                continue
            # Wait briefly for it to be attached
            elem.wait_for(state="attached", timeout=3000)

            # Walk up to the nearest clickable parent (if the located element isn't directly clickable)
            # Often the span is inside a button; we click the closest element with role="button" or aria-pressed
            clickable = elem
            if not (clickable.get_attribute("role") or clickable.get_attribute("aria-pressed")):
                # Try parent, grandparent
                parent = elem.locator('xpath=ancestor::*[self::button or self::div[@role="button"] or @aria-pressed][1]')
                if parent.count() > 0:
                    clickable = parent

            state = clickable.get_attribute("aria-pressed")
            if state == "true":
                log.info("[deepseek] Deep thinking already ON (selector: %s)", sel)
                return
            else:
                clickable.click(timeout=5000)
                # Wait until pressed becomes true
                clickable.wait_for_function(
                    "el => el.getAttribute('aria-pressed') === 'true'",
                    timeout=5000,
                )
                log.info("[deepseek] Deep thinking enabled (selector: %s)", sel)
                return
        except Exception:
            continue

    # If we get here, no selector worked
    log.warning("[deepseek] Could not find/toggle Deep Thinking button. Taking debug screenshot.")
    # try:
    #     _page.screenshot(path=f"deepseek_toggle_fail_{int(time.time())}.png")
    # except Exception:
    #     pass
    log.warning("[deepseek] Deep Thinking mode will NOT be active for this request.")


# ---------------------------------------------------------------------------
# Core Playwright call
# ---------------------------------------------------------------------------
def _playwright_call(prompt: str, system: Optional[str]) -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    _ensure_session()
    _page.goto(DEEPSEEK_HOME)
    _page.wait_for_selector(SEL_TEXTAREA, state="visible", timeout=20_000)

    if DEEP_THINK_ENABLED:
        _enable_deep_thinking()
    
    if DEEP_EXPERT:
        _enable_deep_expert()

    # Count existing RESPONSE blocks before sending (ignore thinking blocks)
    existing_count = _page.locator(SEL_RESPONSE).count()
    log.info("[deepseek] Existing response blocks before send: %d", existing_count)

    _page.locator(SEL_TEXTAREA).fill(full_prompt)
    _page.press(SEL_TEXTAREA, "Enter")

    # Step 1: Wait for thinking block to appear (confirms message was sent)
    try:
        _page.wait_for_selector(SEL_THINKING, state="visible", timeout=30_000)
        log.info("[deepseek] Thinking started...")
    except Exception:
        log.warning("[deepseek] No thinking block detected — proceeding anyway")

    # Step 2: Wait for thinking to FINISH (thinking block collapses/hides)
    # The thinking div gets a collapsed state; wait for a NEW response block
    _page.wait_for_function(
        f"() => document.querySelectorAll('{SEL_RESPONSE}').length > {existing_count}",
        timeout=RESPONSE_TIMEOUT,
    )
    log.info("[deepseek] Response block appeared — thinking complete")

    # Step 3: Poll the response block until text is stable
    resp_locator = _page.locator(SEL_RESPONSE).last
    last   = ""
    stable = 0

    while stable < STABLE_CHECKS:
        _page.wait_for_timeout(POLL_INTERVAL_MS)
        current = resp_locator.inner_text()
        if current and current == last:
            stable += 1
        else:
            stable = 0
            last   = current

    if not last:
        raise ValueError("DeepSeek returned empty response")

    log.info("[deepseek] Response stable (%d chars)", len(last))
    _logout()
    return last
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
# Sanitize control characters inside JSON string values
# ---------------------------------------------------------------------------
def _sanitize_json(raw: str) -> str:
    """
    Replace literal unescaped control characters (newline, carriage return,
    tab) inside JSON string values. DeepSeek via Playwright sometimes returns
    these literally inside strings — valid in display but invalid in JSON.
    """
    def _fix(m):
        s = m.group(0)
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    return re.sub(r'"(?:[^"\\]|\\.)*"', _fix, raw, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ask_deepseek_text(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.4,       # accepted for signature compatibility; not used by UI
    context: str = "deepseek_text",
    use_search: bool = False,        # accepted for signature compatibility; not used here
    return_raw: bool = False,
) -> Optional[dict]:
    """
    Send prompt to DeepSeek via Playwright browser automation.
    Returns parsed JSON dict, or None on failure.

    Signature is intentionally identical to ask_gemini_text() so callers
    can swap between the two without any changes.
    """
    last_error = ""
    raw        = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] DeepSeek attempt %d/%d", context, attempt, MAX_RETRIES)

            raw = _playwright_call(prompt, system)
            if return_raw:
                return raw
            raw    = _strip_fences(raw)
            raw    = _sanitize_json(raw)
            result = json.loads(raw)

            log.info("[%s] DeepSeek responded on attempt %d", context, attempt)
            return result

        except json.JSONDecodeError as exc:
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


def ask_deepseek_review_playwright(
    system: str,
    prompt: str,
    context: str = "learning_hub",
) -> Optional[str]:
    """
    Learning hub weekly review via Playwright DeepSeek with Deep Thinking mode.
    Returns raw text string (not parsed JSON) — learning_hub handles parsing.
    Returns None on failure.
    """
    full_prompt = f"{system}\n\n{prompt}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("[%s] Playwright DeepSeek review attempt %d/%d", context, attempt, MAX_RETRIES)
            raw = _playwright_call(full_prompt, system=None)
            if raw:
                return raw
        except Exception as e:
            log.warning("[%s] Playwright attempt %d failed: %s", context, attempt, e)
            global _logged_in
            _logged_in = False
            if attempt < MAX_RETRIES:
                time.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
    log.error("[%s] All Playwright attempts failed", context)
    return None