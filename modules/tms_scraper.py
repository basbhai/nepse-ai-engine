"""
tms_scraper.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2
Purpose : Authenticate with TMS49 (NEPSE Trading Management System) and
          fetch live market data — top securities, indices.

Auth flow : Every run does a fresh login (no session caching).
            Gemini 2.5 Flash-Lite solves the captcha from raw image bytes
            — no PNG file is written to disk.

Designed to run on GitHub Actions (no persistent filesystem needed).
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, base64, logging, json
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
import requests
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [TMS] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TMS_BASE        = "https://tms49.nepsetms.com.np"
TMS_AUTH_URL    = f"{TMS_BASE}/tmsapi/authApi/authenticate"
TMS_TOP25_URL   = f"{TMS_BASE}/tmsapi/rtApi/ws/top25securities"
TMS_INDICES_URL = f"{TMS_BASE}/tmsapi/exchangeIndex/indices"
MEMBER_CODE     = "49"

USERNAME       = os.getenv("TMS_USERNAME")
PASSWORD       = os.getenv("TMS_PASSWORD")
REQUEST_OWNER  = os.getenv("TMS_REQUEST_OWNER", "109268")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

NST = timezone(timedelta(hours=5, minutes=45))

BASE_HEADERS = {
    "Accept":             "application/json, text/plain, */*",
    "Accept-Encoding":    "gzip, deflate, br, zstd",
    "Accept-Language":    "en-GB,en-US;q=0.9,en;q=0.8",
    "Priority":           "u=1, i",
    "Sec-Ch-Ua":          '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "Sec-Ch-Ua-Mobile":   "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest":     "empty",
    "Sec-Fetch-Mode":     "cors",
    "Sec-Fetch-Site":     "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI — new google.genai SDK
# ══════════════════════════════════════════════════════════════════════════════

def gemini_solve_captcha(image_bytes: bytes) -> str | None:
    """
    Solves TMS captcha using the stable Gemini 2.5 Flash model.
    Optimized for the new google-genai SDK.
    """
    from google import genai
    from google.genai import types

    try:
        # Use the clean stable string to avoid 404 errors
        # 'gemini-2.5-flash' is the standard ID for the v1beta/v1 endpoints
        TARGET_MODEL = "gemini-2.5-flash"

        client = genai.Client(api_key=GEMINI_API_KEY)
        
        log.info(f"Solving captcha with {TARGET_MODEL}...")

        response = client.models.generate_content(
            model=TARGET_MODEL, 
            contents=[
                "Solve this captcha image. Output ONLY the characters. No spaces, no punctuation.",
                types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            ]   
        )
        
        if response and response.text:
            # Cleanup: ensure no spaces or newlines trip up the NEPSE login
            solved_text = response.text.strip().replace(" ", "")
            log.info(f"✅ Gemini solved it: {solved_text}")
            return solved_text
        
        log.warning("Gemini returned an empty response.")
        return None

    except Exception as e:
        # If you still get a 404, check your API Key permissions for v1beta
        log.error(f"❌ Gemini solve failed: {str(e)}")
        return None
# ══════════════════════════════════════════════════════════════════════════════
# COOKIE EXTRACTION — parse Set-Cookie headers directly
# ══════════════════════════════════════════════════════════════════════════════

def extract_cookies_from_response(resp: requests.Response) -> dict:
    """
    Extract _rid, _aid, XSRF-TOKEN from Set-Cookie response headers.
    Uses stdlib SimpleCookie for robust parsing.
    """
    cookies = {}
    targets = {"_rid", "_aid", "XSRF-TOKEN"}

    # requests stores raw headers — getlist gives all Set-Cookie values
    raw_set_cookies = []
    try:
        raw_set_cookies = resp.raw.headers.getlist("Set-Cookie")
    except Exception:
        pass

    # Fallback: single header value (may only have last one)
    if not raw_set_cookies:
        val = resp.headers.get("set-cookie", "")
        if val:
            raw_set_cookies = [val]

    log.info("Set-Cookie headers found: %d", len(raw_set_cookies))

    for raw in raw_set_cookies:
        # SimpleCookie parses "name=value; attr=val; ..." correctly
        sc = SimpleCookie()
        try:
            sc.load(raw)
        except Exception:
            pass
        for name, morsel in sc.items():
            if name in targets:
                cookies[name] = morsel.value
                log.info("  Parsed cookie: %s = %s...", name, morsel.value[:40])

    # If SimpleCookie missed any, fall back to manual split
    for raw in raw_set_cookies:
        first = raw.split(";")[0].strip()
        if "=" in first:
            name, _, value = first.partition("=")
            name = name.strip()
            if name in targets and name not in cookies:
                cookies[name] = value.strip()
                log.info("  Manual parse cookie: %s = %s...", name, value[:40])

    missing = targets - set(cookies.keys())
    if missing:
        log.warning("Could not extract cookies: %s", missing)

    return cookies


def build_host_session_id(session_uuid: str) -> str:
    """base64('MjQ=-<uuid>') — MjQ= is base64('24'), your internal user ID."""
    return base64.b64encode(f"MjQ=-{session_uuid}".encode()).decode()





# ══════════════════════════════════════════════════════════════════════════════
# SESSION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_session(cookies: dict) -> requests.Session:
    session = requests.Session()
    domain  = "tms49.nepsetms.com.np"
    for name, value in cookies.items():
        if name in ("_rid", "_aid", "XSRF-TOKEN"):
            session.cookies.set(name, value, domain=domain, path="/")
    return session


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════════

def fresh_login() -> tuple[requests.Session, str]:
    log.info("Fresh login — Gemini captcha solve...")
    tmp     = requests.Session()
    hdr     = {**BASE_HEADERS, "Referer": f"{TMS_BASE}/login",
               "Content-Type": "application/json"}

    # Captcha ID
    r          = tmp.get(f"{TMS_BASE}/tmsapi/authApi/captcha/id",
                         headers=hdr, timeout=10)
    captcha_id = r.json()["id"]
    log.info("Captcha ID: %s", captcha_id)

    # Captcha image
    r = tmp.get(
        f"{TMS_BASE}/tmsapi/authApi/captcha/image/{captcha_id}",
        headers={**BASE_HEADERS, "Referer": f"{TMS_BASE}/login",
                 "Accept": "image/*,*/*;q=0.8",
                 "Sec-Fetch-Dest": "image", "Sec-Fetch-Mode": "no-cors"},
        timeout=10,
    )
    image_bytes = r.content
    log.info("Captcha image fetched — %d bytes (in-memory only)", len(image_bytes))

    # Gemini solve
    solved = gemini_solve_captcha(image_bytes)

    # Authenticate
    auth = tmp.post(
        TMS_AUTH_URL, headers=hdr,
        json={"userName": USERNAME,
              "password":  base64.b64encode(PASSWORD.encode()).decode(),
              "captchaIdentifier": captcha_id,
              "userCaptcha": solved,
              "jwt": "", "otp": ""},
        timeout=15,
    )

    log.info("Auth HTTP %d", auth.status_code)
    if auth.status_code != 200:
        raise RuntimeError(f"Login failed: HTTP {auth.status_code} — {auth.text[:300]}")

    # ── DIAGNOSTIC: log full login response body ──────────────────────────
    try:
        print("\n" + "─" * 60)

    except Exception:
        print("LOGIN BODY (raw):", auth.text[:1000])

    # Extract cookies from Set-Cookie headers
    cookies = extract_cookies_from_response(auth)

    # Verify we got what we need
    if not cookies.get("_rid") or not cookies.get("_aid"):
        raise RuntimeError(
            f"Missing _rid/_aid after login. Got: {list(cookies.keys())}"
        )

    # Build host-session-id
    try:
        body         = auth.json()
        session_uuid = (
            body.get("sessionId") or
            body.get("session_id") or
            (body.get("data") or {}).get("sessionId") or
            captcha_id
        )
    except Exception:
        session_uuid = captcha_id

    host_session_id = build_host_session_id(session_uuid)
    log.info("host-session-id: %s...", host_session_id[:40])

    session = build_session(cookies)
    return session, host_session_id


def get_session() -> tuple[requests.Session, str]:
    """
    Always performs a fresh login.
    Returns (session, host_session_id).
    No caching — designed for GitHub Actions where each run is ephemeral.
    """
    session, host_sid = fresh_login()
    return session, host_sid


# ══════════════════════════════════════════════════════════════════════════════
# HEADERS + FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

def dashboard_headers(session: requests.Session, host_session_id: str) -> dict:
    return {
        **BASE_HEADERS,
        "Referer":         f"{TMS_BASE}/tms/mwDashboard",
        "Membercode":      MEMBER_CODE,
        "Request-Owner":   REQUEST_OWNER,
        "Host-Session-Id": host_session_id,
        "X-Xsrf-Token":    session.cookies.get("XSRF-TOKEN", ""),
    }


def fetch_top25(session, host_sid) -> list:
    r = session.get(TMS_TOP25_URL,
                    headers=dashboard_headers(session, host_sid), timeout=15)
    log.info("top25 → HTTP %d", r.status_code)
    if r.status_code == 500:
        log.warning("500 — market closed (expected outside 10:45–15:00 NST)")
        return []
    r.raise_for_status()
    data    = r.json()
    payload = data.get("payload", data)
    return (payload.get("data") if isinstance(payload, dict) else payload) or []


def fetch_indices(session, host_sid) -> list:
    r = session.get(TMS_INDICES_URL,
                    headers=dashboard_headers(session, host_sid), timeout=15)
    log.info("indices → HTTP %d", r.status_code)
    r.raise_for_status()
    return r.json()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TMS49 -- Fresh login + data test")
    print("=" * 60 + "\n")

    session, host_sid = get_session()

    print(f"\n Session ready — fresh login every run")
    print(f"   _rid             : {session.cookies.get('_rid','?')[:40]}...")
    print(f"   XSRF-TOKEN       : {session.cookies.get('XSRF-TOKEN','?')}")
    print(f"   host-session-id  : {host_sid[:40]}...")
    print(f"   request-owner    : {REQUEST_OWNER}")

    print("\n-- Test 1: /exchangeIndex/indices -------------------")
    try:
        indices = fetch_indices(session, host_sid)
        for idx in indices:
            print(f"  {idx.get('exchangeIndexId'):>3}  {idx.get('indexCode')}")
        print(f"  -> {len(indices)} indices OK")
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n-- Test 2: /rtApi/ws/top25securities ----------------")
    try:
        secs = fetch_top25(session, host_sid)
        if secs:
            print(f"  {'Symbol':<10} {'LTP':>8} {'Chg%':>7} {'Volume':>10}")
            for s in secs[:5]:
                print(f"  {s.get('symbol','?'):<10} "
                      f"{s.get('ltp',0):>8.2f} "
                      f"{s.get('percentChange',0):>+7.2f}% "
                      f"{s.get('volume',0):>10,}")
            print(f"  -> {len(secs)} securities OK")
        else:
            print("  -> 0 results (market closed — retest Sun-Thu 10:45-15:00 NST)")
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n" + "=" * 60 + "\n")