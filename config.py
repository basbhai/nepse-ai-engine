"""
config.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Central configuration: shared constants and environment setup.
Import from here instead of re-defining in every module.
"""

import os
from datetime import timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

# ── Nepal Standard Time ───────────────────────────────────────────────────────
NST = timezone(timedelta(hours=5, minutes=45))

# ── Gemini AI ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Shared browser User-Agent ─────────────────────────────────────────────────
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)
