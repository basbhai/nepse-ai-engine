"""
calendar_guard.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2
Purpose : Determine if NEPSE is currently open for trading.
          Called by EVERY GitHub Actions workflow as the first gate.
          If is_open() returns False, the workflow exits immediately.

Nepal Standard Time : UTC+5:45  (non-standard offset — handled manually)
Market hours        : Sunday–Thursday, 10:45 AM – 3:00 PM NST
Holiday source      : Two-layer system:
                        Layer 1 — Hardcoded FY 2082/83 BS gazette list
                        Layer 2 — SETTINGS tab key: Is_Holiday
                                  Format: "YYYY-MM-DD:Reason,YYYY-MM-DD:Reason"
                                  Written by scraper / news / gemini via
                                  sheets.update_setting("Is_Holiday", ...)
                      Terai-specific holidays EXCLUDED (NEPSE is Kathmandu)

Sheets fallback     : If Sheets is unreachable at import, layer 1 is used.
                      System never blocks trading due to a Sheets timeout.

Ad-hoc closure flow:
  scraper / news / gemini detects bandh or NEPSE closure notice
      → calls flag_adhoc_closure(date, reason, source)
      → appends to Is_Holiday in SETTINGS tab
      → next GitHub Actions container (fresh import) picks it up

Usage (in every workflow entry point):
──────────────────────────────────────
    from calendar_guard import is_open, get_status
    import sys

    guard = get_status()
    print(guard["message"])

    if not guard["is_open"]:
        sys.exit(0)

─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime, date, time, timezone, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TIMEZONE
# ══════════════════════════════════════════════════════════════════════════════

_NST_OFFSET = timedelta(hours=5, minutes=45)
NST = timezone(_NST_OFFSET, name="NST")


def now_nst() -> datetime:
    """Return current datetime in Nepal Standard Time."""
    return datetime.now(tz=timezone.utc).astimezone(NST)


def today_nst() -> date:
    """Return current date in Nepal Standard Time."""
    return now_nst().date()


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRADING DAY DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

# Python weekday(): Monday=0, Tuesday=1, Wednesday=2, Thursday=3,
#                   Friday=4,   Saturday=5, Sunday=6
TRADING_DAYS = frozenset({ 0, 1, 2, 3,4})  #  Mon, Tue, Wed, Thu, fri

WEEKDAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. SESSION WINDOWS (all NST)
# ══════════════════════════════════════════════════════════════════════════════

MARKET_OPEN  = time(10, 45)   # Regular trading session open
MARKET_CLOSE = time(15,  0)   # Regular trading session close

PREOPEN_START = time(10, 30)  # morning_brief.yml window start
PREOPEN_END   = time(10, 45)  # morning_brief.yml window end / market open

EOD_START = time(15,  0)      # eod_audit.yml window start
EOD_END   = time(15, 30)      # eod_audit.yml window end


# ══════════════════════════════════════════════════════════════════════════════
# 4. LAYER 1 — HARDCODED HOLIDAYS (FY 2082/83 BS, 2026–2027 Gregorian)
#    Source  : Nepal Federal Government Gazette
#    Scope   : National holidays + Kathmandu Valley holidays
#    Excluded: Terai-specific dates (NEPSE operates from Kathmandu)
#
#    DO NOT add ad-hoc closures here.
#    Use flag_adhoc_closure() — it writes to SETTINGS tab (Is_Holiday key).
# ══════════════════════════════════════════════════════════════════════════════

_GAZETTE_HOLIDAYS: dict[date, str] = {

    # ── 2026 ──────────────────────────────────────────────────────────────────

    date(2026,  4, 14): "Nepali New Year (Baisakh 1, 2083 BS)",

    date(2026,  5,  1): "Buddha Jayanti / International Workers' Day",
    date(2026,  5, 29): "Republic Day",

    date(2026,  8, 28): "Rakshya Bandhan",
    date(2026,  8, 29): "Gai Jatra (Kathmandu Valley)",

    date(2026,  9,  4): "Krishna Janmashtami",
    date(2026,  9, 19): "Constitution Day (National Day)",
    date(2026,  9, 25): "Indra Jatra (Kathmandu Valley)",

    date(2026, 10, 11): "Ghatasthapana",
    date(2026, 10, 17): "Dashain — Phulpati",
    date(2026, 10, 18): "Dashain — Maha Ashtami",
    date(2026, 10, 19): "Dashain — Maha Navami",
    date(2026, 10, 20): "Dashain — Vijaya Dashami",
    date(2026, 10, 21): "Dashain — Ekadashi",
    date(2026, 10, 22): "Dashain — Dwadashi",
    date(2026, 10, 23): "Dashain — Trayodashi",

    date(2026, 11,  8): "Tihar — Laxmi Puja",
    date(2026, 11,  9): "Tihar — Gobardhan Puja",
    date(2026, 11, 10): "Tihar — Mha Puja / Newari New Year",
    date(2026, 11, 11): "Tihar — Bhai Tika",
    date(2026, 11, 12): "Tihar — Bhai Tika (extra day)",
    date(2026, 11, 15): "Chhath Parva",

    date(2026, 12, 24): "Udhauli Parva / Yomari Punhi",
    date(2026, 12, 25): "Christmas Day",
    date(2026, 12, 30): "Tamu Lhosar",

    # ── 2027 ──────────────────────────────────────────────────────────────────

    date(2027,  1, 11): "Prithvi Jayanti (National Unity Day)",
    date(2027,  1, 15): "Maghe Sankranti / Maghi Parva",
    date(2027,  1, 30): "Martyrs' Day",

    date(2027,  2,  7): "Sonam Lhosar",
    date(2027,  2, 19): "National Democracy Day",

    date(2027,  3,  6): "Mahashivaratri",
    date(2027,  3,  8): "International Women's Day",
    date(2027,  3,  9): "Gyalpo Lhosar",
    date(2027,  3, 21): "Holi / Fagu Purnima (Hilly Districts)",

    date(2027,  4,  6): "Ghode Jatra (Kathmandu Valley)",
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. LAYER 2 — SETTINGS TAB: Is_Holiday key
#
#    Key     : Is_Holiday
#    Value   : Comma-separated entries, each "YYYY-MM-DD:Reason"
#              e.g. "2026-06-15:Bandh,2026-11-20:Election day"
#    Written : by flag_adhoc_closure() via sheets.update_setting()
#    Read    : once at module import via sheets.get_setting()
#
#    Loaded ONCE at import. GitHub Actions = fresh container every 6 min
#    = effectively always current without any caching logic needed.
#
#    On any Sheets error → warning logged, gazette list used alone.
#    System NEVER blocks due to a Sheets timeout.
# ══════════════════════════════════════════════════════════════════════════════

IS_HOLIDAY_SETTING_KEY = "Is_Holiday"


def _parse_is_holiday_value(raw: str) -> dict[date, str]:
    """
    Parse the Is_Holiday SETTINGS value into a dict[date, str].

    Expected format: "YYYY-MM-DD:Reason,YYYY-MM-DD:Reason"
    Malformed entries are logged and skipped — never crash.
    """
    closures: dict[date, str] = {}
    if not raw or not raw.strip():
        return closures

    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            logger.warning(
                "calendar_guard: Is_Holiday entry missing ':' separator — "
                "skipped: '%s'", entry
            )
            continue
        date_part, _, reason = entry.partition(":")
        date_part = date_part.strip()
        reason    = reason.strip() or "Ad-hoc closure"
        try:
            d = date.fromisoformat(date_part)
            closures[d] = reason
        except ValueError:
            logger.warning(
                "calendar_guard: Is_Holiday invalid date '%s' — skipped",
                date_part
            )

    return closures


def _load_settings_closures() -> dict[date, str]:
    """
    Read Is_Holiday from SETTINGS tab via sheets.get_setting().
    Returns empty dict on any failure — never raises.
    Called once at module import.
    """
    try:
        from sheets import get_setting  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "calendar_guard: sheets.py not importable — "
            "gazette holidays only (Is_Holiday not loaded)"
        )
        return {}

    try:
        raw = get_setting(IS_HOLIDAY_SETTING_KEY, default="")
        if not raw:
            return {}
        closures = _parse_is_holiday_value(str(raw))
        if closures:
            logger.info(
                "calendar_guard: loaded %d ad-hoc closure(s) from "
                "SETTINGS[Is_Holiday]", len(closures)
            )
        return closures

    except Exception as exc:
        logger.warning(
            "calendar_guard: could not read SETTINGS[Is_Holiday] (%s) — "
            "gazette holidays only", exc
        )
        return {}


# Load at import — empty dict if Sheets unavailable
_SETTINGS_CLOSURES: dict[date, str] = _load_settings_closures()

# Final merged lookup. Settings closures override gazette on date conflict.
ALL_HOLIDAYS: dict[date, str] = {**_GAZETTE_HOLIDAYS, **_SETTINGS_CLOSURES}

GAZETTE_HOLIDAY_COUNT  = len(_GAZETTE_HOLIDAYS)
SETTINGS_CLOSURE_COUNT = len(_SETTINGS_CLOSURES)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FLAG AD-HOC CLOSURE — called by scraper / news / gemini
# ══════════════════════════════════════════════════════════════════════════════

def flag_adhoc_closure(
    closure_date: date,
    reason: str,
    source: str = "unknown",
) -> bool:
    """
    Append an ad-hoc NEPSE closure to SETTINGS[Is_Holiday].

    Called by any module that detects a closure announcement:
        scraper.py     — bandh in ShareSansar headlines
        geo_sentiment  — geopolitical disruption
        gemini_filter  — NEPSE official notice in news feed

    The entry is appended to the existing Is_Holiday value (never overwrites
    previous closures). Format stored: "YYYY-MM-DD:Reason [src:source]"

    The current process's ALL_HOLIDAYS is NOT updated — the next GitHub
    Actions run (fresh import, 6 min later) will pick it up automatically.
    Call reload_closures() if you need immediate effect in the same process.

    Returns True on success, False on failure (never raises).
    """
    try:
        from sheets import get_setting, update_setting  # noqa: PLC0415
    except ImportError:
        logger.error("flag_adhoc_closure: sheets.py not importable")
        return False

    try:
        new_entry  = f"{closure_date.isoformat()}:{reason} [src:{source}]"
        existing   = str(get_setting(IS_HOLIDAY_SETTING_KEY, default="") or "")
        updated    = f"{existing},{new_entry}" if existing.strip() else new_entry

        ok = update_setting(
            IS_HOLIDAY_SETTING_KEY,
            updated,
            description="Ad-hoc NEPSE closures. Format: YYYY-MM-DD:Reason [src:module]",
        )
        if ok:
            logger.info(
                "flag_adhoc_closure: added closure for %s — %s [src:%s]",
                closure_date, reason, source,
            )
        return bool(ok)

    except Exception as exc:
        logger.error("flag_adhoc_closure: failed — %s", exc)
        return False


def reload_closures() -> int:
    """
    Re-fetch Is_Holiday from SETTINGS and rebuild ALL_HOLIDAYS in-place.
    Not needed in normal GitHub Actions flow (each run is a fresh import).
    Useful for long-running local processes or tests.
    Returns number of settings closures loaded.
    """
    global _SETTINGS_CLOSURES, ALL_HOLIDAYS, SETTINGS_CLOSURE_COUNT
    _SETTINGS_CLOSURES  = _load_settings_closures()
    ALL_HOLIDAYS        = {**_GAZETTE_HOLIDAYS, **_SETTINGS_CLOSURES}
    SETTINGS_CLOSURE_COUNT = len(_SETTINGS_CLOSURES)
    return SETTINGS_CLOSURE_COUNT


# ══════════════════════════════════════════════════════════════════════════════
# 7. CORE LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_day(d: Optional[date] = None) -> bool:
    """Return True if `d` is a NEPSE trading day (not weekend, not holiday)."""
    if d is None:
        d = today_nst()
    if d.weekday() not in TRADING_DAYS:
        return False
    if d in ALL_HOLIDAYS:
        return False
    return True


def is_open(dt: Optional[datetime] = None) -> bool:
    """
    Return True if NEPSE is open at the given datetime (default: now NST).
    Primary boolean gate used by all GitHub Actions workflows.
    """
    if dt is None:
        dt = now_nst()
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=NST)
        else:
            dt = dt.astimezone(NST)

    if not is_trading_day(dt.date()):
        return False
    return MARKET_OPEN <= dt.time() < MARKET_CLOSE


def is_preopen(dt: Optional[datetime] = None) -> bool:
    """Return True if in pre-open window (10:30–10:45 AM NST, trading day only)."""
    if dt is None:
        dt = now_nst()
    else:
        dt = dt.astimezone(NST)
    if not is_trading_day(dt.date()):
        return False
    return PREOPEN_START <= dt.time() < PREOPEN_END


def is_eod_window(dt: Optional[datetime] = None) -> bool:
    """Return True if in EOD audit window (3:00–3:30 PM NST, trading day only)."""
    if dt is None:
        dt = now_nst()
    else:
        dt = dt.astimezone(NST)
    if not is_trading_day(dt.date()):
        return False
    return EOD_START <= dt.time() < EOD_END


def next_open_datetime() -> datetime:
    """
    Return the next market open datetime in NST.
    - Before open on a trading day → today's open.
    - Currently open or after close → next trading day's open.
    """
    dt = now_nst()
    d  = dt.date()
    t  = dt.time()

    if is_trading_day(d) and t < MARKET_OPEN:
        return datetime.combine(d, MARKET_OPEN, tzinfo=NST)

    candidate = d + timedelta(days=1)
    for _ in range(14):
        if is_trading_day(candidate):
            return datetime.combine(candidate, MARKET_OPEN, tzinfo=NST)
        candidate += timedelta(days=1)

    return datetime.combine(d + timedelta(days=14), MARKET_OPEN, tzinfo=NST)


# ══════════════════════════════════════════════════════════════════════════════
# 8. STATUS DICT
# ══════════════════════════════════════════════════════════════════════════════

def get_status(dt: Optional[datetime] = None) -> dict:
    """
    Return a full status dictionary describing the current market state.

    Keys
    ────
    is_open             bool   Primary gate
    is_preopen          bool   Morning briefing window (10:30–10:45 NST)
    is_eod_window       bool   EOD audit window (15:00–15:30 NST)
    is_trading_day      bool   Today is Sun–Thu and not a holiday
    is_holiday          bool   Today is a public/ad-hoc holiday
    holiday_name        str    Name of today's holiday, or ""
    holiday_source      str    "gazette" | "settings" | ""
    weekday_name        str    "Sunday", "Monday", …
    nst_time            str    "HH:MM:SS"
    nst_date            str    "YYYY-MM-DD"
    next_open           str    "YYYY-MM-DD HH:MM NST"
    minutes_to_open     int    0 if open now
    session             str    "OPEN" | "PRE_OPEN" | "EOD" | "CLOSED"
    settings_closures   int    Ad-hoc closures loaded from SETTINGS[Is_Holiday]
    message             str    Human-readable one-liner for logs / Telegram
    """
    if dt is None:
        dt = now_nst()
    else:
        dt = dt.astimezone(NST)

    d = dt.date()
    t = dt.time()

    _is_holiday   = d in ALL_HOLIDAYS
    _holiday_name = ALL_HOLIDAYS.get(d, "")
    _holiday_src  = (
        "settings" if d in _SETTINGS_CLOSURES else
        "gazette"  if d in _GAZETTE_HOLIDAYS  else
        ""
    )
    _is_tday   = is_trading_day(d)
    _is_open   = _is_tday and MARKET_OPEN <= t < MARKET_CLOSE
    _is_pre    = _is_tday and PREOPEN_START <= t < PREOPEN_END
    _is_eod    = _is_tday and EOD_START <= t < EOD_END
    _weekday   = WEEKDAY_NAMES[d.weekday()]
    _nst_time  = dt.strftime("%H:%M:%S")
    _nst_date  = d.isoformat()
    _next_open = next_open_datetime()
    _mins      = 0 if _is_open else max(0, int((_next_open - dt).total_seconds() // 60))

    if _is_open:
        session = "OPEN"
    elif _is_pre:
        session = "PRE_OPEN"
    elif _is_eod:
        session = "EOD"
    else:
        session = "CLOSED"

    if _is_open:
        msg = f"✅ NEPSE OPEN | {_weekday} {_nst_date} {_nst_time} NST"
    elif _is_holiday:
        src_tag = " [ad-hoc]" if _holiday_src == "settings" else ""
        msg = (
            f"🎉 Market CLOSED — Holiday{src_tag}: {_holiday_name} | "
            f"Next open: {_next_open.strftime('%Y-%m-%d %H:%M')} NST"
        )
    elif not _is_tday:
        msg = (
            f"📅 Market CLOSED — {_weekday} is a non-trading day | "
            f"Next open: {_next_open.strftime('%Y-%m-%d %H:%M')} NST"
        )
    elif t < MARKET_OPEN:
        msg = (
            f"⏳ Pre-market | {_weekday} {_nst_date} | "
            f"Opens 10:45 AM NST — {_mins} min away"
        )
    else:
        hrs, rem = divmod(_mins, 60)
        msg = (
            f"🔒 Market CLOSED for today | "
            f"Next open: {_next_open.strftime('%Y-%m-%d %H:%M')} NST "
            f"({hrs}h {rem}m away)"
        )

    return {
        "is_open":            _is_open,
        "is_preopen":         _is_pre,
        "is_eod_window":      _is_eod,
        "is_trading_day":     _is_tday,
        "is_holiday":         _is_holiday,
        "holiday_name":       _holiday_name,
        "holiday_source":     _holiday_src,
        "weekday_name":       _weekday,
        "nst_time":           _nst_time,
        "nst_date":           _nst_date,
        "next_open":          _next_open.strftime("%Y-%m-%d %H:%M NST"),
        "minutes_to_open":    _mins,
        "session":            session,
        "settings_closures":  SETTINGS_CLOSURE_COUNT,
        "message":            msg,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
#    python calendar_guard.py                     — live check
#    python calendar_guard.py 2026-10-20 14:00    — simulate datetime
#    Exit: 0 = market open, 1 = closed
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    if len(sys.argv) == 3:
        try:
            naive_dt = datetime.strptime(
                f"{sys.argv[1]} {sys.argv[2]}", "%Y-%m-%d %H:%M"
            )
            test_dt = naive_dt.replace(tzinfo=NST)
            s = get_status(test_dt)
            print(f"\n[TEST] Simulating: {sys.argv[1]} {sys.argv[2]} NST\n")
        except ValueError as e:
            print(f"Error: {e}\nUsage: python calendar_guard.py YYYY-MM-DD HH:MM")
            sys.exit(2)
    else:
        s = get_status()
        print("\n[LIVE] Current Nepal Standard Time\n")

    print(s["message"])
    print(
        f"\nHolidays loaded — gazette: {GAZETTE_HOLIDAY_COUNT}, "
        f"SETTINGS[Is_Holiday] (ad-hoc): {SETTINGS_CLOSURE_COUNT}"
    )
    print("\nFull status:")
    print(json.dumps(s, indent=2))
    sys.exit(0 if s["is_open"] else 1)
