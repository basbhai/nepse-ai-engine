"""
test_atrad_endpoints.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Standalone test — no DB, no imports from nepse-engine.

Tests 3 ATrad endpoints using the existing session:
  1. fullWatch       — current (542 symbols, adv/dec computed manually)
  2. getTradeStats   — NEW (authoritative adv/dec/unchanged from exchange)
  3. getSectorDataAll — NEW (NEPSE index, sensitive index, turnover, volume)

Run from anywhere on Ubuntu:
    cd ~/nepse-engine
    python test_atrad_endpoints.py

Reads credentials from .env in current directory.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import ast
import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("atrad_test")

# ── Credentials ───────────────────────────────────────────────────────────────
BASE_URL    = "https://tms.roadshowsecurities.com.np/atsweb"
USERNAME    = os.getenv("TMS_ROADSHOW_USER")
PASSWORD    = os.getenv("TMS_ROADSHOW_PASS")
WATCH_ID    = os.getenv("TMS_ROADSHOW_WATCH_ID", "8643")
BOOK_DEF_ID = "1"

if not USERNAME or not PASSWORD:
    raise SystemExit("❌  TMS_ROADSHOW_USER / TMS_ROADSHOW_PASS not set in .env")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {
        "User-Agent"       : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept"           : "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With" : "XMLHttpRequest",
        "Referer"          : f"{BASE_URL}/login",
    }


def _parse(r: requests.Response) -> dict:
    """ATrad responses use Python literals, not JSON."""
    text = (r.text.strip()
            .replace("true",  "True")
            .replace("false", "False")
            .replace("null",  "None"))
    try:
        return ast.literal_eval(text)
    except Exception as e:
        log.error("Parse error: %s | raw: %s", e, r.text[:300])
        return {}


def _sf(val, default=0.0) -> float:
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return default


def _si(val, default=0) -> int:
    try:
        return int(str(val).replace(",", ""))
    except Exception:
        return default


def _ts() -> str:
    return str(int(time.time() * 1000))


# ── Step 1: Login ─────────────────────────────────────────────────────────────
def login() -> requests.Session:
    print("\n" + "═" * 60)
    print("  STEP 1 — LOGIN")
    print("═" * 60)

    s = requests.Session()
    s.headers.update(_headers())

    r = s.get(
        f"{BASE_URL}/login",
        params={
            "action"           : "login",
            "format"           : "json",
            "txtUserName"      : USERNAME,
            "txtPassword"      : PASSWORD,
            "dojo.preventCache": _ts(),
        },
        timeout=15,
    )
    data = _parse(r)

    print(f"  HTTP status : {r.status_code}")
    print(f"  code        : {data.get('code')}")
    print(f"  description : {data.get('description')}")
    print(f"  broker_code : {data.get('broker_code')}")
    print(f"  watchID     : {data.get('watchID')}")

    if data.get("code") != "0":
        raise SystemExit(f"❌  Login failed: {data}")

    print("  ✅  Login successful")
    return s


# ── Step 2: getTradeStats (NEW) ───────────────────────────────────────────────
def test_trade_stats(s: requests.Session):
    print("\n" + "═" * 60)
    print("  STEP 2 — getTradeStats  (NEW — authoritative adv/dec)")
    print("  URL: /marketdetails?action=getTradeStats")
    print("═" * 60)

    r = s.get(
        f"{BASE_URL}/marketdetails",
        params={
            "action"           : "getTradeStats",
            "format"           : "json",
            "dojo.preventCache": _ts(),
        },
        timeout=10,
    )
    data = _parse(r)

    print(f"  HTTP status : {r.status_code}")
    print(f"  Raw response: {r.text[:500]}")
    print()

    if data.get("code") == "0":
        d = data.get("data", {})
        print(f"  ✅  Parsed:")
        print(f"      advancing  (up)       : {d.get('up')}")
        print(f"      declining  (down)     : {d.get('down')}")
        print(f"      unchanged             : {d.get('unchanged')}")
        print(f"      total                 : {d.get('total')}")

        # Compare what our manual count would give
        up  = _si(d.get("up",        0))
        dn  = _si(d.get("down",      0))
        unc = _si(d.get("unchanged", 0))
        tot = _si(d.get("total",     0))
        if tot > 0:
            score = round((up - dn) / tot * 100, 2)
            print(f"\n      → breadth_score (our formula) : {score}%")
    else:
        print(f"  ❌  Failed: {data}")

    return data


# ── Step 3: getSectorDataAll (NEW) ────────────────────────────────────────────
def test_sector_data(s: requests.Session):
    print("\n" + "═" * 60)
    print("  STEP 3 — getSectorDataAll  (NEW — NEPSE index + market totals)")
    print("  URL: /sector?action=getSectorDataAll")
    print("═" * 60)

    r = s.get(
        f"{BASE_URL}/sector",
        params={
            "action"           : "getSectorDataAll",
            "format"           : "json",
            "exchange"         : "NEPSE",
            "sectorId"         : "NEPSE",
            "sectorIdSL"       : "SENSIND",
            "dojo.preventCache": _ts(),
        },
        timeout=10,
    )
    data = _parse(r)

    print(f"  HTTP status : {r.status_code}")
    print(f"  Raw response: {r.text[:500]}")
    print()

    if data.get("code") == "0":
        sectors = data.get("data", {}).get("sector", [])
        if sectors:
            d = sectors[0]
            print(f"  ✅  Parsed (first/only row):")
            print(f"      pr1  NEPSE index        : {d.get('pr1')}  → {_sf(d.get('pr1')):.2f}")
            print(f"      pr2  Sensitive index    : {d.get('pr2')}  → {_sf(d.get('pr2')):.2f}")
            print(f"      n1   index change (abs) : {d.get('n1')}   → {_sf(d.get('n1')):.2f}")
            print(f"      p1   index change (%)   : {d.get('p1')}   → {_sf(d.get('p1')):.2f}%")
            print(f"      n2   sensitive chg (abs): {d.get('n2')}")
            print(f"      p2   sensitive chg (%)  : {d.get('p2')}")
            print(f"      v    total volume        : {d.get('v')}    → {_si(d.get('v')):,}")
            print(f"      to   total turnover NPR  : {d.get('to')}")
            print(f"      tr   total trades        : {d.get('tr')}   → {_si(d.get('tr')):,}")
            print(f"      marketCashIn            : {d.get('marketCashIn')!r}")
            print(f"      marketCashInVal         : {d.get('marketCashInVal')!r}")
            print(f"      marketCashOutVal        : {d.get('marketCashOutVal')!r}")
            print()
            print(f"  All keys in response: {list(d.keys())}")
        else:
            print("  ⚠️   sector list is empty")
    else:
        print(f"  ❌  Failed: {data}")

    return data


# ── Step 4: fullWatch (EXISTING) ──────────────────────────────────────────────
def test_full_watch(s: requests.Session):
    print("\n" + "═" * 60)
    print("  STEP 4 — fullWatch  (EXISTING — 542 symbols)")
    print("  URL: /watch?action=fullWatch")
    print("═" * 60)

    r = s.get(
        f"{BASE_URL}/watch",
        params={
            "action"           : "fullWatch",
            "format"           : "json",
            "exchange"         : "NEPSE",
            "bookDefId"        : BOOK_DEF_ID,
            "watchId"          : WATCH_ID,
            "lastUpdatedId"    : "0",
            "dojo.preventCache": _ts(),
        },
        timeout=30,
    )
    data = _parse(r)

    print(f"  HTTP status  : {r.status_code}")
    print(f"  code         : {data.get('code')}")

    if data.get("code") == "0":
        records = data.get("data", {}).get("watch", [])
        print(f"  ✅  Total symbols returned: {len(records)}")

        if records:
            # Manual adv/dec count (current approach)
            adv = sum(1 for rec in records if _sf(rec.get("perchange", 0)) > 0)
            dec = sum(1 for rec in records if _sf(rec.get("perchange", 0)) < 0)
            unc = sum(1 for rec in records if _sf(rec.get("perchange", 0)) == 0)
            tot = len(records)
            score = round((adv - dec) / tot * 100, 2) if tot > 0 else 0.0

            print(f"\n  Manually computed from perchange field:")
            print(f"      advancing : {adv}")
            print(f"      declining : {dec}")
            print(f"      unchanged : {unc}")
            print(f"      total     : {tot}")
            print(f"      breadth_score : {score}%")

            print(f"\n  First symbol sample:")
            rec = records[0]
            print(f"      keys available: {list(rec.keys())}")
            print(f"      symbol    : {rec.get('security')}")
            print(f"      ltp       : {rec.get('tradeprice')}")
            print(f"      perchange : {rec.get('perchange')}")
            print(f"      totvolume : {rec.get('totvolume')}")
    else:
        print(f"  ❌  Failed: {data}")

    return data


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  ATrad Endpoint Test — NEPSE AI Engine")
    print("█" * 60)

    session = login()

    stats  = test_trade_stats(session)
    sector = test_sector_data(session)
    watch  = test_full_watch(session)

    print("\n" + "═" * 60)
    print("  SUMMARY — adv/dec comparison")
    print("═" * 60)

    # getTradeStats numbers
    if stats.get("code") == "0":
        d = stats.get("data", {})
        print(f"  getTradeStats  → up={d.get('up')}  down={d.get('down')}  unch={d.get('unchanged')}  total={d.get('total')}")

    # fullWatch manual count
    if watch.get("code") == "0":
        records = watch.get("data", {}).get("watch", [])
        adv = sum(1 for rec in records if _sf(rec.get("perchange", 0)) > 0)
        dec = sum(1 for rec in records if _sf(rec.get("perchange", 0)) < 0)
        unc = sum(1 for rec in records if _sf(rec.get("perchange", 0)) == 0)
        print(f"  fullWatch calc → up={adv}  down={dec}  unch={unc}  total={len(records)}")

    # NEPSE index
    if sector.get("code") == "0":
        sectors = sector.get("data", {}).get("sector", [])
        if sectors:
            d = sectors[0]
            print(f"  NEPSE index    → {_sf(d.get('pr1')):.2f}  ({_sf(d.get('p1')):+.2f}%)")

    print("\n  Done.\n")