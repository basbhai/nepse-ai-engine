"""
modules/atrad_scraper.py — ATrad TMS Live Market Data
Fully fixed numeric conversion + safe column handling.
"""

import os
import ast
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from sheets import write_row, upsert_row
from config import NST

load_dotenv()

log = logging.getLogger("atrad_scraper")

BASE_URL    = "https://tms.roadshowsecurities.com.np/atsweb"
USERNAME    = os.getenv("TMS_ROADSHOW_USER")
PASSWORD    = os.getenv("TMS_ROADSHOW_PASS")
WATCH_ID    = os.getenv("TMS_ROADSHOW_WATCH_ID", "8643")
BOOK_DEF_ID = "1"

_session      = None
_last_login   = None


def _get_headers() -> dict:
    return {
        "User-Agent"       : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept"           : "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With" : "XMLHttpRequest",
        "Referer"          : f"{BASE_URL}/login",
    }


def _parse(r: requests.Response) -> dict:
    text = r.text
    text = (text
            .replace("true",  "True")
            .replace("false", "False")
            .replace("null",  "None"))
    try:
        return ast.literal_eval(text)
    except Exception as e:
        log.error(f"Parse error: {e} | raw: {text[:200]}")
        return {}


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return default


def login() -> bool:
    global _session, _last_login
    s = requests.Session()
    s.headers.update(_get_headers())

    try:
        r = s.get(
            f"{BASE_URL}/login",
            params={
                "action"           : "login",
                "format"           : "json",
                "txtUserName"      : USERNAME,
                "txtPassword"      : PASSWORD,
                "dojo.preventCache": str(int(time.time() * 1000)),
            },
            timeout=15,
        )
        data = _parse(r)

        if data.get("code") == "0":
            _session    = s
            _last_login = datetime.now(NST)
            log.info(f"ATrad login success | watchID={data.get('watchID', WATCH_ID)} | broker={data.get('broker_code')}")
            return True
        else:
            log.error(f"ATrad login failed: {data}")
            return False
    except Exception as e:
        log.error(f"ATrad login exception: {e}")
        return False


def keepalive() -> bool:
    global _session
    if _session is None:
        return login()

    try:
        r = _session.get(
            f"{BASE_URL}/login",
            params={
                "action"           : "checkUserSession",
                "format"           : "json",
                "txtUserName"      : USERNAME,
                "dojo.preventCache": str(int(time.time() * 1000)),
            },
            timeout=10,
        )
        data = _parse(r)
        if data.get("code") == "0":
            return True
        else:
            log.warning("ATrad session expired — re-logging in")
            return login()
    except Exception as e:
        log.error(f"ATrad keepalive failed: {e} — attempting re-login")
        return login()


def _ensure_session() -> bool:
    if _session is None:
        return login()
    return True


# ====================== FIXED MARKET WATCH ======================
def fetch_market_watch(write_db: bool = True) -> pd.DataFrame:
    if not _ensure_session():
        log.error("ATrad fetch_market_watch: no session")
        return pd.DataFrame()

    now_nst = datetime.now(NST)

    try:
        r = _session.get(
            f"{BASE_URL}/watch",
            params={
                "action"           : "fullWatch",
                "format"           : "json",
                "exchange"         : "NEPSE",
                "bookDefId"        : BOOK_DEF_ID,
                "watchId"          : WATCH_ID,
                "lastUpdatedId"    : "0",
                "dojo.preventCache": str(int(time.time() * 1000)),
            },
            timeout=30,
        )
        data = _parse(r)

        if data.get("code") != "0":
            log.error(f"fullWatch failed: {data.get('description')}")
            return pd.DataFrame()

        records = data["data"]["watch"]
        if not records:
            log.warning("fullWatch returned empty watch list")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "ltp" in df.columns:
            df = df.drop(columns=["ltp"])

        # Rename columns
        col_map = {
            "security": "symbol", "tradeprice": "ltp", "openingprice": "open",
            "highpx": "high", "lowpx": "low", "vwap": "vwap", "avgprice": "avg_price",
            "bidprice": "bid_price", "bidqty": "bid_qty", "askprice": "ask_price",
            "askqty": "ask_qty", "totvolume": "volume", "totturnover": "turnover",
            "tottrades": "total_trades", "netchange": "net_change", "perchange": "pct_change",
            "lowdpr": "low_dpr", "highdpr": "high_dpr", "popdprlow": "prev_dpr_low",
            "popdprhigh": "prev_dpr_high", "maxalqty": "max_al_qty", "lasttradedtime": "time",
        }
        df = df.rename(columns=col_map).reset_index(drop=True)

        # ── SAFE NUMERIC CONVERSION (This was the bug) ─────────────────────
        num_cols = ["ltp", "open", "high", "low", "vwap", "avg_price",
                    "bid_price", "bid_qty", "ask_price", "ask_qty",
                    "volume", "total_trades", "net_change", "pct_change",
                    "low_dpr", "high_dpr", "prev_dpr_low", "prev_dpr_high"]

        for col in num_cols:
            if col in df.columns:
                vals = [str(x).replace(",", "") for x in df[col].tolist()]
                arr = pd.to_numeric(vals, errors="coerce")
                df[col] = np.where(np.isnan(arr), 0.0, arr)
            else:
                df[col] = 0.0

        df = df.loc[:, ~df.columns.duplicated()]

        # ── Derived signals ───────────────────────────────────────────────
        ltp      = df["ltp"].to_numpy()
        vwap     = df["vwap"].to_numpy()
        bid_qty  = df["bid_qty"].to_numpy()
        ask_qty  = df["ask_qty"].to_numpy()
        low_dpr  = df["low_dpr"].to_numpy()
        high_dpr = df["high_dpr"].to_numpy()

        vwap_dev = np.where(vwap > 0, (ltp - vwap) / vwap, 0.0).round(6)
        df.insert(len(df.columns), "vwap_dev", vwap_dev)

        total_depth   = bid_qty + ask_qty
        bid_ask_ratio = np.where(total_depth > 0, bid_qty / total_depth, 0.5).round(4)
        df.insert(len(df.columns), "bid_ask_ratio", bid_ask_ratio)

        dpr_range     = high_dpr - low_dpr
        dpr_proximity = np.where(dpr_range > 0, (ltp - low_dpr) / dpr_range, 0.5).round(4)
        df.insert(len(df.columns), "dpr_proximity", dpr_proximity)

        # Date & Time
        df["date"] = now_nst.strftime("%Y-%m-%d")
        if "time" not in df.columns or df["time"].eq("").all():
            df["time"] = now_nst.strftime("%H:%M:%S")

        if write_db and not df.empty:
            _write_market_watch(df, now_nst)

        log.info(f"✅ ATrad market watch: {len(df)} symbols fetched successfully")
        return df

    except Exception as e:
        log.error(f"fetch_market_watch exception: {e}", exc_info=True)
        return pd.DataFrame()


def _write_market_watch(df: pd.DataFrame, now_nst: datetime) -> None:
    keep_cols = [
        "date", "time", "symbol", "ltp", "open", "high", "low", "vwap", "avg_price",
        "bid_price", "bid_qty", "ask_price", "ask_qty", "volume", "turnover",
        "total_trades", "net_change", "pct_change", "low_dpr", "high_dpr",
        "prev_dpr_low", "prev_dpr_high", "max_al_qty", "vwap_dev",
        "bid_ask_ratio", "dpr_proximity",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    rows = df[existing].astype(str).to_dict(orient="records")

    fallback_time = now_nst.strftime("%H:%M:%S")
    for row in rows:
        if not row.get("time"):
            row["time"] = fallback_time
        try:
            upsert_row("atrad_market_watch", row, conflict_columns=["date", "time", "symbol"])
        except Exception as e:
            log.debug(f"Write skip {row.get('symbol')}: {e}")


# Keep the rest of the functions (fetch_order_book, fetch_trades, get_ltp, run) unchanged
# ... (they are already correct)

def fetch_order_book(symbol: str) -> dict:
    if not _ensure_session(): return {}
    try:
        r = _session.get(f"{BASE_URL}/marketdetails", params={
            "action": "getOrderBook", "format": "json", "security": symbol,
            "board": "1", "dojo.preventCache": str(int(time.time() * 1000))
        }, timeout=15)
        data = _parse(r)
        if data.get("code") != "0": return {}

        book = data["data"]["orderbook"][0]
        bids = book.get("bid", [])
        asks = book.get("ask", [])

        if bids and asks:
            bid_arr = np.array([[_safe_float(b.get("price",0)), _safe_float(b.get("qty",0))] for b in bids])
            ask_arr = np.array([[_safe_float(a.get("price",0)), _safe_float(a.get("qty",0))] for a in asks])
            tb = bid_arr[:,1].sum()
            ta = ask_arr[:,1].sum()
            td = tb + ta
            imb = tb / td if td > 0 else 0.5
            bb = bid_arr[0,0] if len(bid_arr)>0 else 0.0
            ba = ask_arr[0,0] if len(ask_arr)>0 else 0.0
            sp = ba - bb
            sp_pct = (sp / bb * 100) if bb > 0 else 0.0
        else:
            imb = bb = ba = sp = sp_pct = 0.0

        return {"symbol":symbol, "bids":bids, "asks":asks, "total_bid_qty":float(tb),"total_ask_qty":float(ta),
                "imbalance":round(imb,4), "best_bid":bb, "best_ask":ba, "spread":round(sp,2), "spread_pct":round(sp_pct,4),
                "fetched_at":datetime.now(NST).strftime("%H:%M:%S")}
    except Exception as e:
        log.error(f"fetch_order_book {symbol}: {e}")
        return {}

def get_ltp_live(symbol: str) -> Optional[dict]:
    """
    Fetch live price + market data for a single symbol via getQuickWatch.
    Used by execution_monitor (30-sec loop) — always fresh, no DB read.
    Returns dict with ltp, vwap, bid_price, bid_qty, ask_price, ask_qty,
    low_dpr, high_dpr, volume, or None on failure.
    """
    if not _ensure_session():
        return None
    try:
        r = _session.get(
            f"{BASE_URL}/watch",
            params={
                "action":                "getQuickWatch",
                "format":                "json",
                "exchange":              "NEPSE",
                "bookDefId":             "1",
                "securityid":            symbol,
                "watchId":               WATCH_ID,
                "isquickwatchsecurity":  "true",
                "lastUpdatedId":         "0",
                "dojo.preventCache":     str(int(time.time() * 1000)),
            },
            timeout=10,
        )
        data = _parse(r)
        if data.get("code") != "0":
            return None

        w = data.get("data", {}).get("watch", [])
        if not w:
            return None
        w = w[0]

        def _f(key):
            try:
                return float(str(w.get(key, "0")).replace(",", "") or 0)
            except (ValueError, TypeError):
                return 0.0

        return {
            "symbol":    symbol,
            "ltp":       _f("ltp"),
            "vwap":      _f("vwap"),
            "bid_price": _f("bidprice"),
            "bid_qty":   _f("bidqty"),
            "ask_price": _f("askprice"),
            "ask_qty":   _f("askqty"),
            "low_dpr":   _f("lowdpr"),
            "high_dpr":  _f("highdpr"),
            "volume":    _f("totvolume"),
            "vwap_dev":  (_f("ltp") - _f("vwap")) / _f("vwap") if _f("vwap") > 0 else 0.0,
        }
    except Exception as e:
        log.error("get_ltp_live(%s): %s", symbol, e)
        return None

def run():
    if not login():
        log.error("ATrad login failed")
        return None
    return fetch_market_watch(write_db=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    df = run()
    if df is not None and not df.empty:
        print(f"\n✅ Fetched {len(df)} symbols")
        print(df[["symbol", "ltp", "vwap", "vwap_dev", "bid_ask_ratio", "dpr_proximity"]].head(10).to_string())