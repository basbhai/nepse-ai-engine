"""
modules/broker_flow_scraper.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Smart Money Flow + Holdings Tracker
Source : https://sharehubnepal.com (JWT auth, 30-min token expiry)
Purpose: Fetch broker accumulation/distribution + aggressive holdings at
         9 PM NST, store in DB, send Telegram (all users) + Email (detailed).

API calls (4 total, one login):
  1. broker-accumulation?duration=1D     → who bought today
  2. broker-distribution?duration=1D     → who sold today
  3. broker-accumulation?duration=1W     → weekly buying trend
  4. broker-aggressive-holdings?EquityOnly=true → concentrated holdings snapshot

Tables written:
  broker_flow     — daily acc/dist flow per symbol
  broker_holdings — concentrated holdings snapshot (~58 symbols)

No backfill — runs once daily at 9 PM NST via summary_workflow.

─────────────────────────────────────────────────────────────────────────────
Run:
    python -m modules.broker_flow_scraper           # full run
    python -m modules.broker_flow_scraper --dry-run # fetch + print, no DB/notify
─────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL     = "https://sharehubnepal.com"
LOGIN_URL    = f"{BASE_URL}/account/api/v1/auth/login/email"
ACC_URL      = f"{BASE_URL}/data/api/v1/floorsheet-analysis/broker-accumulation"
DIST_URL     = f"{BASE_URL}/data/api/v1/floorsheet-analysis/broker-distribution"
HOLDINGS_URL = f"{BASE_URL}/data/api/v1/floorsheet-analysis/broker-aggressive-holdings"

SH_EMAIL = os.getenv("SHAREHUB_EMAIL", "basbhai2026@gmail.com")
SH_PASS  = os.getenv("SHAREHUB_PASSWORD", "Mahanatma@021")

NST   = timezone(timedelta(hours=5, minutes=45))
TOP_N = 3

HEADERS = {
    "Content-Type": "application/json",
    "referer":      "https://sharehubnepal.com",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _login() -> str:
    """Login to Sharehub, return JWT accessToken. Raises on failure."""
    log.info("Sharehub: logging in as %s", SH_EMAIL)
    r = requests.post(
        LOGIN_URL,
        json={"email": SH_EMAIL, "password": SH_PASS},
        headers=HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    token = r.json()["data"]["accessToken"]
    log.info("Sharehub: login OK")
    return token


def _auth_headers(token: str) -> dict:
    return {**HEADERS, "Authorization": f"Bearer {token}"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(url: str, params: dict, token: str) -> list:
    r = requests.get(url, params=params, headers=_auth_headers(token), timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise ValueError(f"Sharehub API error: {data.get('message', 'unknown')}")
    content = data["data"]["content"]
    log.info("  %-50s → %d rows", f"{url.split('/')[-1]} {params}", len(content))
    return content


def fetch_all(token: str) -> tuple[list, list, list, list]:
    """
    Fetch all 4 endpoints sequentially.
    All complete well within 30-min token window.
    Returns (acc_1d, dist_1d, acc_1w, holdings).
    """
    log.info("Fetching 4 broker endpoints...")
    acc_1d   = _fetch(ACC_URL,      {"duration": "1D"},     token)
    dist_1d  = _fetch(DIST_URL,     {"duration": "1D"},     token)
    acc_1w   = _fetch(ACC_URL,      {"duration": "1W"},     token)
    holdings = _fetch(HOLDINGS_URL, {"EquityOnly": "true"}, token)
    return acc_1d, dist_1d, acc_1w, holdings


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AGGREGATE FLOW PER SYMBOL
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_flow(rows: list) -> dict:
    """
    Aggregate broker-level rows into per-symbol totals.
    Returns {symbol: {name, total_amount, total_qty, broker_count,
                       top_broker, top_broker_pct, brokers_json}}
    """
    by_symbol = defaultdict(list)
    for row in rows:
        sym = row.get("symbol", "")
        if sym:
            by_symbol[sym].append(row)

    result = {}
    for sym, broker_rows in by_symbol.items():
        total_amount = sum(r.get("accumulationAmount", 0) for r in broker_rows)
        total_qty    = sum(r.get("accumulationQuantity", 0) for r in broker_rows)
        broker_count = len(broker_rows)

        top = max(broker_rows, key=lambda r: r.get("accumulationAmount", 0))
        top_broker_pct = round(
            top.get("accumulationAmount", 0) / total_amount * 100, 2
        ) if total_amount > 0 else 0

        brokers_json = json.dumps([
            {
                "broker":   r.get("brokerName", ""),
                "amount":   round(r.get("accumulationAmount", 0), 2),
                "qty":      r.get("accumulationQuantity", 0),
                "acc_pct":  round(r.get("accumulationPercentage", 0), 2),
                "avg_rate": round(r.get("averageRate", 0), 2),
            }
            for r in sorted(broker_rows,
                            key=lambda x: x.get("accumulationAmount", 0),
                            reverse=True)
        ])

        result[sym] = {
            "name":           broker_rows[0].get("name", sym),
            "total_amount":   round(total_amount, 2),
            "total_qty":      total_qty,
            "broker_count":   broker_count,
            "top_broker":     top.get("brokerName", ""),
            "top_broker_pct": top_broker_pct,
            "brokers_json":   brokers_json,
        }
    return result


def build_flow_records(
    acc_1d: list, dist_1d: list, acc_1w: list, trade_date: str
) -> list[dict]:
    """Merge acc/dist data into per-symbol broker_flow rows."""
    agg_acc_1d  = _aggregate_flow(acc_1d)
    agg_dist_1d = _aggregate_flow(dist_1d)
    agg_acc_1w  = _aggregate_flow(acc_1w)

    all_symbols = set(agg_acc_1d.keys()) | set(agg_dist_1d.keys())
    records = []

    for sym in all_symbols:
        a1d = agg_acc_1d.get(sym, {})
        d1d = agg_dist_1d.get(sym, {})
        a1w = agg_acc_1w.get(sym, {})

        acc_amt  = a1d.get("total_amount", 0)
        dist_amt = d1d.get("total_amount", 0)
        net_flow = round(acc_amt - dist_amt, 2)

        if net_flow > 0:
            bias = "ACCUMULATION"
        elif net_flow < 0:
            bias = "DISTRIBUTION"
        else:
            bias = "NEUTRAL"

        acc_amt_1w = a1w.get("total_amount", 0)

        records.append({
            "date":   trade_date,
            "symbol": sym,
            "name":   a1d.get("name") or d1d.get("name") or sym,

            "acc_broker_count_1d":   str(a1d.get("broker_count", 0)),
            "acc_amount_1d":         str(acc_amt),
            "acc_qty_1d":            str(a1d.get("total_qty", 0)),
            "acc_top_broker_1d":     a1d.get("top_broker", ""),
            "acc_top_broker_pct_1d": str(a1d.get("top_broker_pct", 0)),

            "dist_broker_count_1d":   str(d1d.get("broker_count", 0)),
            "dist_amount_1d":         str(dist_amt),
            "dist_qty_1d":            str(d1d.get("total_qty", 0)),
            "dist_top_broker_1d":     d1d.get("top_broker", ""),
            "dist_top_broker_pct_1d": str(d1d.get("top_broker_pct", 0)),

            "net_flow_1d":  str(net_flow),
            "flow_bias_1d": bias,

            "acc_broker_count_1w":   str(a1w.get("broker_count", 0)),
            "acc_amount_1w":         str(acc_amt_1w),
            "acc_qty_1w":            str(a1w.get("total_qty", 0)),
            "acc_top_broker_1w":     a1w.get("top_broker", ""),
            "acc_top_broker_pct_1w": str(a1w.get("top_broker_pct", 0)),

            "dist_broker_count_1w":   "0",
            "dist_amount_1w":         "0",
            "dist_qty_1w":            "0",
            "dist_top_broker_1w":     "",
            "dist_top_broker_pct_1w": "0",

            "net_flow_1w":  str(round(acc_amt_1w, 2)),
            "flow_bias_1w": "ACCUMULATION" if acc_amt_1w > 0 else "NEUTRAL",

            "acc_brokers_1d_json":  a1d.get("brokers_json", "[]"),
            "dist_brokers_1d_json": d1d.get("brokers_json", "[]"),
            "acc_brokers_1w_json":  a1w.get("brokers_json", "[]"),

            "created_at": datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"),
        })

    log.info("Built %d broker_flow records", len(records))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD HOLDINGS RECORDS
# ══════════════════════════════════════════════════════════════════════════════

def build_holdings_records(holdings: list, trade_date: str) -> list[dict]:
    """Convert raw holdings API response into broker_holdings rows."""
    records = []
    for h in holdings:
        sym        = h.get("symbol", "")
        total_qty  = h.get("totalQuantity", 0)
        hold_qty   = h.get("holdQuantity", 0)
        hold_pct   = round(hold_qty / total_qty * 100, 2) if total_qty > 0 else 0
        top3_pct   = h.get("topThreeBrokersHoldingPercentage", 0)
        public_pct = h.get("publicTradePercentage", 0)

        # Stealth score: high concentration + low public leakage = smart money
        stealth_score = round(top3_pct * (1 - public_pct / 100), 2)

        top_brokers = h.get("topBrokers", [])
        b1 = top_brokers[0] if len(top_brokers) > 0 else {}
        b2 = top_brokers[1] if len(top_brokers) > 1 else {}
        b3 = top_brokers[2] if len(top_brokers) > 2 else {}

        records.append({
            "date":   trade_date,
            "symbol": sym,
            "name":   h.get("name", sym),

            "total_involved_brokers": str(h.get("totalInvolvedBrokers", 0)),
            "top3_holding_pct":       str(top3_pct),
            "total_qty":              str(total_qty),
            "hold_qty":               str(hold_qty),
            "hold_pct":               str(hold_pct),
            "public_trade_pct":       str(public_pct),
            "stealth_score":          str(stealth_score),

            "ltp":        str(h.get("ltp", 0)),
            "change":     str(h.get("change", 0)),
            "change_pct": str(h.get("changePercentage", 0)),

            "top_broker_1_name": b1.get("name", ""),
            "top_broker_1_code": b1.get("code", ""),
            "top_broker_1_hold": str(b1.get("holdQuantity", 0)),
            "top_broker_1_pct":  str(b1.get("holdingPercentage", 0)),

            "top_broker_2_name": b2.get("name", ""),
            "top_broker_2_code": b2.get("code", ""),
            "top_broker_2_hold": str(b2.get("holdQuantity", 0)),
            "top_broker_2_pct":  str(b2.get("holdingPercentage", 0)),

            "top_broker_3_name": b3.get("name", ""),
            "top_broker_3_code": b3.get("code", ""),
            "top_broker_3_hold": str(b3.get("holdQuantity", 0)),
            "top_broker_3_pct":  str(b3.get("holdingPercentage", 0)),

            "top_brokers_json": json.dumps([
                {
                    "code":              b.get("code", ""),
                    "name":              b.get("name", ""),
                    "totalBuyQuantity":  b.get("totalBuyQuantity", 0),
                    "holdQuantity":      b.get("holdQuantity", 0),
                    "holdingPercentage": b.get("holdingPercentage", 0),
                }
                for b in top_brokers
            ]),

            "created_at": datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"),
        })

    log.info("Built %d broker_holdings records", len(records))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DB WRITE
# ══════════════════════════════════════════════════════════════════════════════

def save_flow_records(records: list[dict]) -> int:
    from sheets import upsert_row
    written = 0
    for rec in records:
        try:
            upsert_row("broker_flow", rec, conflict_columns=["symbol", "date"])
            written += 1
        except Exception as e:
            log.error("broker_flow upsert failed %s: %s", rec.get("symbol"), e)
    log.info("broker_flow: %d/%d saved", written, len(records))
    return written


def save_holdings_records(records: list[dict]) -> int:
    from sheets import upsert_row
    written = 0
    for rec in records:
        try:
            upsert_row("broker_holdings", rec, conflict_columns=["symbol", "date"])
            written += 1
        except Exception as e:
            log.error("broker_holdings upsert failed %s: %s", rec.get("symbol"), e)
    log.info("broker_holdings: %d/%d saved", written, len(records))
    return written


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cr(amount: float) -> str:
    """Format NPR as Cr/L."""
    if amount >= 1_00_00_000:
        return f"{amount / 1_00_00_000:.1f}Cr"
    elif amount >= 1_00_000:
        return f"{amount / 1_00_000:.1f}L"
    return f"{amount:,.0f}"


def _top_flow_by_broker_count(records: list[dict], bias: str, n: int = TOP_N) -> list[dict]:
    if bias == "ACCUMULATION":
        filtered = [r for r in records if float(r["acc_amount_1d"]) > 0]
        return sorted(filtered, key=lambda r: int(r["acc_broker_count_1d"]), reverse=True)[:n]
    else:
        filtered = [r for r in records if float(r["dist_amount_1d"]) > 0]
        return sorted(filtered, key=lambda r: int(r["dist_broker_count_1d"]), reverse=True)[:n]


def _top_flow_by_volume(records: list[dict], bias: str, n: int = TOP_N) -> list[dict]:
    if bias == "ACCUMULATION":
        filtered = [r for r in records if float(r["acc_amount_1d"]) > 0]
        return sorted(filtered, key=lambda r: float(r["acc_amount_1d"]), reverse=True)[:n]
    else:
        filtered = [r for r in records if float(r["dist_amount_1d"]) > 0]
        return sorted(filtered, key=lambda r: float(r["dist_amount_1d"]), reverse=True)[:n]


def _top_holdings_by_stealth(records: list[dict], n: int = TOP_N) -> list[dict]:
    return sorted(records, key=lambda r: float(r["stealth_score"]), reverse=True)[:n]


def _broker_names_short(brokers_json: str, n: int = 5) -> str:
    try:
        brokers = json.loads(brokers_json)
        names = [b["broker"].split(" ")[0] for b in brokers[:n]]
        if len(brokers) > n:
            names.append(f"+{len(brokers)-n}")
        return ", ".join(names)
    except Exception:
        return ""


def _broker_names_full(brokers_json: str) -> str:
    try:
        brokers = json.loads(brokers_json)
        return "\n".join(
            f"  • {b['broker']}: NPR {_cr(b['amount'])} | "
            f"{b['qty']:,} shares | {b['acc_pct']}% | avg {b['avg_rate']:.1f}"
            for b in brokers
        )
    except Exception:
        return ""


def _holdings_broker_detail(top_brokers_json: str) -> str:
    try:
        brokers = json.loads(top_brokers_json)
        return "\n".join(
            f"  • [{b['code']}] {b['name']}: "
            f"hold {b['holdQuantity']:,} ({b['holdingPercentage']}%) | "
            f"bought {b['totalBuyQuantity']:,}"
            for b in brokers
        )
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TELEGRAM MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

def build_telegram_message(
    flow_records: list[dict],
    holdings_records: list[dict],
    trade_date: str,
) -> str:
    sep = "▬" * 22
    lines = [sep, f"🏦 *Smart Money Flow — {trade_date}*", sep, ""]

    # ── Accumulation by broker count ────────────────────────────────────────
    lines.append("🟢 *Accumulation — Top Broker Count*")
    for i, r in enumerate(_top_flow_by_broker_count(flow_records, "ACCUMULATION"), 1):
        names = _broker_names_short(r["acc_brokers_1d_json"])
        lines.append(
            f"{i}. *{r['symbol']}* — {r['acc_broker_count_1d']} brokers | "
            f"NPR {_cr(float(r['acc_amount_1d']))}"
        )
        if names:
            lines.append(f"   _{names}_")
    lines.append("")

    # ── Accumulation by volume ───────────────────────────────────────────────
    lines.append("🟢 *Accumulation — Top Volume*")
    for i, r in enumerate(_top_flow_by_volume(flow_records, "ACCUMULATION"), 1):
        names = _broker_names_short(r["acc_brokers_1d_json"])
        lines.append(
            f"{i}. *{r['symbol']}* — NPR {_cr(float(r['acc_amount_1d']))} | "
            f"{r['acc_broker_count_1d']} brokers"
        )
        if names:
            lines.append(f"   _{names}_")
    lines.append("")

    # ── Stealth holdings ─────────────────────────────────────────────────────
    if holdings_records:
        lines.append("🔵 *Stealth — Highest Broker Concentration*")
        for i, h in enumerate(_top_holdings_by_stealth(holdings_records), 1):
            b1 = h["top_broker_1_name"].split(" ")[0]
            b2 = h["top_broker_2_name"].split(" ")[0]
            b3 = h["top_broker_3_name"].split(" ")[0]
            lines.append(
                f"{i}. *{h['symbol']}* — top3: {h['top3_holding_pct']}% | "
                f"public: {h['public_trade_pct']}% | score: {h['stealth_score']}"
            )
            lines.append(
                f"   _{b1}({h['top_broker_1_pct']}%), "
                f"{b2}({h['top_broker_2_pct']}%), "
                f"{b3}({h['top_broker_3_pct']}%)_"
            )
        lines.append("")

    lines += [sep, f"🕒 _{datetime.now(tz=NST).strftime('%H:%M NST')}_"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — EMAIL BODY
# ══════════════════════════════════════════════════════════════════════════════

def build_email_body(
    flow_records: list[dict],
    holdings_records: list[dict],
    trade_date: str,
) -> str:
    sep  = "=" * 60
    sep2 = "─" * 60
    now  = datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M NST")

    lines = [sep, f"NEPSE SMART MONEY FLOW — {trade_date}",
             f"Generated: {now}", sep, ""]

    # Flow sections
    for title, top_recs, json_key, amount_key, count_key in [
        ("ACCUMULATION — TOP BY BROKER COUNT",
         _top_flow_by_broker_count(flow_records, "ACCUMULATION"),
         "acc_brokers_1d_json", "acc_amount_1d", "acc_broker_count_1d"),
        ("ACCUMULATION — TOP BY VOLUME",
         _top_flow_by_volume(flow_records, "ACCUMULATION"),
         "acc_brokers_1d_json", "acc_amount_1d", "acc_broker_count_1d"),
        ("DISTRIBUTION — TOP BY BROKER COUNT",
         _top_flow_by_broker_count(flow_records, "DISTRIBUTION"),
         "dist_brokers_1d_json", "dist_amount_1d", "dist_broker_count_1d"),
        ("DISTRIBUTION — TOP BY VOLUME",
         _top_flow_by_volume(flow_records, "DISTRIBUTION"),
         "dist_brokers_1d_json", "dist_amount_1d", "dist_broker_count_1d"),
    ]:
        lines += [sep2, f"  {title}", sep2]
        for i, r in enumerate(top_recs, 1):
            lines.append(
                f"\n{i}. {r['symbol']} — {r['name']}\n"
                f"   Total: NPR {float(r[amount_key]):,.0f} | {r[count_key]} brokers"
            )
            detail = _broker_names_full(r[json_key])
            if detail:
                lines.append(detail)
        lines.append("")

    # Holdings section
    if holdings_records:
        lines += [sep, "STEALTH ACCUMULATION — AGGRESSIVE HOLDINGS SNAPSHOT", sep]
        lines.append(f"Symbols with concentrated holdings today: {len(holdings_records)}\n")
        for h in _top_holdings_by_stealth(holdings_records, n=10):
            lines.append(
                f"\n{h['symbol']} — {h['name']}\n"
                f"  LTP: NPR {h['ltp']} ({h['change_pct']}%) | "
                f"Stealth Score: {h['stealth_score']}\n"
                f"  Top 3 holding: {h['top3_holding_pct']}% of {int(h['total_qty']):,} shares\n"
                f"  Still held: {int(h['hold_qty']):,} ({h['hold_pct']}%) | "
                f"Public leaked: {h['public_trade_pct']}%\n"
                f"  Total brokers involved: {h['total_involved_brokers']}"
            )
            detail = _holdings_broker_detail(h["top_brokers_json"])
            if detail:
                lines.append(detail)
        lines.append("")

    # Weekly context
    lines += [sep, "1W ACCUMULATION — TOP 5", sep]
    for r in sorted(flow_records,
                    key=lambda r: float(r["acc_amount_1w"]),
                    reverse=True)[:5]:
        lines.append(
            f"\n{r['symbol']} — {r['name']}\n"
            f"  Week total: NPR {float(r['acc_amount_1w']):,.0f} | "
            f"{r['acc_broker_count_1w']} brokers"
        )
        detail = _broker_names_full(r["acc_brokers_1w_json"])
        if detail:
            lines.append(detail)

    lines += ["", sep,
              f"Flow symbols: {len(flow_records)} | "
              f"Holdings symbols: {len(holdings_records)}", sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════════════════

def send_notifications(
    flow_records: list[dict],
    holdings_records: list[dict],
    trade_date: str,
) -> None:
    from helper.notifier import send_telegram, send_email

    tg_msg = build_telegram_message(flow_records, holdings_records, trade_date)
    tg_ok  = send_telegram(tg_msg, parse_mode="Markdown")
    log.info("Telegram broker flow: %s", "sent" if tg_ok else "failed")

    email_body = build_email_body(flow_records, holdings_records, trade_date)
    email_ok   = send_email(
        subject=f"NEPSE Smart Money Flow — {trade_date}",
        body=email_body,
    )
    log.info("Email broker flow: %s", "sent" if email_ok else "failed/not configured")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SHEETS HELPERS (for claude_analyst)
# ══════════════════════════════════════════════════════════════════════════════

def get_broker_flow(symbol: str, trade_date: Optional[str] = None) -> Optional[dict]:
    """
    Get broker_flow row for a symbol on a given date (default: today).
    Called by claude_analyst._load_context(). Fails silently → returns None.
    """
    try:
        from sheets import read_tab
        target = trade_date or date.today().strftime("%Y-%m-%d")
        for r in read_tab("broker_flow"):
            if r.get("symbol") == symbol and r.get("date") == target:
                return r
        return None
    except Exception as e:
        log.debug("get_broker_flow(%s): %s", symbol, e)
        return None


def get_broker_holdings(symbol: str, trade_date: Optional[str] = None) -> Optional[dict]:
    """
    Get broker_holdings row for a symbol on a given date (default: today).
    Called by claude_analyst._load_context(). Fails silently → returns None.
    """
    try:
        from sheets import read_tab
        target = trade_date or date.today().strftime("%Y-%m-%d")
        for r in read_tab("broker_holdings"):
            if r.get("symbol") == symbol and r.get("date") == target:
                return r
        return None
    except Exception as e:
        log.debug("get_broker_holdings(%s): %s", symbol, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> bool:
    """
    Full pipeline: login → fetch → build → save → notify.
    Returns True on success. Called by summary_workflow.py.
    """
    trade_date = date.today().strftime("%Y-%m-%d")
    log.info("broker_flow_scraper: %s (dry_run=%s)", trade_date, dry_run)

    try:
        token = _login()
    except Exception as e:
        log.error("Sharehub login failed: %s", e)
        return False

    try:
        acc_1d, dist_1d, acc_1w, holdings = fetch_all(token)
    except Exception as e:
        log.error("Sharehub fetch failed: %s", e)
        return False

    flow_records     = build_flow_records(acc_1d, dist_1d, acc_1w, trade_date)
    holdings_records = build_holdings_records(holdings, trade_date)

    if not flow_records and not holdings_records:
        log.warning("No records built — skipping")
        return False

    if dry_run:
        log.info("[DRY-RUN] %d flow + %d holdings records",
                 len(flow_records), len(holdings_records))
        log.info("── Top 3 acc (broker count):")
        for r in _top_flow_by_broker_count(flow_records, "ACCUMULATION"):
            log.info("  %s: %s brokers | NPR %s",
                     r["symbol"], r["acc_broker_count_1d"],
                     _cr(float(r["acc_amount_1d"])))
        log.info("── Top 3 stealth (holdings):")
        for h in _top_holdings_by_stealth(holdings_records):
            log.info("  %s: stealth=%.1f | top3=%.1f%% | public=%.2f%%",
                     h["symbol"], float(h["stealth_score"]),
                     float(h["top3_holding_pct"]), float(h["public_trade_pct"]))
        return True

    flow_saved     = save_flow_records(flow_records)
    holdings_saved = save_holdings_records(holdings_records)

    if flow_saved == 0 and holdings_saved == 0:
        log.error("Nothing saved — skipping notifications")
        return False

    try:
        send_notifications(flow_records, holdings_records, trade_date)
    except Exception as e:
        log.error("Notification failed: %s", e)

    log.info("Done — %d flow | %d holdings symbols tracked",
             len(flow_records), len(holdings_records))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [BROKER_FLOW] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    parser = argparse.ArgumentParser(description="NEPSE Broker Flow + Holdings Scraper")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and print — no DB write, no notifications")
    args = parser.parse_args()
    sys.exit(0 if run(dry_run=args.dry_run) else 1)