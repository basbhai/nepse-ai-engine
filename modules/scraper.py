"""
scraper.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 2
Purpose : Fetch all live market data needed by the signal pipeline.

Two data sources:
  1. TMS49 API      — real-time prices for all ~335 securities
                      (LTP, Open, High, Low, Volume, VWAP, % change etc.)
  2. ShareSansar    — Conf. score per symbol (proprietary momentum score)
                      Table at: https://www.sharesansar.com/today-share-price
                      Server-rendered HTML — no JS, no captcha, no login.

Output (in memory, passed to indicators.py):
  get_all_market_data() → dict[symbol, PriceRow]

Side effect (written to Sheets):
  MARKET_BREADTH tab — advancing/declining counts, turnover, 52W stats

Gate:
  calendar_guard.is_open() is checked by the CALLER (main.py / trading.yml).
  scraper.py itself does not gate — it can be called for EOD/testing too.

Column mapping from ShareSansar table (matches OmitNomis CSV structure):
  Symbol | Conf. | Open | High | Low | Close | LTP | Close-LTP |
  Close-LTP% | VWAP | Vol | Prev.Close | Turnover | Trans. |
  Diff | Range | Diff% | Range% | VWAP% | 120Days | 180Days |
  52W_High | 52W_Low

─────────────────────────────────────────────────────────────────────────────
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Internal imports
from modules.tms_scraper import get_session, fetch_top25, fetch_indices, dashboard_headers

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NST = timezone(timedelta(hours=5, minutes=45))

SHARESANSAR_URL  = "https://www.sharesansar.com/today-share-price"
SHARESANSAR_TIMEOUT = 20  # seconds
SHARESANSAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}

# Conf. score thresholds (from handoff Section 10.2)
CONF_BULLISH_THRESHOLD  = 50   # above = bullish momentum
CONF_BEARISH_THRESHOLD  = 35   # below = weak/bearish


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PriceRow:
    """
    Complete market data for one security.
    Merges TMS real-time data + ShareSansar Conf. score.
    This is the unit passed to indicators.py.
    """
    symbol:        str
    ltp:           float = 0.0
    open_price:    float = 0.0
    high:          float = 0.0
    low:           float = 0.0
    close:         float = 0.0
    prev_close:    float = 0.0
    change:        float = 0.0       # LTP - prev_close
    change_pct:    float = 0.0       # % change
    volume:        int   = 0
    turnover:      float = 0.0       # NPR
    transactions:  int   = 0
    vwap:          float = 0.0
    high_52w:      float = 0.0
    low_52w:       float = 0.0
    avg_120d:      float = 0.0       # 120-day average price
    avg_180d:      float = 0.0       # 180-day average price
    conf_score:    float = 0.0       # ShareSansar proprietary momentum score
    conf_signal:   str   = "NEUTRAL" # BULLISH / BEARISH / NEUTRAL
    source:        str   = "tms"     # tms | sharesansar | merged
    timestamp:     str   = field(default_factory=lambda: datetime.now(
                                     tz=timezone(timedelta(hours=5, minutes=45))
                                 ).strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> dict:
        return asdict(self)
# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 0 — share shansaar live
# ══════════════════════════════════════════════════════════════════════════════

SHARESANSAR_LIVE_URL = "https://www.sharesansar.com/live-trading"

def fetch_live_trading() -> dict[str, PriceRow]:
    """
    Fetch live prices from ShareSansar live-trading page.
    No login, no captcha, ~5 sec, ~1 min delay acceptable.
    Columns: S.No, Symbol, LTP, Point Change, % Change, 
             Open, High, Low, Volume, Prev. Close
    """
    logger.info("ShareSansar live-trading: fetching...")
    try:
        resp = requests.get(
            SHARESANSAR_LIVE_URL,
            headers=SHARESANSAR_HEADERS,
            timeout=SHARESANSAR_TIMEOUT,
        )
        resp.raise_for_status()

        soup  = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if not table:
            logger.warning("ShareSansar live-trading: no table found")
            return {}

        rows   = table.find_all("tr")
        prices = {}

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < 10:
                continue

            def cell(i):
                return cells[i].get_text(strip=True).replace(",", "")

            symbol = cell(1).upper()
            if not symbol:
                continue

            ltp        = _safe_float(cell(2))
            prev_close = _safe_float(cell(9))
            change_pct = _safe_float(cell(4))

            prices[symbol] = PriceRow(
                symbol      = symbol,
                ltp         = ltp,
                open_price  = _safe_float(cell(5)),
                high        = _safe_float(cell(6)),
                low         = _safe_float(cell(7)),
                close       = prev_close,
                prev_close  = prev_close,
                change      = _safe_float(cell(3)),
                change_pct  = change_pct,
                volume      = _safe_int(cell(8)),
                source      = "ss_live",
            )

        logger.info("ShareSansar live-trading: %d symbols", len(prices))
        return prices

    except Exception as exc:
        logger.error("ShareSansar live-trading failed: %s", exc)
        return {}
# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — TMS49 API
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(value, default: float = 0.0) -> float:
    """Convert TMS API value to float safely."""
    try:
        if value is None or value == "" or value == "N/A":
            return default
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Convert TMS API value to int safely."""
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).replace(",", "").strip()))
    except (ValueError, TypeError):
        return default


def fetch_tms_prices() -> dict[str, PriceRow]:
    """
    Fetch real-time price data for all securities from TMS49 API.
    Uses fresh login every call (no caching — GitHub Actions ephemeral).

    Returns:
        dict[symbol, PriceRow] — keyed by uppercase symbol string.
        Empty dict on failure (never raises — caller handles gracefully).

    TMS49 field mapping (from observed API response):
        symbol          → symbol
        ltp             → ltp
        openPrice       → open_price
        highPrice       → high
        lowPrice        → low
        closingPrice    → close  (previous session close)
        percentChange   → change_pct
        volume          → volume
        turnover        → turnover
        totalTrades     → transactions
        averageTradePrice → vwap
        fiftyTwoWeekHigh  → high_52w
        fiftyTwoWeekLow   → low_52w
    """
    logger.info("TMS49: starting fresh login + price fetch...")
    try:
        session, host_sid = get_session()
        raw_securities = fetch_top25(session, host_sid)  # returns ALL securities despite name

        if not raw_securities:
            logger.warning("TMS49: returned 0 securities (market closed?)")
            return {}

        prices: dict[str, PriceRow] = {}

        for sec in raw_securities:
            symbol = str(sec.get("symbol", "")).strip().upper()
            if not symbol:
                continue

            ltp        = _safe_float(sec.get("ltp"))
            prev_close = _safe_float(sec.get("closingPrice") or sec.get("previousClose"))
            change_pct = _safe_float(sec.get("percentChange"))

            # Derive absolute change if API doesn't give it directly
            change = round(ltp - prev_close, 2) if ltp and prev_close else _safe_float(sec.get("change"))

            prices[symbol] = PriceRow(
                symbol       = symbol,
                ltp          = ltp,
                open_price   = _safe_float(sec.get("openPrice")),
                high         = _safe_float(sec.get("highPrice")),
                low          = _safe_float(sec.get("lowPrice")),
                close        = _safe_float(sec.get("closingPrice") or sec.get("closePrice")),
                prev_close   = prev_close,
                change       = change,
                change_pct   = change_pct,
                volume       = _safe_int(sec.get("volume") or sec.get("totalVolume")),
                turnover     = _safe_float(sec.get("turnover") or sec.get("totalTurnover")),
                transactions = _safe_int(sec.get("totalTrades") or sec.get("noOfTransactions")),
                vwap         = _safe_float(sec.get("averageTradePrice") or sec.get("vwap")),
                high_52w     = _safe_float(sec.get("fiftyTwoWeekHigh") or sec.get("high52Week")),
                low_52w      = _safe_float(sec.get("fiftyTwoWeekLow")  or sec.get("low52Week")),
                source       = "tms",
            )

        logger.info("TMS49: fetched %d securities", len(prices))
        return prices

    except Exception as exc:
        logger.error("TMS49 fetch failed: %s", exc)
        return {}


def fetch_tms_indices() -> list[dict]:
    """
    Fetch NEPSE index and sector sub-indices from TMS49.

    Returns:
        List of index dicts with keys:
            indexCode, currentValue, previousValue,
            percentChange, absoluteChange
        Empty list on failure.
    """
    logger.info("TMS49: fetching indices...")
    try:
        session, host_sid = get_session()
        raw = fetch_indices(session, host_sid)

        indices = []
        for idx in (raw or []):
            indices.append({
                "index_code":      str(idx.get("indexCode", "")).strip(),
                "current_value":   _safe_float(idx.get("currentValue") or idx.get("indexValue")),
                "previous_value":  _safe_float(idx.get("previousValue") or idx.get("previousClose")),
                "change_pct":      _safe_float(idx.get("percentChange")),
                "absolute_change": _safe_float(idx.get("absoluteChange") or idx.get("change")),
            })

        logger.info("TMS49: fetched %d indices", len(indices))
        return indices

    except Exception as exc:
        logger.error("TMS49 indices fetch failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — SHARESANSAR CONF. SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _parse_sharesansar_number(text: str) -> float:
    """
    Parse a number from ShareSansar table cell.
    Handles commas, dashes, empty strings.
    """
    if not text:
        return 0.0
    cleaned = text.strip().replace(",", "").replace("\xa0", "")
    if cleaned in ("-", "", "N/A", "—"):
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def fetch_sharesansar_data() -> dict[str, dict]:
    """
    Scrape today's share price table from ShareSansar.
    
    Table columns (in order):
        S.No | Symbol | Conf. | Open | High | Low | Close | LTP |
        Close-LTP | Close-LTP% | VWAP | Vol | Prev.Close |
        Turnover | Trans. | Diff | Range | Diff% | Range% |
        VWAP% | 120Days | 180Days | 52W_High | 52W_Low
    """
    logger.info("ShareSansar: scraping today-share-price table...")

    try:
        resp = requests.get(
            SHARESANSAR_URL,
            headers=SHARESANSAR_HEADERS,
            timeout=SHARESANSAR_TIMEOUT,
        )
        resp.raise_for_status()

    except requests.exceptions.Timeout:
        logger.warning("ShareSansar: request timed out after %ds — conf scores unavailable", SHARESANSAR_TIMEOUT)
        return {}
    except requests.exceptions.RequestException as exc:
        logger.warning("ShareSansar: request failed (%s) — conf scores unavailable", exc)
        return {}

    try:
        soup = BeautifulSoup(resp.text, "html.parser")

        table = (
            soup.find("table", {"id": "headFixed"}) or
            soup.find("table", class_=lambda c: c and "table" in c) or
            soup.find("table")
        )

        if not table:
            logger.warning("ShareSansar: could not find price table in HTML")
            return {}

        rows = table.find_all("tr")
        if len(rows) < 2:
            logger.warning("ShareSansar: table has fewer than 2 rows")
            return {}

        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        logger.info("ShareSansar: table headers = %s", headers[:10])

        data: dict[str, dict] = {}

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            # S.No is cells[0], Symbol is cells[1]
            symbol_cell = cells[1].get_text(strip=True).upper()
            if not symbol_cell or symbol_cell == "SYMBOL":
                continue

            def get_cell(idx: int) -> str:
                if idx < len(cells):
                    return cells[idx].get_text(strip=True)
                return ""

            # Column mapping (0-indexed):
            # 0=S.No, 1=Symbol, 2=Conf., 3=Open, 4=High, 5=Low,
            # 6=Close, 7=LTP, 8=Close-LTP, 9=Close-LTP%,
            # 10=VWAP, 11=Vol, 12=Prev.Close, 13=Turnover, 14=Trans.,
            # 15=Diff, 16=Range, 17=Diff%, 18=Range%, 19=VWAP%,
            # 20=120Days, 21=180Days, 22=52W_High, 23=52W_Low

            data[symbol_cell] = {
                "conf_score":   _parse_sharesansar_number(get_cell(2)),
                "open_price":   _parse_sharesansar_number(get_cell(3)),
                "high":         _parse_sharesansar_number(get_cell(4)),
                "low":          _parse_sharesansar_number(get_cell(5)),
                "close":        _parse_sharesansar_number(get_cell(6)),
                "ltp":          _parse_sharesansar_number(get_cell(7)),
                "vwap":         _parse_sharesansar_number(get_cell(10)),
                "volume":       int(_parse_sharesansar_number(get_cell(11))),
                "prev_close":   _parse_sharesansar_number(get_cell(12)),
                "turnover":     _parse_sharesansar_number(get_cell(13)),
                "transactions": int(_parse_sharesansar_number(get_cell(14))),
                "avg_120d":     _parse_sharesansar_number(get_cell(20)),
                "avg_180d":     _parse_sharesansar_number(get_cell(21)),
                "high_52w":     _parse_sharesansar_number(get_cell(22)),
                "low_52w":      _parse_sharesansar_number(get_cell(23)),
            }
        logger.info("ShareSansar: parsed %d symbols", len(data))
        return data

    except Exception as exc:
        logger.error("ShareSansar: parse failed — %s", exc)
        return {}

# ══════════════════════════════════════════════════════════════════════════════
# MERGE — TMS + ShareSansar
# ══════════════════════════════════════════════════════════════════════════════

def _conf_signal(score: float) -> str:
    """Classify Conf. score into signal label."""
    if score >= CONF_BULLISH_THRESHOLD:
        return "BULLISH"
    if score <= CONF_BEARISH_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def merge_market_data(
    tms_data: dict[str, PriceRow],
    ss_data:  dict[str, dict],
) -> dict[str, PriceRow]:
    """
    Merge TMS real-time prices with ShareSansar enrichment data.

    Strategy:
    - TMS is the primary source for real-time price fields
      (LTP, change%, volume are live — ShareSansar is delayed)
    - ShareSansar fills in: conf_score, avg_120d, avg_180d
    - For fields where TMS returns 0 and ShareSansar has a value,
      ShareSansar backfills (e.g. 52W high/low, prev_close)

    Returns merged dict. Symbols only in TMS (no SS data) are kept
    with conf_score=0 — they will be filtered by filter_engine.py.
    """
    merged: dict[str, PriceRow] = {}
    ss_matched = 0
    ss_missing = 0

    for symbol, row in tms_data.items():
        ss = ss_data.get(symbol, {})

        if ss:
            ss_matched += 1
            conf = ss.get("conf_score", 0.0)

            # Backfill zeros from ShareSansar
            row.high_52w     = row.high_52w     or ss.get("high_52w", 0.0)
            row.low_52w      = row.low_52w      or ss.get("low_52w",  0.0)
            row.prev_close   = row.prev_close   or ss.get("prev_close", 0.0)
            row.avg_120d     = ss.get("avg_120d", 0.0)
            row.avg_180d     = ss.get("avg_180d", 0.0)
            row.conf_score   = conf
            row.conf_signal  = _conf_signal(conf)
            row.source       = "merged"

        else:
            ss_missing += 1
            row.conf_signal = "NEUTRAL"
            row.source      = "tms_only"

        merged[symbol] = row

    # Symbols in ShareSansar but not in TMS (not actively traded today)
    # — add them with conf score only, for informational purposes
    for symbol, ss in ss_data.items():
        if symbol not in merged:
            conf = ss.get("conf_score", 0.0)
            merged[symbol] = PriceRow(
                symbol       = symbol,
                ltp          = ss.get("ltp", 0.0),
                open_price   = ss.get("open_price", 0.0),
                high         = ss.get("high", 0.0),
                low          = ss.get("low", 0.0),
                close        = ss.get("close", 0.0),
                prev_close   = ss.get("prev_close", 0.0),
                volume       = ss.get("volume", 0),
                turnover     = ss.get("turnover", 0.0),
                transactions = ss.get("transactions", 0),
                vwap         = ss.get("vwap", 0.0),
                high_52w     = ss.get("high_52w", 0.0),
                low_52w      = ss.get("low_52w", 0.0),
                avg_120d     = ss.get("avg_120d", 0.0),
                avg_180d     = ss.get("avg_180d", 0.0),
                conf_score   = conf,
                conf_signal  = _conf_signal(conf),
                source       = "ss_only",
            )

    logger.info(
        "Merge complete: %d total | %d TMS+SS | %d TMS-only | %d SS-only",
        len(merged), ss_matched, ss_missing,
        len(ss_data) - ss_matched,
    )
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MARKET BREADTH — compute + write to Sheets
# ══════════════════════════════════════════════════════════════════════════════

def compute_market_breadth(
    merged_data: dict[str, PriceRow],
    indices:     list[dict],
) -> dict:
    """
    Compute market breadth statistics from merged price data.

    Returns dict ready to write to MARKET_BREADTH tab.
    """
    advancing  = 0
    declining  = 0
    unchanged  = 0
    new_52w_high = 0
    new_52w_low  = 0
    total_turnover = 0.0
    total_volume   = 0

    for row in merged_data.values():
        if row.ltp <= 0:
            continue

        # Direction
        if row.change_pct > 0:
            advancing += 1
        elif row.change_pct < 0:
            declining += 1
        else:
            unchanged += 1

        # 52W extremes — LTP at or above/below 52W mark
        if row.high_52w > 0 and row.ltp >= row.high_52w:
            new_52w_high += 1
        if row.low_52w > 0 and row.ltp <= row.low_52w:
            new_52w_low += 1

        total_turnover += row.turnover
        total_volume   += row.volume

    # Simple breadth score: (advancing - declining) / total active
    total_active = advancing + declining + unchanged
    breadth_score = round(
        (advancing - declining) / total_active * 100, 2
    ) if total_active > 0 else 0.0

    # Market signal from breadth
    if breadth_score >= 30:
        market_signal = "STRONGLY_BULLISH"
    elif breadth_score >= 10:
        market_signal = "BULLISH"
    elif breadth_score >= -10:
        market_signal = "NEUTRAL"
    elif breadth_score >= -30:
        market_signal = "BEARISH"
    else:
        market_signal = "STRONGLY_BEARISH"

    # Find NEPSE composite index
    nepse_value = 0.0
    nepse_change_pct = 0.0
    for idx in indices:
        code = idx.get("index_code", "").upper()
        if "NEPSE" in code and "FLOAT" not in code and "SENSITIVE" not in code:
            nepse_value      = idx.get("current_value", 0.0)
            nepse_change_pct = idx.get("change_pct", 0.0)
            break

    nst_now = datetime.now(tz=timezone(timedelta(hours=5, minutes=45)))

    return {
        "Date":              nst_now.strftime("%Y-%m-%d"),
        "Advancing":         advancing,
        "Declining":         declining,
        "Unchanged":         unchanged,
        "New_52W_High":      new_52w_high,
        "New_52W_Low":       new_52w_low,
        "Total_Turnover_NPR": round(total_turnover, 2),
        "Total_Volume":      total_volume,
        "Breadth_Score":     breadth_score,
        "Market_Signal":     market_signal,
        "NEPSE_Index":       nepse_value,
        "NEPSE_Change_Pct":  nepse_change_pct,
        "Timestamp":         nst_now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_market_breadth(breadth: dict) -> bool:
    """
    Write market breadth snapshot to Neon MARKET_BREADTH table.
    Upserts on date — safe to call multiple times per day.

    Normalises Title_Case keys from compute_market_breadth() to
    snake_case to match DB column names.

    Returns True on success, False on failure (never raises).
    """
    # Key map: compute_market_breadth() output → DB column names
    key_map = {
        "Date":               "date",
        "Advancing":          "advancing",
        "Declining":          "declining",
        "Unchanged":          "unchanged",
        "New_52W_High":       "new_52w_high",
        "New_52W_Low":        "new_52w_low",
        "Total_Turnover_NPR": "total_turnover_npr",
        "Total_Volume":       "total_volume",
        "Breadth_Score":      "breadth_score",
        "Market_Signal":      "market_signal",
        "NEPSE_Index":        "nepse_index",
        "NEPSE_Change_Pct":   "nepse_change_pct",
        "Timestamp":          "timestamp",
    }
    normalised = {
        key_map.get(k, k.lower()): str(v) if v is not None else ""
        for k, v in breadth.items()
    }

    try:
        from db import write_market_breadth as db_write  # noqa
        ok = db_write(normalised)
        if ok:
            logger.info(
                "MARKET_BREADTH → Neon: adv=%s dec=%s score=%s signal=%s nepse=%s",
                normalised.get("advancing"),
                normalised.get("declining"),
                normalised.get("breadth_score"),
                normalised.get("market_signal"),
                normalised.get("nepse_index"),
            )
        return ok
    except ImportError:
        logger.warning("db package not importable — market breadth not written")
        return False
    except Exception as exc:
        logger.error("write_market_breadth failed: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def get_all_market_data(
    write_breadth: bool = True,
    ss_fallback_ok: bool = True,
) -> dict[str, PriceRow]:

    # Step 1: Live prices from ShareSansar (fast, no login)
    live_data = fetch_live_trading()
    if not live_data:
        logger.error("ShareSansar live-trading returned no data")
        return {}

    # Step 2: Conf scores + 120d/180d/52W from today-share-price
    ss_data = fetch_sharesansar_data()

    # Step 3: Enrich live prices with conf scores
    for symbol, row in live_data.items():
        ss = ss_data.get(symbol, {})
        if ss:
            row.conf_score  = ss.get("conf_score", 0.0)
            row.conf_signal = _conf_signal(row.conf_score)
            row.avg_120d    = ss.get("avg_120d", 0.0)
            row.avg_180d    = ss.get("avg_180d", 0.0)
            row.high_52w    = ss.get("high_52w", 0.0)
            row.low_52w     = ss.get("low_52w", 0.0)
            row.turnover    = ss.get("turnover", 0.0)
            row.transactions= ss.get("transactions", 0)
            row.vwap        = ss.get("vwap", 0.0)
            row.source      = "ss_merged"
        else:
            row.conf_signal = "NEUTRAL"

    if write_breadth and live_data:
        breadth = compute_market_breadth(live_data, [])
        write_market_breadth(breadth)

    logger.info("scraper: %d symbols | source=sharesansar_only", len(live_data))
    return live_data

def get_watchlist_data(symbols: list[str]) -> dict[str, PriceRow]:
    """
    Fetch market data filtered to watchlist symbols only.
    Useful for quick runs when you only care about tracked stocks.

    Args:
        symbols: List of uppercase symbol strings e.g. ["NABIL", "HBL"]

    Returns:
        Subset of get_all_market_data() filtered to requested symbols.
    """
    all_data = get_all_market_data(write_breadth=False)
    watchlist_upper = {s.upper() for s in symbols}
    filtered = {sym: row for sym, row in all_data.items() if sym in watchlist_upper}
    logger.info("Watchlist filter: %d/%d symbols found", len(filtered), len(symbols))
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# CLI — test runner
#   python -m modules.scraper             → full run, print summary
#   python -m modules.scraper --ss-only    → ShareSansar scrape only (no TMS login)
#   python -m modules.scraper NABIL HBL    → watchlist mode for named symbols
#   python -m modules.scraper --tms-only   → TMS only (skip ShareSansar)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SCRAPER] %(levelname)s: %(message)s",
    )

    args = sys.argv[1:]

    # ── ShareSansar only ────────────────────────────────────────────────────
    if "--ss-only" in args:
        print("\n" + "=" * 60)
        print("  ShareSansar ONLY — conf scores test")
        print("=" * 60 + "\n")

        ss = fetch_sharesansar_data()
        if ss:
            print(f"  Symbols parsed: {len(ss)}\n")
            print(f"  {'Symbol':<12} {'Conf':>6} {'Signal':<10} {'120d':>8} {'180d':>8}")
            print("  " + "-" * 50)
            # Show first 10, sorted by conf desc
            top = sorted(ss.items(), key=lambda x: x[1].get("conf_score", 0), reverse=True)[:10]
            for sym, d in top:
                score = d.get("conf_score", 0)
                signal = _conf_signal(score)
                print(f"  {sym:<12} {score:>6.2f} {signal:<10} "
                      f"{d.get('avg_120d',0):>8.2f} {d.get('avg_180d',0):>8.2f}")
        else:
            print("  FAILED — no data returned")

    # ── TMS only ────────────────────────────────────────────────────────────
    elif "--tms-only" in args:
        print("\n" + "=" * 60)
        print("  TMS49 ONLY — live prices test")
        print("=" * 60 + "\n")

        tms = fetch_tms_prices()
        if tms:
            print(f"  Securities: {len(tms)}\n")
            print(f"  {'Symbol':<12} {'LTP':>8} {'Chg%':>7} {'Vol':>10} {'Turnover':>14}")
            print("  " + "-" * 60)
            top = sorted(tms.values(), key=lambda r: r.turnover, reverse=True)[:10]
            for row in top:
                print(f"  {row.symbol:<12} {row.ltp:>8.2f} {row.change_pct:>+7.2f}% "
                      f"{row.volume:>10,} {row.turnover:>14,.0f}")
        else:
            print("  FAILED — market may be closed")

    # ── Watchlist mode ───────────────────────────────────────────────────────
    elif args and not args[0].startswith("--"):
        symbols = [a.upper() for a in args]
        print(f"\n  Watchlist mode: {symbols}\n")
        data = get_watchlist_data(symbols)
        for sym, row in data.items():
            print(f"  {sym}: LTP={row.ltp} chg={row.change_pct:+.2f}% "
                  f"vol={row.volume:,} conf={row.conf_score} [{row.conf_signal}]")
        missing = set(symbols) - set(data.keys())
        if missing:
            print(f"\n  Not found: {missing}")

    # ── Full run ─────────────────────────────────────────────────────────────
    else:
        print("\n" + "=" * 60)
        print("  NEPSE Scraper — Full market data fetch")
        print("=" * 60 + "\n")

        data = get_all_market_data(write_breadth=True)

        if not data:
            print("  FAILED — no data returned")
            sys.exit(1)

        # Summary stats
        bullish  = sum(1 for r in data.values() if r.conf_signal == "BULLISH")
        bearish  = sum(1 for r in data.values() if r.conf_signal == "BEARISH")
        neutral  = sum(1 for r in data.values() if r.conf_signal == "NEUTRAL")
        adv      = sum(1 for r in data.values() if r.change_pct > 0)
        dec      = sum(1 for r in data.values() if r.change_pct < 0)
        merged_c = sum(1 for r in data.values() if r.source == "merged")

        print(f"  Total symbols    : {len(data)}")
        print(f"  Fully merged     : {merged_c}")
        print(f"  Advancing        : {adv} | Declining: {dec}")
        print(f"  Conf — Bullish   : {bullish} | Neutral: {neutral} | Bearish: {bearish}")
        print()
        print(f"  {'Symbol':<12} {'LTP':>8} {'Chg%':>7} {'Conf':>6} {'Signal':<12} {'Source'}")
        print("  " + "-" * 65)

        # Top 15 by turnover
        top = sorted(data.values(), key=lambda r: r.turnover, reverse=True)[:15]
        for row in top:
            print(f"  {row.symbol:<12} {row.ltp:>8.2f} {row.change_pct:>+7.2f}% "
                  f"{row.conf_score:>6.1f} {row.conf_signal:<12} {row.source}")

        print("\n  ✅ Done — MARKET_BREADTH written to Sheets")
        print("=" * 60 + "\n")