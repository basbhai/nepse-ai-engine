#!/usr/bin/env python3
"""
modules/atrad_mcp_server.py — MCP server exposing ATrad TMS live market
data (modules/atrad_scraper.py) as read-only tools.

Credentials (TMS_ROADSHOW_USER / TMS_ROADSHOW_PASS / TMS_ROADSHOW_WATCH_ID)
are read from the project root .env file — same variables atrad_scraper.py
already uses. Login happens lazily on first tool call.

Read-only by design: fetch_market_watch is always called with
write_db=False here, so no tool in this file writes to the database or
places any order.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # for sheets.py / config.py imports inside atrad_scraper
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import atrad_scraper`

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import atrad_scraper as atrad
from mcp.server import MCPServer

mcp = MCPServer("atrad-scraper")


def _df_to_records(df):
    """Convert a pandas DataFrame to JSON-safe list[dict] (handles numpy dtypes)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


@mcp.tool()
def get_market_watch(symbol: str | None = None) -> list[dict]:
    """Fetch the live ATrad market watch (LTP, OHLC, VWAP, bid/ask depth,
    volume, DPR bands, and derived vwap_dev / bid_ask_ratio / dpr_proximity)
    for all NEPSE symbols on the configured watch list. Never writes to the
    database. Pass `symbol` to filter to a single ticker.
    """
    df = atrad.fetch_market_watch(write_db=False)
    if symbol and not df.empty and "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    return _df_to_records(df)


@mcp.tool()
def get_trade_stats() -> dict:
    """Get market-wide advancing/declining/unchanged counts from ATrad."""
    return atrad.fetch_trade_stats()


@mcp.tool()
def get_nepse_index() -> dict:
    """Get the live NEPSE composite + Sensitive index values, market
    volume, turnover, and trade count from ATrad.
    """
    return atrad.fetch_nepse_index()


@mcp.tool()
def get_order_book(symbol: str) -> dict:
    """Get live bid/ask order book depth, imbalance, and spread for a
    single symbol from ATrad.
    """
    return atrad.fetch_order_book(symbol)


@mcp.tool()
def get_ltp(symbol: str) -> dict | None:
    """Get a fresh single-symbol quote (LTP, VWAP, bid/ask, DPR band,
    volume) from ATrad. No DB read/write — always live.
    """
    return atrad.get_ltp_live(symbol)


if __name__ == "__main__":
    mcp.run()
