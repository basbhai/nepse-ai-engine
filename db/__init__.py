"""
db/__init__.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Single import point for all database operations.

Every other module does:
    from db import get_setting, write_row, read_tab

No module ever imports directly from db.connection, db.schema, or db.migrations.
All public API lives here.
─────────────────────────────────────────────────────────────────────────────
"""

# Re-export everything from sheets.py as the public API
from sheets import (
    # Core CRUD
    read_tab,
    read_tab_where,
    write_row,
    upsert_row,
    batch_write,
    batch_upsert,
    update_row,
    delete_rows,
    clear_table,

    # Settings
    get_setting,
    update_setting,
    get_all_settings,

    # Domain helpers
    write_signal,
    update_trade_outcome,
    get_open_positions,
    get_pending_trades,
    get_watchlist,
    get_watchlist_symbols,
    write_market_breadth,
    write_indicators_batch,
    read_today_indicators,
    write_geo_snapshot,
    get_latest_geo,
    get_macro_data,
    upsert_macro,
    write_candle_pattern,
    log_lesson,

    # Admin
    initialize_all_tabs,
    table_exists,
    get_row_count,
    run_raw_sql,
)

# Schema reference — useful for other modules to inspect column lists
from db.schema import TABS, TABLE_COLUMNS

# Connection health check
from db.connection import test_connection, close_pool

__all__ = [
    # Core CRUD
    "read_tab", "read_tab_where", "write_row", "upsert_row",
    "batch_write", "batch_upsert", "update_row", "delete_rows", "clear_table",
    # Settings
    "get_setting", "update_setting", "get_all_settings",
    # Domain helpers
    "write_signal", "update_trade_outcome", "get_open_positions",
    "get_pending_trades", "get_watchlist", "get_watchlist_symbols",
    "write_market_breadth", "write_indicators_batch", "read_today_indicators",
    "write_geo_snapshot", "get_latest_geo", "get_macro_data",
    "upsert_macro", "write_candle_pattern", "log_lesson",
    # Admin
    "initialize_all_tabs", "table_exists", "get_row_count", "run_raw_sql",
    # Schema
    "TABS", "TABLE_COLUMNS",
    # Connection
    "test_connection", "close_pool",
]
