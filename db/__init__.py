"""
db/__init__.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Package marker for the db/ directory.

IMPORT RULE FOR ALL MODULES:
    from sheets import get_setting, write_row, read_tab   ← always use this

DO NOT import from db directly — db/__init__.py intentionally does not
re-export sheets.py functions to avoid circular imports.

    sheets.py          → from db.connection import _db       ✅
    db/__init__.py     → does NOT import sheets.py           ✅ (no loop)
    any other module   → from sheets import ...              ✅

Schema and connection utilities are available if needed:
    from db.schema     import TABS, TABLE_COLUMNS
    from db.connection import test_connection
    from db.migrations import run_migrations
─────────────────────────────────────────────────────────────────────────────
"""

# Only safe imports — these do NOT import sheets.py
from db.schema     import TABS, TABLE_COLUMNS, TABLE_DDL
from db.connection import test_connection, close_pool
from db.migrations import run_migrations

__all__ = [
    "TABS",
    "TABLE_COLUMNS",
    "TABLE_DDL",
    "test_connection",
    "close_pool",
    "run_migrations",
]