"""
db/migrations.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Automatic schema migration runner.

How it works (ORM-style, no manual ALTER TABLE ever):
    1. schema.py defines TABLE_DDL (CREATE TABLE) and TABLE_COLUMNS (truth)
    2. migrations.py inspects what actually exists in Neon
    3. Diffs truth vs reality — finds missing tables and missing columns
    4. Auto-applies: CREATE TABLE or ALTER TABLE ADD COLUMN as needed
    5. Records every change in db_schema with a timestamp

Adding a new column:
    → Add it to TABLE_COLUMNS in schema.py
    → Run: python -m db.migrations
    → Done. No manual SQL, no numbered functions, no 002/003/004.

Adding a new table:
    → Add DDL to TABLE_DDL in schema.py
    → Add columns to TABLE_COLUMNS in schema.py
    → Add key to TABS in schema.py
    → Run: python -m db.migrations
    → Done.

CLI:
    python -m db.migrations            # auto-detect and apply all drift
    python -m db.migrations status     # show table/column status
    python -m db.migrations reset      # DROP all tables and recreate (⚠️ destructive)
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime

from db.connection import _db
from db.schema import TABLE_DDL, TABLE_COLUMNS, TABS
from config import NST

log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# BOOTSTRAP — db_schema tracking table
# ─────────────────────────────────────────

def _bootstrap():
    """
    Create the db_schema tracking table if it doesn't exist.
    Always safe to run — uses CREATE TABLE IF NOT EXISTS.
    """
    with _db() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS db_schema (
                id             SERIAL PRIMARY KEY,
                migration_id   TEXT NOT NULL,
                name           TEXT NOT NULL,
                applied_at     TIMESTAMPTZ DEFAULT NOW(),
                status         TEXT DEFAULT 'applied',
                notes          TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_schema_migration_id
                ON db_schema (migration_id);
        """)


def _record(migration_id: str, name: str, notes: str = ""):
    """Record a migration in db_schema. Upserts — safe to call twice."""
    with _db() as cur:
        cur.execute("""
            INSERT INTO db_schema (migration_id, name, applied_at, status, notes)
            VALUES (%s, %s, NOW(), 'applied', %s)
            ON CONFLICT (migration_id) DO UPDATE
                SET applied_at = NOW(),
                    status     = 'applied',
                    notes      = EXCLUDED.notes
        """, (migration_id, name, notes))


# ─────────────────────────────────────────
# INTROSPECTION — what does Neon have now?
# ─────────────────────────────────────────

def get_existing_tables() -> set[str]:
    """Return set of table names currently in the public schema."""
    with _db() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type   = 'BASE TABLE'
        """)
        return {row["table_name"] for row in cur.fetchall()}


def get_existing_columns(table: str) -> set[str]:
    """Return set of column names currently in a table."""
    with _db() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = %s
        """, (table,))
        return {row["column_name"] for row in cur.fetchall()}


def get_all_existing_columns() -> dict[str, set[str]]:
    """Return {table_name: {col1, col2, ...}} for all tables in DB."""
    with _db() as cur:
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """)
        result: dict[str, set[str]] = {}
        for row in cur.fetchall():
            t = row["table_name"]
            if t not in result:
                result[t] = set()
            result[t].add(row["column_name"])
        return result


# ─────────────────────────────────────────
# DIFF — what needs to change?
# ─────────────────────────────────────────

def compute_drift() -> dict:
    """
    Compare schema.py (truth) against the live Neon DB.

    Returns:
        {
            "missing_tables":  ["table1", ...],
            "missing_columns": {"table2": ["col_a", "col_b"], ...},
            "extra_tables":    ["orphan_table", ...],  # info only, not touched
        }
    """
    existing_tables  = get_existing_tables()
    existing_columns = get_all_existing_columns()

    # Tables defined in schema but not in DB
    schema_tables   = set(TABLE_DDL.keys())
    missing_tables  = schema_tables - existing_tables

    # Columns defined in TABLE_COLUMNS but not in DB table
    missing_columns: dict[str, list[str]] = {}
    for table, expected_cols in TABLE_COLUMNS.items():
        if table not in existing_tables:
            continue   # will be created by CREATE TABLE — skip column diff
        actual_cols = existing_columns.get(table, set())
        missing     = [c for c in expected_cols if c not in actual_cols]
        if missing:
            missing_columns[table] = missing

    # Tables in DB but not in schema (informational only)
    extra_tables = existing_tables - schema_tables - {"db_schema"}

    return {
        "missing_tables":  sorted(missing_tables),
        "missing_columns": missing_columns,
        "extra_tables":    sorted(extra_tables),
    }


# ─────────────────────────────────────────
# APPLY — fix the drift
# ─────────────────────────────────────────

def create_table(table: str) -> bool:
    """
    Run CREATE TABLE IF NOT EXISTS from TABLE_DDL.
    Includes indexes defined in the DDL block.
    """
    ddl = TABLE_DDL.get(table)
    if not ddl:
        log.error("No DDL found for table: %s", table)
        return False
    try:
        with _db() as cur:
            cur.execute(ddl)
        log.info("  ✅ Created table: %s", table)
        return True
    except Exception as e:
        log.error("  ❌ Failed to create table %s: %s", table, e)
        return False


def add_column(table: str, column: str) -> bool:
    """
    ALTER TABLE to add a missing column as TEXT (nullable).
    Uses IF NOT EXISTS — safe to run multiple times.
    """
    try:
        with _db() as cur:
            cur.execute(f"""
                ALTER TABLE "{table}"
                ADD COLUMN IF NOT EXISTS "{column}" TEXT
            """)
        log.info("  ✅ Added column: %s.%s", table, column)
        return True
    except Exception as e:
        log.error("  ❌ Failed to add column %s.%s: %s", table, column, e)
        return False


def apply_drift(drift: dict) -> dict:
    """
    Apply all detected drift — create missing tables, add missing columns.
    Records each change in db_schema.

    Returns:
        {
            "tables_created":  int,
            "columns_added":   int,
            "tables_failed":   int,
            "columns_failed":  int,
        }
    """
    stats = dict(tables_created=0, columns_added=0, tables_failed=0, columns_failed=0)

    # ── Create missing tables ──────────────────────────────────────────────
    for table in drift["missing_tables"]:
        log.info("Creating missing table: %s", table)
        ok = create_table(table)
        if ok:
            stats["tables_created"] += 1
            _record(
                f"auto_create_{table}",
                f"auto: create table {table}",
                notes="auto-detected missing table",
            )
        else:
            stats["tables_failed"] += 1

    # ── Add missing columns ────────────────────────────────────────────────
    for table, columns in drift["missing_columns"].items():
        log.info("Adding %d missing column(s) to %s: %s", len(columns), table, columns)
        for col in columns:
            ok = add_column(table, col)
            if ok:
                stats["columns_added"] += 1
                _record(
                    f"auto_add_col_{table}_{col}",
                    f"auto: add column {table}.{col}",
                    notes="auto-detected missing column",
                )
            else:
                stats["columns_failed"] += 1

    return stats


# ─────────────────────────────────────────
# SEED — default data
# ─────────────────────────────────────────

def seed_settings():
    """
    Seed settings table with defaults.
    ON CONFLICT DO NOTHING — never overwrites user changes.
    Safe to run on every init.
    """
    defaults = [
        # Key                      Value       Description
        ("PAPER_MODE",             "true",     "If true, no real orders placed"),
        ("CAPITAL_TOTAL_NPR",      "100000",   "Total trading capital in NPR"),
        ("MAX_POSITION_PCT",       "10",       "Max single position % of capital"),
        ("STOP_LOSS_DEFAULT_PCT",  "5",        "Default stop loss %"),
        ("RSI_OVERSOLD",           "40",       "RSI level = oversold"),
        ("RSI_OVERBOUGHT",         "70",       "RSI level = overbought"),
        ("MIN_VOLUME_RATIO",       "1.5",      "Min volume vs 180d avg to qualify"),
        ("MIN_TECH_SCORE",         "3",        "Min tech score to pass filter"),
        ("MIN_CONF_SCORE",         "50",       "Min ShareSansar confidence score"),
        ("FD_RATE_PCT",            "8.0",      "Current FD rate — update monthly"),
        ("BROKERAGE_RATE_PCT",     "0.4",      "Broker commission %"),
        ("SEBON_FEE_PCT",          "0.015",    "SEBON fee %"),
        ("DP_FEE_NPR",             "25",       "Depository fee per transaction NPR"),
        ("TELEGRAM_ENABLED",       "false",    "Send Telegram alerts"),
        ("TELEGRAM_CHAT_ID",       "",         "Your Telegram chat ID"),
        ("MARKET_OPEN_TIME",       "10:45",    "NEPSE open time NST"),
        ("MARKET_CLOSE_TIME",      "15:00",    "NEPSE close time NST"),
        ("LAST_MACRO_UPDATE",      "",         "Date of last macro data entry"),
    ]

    inserted = 0
    with _db() as cur:
        for key, value, description in defaults:
            cur.execute("""
                INSERT INTO settings (key, value, description, last_updated, set_by)
                VALUES (%s, %s, %s, NOW()::text, 'system')
                ON CONFLICT (key) DO NOTHING
            """, (key, value, description))
            if cur.rowcount:
                inserted += 1

    if inserted:
        log.info("Seeded %d default settings", inserted)
        _record("seed_settings", "seed default settings",
                notes=f"{inserted} keys inserted")


# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────

def run_migrations() -> dict:
    """
    Full auto-migration pipeline:
        1. Bootstrap db_schema tracking table
        2. Compute drift (missing tables + columns)
        3. Apply all drift
        4. Seed default data
        5. Return summary

    Called by initialize_all_tabs() in db/sheets.py.
    Called by CLI: python -m db.migrations
    """
    _bootstrap()

    drift = compute_drift()
    needs_work = drift["missing_tables"] or drift["missing_columns"]

    if not needs_work:
        log.info("Schema is up to date — no changes needed")
        seed_settings()
        return {"status": "up_to_date", "tables_created": 0, "columns_added": 0}

    log.info(
        "Drift detected — %d missing tables, %d tables with missing columns",
        len(drift["missing_tables"]),
        len(drift["missing_columns"]),
    )

    stats = apply_drift(drift)
    seed_settings()

    stats["status"] = "migrated"
    return stats


def migration_status() -> dict:
    """
    Return full current state — what exists vs what schema expects.
    Used by CLI status command.
    """
    _bootstrap()
    existing_tables  = get_existing_tables()
    existing_columns = get_all_existing_columns()

    report = {}
    for table, expected_cols in TABLE_COLUMNS.items():
        if table not in existing_tables:
            report[table] = {"status": "MISSING", "missing_columns": expected_cols}
            continue

        actual_cols = existing_columns.get(table, set())
        missing     = [c for c in expected_cols if c not in actual_cols]
        extra       = [c for c in actual_cols
                       if c not in expected_cols
                       and c not in ("id", "inserted_at")]

        report[table] = {
            "status":          "OK" if not missing else "DRIFT",
            "expected":        len(expected_cols),
            "actual":          len(actual_cols),
            "missing_columns": missing,
            "extra_columns":   extra,
        }

    return report


def reset_schema():
    """
    DROP all managed tables and recreate from scratch.
    ⚠️  DESTRUCTIVE — all data is lost. Use only in dev.
    """
    log.warning("RESET: dropping all tables...")
    tables = list(TABLE_DDL.keys())
    # Reverse order to avoid FK issues (if any added later)
    with _db() as cur:
        for table in reversed(tables):
            if table == "db_schema":
                continue
            cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            log.warning("  Dropped: %s", table)

    log.info("RESET: recreating schema...")
    run_migrations()
    log.info("RESET complete")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MIGRATIONS] %(levelname)s: %(message)s",
    )
    from log_config import attach_file_handler
    attach_file_handler(__name__)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "apply"

    if cmd == "status":
        print("\n  Checking schema drift...\n")
        report = migration_status()
        print(f"  {'Table':<25} {'Status':<8} {'Expected':>8} {'Actual':>8}  Missing Columns")
        print("  " + "─" * 75)
        for table, info in sorted(report.items()):
            status = info["status"]
            icon   = "✅" if status == "OK" else ("❌" if status == "MISSING" else "⚠️ ")
            exp    = info.get("expected", "—")
            act    = info.get("actual",   "—")
            miss   = ", ".join(info.get("missing_columns", [])) or "—"
            print(f"  {icon} {table:<23} {status:<8} {str(exp):>8} {str(act):>8}  {miss}")
        print()

    elif cmd == "reset":
        confirm = input("  ⚠️  This will DELETE ALL DATA. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            reset_schema()
            print("  ✅ Reset complete")
        else:
            print("  Cancelled")

    else:  # apply (default)
        print("\n  Running auto-migration...\n")
        result = run_migrations()
        status = result.get("status", "unknown")
        if status == "up_to_date":
            print("  ✅ Schema is up to date — nothing to do\n")
        else:
            print(f"  ✅ Migration complete:")
            print(f"     Tables created : {result.get('tables_created', 0)}")
            print(f"     Columns added  : {result.get('columns_added', 0)}")
            if result.get("tables_failed") or result.get("columns_failed"):
                print(f"     ⚠️  Tables failed : {result.get('tables_failed', 0)}")
                print(f"     ⚠️  Columns failed: {result.get('columns_failed', 0)}")
            print()