"""
analysis/backup_sync.py
=======================
NEPSE AI Engine — Neon PostgreSQL Backup Sync

Syncs 20 critical irreplaceable tables from local PostgreSQL → Neon cloud.

Strategy:
  1. First run: CREATE TABLE IF NOT EXISTS on Neon (schema from local)
  2. Every run: INSERT ... ON CONFLICT DO UPDATE (upsert all rows)
  3. Auto-detects new columns: ALTER TABLE IF NOT EXISTS
  4. Runs every 15 min via systemd timer — zero changes to existing modules

Trigger:
  systemd timer: nepse-backup.timer (every 15 min during trading hours)
  Also called manually: python -m analysis.backup_sync

Tables backed up (irreplaceable knowledge):
  settings              — MARKET_STATE, PAPER_MODE, all dynamic config
  financials            — win_rate, profit_factor, loss_streak (stateful)
  nrb_monthly           — manually entered NRB macro data
  fd_rate_summary       — monthly FD rate trend history
  learning_hub          — all lessons Claude reads before every decision
  market_log            — every BUY/WAIT/AVOID signal + eval fields
  trade_journal         — closed trade outcomes with causal attribution
  daily_context_log     — nightly Gemini summaries (GPT reads these)
  gate_misses           — FALSE_BLOCK/CORRECT_BLOCK stamps
  gate_proposals        — threshold proposals from GPT
  claude_audit          — weekly accuracy tracking
  paper_capital         — current capital NPR (single stateful row)
  paper_portfolio       — open/closed positions with WACC and cost basis
  paper_trade_log       — full transaction history for capital reconciliation
  monthly_council_agenda    — agenda items per council run
  monthly_council_log       — full model deliberation transcripts
  monthly_council_checklist — go/stop triggers per month (Claude reads these)
  monthly_override          — confidence score + buy_blocked flag (Claude reads at startup)
  accuracy_review_log   — DeepSeek monthly stats — not reconstructible
  system_proposals      — all architectural proposals + approval audit trail

NOT backed up (re-scrapable):
  price_history, indicators, candle_signals, candle_patterns,
  atrad_market_watch, floorsheet, floorsheet_signals,
  geopolitical_data, nepal_pulse, market_breadth, nepse_indices,
  share_sectors, fd_rates, fundamentals, fundamental_beta,
  sector_momentum, corporate_events, dividend_announcements,
  dividend_pattern_study, backtest_results, capital_allocation,
  financial_advisor, macro_stat_results, international_prices,
  portfolio, paper_users

ENV:
  DATABASE_URL      — local PostgreSQL (source)
  DATABASE_URL_NEON — Neon PostgreSQL (destination)
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [BACKUP_SYNC] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_URL = os.getenv("DATABASE_URL", "")
NEON_URL  = os.getenv("DATABASE_URL_NEON", "")

# Tables to sync — order matters for foreign key safety
BACKUP_TABLES = [
    # ── System config (sync first — everything else reads these) ──────────────
    "settings",               # dynamic config — MARKET_STATE, PAPER_MODE, thresholds
    "financials",             # KPIs — win_rate, loss_streak: stateful, not reconstructible

    # ── Macro inputs (manually entered) ───────────────────────────────────────
    "nrb_monthly",            # NRB macro data — cannot be auto-scraped
    "fd_rate_summary",        # monthly FD trend — seasonal patterns over months

    # ── AI knowledge base ─────────────────────────────────────────────────────
    "learning_hub",           # lessons Claude reads — most irreplaceable
    "daily_context_log",      # nightly Gemini summaries — GPT weekly review

    # ── Trading signals & outcomes ────────────────────────────────────────────
    "market_log",             # all BUY/WAIT/AVOID signals + eval fields
    "trade_journal",          # closed trade outcomes — core GPT learning

    # ── Self-improvement loop ─────────────────────────────────────────────────
    "gate_misses",            # FALSE_BLOCK/CORRECT_BLOCK stamps
    "gate_proposals",         # threshold proposals — config evolution trail
    "claude_audit",           # weekly accuracy tracking — needs full history

    # ── Paper trading capital (single stateful row — not reconstructible) ─────
    "paper_capital",          # current capital NPR — reconciled from all trades
    "paper_portfolio",        # open/closed positions with WACC and cost basis
    "paper_trade_log",        # full transaction ledger — needed for reconciliation

    # ── Monthly council (deliberation history + runtime config) ──────────────
    "monthly_council_agenda",    # agenda items per run — what was deliberated
    "monthly_council_log",       # full model transcripts — irreplaceable record
    "monthly_council_checklist", # go/stop triggers per month — Claude reads
    "monthly_override",          # confidence score + buy_blocked — Claude reads at startup

    # ── Accuracy & proposals ──────────────────────────────────────────────────
    "accuracy_review_log",    # DeepSeek monthly stats — not reconstructible
    "system_proposals",       # architectural proposals + approval audit trail

    # ── Political event pattern learning system ───────────────────────────────
    "news_effect_patterns",   # validated political patterns — seeded + council-updated
    "pattern_validation_log", # lag predictions + accuracy outcomes — irreplaceable
]

# Primary keys per table — used for ON CONFLICT
PRIMARY_KEYS = {
    "settings":                  ["key"],
    "financials":                ["id"],
    "nrb_monthly":               ["id"],
    "fd_rate_summary":           ["id"],
    "learning_hub":              ["id"],
    "daily_context_log":         ["date"],
    "market_log":                ["id"],
    "trade_journal":             ["id"],
    "gate_misses":               ["id"],
    "gate_proposals":            ["id"],
    "claude_audit":              ["id"],
    "paper_capital":             ["telegram_id"],
    "paper_portfolio":           ["id"],
    "paper_trade_log":           ["id"],
    "monthly_council_agenda":    ["id"],
    "monthly_council_log":       ["id"],
    "monthly_council_checklist": ["id"],
    "monthly_override":          ["id"],
    "accuracy_review_log":       ["id"],
    "system_proposals":          ["id"],
    "news_effect_patterns":      ["id"],
    "pattern_validation_log":    ["id"],
}


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _connect(url: str, label: str):
    """Open psycopg2 connection. Returns connection or None on failure."""
    try:
        conn = psycopg2.connect(url)
        conn.autocommit = False
        return conn
    except Exception as e:
        log.error("Cannot connect to %s: %s", label, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA SYNC
# ─────────────────────────────────────────────────────────────────────────────

def _get_local_columns(local_conn, table: str) -> list[dict]:
    """
    Get column definitions from local DB for a table.
    Returns list of {column_name, data_type, is_nullable, column_default}.
    """
    with local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_name = %s
              AND table_schema = 'public'
            ORDER BY ordinal_position
        """, (table,))
        return cur.fetchall()


def _get_neon_columns(neon_conn, table: str) -> set[str]:
    """Get existing column names from Neon table."""
    try:
        with neon_conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                  AND table_schema = 'public'
            """, (table,))
            return {row[0] for row in cur.fetchall()}
    except Exception:
        return set()


def _map_type(pg_type: str) -> str:
    """Map PostgreSQL data type to safe DDL type."""
    mapping = {
        "integer":                     "INTEGER",
        "bigint":                      "BIGINT",
        "smallint":                    "SMALLINT",
        "numeric":                     "NUMERIC",
        "double precision":            "DOUBLE PRECISION",
        "real":                        "REAL",
        "boolean":                     "BOOLEAN",
        "text":                        "TEXT",
        "character varying":           "TEXT",
        "character":                   "TEXT",
        "timestamp without time zone": "TIMESTAMP",
        "timestamp with time zone":    "TIMESTAMPTZ",
        "date":                        "DATE",
        "time without time zone":      "TIME",
        "json":                        "JSON",
        "jsonb":                       "JSONB",
        "uuid":                        "UUID",
    }
    return mapping.get(pg_type.lower(), "TEXT")


def _ensure_table(local_conn, neon_conn, table: str) -> bool:
    """
    CREATE TABLE IF NOT EXISTS on Neon using local schema.
    Then ALTER TABLE to add any missing columns.
    Returns True if table is ready.
    """
    columns = _get_local_columns(local_conn, table)
    if not columns:
        log.warning("Table %s has no columns in local DB — skipping", table)
        return False

    pks = PRIMARY_KEYS.get(table, ["id"])

    # Build column definitions
    col_defs = []
    for col in columns:
        name     = col["column_name"]
        dtype    = _map_type(col["data_type"])
        nullable = "" if col["is_nullable"] == "YES" else " NOT NULL"

        # Handle serial/sequence defaults — use BIGSERIAL for id columns
        default = col.get("column_default") or ""
        if "nextval" in str(default) and name == "id":
            col_defs.append(f'"{name}" BIGSERIAL')
        else:
            col_defs.append(f'"{name}" {dtype}{nullable}')

    # Add primary key constraint
    pk_str   = ", ".join(f'"{pk}"' for pk in pks)
    col_defs.append(f"PRIMARY KEY ({pk_str})")

    ddl = (
        f'CREATE TABLE IF NOT EXISTS "{table}" (\n'
        f'  {chr(10)+"  ,".join(col_defs)}\n'
        f');'
    )

    try:
        with neon_conn.cursor() as cur:
            cur.execute(ddl)
        neon_conn.commit()
        log.info("✅ Table %s ensured on Neon", table)
    except Exception as e:
        neon_conn.rollback()
        log.error("Failed to create table %s on Neon: %s", table, e)
        return False

    # Add any missing columns (schema drift)
    neon_cols = _get_neon_columns(neon_conn, table)
    for col in columns:
        name  = col["column_name"]
        dtype = _map_type(col["data_type"])
        if name not in neon_cols:
            try:
                with neon_conn.cursor() as cur:
                    cur.execute(
                        f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{name}" {dtype};'
                    )
                neon_conn.commit()
                log.info("Added missing column %s.%s to Neon", table, name)
            except Exception as e:
                neon_conn.rollback()
                log.warning("Could not add column %s.%s: %s", table, name, e)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# DATA SYNC
# ─────────────────────────────────────────────────────────────────────────────

def _sync_table(local_conn, neon_conn, table: str) -> dict:
    """
    Upsert all rows from local → Neon for one table.
    Returns stats: {upserted, errors}
    """
    pks   = PRIMARY_KEYS.get(table, ["id"])
    stats = {"upserted": 0, "errors": 0}

    # Fetch all rows from local
    try:
        with local_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f'SELECT * FROM "{table}"')
            rows = cur.fetchall()
    except Exception as e:
        log.error("Failed to read %s from local: %s", table, e)
        stats["errors"] += 1
        return stats

    if not rows:
        log.info("Table %s — 0 rows (empty)", table)
        return stats

    # Get Neon columns to filter out any columns that don't exist yet
    neon_cols = _get_neon_columns(neon_conn, table)

    # Build upsert SQL — filter to Neon-existing cols only
    all_cols = [c for c in rows[0].keys() if c in neon_cols]
    if not all_cols:
        log.warning("No matching columns for %s — skipping", table)
        return stats

    col_names  = [f'"{c}"' for c in all_cols]
    col_values = [f"%({c})s" for c in all_cols]

    # ON CONFLICT DO UPDATE — update all non-PK columns
    update_cols = [c for c in all_cols if c not in pks]
    if update_cols:
        update_str      = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        conflict_action = f"DO UPDATE SET {update_str}"
    else:
        conflict_action = "DO NOTHING"

    pk_conflict = ", ".join(f'"{pk}"' for pk in pks)

    sql = (
        f'INSERT INTO "{table}" ({", ".join(col_names)}) '
        f'VALUES ({", ".join(col_values)}) '
        f'ON CONFLICT ({pk_conflict}) {conflict_action}'
    )

    # Upsert in batches of 500
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        batch    = rows[i:i + BATCH]
        filtered = [{c: row.get(c) for c in all_cols} for row in batch]
        try:
            with neon_conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, sql, filtered, page_size=BATCH)
            neon_conn.commit()
            stats["upserted"] += len(batch)
        except Exception as e:
            neon_conn.rollback()
            log.error(
                "Batch upsert failed for %s (rows %d-%d): %s",
                table, i, i + BATCH, e,
            )
            stats["errors"] += len(batch)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM ALERT
# ─────────────────────────────────────────────────────────────────────────────

def _alert_admin(message: str) -> None:
    """Send backup failure alert to admin via error bot."""
    try:
        import requests
        token   = os.getenv("TELEGRAM_ERROR_BOT", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":    chat_id,
                "text":       f"🔴 *NEPSE Backup Sync Failed*\n\n{message}",
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
    except Exception as e:
        log.error("Failed to send Telegram alert: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run(tables: Optional[list[str]] = None, dry_run: bool = False) -> bool:
    """
    Main sync runner.

    Args:
        tables:  Specific tables to sync. None = all BACKUP_TABLES.
        dry_run: If True, connects and checks schema but does not write.

    Returns:
        True if all tables synced successfully, False if any errors.
    """
    nst_now = datetime.now(tz=NST)
    log.info("=" * 60)
    log.info("BACKUP SYNC starting — %s NST", nst_now.strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    if not LOCAL_URL:
        log.error("DATABASE_URL not set in .env")
        return False
    if not NEON_URL:
        log.error("DATABASE_URL_NEON not set in .env")
        _alert_admin("DATABASE_URL_NEON not configured — backup cannot run.")
        return False

    # Connect to both
    local_conn = _connect(LOCAL_URL, "local")
    neon_conn  = _connect(NEON_URL,  "Neon")

    if not local_conn:
        _alert_admin("Cannot connect to local PostgreSQL — backup aborted.")
        return False
    if not neon_conn:
        _alert_admin("Cannot connect to Neon PostgreSQL — backup aborted.")
        if local_conn:
            local_conn.close()
        return False

    target_tables = tables or BACKUP_TABLES
    total_stats   = {"upserted": 0, "errors": 0}
    failed_tables = []

    try:
        for table in target_tables:
            log.info("─── Syncing: %s", table)

            # Ensure table exists on Neon with correct schema
            ready = _ensure_table(local_conn, neon_conn, table)
            if not ready:
                failed_tables.append(table)
                total_stats["errors"] += 1
                continue

            if dry_run:
                log.info("[DRY-RUN] Would sync %s — schema check only", table)
                continue

            # Sync data
            stats = _sync_table(local_conn, neon_conn, table)
            total_stats["upserted"] += stats["upserted"]
            total_stats["errors"]   += stats["errors"]

            if stats["errors"] > 0:
                failed_tables.append(table)
                log.warning(
                    "Table %s: %d upserted, %d errors",
                    table, stats["upserted"], stats["errors"],
                )
            else:
                log.info(
                    "Table %s: ✅ %d rows synced",
                    table, stats["upserted"],
                )

    finally:
        local_conn.close()
        neon_conn.close()

    # Summary
    log.info("=" * 60)
    log.info(
        "BACKUP SYNC complete — %d rows upserted | %d errors | %d tables failed",
        total_stats["upserted"], total_stats["errors"], len(failed_tables),
    )
    log.info("=" * 60)

    if failed_tables:
        msg = (
            f"Backup sync completed with errors.\n"
            f"Failed tables: {', '.join(failed_tables)}\n"
            f"Total errors: {total_stats['errors']}"
        )
        _alert_admin(msg)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEPSE Backup Sync — local → Neon")
    parser.add_argument(
        "--tables", nargs="+", default=None,
        help="Specific tables to sync (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check schema only — no data writes"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all tables that will be backed up"
    )
    args = parser.parse_args()

    if args.list:
        print("\nTables scheduled for backup:")
        for t in BACKUP_TABLES:
            pks = PRIMARY_KEYS.get(t, ["id"])
            print(f"  {t:<30} PK: {pks}")
        sys.exit(0)

    success = run(tables=args.tables, dry_run=args.dry_run)
    sys.exit(0 if success else 1)