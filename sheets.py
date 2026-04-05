"""
db/sheets.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Public database API.

Drop-in replacement for the old Google Sheets sheets.py.
Every function signature is IDENTICAL — no other module needs changes.

All reads/writes go to Neon PostgreSQL via db/connection.py.
Table definitions live in db/schema.py.
Schema versioning lives in db/migrations.py.

Import pattern (unchanged from old sheets.py):
    from db import get_setting, write_row, read_tab
    from db import update_setting, write_signal, update_trade_outcome
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from db.connection import _db
from db.schema import TABS, TABLE_COLUMNS


log = logging.getLogger(__name__)

# Nepal Standard Time offset for timestamps
NST_OFFSET = "+05:45"


def _now_nst() -> str:
    """Current datetime as NST string for timestamp columns."""
    from datetime import timezone, timedelta
    nst = timezone(timedelta(hours=5, minutes=45))
    return datetime.now(tz=nst).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_table(tab_key: str) -> str:
    """Resolve TABS key or literal table name → lowercase table name."""
    return TABS.get(tab_key, tab_key).lower()


def _safe_columns(table: str) -> list[str]:
    """Get column list for table — raises if unknown."""
    cols = TABLE_COLUMNS.get(table)
    if cols is None:
        raise ValueError(
            f"Unknown table: '{table}'. "
            f"Add it to db/schema.py TABLE_COLUMNS."
        )
    return cols


# ══════════════════════════════════════════════════════════════════════════════
# CORE READ/WRITE — used everywhere
# ══════════════════════════════════════════════════════════════════════════════

def read_tab(tab_key: str, limit: int = None) -> list[dict]:
    """
    Read all rows from a table. Returns list of dicts.
    Equivalent to old: worksheet.get_all_records()

    Args:
        tab_key: Key from TABS dict (e.g. "market_log") or literal table name
        limit:   Optional row limit (most recent first)

    Returns:
        List of row dicts. Empty list if table empty or missing.

    Example:
        rows = read_tab("watchlist")
        for row in rows:
            print(row["symbol"], row["sector"])
    """
    table = _resolve_table(tab_key)
    try:
        with _db() as cur:
            if limit:
                cur.execute(
                    f'SELECT * FROM "{table}" ORDER BY id DESC LIMIT %s',
                    (limit,),
                )
            else:
                cur.execute(f'SELECT * FROM "{table}" ORDER BY id')
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log.error("read_tab(%s) failed: %s", table, e)
        return []


def read_tab_where(
    tab_key: str,
    filters: dict[str, Any],
    limit: int = None,
    order_by: str = "id",
    desc: bool = False,
) -> list[dict]:
    """
    Read rows matching filter conditions.

    Args:
        tab_key:  Table key or name
        filters:  Dict of {column: value} — ANDed together
        limit:    Optional row limit
        order_by: Column to sort by (default "id")
        desc:     Sort descending if True

    Returns:
        List of matching row dicts

    Example:
        pending = read_tab_where("market_log", {"outcome": "PENDING"})
        nabil   = read_tab_where("portfolio", {"symbol": "NABIL", "status": "OPEN"})
    """
    table = _resolve_table(tab_key)
    try:
        where_parts = [f'"{col}" = %s' for col in filters]
        where_sql   = " AND ".join(where_parts) if where_parts else "TRUE"
        order_dir   = "DESC" if desc else "ASC"
        limit_sql   = f"LIMIT {int(limit)}" if limit else ""

        sql = f'''
            SELECT * FROM "{table}"
            WHERE {where_sql}
            ORDER BY "{order_by}" {order_dir}
            {limit_sql}
        '''
        with _db() as cur:
            cur.execute(sql, list(filters.values()))
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("read_tab_where(%s, %s) failed: %s", table, filters, e)
        return []


def write_row(tab_key: str, row_data: dict) -> bool:
    """
    Insert a single row. Ignores unknown columns gracefully.
    Equivalent to old: worksheet.append_row(...)

    Args:
        tab_key:  Table key or name
        row_data: Dict of {column: value}. Unknown columns skipped.

    Returns:
        True on success

    Example:
        write_row("market_log", {
            "symbol": "NABIL",
            "action": "BUY",
            "confidence": "78",
            "entry_price": "1245",
            "timestamp": "2026-03-14 11:30:00",
        })
    """
    table   = _resolve_table(tab_key)
    columns = _safe_columns(table)

    # Auto-timestamp if column exists and value not provided
    if "timestamp" in columns and "timestamp" not in row_data:
        row_data = {**row_data, "timestamp": _now_nst()}

    # Only insert columns that exist in schema
    valid_cols = [c for c in columns if c in row_data]
    if not valid_cols:
        log.warning("write_row(%s): no valid columns in row_data", table)
        return False

    col_sql  = ", ".join(f'"{c}"' for c in valid_cols)
    val_sql  = ", ".join("%s" for _ in valid_cols)
    values   = [str(row_data[c]) if row_data[c] is not None else None
                for c in valid_cols]

    try:
        with _db() as cur:
            cur.execute(
                f'INSERT INTO "{table}" ({col_sql}) VALUES ({val_sql})',
                values,
            )
        log.info("write_row(%s): inserted %d columns", table, len(valid_cols))
        return True
    except Exception as e:
        log.error("write_row(%s) failed: %s", table, e)
        return False


def upsert_row(
    tab_key: str,
    row_data: dict,
    conflict_columns: list[str],
) -> bool:
    """
    Insert or update a row on conflict.
    Essential for INDICATORS (symbol+date unique) and SETTINGS (key unique).

    Args:
        tab_key:          Table key or name
        row_data:         Dict of {column: value}
        conflict_columns: Columns that define uniqueness (ON CONFLICT target)

    Returns:
        True on success

    Example:
        # Upsert today's indicators for NABIL
        upsert_row("indicators", {
            "symbol": "NABIL",
            "date": "2026-03-14",
            "rsi_14": "58.6",
            "tech_signal": "BULLISH",
        }, conflict_columns=["symbol", "date"])

        # Upsert a setting
        upsert_row("settings", {
            "key": "PAPER_MODE",
            "value": "false",
        }, conflict_columns=["key"])
    """
    table   = _resolve_table(tab_key)
    columns = _safe_columns(table)

    if "timestamp" in columns and "timestamp" not in row_data:
        row_data = {**row_data, "timestamp": _now_nst()}

    valid_cols = [c for c in columns if c in row_data]
    if not valid_cols:
        log.warning("upsert_row(%s): no valid columns", table)
        return False

    col_sql      = ", ".join(f'"{c}"' for c in valid_cols)
    val_sql      = ", ".join("%s" for _ in valid_cols)
    conflict_sql = ", ".join(f'"{c}"' for c in conflict_columns)
    update_cols  = [c for c in valid_cols if c not in conflict_columns]
    update_sql   = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    if not update_sql:
        update_sql = f'"{conflict_columns[0]}" = EXCLUDED."{conflict_columns[0]}"'

    values = [str(row_data[c]) if row_data[c] is not None else None
              for c in valid_cols]

    try:
        with _db() as cur:
            cur.execute(
                f'''
                INSERT INTO "{table}" ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT ({conflict_sql})
                DO UPDATE SET {update_sql}
                ''',
                values,
            )
        return True
    except Exception as e:
        log.error("upsert_row(%s) failed: %s", table, e)
        return False


def batch_write(tab_key: str, rows: list[dict]) -> int:
    """
    Bulk insert multiple rows in a single transaction.
    Used by indicators.py (335 rows/day), scraper.py market breadth, etc.

    Args:
        tab_key: Table key or name
        rows:    List of row dicts

    Returns:
        Number of rows successfully inserted

    Example:
        results = compute_all_indicators(market_data, cache)
        rows = [dataclasses.asdict(r) for r in results.values()]
        inserted = batch_write("indicators", rows)
    """
    if not rows:
        return 0

    table   = _resolve_table(tab_key)
    columns = _safe_columns(table)

    if not columns:
        log.error("batch_write(%s): no columns defined", table)
        return 0

    # Determine valid columns from first row
    sample     = rows[0]
    valid_cols = [c for c in columns if c in sample]
    if not valid_cols:
        log.warning("batch_write(%s): no matching columns", table)
        return 0

    col_sql = ", ".join(f'"{c}"' for c in valid_cols)
    val_sql = ", ".join("%s" for _ in valid_cols)

    def _row_values(row: dict) -> list:
        return [str(row[c]) if row.get(c) is not None else None
                for c in valid_cols]

    try:
        import psycopg2.extras
        all_values = [_row_values(r) for r in rows]
        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f'INSERT INTO "{table}" ({col_sql}) VALUES ({val_sql})',
                all_values,
                page_size=100,
            )
        log.info("batch_write(%s): %d rows inserted", table, len(rows))
        return len(rows)
    except Exception as e:
        log.error("batch_write(%s) failed: %s", table, e)
        return 0


def batch_upsert(
    tab_key: str,
    rows: list[dict],
    conflict_columns: list[str],
) -> int:
    """
    Bulk upsert. Used by indicators.py — overwrites today's indicators
    on re-run without creating duplicates.

    Args:
        tab_key:          Table key or name
        rows:             List of row dicts
        conflict_columns: Columns that define uniqueness

    Returns:
        Number of rows upserted

    Example:
        batch_upsert("indicators", indicator_rows,
                     conflict_columns=["symbol", "date"])
    """
    if not rows:
        return 0

    table   = _resolve_table(tab_key)
    columns = _safe_columns(table)
    sample  = rows[0]

    valid_cols   = [c for c in columns if c in sample]
    col_sql      = ", ".join(f'"{c}"' for c in valid_cols)
    val_sql      = ", ".join("%s" for _ in valid_cols)
    conflict_sql = ", ".join(f'"{c}"' for c in conflict_columns)
    update_cols  = [c for c in valid_cols if c not in conflict_columns]
    update_sql   = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)

    if not update_sql:
        update_sql = f'"{conflict_columns[0]}" = EXCLUDED."{conflict_columns[0]}"'

    def _row_values(row: dict) -> list:
        return [str(row[c]) if row.get(c) is not None else None
                for c in valid_cols]

    try:
        import psycopg2.extras
        all_values = [_row_values(r) for r in rows]
        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f'''
                INSERT INTO "{table}" ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT ({conflict_sql})
                DO UPDATE SET {update_sql}
                ''',
                all_values,
                page_size=100,
            )
        log.info("batch_upsert(%s): %d rows", table, len(rows))
        return len(rows)
    except Exception as e:
        log.error("batch_upsert(%s) failed: %s", table, e)
        return 0



def update_row(
    tab_key: str,
    updates: dict[str, Any],
    where: dict[str, Any],
) -> int:
    """
    Update columns in rows matching where clause.

    Args:
        tab_key:  Table key or name
        updates:  Dict of {column: new_value}
        where:    Dict of {column: match_value} — ANDed

    Returns:
        Number of rows updated

    Example:
        # Close a trade
        update_row("market_log",
            updates={"outcome": "WIN", "actual_pnl": "3450", "exit_date": "2026-03-14"},
            where={"symbol": "NABIL", "outcome": "PENDING"},
        )
    """
    table = _resolve_table(tab_key)
    if not updates or not where:
        log.warning("update_row(%s): empty updates or where clause", table)
        return 0

    set_parts   = [f'"{k}" = %s' for k in updates]
    where_parts = [f'"{k}" = %s' for k in where]
    set_sql     = ", ".join(set_parts)
    where_sql   = " AND ".join(where_parts)
    values      = list(updates.values()) + list(where.values())

    try:
        with _db() as cur:
            cur.execute(
                f'UPDATE "{table}" SET {set_sql} WHERE {where_sql}',
                values,
            )
            count = cur.rowcount
        log.info("update_row(%s): %d rows updated", table, count)
        return count
    except Exception as e:
        log.error("update_row(%s) failed: %s", table, e)
        return 0


def delete_rows(tab_key: str, where: dict[str, Any]) -> int:
    """
    Delete rows matching where clause.

    Args:
        tab_key: Table key or name
        where:   Dict of {column: value} — ANDed

    Returns:
        Number of rows deleted

    Example:
        delete_rows("corporate_events", {"status": "EXPIRED"})
    """
    table = _resolve_table(tab_key)
    if not where:
        log.error("delete_rows(%s): refusing delete with empty where clause", table)
        return 0

    where_parts = [f'"{k}" = %s' for k in where]
    where_sql   = " AND ".join(where_parts)

    try:
        with _db() as cur:
            cur.execute(
                f'DELETE FROM "{table}" WHERE {where_sql}',
                list(where.values()),
            )
            count = cur.rowcount
        log.info("delete_rows(%s): %d rows deleted", table, count)
        return count
    except Exception as e:
        log.error("delete_rows(%s) failed: %s", table, e)
        return 0


def clear_table(tab_key: str) -> bool:
    """
    Delete ALL rows from a table (keeps structure).
    Used by indicators.py to wipe yesterday's data before writing fresh.

    Args:
        tab_key: Table key or name

    Returns:
        True on success
    """
    table = _resolve_table(tab_key)
    try:
        with _db() as cur:
            cur.execute(f'DELETE FROM "{table}"')
            count = cur.rowcount
        log.info("clear_table(%s): deleted %d rows", table, count)
        return True
    except Exception as e:
        log.error("clear_table(%s) failed: %s", table, e)
        return False

# ══════════════════════════════════════════════════════════════════════════════
# NEPAL PULSE.
# ══════════════════════════════════════════════════════════════════════════════

def write_nepal_pulse(pulse_data: dict) -> bool:
    """Write a Nepal domestic market snapshot. Called by nepal_pulse.py."""
    return write_row("nepal_pulse", pulse_data)

def get_latest_pulse() -> Optional[dict]:
    """Return the most recent Nepal pulse snapshot."""
    rows = read_tab("nepal_pulse", limit=1)
    return rows[0] if rows else None

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS — get/set key-value config
# ══════════════════════════════════════════════════════════════════════════════

def get_setting(key: str, default: str = "") -> str:
    """
    Get a setting value by key.

    Args:
        key:     Setting key (e.g. "PAPER_MODE", "CAPITAL_TOTAL_NPR")
        default: Returned if key not found

    Returns:
        String value, or default

    Example:
        paper_mode = get_setting("PAPER_MODE", "true") == "true"
        capital    = float(get_setting("CAPITAL_TOTAL_NPR", "100000"))
    """
    try:
        with _db() as cur:
            cur.execute(
                'SELECT value FROM settings WHERE key = %s',
                (key,),
            )
            row = cur.fetchone()
            return row["value"] if row else default
    except Exception as e:
        log.error("get_setting(%s) failed: %s", key, e)
        return default


def update_setting(key: str, value: str, set_by: str = "system") -> bool:
    """
    Update or insert a setting.

    Args:
        key:    Setting key
        value:  New value (always stored as string)
        set_by: Who changed it ("system" or "user")

    Returns:
        True on success

    Example:
        update_setting("LAST_MACRO_UPDATE", "2026-03-01", set_by="user")
        update_setting("PAPER_MODE", "false", set_by="user")
    """
    try:
        with _db() as cur:
            cur.execute(
                """
                INSERT INTO settings (key, value, last_updated, set_by)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                    SET value        = EXCLUDED.value,
                        last_updated = EXCLUDED.last_updated,
                        set_by       = EXCLUDED.set_by
                """,
                (key, str(value), _now_nst(), set_by),
            )
        log.info("update_setting: %s = %s (by %s)", key, value, set_by)
        return True
    except Exception as e:
        log.error("update_setting(%s) failed: %s", key, e)
        return False


def get_all_settings() -> dict[str, str]:
    """
    Return all settings as a flat dict.

    Example:
        cfg = get_all_settings()
        if cfg.get("PAPER_MODE") == "true":
            ...
    """
    try:
        with _db() as cur:
            cur.execute("SELECT key, value FROM settings ORDER BY key")
            return {r["key"]: r["value"] for r in cur.fetchall()}
    except Exception as e:
        log.error("get_all_settings failed: %s", e)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC HELPERS
# These wrap the core functions with business logic.
# ══════════════════════════════════════════════════════════════════════════════

def write_signal(signal_data: dict) -> bool:
    """
    Write a trading signal to market_log.
    Timestamps automatically. Outcome defaults to PENDING.

    Args:
        signal_data: Dict matching market_log columns

    Returns:
        True on success

    Example:
        write_signal({
            "symbol": "NABIL",
            "action": "BUY",
            "confidence": "78",
            "entry_price": "1245",
            "stop_loss": "1183",
            "target": "1370",
            "allocation_npr": "10000",
            "rsi_14": "42.3",
            "tech_signal": "BULLISH",
            "reasoning": "RSI oversold + golden cross + volume surge",
        })
    """
    from datetime import timezone, timedelta
    nst = timezone(timedelta(hours=5, minutes=45))
    now = datetime.now(tz=nst)

    data = {
        "date":      now.strftime("%Y-%m-%d"),
        "time":      now.strftime("%H:%M:%S"),
        "outcome":   "PENDING",
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        **signal_data,
    }
    return write_row("market_log", data)


def update_trade_outcome(
    symbol: str,
    entry_date: str,
    outcome: str,
    actual_pnl: float = 0.0,
    exit_price: float = 0.0,
    exit_reason: str = "",
) -> bool:
    """
    Update the outcome of a PENDING trade in market_log.

    Args:
        symbol:     Stock symbol
        entry_date: Date the trade was entered (YYYY-MM-DD)
        outcome:    "WIN", "LOSS", or "BREAKEVEN"
        actual_pnl: Actual profit/loss in NPR
        exit_price: Price at exit
        exit_reason: Why the trade was closed

    Returns:
        True if a row was updated

    Example:
        update_trade_outcome("NABIL", "2026-03-10", "WIN",
                             actual_pnl=3450, exit_price=1312,
                             exit_reason="Target hit")
    """
    count = update_row(
        "market_log",
        updates={
            "outcome":     outcome,
            "actual_pnl":  str(actual_pnl),
            "exit_price":  str(exit_price),
            "exit_date":   _now_nst()[:10],
            "exit_reason": exit_reason,
        },
        where={
            "symbol":  symbol,
            "date":    entry_date,
            "outcome": "PENDING",
        },
    )
    if count == 0:
        log.warning(
            "update_trade_outcome: no PENDING trade found for %s on %s",
            symbol, entry_date,
        )
    return count > 0


def get_open_positions() -> list[dict]:
    """
    Return all OPEN positions from portfolio table.

    Example:
        positions = get_open_positions()
        for p in positions:
            print(p["symbol"], p["pnl_pct"])
    """
    return read_tab_where("portfolio", {"status": "OPEN"})


def get_pending_trades() -> list[dict]:
    """Return all trades with outcome = PENDING from market_log."""
    return read_tab_where("market_log", {"outcome": "PENDING"})


def get_watchlist() -> list[dict]:
    """
    Return share_sectors as sector reference.
    Legacy name kept for backward compatibility.
    Watchlist table was dropped — share_sectors is the source of truth.
    """
    return read_tab("share_sectors")


def get_watchlist_symbols() -> list[str]:
    """
    Return all stock symbols from share_sectors.
    Legacy name kept for backward compatibility.
    """
    rows = read_tab("share_sectors")
    return [r["symbol"] for r in rows if r.get("symbol")]


def write_market_breadth(breadth_data: dict) -> bool:
    """
    Write market breadth snapshot. Upserts on date to avoid duplicates.

    Example:
        write_market_breadth({
            "date": "2026-03-14",
            "advancing": "180",
            "declining": "120",
            "breadth_score": "65",
            "market_signal": "BULLISH",
        })
    """
    return upsert_row("market_breadth", breadth_data, conflict_columns=["date"])


def write_indicators_batch(indicator_rows: list[dict]) -> int:
    """
    Write all indicator results for the day. Upserts on symbol+date.
    Called once daily by indicators.py run_daily_indicators().

    Args:
        indicator_rows: List of dicts from IndicatorResult dataclass

    Returns:
        Number of rows written

    Example:
        rows = [dataclasses.asdict(r) for r in results.values()]
        write_indicators_batch(rows)
    """
    return batch_upsert("indicators", indicator_rows,
                        conflict_columns=["symbol", "date"])


def read_today_indicators(date: str = None) -> dict[str, dict]:
    """
    Read today's pre-computed indicators.
    Called by filter_engine.py during the 6-min trading loop.

    Args:
        date: Date string YYYY-MM-DD (defaults to today NST)

    Returns:
        Dict[symbol, row_dict]

    Example:
        indicators = read_today_indicators()
        nabil = indicators.get("NABIL")
        if nabil and float(nabil["rsi_14"]) < 40:
            ...
    """
    from datetime import timezone, timedelta
    if date is None:
        nst  = timezone(timedelta(hours=5, minutes=45))
        date = datetime.now(tz=nst).strftime("%Y-%m-%d")

    rows = read_tab_where("indicators", {"date": date})
    return {r["symbol"]: r for r in rows if r.get("symbol")}


def write_geo_snapshot(geo_data: dict) -> bool:
    """
    Write a geopolitical/macro market snapshot.
    Called every 30 min by geo_sentiment.py.

    Example:
        write_geo_snapshot({
            "crude_price": "74.2",
            "vix": "18.4",
            "nifty": "22150",
            "geo_score": "65",
        })
    """
    return write_row("geo_data", geo_data)


def get_latest_geo() -> Optional[dict]:
    """Return the most recent geopolitical snapshot."""
    rows = read_tab("geo_data", limit=1)
    return rows[0] if rows else None


def get_macro_data() -> dict[str, str]:
    """
    Return all macro indicators as a flat {indicator: value} dict.
    Used by filter_engine.py and signal.py.

    Example:
        macro = get_macro_data()
        policy_rate = float(macro.get("Policy_Rate", "5.5"))
    """
    try:
        with _db() as cur:
            cur.execute("SELECT indicator, value FROM macro_data")
            return {r["indicator"]: r["value"] for r in cur.fetchall()}
    except Exception as e:
        log.error("get_macro_data failed: %s", e)
        return {}


def upsert_macro(indicator: str, value: str, source: str = "", unit: str = "") -> bool:
    """
    Insert or update a single macro indicator.
    Called when manually entering NRB data.

    Example:
        upsert_macro("Policy_Rate", "5.0", source="NRB", unit="%")
        upsert_macro("CPI_YoY", "4.8", source="NRB", unit="%")
    """
    return upsert_row("macro_data", {
        "indicator":    indicator,
        "value":        value,
        "source":       source,
        "unit":         unit,
        "last_updated": _now_nst(),
    }, conflict_columns=["indicator"])


def write_candle_pattern(pattern_data: dict) -> bool:
    """
    Insert or update a candle pattern definition.
    Upserts on pattern_name.

    Example:
        write_candle_pattern({
            "pattern_name": "Bullish Engulfing",
            "type": "BULLISH",
            "tier": "1",
            "nepal_win_rate_pct": "68",
        })
    """
    return upsert_row("candle_patterns", pattern_data,
                      conflict_columns=["pattern_name"])


def log_lesson(lesson_data: dict) -> bool:
    """
    Write a lesson to learning_hub.

    Example:
        log_lesson({
            "symbol": "NABIL",
            "pattern": "Bullish Engulfing",
            "lesson": "Works best when RSI < 45 and volume > 2x",
            "outcome": "WIN",
            "pnl_npr": "3450",
        })
    """
    return write_row("learning_hub", lesson_data)

#=============================================================================
# NEPSE INDICES — add these functions to sheets.py
# Section: after write_market_breadth(), before initialize_all_tabs()
# =============================================================================
 
def write_index_batch(rows: list[dict]) -> int:
    """
    Bulk upsert index rows into nepse_indices table.
    ON CONFLICT (date, index_id) DO UPDATE — safe to re-run.
 
    Uses psycopg2.extras.execute_batch for performance.
    Called by modules/index_scraper.py only.
 
    Args:
        rows: List of dicts with keys:
              date, index_id, index_name, current_value,
              change_abs, change_pct, turnover, source
 
    Returns:
        Number of rows written.
 
    Example:
        from sheets import write_index_batch
        write_index_batch(parsed_rows)
    """
    if not rows:
        return 0
    try:
        import psycopg2.extras
 
        columns = ["date", "index_id", "index_name", "current_value",
                   "change_abs", "change_pct", "turnover", "source"]
        values  = [tuple(r.get(c, "") for c in columns) for r in rows]
        col_sql = ", ".join(f'"{c}"' for c in columns)
        val_sql = ", ".join(["%s"] * len(columns))
        upd_sql = ", ".join(
            f'"{c}" = EXCLUDED."{c}"'
            for c in columns if c not in ("date", "index_id")
        )
 
        with _db() as cur:
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO nepse_indices ({col_sql})
                VALUES ({val_sql})
                ON CONFLICT (date, index_id)
                DO UPDATE SET {upd_sql}
                """,
                values,
                page_size=500,
            )
        log.info("write_index_batch: %d rows upserted", len(rows))
        return len(rows)
    except Exception as e:
        log.error("write_index_batch failed: %s", e)
        return 0
 
 
def get_latest_index_date(index_id: int) -> Optional[str]:
    """
    Return the most recent date stored for one index_id.
    Used by index_scraper.py for incremental fetch (resume-friendly).
 
    Args:
        index_id: 1-17
 
    Returns:
        Date string YYYY-MM-DD or None if no data yet.
 
    Example:
        latest = get_latest_index_date(1)   # NEPSE composite
        latest = get_latest_index_date(2)   # Banking
    """
    try:
        with _db() as cur:
            cur.execute(
                "SELECT MAX(date) AS latest FROM nepse_indices WHERE index_id = %s",
                (str(index_id),)
            )
            row = cur.fetchone()
            return row["latest"] if row and row["latest"] else None
    except Exception as e:
        log.error("get_latest_index_date(%s) failed: %s", index_id, e)
        return None
 
 
def get_index_coverage() -> list[dict]:
    """
    Return row count + date range per index.
    Used by index_scraper.py print_status() for CLI display.
 
    Returns:
        List of dicts: {index_id, index_name, row_count, earliest, latest}
        Sorted by index_id ascending.
 
    Example:
        rows = get_index_coverage()
        for r in rows:
            print(r["index_name"], r["row_count"], r["earliest"], r["latest"])
    """
    try:
        with _db() as cur:
            cur.execute("""
                SELECT
                    index_id::int  AS index_id,
                    index_name,
                    COUNT(*)       AS row_count,
                    MIN(date)      AS earliest,
                    MAX(date)      AS latest
                FROM nepse_indices
                GROUP BY index_id, index_name
                ORDER BY index_id
            """)
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("get_index_coverage failed: %s", e)
        return []
 
 
def read_index_history(
    index_id:  int,
    from_date: str = None,
    to_date:   str = None,
) -> list[dict]:
    """
    Read stored data for one index_id, sorted ascending by date.
    Used by backtester.py and statistical analysis.
 
    Args:
        index_id:  1-17
        from_date: YYYY-MM-DD (optional)
        to_date:   YYYY-MM-DD (optional)
 
    Returns:
        List of row dicts sorted by date ASC.
 
    Example:
        from sheets import read_index_history
        nepse   = read_index_history(1, "2023-07-15", "2026-03-20")
        banking = read_index_history(2)
    """
    try:
        conditions = ["index_id = %s"]
        params     = [str(index_id)]
        if from_date:
            conditions.append("date >= %s")
            params.append(from_date)
        if to_date:
            conditions.append("date <= %s")
            params.append(to_date)
        where = " AND ".join(conditions)
        with _db() as cur:
            cur.execute(
                f"SELECT * FROM nepse_indices WHERE {where} ORDER BY date ASC",
                tuple(params)
            )
            return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.error("read_index_history(%s) failed: %s", index_id, e)
        return []
 
 
def read_all_indices_wide(
    from_date: str = None,
    to_date:   str = None,
) -> list[dict]:
    """
    Read all 17 indices pivoted to wide format.
    One row per date, one column per index name.
    Ready for pandas DataFrame and Spearman correlation matrix.
 
    Returns:
        List of dicts:
        {"date": "2024-01-15", "NEPSE": 2145.23, "Banking": 1432.10, ...}
 
    Example (in backtester.py):
        from sheets import read_all_indices_wide
        import pandas as pd
 
        rows = read_all_indices_wide("2023-07-15", "2026-03-20")
        df   = pd.DataFrame(rows).set_index("date")
        corr = df.pct_change().corr(method="spearman")
        print(corr["NEPSE"].sort_values())
    """
    try:
        conditions = []
        params     = []
        if from_date:
            conditions.append("date >= %s")
            params.append(from_date)
        if to_date:
            conditions.append("date <= %s")
            params.append(to_date)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
 
        with _db() as cur:
            cur.execute(
                f"SELECT date, index_name, current_value "
                f"FROM nepse_indices {where} ORDER BY date, index_id",
                tuple(params) if params else None
            )
            rows = [dict(r) for r in cur.fetchall()]
 
        from collections import defaultdict
        by_date: dict[str, dict] = defaultdict(dict)
        for row in rows:
            try:
                val = float(row["current_value"]) if row["current_value"] else None
            except (ValueError, TypeError):
                val = None
            by_date[row["date"]][row["index_name"]] = val
 
        return [{"date": d, **vals} for d, vals in sorted(by_date.items())]
 
    except Exception as e:
        log.error("read_all_indices_wide failed: %s", e)
        return []
 

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA / ADMIN
# ══════════════════════════════════════════════════════════════════════════════

def initialize_all_tabs() -> dict:
    """
    Run all pending migrations to ensure all tables exist.
    Drop-in replacement for old initialize_all_tabs().

    Returns:
        Dict of {migration_id: status}

    Example:
        results = initialize_all_tabs()
        # {"001": "applied"} or {"001": "skipped"}
    """
    from db.migrations import run_migrations
    return run_migrations()


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        with _db() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
                """,
                (table_name.lower(),),
            )
            row = cur.fetchone()
            return bool(row["exists"]) if row else False
    except Exception as e:
        log.error("table_exists(%s) failed: %s", table_name, e)
        return False


def get_row_count(tab_key: str) -> int:
    """Return number of rows in a table."""
    table = _resolve_table(tab_key)
    try:
        with _db() as cur:
            cur.execute(f'SELECT COUNT(*) AS cnt FROM "{table}"')
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0
    except Exception as e:
        log.error("get_row_count(%s) failed: %s", table, e)
        return 0


def run_raw_sql(sql: str, params: tuple = None) -> list[dict]:
    """
    Execute raw SQL and return results as list of dicts.
    For complex queries not covered by the standard API.

    Args:
        sql:    SQL query string (use %s for parameters)
        params: Query parameters tuple

    Returns:
        List of row dicts, or empty list

    Example:
        rows = run_raw_sql(
            "SELECT symbol, COUNT(*) as signals FROM market_log GROUP BY symbol ORDER BY signals DESC LIMIT 10"
        )
        rows = run_raw_sql(
            "SELECT * FROM market_log WHERE date >= %s AND outcome = %s",
            ("2026-01-01", "WIN")
        )
    """
    try:
        with _db() as cur:
            cur.execute(sql, params)
            return [dict(r) for r in (cur.fetchall() or [])]
    except Exception as e:
        log.error("run_raw_sql failed: %s", e)
        return []
