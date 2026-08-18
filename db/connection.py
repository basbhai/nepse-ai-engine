"""
db/connection.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — PostgreSQL connection layer.

Responsibilities:
    - Load DATABASE_URL from .env
    - Provide _db() context manager for all queries
    - Connection pooling via psycopg2 ThreadedConnectionPool
    - Retry logic with exponential back-off
    - Never expose raw connections outside this module

Usage:
    from db.connection import _db

    with _db() as cur:
        cur.execute("SELECT * FROM watchlist WHERE symbol = %s", ("NABIL",))
        rows = cur.fetchall()   # list of dicts (RealDictCursor)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import logging
import threading
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "false").lower() == "true"

DATABASE_URL = (
    os.getenv("BACKTEST_DATABASE_URL")
    if BACKTEST_MODE
    else os.getenv("DATABASE_URL", "")
)
MAX_RETRIES  = 3
RETRY_DELAY  = 5     # seconds base
POOL_MIN     = 1
POOL_MAX     = int(os.getenv("DB_POOL_MAX", "12"))

# ─────────────────────────────────────────
# CONNECTION POOL (lazy, thread-safe)
# ─────────────────────────────────────────
_pool      = None
_pool_lock = threading.Lock()


def _get_pool():
    """
    Lazy-init ThreadedConnectionPool.
    Called once on first _db() use. Thread-safe via double-checked lock.
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool

        try:
            from psycopg2 import pool as pg_pool
        except ImportError:
            raise RuntimeError(
                "psycopg2-binary not installed.\n"
                "Run: pip install psycopg2-binary"
            )

        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL not set in .env\n"
                "Format: postgresql://user:pass@host/db"
            )

        _pool = pg_pool.ThreadedConnectionPool(POOL_MIN, POOL_MAX, DATABASE_URL)
        log.info("DB connection pool ready (min=%d max=%d)", POOL_MIN, POOL_MAX)
        return _pool


@contextmanager
def _db():
    """
    Context manager for all DB access. Yields a RealDictCursor.

    - Pulls a connection from the pool, returns it on exit
    - Auto-commits on clean exit, rolls back on exception
    - Retries acquiring a connection up to MAX_RETRIES times. A pooled
      connection that Postgres or the OS silently closed while idle looks
      fine to the pool and only fails once used, so each acquired
      connection gets a cheap SELECT 1 probe before being handed to the
      caller; a dead one is discarded (not recycled) and retried
      immediately — reconnecting is a ~20ms local operation, so only
      later attempts (a real outage, not one stale connection) pay the
      back-off sleep
    - A failure *inside* the `with` block (i.e. during your own queries)
      is not retried — your queries may have already partially run, so
      it's raised as-is rather than silently rerun from a fresh connection
    - Rows returned as dicts: row["symbol"] not row[0]

    Usage:
        with _db() as cur:
            cur.execute("SELECT * FROM market_log WHERE outcome = %s", ("PENDING",))
            rows = cur.fetchall()

        with _db() as cur:
            cur.execute(
                "INSERT INTO watchlist (symbol, sector) VALUES (%s, %s)",
                ("NABIL", "Banking"),
            )
    """
    import psycopg2
    import psycopg2.extras

    conn     = None
    cur      = None
    last_exc = None

    for attempt in range(MAX_RETRIES):
        try:
            pool = _get_pool()
            conn = pool.getconn()

            try:
                probe = conn.cursor()
                probe.execute("SELECT 1")
                probe.fetchone()
                probe.close()
                conn.rollback()  # close the implicit transaction the probe opened
            except psycopg2.OperationalError:
                pool.putconn(conn, close=True)
                conn = None
                raise

            conn.autocommit = False
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            break

        except psycopg2.OperationalError as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * attempt  # 0 on the first retry, 5s+ after
                log.warning(
                    "DB connection error (attempt %d/%d), retrying%s: %s",
                    attempt + 1, MAX_RETRIES,
                    f" in {wait}s" if wait else "", e,
                )
                if wait:
                    time.sleep(wait)
    else:
        raise RuntimeError(f"DB failed after {MAX_RETRIES} attempts: {last_exc}")

    try:
        yield cur
        conn.commit()
    except psycopg2.OperationalError:
        # Connection died mid-query, not just at acquisition — discard it
        # instead of recycling a broken connection back into the pool.
        try: conn.rollback()
        except Exception: pass
        try:
            _get_pool().putconn(conn, close=True)
        except Exception:
            pass
        conn = None
        raise
    except Exception as e:
        try: conn.rollback()
        except Exception: pass
        log.error("DB error: %s", e)
        raise
    finally:
        if conn:
            try:
                _get_pool().putconn(conn)
            except Exception:
                pass


def test_connection() -> bool:
    """Quick health check — returns True if the DB is reachable."""
    try:
        with _db() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() is not None
    except Exception as e:
        log.error("Connection test failed: %s", e)
        return False


def close_pool():
    """Close all pooled connections. Call at process exit in long-running scripts."""
    global _pool
    if _pool:
        try:
            _pool.closeall()
            log.info("Connection pool closed")
        except Exception as e:
            log.warning("Error closing pool: %s", e)
        _pool = None
