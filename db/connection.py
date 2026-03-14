"""
db/connection.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Neon PostgreSQL connection layer.

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
DATABASE_URL = os.getenv("DATABASE_URL", "")
MAX_RETRIES  = 3
RETRY_DELAY  = 5     # seconds base
POOL_MIN     = 1
POOL_MAX     = 5     # Neon free allows 20

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
                "Format: postgresql://user:pass@host/db?sslmode=require\n"
                "Get it from: neon.tech → project → Connection Details"
            )

        _pool = pg_pool.ThreadedConnectionPool(POOL_MIN, POOL_MAX, DATABASE_URL)
        log.info("Neon connection pool ready (min=%d max=%d)", POOL_MIN, POOL_MAX)
        return _pool


@contextmanager
def _db():
    """
    Context manager for all DB access. Yields a RealDictCursor.

    - Pulls connection from pool, returns it on exit
    - Auto-commits on clean exit, rolls back on exception
    - Retries MAX_RETRIES times with linear back-off on OperationalError
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
    last_exc = None

    for attempt in range(MAX_RETRIES):
        try:
            pool = _get_pool()
            conn = pool.getconn()
            conn.autocommit = False
            cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            yield cur

            conn.commit()
            return

        except psycopg2.OperationalError as e:
            last_exc = e
            if conn:
                try: conn.rollback()
                except Exception: pass
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                log.warning(
                    "DB connection error (attempt %d/%d), retry in %ds: %s",
                    attempt + 1, MAX_RETRIES, wait, e,
                )
                time.sleep(wait)

        except Exception as e:
            if conn:
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
                conn = None

    raise RuntimeError(f"DB failed after {MAX_RETRIES} attempts: {last_exc}")


def test_connection() -> bool:
    """Quick health check — returns True if Neon is reachable."""
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
