"""
web_ui/dashboard_api.py
NEPSE AI Engine — Dashboard read-only API

Run:
    uvicorn web_ui.dashboard_api:app --port 8766 --reload
"""

import logging
import sys
import os
import datetime
import decimal
from datetime import date

import re as _re
import io as _io
import csv as _csv

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import _db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DASH_API] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    try:
        with _db() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS saved_queries (
                  id SERIAL PRIMARY KEY,
                  name TEXT NOT NULL,
                  description TEXT,
                  sql TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT NOW()
                )
            """)
        log.info("saved_queries table ensured")
    except Exception as e:
        log.exception("Failed to ensure saved_queries table: %s", e)
    yield

app = FastAPI(title="NEPSE AI — Dashboard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


def _safe(val):
    """Convert psycopg2 non-JSON-serialisable types to plain Python."""
    if isinstance(val, (datetime.datetime, datetime.date, datetime.time)):
        return val.isoformat()
    if isinstance(val, decimal.Decimal):
        return float(val)
    return val


def _rows(cur) -> list[dict]:
    return [{k: _safe(v) for k, v in dict(r).items()} for r in (cur.fetchall() or [])]


def _safe_float(val):
    try:
        return float(val) if val not in (None, "", "null", "None") else None
    except (ValueError, TypeError):
        return None


# ── Static ────────────────────────────────────────────────────────────────────

_NRB_HTML = os.path.join(os.path.dirname(__file__), "nrb_entry.html")


@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(_HTML)


@app.get("/nrb")
def serve_nrb():
    return FileResponse(_NRB_HTML)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "dashboard_api"}


# ── Summary ───────────────────────────────────────────────────────────────────

@app.get("/dashboard/summary")
def get_summary():
    try:
        with _db() as cur:
            # paper_capital
            cur.execute(
                "SELECT current_capital, total_trades, total_wins, total_losses "
                "FROM paper_capital WHERE test_mode = 'false' ORDER BY id DESC LIMIT 1"
            )
            cap = cur.fetchone()
            cap = dict(cap) if cap else {}

            # open paper positions + ltp join
            cur.execute("""
                SELECT pp.symbol, pp.total_shares, pp.wacc,
                       amw.ltp AS current_ltp
                FROM paper_portfolio pp
                LEFT JOIN (
                    SELECT DISTINCT ON (symbol) symbol, ltp
                    FROM atrad_market_watch
                    ORDER BY symbol, date DESC, time DESC
                ) amw ON amw.symbol = pp.symbol
                WHERE pp.status = 'OPEN' AND pp.test_mode = 'false'
            """)
            positions = _rows(cur)

            unrealized_pnl = 0.0
            for p in positions:
                shares = _safe_float(p.get("total_shares")) or 0
                wacc   = _safe_float(p.get("wacc")) or 0
                ltp    = _safe_float(p.get("current_ltp")) or 0
                if ltp and wacc:
                    unrealized_pnl += shares * (ltp - wacc)

            # trade_journal aggregate
            cur.execute("""
                SELECT
                    COUNT(*)                                                          AS total,
                    SUM(CASE WHEN result = 'WIN'  THEN 1 ELSE 0 END)                 AS wins,
                    SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END)                 AS losses,
                    AVG(CASE WHEN return_pct ~ '^-?[0-9]+\\.?[0-9]*$'
                             THEN return_pct::float ELSE NULL END)                   AS avg_return
                FROM trade_journal
            """)
            tj = dict(cur.fetchone() or {})

            # market_state from settings
            cur.execute(
                "SELECT value FROM settings WHERE key = 'market_state' ORDER BY id DESC LIMIT 1"
            )
            ms = cur.fetchone()
            market_state = ms["value"] if ms else None

            # nepal_score (from nepal_pulse)
            cur.execute(
                "SELECT nepal_score FROM nepal_pulse ORDER BY date DESC, id DESC LIMIT 1"
            )
            np_row = cur.fetchone()
            nepal_score = np_row["nepal_score"] if np_row else None

            # geo_score (from geopolitical_data)
            cur.execute(
                "SELECT geo_score FROM geopolitical_data ORDER BY date DESC, id DESC LIMIT 1"
            )
            geo_row = cur.fetchone()
            geo_score = geo_row["geo_score"] if geo_row else None

            # NEPSE index last 30 days
            cur.execute("""
                SELECT date, current_value, change_pct, index_name
                FROM nepse_indices
                WHERE index_name ILIKE '%nepse%' OR index_id = 'NEPSE'
                ORDER BY date DESC
                LIMIT 30
            """)
            nepse_30d = list(reversed(_rows(cur)))

            # recent 5 market_log signals
            cur.execute("""
                SELECT date, symbol, action, confidence, primary_signal, outcome, actual_pnl
                FROM market_log
                ORDER BY date DESC, id DESC
                LIMIT 5
            """)
            recent_signals = _rows(cur)

        total  = int(tj.get("total") or 0)
        wins   = int(tj.get("wins")  or 0)
        losses = int(tj.get("losses") or 0)
        win_rate = round(wins / total * 100, 1) if total > 0 else 0.0

        return {
            "paper_capital":    cap.get("current_capital"),
            "open_positions":   len(positions),
            "unrealized_pnl":   round(unrealized_pnl, 2),
            "trade_journal": {
                "total_trades": total,
                "wins":         wins,
                "losses":       losses,
                "win_rate":     win_rate,
                "avg_return":   round(_safe_float(tj.get("avg_return")) or 0, 2),
            },
            "market_state":    market_state,
            "nepal_score":     nepal_score,
            "geo_score":       geo_score,
            "nepse_30d":       nepse_30d,
            "recent_signals":  recent_signals,
        }
    except Exception as e:
        log.exception("summary failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Market Log ────────────────────────────────────────────────────────────────

@app.get("/dashboard/market_log")
def get_market_log():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM market_log ORDER BY date DESC, id DESC LIMIT 200")
            rows = _rows(cur)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("market_log failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Learning Hub ──────────────────────────────────────────────────────────────

@app.get("/dashboard/learning_hub")
def get_learning_hub():
    try:
        with _db() as cur:
            cur.execute(
                "SELECT * FROM learning_hub WHERE active = 'true' ORDER BY id"
            )
            active = _rows(cur)

            superseded_ids = []
            for lesson in active:
                sid = lesson.get("supersedes_lesson_id")
                if sid and str(sid).isdigit():
                    superseded_ids.append(int(sid))

            old_lessons: dict = {}
            if superseded_ids:
                cur.execute(
                    "SELECT * FROM learning_hub WHERE id = ANY(%s)",
                    (superseded_ids,),
                )
                for row in _rows(cur):
                    old_lessons[str(row["id"])] = row

        return {"lessons": active, "superseded_lessons": old_lessons}
    except Exception as e:
        log.exception("learning_hub failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Audit ─────────────────────────────────────────────────────────────────────

@app.get("/dashboard/audit")
def get_audit():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM claude_audit ORDER BY review_week ASC")
            rows = _rows(cur)
        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("audit failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Council ───────────────────────────────────────────────────────────────────

@app.get("/dashboard/council")
def get_council():
    try:
        with _db() as cur:
            cur.execute(
                "SELECT * FROM monthly_council_log ORDER BY run_month DESC, id ASC"
            )
            log_rows = _rows(cur)

            cur.execute(
                "SELECT * FROM monthly_council_agenda ORDER BY run_month DESC, item_number ASC"
            )
            agenda_rows = _rows(cur)

            cur.execute(
                "SELECT * FROM monthly_council_checklist ORDER BY run_month DESC"
            )
            checklist_rows = _rows(cur)

        return {
            "log":       log_rows,
            "agenda":    agenda_rows,
            "checklist": checklist_rows,
        }
    except Exception as e:
        log.exception("council failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/dashboard/trades")
def get_trades():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM trade_journal ORDER BY entry_date DESC")
            tj_rows = _rows(cur)

            cur.execute("SELECT * FROM paper_portfolio ORDER BY first_buy_date DESC")
            pp_rows = _rows(cur)

        return {"trade_journal": tj_rows, "paper_portfolio": pp_rows}
    except Exception as e:
        log.exception("trades failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/dashboard/portfolio")
def get_portfolio():
    try:
        with _db() as cur:
            cur.execute(
                "SELECT * FROM paper_portfolio "
                "WHERE status = 'OPEN' AND test_mode = 'false'"
            )
            open_positions = _rows(cur)

            cur.execute(
                "SELECT current_capital FROM paper_capital "
                "WHERE test_mode = 'false' ORDER BY id DESC LIMIT 1"
            )
            cap = cur.fetchone()
            cash = cap["current_capital"] if cap else None

            symbols = [p["symbol"] for p in open_positions if p.get("symbol")]
            ltp_map: dict = {}
            if symbols:
                cur.execute("""
                    SELECT DISTINCT ON (symbol) symbol, ltp
                    FROM atrad_market_watch
                    WHERE symbol = ANY(%s)
                    ORDER BY symbol, date DESC, time DESC
                """, (symbols,))
                for row in _rows(cur):
                    ltp_map[row["symbol"]] = row["ltp"]

        for p in open_positions:
            sym  = p.get("symbol", "")
            ltp  = _safe_float(ltp_map.get(sym)) or 0
            wacc = _safe_float(p.get("wacc")) or 0
            shrs = _safe_float(p.get("total_shares")) or 0
            p["current_ltp"] = ltp_map.get(sym)
            if ltp and wacc:
                p["unrealized_pnl_pct"] = round((ltp - wacc) / wacc * 100, 2)
                p["unrealized_pnl_npr"] = round(shrs * (ltp - wacc), 2)
            else:
                p["unrealized_pnl_pct"] = None
                p["unrealized_pnl_npr"] = None

        return {"open_positions": open_positions, "cash_balance": cash}
    except Exception as e:
        log.exception("portfolio failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Reporter ───────────────────────────────────────────────────────────────────

_SQL_DANGER = _re.compile(r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER)\b', _re.IGNORECASE)
_PARAM_RE   = _re.compile(r':([a-zA-Z_][a-zA-Z0-9_]*)')


def _check_sql(sql: str):
    if _SQL_DANGER.search(sql):
        raise HTTPException(status_code=400, detail="Only SELECT queries allowed.")


def _build_psycopg2_sql(sql: str) -> str:
    return _PARAM_RE.sub(r'%(\1)s', sql)


class _SaveQueryBody(BaseModel):
    name: str
    description: Optional[str] = None
    sql: str


class _RunQueryBody(BaseModel):
    sql: str
    params: dict = {}


class _DownloadBody(BaseModel):
    sql: str
    params: dict = {}
    filename: str = "export"


@app.get("/reporter/tables")
def reporter_tables():
    try:
        with _db() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [r["table_name"] for r in (cur.fetchall() or [])]
        return {"tables": tables}
    except Exception as e:
        log.exception("reporter_tables failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reporter/sectors")
def reporter_sectors():
    try:
        with _db() as cur:
            cur.execute(
                "SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL ORDER BY sector"
            )
            rows = [r["sector"] for r in (cur.fetchall() or [])]
        return {"sectors": rows}
    except Exception as e:
        log.exception("reporter_sectors failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reporter/queries")
def reporter_list_queries():
    try:
        with _db() as cur:
            cur.execute(
                "SELECT id, name, description, created_at FROM saved_queries ORDER BY id DESC"
            )
            rows = _rows(cur)
        return {"queries": rows}
    except Exception as e:
        log.exception("reporter_list_queries failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reporter/queries")
def reporter_save_query(body: _SaveQueryBody):
    _check_sql(body.sql)
    try:
        with _db() as cur:
            cur.execute(
                "INSERT INTO saved_queries (name, description, sql) VALUES (%s, %s, %s) RETURNING id",
                (body.name, body.description, body.sql),
            )
            row = cur.fetchone()
        return {"id": row["id"], "name": body.name}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reporter_save_query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reporter/queries/{query_id}")
def reporter_get_query(query_id: int):
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM saved_queries WHERE id = %s", (query_id,))
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Query not found")
        return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reporter_get_query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reporter/run")
def reporter_run(body: _RunQueryBody):
    _check_sql(body.sql)
    try:
        psql = _build_psycopg2_sql(body.sql)
        params = {k: v for k, v in body.params.items() if v not in (None, "", [])}
        with _db() as cur:
            cur.execute(psql, params or None)
            rows = _rows(cur)
        return {"rows": rows, "count": len(rows)}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reporter_run failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reporter/download/csv")
def reporter_download_csv(body: _DownloadBody):
    _check_sql(body.sql)
    try:
        psql = _build_psycopg2_sql(body.sql)
        params = {k: v for k, v in body.params.items() if v not in (None, "", [])}
        with _db() as cur:
            cur.execute(psql, params or None)
            rows = _rows(cur)

        if not rows:
            raise HTTPException(status_code=404, detail="No rows returned")

        buf = _io.StringIO()
        writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        buf.seek(0)

        filename = (body.filename or "export").strip()
        if not filename.endswith(".csv"):
            filename += ".csv"

        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reporter_download_csv failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reporter/download/xlsx")
def reporter_download_xlsx(body: _DownloadBody):
    _check_sql(body.sql)
    try:
        import openpyxl
        psql = _build_psycopg2_sql(body.sql)
        params = {k: v for k, v in body.params.items() if v not in (None, "", [])}
        with _db() as cur:
            cur.execute(psql, params or None)
            rows = _rows(cur)

        if not rows:
            raise HTTPException(status_code=404, detail="No rows returned")

        wb = openpyxl.Workbook()
        ws = wb.active
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h) for h in headers])

        buf = _io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        filename = (body.filename or "export").strip()
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"

        return StreamingResponse(
            iter([buf.read()]),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("reporter_download_xlsx failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
