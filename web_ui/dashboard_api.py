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
from sheets import upsert_row, read_tab

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


def _to_str(val) -> Optional[str]:
    if val is None:
        return None
    return str(val)


# ── NRB Model ─────────────────────────────────────────────────────────────────

class NRBPayload(BaseModel):
    period:                                str
    fiscal_year:                           str
    month_number:                          str
    is_annual:                             str = "false"
    policy_rate:                           Optional[str] = None
    bank_rate:                             Optional[str] = None
    crr_percentage:                        Optional[str] = None
    slr_percentage:                        Optional[str] = None
    lending_rate_pct:                      Optional[str] = None
    deposit_rate_pct:                      Optional[str] = None
    interbank_rate_pct:                    Optional[str] = None
    tbill_91d_rate_pct:                    Optional[str] = None
    cpi_inflation:                         Optional[str] = None
    credit_growth_rate:                    Optional[str] = None
    private_sector_credit_growth_yoy_pct:  Optional[str] = None
    m2_growth_yoy_pct:                     Optional[str] = None
    deposit_growth_yoy_pct:                Optional[str] = None
    npl_ratio_pct:                         Optional[str] = None
    liquidity_injected_billion:            Optional[str] = None
    liquidity_status:                      Optional[str] = None
    remittance_yoy_change_pct:             Optional[str] = None
    remittance_total_billion_npr:          Optional[str] = None
    fx_reserve_months:                     Optional[str] = None
    usd_npr_rate:                          Optional[str] = None
    bop_overall_balance_usd_m:             Optional[str] = None
    bop_current_account_usd_m:             Optional[str] = None
    bop_capital_account_usd_m:             Optional[str] = None
    bop_trade_deficit_usd_m:               Optional[str] = None
    bop_status:                            Optional[str] = None
    bop_trend:                             Optional[str] = None
    bop_impact_on_nepse:                   Optional[str] = None
    nepse_index_value:                     Optional[str] = None
    nepse_index:                           Optional[str] = None
    market_cap_billion_npr:                Optional[str] = None
    overall_sentiment:                     Optional[str] = None
    forward_guidance:                      Optional[str] = None
    key_risks:                             Optional[str] = None


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
                "SELECT value FROM settings WHERE key = 'MARKET_STATE' ORDER BY id DESC LIMIT 1"
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


# ── Broker Flow ───────────────────────────────────────────────────────────────

@app.get("/dashboard/broker_flow")
def get_broker_flow():
    """Return top 8 accumulation and top 8 distribution rows from latest broker_flow date."""
    try:
        with _db() as cur:
            cur.execute("SELECT MAX(date) as latest FROM broker_flow")
            row = cur.fetchone()
            latest = row["latest"] if row else None
            if not latest:
                return {"date": None, "accumulation": [], "distribution": []}

            cur.execute("""
                SELECT symbol, name,
                       acc_broker_count_1d, acc_qty_1d, acc_amount_1d,
                       acc_top_broker_1d, acc_top_broker_pct_1d,
                       net_flow_1d, flow_bias_1d
                FROM broker_flow
                WHERE date = %s
                  AND flow_bias_1d = 'ACCUMULATION'
                  AND NULLIF(acc_qty_1d, '') IS NOT NULL
                ORDER BY (
                    CAST(NULLIF(acc_amount_1d, '0') AS FLOAT) *
                    CAST(NULLIF(acc_qty_1d, '0') AS FLOAT) /
                    NULLIF(CAST(NULLIF(acc_broker_count_1d, '0') AS FLOAT), 0)
                ) DESC NULLS LAST
                LIMIT 8
            """, (latest,))
            acc_rows = [dict(r) for r in cur.fetchall()]

            cur.execute("""
                SELECT symbol, name,
                       dist_broker_count_1d, dist_qty_1d, dist_amount_1d,
                       dist_top_broker_1d, dist_top_broker_pct_1d,
                       net_flow_1d, flow_bias_1d
                FROM broker_flow
                WHERE date = %s
                  AND flow_bias_1d = 'DISTRIBUTION'
                  AND NULLIF(dist_qty_1d, '') IS NOT NULL
                ORDER BY (
                    CAST(NULLIF(dist_amount_1d, '0') AS FLOAT) *
                    CAST(NULLIF(dist_qty_1d, '0') AS FLOAT) /
                    NULLIF(CAST(NULLIF(dist_broker_count_1d, '0') AS FLOAT), 0)
                ) DESC NULLS LAST
                LIMIT 8
            """, (latest,))
            dist_rows = [dict(r) for r in cur.fetchall()]

            return {
                "date": latest,
                "accumulation": [_safe(r) for r in acc_rows],
                "distribution":  [_safe(r) for r in dist_rows],
            }
    except Exception as e:
        log.exception("broker_flow endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Market Log ────────────────────────────────────────────────────────────────

@app.get("/dashboard/market_log")
def get_market_log():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM market_log ORDER BY date DESC, id DESC LIMIT 10")
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
                "SELECT DISTINCT run_month FROM monthly_council_log "
                "WHERE run_month ~ '^[0-9]{4}-[0-9]{2}$' "
                "ORDER BY run_month DESC LIMIT 1"
            )
            row = cur.fetchone()
            latest_month = row["run_month"] if row else None

            if not latest_month:
                return {"log": [], "agenda": [], "checklist": [], "proposals": [], "run_month": None}

            cur.execute(
                "SELECT * FROM monthly_council_log WHERE run_month = %s ORDER BY id ASC",
                (latest_month,)
            )
            log_rows = _rows(cur)

            cur.execute(
                "SELECT * FROM monthly_council_agenda WHERE run_month = %s ORDER BY item_number ASC",
                (latest_month,)
            )
            agenda_rows = _rows(cur)

            cur.execute(
                "SELECT * FROM monthly_council_checklist WHERE run_month = %s ORDER BY id ASC",
                (latest_month,)
            )
            checklist_rows = _rows(cur)

            cur.execute(
                "SELECT * FROM system_proposals WHERE run_month = %s ORDER BY id ASC",
                (latest_month,)
            )
            proposals_rows = _rows(cur)

        return {
            "log":       log_rows,
            "agenda":    agenda_rows,
            "checklist": checklist_rows,
            "proposals": proposals_rows,
            "run_month": latest_month,
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
        params = {k: (v if v not in (None, "", []) else None) for k, v in body.params.items()}
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


# ── NRB Routes ────────────────────────────────────────────────────────────────

@app.get("/nrb/periods")
def get_nrb_periods():
    try:
        rows = read_tab("nrb_monthly") or []
        rows.sort(key=lambda r: (r.get("fiscal_year",""), int(r.get("month_number") or 0)), reverse=True)
        return {
            "periods": [
                {
                    "period":       r.get("period"),
                    "fiscal_year":  r.get("fiscal_year"),
                    "month_number": r.get("month_number"),
                    "inserted_at":  str(r.get("inserted_at", "")),
                }
                for r in rows
            ]
        }
    except Exception as e:
        log.exception("get_nrb_periods failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nrb/upsert")
def upsert_nrb(payload: NRBPayload):
    try:
        row = {
            "period":                               payload.period,
            "fiscal_year":                          payload.fiscal_year,
            "month_number":                         payload.month_number,
            "is_annual":                            payload.is_annual,
            "policy_rate":                          _to_str(payload.policy_rate),
            "bank_rate":                            _to_str(payload.bank_rate),
            "crr_percentage":                       _to_str(payload.crr_percentage),
            "slr_percentage":                       _to_str(payload.slr_percentage),
            "lending_rate_pct":                     _to_str(payload.lending_rate_pct),
            "deposit_rate_pct":                     _to_str(payload.deposit_rate_pct),
            "interbank_rate_pct":                   _to_str(payload.interbank_rate_pct),
            "tbill_91d_rate_pct":                   _to_str(payload.tbill_91d_rate_pct),
            "cpi_inflation":                        _to_str(payload.cpi_inflation),
            "credit_growth_rate":                   _to_str(payload.credit_growth_rate),
            "private_sector_credit_growth_yoy_pct": _to_str(payload.private_sector_credit_growth_yoy_pct),
            "m2_growth_yoy_pct":                    _to_str(payload.m2_growth_yoy_pct),
            "deposit_growth_yoy_pct":               _to_str(payload.deposit_growth_yoy_pct),
            "npl_ratio_pct":                        _to_str(payload.npl_ratio_pct),
            "liquidity_injected_billion":           _to_str(payload.liquidity_injected_billion),
            "liquidity_status":                     _to_str(payload.liquidity_status),
            "remittance_yoy_change_pct":            _to_str(payload.remittance_yoy_change_pct),
            "remittance_total_billion_npr":         _to_str(payload.remittance_total_billion_npr),
            "fx_reserve_months":                    _to_str(payload.fx_reserve_months),
            "usd_npr_rate":                         _to_str(payload.usd_npr_rate),
            "bop_overall_balance_usd_m":            _to_str(payload.bop_overall_balance_usd_m),
            "bop_current_account_usd_m":            _to_str(payload.bop_current_account_usd_m),
            "bop_capital_account_usd_m":            _to_str(payload.bop_capital_account_usd_m),
            "bop_trade_deficit_usd_m":              _to_str(payload.bop_trade_deficit_usd_m),
            "bop_status":                           _to_str(payload.bop_status),
            "bop_trend":                            _to_str(payload.bop_trend),
            "bop_impact_on_nepse":                  _to_str(payload.bop_impact_on_nepse),
            "nepse_index_value":                    _to_str(payload.nepse_index_value),
            "nepse_index":                          _to_str(payload.nepse_index or payload.nepse_index_value),
            "market_cap_billion_npr":               _to_str(payload.market_cap_billion_npr),
            "overall_sentiment":                    _to_str(payload.overall_sentiment),
            "forward_guidance":                     _to_str(payload.forward_guidance),
            "key_risks":                            _to_str(payload.key_risks),
        }
        row = {k: v for k, v in row.items() if v is not None}
        ok = upsert_row("nrb_monthly", row, conflict_columns=["period"])
        if ok:
            log.info("Upserted NRB record: %s", payload.period)
            return {
                "status":         "success",
                "period":         payload.period,
                "message":        f"NRB data for {payload.period} saved successfully.",
                "fields_written": len(row),
            }
        else:
            raise HTTPException(status_code=500, detail="upsert_row returned False")
    except HTTPException:
        raise
    except Exception as e:
        log.exception("upsert_nrb failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/watching")
def get_stealth_watching():
    try:
        today = date.today()
        with _db() as cur:
            cur.execute("""
                SELECT DISTINCT ON (symbol, broker_id) *
                FROM stealth_signals
                WHERE status = 'WATCHING'
                ORDER BY symbol, broker_id, signal_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            if r.get("signal_date"):
                try:
                    sig_date = date.fromisoformat(str(r["signal_date"]))
                    r["days_watching"] = (today - sig_date).days
                except Exception:
                    r["days_watching"] = None
            else:
                r["days_watching"] = None

        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("stealth/watching failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/triggered")
def get_stealth_triggered():
    try:
        today = date.today()
        with _db() as cur:
            cur.execute("""
                SELECT DISTINCT ON (s.symbol, s.broker_id) s.*,
                    p.close AS current_price
                FROM stealth_signals s
                LEFT JOIN LATERAL (
                    SELECT COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
                    FROM price_history
                    WHERE symbol = s.symbol
                    AND close IS NOT NULL AND close != ''
                    AND close ~ '^[0-9]+\\.?[0-9]*$'
                    ORDER BY date DESC
                    LIMIT 1
                ) p ON true
                WHERE s.status = 'TRIGGERED'
                ORDER BY s.symbol, s.broker_id, s.trigger_date DESC, s.signal_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            # Days since trigger
            if r.get("trigger_date"):
                try:
                    tdate = date.fromisoformat(str(r["trigger_date"]))
                    r["days_since_trigger"] = (today - tdate).days
                except Exception:
                    r["days_since_trigger"] = None

            # Live return %
            try:
                tp = float(r["trigger_price"]) if r.get("trigger_price") else None
                cp = float(r["current_price"]) if r.get("current_price") else None
                if tp and cp and tp > 0:
                    r["live_return_pct"] = round((cp - tp) / tp * 100, 2)
                else:
                    r["live_return_pct"] = None
            except Exception:
                r["live_return_pct"] = None

        return {"rows": rows, "count": len(rows)}
    except Exception as e:
        log.exception("stealth/triggered failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/history")
def get_stealth_history():
    try:
        with _db() as cur:
            cur.execute("""
                SELECT * FROM stealth_signals
                WHERE status = 'CLOSED'
                ORDER BY close_date DESC
            """)
            rows = _rows(cur)

        for r in rows:
            # Days held (trigger → close)
            try:
                td = date.fromisoformat(str(r["trigger_date"])) if r.get("trigger_date") else None
                cd = date.fromisoformat(str(r["close_date"]))   if r.get("close_date")   else None
                r["days_held"] = (cd - td).days if (td and cd) else None
            except Exception:
                r["days_held"] = None

            # WIN/LOSS
            try:
                rp = float(r["return_pct"]) if r.get("return_pct") else None
                r["result"] = "WIN" if rp and rp > 0 else ("LOSS" if rp and rp < 0 else "BREAKEVEN")
            except Exception:
                r["result"] = None

        # Summary stats
        returns = []
        for r in rows:
            try:
                rp = float(r["return_pct"]) if r.get("return_pct") else None
                if rp is not None:
                    returns.append(rp)
            except Exception:
                pass

        win_rate   = round(sum(1 for x in returns if x > 0) / len(returns) * 100, 1) if returns else None
        avg_return = round(sum(returns) / len(returns), 2) if returns else None
        sorted_ret = sorted(returns)
        median_ret = sorted_ret[len(sorted_ret)//2] if sorted_ret else None

        return {
            "rows":    rows,
            "count":   len(rows),
            "summary": {
                "total_closed":  len(rows),
                "win_rate":      win_rate,
                "avg_return":    avg_return,
                "median_return": median_ret,
            }
        }
    except Exception as e:
        log.exception("stealth/history failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stealth/stats")
def get_stealth_stats():
    try:
        with _db() as cur:
            cur.execute("SELECT * FROM stealth_signals ORDER BY signal_date DESC")
            all_rows = _rows(cur)

        brokers = {}
        total_watching   = 0
        total_triggered  = 0
        total_closed     = 0
        all_returns      = []

        for r in all_rows:
            bid  = str(r.get("broker_id", ""))
            bname = r.get("broker_name", bid)
            status = r.get("status", "")

            if bid not in brokers:
                brokers[bid] = {
                    "broker_id":       bid,
                    "broker_name":     bname,
                    "signal_count":    0,
                    "triggered_count": 0,
                    "closed_count":    0,
                    "returns":         [],
                }

            brokers[bid]["signal_count"] += 1

            if status == "WATCHING":
                total_watching += 1
            elif status == "TRIGGERED":
                total_triggered += 1
                brokers[bid]["triggered_count"] += 1
            elif status == "CLOSED":
                total_closed += 1
                brokers[bid]["triggered_count"] += 1
                brokers[bid]["closed_count"]    += 1
                try:
                    rp = float(r["return_pct"]) if r.get("return_pct") else None
                    if rp is not None:
                        brokers[bid]["returns"].append(rp)
                        all_returns.append(rp)
                except Exception:
                    pass

        # Compute per-broker stats
        broker_stats = []
        for bid, b in brokers.items():
            rets = b.pop("returns", [])
            b["trigger_rate"] = round(b["triggered_count"] / b["signal_count"] * 100, 1) \
                                if b["signal_count"] > 0 else None
            b["avg_return"]   = round(sum(rets) / len(rets), 2) if rets else None
            b["win_rate"]     = round(sum(1 for x in rets if x > 0) / len(rets) * 100, 1) \
                                if rets else None
            # Historical hit rates from grid search
            hit_rates = {"56": 82.4, "48": 80.0, "49": 78.3, "38": 77.8, "58": 60.0}
            b["validated_hit_rate"] = hit_rates.get(bid)
            broker_stats.append(b)

        broker_stats.sort(key=lambda x: -(x.get("validated_hit_rate") or 0))

        # Overall stats
        overall_hit_rate   = round(sum(1 for x in all_returns if x > 0) / len(all_returns) * 100, 1) if all_returns else None
        overall_avg_return = round(sum(all_returns) / len(all_returns), 2) if all_returns else None
        sorted_all         = sorted(all_returns)
        overall_median     = sorted_all[len(sorted_all)//2] if sorted_all else None

        return {
            "broker_stats": broker_stats,
            "overall": {
                "total_signals":  len(all_rows),
                "watching":       total_watching,
                "triggered":      total_triggered,
                "closed":         total_closed,
                "hit_rate":       overall_hit_rate,
                "avg_return":     overall_avg_return,
                "median_return":  overall_median,
            }
        }
    except Exception as e:
        log.exception("stealth/stats failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Logs ──────────────────────────────────────────────────────────────────

@app.get("/dashboard/logs/list")
def get_logs_list():
    try:
        import glob as _glob
        logs_dir = os.path.expanduser("~/nepse-engine/logs")
        if not os.path.isdir(logs_dir):
            return {"files": []}

        files = []
        for fpath in _glob.glob(os.path.join(logs_dir, "**", "main_*.log"), recursive=True):
            try:
                stat = os.stat(fpath)
                mtime = stat.st_mtime
                size = stat.st_size
                rel_path = os.path.relpath(fpath, logs_dir)
                dt = datetime.datetime.fromtimestamp(mtime)
                files.append({
                    "filename": os.path.basename(fpath),
                    "path": rel_path.replace("\\", "/"),
                    "date": dt.strftime("%Y-%m-%d"),
                    "time": dt.strftime("%H:%M:%S"),
                    "size": size,
                })
            except Exception:
                pass

        files.sort(key=lambda x: (x["date"], x["time"]), reverse=True)
        return {"files": files}
    except Exception as e:
        log.exception("logs/list failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/logs/file")
def get_logs_file(path: str):
    try:
        logs_dir = os.path.expanduser("~/nepse-engine/logs")
        fpath = os.path.normpath(os.path.join(logs_dir, path))

        if not fpath.startswith(os.path.normpath(logs_dir)) or not os.path.isfile(fpath):
            raise HTTPException(status_code=404, detail="File not found or access denied")

        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()

        lines = []
        for raw in raw_lines:
            line = raw.rstrip("\n\r")

            # Truncate raw: content
            if "raw: " in line:
                idx = line.find("raw: ")
                prefix = line[:idx + 5]
                content = line[idx + 5:]
                if len(content) > 120:
                    content = content[:120] + "…[truncated]"
                line = prefix + content

            # Truncate HTML
            if "<!DOCTYPE" in line or "<!doctype" in line.lower():
                idx = line.lower().find("<!doctype")
                line = line[:idx] + "<!DOCTYPE...…[truncated]"
            elif line.lstrip().startswith("<") and ("html" in line.lower() or "body" in line.lower()):
                line = line[:100] + "…[truncated]" if len(line) > 100 else line

            lines.append(line)

        return {"lines": lines}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("logs/file failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
