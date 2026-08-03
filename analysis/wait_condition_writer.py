# -*- coding: utf-8 -*-
"""
analysis/wait_condition_writer.py — NEPSE AI Engine
=====================================================
Narrow, guarded write CLI for the Claude wait-condition monitor
(agents/claude_wait_monitor, invoked headlessly via systemd at 10:30 AM NST).

The monitor itself reads market_log + indicators + price_history through the
read-only nepse-engine MCP server and reasons about whether each WAIT
signal's condition has been met. This script is the ONLY thing it is allowed
to call to actually write anything — and even this script only exposes two
narrow operations, both of which:
  - go through sheets.update_row() — never raw SQL (inviolable rule 1)
  - only ever UPDATE an existing row, never INSERT (inviolable rule 2/3)
  - refuse to touch any row that is not currently action='WAIT' AND
    outcome='PENDING' at call time, regardless of what id is passed in

Operations:
  mark-met          outcome -> CONDITION_MET (flag only; does not touch
                    paper_portfolio/paper_capital, no auto-trade)
  update-condition  rewrite wait_condition / wait_condition_parsed in place
                    on the same row, stamping last_reviewed_date
  notify            send a Telegram/Discord message via helper.notifier
                    (routes per the existing NOTIFY_CHANNEL setting)

CLI:
    python -m analysis.wait_condition_writer mark-met --id 123
    python -m analysis.wait_condition_writer update-condition --id 123 \
        --condition "..." --parsed '{"requirements": [...], "logic": "ALL"}'
    python -m analysis.wait_condition_writer notify --message "..."
    echo "..." | python -m analysis.wait_condition_writer notify   # stdin also works
"""

import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sheets import run_raw_sql, update_row

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WAIT_WRITER] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

NST = ZoneInfo("Asia/Kathmandu")


def _today() -> str:
    return datetime.now(NST).strftime("%Y-%m-%d")


def _get_wait_row(row_id: int) -> dict | None:
    rows = run_raw_sql(
        "SELECT id, symbol, action, outcome FROM market_log WHERE id = %s",
        (row_id,),
    )
    return rows[0] if rows else None


def _refuse_unless_pending_wait(row_id: int) -> dict | None:
    """Returns the row if it's safe to write to, else logs why and returns None."""
    row = _get_wait_row(row_id)
    if not row:
        log.error("No market_log row with id=%s", row_id)
        return None
    if row["action"] != "WAIT" or row["outcome"] != "PENDING":
        log.error(
            "REFUSED: id=%s (%s) is action=%s outcome=%s — only PENDING WAIT "
            "rows may be written to.",
            row_id, row.get("symbol"), row.get("action"), row.get("outcome"),
        )
        return None
    return row


def mark_met(row_id: int) -> int:
    row = _refuse_unless_pending_wait(row_id)
    if not row:
        return 1
    updated = update_row(
        "market_log",
        updates={"outcome": "CONDITION_MET", "last_reviewed_date": _today()},
        where={"id": str(row_id)},
    )
    if updated:
        log.info("id=%s (%s) -> outcome=CONDITION_MET", row_id, row["symbol"])
        return 0
    log.warning("update_row returned 0 rows for id=%s (%s)", row_id, row["symbol"])
    return 1


def update_condition(row_id: int, condition: str, parsed: str | None) -> int:
    row = _refuse_unless_pending_wait(row_id)
    if not row:
        return 1
    updates = {"wait_condition": condition, "last_reviewed_date": _today()}
    if parsed:
        updates["wait_condition_parsed"] = parsed
    updated = update_row("market_log", updates=updates, where={"id": str(row_id)})
    if updated:
        log.info("id=%s (%s) — wait_condition rewritten", row_id, row["symbol"])
        return 0
    log.warning("update_row returned 0 rows for id=%s (%s)", row_id, row["symbol"])
    return 1


def notify(message: str) -> int:
    from helper.notifier import send_telegram
    ok = send_telegram(message, parse_mode="Markdown")
    log.info("Notification sent=%s (%d chars)", ok, len(message))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Guarded market_log WAIT-condition writer (mark-met / "
                     "update-condition / notify only — no other writes possible)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_met = sub.add_parser("mark-met", help="Stamp outcome=CONDITION_MET")
    p_met.add_argument("--id", type=int, required=True)

    p_upd = sub.add_parser("update-condition", help="Rewrite wait_condition in place")
    p_upd.add_argument("--id", type=int, required=True)
    p_upd.add_argument("--condition", type=str, required=True)
    p_upd.add_argument("--parsed", type=str, default=None,
                        help="Optional JSON string for wait_condition_parsed")

    p_notify = sub.add_parser("notify", help="Send Telegram/Discord message")
    p_notify.add_argument("--message", type=str, default=None,
                           help="If omitted, message is read from stdin")

    args = parser.parse_args()

    if args.cmd == "mark-met":
        sys.exit(mark_met(args.id))
    elif args.cmd == "update-condition":
        sys.exit(update_condition(args.id, args.condition, args.parsed))
    elif args.cmd == "notify":
        message = args.message if args.message is not None else sys.stdin.read()
        if not message.strip():
            log.error("notify: empty message")
            sys.exit(1)
        sys.exit(notify(message))


if __name__ == "__main__":
    main()
