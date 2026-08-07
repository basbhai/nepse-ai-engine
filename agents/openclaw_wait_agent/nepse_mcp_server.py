#!/usr/bin/env python3
"""
NEPSE Engine MCP Server (dynamic version)
Exposes schema introspection + generic parameterized SQL as MCP tools,
plus a few convenience wrappers for common alert/notification actions.

Design goals vs. the original version:
- No hardcoded per-question functions (get_pending_waits, check_wait_condition, etc).
  The agent discovers the schema itself and writes its own SQL.
- No f-string SQL interpolation. Everything goes through parameterized
  queries via the reporter API's existing `params` support.
- Read-only: run_query only allows SELECT. There is no write path — no tool
  in this file can INSERT/UPDATE/DELETE against the database.
"""
import os
import re
import requests
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load this agent's own .env regardless of who launches the process
# (Claude Code's MCP client runs it with an arbitrary cwd).
load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTER_API = os.getenv('REPORTER_API_URL', 'http://localhost:8766')
TELEGRAM_BOT = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT = os.getenv('TELEGRAM_CHAT_ID')

# Basic guardrails on the read-only query tool.
SELECT_ONLY_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE)\b",
    re.IGNORECASE,
)
MULTI_STATEMENT_RE = re.compile(r";\s*\S")  # semicolon followed by more content


class NepseEngine:
    """Thin wrapper around the nepse-engine reporter API."""

    def query(self, sql: str, params: dict | None = None) -> list[dict] | dict:
        """Execute a parameterized SQL query via the reporter API."""
        try:
            response = requests.post(
                f'{REPORTER_API}/reporter/run',
                json={'sql': sql, 'params': params or {}},
                timeout=10,
            )
            if response.status_code != 200:
                return {"error": response.text}
            return response.json().get('rows', [])
        except Exception as e:
            return {"error": str(e)}

    def get_schema(self, table: str | None = None) -> list[dict]:
        """Introspect table/column structure via information_schema.
        If table is None, returns columns for all tables in 'public'.
        """
        sql = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """
        params = {}
        if table:
            sql += " AND table_name = %(table)s"
            params["table"] = table
        sql += " ORDER BY table_name, ordinal_position"
        return self.query(sql, params)

    def list_tables(self) -> list[dict]:
        sql = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        return self.query(sql)

    def run_query(self, sql: str, params: dict | None = None) -> list[dict] | dict:
        """Generic read-only query. Only SELECT, single statement, parameterized."""
        if not SELECT_ONLY_RE.match(sql):
            return {"error": "Only SELECT statements are allowed via run_query."}
        if FORBIDDEN_KEYWORDS_RE.search(sql):
            return {"error": "Query contains a disallowed keyword."}
        if MULTI_STATEMENT_RE.search(sql):
            return {"error": "Multiple statements are not allowed."}
        return self.query(sql, params)

    def send_telegram_alert(self, message: str) -> bool:
        if not TELEGRAM_BOT or not TELEGRAM_CHAT:
            return False
        try:
            requests.post(
                f'https://api.telegram.org/bot{TELEGRAM_BOT}/sendMessage',
                json={'chat_id': TELEGRAM_CHAT, 'text': message},
                timeout=5,
            )
            return True
        except Exception:
            return False


# MCP Tool Definitions
TOOLS = [
    {
        "name": "list_tables",
        "description": "List all tables in the nepse-engine database (public schema).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_schema",
        "description": (
            "Get column names, types, and nullability for one table or all tables. "
            "Call this before writing a query if you're unsure of the schema."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "Optional table name. Omit to get schema for all tables.",
                }
            },
        },
    },
    {
        "name": "run_query",
        "description": (
            "Run a read-only, parameterized SELECT query against the nepse-engine "
            "database. Use get_schema first if you don't know the table/column names. "
            "Use %(name)s placeholders in sql and supply matching values in params."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement."},
                "params": {
                    "type": "object",
                    "description": "Named parameters referenced in the SQL as %(name)s.",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "send_alert",
        "description": "Send a Telegram alert notification.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Message to send"}},
            "required": ["message"],
        },
    },
]


def process_tool_call(tool_name: str, tool_input: dict):
    """Dispatch an MCP tool call to the engine."""
    engine = NepseEngine()

    if tool_name == "list_tables":
        return engine.list_tables()

    elif tool_name == "get_schema":
        return engine.get_schema(table=tool_input.get("table"))

    elif tool_name == "run_query":
        return engine.run_query(tool_input["sql"], tool_input.get("params"))

    elif tool_name == "send_alert":
        success = engine.send_telegram_alert(tool_input["message"])
        return {"sent": success}

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# Real MCP protocol server (stdio transport), for use by Claude Code / any
# MCP client via `claude mcp add`. Thin wrappers around NepseEngine — read-only,
# no tool here can write to the database.
from mcp.server import MCPServer

mcp = MCPServer("nepse-engine")
_engine = NepseEngine()


@mcp.tool()
def list_tables() -> list[dict]:
    """List all tables in the nepse-engine database (public schema)."""
    return _engine.list_tables()


@mcp.tool()
def get_schema(table: str | None = None) -> list[dict]:
    """Get column names, types, and nullability for one table or all tables.
    Call this before writing a query if you're unsure of the schema.
    """
    return _engine.get_schema(table)


@mcp.tool()
def run_query(sql: str, params: dict | None = None) -> list[dict] | dict:
    """Run a read-only, parameterized SELECT query against the nepse-engine
    database. Use get_schema first if you don't know the table/column names.
    Use %(name)s placeholders in sql and supply matching values in params.
    """
    return _engine.run_query(sql, params)


@mcp.tool()
def send_alert(message: str) -> dict:
    """Send a Telegram alert notification."""
    return {"sent": _engine.send_telegram_alert(message)}


if __name__ == "__main__":
    mcp.run()