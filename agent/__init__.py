"""
agent/__init__.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Agentic WAIT Monitor
Public interface: run_wait_monitor() only.

To enable:  UPDATE settings SET value='true' WHERE key='AGENTIC_WAIT_MONITOR';
To disable: UPDATE settings SET value='false' WHERE key='AGENTIC_WAIT_MONITOR';
To purge:   Delete this agent/ directory. Remove 3 lines from main.py.
─────────────────────────────────────────────────────────────────────────────
"""

from agent.agent import run_wait_monitor

__all__ = ["run_wait_monitor"]
