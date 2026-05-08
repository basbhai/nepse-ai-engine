"""
agent/__init__.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Agentic WAIT Monitor
Public interface: run_wait_monitor() and run_wait_pipeline().

To enable:  UPDATE settings SET value='true' WHERE key='AGENTIC_WAIT_MONITOR';
To disable: UPDATE settings SET value='false' WHERE key='AGENTIC_WAIT_MONITOR';
To purge:   Delete this agent/ directory. Remove 3 lines from main.py.
─────────────────────────────────────────────────────────────────────────────
"""

from agent.agent import run_wait_monitor
from agent.wait_pipeline import run_wait_pipeline

__all__ = ["run_wait_monitor", "run_wait_pipeline"]
