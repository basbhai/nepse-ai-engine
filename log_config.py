"""
log_config.py — NEPSE AI Engine
================================
Attaches a timestamped FileHandler to the root logger so every module's
log output (which propagates to root) is captured in a per-run text file.

Usage — call once at the top of every `if __name__ == "__main__":` block,
AFTER any `logging.basicConfig(...)` call:

    from log_config import attach_file_handler
    attach_file_handler(__name__)

The console output is unchanged.  File is created under logs/ at the repo root.
Filename format:  logs/<module>_<YYYYMMDD_HHMMSS>.txt
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

_ROOT     = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_ROOT, "logs")
_NST      = ZoneInfo("Asia/Kathmandu")

# Track installed names so repeated imports / calls are safe
_installed: set[str] = set()


def attach_file_handler(module_name: str) -> str | None:
    """
    Add a FileHandler to the root logger.

    Parameters
    ----------
    module_name : str
        Pass ``__name__`` from the calling module.
        e.g. "analysis.learning_hub" → log file ``learning_hub_20260412_174500.txt``

    Returns
    -------
    str | None
        Absolute path to the log file, or None if already installed.
    """
    if module_name in _installed:
        return None
    _installed.add(module_name)

    os.makedirs(_LOGS_DIR, exist_ok=True)

    # Derive a short, clean name
    if module_name in ("__main__", "__mp_main__"):
        short = "main"
    else:
        short = module_name.split(".")[-1]

    timestamp = datetime.now(_NST).strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(_LOGS_DIR, f"{short}_{timestamp}.txt")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    root = logging.getLogger()
    root.addHandler(fh)
    root.info("File logging → %s", log_path)
    return log_path
