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
Filename format:  logs/<YYYY>/<MonthName>/<DD>/<module>_<YYYYMMDD_HHMMSS>.txt
"""

import inspect
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

_ROOT = os.path.dirname(os.path.abspath(__file__))
_NST  = ZoneInfo("Asia/Kathmandu")

# Track installed names so repeated imports / calls are safe
_installed: set[str] = set()


def _entry_point_name(fallback: str) -> str:
    """
    Walk the call stack to the bottom frame and derive the short script name.
    Falls back to fallback if stack inspection fails for any reason.
    """
    try:
        frames = inspect.stack()
        bottom = frames[-1]
        filename = bottom.filename
        if filename and filename not in ("<stdin>", "<string>", ""):
            return os.path.splitext(os.path.basename(filename))[0]
    except Exception:
        pass
    return fallback


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
        Absolute path to the log file, or None if already installed or logs disabled.
    """
    if module_name in _installed:
        return None
    _installed.add(module_name)

    # Settings gate — skip file creation if LOGS_ENABLED is not true/1
    try:
        import sys
        _modules_dir = os.path.join(_ROOT, "modules")
        if _modules_dir not in sys.path:
            sys.path.insert(0, _modules_dir)
        # sheets.py lives at the repo root
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from sheets import get_setting
        val = get_setting("LOGS_ENABLED", "")
        if val.strip().lower() not in ("true", "1"):
            return None
    except Exception:
        return None

    # Derive a short, clean name from the true entry-point script
    fallback = "main" if module_name in ("__main__", "__mp_main__") else module_name.split(".")[-1]
    short = _entry_point_name(fallback)

    now = datetime.now(_NST)
    year       = now.strftime("%Y")
    month_name = now.strftime("%B")   # full English month name, e.g. "April"
    day        = now.strftime("%d")
    timestamp  = now.strftime("%Y%m%d_%H%M%S")

    log_dir  = os.path.join(_ROOT, "logs", year, month_name, day)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{short}_{timestamp}.txt")

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
