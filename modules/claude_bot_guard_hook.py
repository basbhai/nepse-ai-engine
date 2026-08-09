#!/usr/bin/env python3
"""
PreToolUse guard for headless bot-triggered Claude runs (/claude, /play).
Denies Edit/Write on any *.py or .env file that already exists on disk
(i.e. modifying live code or secrets). Allows Write to brand-new paths
(reports, one-off scripts, CSVs, etc. — "copy files") regardless of
extension, and allows Edit/Write on non-.py/.env existing files.

WORKSPACE_DIR (/home/shanvi/claude_agent_reports) is the bot's own scratch
directory — it's fully exempt from the "existing .py file" restriction (the
bot can freely create AND edit .py files there across runs, not just
create brand-new ones), since nothing under it is live project code. .env
stays blocked everywhere, including inside WORKSPACE_DIR — no exception for
secrets.

Bash is denied by default (/play — Playwright tools only, no shell). It is
allowed only when CLAUDE_BOT_GUARD_MODE=ondemand (set by trading_core.py for
/claude runs only), and even then a best-effort regex scan still blocks any
command that looks like it writes to a .env file, or to a .py file outside
WORKSPACE_DIR, via shell redirection, sed/perl -i, tee, or cp/mv — mirroring
the Edit/Write rule above. This scan is best-effort, not airtight (arbitrary
shell can obfuscate a write), so /claude admins are trusted more than /play
by this widening.
"""
import json
import os
import re
import sys

WORKSPACE_DIR = "/home/shanvi/claude_agent_reports"


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def _in_workspace(path: str) -> bool:
    try:
        resolved = os.path.realpath(path)
    except Exception:
        resolved = path
    return resolved == WORKSPACE_DIR or resolved.startswith(WORKSPACE_DIR + os.sep)


# Best-effort patterns for a Bash command that writes into a .env file, or a
# .py file, via redirection (>, >>, >|), in-place sed/perl edits, tee, or
# cp/mv destinations.
_ENV_EXT = r"\.env\S*"
_PY_EXT = r"\S*\.py"
_ENV_PATTERNS = [
    re.compile(r">>?\|?\s*" + _ENV_EXT),
    re.compile(r"\bsed\b[^|;&\n]*-i[^|;&\n]*" + _ENV_EXT),
    re.compile(r"\bperl\b[^|;&\n]*-i[^|;&\n]*" + _ENV_EXT),
    re.compile(r"\btee\b[^|;&\n]*" + _ENV_EXT),
    re.compile(r"\b(?:cp|mv)\b[^|;&\n]*" + _ENV_EXT),
    re.compile(r"\bdd\b[^|;&\n]*of=" + _ENV_EXT),
]
_PY_PATTERNS = [
    re.compile(r">>?\|?\s*" + _PY_EXT),
    re.compile(r"\bsed\b[^|;&\n]*-i[^|;&\n]*" + _PY_EXT),
    re.compile(r"\bperl\b[^|;&\n]*-i[^|;&\n]*" + _PY_EXT),
    re.compile(r"\btee\b[^|;&\n]*" + _PY_EXT),
    re.compile(r"\b(?:cp|mv)\b[^|;&\n]*" + _PY_EXT),
    re.compile(r"\bdd\b[^|;&\n]*of=" + _PY_EXT),
]


def bash_blocked_reason(command: str) -> str | None:
    """Return 'env', 'py', or None. WORKSPACE_DIR is exempt from the 'py'
    check (substring match on the command — best-effort, like the rest of
    this scan) but never from the 'env' check."""
    if any(p.search(command) for p in _ENV_PATTERNS):
        return "env"
    if WORKSPACE_DIR not in command and any(p.search(command) for p in _PY_PATTERNS):
        return "py"
    return None


def main() -> None:
    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open on parse error — don't crash the run over a hook bug

    tool_name = hook_input.get("tool_name")

    if tool_name == "Bash":
        command = (hook_input.get("tool_input") or {}).get("command") or ""
        if os.environ.get("CLAUDE_BOT_GUARD_MODE") != "ondemand":
            deny("Blocked: Bash is not available to this bot — use the provided MCP/Playwright tools only.")
        reason = bash_blocked_reason(command)
        if reason == "env":
            deny(f"Blocked: this bot may not use Bash to write to a .env file ({command[:200]}).")
        if reason == "py":
            deny(
                f"Blocked: this bot may not use Bash to write to a .py file outside "
                f"{WORKSPACE_DIR} ({command[:200]})."
            )
        sys.exit(0)

    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        sys.exit(0)

    file_path = (hook_input.get("tool_input") or {}).get("file_path") or \
                (hook_input.get("tool_input") or {}).get("notebook_path")
    if not file_path:
        sys.exit(0)

    base = os.path.basename(file_path)

    if base == ".env" or base.endswith(".env"):
        deny(f"Blocked: this bot may never write or edit an .env file ({file_path}).")

    is_py = base.endswith(".py") or base.endswith(".ipynb")
    if is_py and os.path.exists(file_path) and not _in_workspace(file_path):
        deny(
            f"Blocked: this bot may not modify an existing core Python file "
            f"({file_path}). Creating a brand-new file is fine, and anything "
            f"under {WORKSPACE_DIR} is always editable."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
