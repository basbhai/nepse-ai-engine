#!/usr/bin/env bash
# agents/claude_wait_monitor/run_wait_monitor.sh
# ─────────────────────────────────────────────────────────────────────────────
# Invoked by nepse-claude-wait-monitor.timer at 10:30 AM NST daily.
# Skips silently (exit 0) on non-trading days via calendar_guard — the
# systemd timer itself just fires every day, holiday/weekend logic lives here,
# same pattern as the other workflows in this repo.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="/home/shanvi/nepse-engine"
PROMPT_FILE="$REPO_DIR/agents/claude_wait_monitor/wait_monitor_prompt.md"
CLAUDE_BIN="$HOME/.local/bin/claude"

cd "$REPO_DIR"
source venv/bin/activate

if ! python3 -c "
from calendar_guard import is_trading_day, today_nst
import sys
sys.exit(0 if is_trading_day(today_nst()) else 1)
"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [wait-monitor] Not a trading day — skipping."
    exit 0
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [wait-monitor] Trading day confirmed — running Claude wait monitor."

"$CLAUDE_BIN" -p "$(cat "$PROMPT_FILE")" \
    --model claude-sonnet-5 \
    --allowedTools "mcp__nepse-engine__list_tables mcp__nepse-engine__get_schema mcp__nepse-engine__run_query Bash"

echo "$(date '+%Y-%m-%d %H:%M:%S') [wait-monitor] Run complete."
