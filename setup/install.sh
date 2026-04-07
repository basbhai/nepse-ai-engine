#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup/install.sh — NEPSE AI Engine
# One-shot setup for fresh Ubuntu 22.04/24.04 desktop.
# Run ONCE after cloning the repo and filling in .env
#
# Usage:
#   cd /opt/nepse-engine
#   cp setup/.env.example .env
#   nano .env                    ← fill in all secrets
#   bash setup/install.sh
#
# What this does:
#   1. Installs system packages (Python, rtcwake, etc.)
#   2. Creates a Python virtual environment
#   3. Installs all Python dependencies
#   4. Copies systemd service + timer files
#   5. Enables and starts all timers
#   6. Adds sudoers rule for rtcwake (wake/sleep)
#   7. Runs DB migrations
#   8. Seeds initial settings
#   9. Runs a dry-run test of main.py
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}✅ $1${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $1${RESET}"; }
info() { echo -e "${CYAN}   $1${RESET}"; }
fail() { echo -e "${RED}❌ $1${RESET}"; exit 1; }
step() { echo -e "\n${BOLD}── $1${RESET}"; }

# ── Must run from project root ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || fail "Cannot cd to project dir: $PROJECT_DIR"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         NEPSE AI Engine — Ubuntu Setup                      ║"
echo "║         Project: $PROJECT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Check .env exists ─────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    fail ".env not found. Copy setup/.env.example to .env and fill in secrets first."
fi
ok ".env found"

# ── Get current user ─────────────────────────────────────────────────────────
NEPSE_USER="$(whoami)"
info "Installing for user: $NEPSE_USER"
info "Project directory:   $PROJECT_DIR"

# ── Step 1: System packages ───────────────────────────────────────────────────
step "1/9  Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    util-linux \
    git curl wget \
    libpq-dev \
    build-essential \
    python3-dev \
    2>/dev/null
ok "System packages installed"

# ── Step 2: Python venv ───────────────────────────────────────────────────────
step "2/9  Creating Python virtual environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "venv created"
else
    ok "venv already exists"
fi

# Activate venv
source venv/bin/activate

# ── Step 3: Python dependencies ───────────────────────────────────────────────
step "3/9  Installing Python dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Python dependencies installed"

# Install google-genai (new SDK — not in requirements.txt yet)
pip install google-genai -q 2>/dev/null || warn "google-genai install failed — check manually"
pip install yfinance -q 2>/dev/null || warn "yfinance install failed"

# ── Step 4: Systemd services ──────────────────────────────────────────────────
step "4/9  Installing systemd services"
SYSTEMD_DIR="$HOME/.config/systemd/user"
mkdir -p "$SYSTEMD_DIR"

# Service files use %i for user — replace with actual username in service copies
for svc in nepse-bot nepse-morning nepse-trading nepse-eod nepse-weekly nepse-sleep; do
    # Replace %i with actual user in service file before installing
    sed "s|%i|$NEPSE_USER|g; s|/opt/nepse-engine|$PROJECT_DIR|g" \
        "setup/${svc}.service" > "$SYSTEMD_DIR/${svc}.service"
    ok "Installed $svc.service"
done

for tmr in nepse-morning nepse-trading nepse-eod nepse-weekly nepse-sleep; do
    cp "setup/${tmr}.timer" "$SYSTEMD_DIR/${tmr}.timer"
    ok "Installed $tmr.timer"
done

# Reload systemd user daemon
systemctl --user daemon-reload
ok "systemd daemon reloaded"

# ── Step 5: Enable and start timers ──────────────────────────────────────────
step "5/9  Enabling systemd timers"

systemctl --user enable nepse-morning.timer
systemctl --user enable nepse-trading.timer
systemctl --user enable nepse-eod.timer
systemctl --user enable nepse-weekly.timer
systemctl --user enable nepse-sleep.timer
systemctl --user enable nepse-bot.service

# Start timers now (they'll fire at the next scheduled time)
systemctl --user start nepse-morning.timer
systemctl --user start nepse-trading.timer
systemctl --user start nepse-eod.timer
systemctl --user start nepse-weekly.timer
systemctl --user start nepse-sleep.timer

# Start Telegram bot immediately
systemctl --user start nepse-bot.service

ok "All timers enabled and started"

# Enable lingering so services run even when you're not logged in
sudo loginctl enable-linger "$NEPSE_USER"
ok "Linger enabled for $NEPSE_USER (services run without login)"

# ── Step 6: Sudoers for rtcwake (wake/sleep) ──────────────────────────────────
step "6/9  Adding sudoers rule for rtcwake + suspend"
SUDOERS_FILE="/etc/sudoers.d/nepse-wakesleep"
if [ ! -f "$SUDOERS_FILE" ]; then
    echo "$NEPSE_USER ALL=(ALL) NOPASSWD: /usr/sbin/rtcwake" | sudo tee "$SUDOERS_FILE" > /dev/null
    echo "$NEPSE_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl suspend" | sudo tee -a "$SUDOERS_FILE" > /dev/null
    sudo chmod 440 "$SUDOERS_FILE"
    ok "Sudoers rule added (rtcwake + suspend without password)"
else
    ok "Sudoers rule already exists"
fi

# ── Step 7: DB migrations ────────────────────────────────────────────────────
step "7/9  Running database migrations"
python -m db.migrations && ok "DB migrations complete" || warn "DB migrations failed — check DATABASE_URL in .env"

# ── Step 8: Seed settings ────────────────────────────────────────────────────
step "8/9  Seeding default settings"
python -c "
from db.migrations import seed_settings
seed_settings()
print('Settings seeded')
" && ok "Settings seeded" || warn "Settings seed failed — may already exist (OK)"

# ── Step 9: Dry-run smoke test ────────────────────────────────────────────────
step "9/9  Smoke test (dry-run)"
python main.py --dry-run --skip-guard && ok "Smoke test passed" || warn "Smoke test had warnings — check output above"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  NEPSE AI Engine installed successfully!                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo ""
echo -e "${BOLD}System schedule:${RESET}"
echo "  10:20 AM NST  → Machine wakes (set by rtcwake)"
echo "  10:30 AM NST  → Morning workflow (history, indicators, briefing)"
echo "  10:45 AM NST  → Trading loop starts (every 6 min)"
echo "   3:00 PM NST  → Market closes"
echo "   3:15 PM NST  → EOD workflow (auditor)"
echo "   3:45 PM NST  → Machine sleeps"
echo "  Sunday 5:45PM → Weekly review (machine stays on Sundays)"
echo ""
echo -e "${BOLD}Useful commands:${RESET}"
echo "  View trading logs:    journalctl --user -u nepse-trading -f"
echo "  View morning logs:    journalctl --user -u nepse-morning -f"
echo "  View EOD logs:        journalctl --user -u nepse-eod -f"
echo "  View Telegram bot:    journalctl --user -u nepse-bot -f"
echo "  Timer status:         systemctl --user list-timers"
echo "  Test trading now:     python main.py --dry-run --skip-guard"
echo "  Go LIVE:              edit setup/nepse-trading.service → change -paper to -live"
echo "                        then: systemctl --user daemon-reload"
echo "                              systemctl --user restart nepse-trading.timer"
echo ""
echo -e "${YELLOW}⚠️  Default mode is PAPER TRADING. To go live, see README.md.${RESET}"
echo ""
