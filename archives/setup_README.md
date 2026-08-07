# NEPSE AI Engine — Local Ubuntu Setup Guide

**Version:** April 6, 2026  
**Phase:** Phase 3 — Active Build + Backtesting Complete  
**Pipeline:** filter_engine ✅ → gemini_filter ⚠️ → claude_analyst ✅ → auditor ✅  
**Blocker:** `gemini_filter.py` — 2 testing hardcodes NOT reverted (since March 23)

---

## Before You Start — Critical Blockers

**Fix these BEFORE running anything. They take 5 minutes.**

### BLOCKER 1 & 2: `gemini_filter.py` hardcodes

Open `gemini_filter.py` and find these two lines:

```python
# BROKEN — allows unlimited positions
slots_remaining = 99

# BROKEN — never blocks when portfolio full
if slots_remaining == 0 and len(open_positions) >= 99:
```

Fix them to:

```python
# CORRECT
slots_remaining = max(0, 3 - len(open_positions))

# CORRECT
if slots_remaining == 0 and len(open_positions) >= 3:
```

**Do not skip this. These have been unfixed since March 23.**

---

## Step 1 — Prepare the Machine

### 1.1 Ubuntu requirements

Ubuntu 22.04 or 24.04 desktop. Fresh install is fine.

Make sure the machine BIOS has:
- **Wake on RTC** enabled (so `rtcwake` can wake from sleep)
- **Secure Boot** OFF (optional but avoids driver headaches)

To check if your BIOS supports RTC wake:

```bash
sudo rtcwake -m no -s 60
# If no error, RTC wake works. Machine will schedule a wake 60s from now.
# Cancel it: sudo rtcwake -m disable
```

### 1.2 Clone the project

```bash
sudo mkdir -p /opt/nepse-engine
sudo chown $USER:$USER /opt/nepse-engine
cd /opt/nepse-engine
git clone <your-github-repo-url> .
```

---

## Step 2 — Configure Environment

```bash
cp setup/.env.example .env
nano .env
```

Fill in every value. Reference:

| Variable | Description |
|---|---|
| `DATABASE_URL` | Neon PostgreSQL. Format: `postgresql://user:pass@host/db?sslmode=require` |
| `GEMINI_API_KEY` | Google Gemini — screening, news, captcha, wealth mgmt |
| `GEMINI_MODEL` | `gemini-2.5-flash` — NOT `gemini-2.5-flash-lite` (causes 404) |
| `OPENROUTER_API_KEY` | For Claude, GPT-4o, DeepSeek (single billing) |
| `CLAUDE_MODEL` | `anthropic/claude-sonnet-4-5` |
| `GPT_MODEL` | `openai/gpt-4o` |
| `DEEPSEEK_MODEL` | `deepseek/deepseek-r1` |
| `TMS_USERNAME` | TMS49 broker login username |
| `TMS_PASSWORD` | TMS49 broker login password |
| `TMS_REQUEST_OWNER` | `109268` |
| `TMS_CLIENT_ID` | `2181770` |
| `MEROSHARE_USERNAME` | Meroshare CDSC username |
| `MEROSHARE_PASSWORD` | Meroshare CDSC password |
| `MEROSHARE_DP_ID` | Your DP ID |
| `MEROSHARE_DEMAT` | `1301180000232764` |
| `TELEGRAM_BOT_TOKEN` | From @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your personal chat ID: `5432461414` |
| `EMAIL_USER` | Gmail address (optional) |
| `EMAIL_APP_PASS` | Gmail app password (not account password) |
| `PAPER_MODE` | `true` — do NOT change to `false` until 55% WR proven |

---

## Step 3 — Run the Installer

```bash
cd /opt/nepse-engine
bash setup/install.sh
```

This does everything in order:

1. Installs system packages (`python3`, `util-linux` for `rtcwake`, `libpq-dev`, etc.)
2. Creates Python virtual environment at `venv/`
3. Installs all Python dependencies from `requirements.txt`
4. Copies systemd service + timer files to `~/.config/systemd/user/`
5. Enables and starts all timers
6. Enables `loginctl linger` so services run without you being logged in
7. Adds sudoers rule for `rtcwake` + `systemctl suspend` (no password needed)
8. Runs DB migrations
9. Seeds default settings
10. Runs a dry-run smoke test of `main.py`

**Expected output at the end:**

```
✅ NEPSE AI Engine installed successfully!

System schedule:
  10:20 AM NST  → Machine wakes (set by rtcwake)
  10:30 AM NST  → Morning workflow (history, indicators, briefing)
  10:45 AM NST  → Trading loop starts (every 6 min)
   3:00 PM NST  → Market closes
   3:15 PM NST  → EOD workflow (auditor)
   3:45 PM NST  → Machine sleeps
  Sunday 5:45PM → Weekly review (machine stays on Sundays)
```

---

## Step 4 — Verify Everything Is Running

### Check timer status

```bash
systemctl --user list-timers
```

You should see all 5 timers listed with their next trigger time:

```
NEXT                         UNIT
Mon 2026-04-07 04:45:00 UTC  nepse-morning.timer
Mon 2026-04-07 05:00:00 UTC  nepse-trading.timer
Mon 2026-04-07 09:30:00 UTC  nepse-eod.timer
Sun 2026-04-12 12:00:00 UTC  nepse-weekly.timer
Mon 2026-04-07 10:00:00 UTC  nepse-sleep.timer
```

### Check Telegram bot is running

```bash
systemctl --user status nepse-bot
```

Should show `active (running)`. Then message your bot on Telegram — it should respond.

### View live logs

```bash
# Trading loop (most important)
journalctl --user -u nepse-trading -f

# Morning workflow
journalctl --user -u nepse-morning -f

# EOD
journalctl --user -u nepse-eod -f

# Telegram bot
journalctl --user -u nepse-bot -f

# All NEPSE logs together
journalctl --user -f | grep nepse
```

---

## Step 5 — Test Before First Real Run

### Smoke test (no DB writes, no API calls)

```bash
source venv/bin/activate
python main.py --dry-run --skip-guard
```

### Test morning workflow manually

```bash
python -m workflows.morning_workflow --dry-run --skip-guard
```

### Test with real API calls but outside market hours

```bash
python main.py -paper --skip-guard
```

This runs the full pipeline (scraper → filter → Gemini → Claude) and writes to DB. Good for verifying everything connects.

---

## Daily Operations — What Happens Automatically

You do nothing. The system manages itself:

| NST Time | What Happens | You See |
|---|---|---|
| 10:20 AM | Machine wakes from sleep | Screen turns on |
| 10:30 AM | Morning workflow runs | Telegram briefing message |
| 10:45 AM | Trading loop starts | Loop runs every 6 min silently |
| Market hours | AI analyzes, signals fire | Telegram BUY alerts if signals found |
| 3:00 PM | Market closes | Loop exits cleanly |
| 3:15 PM | EOD auditor runs | Telegram EOD summary |
| 3:45 PM | Machine sleeps | Screen goes dark |
| Sunday 5:45 PM | Weekly review | Telegram weekly report |

---

## Telegram Bot — Your Interface

The bot (`telegram_bot.py`) runs 24/7. It is how you interact with paper trading.

### Your commands

| Command | What It Does |
|---|---|
| `/register` | Register for paper trading |
| `/buy NABIL 10 1245` | Record a paper buy (10 shares at NPR 1245) |
| `/sell NABIL 10 1350` | Record a paper sell |
| `/status` | See your open positions |
| `/pnl` | See your P&L summary |
| `/capital` | See remaining capital |
| `/signal` | See today's AI signals |
| `/mode` | Check paper vs live mode |

### Admin commands

| Command | What It Does |
|---|---|
| `/approve @username` | Approve a paper trading user |
| `/users` | List all registered users |

---

## Going Live — When Ready

**Only after:** Win rate ≥ 55% across 30+ paper trades.

### Step 1 — Update the trading service

```bash
nano ~/.config/systemd/user/nepse-trading.service
```

Change this line:

```ini
# FROM:
ExecStart=/opt/nepse-engine/venv/bin/python main.py -paper

# TO:
ExecStart=/opt/nepse-engine/venv/bin/python main.py -live
```

### Step 2 — Reload and restart

```bash
systemctl --user daemon-reload
systemctl --user restart nepse-trading.timer
```

### Step 3 — Update the DB setting

```bash
source venv/bin/activate
python -c "from sheets import update_setting; update_setting('PAPER_MODE', 'false')"
```

**Live mode activates:**
- Circuit breaker (loss streak > 7 → halt)
- Geo block (combined geo < -3 → halt)  
- BUY alerts go to Telegram with real NPR amounts
- You execute trades manually at your broker

---

## Schedule Reference (UTC ↔ NST Conversion)

NST = UTC + 5:45

| Event | NST | UTC |
|---|---|---|
| Machine wakes | 10:20 AM | 04:35 |
| Morning workflow | 10:30 AM | 04:45 |
| Market opens | 10:45 AM | 05:00 |
| Trading loop runs | Every 6 min | Every 6 min from 05:00 |
| Market closes | 3:00 PM | 09:15 |
| EOD workflow | 3:15 PM | 09:30 |
| Machine sleeps (Mon–Thu) | 3:45 PM | 10:00 |
| Weekly review (Sunday) | 5:45 PM | 12:00 |
| Machine sleeps (Sunday) | 6:15 PM | 12:30 |
| Nightly context summary | 9:00 PM | 15:15 |

---

## Known Bugs (Fix Before Going Live)

| File | Issue | Fix |
|---|---|---|
| `gemini_filter.py` | `slots_remaining = 99` | `max(0, 3 - len(open_positions))` |
| `gemini_filter.py` | Portfolio check `>= 99` | Change to `>= 3` |
| `capital_allocator.py` | Wrong meroshare import | `from modules.meroshare import get_portfolio_summary` |
| `capital_allocator.py` | `%,.0f` crashes log | Replace with `%.0f` |
| `briefing.py` | Wrong meroshare import | Same fix as above |
| `budget.py` | Kelly Criterion error | Fix DeepSeek calculation before live |
| `nepal_pulse.py` | Duplicate `nrb_rate_decision` key | Remove second occurrence in SCORE_MAP |
| `nepal_pulse.py` | `'ict'` typo | Change to `'inflation_pct'` |

---

## Useful Commands Reference

```bash
# View all timer schedules
systemctl --user list-timers

# Manually trigger morning workflow now
systemctl --user start nepse-morning.service

# Manually trigger one trading cycle now
systemctl --user start nepse-trading.service

# Watch trading loop in real time
journalctl --user -u nepse-trading -f

# Check if Telegram bot is alive
systemctl --user status nepse-bot

# Restart Telegram bot
systemctl --user restart nepse-bot

# Run dry-run test
python main.py --dry-run --skip-guard

# Check next wake time
python setup/sleep_scheduler.py --status

# Manually suspend machine now (schedules RTC wake first)
python setup/sleep_scheduler.py

# Stop all NEPSE services temporarily
systemctl --user stop nepse-trading.timer nepse-morning.timer nepse-eod.timer

# Start them again
systemctl --user start nepse-trading.timer nepse-morning.timer nepse-eod.timer

# Check DB connection
python -c "from db.connection import test_connection; test_connection()"

# Seed/reset DB settings
python -m db.migrations
```

---

## What Each File Does

| File | Purpose |
|---|---|
| `main.py` | Trading loop. Called every 6 min. `-paper` or `-live`. |
| `workflows/morning_workflow.py` | 10:30 AM sequence. History, indicators, briefing. |
| `workflows/eod_workflow.py` | 3:15 PM sequence. Auditor, recommendation tracker. |
| `workflows/weekly_workflow.py` | Sunday 5:45 PM. Learning Hub, capital allocator. |
| `setup/install.sh` | One-shot Ubuntu setup. Run once. |
| `setup/sleep_scheduler.py` | Sets RTC wake alarm + suspends machine. |
| `setup/nepse-bot.service` | Telegram bot — persistent, always running. |
| `setup/nepse-morning.service/.timer` | Morning workflow trigger. |
| `setup/nepse-trading.service/.timer` | Trading loop trigger (every 6 min). |
| `setup/nepse-eod.service/.timer` | EOD workflow trigger. |
| `setup/nepse-weekly.service/.timer` | Weekly review trigger. |
| `setup/nepse-sleep.service/.timer` | Sleep scheduler trigger. |
| `setup/.env.example` | Environment variable template. |

---

## Monthly Cost

| Item | Cost |
|---|---|
| Claude (22 trading days, 3-5 calls/day) | ~$2–3 |
| Gemini Flash (all uses) | ~$0.50 |
| GPT Sunday review | ~$0.06 |
| DeepSeek Kelly | ~$0.02 |
| **TOTAL** | **~$2.60–3.60/month** |

Within NPR 500 budget.

---

## Core Rules — Never Break These

- RSI is **context only** — never a standalone buy trigger (Karki 2023: -4.81% annualized)
- Max single position: **10% of total capital**
- Default stop loss: **3% hard stop** — no emotional override
- Trailing stop: activates at **+5% gain**, trails 3% below peak
- Max simultaneous positions: **3**
- Never trade on **bandh day** — zero liquidity
- Geo combined < -3 → **automatic block**
- Loss streak > 7 → **circuit breaker**
- `PAPER_MODE = true` until **55% WR proven across 30+ trades**
- Hard blocked sectors: **Manufacturing** (PF=0.06), **Commercial Banks** (PF=0.92), **Hotels** (PF=0.32)

---

**Start with paper trading. Prove the system. Scale at confidence > 75%.**  
**The Learning Hub is the real long-term asset. Record every trade. No exceptions.**
