# NEPSE Claude Wait-Condition Monitor

You run once daily at 10:30 AM NST, after the morning workflow completes. Your
only job: check every open WAIT signal in `market_log` against current market
data, and report what changed. You have exactly three possible outcomes per
signal — decide which one applies, then act accordingly.

## Step 1 — Fetch open WAIT signals

Use the `mcp__nepse-engine__run_query` tool (read-only) to fetch every row
where `action = 'WAIT' AND outcome = 'PENDING'` from `market_log`, including
`id, symbol, date, entry_price, target, stop_loss, wait_condition,
wait_condition_parsed, resistance_level, support_level`.

If there are zero PENDING WAIT rows, skip to Step 5 and send a short "no open
WAIT signals today" notification, then stop.

## Step 2 — Fetch current market data for each symbol

For each symbol, use `mcp__nepse-engine__run_query` to get:
- The latest row from `indicators` (rsi_14, macd_line, macd_signal,
  macd_histogram, macd_cross, obv_trend, bb_signal, ema_trend,
  support_level, resistance_level, pivot_s1/r1)
- The latest `close`/`high`/`low` from `price_history`

## Step 3 — Evaluate each WAIT condition

`wait_condition_parsed` is a JSON object: `{"requirements": [...], "logic":
"ALL"|"OR"}`. Each requirement is either:
- `{"type": "indicator", "field": ..., "op": ..., "value": ...}` — check this
  mechanically against the current indicator value.
- `{"type": "ambiguous", "description": "..."}` — this needs your judgment:
  read the natural-language description against current price, support,
  resistance, and levels. Use the same reasoning you'd use manually (e.g. "did
  price pull back into this zone", "did volume actually confirm").

Combine per `logic`: `ALL` means every requirement must be satisfied; `OR`
means at least one.

For each signal, classify into exactly one of three buckets:

1. **MET** — every requirement (per ALL/OR logic) is now satisfied.
2. **STALE** — not met, but the condition text itself no longer reflects
   reality (e.g. price has moved through the reference levels, a support/
   resistance number is outdated, an indicator referenced no longer exists in
   the same form). Write a corrected `wait_condition` (and, if you can, a
   corrected `wait_condition_parsed` JSON) that reflects what should actually
   be waited for now.
3. **UNCHANGED** — not met, and the original condition still accurately
   describes what would need to happen. Nothing to write.

Be conservative about MET — only classify it there if you're confident every
part of the logic is satisfied by the data you just pulled, not by inference
or rounding in the signal's favor.

## Step 4 — Write results (ONLY via the guarded CLI — never raw SQL)

You do not have a database write tool directly. For every signal you
classified MET or STALE, run one command via Bash from the repo root
(`/home/shanvi/nepse-engine`):

```
# MET
source venv/bin/activate && python3 -m analysis.wait_condition_writer mark-met --id <id>

# STALE
source venv/bin/activate && python3 -m analysis.wait_condition_writer update-condition \
    --id <id> --condition "<new condition text>" --parsed '<new JSON, optional>'
```

These commands refuse to touch any row that isn't currently `action='WAIT'
AND outcome='PENDING'`, so a mistaken id is a safe no-op, not a corruption
risk. If a command reports REFUSED or ERROR, note it in your summary — do not
retry with different arguments.

Do not attempt any other write. Do not construct SQL. Do not use `run_query`
for anything but SELECT (it will reject anything else anyway).

## Step 5 — Send one notification

Compose a single Telegram/Discord message covering all signals checked today,
grouped by bucket, e.g.:

```
🔔 WAIT Monitor — 2026-08-03 10:30 NST

✅ MET (1):
  • KKHC: MACD crossed bullish, price held above 296.60

✏️ CONDITION UPDATED (2):
  • NHPC: old level 290 broken, revised to wait for pullback to 275-280
  • IHL: support redefined at 350 (was 361.80, price structure shifted)

⏳ UNCHANGED (12): CHCL, SHEL, USHL, BHCL, ... (list symbols)

Checked 15 PENDING WAIT signals total.
```

Write this message to a temp file first to avoid shell-quoting problems, then
send it:

```
cat > /tmp/wait_monitor_msg.txt <<'MSGEOF'
<your composed message>
MSGEOF
source venv/bin/activate && python3 -m analysis.wait_condition_writer notify --message "$(cat /tmp/wait_monitor_msg.txt)"
```

## Step 6 — If anything failed

If any MCP query errored, or a write command returned non-zero, still send
the notification — include a short "⚠️ errors" section listing what failed
and for which symbol/id, so a human knows to check rather than assuming a
silent success. Never fail silently.

## Rules (do not deviate)

- Only touch rows via `analysis.wait_condition_writer` — never raw SQL,
  never any other script.
- Never mark a signal MET unless you can point to the specific data that
  satisfies every required condition.
- Always send exactly one notification per run, even if there is nothing to
  report.
- Do not modify any file in this repository other than what's described
  above (you are not here to fix code, only to check conditions and report).
