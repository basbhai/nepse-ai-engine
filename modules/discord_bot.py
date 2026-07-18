"""
modules/discord_bot.py
──────────────────────
Nepal paper trading journal bot via Discord — read-only commands.
Mirrors the read commands of telegram_bot.py; all business logic
lives in modules/trading_core.py.

Run:
    python -m modules.discord_bot

ENV vars (all required unless noted):
    DISCORD_BOT_TOKEN   — bot token from Discord Developer Portal
    DISCORD_GUILD_ID    — guild to sync slash commands to instantly
    DISCORD_ADMIN_ID    — Discord snowflake of the admin user
    DISCORD_CHANNEL_ID  — (optional) default channel for future push alerts

Auth model:
    Every command resolves the calling Discord snowflake to a canonical
    telegram_id via resolve_telegram_id(discord_id, "DISCORD") from sheets.py.
    The paper_users table stores discord_id alongside telegram_id — a user
    must register at the dashboard (/register page) to link their Discord
    account before commands work.

Commands (all replies ephemeral):
    /help     — list commands, no auth required
    /mode     — PAPER/LIVE + sandbox indicator
    /capital  — paper capital state for the calling user
    /status   — open positions for the calling user
    /pnl      — closed trades + win rate for the calling user

NOT implemented here (dashboard-only or follow-up task):
    /register  — registration is dashboard-only by design
    /buy /sell /pause /resume /council_agenda — follow-up task
"""

import os
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation

import discord
from discord import app_commands

from config import NST
from sheets import run_raw_sql, resolve_telegram_id
from modules.trading_core import (
    get_paper_capital,
    get_paper_position,
    mode_label,
    fmt_npr,
    win_rate_str,
    market_gate_message,
    sandbox_label,
    circuit_label,
    hold_days,
    MAX_POSITIONS,
    calc_buy_fees, calc_sell_fees,
    execute_buy, execute_sell,
    lookup_symbols, count_open_positions,
)
from modules.trading_core import _circuit_breakers

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ─── ENV ─────────────────────────────────────────────────────────────────────
_token = os.environ.get("DISCORD_BOT_TOKEN")
if not _token:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is not set. "
        "Add it to your .env file and restart."
    )
DISCORD_BOT_TOKEN = _token

_guild_id = os.environ.get("DISCORD_GUILD_ID")
if not _guild_id:
    raise RuntimeError(
        "DISCORD_GUILD_ID is not set. "
        "Add it to your .env file and restart."
    )
GUILD = discord.Object(id=int(_guild_id))

DISCORD_ADMIN_ID  = os.environ.get("DISCORD_ADMIN_ID", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")

# Dashboard base URL for the registration link shown to unregistered users.
# Override with DASHBOARD_URL in .env if deploying to a public hostname.
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8766")
REGISTER_URL  = f"{DASHBOARD_URL}/register"


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def _is_admin(discord_user_id: int) -> bool:
    return str(discord_user_id) == DISCORD_ADMIN_ID


async def _resolve(interaction: discord.Interaction) -> str | None:
    """
    Resolve the interaction's Discord user to a canonical telegram_id.
    Replies ephemerally and returns None if the user is not registered.
    """
    tid = resolve_telegram_id(str(interaction.user.id), "DISCORD")
    if tid is None:
        await interaction.followup.send(
            "You're not registered. Register at the dashboard:\n"
            f"{REGISTER_URL}",
            ephemeral=True,
        )
    return tid


# ─── Bot setup ───────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.guilds = True
# message_content intent needed if we ever read message text; harmless to set now.
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


@bot.event
async def on_ready():
    try:
        synced = await tree.sync(guild=GUILD)
        log.info(
            "Discord bot ready: %s | guild commands synced: %d | guilds: %d",
            bot.user,
            len(synced),
            len(bot.guilds),
        )
    except Exception:
        log.exception("Failed to sync slash commands to guild %s", _guild_id)


# ═══════════════════════════════════════════════════════════════════════════
# /help
# ═══════════════════════════════════════════════════════════════════════════

@tree.command(guild=GUILD, name="help", description="List available commands")
async def cmd_help(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        embed = discord.Embed(
            title=f"Nepal Paper Trading Bot {sandbox_label()}",
            description=mode_label(),
            colour=discord.Colour.blurple(),
        )
        embed.add_field(
            name="Status commands (anytime)",
            value=(
                "`/capital` — your paper capital state\n"
                "`/status`  — your open positions\n"
                "`/pnl`     — closed trades + win rate\n"
                "`/mode`    — current trading mode\n"
            ),
            inline=False,
        )
        embed.add_field(
            name="Registration",
            value=(
                f"Register or link your Discord account at:\n{REGISTER_URL}"
            ),
            inline=False,
        )
        embed.set_footer(text="All replies are visible only to you.")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception:
        log.exception("cmd_help failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /mode
# ═══════════════════════════════════════════════════════════════════════════

@tree.command(guild=GUILD, name="mode", description="Show current trading mode (PAPER/LIVE + sandbox)")
async def cmd_mode(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return
        label = mode_label()
        cb    = circuit_label(tid)
        await interaction.followup.send(
            f"{label} | {cb}",
            ephemeral=True,
        )
    except Exception:
        log.exception("cmd_mode failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /capital
# ═══════════════════════════════════════════════════════════════════════════

@tree.command(guild=GUILD, name="capital", description="Show your paper capital state")
async def cmd_capital(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return

        cap    = get_paper_capital(tid)
        start  = Decimal(str(cap.get("starting_capital", "100000")))
        curr   = Decimal(str(cap.get("current_capital",  "100000")))
        pnl    = Decimal(str(cap.get("total_realised_pnl", "0")))
        fees   = Decimal(str(cap.get("total_fees_paid", "0")))
        cgt    = Decimal(str(cap.get("total_cgt_paid", "0")))
        growth = (
            ((curr - start) / start * 100).quantize(Decimal("0.01"))
            if start else Decimal("0")
        )

        locked_rows = run_raw_sql(
            "SELECT COALESCE(SUM(total_cost::numeric), 0) AS lk "
            "FROM paper_portfolio WHERE telegram_id = %s AND status = 'OPEN'",
            (str(tid),),
        )
        locked = Decimal(str(locked_rows[0]["lk"])) if locked_rows else Decimal("0")

        embed = discord.Embed(
            title=f"Paper Capital | {mode_label()}",
            colour=discord.Colour.green() if pnl >= 0 else discord.Colour.red(),
            timestamp=datetime.now(NST),
        )
        embed.add_field(name="Starting",       value=fmt_npr(start),          inline=True)
        embed.add_field(name="Current cash",   value=fmt_npr(curr),           inline=True)
        embed.add_field(name="Locked trades",  value=fmt_npr(locked),         inline=True)
        embed.add_field(name="Realised P&L",   value=fmt_npr(pnl, sign=True), inline=True)
        embed.add_field(name="Fees paid",      value=fmt_npr(fees),           inline=True)
        embed.add_field(name="CGT paid",       value=fmt_npr(cgt),            inline=True)
        embed.add_field(name="Overall growth", value=f"{growth:+.2f}%",       inline=True)
        embed.add_field(name="Win rate",       value=win_rate_str(cap),       inline=True)
        embed.add_field(
            name="Total trades",
            value=str(cap.get("total_trades", "0")),
            inline=True,
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception:
        log.exception("cmd_capital failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /status
# ═══════════════════════════════════════════════════════════════════════════

@tree.command(guild=GUILD, name="status", description="Show your open positions")
async def cmd_status(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return

        positions = run_raw_sql(
            "SELECT * FROM paper_portfolio "
            "WHERE telegram_id = %s AND status = 'OPEN' ORDER BY id",
            (str(tid),),
        )
        cap  = get_paper_capital(tid)
        curr = Decimal(str(cap.get("current_capital", "0")))

        if not positions:
            await interaction.followup.send(
                f"No open positions | {mode_label()}\nAvailable cash: {fmt_npr(curr)}",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f"Open Positions ({len(positions)}/{MAX_POSITIONS}) | {mode_label()}",
            colour=discord.Colour.blurple(),
            timestamp=datetime.now(NST),
        )
        for pos in positions:
            sym    = pos["symbol"]
            shares = pos.get("total_shares", "0")
            wacc   = pos.get("wacc", "0")
            cost   = pos.get("total_cost", "0")
            days   = hold_days(pos.get("first_buy_date") or "")
            buys   = pos.get("buy_count", "1")
            embed.add_field(
                name=f"📌 {sym}",
                value=(
                    f"Shares: {float(shares):.0f} | WACC: {fmt_npr(wacc)}\n"
                    f"Cost basis: {fmt_npr(cost)} | Held: {days}d | Avg: {buys}×"
                ),
                inline=False,
            )
        embed.set_footer(text=f"Available cash: {fmt_npr(curr)}")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception:
        log.exception("cmd_status failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /pnl
# ═══════════════════════════════════════════════════════════════════════════

@tree.command(guild=GUILD, name="pnl", description="Show closed trades and win rate")
async def cmd_pnl(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return

        cap    = get_paper_capital(tid)
        # Mirror telegram_bot.py's /pnl: queries paper_portfolio CLOSED rows
        trades = run_raw_sql(
            "SELECT * FROM paper_portfolio "
            "WHERE telegram_id = %s AND status = 'CLOSED' "
            "ORDER BY id DESC LIMIT 20",
            (str(tid),),
        )

        pnl_total  = cap.get("total_realised_pnl", "0")
        fees_total = cap.get("total_fees_paid", "0")
        cgt_total  = cap.get("total_cgt_paid", "0")

        colour = (
            discord.Colour.green()
            if float(pnl_total or 0) >= 0
            else discord.Colour.red()
        )
        embed = discord.Embed(
            title=f"Closed Trades | {mode_label()}",
            colour=colour,
            timestamp=datetime.now(NST),
        )
        embed.add_field(name="Win rate",     value=win_rate_str(cap),                  inline=True)
        embed.add_field(name="Realised P&L", value=fmt_npr(pnl_total,  sign=True),     inline=True)
        embed.add_field(name="Total fees",   value=fmt_npr(fees_total),                inline=True)
        embed.add_field(name="Total CGT",    value=fmt_npr(cgt_total),                 inline=True)

        if not trades:
            embed.description = "No closed trades yet."
        else:
            lines = []
            for t in trades:
                em  = "🟢" if t.get("result") == "WIN" else ("🔴" if t.get("result") == "LOSS" else "⚪")
                pnl = float(t.get("net_pnl") or 0)
                cgt = float(t.get("cgt_paid") or 0)
                lines.append(
                    f"{em} **{t['symbol']}** | {t.get('exit_date', '?')}\n"
                    f"  {float(t.get('exit_shares') or 0):.0f}sh @ "
                    f"{fmt_npr(t.get('exit_price', '0'))} | "
                    f"Net: {fmt_npr(pnl, sign=True)} | CGT: {fmt_npr(cgt)}"
                )
            # Discord embed field values are capped at 1024 chars; split if needed
            chunk = "\n".join(lines[:10])
            embed.add_field(name="Recent (last 20)", value=chunk or "—", inline=False)
            if len(lines) > 10:
                embed.add_field(
                    name="​",
                    value="\n".join(lines[10:]),
                    inline=False,
                )

        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception:
        log.exception("cmd_pnl failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /buy
# ═══════════════════════════════════════════════════════════════════════════

class BuyView(discord.ui.View):
    def __init__(self, caller_id: int, tid: str, symbol: str,
                 shares: Decimal, price: Decimal):
        super().__init__(timeout=60)
        self._caller_id = caller_id
        self._tid        = tid
        self._symbol     = symbol
        self._shares     = shares
        self._price      = price
        self._done       = False

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self._caller_id:
            await interaction.response.send_message(
                "Not your order.", ephemeral=True
            )
            return False
        return True

    def _disable_all(self):
        for item in self.children:
            item.disabled = True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._guard(interaction):
            return
        if self._done:
            await interaction.response.defer()
            return
        self._done = True
        self._disable_all()
        try:
            r = execute_buy(self._tid, self._symbol, self._shares, self._price)
            avg_txt = "_(averaged)_" if r["averaged"] else "_(new position)_"
            embed = discord.Embed(
                title=f"✅ BUY recorded {avg_txt}",
                colour=discord.Colour.green(),
                timestamp=datetime.now(NST),
            )
            embed.add_field(name="Symbol",    value=r["symbol"],             inline=True)
            embed.add_field(name="Shares",    value=f"{r['shares']:.0f}",    inline=True)
            embed.add_field(name="Price",     value=fmt_npr(r["price"]),     inline=True)
            embed.add_field(name="New WACC",  value=fmt_npr(r["new_wacc"]),  inline=True)
            embed.add_field(name="Capital remaining", value=fmt_npr(r["cap_after"]), inline=True)
            await interaction.response.edit_message(embed=embed, view=self)
        except ValueError as e:
            embed = discord.Embed(
                title="❌ BUY failed",
                description=str(e),
                colour=discord.Colour.red(),
            )
            await interaction.response.edit_message(embed=embed, view=self)
        except Exception:
            log.exception("execute_buy failed for user %s symbol %s", self._caller_id, self._symbol)
            embed = discord.Embed(
                title="❌ BUY failed",
                description="An internal error occurred. Please try again.",
                colour=discord.Colour.red(),
            )
            await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._guard(interaction):
            return
        if self._done:
            await interaction.response.defer()
            return
        self._done = True
        self._disable_all()
        embed = discord.Embed(title="❌ BUY cancelled", colour=discord.Colour.red())
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self):
        self._disable_all()
        # on_timeout has no interaction; the message is edited via the stored reference if available


@tree.command(guild=GUILD, name="buy", description="Open or average into a position")
@app_commands.describe(
    symbol="Stock symbol e.g. NABIL",
    shares="Number of shares",
    price="Price per share in NPR",
)
async def cmd_buy(interaction: discord.Interaction, symbol: str, shares: int, price: float):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return

        if _circuit_breakers.get(tid):
            await interaction.followup.send(
                "🔴 Circuit breaker ACTIVE. Use /resume first.",
                ephemeral=True,
            )
            return

        gate = market_gate_message()
        if gate:
            await interaction.followup.send(gate, ephemeral=True)
            return

        sym = symbol.upper()
        matches = lookup_symbols(sym)
        if not matches:
            await interaction.followup.send(
                f"❌ Symbol **{sym}** not found in NEPSE.",
                ephemeral=True,
            )
            return
        if matches[0].upper() != sym:
            suggestion = ", ".join(f"`{s}`" for s in matches)
            await interaction.followup.send(
                f"❌ Symbol **{sym}** not found. Did you mean: {suggestion}?",
                ephemeral=True,
            )
            return
        sym = matches[0]

        try:
            d_shares = Decimal(str(int(shares)))
            d_price  = Decimal(str(price))
        except (InvalidOperation, ValueError) as e:
            await interaction.followup.send(
                f"❌ Parse error: {e}", ephemeral=True
            )
            return

        fees      = calc_buy_fees(d_price, d_shares)
        total_out = fees["total_cost"]
        cap       = get_paper_capital(tid)
        available = Decimal(str(cap.get("current_capital", "0")))

        if total_out > available:
            await interaction.followup.send(
                f"❌ Insufficient capital.\n"
                f"Required:  {fmt_npr(total_out)}\n"
                f"Available: {fmt_npr(available)}",
                ephemeral=True,
            )
            return

        existing = get_paper_position(tid, sym)
        avg_note = ""
        if existing:
            old_shares = Decimal(str(existing["total_shares"]))
            old_cost   = Decimal(str(existing["total_cost"]))
            new_shares = old_shares + d_shares
            new_wacc   = ((old_cost + total_out) / new_shares).quantize(
                Decimal("0.0001"), rounding="ROUND_HALF_UP"
            )
            avg_note = (
                f"Averaging: hold {old_shares:.0f}sh @ WACC {fmt_npr(existing['wacc'])}\n"
                f"New WACC after: {fmt_npr(new_wacc)}"
            )

        embed = discord.Embed(
            title=f"Confirm BUY | {mode_label()}",
            colour=discord.Colour.blurple(),
            timestamp=datetime.now(NST),
        )
        embed.add_field(name="Symbol",      value=f"{sym}",                    inline=True)
        embed.add_field(name="Shares",      value=f"{d_shares:.0f}",           inline=True)
        embed.add_field(name="Price",       value=fmt_npr(d_price),            inline=True)
        embed.add_field(name="Gross",       value=fmt_npr(fees["gross"]),      inline=True)
        embed.add_field(name="Brokerage",   value=fmt_npr(fees["brokerage"]),  inline=True)
        embed.add_field(name="SEBON",       value=fmt_npr(fees["sebon"]),      inline=True)
        embed.add_field(name="DP fee",      value=fmt_npr(fees["dp_fee"]),     inline=True)
        embed.add_field(name="Total fees",  value=fmt_npr(fees["total_fees"]), inline=True)
        embed.add_field(name="Total out",   value=fmt_npr(total_out),          inline=True)
        embed.add_field(name="Available",   value=fmt_npr(available),          inline=True)
        embed.add_field(name="After buy",   value=fmt_npr(available - total_out), inline=True)
        if avg_note:
            embed.add_field(name="Averaging", value=avg_note, inline=False)
        embed.set_footer(text="Expires in 60 seconds.")

        view = BuyView(interaction.user.id, tid, sym, d_shares, d_price)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    except Exception:
        log.exception("cmd_buy failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ═══════════════════════════════════════════════════════════════════════════
# /sell
# ═══════════════════════════════════════════════════════════════════════════

class SellView(discord.ui.View):
    def __init__(self, caller_id: int, tid: str, symbol: str,
                 shares: Decimal, price: Decimal):
        super().__init__(timeout=60)
        self._caller_id = caller_id
        self._tid        = tid
        self._symbol     = symbol
        self._shares     = shares
        self._price      = price
        self._done       = False

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self._caller_id:
            await interaction.response.send_message(
                "Not your order.", ephemeral=True
            )
            return False
        return True

    def _disable_all(self):
        for item in self.children:
            item.disabled = True

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._guard(interaction):
            return
        if self._done:
            await interaction.response.defer()
            return
        self._done = True
        self._disable_all()
        try:
            r  = execute_sell(self._tid, self._symbol, self._shares, self._price)
            em = "🟢" if r["result"] == "WIN" else ("🔴" if r["result"] == "LOSS" else "⚪")
            embed = discord.Embed(
                title="✅ SELL recorded",
                colour=discord.Colour.green() if r["result"] == "WIN" else discord.Colour.red(),
                timestamp=datetime.now(NST),
            )
            embed.add_field(name="Symbol",   value=f"{em} {r['symbol']} — {r['result']}", inline=False)
            embed.add_field(name="Net P&L",  value=fmt_npr(r["fees"]["net_pnl"], sign=True), inline=True)
            embed.add_field(name="CGT paid", value=fmt_npr(r["fees"]["cgt"]),               inline=True)
            embed.add_field(name="Capital now", value=fmt_npr(r["cap_after"]),              inline=True)
            if r["remaining"] > 0:
                embed.add_field(
                    name="Remaining",
                    value=f"{r['remaining']:.0f} shares still open",
                    inline=False,
                )
            await interaction.response.edit_message(embed=embed, view=self)
        except ValueError as e:
            embed = discord.Embed(
                title="❌ SELL failed",
                description=str(e),
                colour=discord.Colour.red(),
            )
            await interaction.response.edit_message(embed=embed, view=self)
        except Exception:
            log.exception("execute_sell failed for user %s symbol %s", self._caller_id, self._symbol)
            embed = discord.Embed(
                title="❌ SELL failed",
                description="An internal error occurred. Please try again.",
                colour=discord.Colour.red(),
            )
            await interaction.response.edit_message(embed=embed, view=self)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._guard(interaction):
            return
        if self._done:
            await interaction.response.defer()
            return
        self._done = True
        self._disable_all()
        embed = discord.Embed(title="❌ SELL cancelled", colour=discord.Colour.red())
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self):
        self._disable_all()


@tree.command(guild=GUILD, name="sell", description="Close or partially close a position")
@app_commands.describe(
    symbol="Stock symbol e.g. NABIL",
    shares="Number of shares to sell",
    price="Price per share in NPR",
)
async def cmd_sell(interaction: discord.Interaction, symbol: str, shares: int, price: float):
    try:
        await interaction.response.defer(ephemeral=True)
        tid = await _resolve(interaction)
        if tid is None:
            return

        gate = market_gate_message()
        if gate:
            await interaction.followup.send(gate, ephemeral=True)
            return

        sym = symbol.upper()
        matches = lookup_symbols(sym)
        if not matches:
            await interaction.followup.send(
                f"❌ Symbol **{sym}** not found in NEPSE.",
                ephemeral=True,
            )
            return
        if matches[0].upper() != sym:
            suggestion = ", ".join(f"`{s}`" for s in matches)
            await interaction.followup.send(
                f"❌ Symbol **{sym}** not found. Did you mean: {suggestion}?",
                ephemeral=True,
            )
            return
        sym = matches[0]

        try:
            d_shares = Decimal(str(int(shares)))
            d_price  = Decimal(str(price))
        except (InvalidOperation, ValueError) as e:
            await interaction.followup.send(
                f"❌ Parse error: {e}", ephemeral=True
            )
            return

        pos = get_paper_position(tid, sym)
        if not pos:
            await interaction.followup.send(
                f"❌ No open position for **{sym}**. Use /status.",
                ephemeral=True,
            )
            return

        held = Decimal(str(pos["total_shares"]))
        if d_shares > held:
            await interaction.followup.send(
                f"❌ You hold {held:.0f} shares of {sym}. Cannot sell {d_shares:.0f}.",
                ephemeral=True,
            )
            return

        wacc      = Decimal(str(pos["wacc"]))
        held_days = hold_days(pos.get("first_buy_date") or "")
        fees      = calc_sell_fees(d_price, d_shares, wacc, held_days)
        remaining = held - d_shares
        em        = "🟢" if fees["net_pnl"] > 0 else "🔴"

        colour = discord.Colour.green() if fees["net_pnl"] >= 0 else discord.Colour.red()
        embed = discord.Embed(
            title=f"Confirm SELL | {mode_label()}",
            colour=colour,
            timestamp=datetime.now(NST),
        )
        embed.add_field(name="Symbol",      value=sym,                          inline=True)
        embed.add_field(name="Shares",      value=f"{d_shares:.0f}",            inline=True)
        embed.add_field(name="Price",       value=fmt_npr(d_price),             inline=True)
        embed.add_field(name="Entry WACC",  value=fmt_npr(wacc),                inline=True)
        embed.add_field(name="Gross sale",  value=fmt_npr(fees["gross"]),       inline=True)
        embed.add_field(name="Brokerage",   value=fmt_npr(fees["brokerage"]),   inline=True)
        embed.add_field(name="SEBON",       value=fmt_npr(fees["sebon"]),       inline=True)
        embed.add_field(name="DP fee",      value=fmt_npr(fees["dp_fee"]),      inline=True)
        embed.add_field(name="CGT (7.5%)",  value=fmt_npr(fees["cgt"]),         inline=True)
        embed.add_field(name="Total fees",  value=fmt_npr(fees["total_fees"]),  inline=True)
        embed.add_field(name=f"{em} Net P&L", value=fmt_npr(fees["net_pnl"], sign=True), inline=True)
        embed.add_field(name="Proceeds",    value=fmt_npr(fees["net_proceeds"]), inline=True)
        if remaining > 0:
            embed.add_field(
                name="Remaining",
                value=f"{remaining:.0f} shares remain at WACC {fmt_npr(wacc)}",
                inline=False,
            )
        embed.set_footer(text="Expires in 60 seconds.")

        view = SellView(interaction.user.id, tid, sym, d_shares, d_price)
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    except Exception:
        log.exception("cmd_sell failed for user %s", interaction.user.id)
        await _safe_error(interaction)


# ─── Shared error reply ───────────────────────────────────────────────────────

async def _safe_error(interaction: discord.Interaction):
    """Send a generic error reply without raising. Primary path is followup (defer always runs first)."""
    msg = "An internal error occurred. Please try again in a moment."
    try:
        await interaction.followup.send(msg, ephemeral=True)
    except Exception:
        try:
            await interaction.response.send_message(msg, ephemeral=True)
        except Exception:
            log.exception("_safe_error itself failed for interaction %s", interaction.id)


# ─── Entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Discord bot...")
    bot.run(DISCORD_BOT_TOKEN)
