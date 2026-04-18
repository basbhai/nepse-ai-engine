"""
main.py — NEPSE AI Engine
─────────────────────────────────────────────────────────────────────────────
Single trading loop orchestrator. Called every 6 minutes by systemd timer
during market hours (10:45 AM – 3:00 PM NST, Sun–Thu).

Usage:
    python main.py            → paper trading (default, safe)
    python main.py -paper     → paper trading (explicit)
    python main.py -live      → live trading (CAUTION: sets PAPER_MODE=false)
    python main.py --dry-run  → full pipeline, zero DB writes, zero API calls
    python main.py --skip-guard → bypass calendar_guard (for manual testing)

Paper mode differences vs live:
    - No circuit breaker check
    - No geo block check
    - BUY signals auto-written to paper_portfolio
    - All log lines prefixed [PAPER]

Live mode:
    - Circuit breaker enforced (loss_streak > 7 → halt)
    - Geo block enforced (combined_geo < -3 → halt)
    - BUY signals sent to notifier → Telegram
    - All log lines prefixed [LIVE] ⚠️

Architecture:
    main.py owns the pipeline order. Modules receive data — they do NOT call
    each other. This replaces the old pattern where claude_analyst called
    gemini_filter which called filter_engine.

Pipeline:
    calendar_guard → scraper → geo_sentiment → nepal_pulse (30-min throttle)
    → filter_engine → gemini_filter → claude_analyst
    → [paper: paper_portfolio write] [live: notifier]
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

NST = timezone(timedelta(hours=5, minutes=45))

# ── Nepal pulse throttle: only re-run if last update > 30 min ago ─────────────
NEPAL_PULSE_INTERVAL_MIN = 30


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(
        description="NEPSE AI Engine — trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              paper trading (default)
  python main.py -paper       paper trading (explicit)
  python main.py -live        live trading  (CAUTION)
  python main.py --dry-run    simulate full run, no writes
  python main.py --skip-guard ignore calendar_guard (testing only)
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-paper", action="store_true", help="Paper trading mode (default)")
    mode.add_argument("-live",  action="store_true", help="Live trading mode — real signals")
    parser.add_argument("--dry-run",    action="store_true", help="No DB writes, no API calls")
    parser.add_argument("--skip-guard", action="store_true", help="Bypass calendar_guard check")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_mode(args) -> bool:
    """
    Determine paper_mode boolean.
    Priority: CLI flag > DB setting > default (paper=True).
    Returns True if paper mode, False if live.
    """
    if args.live:
        return False
    if args.paper:
        return True
    # Read from DB if no CLI flag given
    try:
        from sheets import get_setting
        db_val = get_setting("PAPER_MODE", "true").lower()
        return db_val == "true"
    except Exception as e:
        log.warning("Could not read PAPER_MODE from DB: %s — defaulting to paper", e)
        return True


def _set_mode_in_db(paper_mode: bool, dry_run: bool) -> None:
    """Persist current mode to settings table so all modules agree."""
    if dry_run:
        return
    try:
        from sheets import update_setting
        update_setting("PAPER_MODE", "true" if paper_mode else "false")
    except Exception as e:
        log.warning("Could not write PAPER_MODE to DB: %s", e)


def _mode_label(paper_mode: bool) -> str:
    return "[PAPER]" if paper_mode else "[LIVE] ⚠️"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CIRCUIT BREAKER (live only)
# ══════════════════════════════════════════════════════════════════════════════

def _check_circuit_breaker() -> bool:
    """
    Returns True (blocked) if loss_streak > 7.
    Only called in live mode. Paper mode always returns False.
    """
    try:
        from sheets import get_setting, read_tab
        # Check explicit circuit breaker flag first
        cb = get_setting("CIRCUIT_BREAKER", "").lower()
        if cb == "true":
            log.warning("CIRCUIT BREAKER ACTIVE (CIRCUIT_BREAKER=true in settings)")
            return True
        # Also check live loss streak from financials table
        rows = read_tab("financials")
        kpis = {r.get("kpi_name", ""): r.get("current_value", "0") for r in rows}
        streak = int(kpis.get("current_loss_streak", "0") or "0")
        if streak > 7:
            log.warning("CIRCUIT BREAKER: loss_streak=%d > 7 — halting", streak)
            return True
        return False
    except Exception as e:
        log.error("Circuit breaker check failed: %s — allowing run", e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GEO BLOCK (live only)
# ══════════════════════════════════════════════════════════════════════════════

def _check_geo_block() -> tuple[bool, int]:
    """
    Returns (blocked, combined_geo_score).
    Blocks if combined_geo < -3. Live mode only.
    """
    try:
        from sheets import get_latest_geo, get_latest_pulse
        geo_row    = get_latest_geo()
        pulse_row  = get_latest_pulse()
        geo_score  = int(geo_row.get("geo_score",   0) if geo_row   else 0)
        nepal_score= int(pulse_row.get("nepal_score",0) if pulse_row else 0)
        combined   = geo_score + nepal_score
        if combined < -3:
            log.warning("GEO BLOCK: combined_geo=%+d < -3 — halting", combined)
            return True, combined
        return False, combined
    except Exception as e:
        log.error("Geo block check failed: %s — allowing run", e)
        return False, 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NEPAL PULSE THROTTLE
# ══════════════════════════════════════════════════════════════════════════════

def _should_run_nepal_pulse() -> bool:
    """
    Returns True if nepal_pulse should run this cycle.
    Only runs if last run was > 30 minutes ago (token-saving throttle).
    """
    try:
        from sheets import get_latest_pulse
        row = get_latest_pulse()
        if not row:
            return True
        ts_str = row.get("timestamp") or row.get("date", "")
        if not ts_str:
            return True
        # Try to parse timestamp
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                ts = datetime.strptime(ts_str[:19], fmt).replace(tzinfo=NST)
                age_min = (datetime.now(tz=NST) - ts).total_seconds() / 60
                return age_min >= NEPAL_PULSE_INTERVAL_MIN
            except ValueError:
                continue
        return True
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUY → PAPER PORTFOLIO AUTO-WRITE
# ══════════════════════════════════════════════════════════════════════════════

def _write_buy_to_paper_portfolio(result, dry_run: bool) -> bool:
    """
    When Claude produces a BUY in paper mode, auto-insert into paper_portfolio.
    This is the missing step identified in opus handoff (Section 3.6).

    Uses TELEGRAM_CHAT_ID as the paper trading user for system-generated trades.
    These are AI-suggested entries — the user still decides via telegram_bot.
    So we write a PENDING row that telegram_bot can confirm.

    Actually: we write to market_log as BUY (already done by claude_analyst).
    Paper portfolio writes happen via telegram_bot /buy command.
    Here we just send the signal to Telegram so the user can act.
    """
    if dry_run:
        log.info("[DRY-RUN] Would notify BUY: %s", result.symbol)
        return True
    try:
        from helper.notifier import send_buy_signal
        send_buy_signal(result)
        return True
    except Exception as e:
        log.error("send_buy_signal failed for %s: %s", result.symbol, e)
        return False


def _write_buy_to_live_portfolio(result, dry_run: bool) -> bool:
    """
    Live mode: notify only. User executes manually via broker.
    We never auto-execute orders — user always has final control.
    """
    if dry_run:
        log.info("[DRY-RUN] Would send live BUY alert: %s", result.symbol)
        return True
    try:
        from helper.notifier import send_buy_signal
        send_buy_signal(result)
        return True
    except Exception as e:
        log.error("Live send_buy_signal failed for %s: %s", result.symbol, e)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def _write_near_misses_to_db(near_misses: list, date_str: str, dry_run: bool) -> None:
    """
    Upsert near-misses to gate_misses table.
    ON CONFLICT (symbol, date) → skip. First block of day wins.
    6-min loop runs 28x/day — upsert prevents duplicates.
    """
    if dry_run or not near_misses:
        return
    try:
        from sheets import upsert_row
        from datetime import datetime, timedelta, timezone
        nst = timezone(timedelta(hours=5, minutes=45))

        for nm in near_misses:
            upsert_row("gate_misses", {
                "date":                     date_str,
                "symbol":                   nm.symbol,
                "sector":                   nm.sector,
                "gate_reason":              nm.gate_reason,
                "gate_category":            nm.gate_category,
                "price_at_block":           str(nm.price_at_block),
                "market_state":             nm.market_state,
                "tech_score":               str(nm.tech_score),
                "conf_score":               str(nm.conf_score),
                "composite_score_would_be": str(nm.composite_score_would_be),
                "volume_os_ratio":          str(nm.volume_os_ratio) if hasattr(nm, "volume_os_ratio") else "0",
                "tracking_days":            "0",

            }, conflict_columns=["symbol", "date"])
    except Exception as e:
        log.warning("_write_near_misses_to_db failed (non-fatal): %s", e)

def run_trading_loop(paper_mode: bool, dry_run: bool, skip_guard: bool) -> int:
    """
    Full trading pipeline. Returns exit code (0=ok, 1=blocked, 2=error).
    """
    label    = _mode_label(paper_mode)
    now_nst  = datetime.now(tz=NST)
    date_str = now_nst.strftime("%Y-%m-%d")

    log.info("=" * 65)
    log.info("%s NEPSE AI Engine — %s NST", label, now_nst.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    # ── Step 1: Calendar guard ────────────────────────────────────────────────
    if not skip_guard:
        try:
            from calendar_guard import is_open, get_status
            guard = get_status()
            log.info("Calendar: %s", guard["message"])
            if not guard["is_open"]:
                log.info("%s Market closed — exiting cleanly", label)
                return 1
        except Exception as e:
            log.error("calendar_guard failed: %s — aborting for safety", e)
            return 2
    else:
        log.warning("%s --skip-guard active — bypassing calendar check", label)

    # ── Step 2: Sync mode to DB ───────────────────────────────────────────────
    _set_mode_in_db(paper_mode, dry_run)
    log.info("%s Mode: %s | dry_run=%s", label,
             "PAPER" if paper_mode else "LIVE", dry_run)

    # ── Step 3: Live-only gates ───────────────────────────────────────────────
    if not paper_mode:
        blocked = _check_circuit_breaker()
        if blocked:
            try:
                from helper.notifier import send_error_alert
                send_error_alert("🚨 Circuit breaker active — trading halted")
            except Exception:
                pass
            return 1

        geo_blocked, combined_geo = _check_geo_block()
        if geo_blocked:
            try:
                from helper.notifier import send_error_alert
                send_error_alert(
                    f"⚠️ Geo block active — combined_geo={combined_geo:+d} < -3"
                )
            except Exception:
                pass
            return 1
        log.info("%s Gates passed — geo=%+d", label, combined_geo)
    else:
        log.info("%s Paper mode — circuit breaker + geo block skipped", label)

    # ── Step 4: Scraper — live prices ─────────────────────────────────────────
    log.info("%s [1/5] Scraper — fetching live prices...", label)
    market_data = {}
    try:
        from modules.scraper import get_all_market_data
        market_data = get_all_market_data(write_breadth=not dry_run)
        log.info("%s Scraper: %d symbols", label, len(market_data))
    except Exception as e:
        log.error("%s Scraper failed: %s", label, e)
        # Non-fatal — filter_engine will try again internally
        market_data = {}

    if not market_data:
        log.warning("%s No market data — cannot run filter. Exiting.", label)
        return 2

    # ── Step 5: Geo sentiment update ─────────────────────────────────────────
    log.info("%s [2/5] Geo sentiment update...", label)
    try:
        from modules.geo_sentiment import run as run_geo
        if not dry_run:
            run_geo()
        else:
            log.info("[DRY-RUN] Skipping geo_sentiment write")
    except Exception as e:
        log.warning("%s geo_sentiment failed (non-fatal): %s", label, e)

    # ── Step 6: Nepal pulse (30-min throttle) ─────────────────────────────────
    if _should_run_nepal_pulse():
        log.info("%s [3/5] Nepal pulse (due) — running...", label)
        try:
            from modules.nepal_pulse import run as run_nepal
            if not dry_run:
                run_nepal()
            else:
                log.info("[DRY-RUN] Skipping nepal_pulse write")
        except Exception as e:
            log.warning("%s nepal_pulse failed (non-fatal): %s", label, e)
    else:
        log.info("%s [3/5] Nepal pulse — throttled (< %d min since last run)",
                 label, NEPAL_PULSE_INTERVAL_MIN)

    # ── Step 7: Filter engine ─────────────────────────────────────────────────
    log.info("%s [4/5] Filter engine — ranking candidates...", label)
    candidates = []
    try:
        from filter_engine import run_filter
        candidates = run_filter(market_data=market_data, top_n=10, date=date_str)
        log.info("%s Filter: %d candidates", label, len(candidates))
            # Write near-misses (non-fatal — never block the trading loop)
        try:
            from filter_engine import get_last_near_misses
            _write_near_misses_to_db(get_last_near_misses(), date_str, dry_run)
            log.debug("%s Near-misses logged: %d", label, len(get_last_near_misses()))
        except Exception as e:
            log.warning("%s Near-miss write failed (non-fatal): %s", label, e)

        if candidates:
            for i, c in enumerate(candidates[:5], 1):
                log.info("%s   #%d %s score=%.1f rsi=%.1f %s",
                         label, i, c.symbol, c.composite_score,
                         c.rsi_14, c.primary_signal)
    except Exception as e:
        log.error("%s filter_engine failed: %s", label, e)
        return 2

    if not candidates:
        log.info("%s No candidates passed filter gates — done for this cycle", label)
        return 0

    # ── Step 8: Gemini filter ─────────────────────────────────────────────────
    log.info("%s [5/5] Gemini filter — screening candidates...", label)
    flags = []
    try:
        from gemini_filter import run_gemini_filter
        if not dry_run:
            flags = run_gemini_filter(
                candidates=candidates,
                market_data=market_data,
                date=date_str,
            )
        else:
            log.info("[DRY-RUN] Skipping Gemini call — would screen %d candidates",
                     len(candidates))
        log.info("%s Gemini: %d flag(s)", label, len(flags))
    except Exception as e:
        log.error("%s gemini_filter failed: %s", label, e)
        # Non-fatal: fall through with no flags

    if not flags:
        log.info("%s No flags from Gemini — nothing for Claude this cycle", label)
        return 0

    # ── Step 9: Claude analyst ────────────────────────────────────────────────
    log.info("%s Claude analyst — deep analysis on %d flag(s)...", label, len(flags))
    results = []
    try:
        from claude_analyst import run_analysis
        if not dry_run:
            results = run_analysis(flags)
        else:
            log.info("[DRY-RUN] Skipping Claude calls — would analyze: %s",
                     ", ".join(f.symbol for f in flags))
        log.info("%s Claude: %d result(s)", label, len(results))
    except Exception as e:
        log.error("%s claude_analyst failed: %s", label, e)
        return 2

    # ── Step 10: Handle results ───────────────────────────────────────────────
    buy_count  = 0
    wait_count = 0
    avoid_count= 0

    for result in results:
        action = getattr(result, "action", "").upper()
        sym    = getattr(result, "symbol", "?")

        if action == "BUY":
            buy_count += 1
            log.info("%s 🟢 BUY: %s | entry=%.2f stop=%.2f target=%.2f alloc=NPR %.0f",
                     label,
                     sym,
                     float(getattr(result, "entry_price",  0) or 0),
                     float(getattr(result, "stop_loss",    0) or 0),
                     float(getattr(result, "target_price", 0) or 0),
                     float(getattr(result, "allocation_npr", 0) or 0),
            )
            if paper_mode:
                _write_buy_to_paper_portfolio(result, dry_run)
            else:
                _write_buy_to_live_portfolio(result, dry_run)

        elif action == "WAIT":
            wait_count += 1
            log.info("%s 🟡 WAIT: %s", label, sym)

        elif action == "AVOID":
            avoid_count += 1
            log.info("%s 🔴 AVOID: %s", label, sym)

        else:
            log.info("%s ❓ %s: %s", label, action, sym)

    # ── Step 11: Summary ──────────────────────────────────────────────────────
    log.info("─" * 65)
    log.info("%s Cycle complete — BUY:%d WAIT:%d AVOID:%d",
             label, buy_count, wait_count, avoid_count)
    log.info("%s Finished at %s NST",
             label, datetime.now(tz=NST).strftime("%H:%M:%S"))
    log.info("=" * 65)

    return 0


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from log_config import attach_file_handler
    attach_file_handler(__name__)
    args       = _parse_args()
    paper_mode = _resolve_mode(args)

    try:
        exit_code = run_trading_loop(
            paper_mode  = paper_mode,
            dry_run     = args.dry_run,
            skip_guard  = args.skip_guard,
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log.critical("Unhandled exception in main: %s", e, exc_info=True)
        try:
            from helper.notifier import send_error_alert
            send_error_alert(f"🔥 main.py crashed: {e}")
        except Exception:
            pass
        sys.exit(2)
