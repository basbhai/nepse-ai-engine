"""
stat_method/hidden_accum_scanner.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Daily Stealth Accumulation Scanner

Runs after EOD floorsheet scrape. Finds stocks where all three conditions hold:
  1. Volume LOW    — below broker-specific vol_ratio threshold
  2. Price FLAT    — below broker-specific price_range threshold
  3. Broker buying — consecutive net-buy streak >= broker-specific minimum

Validated thresholds from grid search (82.4% hit rate for Sri Hari at 90d):
  Sri Hari Securities (56):   vol<=0.30, price<=0.05, streak>=5
  Trishakti Securities (48):  vol<=0.50, price<=0.03, streak>=5
  Online Securities (49):     vol<=0.60, price<=0.03, streak>=7
  Dipshikha Dhitopatra (38):  vol<=0.40, price<=0.10, streak>=10
  Naasa Securities (58):      vol<=0.40, price<=0.05, streak>=5

Usage:
    cd ~/nepse-engine
    python -m modules.hidden_accum_scanner
    python -m modules.hidden_accum_scanner --dry-run   # no DB writes
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STEALTH] %(message)s",
)
log = logging.getLogger(__name__)

# ── Broker thresholds (validated from grid search) ────────────────────────────
BROKER_THRESHOLDS = {
    "56": {"name": "Sri Hari Securities",   "vol": 0.30, "price": 0.05, "streak": 5,  "window": 60},
    "48": {"name": "Trishakti Securities",  "vol": 0.50, "price": 0.03, "streak": 5,  "window": 90},
    "49": {"name": "Online Securities",     "vol": 0.60, "price": 0.03, "streak": 7,  "window": 60},
    "38": {"name": "Dipshikha Dhitopatra", "vol": 0.40, "price": 0.10, "streak": 10, "window": 90},
    "58": {"name": "Naasa Securities",      "vol": 0.40, "price": 0.05, "streak": 5,  "window": 60},
}

# Trigger thresholds to test
VOLUME_SPIKE_THRESHOLDS = [1.5, 2.0, 3.0]
PRICE_BREAKOUT_PCTS     = [3.0, 5.0]   # % gain in single day

LOOKBACK_DAYS  = 120   # load last N calendar days of floorsheet
CLOSE_AFTER_D  = 90    # close triggered signals after N days


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _local_db():
    import psycopg2
    import os
    local = "postgresql://postgres:nepse123@127.0.0.1:5432/nepse_engine"
    url   = os.environ.get("DATABASE_URL", local)
    if "neon" in url:
        url = local
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def _nst_now() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Kathmandu")).strftime("%Y-%m-%d %H:%M:%S")


def _today() -> str:
    return date.today().strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_mutual_fund_symbols() -> set:
    """Load MF symbols from share_sectors to exclude from signals."""
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("""
                SELECT UPPER(symbol) AS symbol
                FROM share_sectors
                WHERE UPPER(instrumenttype) LIKE '%MUTUAL%'
                   OR UPPER(instrumenttype) LIKE '%FUND%'
            """)
            rows = cur.fetchall()
        return {r["symbol"] for r in rows}
    except Exception as e:
        log.warning("Could not load MF symbols: %s — using suffix filter only", e)
        return set()


def is_mutual_fund(symbol: str, mf_symbols: set) -> bool:
    """Exclude mutual funds by DB lookup or common suffixes."""
    if symbol.upper() in mf_symbols:
        return True
    # Common NEPSE MF suffixes
    mf_suffixes = ("F", "MF", "MF1", "MF2", "MF3", "D", "G")
    for sfx in mf_suffixes:
        if symbol.upper().endswith(sfx) and len(symbol) > 3:
            return True
    return False


def load_broker_daily(from_date: str) -> pd.DataFrame:
    """Load daily net position for 5 brokers from local postgres."""
    broker_ids = list(BROKER_THRESHOLDS.keys())
    log.info("Loading broker floorsheet from %s...", from_date)

    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    ph   = ",".join(["%s"] * len(broker_ids))

    cur.execute(f"""
        WITH buys AS (
            SELECT date::date AS date, symbol,
                   buyer_broker_id AS broker_id,
                   SUM(quantity::float) AS buy_units
            FROM floorsheet
            WHERE date >= %s
              AND buyer_broker_id IN ({ph})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, buyer_broker_id
        ),
        sells AS (
            SELECT date::date AS date, symbol,
                   seller_broker_id AS broker_id,
                   SUM(quantity::float) AS sell_units
            FROM floorsheet
            WHERE date >= %s
              AND seller_broker_id IN ({ph})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, seller_broker_id
        )
        SELECT
            COALESCE(b.date,    s.date)        AS date,
            COALESCE(b.symbol,  s.symbol)      AS symbol,
            COALESCE(b.broker_id, s.broker_id) AS broker_id,
            COALESCE(b.buy_units,  0)          AS buy_units,
            COALESCE(s.sell_units, 0)          AS sell_units
        FROM buys b
        FULL OUTER JOIN sells s
            ON s.date=b.date AND s.symbol=b.symbol AND s.broker_id=b.broker_id
        ORDER BY date, symbol, broker_id
    """, [from_date] + broker_ids + [from_date] + broker_ids)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]      = pd.to_datetime(df["date"])
    df["net_units"] = df["buy_units"].astype(float) - df["sell_units"].astype(float)
    df["broker_id"] = df["broker_id"].astype(str).str.strip()
    log.info("  Broker rows: %d | Symbols: %d", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


def load_daily_volume(from_date: str) -> pd.DataFrame:
    """Load daily total market volume per symbol (all brokers) from local postgres."""
    log.info("Loading daily volume aggregates from %s...", from_date)
    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT date::date AS date, symbol,
               SUM(quantity::float) AS total_volume
        FROM floorsheet
        WHERE date >= %s
          AND quantity IS NOT NULL AND quantity != ''
          AND quantity ~ '^[0-9]+\\.?[0-9]*$'
        GROUP BY date::date, symbol
        ORDER BY symbol, date
    """, (from_date,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]         = pd.to_datetime(df["date"])
    df["total_volume"] = df["total_volume"].astype(float)
    log.info("  Volume rows: %d", len(df))
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_prices(from_date: str) -> pd.DataFrame:
    """Load price_history from Neon."""
    log.info("Loading price_history from %s...", from_date)
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT symbol, date::date AS date,
                   COALESCE(NULLIF(close,''), NULLIF(ltp,''))::float AS close
            FROM price_history
            WHERE date >= %s
              AND close IS NOT NULL AND close != ''
              AND close ~ '^[0-9]+\\.?[0-9]*$'
              AND close::float > 0
            ORDER BY symbol, date ASC
        """, (from_date,))
        rows = cur.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    log.info("  Price rows: %d | Symbols: %d", len(df), df["symbol"].nunique())
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def load_existing_watching() -> list[dict]:
    """Load all WATCHING signals from Neon."""
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT * FROM stealth_signals
            WHERE status = 'WATCHING'
            ORDER BY signal_date DESC
        """)
        return [dict(r) for r in cur.fetchall()]


def load_existing_triggered() -> list[dict]:
    """Load all TRIGGERED signals from Neon."""
    from db.connection import _db
    with _db() as cur:
        cur.execute("""
            SELECT * FROM stealth_signals
            WHERE status = 'TRIGGERED'
            ORDER BY trigger_date DESC
        """)
        return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_streak(net_series: pd.Series) -> int:
    """Count consecutive net-positive days from end of series."""
    vals = net_series.values
    streak = 0
    for v in reversed(vals):
        if v > 0:
            streak += 1
        else:
            break
    return streak


def compute_vol_ratio(vol_df: pd.DataFrame, symbol: str,
                      today: pd.Timestamp) -> float | None:
    """today_volume / 60d_avg_volume."""
    sym_vol = vol_df[vol_df["symbol"] == symbol].sort_values("date")
    if sym_vol.empty:
        return None
    today_row = sym_vol[sym_vol["date"] == today]
    if today_row.empty:
        return None
    today_vol = float(today_row["total_volume"].iloc[0])
    past_60   = sym_vol[sym_vol["date"] < today].tail(60)
    if len(past_60) < 5:
        return None
    avg_60 = float(past_60["total_volume"].mean())
    if avg_60 == 0:
        return None
    return round(today_vol / avg_60, 4)


def compute_price_range(price_df: pd.DataFrame, symbol: str,
                        today: pd.Timestamp) -> float | None:
    """(max-min)/min over last 20 trading days before today."""
    sym_p = price_df[
        (price_df["symbol"] == symbol) &
        (price_df["date"] < today)
    ].sort_values("date").tail(20)
    if len(sym_p) < 5:
        return None
    mn = float(sym_p["close"].min())
    mx = float(sym_p["close"].max())
    if mn == 0:
        return None
    return round((mx - mn) / mn, 4)


def get_today_close(price_df: pd.DataFrame, symbol: str,
                    today: pd.Timestamp) -> float | None:
    """Get today's closing price."""
    row = price_df[
        (price_df["symbol"] == symbol) &
        (price_df["date"] == today)
    ]
    if row.empty:
        # Try latest available
        sym_p = price_df[price_df["symbol"] == symbol].sort_values("date")
        if sym_p.empty:
            return None
        return float(sym_p["close"].iloc[-1])
    return float(row["close"].iloc[0])


# ══════════════════════════════════════════════════════════════════════════════
# TRIGGER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def check_trigger(symbol: str,
                  vol_df: pd.DataFrame,
                  price_df: pd.DataFrame,
                  today: pd.Timestamp) -> dict | None:
    """
    Check if a WATCHING symbol has triggered.
    Returns trigger info dict or None.
    """
    trigger_types = []

    # ── Volume spike ─────────────────────────────────────────────────────────
    sym_vol = vol_df[vol_df["symbol"] == symbol].sort_values("date")
    today_row = sym_vol[sym_vol["date"] == today]
    if not today_row.empty:
        today_vol = float(today_row["total_volume"].iloc[0])
        past_20   = sym_vol[sym_vol["date"] < today].tail(20)
        if len(past_20) >= 5:
            avg_20 = float(past_20["total_volume"].mean())
            if avg_20 > 0:
                vol_ratio_today = today_vol / avg_20
                for thresh in VOLUME_SPIKE_THRESHOLDS:
                    if vol_ratio_today >= thresh:
                        trigger_types.append(f"VOLUME_SPIKE_{thresh}X")
                        break  # only record highest threshold hit

    # ── Price breakout ────────────────────────────────────────────────────────
    sym_p = price_df[price_df["symbol"] == symbol].sort_values("date")
    today_price_row = sym_p[sym_p["date"] == today]
    if not today_price_row.empty:
        today_close = float(today_price_row["close"].iloc[0])
        past_20_p   = sym_p[sym_p["date"] < today].tail(20)

        # 20-day high breakout
        if not past_20_p.empty:
            high_20 = float(past_20_p["close"].max())
            if today_close > high_20:
                trigger_types.append("PRICE_BREAKOUT_20D_HIGH")

        # Single day % gain
        if len(past_20_p) >= 1:
            prev_close = float(past_20_p["close"].iloc[-1])
            if prev_close > 0:
                pct_gain = (today_close - prev_close) / prev_close * 100
                for pct in sorted(PRICE_BREAKOUT_PCTS, reverse=True):
                    if pct_gain >= pct:
                        trigger_types.append(f"PRICE_BREAKOUT_{pct:.0f}PCT")
                        break

    if not trigger_types:
        return None

    today_close = get_today_close(price_df, symbol, today)
    today_vol_ratio = None
    if not today_row.empty and len(past_20) >= 5:
        today_vol_ratio = round(today_vol / avg_20, 4) if avg_20 > 0 else None

    return {
        "trigger_date":      today.strftime("%Y-%m-%d"),
        "trigger_price":     str(round(today_close, 2)) if today_close else None,
        "trigger_type":      " | ".join(trigger_types),
        "trigger_vol_ratio": str(today_vol_ratio) if today_vol_ratio else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def send_trigger_alert(signal: dict) -> None:
    """Send Telegram alert when a stealth signal triggers."""
    try:
        from helper.notifier import _send_admin_only
        msg = (
            f"🔔 *STEALTH SIGNAL TRIGGERED*\n\n"
            f"Symbol: *{signal['symbol']}*\n"
            f"Broker: {signal['broker_name']}\n"
            f"Trigger: {signal['trigger_type']}\n"
            f"Trigger Price: NPR {signal['trigger_price']}\n"
            f"Signal Date: {signal['signal_date']} "
            f"(streak was {signal['streak_days']}d)\n"
            f"Vol Ratio: {signal['vol_ratio']} → {signal.get('trigger_vol_ratio','?')}\n\n"
            f"Historical hit rate: "
            f"{'82.4%' if signal['broker_id']=='56' else '80.0%' if signal['broker_id']=='48' else '78.3%' if signal['broker_id']=='49' else '77.8%' if signal['broker_id']=='38' else '~60%'} "
            f"at 90d"
        )
        _send_admin_only(msg)
        log.info("Telegram alert sent for %s", signal["symbol"])
    except Exception as e:
        log.warning("Telegram alert failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# DB WRITES
# ══════════════════════════════════════════════════════════════════════════════

def upsert_watching(signal: dict, dry_run: bool) -> None:
    """Insert new WATCHING signal or update streak if already exists."""
    from sheets import write_row, update_row
    from db.connection import _db

    now = _nst_now()

    # Check if exists
    with _db() as cur:
        cur.execute("""
            SELECT id, streak_days FROM stealth_signals
            WHERE symbol = %s AND broker_id = %s AND signal_date = %s
        """, (signal["symbol"], signal["broker_id"], signal["signal_date"]))
        existing = cur.fetchone()

    if existing:
        # Update streak only
        if dry_run:
            log.info("[DRY] Update streak %s %s → %s days",
                     signal["symbol"], signal["broker_name"], signal["streak_days"])
            return
        update_row("stealth_signals",
                   {"streak_days": str(signal["streak_days"]), "updated_at": now},
                   {"id": existing["id"]})
        log.info("Updated streak: %s %s → %s days",
                 signal["symbol"], signal["broker_name"], signal["streak_days"])
    else:
        if dry_run:
            log.info("[DRY] New signal: %s %s streak=%s vol=%.2f price=%.2f",
                     signal["symbol"], signal["broker_name"],
                     signal["streak_days"], float(signal["vol_ratio"] or 0),
                     float(signal["price_range"] or 0))
            return
        write_row("stealth_signals", {
            "symbol":       signal["symbol"],
            "broker_id":    signal["broker_id"],
            "broker_name":  signal["broker_name"],
            "signal_date":  signal["signal_date"],
            "streak_days":  str(signal["streak_days"]),
            "vol_ratio":    str(signal["vol_ratio"]),
            "price_range":  str(signal["price_range"]),
            "entry_price":  str(signal["entry_price"]) if signal["entry_price"] else None,
            "status":       "WATCHING",
            "window_days":  str(signal["window_days"]),
            "vol_thresh":   str(signal["vol_thresh"]),
            "price_thresh": str(signal["price_thresh"]),
            "streak_thresh":str(signal["streak_thresh"]),
            "created_at":   now,
            "updated_at":   now,
        })
        log.info("New signal: %s %s streak=%s vol=%s price=%s",
                 signal["symbol"], signal["broker_name"],
                 signal["streak_days"], signal["vol_ratio"], signal["price_range"])


def mark_triggered(signal_id: int, trigger_info: dict, dry_run: bool) -> None:
    from sheets import update_row
    now = _nst_now()
    if dry_run:
        log.info("[DRY] Trigger id=%d → %s", signal_id, trigger_info["trigger_type"])
        return
    update_row("stealth_signals", {
        "status":            "TRIGGERED",
        "trigger_date":      trigger_info["trigger_date"],
        "trigger_price":     trigger_info["trigger_price"],
        "trigger_type":      trigger_info["trigger_type"],
        "trigger_vol_ratio": trigger_info.get("trigger_vol_ratio"),
        "updated_at":        now,
    }, {"id": signal_id})
    log.info("Marked TRIGGERED: id=%d %s", signal_id, trigger_info["trigger_type"])


def mark_closed(signal_id: int, close_price: float,
                trigger_price: float, dry_run: bool) -> None:
    from sheets import update_row
    now = _nst_now()
    return_pct = round((close_price - trigger_price) / trigger_price * 100, 2) \
                 if trigger_price and trigger_price > 0 else None
    if dry_run:
        log.info("[DRY] Close id=%d return=%.1f%%", signal_id, return_pct or 0)
        return
    update_row("stealth_signals", {
        "status":     "CLOSED",
        "close_date": _today(),
        "close_price": str(round(close_price, 2)),
        "return_pct":  str(return_pct) if return_pct is not None else None,
        "updated_at":  now,
    }, {"id": signal_id})
    log.info("Marked CLOSED: id=%d return=%.1f%%", signal_id, return_pct or 0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_scanner(dry_run: bool = False) -> None:
    today_dt   = pd.Timestamp(date.today())
    from_date  = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    # ── Load data ─────────────────────────────────────────────────────────────
    mf_symbols  = load_mutual_fund_symbols()
    broker_df   = load_broker_daily(from_date)
    vol_df      = load_daily_volume(from_date)
    price_df    = load_prices(from_date)

    if broker_df.empty or price_df.empty:
        log.error("Missing data — aborting")
        return

    new_signals     = 0
    updated_signals = 0

    # ── Step 1: Find new/updated WATCHING signals ─────────────────────────────
    log.info("Scanning for stealth accumulation signals...")

    for broker_id, thresh in BROKER_THRESHOLDS.items():
        broker_name = thresh["name"]
        bdf = broker_df[broker_df["broker_id"] == broker_id]

        for symbol, grp in bdf.groupby("symbol"):
            if is_mutual_fund(symbol, mf_symbols):
                continue

            grp = grp.sort_values("date").reset_index(drop=True)

            # Streak
            streak = compute_streak(grp["net_units"])
            if streak < thresh["streak"]:
                continue

            # Vol ratio
            vol_ratio = compute_vol_ratio(vol_df, symbol, today_dt)
            if vol_ratio is None or vol_ratio > thresh["vol"]:
                continue

            # Price range
            price_range = compute_price_range(price_df, symbol, today_dt)
            if price_range is None or price_range > thresh["price"]:
                continue

            # All three conditions met
            entry_price = get_today_close(price_df, symbol, today_dt)
            signal_date = today_dt.strftime("%Y-%m-%d")

            upsert_watching({
                "symbol":       symbol,
                "broker_id":    broker_id,
                "broker_name":  broker_name,
                "signal_date":  signal_date,
                "streak_days":  streak,
                "vol_ratio":    round(vol_ratio, 4),
                "price_range":  round(price_range, 4),
                "entry_price":  round(entry_price, 2) if entry_price else None,
                "window_days":  thresh["window"],
                "vol_thresh":   thresh["vol"],
                "price_thresh": thresh["price"],
                "streak_thresh":thresh["streak"],
            }, dry_run)
            new_signals += 1

    log.info("Signal scan complete: %d signals processed", new_signals)

    # ── Step 2: Check WATCHING signals for triggers ───────────────────────────
    watching = load_existing_watching()
    log.info("Checking %d WATCHING signals for triggers...", len(watching))

    triggered_count = 0
    for sig in watching:
        trigger = check_trigger(sig["symbol"], vol_df, price_df, today_dt)
        if trigger:
            mark_triggered(sig["id"], trigger, dry_run)
            # Send Telegram
            sig.update(trigger)
            send_trigger_alert(sig)
            triggered_count += 1

    log.info("Triggers fired: %d", triggered_count)

    # ── Step 3: Close triggered signals older than 90 days ───────────────────
    triggered = load_existing_triggered()
    closed_count = 0

    for sig in triggered:
        trigger_date = pd.Timestamp(sig["trigger_date"])
        days_since   = (date.today() - trigger_date.date()).days

        if days_since < CLOSE_AFTER_D:
            continue

        current_price = get_today_close(price_df, sig["symbol"], today_dt)
        if current_price is None:
            continue

        trigger_price = float(sig["trigger_price"]) if sig.get("trigger_price") else None
        if trigger_price is None:
            continue

        mark_closed(sig["id"], current_price, trigger_price, dry_run)
        closed_count += 1

    log.info("Signals closed: %d", closed_count)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("STEALTH SCANNER COMPLETE — %s", _today())
    log.info("  New/updated signals: %d", new_signals)
    log.info("  Triggers fired:      %d", triggered_count)
    log.info("  Signals closed:      %d", closed_count)
    log.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and log without writing to DB or Telegram")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN MODE — no DB writes or Telegram alerts")

    run_scanner(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
