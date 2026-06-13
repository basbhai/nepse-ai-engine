"""
smart_broker_scanner.py
────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Smart Broker Scanner

Monitors 5 specific brokers empirically proven to precede large moves:

  BUY SIGNALS (when these become dominant net buyers in a stock):
    38 — Dipshikha Dhitopatra  avg gain 474% (JBLBP +616%, BJHL +333%)
    56 — Sri Hari Securities    avg gain 456% (RSML +1321%, AHL +25%)
    49 — Online Securities      buyer signal
    48 — Trishakti Securities   avg gain 121% (SKHL +289%, JHAPA +43%)

  SELL SIGNAL (Naasa selling into a rally = bullish confirmation):
    58 — Naasa Securities       top seller in 14 stocks avg +212%

Signal fires when:
  BUY  — broker has been net-positive for MIN_STREAK consecutive days
          AND their net position is unusually large vs their own history
          AND the stock price is still calm (< CALM_PCT gain in last 15d)

  SELL (Naasa) — Naasa is top net seller in a stock for MIN_STREAK days
                 AND another smart broker (38/56/49/48) is also net buying
                 Combined = highest conviction signal

Output: Telegram alert + CSV log

Usage:
    cd ~/nepse-engine
    python stat_method/smart_broker_scanner.py
    python stat_method/smart_broker_scanner.py --days 10   # longer streak
    python stat_method/smart_broker_scanner.py --dry-run
"""

import sys
import csv
import logging
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SMART] %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent / "output"

# ── Tracked brokers ───────────────────────────────────────────────────────────
BUY_BROKERS = {
    "38": "Dipshikha Dhitopatra",
    "56": "Sri Hari Securities",
    "49": "Online Securities",
    "48": "Trishakti Securities",
}
NAASA_ID   = "58"
NAASA_NAME = "Naasa Securities"

# ── Config ────────────────────────────────────────────────────────────────────
MIN_STREAK    = 5      # consecutive net-buy days to trigger
MIN_NET_UNITS = 1000   # minimum net units in streak
CALM_PCT      = 0.15   # price must be < 15% above 15d-ago close
LOOKBACK_DAYS = 180    # days of history for baseline comparison
DOMINANT_MULT = 2.0    # current streak avg > 2x historical avg = dominant
FS_START      = "2023-07-01"


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


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_broker_daily(broker_ids: list[str],
                      from_date: str) -> pd.DataFrame:
    """
    Load daily net position for specific broker IDs across all symbols.
    Returns: DataFrame[date, symbol, broker_id, buy_units, sell_units, net_units]
    """
    log.info("Loading floorsheet for brokers %s from %s...",
             broker_ids, from_date)

    import psycopg2.extras
    conn = _local_db()
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    placeholders = ",".join(["%s"] * len(broker_ids))

    cur.execute(f"""
        WITH buys AS (
            SELECT date::date AS date, symbol,
                   buyer_broker_id AS broker_id,
                   buyer_broker    AS broker_name,
                   SUM(quantity::float) AS buy_units
            FROM floorsheet
            WHERE date >= %s
              AND buyer_broker_id IN ({placeholders})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, buyer_broker_id, buyer_broker
        ),
        sells AS (
            SELECT date::date AS date, symbol,
                   seller_broker_id AS broker_id,
                   SUM(quantity::float) AS sell_units
            FROM floorsheet
            WHERE date >= %s
              AND seller_broker_id IN ({placeholders})
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
            GROUP BY date::date, symbol, seller_broker_id
        )
        SELECT
            COALESCE(b.date,   s.date)        AS date,
            COALESCE(b.symbol, s.symbol)      AS symbol,
            COALESCE(b.broker_id, s.broker_id) AS broker_id,
            b.broker_name,
            COALESCE(b.buy_units,  0)         AS buy_units,
            COALESCE(s.sell_units, 0)         AS sell_units
        FROM buys b
        FULL OUTER JOIN sells s
            ON  s.date      = b.date
            AND s.symbol    = b.symbol
            AND s.broker_id = b.broker_id
        ORDER BY date, symbol, broker_id
    """, [from_date] + broker_ids + [from_date] + broker_ids)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    log.info("  Loaded %d rows", len(rows))
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["date"]       = pd.to_datetime(df["date"])
    df["net_units"]  = df["buy_units"] - df["sell_units"]
    df["broker_id"]  = df["broker_id"].astype(str).str.strip()
    df["buy_units"]  = df["buy_units"].fillna(0)
    df["sell_units"] = df["sell_units"].fillna(0)
    return df.sort_values(["symbol", "broker_id", "date"]).reset_index(drop=True)


def load_price_recent(from_date: str) -> pd.DataFrame:
    """Load price_history for calm price check."""
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
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_current_streak(series: pd.Series) -> tuple[int, float]:
    """
    Find length and total of current consecutive positive/negative run.
    Returns (streak_days, total_net).
    """
    vals = series.values
    if len(vals) == 0:
        return 0, 0.0

    # Walk backwards from end
    streak = 0
    total  = 0.0
    sign   = 1 if vals[-1] > 0 else -1

    for v in reversed(vals):
        if (sign == 1 and v > 0) or (sign == -1 and v < 0):
            streak += 1
            total  += v
        else:
            break

    return streak, total


def is_price_calm(price_df: pd.DataFrame, symbol: str,
                  as_of: pd.Timestamp) -> tuple[bool, float]:
    """Check if price is still calm (<15% gain in last 15 trading days)."""
    sym_p = price_df[price_df["symbol"] == symbol].copy()
    sym_p = sym_p[sym_p["date"] <= as_of].sort_values("date")

    if len(sym_p) < 16:
        return True, 0.0  # not enough data, assume calm

    close_now = sym_p["close"].iloc[-1]
    close_15d = sym_p["close"].iloc[-16]

    if close_15d <= 0:
        return True, 0.0

    gain = (close_now - close_15d) / close_15d
    return gain < CALM_PCT, round(gain * 100, 1)


def compute_baseline(broker_sym_df: pd.DataFrame,
                     streak_start: pd.Timestamp) -> float:
    """
    Compute broker's average daily net in this stock before the current streak.
    """
    pre = broker_sym_df[
        (broker_sym_df["date"] < streak_start) &
        (broker_sym_df["net_units"] > 0)
    ]
    if pre.empty:
        return 0.0
    return float(pre["net_units"].mean())


def scan_buy_signals(df: pd.DataFrame,
                     price_df: pd.DataFrame,
                     min_streak: int,
                     min_units: int) -> list[dict]:
    """Scan for buy broker streaks (Dipshikha, Sri Hari, Online, Trishakti)."""
    signals = []
    today   = df["date"].max()

    buy_df = df[df["broker_id"].isin(BUY_BROKERS.keys())].copy()

    for (symbol, broker_id), grp in buy_df.groupby(["symbol", "broker_id"]):
        grp = grp.sort_values("date").reset_index(drop=True)

        streak_days, streak_total = find_current_streak(grp["net_units"])

        if streak_days < min_streak:
            continue
        if streak_total < min_units:
            continue

        # Streak start date
        streak_start = grp["date"].iloc[-(streak_days)]

        # Price calm check
        calm, gain_pct = is_price_calm(price_df, symbol, today)
        if not calm:
            continue  # move already started

        # Baseline comparison
        baseline_avg = compute_baseline(grp, streak_start)
        avg_daily    = streak_total / streak_days
        dominant     = (baseline_avg == 0 or
                        avg_daily > baseline_avg * DOMINANT_MULT)

        # Get latest price
        sym_p = price_df[price_df["symbol"] == symbol]
        price_now = float(sym_p["close"].iloc[-1]) if len(sym_p) > 0 else None

        broker_name = BUY_BROKERS.get(broker_id, broker_id)

        signals.append({
            "signal_type":    "BUY_BROKER",
            "symbol":         symbol,
            "broker_id":      broker_id,
            "broker_name":    broker_name,
            "streak_days":    streak_days,
            "streak_total":   round(streak_total, 0),
            "avg_daily_net":  round(avg_daily, 0),
            "baseline_avg":   round(baseline_avg, 0),
            "is_dominant":    dominant,
            "streak_start":   streak_start.date(),
            "price_now":      price_now,
            "price_gain_15d": gain_pct,
            "confidence":     "HIGH" if dominant and streak_days >= 8
                              else "MEDIUM" if dominant or streak_days >= 10
                              else "LOW",
        })

    return signals


def scan_naasa_sell(df: pd.DataFrame,
                    buy_signals: list[dict],
                    price_df: pd.DataFrame,
                    min_streak: int) -> list[dict]:
    """
    Scan for Naasa net-selling streaks in stocks where a buy broker
    is also active — combined signal = highest conviction.
    """
    signals = []
    today   = df["date"].max()

    # Symbols where buy brokers are already active
    buy_symbols = {s["symbol"] for s in buy_signals}

    naasa_df = df[df["broker_id"] == NAASA_ID].copy()

    for symbol, grp in naasa_df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Naasa net SELL streak (negative net)
        sell_series = -grp["net_units"]  # flip sign so positive = selling
        streak_days, streak_total = find_current_streak(sell_series)

        if streak_days < min_streak:
            continue
        if streak_total < 500:  # at least 500 units being sold
            continue

        calm, gain_pct = is_price_calm(price_df, symbol, today)

        sym_p     = price_df[price_df["symbol"] == symbol]
        price_now = float(sym_p["close"].iloc[-1]) if len(sym_p) > 0 else None

        # Check if a buy broker is also active in this symbol
        combined = symbol in buy_symbols
        buy_broker_detail = [
            f"{s['broker_name']}({s['streak_days']}d)"
            for s in buy_signals if s["symbol"] == symbol
        ]

        signals.append({
            "signal_type":       "NAASA_SELLING",
            "symbol":            symbol,
            "broker_id":         NAASA_ID,
            "broker_name":       NAASA_NAME,
            "naasa_sell_streak": streak_days,
            "naasa_sell_units":  round(streak_total, 0),
            "combined_signal":   combined,
            "buy_brokers_also":  ", ".join(buy_broker_detail) if buy_broker_detail else "none",
            "price_now":         price_now,
            "price_gain_15d":    gain_pct,
            "price_calm":        calm,
            "confidence":        "HIGH"   if combined and calm else
                                 "MEDIUM" if combined else
                                 "LOW",
        })

    return signals


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_signals(buy_signals: list[dict],
                  naasa_signals: list[dict]) -> None:
    """Print formatted signal report."""

    log.info("\n%s", "=" * 80)
    log.info("SMART BROKER SCANNER — %s", date.today())
    log.info("=" * 80)

    # Buy signals by confidence
    high   = [s for s in buy_signals if s["confidence"] == "HIGH"]
    medium = [s for s in buy_signals if s["confidence"] == "MEDIUM"]
    low    = [s for s in buy_signals if s["confidence"] == "LOW"]

    log.info("\n🔴 HIGH CONFIDENCE BUY SIGNALS (%d)", len(high))
    if high:
        log.info("  %-10s %-22s %6s %10s %8s %8s %8s",
                 "Symbol", "Broker", "Days", "Net Units",
                 "Avg/Day", "Price", "15d Gain")
        log.info("  " + "-" * 75)
        for s in sorted(high, key=lambda x: -x["streak_days"]):
            log.info("  %-10s %-22s %6d %10s %8s %8s %7.1f%%",
                     s["symbol"], s["broker_name"][:21],
                     s["streak_days"],
                     f"{s['streak_total']:,.0f}",
                     f"{s['avg_daily_net']:,.0f}",
                     f"{s['price_now']:.1f}" if s["price_now"] else "N/A",
                     s["price_gain_15d"])

    log.info("\n🟡 MEDIUM CONFIDENCE BUY SIGNALS (%d)", len(medium))
    if medium:
        for s in sorted(medium, key=lambda x: -x["streak_days"]):
            log.info("  %-10s %-22s %dd streak  %.0f net units  price=%s  15d=%.1f%%",
                     s["symbol"], s["broker_name"][:21],
                     s["streak_days"], s["streak_total"],
                     f"{s['price_now']:.1f}" if s["price_now"] else "N/A",
                     s["price_gain_15d"])

    if low:
        log.info("\n⚪ LOW CONFIDENCE (%d) — watching:", len(low))
        for s in sorted(low, key=lambda x: -x["streak_days"]):
            log.info("  %-10s %-22s %dd  %.0f units",
                     s["symbol"], s["broker_name"][:21],
                     s["streak_days"], s["streak_total"])

    # Naasa signals
    naasa_high   = [s for s in naasa_signals if s["confidence"] == "HIGH"]
    naasa_medium = [s for s in naasa_signals if s["confidence"] == "MEDIUM"]

    log.info("\n🔵 NAASA SELLING (bullish confirmation) — HIGH (%d)", len(naasa_high))
    if naasa_high:
        log.info("  %-10s %6s %12s %8s  %-30s",
                 "Symbol", "Days", "Sell Units", "Price", "Buy Brokers Also Active")
        log.info("  " + "-" * 75)
        for s in sorted(naasa_high, key=lambda x: -x["naasa_sell_streak"]):
            log.info("  %-10s %6d %12s %8s  %s",
                     s["symbol"], s["naasa_sell_streak"],
                     f"{s['naasa_sell_units']:,.0f}",
                     f"{s['price_now']:.1f}" if s["price_now"] else "N/A",
                     s["buy_brokers_also"][:45])

    if naasa_medium:
        log.info("\n🔵 NAASA SELLING — MEDIUM (%d) — no confirmed buy broker yet:",
                 len(naasa_medium))
        for s in sorted(naasa_medium, key=lambda x: -x["naasa_sell_streak"]):
            log.info("  %-10s %dd sell streak  %.0f units  price=%s",
                     s["symbol"], s["naasa_sell_streak"],
                     s["naasa_sell_units"],
                     f"{s['price_now']:.1f}" if s["price_now"] else "N/A")

    log.info("\n%s", "=" * 80)
    log.info("SUMMARY")
    log.info("=" * 80)
    log.info("Buy broker signals:  %d (High=%d  Medium=%d  Low=%d)",
             len(buy_signals), len(high), len(medium), len(low))
    log.info("Naasa sell signals:  %d (High=%d  Medium=%d)",
             len(naasa_signals), len(naasa_high), len(naasa_medium))
    combined = [s for s in naasa_signals if s["combined_signal"]]
    log.info("COMBINED (buy+naasa):%d — STRONGEST SIGNALS", len(combined))
    if combined:
        log.info("  → %s", ", ".join(s["symbol"] for s in combined))


def save_csv(buy_signals: list[dict],
             naasa_signals: list[dict]) -> None:
    today    = date.today().strftime("%Y-%m-%d")
    out_path = OUT_DIR / f"smart_broker_{today}.csv"

    all_signals = []
    for s in buy_signals:
        all_signals.append({
            "date":        today,
            "signal_type": s["signal_type"],
            "symbol":      s["symbol"],
            "broker":      s["broker_name"],
            "streak_days": s["streak_days"],
            "net_units":   s["streak_total"],
            "confidence":  s["confidence"],
            "price_now":   s.get("price_now", ""),
            "price_gain_15d": s["price_gain_15d"],
            "notes":       f"dominant={s['is_dominant']} baseline={s['baseline_avg']:.0f}",
        })
    for s in naasa_signals:
        all_signals.append({
            "date":        today,
            "signal_type": s["signal_type"],
            "symbol":      s["symbol"],
            "broker":      s["broker_name"],
            "streak_days": s["naasa_sell_streak"],
            "net_units":   s["naasa_sell_units"],
            "confidence":  s["confidence"],
            "price_now":   s.get("price_now", ""),
            "price_gain_15d": s["price_gain_15d"],
            "notes":       f"combined={s['combined_signal']} buy_also={s['buy_brokers_also']}",
        })

    if all_signals:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_signals[0].keys()))
            writer.writeheader()
            writer.writerows(all_signals)
        log.info("Saved to %s", out_path)


def send_telegram(buy_signals: list[dict],
                  naasa_signals: list[dict]) -> None:
    """Send Telegram alert for high confidence signals."""
    try:
        from helper.notifier import send_admin_message
    except ImportError:
        log.warning("Notifier not available — skipping Telegram")
        return

    high_buy   = [s for s in buy_signals   if s["confidence"] == "HIGH"]
    high_naasa = [s for s in naasa_signals  if s["confidence"] == "HIGH"]
    combined   = [s for s in naasa_signals  if s["combined_signal"]]

    if not high_buy and not high_naasa:
        log.info("No high-confidence signals — skipping Telegram")
        return

    lines = [f"🎯 *Smart Broker Alert — {date.today()}*\n"]

    if combined:
        lines.append("🔥 *COMBINED SIGNALS (highest conviction):*")
        for s in combined:
            lines.append(f"  `{s['symbol']}` — Naasa selling ({s['naasa_sell_streak']}d) "
                         f"+ {s['buy_brokers_also']}")
        lines.append("")

    if high_buy:
        lines.append("🟢 *Smart Money Buying:*")
        for s in sorted(high_buy, key=lambda x: -x["streak_days"]):
            dom = "★" if s["is_dominant"] else ""
            lines.append(f"  `{s['symbol']}` — {s['broker_name']} "
                         f"{s['streak_days']}d streak "
                         f"{s['streak_total']:,.0f} units {dom}")
        lines.append("")

    if high_naasa and not combined:
        lines.append("🔵 *Naasa Selling (bullish):*")
        for s in high_naasa:
            lines.append(f"  `{s['symbol']}` — {s['naasa_sell_streak']}d "
                         f"{s['naasa_sell_units']:,.0f} units sold")

    msg = "\n".join(lines)
    send_admin_message(msg)
    log.info("Telegram sent")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",      type=int,  default=MIN_STREAK,
                        help=f"Min streak days (default {MIN_STREAK})")
    parser.add_argument("--min-units", type=int,  default=MIN_NET_UNITS)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    from_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    from_date = max(from_date, FS_START)

    # Load all 5 broker IDs
    all_broker_ids = list(BUY_BROKERS.keys()) + [NAASA_ID]
    df = load_broker_daily(all_broker_ids, from_date)

    if df.empty:
        log.error("No floorsheet data loaded")
        return

    # Load price for calm check
    price_from = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    price_df   = load_price_recent(price_from)

    log.info("Floorsheet rows: %d | Symbols: %d | Date range: %s → %s",
             len(df), df["symbol"].nunique(),
             df["date"].min().date(), df["date"].max().date())

    # Scan
    buy_signals   = scan_buy_signals(df, price_df, args.days, args.min_units)
    naasa_signals = scan_naasa_sell(df, buy_signals, price_df, args.days)

    # Print
    print_signals(buy_signals, naasa_signals)

    if args.dry_run:
        log.info("Dry run — not saving or sending Telegram")
        return

    # Save CSV
    save_csv(buy_signals, naasa_signals)

    # Telegram
    if not args.no_telegram:
        send_telegram(buy_signals, naasa_signals)


if __name__ == "__main__":
    main()