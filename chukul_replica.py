"""
chukul_replica_v2.py
Replicates player-favourite-history and players-choices from local floorsheet.
Logic reverse-engineered from Chukul's bulk_transactions data.

Algorithm:
- Per symbol, per broker: compute total BUY qty and SELL qty for the day
- Broker is classified as BUY if buy_qty > sell_qty, else SELL
- Only include brokers where max(buy_qty, sell_qty) >= CUMM_THRESHOLD
- player-favourite: aggregate all BUY-brokers qty and SELL-brokers qty per symbol
- players-choices: net dominant side per symbol by amount
"""

import argparse
import json
import logging
from collections import defaultdict
from db.connection import _db

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Thresholds ────────────────────────────────────────────────────────────────
CUMM_THRESHOLD = 6000       # min qty for a broker to be considered "bulk"

STRENGTH_BY_QTY = [
    (100_000, "Very Very Strong"),
    (50_000,  "Very Strong"),
    (20_000,  "Strong"),
    (5_000,   "Normal"),
    (0,       "Weak"),
]

CHOICE_STRENGTH_BY_AMT = [
    (100_000_000, "Strong"),
    (50_000_000,  "Medium"),
    (0,           "Weak"),
]


def _qty_strength(qty):
    for threshold, label in STRENGTH_BY_QTY:
        if qty >= threshold:
            return label
    return "Weak"


def _amt_strength(amt):
    for threshold, label in CHOICE_STRENGTH_BY_AMT:
        if amt >= threshold:
            return label
    return "Weak"


# ── Load floorsheet ───────────────────────────────────────────────────────────
def load_floorsheet(date: str) -> list[dict]:
    with _db() as cur:
        cur.execute("""
            SELECT
                symbol,
                buyer_broker_id,
                seller_broker_id,
                quantity::float  AS qty,
                rate::float      AS rate,
                amount::float    AS amount
            FROM floorsheet
            WHERE (date = %s OR date = %s)
              AND quantity IS NOT NULL AND quantity != ''
              AND quantity ~ '^[0-9]+\\.?[0-9]*$'
              AND quantity::float > 0
        """, [date, date])
        return [dict(r) for r in cur.fetchall()]


def load_prev_symbols(prev_date: str) -> set[str]:
    try:
        rows = load_floorsheet(prev_date)
        return {str(r["symbol"]).upper() for r in rows}
    except Exception:
        return set()


# ── Core computation ──────────────────────────────────────────────────────────
def compute(rows: list[dict], date: str, prev_syms: set[str]) -> tuple[list, dict]:
    """
    Returns (player_favourite, players_choices)
    """
    # (symbol, broker_id) → {buy_qty, sell_qty, buy_amt, sell_amt}
    broker_agg = defaultdict(lambda: {
        "buy_qty": 0.0, "sell_qty": 0.0,
        "buy_amt": 0.0, "sell_amt": 0.0
    })

    for r in rows:
        sym    = str(r["symbol"]).upper().strip()
        qty    = float(r["qty"] or 0)
        amt    = float(r["amount"] or 0)
        rate   = float(r["rate"] or 0)
        if amt == 0 and qty > 0 and rate > 0:
            amt = qty * rate

        buyer  = str(r.get("buyer_broker_id") or "").strip()
        seller = str(r.get("seller_broker_id") or "").strip()

        if buyer and buyer not in ("", "None", "0", "nan"):
            k = (sym, buyer)
            broker_agg[k]["buy_qty"] += qty
            broker_agg[k]["buy_amt"] += amt

        if seller and seller not in ("", "None", "0", "nan"):
            k = (sym, seller)
            broker_agg[k]["sell_qty"] += qty
            broker_agg[k]["sell_amt"] += amt

    # ── Aggregate per symbol ──────────────────────────────────────────────────
    # sym → {buy_qty, sell_qty, buy_amt, sell_amt}
    sym_agg = defaultdict(lambda: {
        "buy_qty": 0.0, "sell_qty": 0.0,
        "buy_amt": 0.0, "sell_amt": 0.0
    })

    for (sym, broker), v in broker_agg.items():
            buy_qty  = v["buy_qty"]
            sell_qty = v["sell_qty"]
            
            # NET position — not gross
            net_qty = buy_qty - sell_qty
            net_amt = v["buy_amt"] - v["sell_amt"]
            
            if abs(net_qty) < CUMM_THRESHOLD:
                continue

            if net_qty > 0:  # net buyer
                sym_agg[sym]["buy_qty"] += net_qty
                sym_agg[sym]["buy_amt"] += net_amt
            else:            # net seller
                sym_agg[sym]["sell_qty"] += abs(net_qty)
                sym_agg[sym]["sell_amt"] += abs(net_amt)
    # ── Build player-favourite ────────────────────────────────────────────────
    fav = []
    for sym, v in sym_agg.items():
        buy_qty  = v["buy_qty"]
        sell_qty = v["sell_qty"]
        buy_amt  = v["buy_amt"]
        sell_amt = v["sell_amt"]

        if buy_qty > 0:
            rate = round(buy_amt / buy_qty, 2) if buy_qty > 0 else 0
            fav.append({
                "date":          date,
                "cnt":           0,   # we don't track block count
                "symbol":        sym,
                "quantity":      round(buy_qty, 0),
                "rate":          rate,
                "amount":        round(buy_amt, 2),
                "cumm_quantity": round(buy_qty, 0),
                "bulk_status":   "BUY",
                "strength":      _qty_strength(buy_qty),
                "local_date":    date,
            })

        if sell_qty > 0:
            rate = round(sell_amt / sell_qty, 2) if sell_qty > 0 else 0
            fav.append({
                "date":          date,
                "cnt":           0,
                "symbol":        sym,
                "quantity":      round(sell_qty, 0),
                "rate":          rate,
                "amount":        round(sell_amt, 2),
                "cumm_quantity": round(sell_qty, 0),
                "bulk_status":   "SELL",
                "strength":      _qty_strength(sell_qty),
                "local_date":    date,
            })

    fav.sort(key=lambda x: x["amount"], reverse=True)

    # ── Build players-choices ─────────────────────────────────────────────────
    choices = []
    for sym, v in sym_agg.items():
        buy_amt  = v["buy_amt"]
        sell_amt = v["sell_amt"]

        if buy_amt == 0 and sell_amt == 0:
            continue

        dominant     = "BUY" if buy_amt >= sell_amt else "SELL"
        dominant_amt = buy_amt if dominant == "BUY" else sell_amt
        status       = "New Entry" if sym not in prev_syms else "Active"

        choices.append({
            "symbol":   sym,
            "amt":      round(dominant_amt, 2),
            "strength": _amt_strength(dominant_amt),
            "b_s":      dominant,
            "status":   status,
        })

    choices.sort(key=lambda x: x["amt"], reverse=True)

    return fav, {"date": date, "data": choices}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",      default="2026-06-05")
    parser.add_argument("--prev-date", default="2026-06-04")
    parser.add_argument("--top",       type=int, default=20)
    args = parser.parse_args()

    log.info(f"Loading floorsheet for {args.date}...")
    rows = load_floorsheet(args.date)
    log.info(f"  {len(rows)} rows loaded")

    if not rows:
        log.error("No data. Check date or DB.")
        return

    prev_syms = load_prev_symbols(args.prev_date)

    fav, choices = compute(rows, args.date, prev_syms)

    print("\n" + "="*60)
    print(f"PLAYER-FAVOURITE  ({args.date})  top {args.top}")
    print("="*60)
    print(json.dumps(fav[:args.top], indent=2))

    print("\n" + "="*60)
    print(f"PLAYERS-CHOICES  ({args.date})  top {args.top}")
    print("="*60)
    print(json.dumps(choices["data"][:args.top], indent=2))

    # ── Validation against Chukul ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("VALIDATION (AKJCL, CFCL, RIDI vs Chukul)")
    print("="*60)
    targets = ["AKJCL", "CFCL", "RIDI", "NHPC", "MFIL"]
    for sym in targets:
        rows_sym = [r for r in fav if r["symbol"] == sym]
        for r in rows_sym:
            print(f"  {sym} {r['bulk_status']:4s} qty={r['quantity']:>10,.0f}  amt={r['amount']:>15,.0f}  {r['strength']}")
        if not rows_sym:
            print(f"  {sym} — not found")


if __name__ == "__main__":
    main()