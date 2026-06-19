#!/usr/bin/env python3
"""
force_close_open_positions.py
──────────────────────────────────────────────────────────────────────────
ONE-TIME cutover script: force-close all currently OPEN paper_portfolio
positions at TODAY'S live intraday low, ahead of the Telegram→Discord
migration and the "fresh start" reset.

Mirrors the EXACT write pattern of execute_sell() in modules/telegram_bot.py
(verified against db/schema.py) — same paper_portfolio columns
(exit_date, exit_price, exit_shares, gross_pnl, sell_fees, cgt_paid,
net_pnl, result), same paper_capital update (additive current_capital,
::int total_trades casts), same paper_trade_log insert (gross_amount,
brokerage, sebon, dp_fee, cgt, total_fees, net_amount, capital_before/after,
wacc_before/after) — so these closes are indistinguishable in the DB from a
real /sell, and DO count toward win_rate / total_trades per your decision.

SAFE BY DEFAULT: running with no flags only PRINTS what would happen.
Nothing is written to the DB until you re-run with --confirm.

IMPORTANT: uses db.connection._db() directly (the same context manager
telegram_bot.py uses for atomic writes) — NOT run_raw_sql(), which cannot
execute DML per project convention.

Usage:
    python force_close_open_positions.py             # dry run, prints only
    python force_close_open_positions.py --confirm    # actually writes

Open positions being closed (from DB query on 2026-06-17):
    5432461414 (admin)  — ALBSL 100sh @ WACC 1104.0450
    5432461414 (admin)  — GBBL   100sh @ WACC  420.0784
    7860783056          — KSBBL 100sh @ WACC  492.0875
"""

import sys
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

from dotenv import load_dotenv
load_dotenv()

from modules.scraper import get_watchlist_data
from sheets import run_raw_sql          # read-only lookups only
from db.connection import _db           # atomic DML — same pattern as telegram_bot.py

# ─── Constants (mirrored from modules/telegram_bot.py) ──────────────────────
SEBON_PCT = Decimal("0.00015")
DP_FEE    = Decimal("25")
CGT_RATE  = Decimal("0.075")

POSITIONS_TO_CLOSE = [
    {"telegram_id": "5432461414", "symbol": "ALBSL", "shares": Decimal("100"), "wacc": Decimal("1104.0450")},
    {"telegram_id": "5432461414", "symbol": "GBBL",  "shares": Decimal("100"), "wacc": Decimal("420.0784")},
    {"telegram_id": "7860783056", "symbol": "KSBBL", "shares": Decimal("100"), "wacc": Decimal("492.0875")},
]


def _brokerage(gross: Decimal) -> Decimal:
    if gross <= Decimal("2500"):
        return Decimal("10")
    elif gross <= Decimal("50000"):
        rate = Decimal("0.0036")
    elif gross <= Decimal("500000"):
        rate = Decimal("0.0033")
    elif gross <= Decimal("2000000"):
        rate = Decimal("0.0031")
    elif gross <= Decimal("10000000"):
        rate = Decimal("0.0027")
    else:
        rate = Decimal("0.0024")
    return (gross * rate).quantize(Decimal("0.01"), ROUND_HALF_UP)


def calc_sell_fees(price: Decimal, shares: Decimal, wacc: Decimal) -> dict:
    """Mirrors modules/telegram_bot.py calc_sell_fees() exactly."""
    gross        = (price * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    brokerage    = _brokerage(gross)
    sebon        = (gross * SEBON_PCT).quantize(Decimal("0.01"), ROUND_HALF_UP)
    total_fees   = brokerage + sebon + DP_FEE
    cost_basis   = (wacc * shares).quantize(Decimal("0.01"), ROUND_HALF_UP)
    gross_pnl    = gross - cost_basis
    cgt          = (max(Decimal("0"), gross_pnl) * CGT_RATE).quantize(Decimal("0.01"), ROUND_HALF_UP)
    net_pnl      = gross_pnl - total_fees - cgt
    net_proceeds = gross - total_fees - cgt
    return {
        "gross": gross, "brokerage": brokerage, "sebon": sebon, "dp_fee": DP_FEE,
        "total_fees": total_fees, "cgt": cgt, "gross_pnl": gross_pnl,
        "net_pnl": net_pnl, "net_proceeds": net_proceeds,
    }


def get_paper_position(telegram_id: str, symbol: str):
    rows = run_raw_sql(
        "SELECT * FROM paper_portfolio WHERE telegram_id = %s AND symbol = %s AND status = 'OPEN'",
        (telegram_id, symbol),
    )
    return rows[0] if rows else None


def get_capital_row(telegram_id: str):
    rows = run_raw_sql("SELECT * FROM paper_capital WHERE telegram_id = %s", (telegram_id,))
    return rows[0] if rows else None


def main():
    confirm = "--confirm" in sys.argv

    symbols = [p["symbol"] for p in POSITIONS_TO_CLOSE]
    print(f"Fetching live intraday data for: {symbols}")
    market_data = get_watchlist_data(symbols)

    today_str = date.today().isoformat()
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"FORCE-CLOSE PREVIEW — {today_str}  ({'WRITE MODE' if confirm else 'DRY RUN — no DB writes'})")
    print(f"{'='*70}\n")

    closes = []
    for pos in POSITIONS_TO_CLOSE:
        sym = pos["symbol"]
        row = market_data.get(sym)
        if row is None:
            print(f"❌ {sym}: no live data returned — SKIPPING. "
                  f"(Market may be closed, or symbol outside ATrad's 542-symbol set.)")
            continue

        low_price = Decimal(str(row.low))
        if low_price <= 0:
            print(f"❌ {sym}: live low price is {low_price} — SKIPPING (invalid).")
            continue

        existing = get_paper_position(pos["telegram_id"], sym)
        if not existing:
            print(f"⚠️  {sym} ({pos['telegram_id']}): no OPEN row found in DB right now "
                  f"(already closed, or data changed since last check) — SKIPPING.")
            continue

        actual_shares = Decimal(str(existing["total_shares"]))
        actual_wacc   = Decimal(str(existing["wacc"]))
        if actual_shares != pos["shares"] or actual_wacc != pos["wacc"]:
            print(f"⚠️  {sym}: DB values differ from script's hardcoded expectation "
                  f"(DB: {actual_shares}sh @ {actual_wacc} vs expected {pos['shares']}sh @ {pos['wacc']}). "
                  f"Using LIVE DB values, not hardcoded ones.")

        fees = calc_sell_fees(low_price, actual_shares, actual_wacc)
        result = "WIN" if fees["net_pnl"] > 0 else ("LOSS" if fees["net_pnl"] < 0 else "BREAKEVEN")

        print(f"--- {sym}  (telegram_id={pos['telegram_id']}, portfolio.id={existing['id']}) ---")
        print(f"  Shares:          {actual_shares}")
        print(f"  Entry WACC:      NPR {actual_wacc}")
        print(f"  Today's low:     NPR {low_price}   <-- forced close price")
        print(f"  Gross sale:      NPR {fees['gross']}")
        print(f"  Brokerage:       NPR {fees['brokerage']}")
        print(f"  SEBON:           NPR {fees['sebon']}")
        print(f"  DP fee:          NPR {fees['dp_fee']}")
        print(f"  CGT:             NPR {fees['cgt']}")
        print(f"  Net P&L:         NPR {fees['net_pnl']}  -> {result}")
        print(f"  Net proceeds:    NPR {fees['net_proceeds']}")
        print()

        closes.append({
            "telegram_id": pos["telegram_id"], "symbol": sym,
            "portfolio_id": existing["id"], "shares": actual_shares, "wacc": actual_wacc,
            "exit_price": low_price, "fees": fees, "result": result,
        })

    if not closes:
        print("Nothing to close — no valid live prices / open rows found. Exiting.")
        return

    print(f"{'='*70}")
    print(f"SUMMARY: {len(closes)} position(s) will be closed.")
    for c in closes:
        print(f"  {c['symbol']:8s} ({c['telegram_id']}): {c['result']:9s} net {c['fees']['net_pnl']:+.2f}")
    print(f"{'='*70}\n")

    if not confirm:
        print("This was a DRY RUN. No database writes were made.")
        print("Review the numbers above. If correct, re-run with --confirm to write.")
        return

    print("⚠️  WRITE MODE — committing to database now...\n")

    for c in closes:
        tid, sym, pf_id, shares, wacc, exit_price, fees, result = (
            c["telegram_id"], c["symbol"], c["portfolio_id"], c["shares"], c["wacc"],
            c["exit_price"], c["fees"], c["result"],
        )
        cap = get_capital_row(tid)
        if not cap:
            print(f"  ❌ {sym}: no paper_capital row for {tid} — SKIPPING WRITE for this position.")
            continue
        available = Decimal(str(cap["current_capital"]))
        new_capital = available + fees["net_proceeds"]
        is_win  = fees["net_pnl"] > 0
        is_loss = fees["net_pnl"] < 0

        with _db() as cur:
            # 1. Close the paper_portfolio row — same columns as real execute_sell
            cur.execute("""
                UPDATE paper_portfolio SET
                    status      = 'CLOSED',
                    exit_date   = %s,
                    exit_price  = %s,
                    exit_shares = %s,
                    gross_pnl   = %s,
                    sell_fees   = %s,
                    cgt_paid    = %s,
                    net_pnl     = %s,
                    result      = %s,
                    updated_at  = %s
                WHERE id = %s
            """, (today_str, str(exit_price), str(shares),
                  str(fees["gross_pnl"]), str(fees["total_fees"]), str(fees["cgt"]),
                  str(fees["net_pnl"]), f"FORCE_CLOSE_{result}", now_str, pf_id))

            # 2. Update paper_capital — additive, same as real execute_sell
            cur.execute("""
                UPDATE paper_capital SET
                    current_capital    = %s,
                    total_realised_pnl = (total_realised_pnl::numeric + %s)::text,
                    total_fees_paid    = (total_fees_paid::numeric + %s)::text,
                    total_cgt_paid     = (total_cgt_paid::numeric + %s)::text,
                    total_trades       = (total_trades::int + 1)::text,
                    total_wins         = (total_wins::int + %s)::text,
                    total_losses       = (total_losses::int + %s)::text,
                    last_updated       = %s
                WHERE telegram_id = %s
            """, (str(new_capital), str(fees["net_pnl"]), str(fees["total_fees"]), str(fees["cgt"]),
                  1 if is_win else 0, 1 if is_loss else 0, now_str, tid))

            # 3. Append paper_trade_log — exact real columns
            cur.execute("""
                INSERT INTO paper_trade_log
                    (telegram_id, symbol, action, shares, price, gross_amount,
                     brokerage, sebon, dp_fee, cgt, total_fees, net_amount,
                     capital_before, capital_after, wacc_before, wacc_after,
                     note, created_at, test_mode)
                VALUES (%s,%s,'SELL',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (str(tid), sym, str(shares), str(exit_price), str(fees["gross"]),
                  str(fees["brokerage"]), str(fees["sebon"]), str(DP_FEE), str(fees["cgt"]),
                  str(fees["total_fees"]), str(fees["net_proceeds"]),
                  str(available), str(new_capital), str(wacc), str(wacc),
                  "FORCE_CLOSE: Telegram->Discord migration cutover", now_str, "false"))

        print(f"  ✅ Closed {sym} for {tid} — {result}, net {fees['net_pnl']:+.2f}, "
              f"capital now {new_capital}")

    print("\nDone. All positions force-closed and written to DB.")
    print("Next step: run the reset/cleanup script for the fresh-start cutover.")


if __name__ == "__main__":
    main()