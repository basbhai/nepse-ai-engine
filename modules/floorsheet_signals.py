"""
modules/floorsheet_signals.py — Compute daily signals from raw floorsheet
Reads from floorsheet table, writes to floorsheet_signals table.
Fully vectorized — no Python loops in computation paths.
Uses max 2 CPU cores via pandas/numpy only.
"""

import logging
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sheets import write_row, run_raw_sql, update_row
from config import NST

log = logging.getLogger("floorsheet_signals")


# ─── Load Raw Floorsheet ─────────────────────────────────────────────────────

def _load_raw(target_date: date) -> pd.DataFrame:
    """Load raw floorsheet rows for a given date."""
    try:
        rows = run_raw_sql("""
            SELECT
                symbol,
                contract_id,
                buyer_broker_id,
                seller_broker_id,
                quantity,
                rate,
                amount,
                trade_time,
                source
            FROM floorsheet
            WHERE date = %s
        """, [target_date.strftime("%Y-%m-%d")])

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Vectorized numeric conversion
        for col in ["quantity", "rate", "amount"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""),
                errors="coerce"
            ).fillna(0.0)

        df["buyer_broker_id"]  = df["buyer_broker_id"].astype(str)
        df["seller_broker_id"] = df["seller_broker_id"].astype(str)

        return df

    except Exception as e:
        log.error(f"_load_raw {target_date}: {e}")
        return pd.DataFrame()


# ─── Compute Signals ──────────────────────────────────────────────────────────

def compute_signals(target_date: date) -> pd.DataFrame:
    """
    Compute all floorsheet signals for every symbol on a given date.
    Fully vectorized using pandas groupby + numpy aggregations.
    Returns DataFrame with one row per symbol.
    """
    df = _load_raw(target_date)
    if df.empty:
        log.warning(f"compute_signals: no raw data for {target_date}")
        return pd.DataFrame()

    date_str = target_date.strftime("%Y-%m-%d")
    source   = df["source"].iloc[0] if "source" in df.columns else "unknown"

    # ── Per-symbol groupby aggregations — vectorized
    grp = df.groupby("symbol")

    agg = grp.agg(
        total_trades         = ("quantity",         "count"),
        total_volume         = ("quantity",         "sum"),
        total_turnover       = ("amount",           "sum"),
        avg_trade_size       = ("quantity",         "mean"),
        median_trade_size    = ("quantity",         "median"),
    ).reset_index()

    # ── VWAP — vectorized per symbol
    vwap_num = grp.apply(
        lambda g: (g["quantity"] * g["rate"]).sum()
    ).reset_index(name="vwap_num")
    vwap_den = grp["quantity"].sum().reset_index(name="vwap_den")
    vwap_df  = vwap_num.merge(vwap_den, on="symbol")
    vwap_df["vwap"] = np.where(
        vwap_df["vwap_den"] > 0,
        vwap_df["vwap_num"] / vwap_df["vwap_den"],
        0.0
    ).round(2)
    agg = agg.merge(vwap_df[["symbol", "vwap"]], on="symbol", how="left")

    # ── Large order detection — qty > 2× median per symbol — vectorized
    # Merge median back onto raw df
    df = df.merge(
        agg[["symbol", "median_trade_size"]],
        on="symbol", how="left"
    )
    df["is_large"] = df["quantity"] > (2 * df["median_trade_size"])

    large_agg = df[df["is_large"]].groupby("symbol").agg(
        large_order_count  = ("quantity", "count"),
        large_order_volume = ("quantity", "sum"),
    ).reset_index()
    agg = agg.merge(large_agg, on="symbol", how="left")
    agg["large_order_count"]  = agg["large_order_count"].fillna(0).astype(int)
    agg["large_order_volume"] = agg["large_order_volume"].fillna(0.0)
    agg["large_order_pct"]    = np.where(
        agg["total_volume"] > 0,
        agg["large_order_volume"] / agg["total_volume"],
        0.0
    ).round(4)

    # ── Buyer/seller pressure — recurring broker activity — vectorized
    buyer_vol  = df.groupby(["symbol", "buyer_broker_id"])["quantity"].sum().reset_index()
    seller_vol = df.groupby(["symbol", "seller_broker_id"])["quantity"].sum().reset_index()

    # Top buyer per symbol
    top_buyer = (buyer_vol
                 .sort_values("quantity", ascending=False)
                 .groupby("symbol")
                 .first()
                 .reset_index()
                 .rename(columns={"buyer_broker_id": "top_buyer_broker_id",
                                   "quantity": "top_buyer_volume"}))

    # Top seller per symbol
    top_seller = (seller_vol
                  .sort_values("quantity", ascending=False)
                  .groupby("symbol")
                  .first()
                  .reset_index()
                  .rename(columns={"seller_broker_id": "top_seller_broker_id",
                                    "quantity": "top_seller_volume"}))

    agg = agg.merge(top_buyer[["symbol", "top_buyer_broker_id", "top_buyer_volume"]],
                    on="symbol", how="left")
    agg = agg.merge(top_seller[["symbol", "top_seller_broker_id", "top_seller_volume"]],
                    on="symbol", how="left")

    # Buyer/seller pressure ratios
    agg["buyer_pressure"] = np.where(
        agg["total_volume"] > 0,
        agg["top_buyer_volume"] / agg["total_volume"],
        0.0
    ).round(4)
    agg["seller_pressure"] = np.where(
        agg["total_volume"] > 0,
        agg["top_seller_volume"] / agg["total_volume"],
        0.0
    ).round(4)

    # ── Broker concentration — top 3 brokers % of volume — vectorized
    top3_buyer = (buyer_vol
                  .sort_values(["symbol", "quantity"], ascending=[True, False])
                  .groupby("symbol")
                  .head(3)
                  .groupby("symbol")["quantity"]
                  .sum()
                  .reset_index(name="top3_buyer_vol"))
    agg = agg.merge(top3_buyer, on="symbol", how="left")
    agg["broker_concentration"] = np.where(
        agg["total_volume"] > 0,
        agg["top3_buyer_vol"] / agg["total_volume"],
        0.0
    ).round(4)

    # ── Institutional flag — large orders + broker concentration
    agg["institutional_flag"] = (
        (agg["large_order_pct"]       > 0.20) &
        (agg["broker_concentration"]  > 0.40)
    ).astype(str).str.lower()  # "true" | "false"

    # ── Final columns
    agg["date"]         = date_str
    agg["source"]       = source
    agg["computed_at"]  = datetime.now(NST).strftime("%Y-%m-%d %H:%M:%S")

    # Round numeric cols
    for col in ["avg_trade_size", "median_trade_size", "total_volume",
                "total_turnover", "large_order_volume"]:
        if col in agg.columns:
            agg[col] = agg[col].round(2)

    log.info(f"compute_signals {target_date}: {len(agg)} symbols computed")
    return agg


# ─── Write Signals ────────────────────────────────────────────────────────────

def write_signals(df: pd.DataFrame) -> int:
    """Write computed signals to floorsheet_signals table."""
    if df.empty:
        return 0

    col_map = {
        "vwap"                : "vwap",
        "total_trades"        : "total_trades",
        "total_volume"        : "total_volume",
        "total_turnover"      : "total_turnover",
        "avg_trade_size"      : "avg_trade_size",
        "large_order_count"   : "large_order_count",
        "large_order_volume"  : "large_order_volume",
        "large_order_pct"     : "large_order_pct",
        "buyer_pressure"      : "buyer_pressure",
        "seller_pressure"     : "seller_pressure",
        "top_buyer_broker_id" : "top_buyer_broker_id",
        "top_seller_broker_id": "top_seller_broker_id",
        "broker_concentration": "broker_concentration",
        "institutional_flag"  : "institutional_flag",
        "source"              : "source",
        "computed_at"         : "computed_at",
    }

    written = 0
    for _, row in df.iterrows():
        try:
            rec = {
                "date"   : row["date"],
                "symbol" : row["symbol"],
            }
            for df_col, db_col in col_map.items():
                if df_col in row.index:
                    val = row[df_col]
                    rec[db_col] = str(val) if pd.notna(val) else None

            write_row("floorsheet_signals", rec)
            written += 1
        except Exception as e:
            log.debug(f"floorsheet_signals write skip {row.get('symbol')}: {e}")

    log.info(f"write_signals: {written} rows written")
    return written


# ─── Run ─────────────────────────────────────────────────────────────────────

def run(target_date: date = None) -> int:
    """
    Compute and write signals for a given date.
    Called by morning_workflow after floorsheet_scraper.run_daily().
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    # Skip weekends
    if target_date.weekday() >= 5:
        log.info(f"floorsheet_signals: {target_date} is weekend — skip")
        return 0

    log.info(f"Computing floorsheet signals for {target_date}")
    df      = compute_signals(target_date)
    written = write_signals(df)
    return written


def run_backfill(start_date: date, end_date: date = None) -> None:
    """Recompute signals for all dates in range. Useful after backfill scrape."""
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    days = pd.date_range(start=start_date, end=end_date, freq="B")
    log.info(f"Signal backfill: {start_date} → {end_date} | {len(days)} days")

    for i, d in enumerate(days):
        target = d.date()
        rows   = run(target)
        if (i + 1) % 20 == 0:
            log.info(f"Progress: {i+1}/{len(days)} days")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="Floorsheet Signal Computation")
    parser.add_argument("--date",  default=None, help="YYYY-MM-DD")
    parser.add_argument("--backfill-start", default=None)
    parser.add_argument("--backfill-end",   default=None)
    args = parser.parse_args()

    if args.backfill_start:
        start = datetime.strptime(args.backfill_start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.backfill_end, "%Y-%m-%d").date() if args.backfill_end else None
        run_backfill(start, end)
    else:
        target = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
        rows   = run(target)
        print(f"Signals computed: {rows} symbols written")
