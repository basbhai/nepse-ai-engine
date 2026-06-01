# -*- coding: utf-8 -*-
"""
spring_loading_screener.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Research Tool

Scans all symbols for the "Spring Loading" pre-entry setup identified from
50%+ gain window analysis (DeepSeek ML findings, June 2026).

Spring Loading criteria (all 6 must pass for Telegram alert):
  1. RSI 28–42 (oversold but not crashed)
  2. MACD histogram negative but improving for 2+ consecutive days
  3. OBV 5-day slope positive (accumulation)
  4. ADX < 25 (pre-breakout, not yet trending)
  5. BB width contracting vs 10-day average (squeeze forming)
  6. Volume below 20-day average (quiet accumulation)

Scoring: stocks meeting 4-5 criteria logged silently to DB.
         Stocks meeting all 6 → Telegram admin alert.

Run: python -m modules.spring_loading_screener
     python -m modules.spring_loading_screener --dry-run   (no DB write, no Telegram)
     python -m modules.spring_loading_screener --all        (show all scores, not just alerts)

Schedule: Add to summary_workflow.py after EOD price_history is updated.

Import rule: from sheets import ... — NEVER from db import ...
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import sys
import json
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta

from config import NST
from sheets import (
    run_raw_sql,
    get_setting,
    upsert_row,
    write_row,
)
from helper.notifier import _send_admin_only

log = logging.getLogger(__name__)

# ── Thresholds (all read from settings with fallback) ─────────────────────────
RSI_MIN          = 28.0
RSI_MAX          = 42.0
ADX_MAX          = 25.0
MACD_IMPROVE_DAYS = 2       # histogram must improve for this many consecutive days
OBV_SLOPE_DAYS   = 5        # OBV slope window
BB_SQUEEZE_RATIO = 0.90     # BB width must be < 90% of 10-day avg
VOLUME_RATIO_MAX = 0.85     # volume < 85% of 20-day avg (quiet)
MIN_HISTORY_DAYS = 60       # skip symbols with less than this many days
ALERT_THRESHOLD  = 6        # all 6 → Telegram
LOG_THRESHOLD    = 4        # 4+ → log to DB silently
LOOKBACK_DAYS    = 90       # days of price history to load per symbol

DRY_RUN = "--dry-run" in sys.argv
SHOW_ALL = "--all" in sys.argv


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _load_price_history(lookback_days: int = LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    """
    Load last N days of price_history for all symbols.
    Returns dict of {symbol: DataFrame} sorted by date ascending.
    """
    cutoff = (datetime.now(tz=NST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    log.info("Loading price_history since %s...", cutoff)
    rows = run_raw_sql(
        """
        SELECT symbol, date, open, high, low, close, ltp, volume
        FROM price_history
        WHERE date >= %s
        ORDER BY symbol, date ASC
        """,
        (cutoff,),
    )

    if not rows:
        log.warning("No price history found")
        return {}

    def _f(v):
        try:
            return float(v) if v not in (None, "", "None") else None
        except Exception:
            return None

    sym_data: dict[str, list] = {}
    for r in rows:
        sym = str(r["symbol"]).upper()
        close = _f(r["close"]) or _f(r["ltp"])
        if close is None:
            continue
        sym_data.setdefault(sym, []).append({
            "date":   r["date"],
            "open":   _f(r["open"]) or close,
            "high":   _f(r["high"]) or close,
            "low":    _f(r["low"])  or close,
            "close":  close,
            "volume": _f(r["volume"]) or 0.0,
        })

    result = {}
    for sym, records in sym_data.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) >= MIN_HISTORY_DAYS:
            result[sym] = df

    log.info("Loaded %d symbols with %d+ days history", len(result), MIN_HISTORY_DAYS)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — IPO EXCLUSION
# ══════════════════════════════════════════════════════════════════════════════

def _get_ipo_cutoffs() -> dict[str, str]:
    """
    Returns {symbol: first_date} for all symbols.
    Used to exclude stocks listed within 60 trading days.
    """
    rows = run_raw_sql(
        """
        SELECT symbol, MIN(date) as first_date
        FROM price_history
        GROUP BY symbol
        """
    )
    return {str(r["symbol"]).upper(): str(r["first_date"]) for r in rows}


def _is_ipo_period(symbol: str, ipo_cutoffs: dict, trading_dates: list, days: int = 60) -> bool:
    """
    Returns True if symbol has fewer than `days` trading days of history.
    """
    first = ipo_cutoffs.get(symbol)
    if not first:
        return True
    sym_dates = [d for d in trading_dates if d >= first]
    return len(sym_dates) < days


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — COMPUTE INDICATORS + SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_spring_score(df: pd.DataFrame) -> dict:
    """
    Compute spring loading score for a symbol.
    Returns dict with score (0-6), criteria breakdown, and key values.
    Uses last row (today's EOD) for scoring.
    """
    result = {
        "score":            0,
        "criteria":         {},
        "rsi":              None,
        "adx":              None,
        "macd_hist":        None,
        "macd_improving":   False,
        "obv_slope_5d":     None,
        "bb_width":         None,
        "bb_squeeze":       False,
        "volume_ratio":     None,
        "volume_quiet":     False,
    }

    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = ta.rsi(close, length=14)
        if rsi is not None and not rsi.empty:
            rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            result["rsi"] = round(rsi_val, 2) if rsi_val else None
            if rsi_val and RSI_MIN <= rsi_val <= RSI_MAX:
                result["score"] += 1
                result["criteria"]["rsi"] = f"RSI={rsi_val:.1f} in [{RSI_MIN}-{RSI_MAX}] ✓"
            else:
                result["criteria"]["rsi"] = f"RSI={rsi_val:.1f} outside range ✗"

        # ── MACD histogram improving ──────────────────────────────────────────
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            hist = macd.get("MACDh_12_26_9")
            if hist is not None:
                last_hist = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else None
                result["macd_hist"] = round(last_hist, 4) if last_hist else None

                # Check if histogram is negative but improving for MACD_IMPROVE_DAYS
                if len(hist) >= MACD_IMPROVE_DAYS + 1:
                    recent = hist.iloc[-(MACD_IMPROVE_DAYS + 1):].dropna()
                    if len(recent) >= MACD_IMPROVE_DAYS + 1:
                        is_negative  = float(recent.iloc[-1]) < 0
                        is_improving = all(
                            recent.iloc[i] > recent.iloc[i - 1]
                            for i in range(1, len(recent))
                        )
                        result["macd_improving"] = is_negative and is_improving
                        if is_negative and is_improving:
                            result["score"] += 1
                            result["criteria"]["macd"] = f"MACD hist={last_hist:.4f} negative+improving {MACD_IMPROVE_DAYS}d ✓"
                        else:
                            result["criteria"]["macd"] = f"MACD hist={last_hist:.4f} not improving ✗"

        # ── OBV 5-day slope ───────────────────────────────────────────────────
        obv = ta.obv(close, volume)
        if obv is not None and len(obv) >= OBV_SLOPE_DAYS + 1:
            obv_clean = obv.dropna()
            if len(obv_clean) >= OBV_SLOPE_DAYS + 1:
                obv_start = float(obv_clean.iloc[-OBV_SLOPE_DAYS - 1])
                obv_end   = float(obv_clean.iloc[-1])
                obv_slope = (obv_end - obv_start) / max(abs(obv_start), 1)
                result["obv_slope_5d"] = round(obv_slope, 6)
                if obv_slope > 0:
                    result["score"] += 1
                    result["criteria"]["obv"] = f"OBV slope={obv_slope:.4f} positive ✓"
                else:
                    result["criteria"]["obv"] = f"OBV slope={obv_slope:.4f} negative ✗"

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col = adx_df.get("ADX_14")
            if adx_col is not None:
                adx_val = float(adx_col.iloc[-1]) if not pd.isna(adx_col.iloc[-1]) else None
                result["adx"] = round(adx_val, 2) if adx_val else None
                if adx_val and adx_val < ADX_MAX:
                    result["score"] += 1
                    result["criteria"]["adx"] = f"ADX={adx_val:.1f} < {ADX_MAX} ✓"
                else:
                    result["criteria"]["adx"] = f"ADX={adx_val:.1f} >= {ADX_MAX} ✗"

        # ── BB width squeeze ──────────────────────────────────────────────────
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty:
            bb_width = bb.get("BBB_20_2.0")
            if bb_width is not None and len(bb_width.dropna()) >= 10:
                curr_width = float(bb_width.iloc[-1])
                avg_width  = float(bb_width.iloc[-10:].mean())
                result["bb_width"] = round(curr_width, 4)
                squeeze = curr_width < avg_width * BB_SQUEEZE_RATIO
                result["bb_squeeze"] = squeeze
                if squeeze:
                    result["score"] += 1
                    result["criteria"]["bb"] = f"BB width={curr_width:.3f} < {BB_SQUEEZE_RATIO}x avg={avg_width:.3f} ✓"
                else:
                    result["criteria"]["bb"] = f"BB width={curr_width:.3f} not contracting ✗"

        # ── Volume quiet ──────────────────────────────────────────────────────
        if len(volume) >= 20:
            curr_vol  = float(volume.iloc[-1])
            avg_vol   = float(volume.iloc[-21:-1].mean())
            vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
            result["volume_ratio"] = round(vol_ratio, 3)
            result["volume_quiet"] = vol_ratio < VOLUME_RATIO_MAX
            if vol_ratio < VOLUME_RATIO_MAX:
                result["score"] += 1
                result["criteria"]["volume"] = f"Volume ratio={vol_ratio:.2f} < {VOLUME_RATIO_MAX} ✓"
            else:
                result["criteria"]["volume"] = f"Volume ratio={vol_ratio:.2f} not quiet ✗"

    except Exception as exc:
        log.warning("_compute_spring_score failed: %s", exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FORMAT TELEGRAM MESSAGE
# ══════════════════════════════════════════════════════════════════════════════

def _format_alert(symbol: str, sector: str, ltp: float, scored: dict) -> str:
    today = datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M")
    criteria_lines = "\n".join(
        f"  {'✅' if '✓' in v else '❌'} {v}"
        for v in scored["criteria"].values()
    )
    return (
        f"🌱 *SPRING LOADING DETECTED* — {symbol}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Sector:   {sector}\n"
        f"LTP:      NPR {ltp:,.2f}\n"
        f"Score:    {scored['score']}/6\n\n"
        f"*Criteria:*\n{criteria_lines}\n\n"
        f"*Key Values:*\n"
        f"  RSI:        {scored['rsi']}\n"
        f"  ADX:        {scored['adx']}\n"
        f"  MACD Hist:  {scored['macd_hist']}\n"
        f"  OBV Slope:  {scored['obv_slope_5d']}\n"
        f"  Vol Ratio:  {scored['volume_ratio']}\n\n"
        f"⚠️ Research signal only — not a BUY signal.\n"
        f"Watch for volume blast (>3x) as entry confirmation.\n"
        f"_{today} NST_"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WRITE TO DB
# ══════════════════════════════════════════════════════════════════════════════

def _write_finding(symbol: str, sector: str, ltp: float, scored: dict, today: str) -> None:
    """
    Write spring loading finding to settings as JSON list (appended).
    Uses settings table key SPRING_LOADING_FINDINGS_{date}.
    Fails silently.
    """
    try:
        key = f"SPRING_LOADING_{today}"
        existing_raw = get_setting(key, "[]")
        try:
            existing = json.loads(existing_raw)
        except Exception:
            existing = []

        existing.append({
            "symbol":       symbol,
            "sector":       sector,
            "ltp":          ltp,
            "score":        scored["score"],
            "rsi":          scored["rsi"],
            "adx":          scored["adx"],
            "macd_hist":    scored["macd_hist"],
            "obv_slope_5d": scored["obv_slope_5d"],
            "volume_ratio": scored["volume_ratio"],
            "criteria":     scored["criteria"],
            "timestamp":    datetime.now(tz=NST).strftime("%Y-%m-%d %H:%M:%S"),
        })

        upsert_row(
            "settings",
            {"key": key, "value": json.dumps(existing), "set_by": "spring_loading_screener"},
            conflict_columns=["key"],
        )
        log.info("Written finding for %s (score=%d) to settings key %s", symbol, scored["score"], key)
    except Exception as exc:
        log.warning("_write_finding failed for %s: %s", symbol, exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_screener() -> list[dict]:
    """
    Main entry point. Scans all symbols, scores, alerts, logs.
    Returns list of findings with score >= LOG_THRESHOLD.
    """
    today = datetime.now(tz=NST).strftime("%Y-%m-%d")
    log.info("=" * 60)
    log.info("spring_loading_screener — %s", today)
    log.info("DRY_RUN=%s", DRY_RUN)

    # Load sector map
    sector_map: dict[str, str] = {}
    try:
        from sheets import read_tab
        rows = read_tab("share_sectors")
        sector_map = {
            r["symbol"].upper(): (r.get("sectorname") or "others").lower()
            for r in rows if r.get("symbol")
        }
    except Exception as exc:
        log.warning("Could not load sector map: %s", exc)

    # Load latest LTP
    ltp_map: dict[str, float] = {}
    try:
        ltp_rows = run_raw_sql(
            """
            SELECT DISTINCT ON (symbol) symbol, close, ltp
            FROM price_history
            ORDER BY symbol, date DESC
            """
        )
        for r in ltp_rows:
            sym = str(r["symbol"]).upper()
            try:
                ltp_map[sym] = float(r["close"] or r["ltp"] or 0)
            except Exception:
                pass
    except Exception as exc:
        log.warning("Could not load LTP map: %s", exc)

    # Load IPO cutoffs
    ipo_cutoffs = _get_ipo_cutoffs()

    # Load price history
    sym_data = _load_price_history(LOOKBACK_DAYS)
    if not sym_data:
        log.error("No price data — aborting")
        return []

    # Get sorted trading dates for IPO check
    all_trading_dates = sorted(set(
        str(d.date()) for df in sym_data.values() for d in df["date"].tolist()
    ))

    findings   = []
    alerts_sent = 0
    logged      = 0

    log.info("Scanning %d symbols...", len(sym_data))

    for sym, df in sym_data.items():
        # IPO exclusion
        if _is_ipo_period(sym, ipo_cutoffs, all_trading_dates, days=60):
            continue

        scored  = _compute_spring_score(df)
        score   = scored["score"]
        sector  = sector_map.get(sym, "others")
        ltp     = ltp_map.get(sym, 0.0)

        if SHOW_ALL:
            log.info(
                "%s [%s] score=%d/6 | RSI=%.1f ADX=%.1f",
                sym, sector, score,
                scored["rsi"] or 0,
                scored["adx"] or 0,
            )

        if score < LOG_THRESHOLD:
            continue

        log.info(
            "FINDING: %s [%s] score=%d/6 LTP=%.2f | RSI=%.1f ADX=%.1f OBV_slope=%.4f",
            sym, sector, score, ltp,
            scored["rsi"]         or 0,
            scored["adx"]         or 0,
            scored["obv_slope_5d"] or 0,
        )

        findings.append({
            "symbol": sym,
            "sector": sector,
            "ltp":    ltp,
            "score":  score,
            "scored": scored,
        })

        # Write to DB
        if not DRY_RUN:
            _write_finding(sym, sector, ltp, scored, today)
            logged += 1

        # Telegram alert if all 6 criteria met
        if score >= ALERT_THRESHOLD:
            msg = _format_alert(sym, sector, ltp, scored)
            if not DRY_RUN:
                _send_admin_only(msg)
                alerts_sent += 1
                log.info("Telegram alert sent for %s", sym)
            else:
                log.info("[DRY_RUN] Would alert: %s", sym)
                print(f"\n--- TELEGRAM PREVIEW ---\n{msg}\n")

    # Summary
    log.info(
        "screener done: %d scanned | %d findings (score>=%d) | %d alerts | %d logged",
        len(sym_data), len(findings), LOG_THRESHOLD, alerts_sent, logged,
    )

    if findings and not DRY_RUN:
        summary = (
            f"🌱 *Spring Loading Screener* — {today}\n"
            f"Scanned: {len(sym_data)} symbols\n"
            f"Findings (≥{LOG_THRESHOLD}/6): {len(findings)}\n"
            f"Full alerts (6/6): {alerts_sent}\n\n"
            + "\n".join(
                f"  {f['symbol']} [{f['sector']}] {f['score']}/6 — RSI {f['scored']['rsi']}"
                for f in sorted(findings, key=lambda x: x["score"], reverse=True)
            )
        )
        _send_admin_only(summary)

    return findings


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SPRING_SCREENER] %(levelname)s: %(message)s",
    )

    try:
        from log_config import attach_file_handler
        attach_file_handler(__name__)
    except Exception:
        pass

    results = run_screener()

    print(f"\n{'='*60}")
    print(f"Spring Loading Screener Results")
    print(f"{'='*60}")
    if results:
        for f in sorted(results, key=lambda x: x["score"], reverse=True):
            print(f"\n{f['symbol']} [{f['sector']}] score={f['score']}/6 LTP={f['ltp']:.2f}")
            for k, v in f["scored"]["criteria"].items():
                print(f"  {v}")
    else:
        print("No findings above threshold today.")
    print()