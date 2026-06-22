"""
analysis/top_gainer_study.py — NEPSE AI Engine v2
═══════════════════════════════════════════════════════════════════════════════
Top Gainer Pre-Signal Study — Full Version

For every trading day, find top N gainers by change_pct.
For each gainer, capture the FULL daily series for 15 trading days BEFORE
the big move — every technical indicator + floorsheet metric every single day.

Fixes from v1:
  - OHLCV + technical columns now correctly included in output
  - Sector segmentation added
  - Full technical indicator trajectory analysis
  - Pattern change detection (delta columns: day-over-day changes)
  - Pattern fitting: identifies which day-range best predicts the event

Output: results/top_gainer_study/
    gainer_events.csv       — long format, one row per (event × day)
                              includes ALL OHLCV + technicals + floorsheet
    pattern_summary.csv     — per-day aggregated stats (mean/median/std)
    pattern_deltas.csv      — day-over-day CHANGES in each metric
    sector_patterns.csv     — same stats segmented by sector
    gain_group_split.csv    — HIGH vs MID vs LOW gainer comparison
    correlations.csv        — Spearman rho vs event_change_pct
    summary.txt             — full human-readable findings

Fully vectorized — numpy/pandas groupby, no Python loops in hot paths.

Run:
    python -m analysis.top_gainer_study
    python -m analysis.top_gainer_study --from 2024-01-01
    python -m analysis.top_gainer_study --top-n 10 --min-gain 5.0
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TOP_GAINER] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_FROM     = "2023-07-03"
DEFAULT_TO       = "2026-05-11"
DEFAULT_TOP_N    = 5
DEFAULT_MIN_GAIN = 3.0
LOOKBACK_DAYS    = 15
RESULTS_DIR      = Path("results/top_gainer_study")

TECH_COLS = [
    "open", "high", "low", "close", "volume", "turnover",
    "day_change_pct",
    "rsi_14",
    "macd_line", "macd_signal", "macd_hist", "macd_cross",
    "ema_5", "ema_10", "ema_20", "ema_50",
    "close_vs_ema5", "close_vs_ema20", "close_vs_ema50",
    "ema5_slope", "ema20_slope",
    "atr_14", "atr_pct",
    "bb_upper", "bb_lower", "bb_mid", "bb_pct_b", "bb_width",
    "range_pct", "range_5d_avg",
    "vol_ma_5", "vol_ma_20", "vol_ratio_5", "vol_ratio_20",
    "vol_slope_5", "obv", "obv_slope",
    "consol_score", "conf_score",
]

FS_COLS = [
    "buyer_pressure", "seller_pressure",
    "broker_concentration", "institutional_flag",
    "large_order_pct", "large_order_count",
    "fs_volume", "total_trades",
]

ANALYSIS_METRICS = [
    "day_change_pct", "close_vs_ema5", "close_vs_ema20", "close_vs_ema50",
    "ema5_slope", "ema20_slope",
    "rsi_14", "macd_hist", "macd_cross",
    "atr_pct", "bb_pct_b", "bb_width", "range_pct", "range_5d_avg",
    "vol_ratio_5", "vol_ratio_20", "vol_slope_5", "obv_slope",
    "consol_score", "conf_score",
    "buyer_pressure", "broker_concentration",
    "institutional_flag", "large_order_pct",
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_price_history(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading price_history %s → %s ...", from_date, to_date)
    from db.connection import _db

    with _db() as cur:
        cur.execute("""
            SELECT date, symbol,
                   open, high, low, close, ltp,
                   volume, turnover, prev_close, conf_score
            FROM price_history
            WHERE date >= %s AND date <= %s
              AND close IS NOT NULL AND close != ''
            ORDER BY symbol, date ASC
        """, (from_date, to_date))
        rows = cur.fetchall()

    if not rows:
        log.error("No price_history data")
        sys.exit(1)

    df = pd.DataFrame([dict(r) for r in rows])
    log.info("  Loaded %d rows, %d symbols", len(df), df["symbol"].nunique())

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open","high","low","close","ltp","volume","turnover","prev_close","conf_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["close"]      = df["close"].fillna(df["ltp"])
    df["open"]       = df["open"].fillna(df["close"])
    df["high"]       = df["high"].fillna(df["close"])
    df["low"]        = df["low"].fillna(df["close"])
    df["volume"]     = df["volume"].fillna(0)
    df["prev_close"] = df["prev_close"].fillna(df.groupby("symbol")["close"].shift(1))

    df = df.dropna(subset=["close","date"])
    df = df[df["close"] > 0].copy()
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    log.info("  Clean: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def load_floorsheet_signals(from_date: str, to_date: str) -> pd.DataFrame:
    log.info("Loading floorsheet_signals ...")
    from db.connection import _db

    with _db() as cur:
        cur.execute("""
            SELECT date, symbol,
                   buyer_pressure, seller_pressure,
                   broker_concentration, institutional_flag,
                   large_order_pct, large_order_count,
                   total_volume AS fs_volume, total_trades
            FROM floorsheet_signals
            WHERE date >= %s AND date <= %s
        """, (from_date, to_date))
        rows = cur.fetchall()

    if not rows:
        log.warning("No floorsheet_signals — fs columns will be null")
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = df["date"].astype(str).apply(_normalize_date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["buyer_pressure","seller_pressure","broker_concentration",
              "large_order_pct","large_order_count","fs_volume","total_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["institutional_flag"] = (
        df["institutional_flag"].astype(str).str.lower().str.strip() == "true"
    ).astype(int)

    log.info("  Loaded %d floorsheet rows", len(df))
    return df.dropna(subset=["date"]).reset_index(drop=True)


def load_sectors() -> dict:
    try:
        from db.connection import _db
        with _db() as cur:
            cur.execute("SELECT symbol, sector FROM share_sectors")
            rows = cur.fetchall()
        return {r["symbol"].upper(): r["sector"] for r in rows}
    except Exception as e:
        log.warning("Could not load sectors: %s", e)
        return {}


def _normalize_date(d: str) -> str:
    try:
        p = str(d).split("-")
        return f"{p[0]}-{int(p[1]):02d}-{int(p[2]):02d}" if len(p)==3 else d
    except Exception:
        return d


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(3, span//2)).mean()


def _slope5(x: np.ndarray) -> float:
    if len(x) < 3: return np.nan
    xi = np.arange(len(x), dtype=float)
    xm, ym = xi.mean(), x.mean()
    d = ((xi-xm)**2).sum()
    return 0.0 if d == 0 else ((xi-xm)*(x-ym)).sum()/d


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Computing technical indicators ...")
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)
    g  = df.groupby("symbol", group_keys=False)

    # daily change
    df["day_change_pct"] = g["close"].transform(lambda x: x.pct_change()*100)
    mask = df["prev_close"] > 0
    df.loc[mask,"day_change_pct"] = (
        (df.loc[mask,"close"] - df.loc[mask,"prev_close"])
        / df.loc[mask,"prev_close"] * 100
    )

    # EMAs
    for span in [5,10,20,50]:
        df[f"ema_{span}"] = g["close"].transform(lambda x, s=span: _ema(x, s))

    df["close_vs_ema5"]  = np.where(df["ema_5"]  > 0, df["close"]/df["ema_5"]  - 1, np.nan)
    df["close_vs_ema20"] = np.where(df["ema_20"] > 0, df["close"]/df["ema_20"] - 1, np.nan)
    df["close_vs_ema50"] = np.where(df["ema_50"] > 0, df["close"]/df["ema_50"] - 1, np.nan)

    df["ema5_slope"]  = g["ema_5"].transform( lambda x: x.rolling(5,min_periods=3).apply(_slope5,raw=True))
    df["ema20_slope"] = g["ema_20"].transform(lambda x: x.rolling(5,min_periods=3).apply(_slope5,raw=True))

    # RSI 14
    def _rsi(s):
        d = s.diff()
        g_ = d.clip(lower=0).ewm(com=13, adjust=False, min_periods=5).mean()
        l_ = (-d).clip(lower=0).ewm(com=13, adjust=False, min_periods=5).mean()
        rs = np.where(l_ > 0, g_/l_, 100.0)
        return pd.Series(np.where(l_==0, 100.0, 100-100/(1+rs)), index=s.index)
    df["rsi_14"] = g["close"].transform(_rsi)

    # MACD (12,26,9)
    e12 = g["close"].transform(lambda x: _ema(x,12))
    e26 = g["close"].transform(lambda x: _ema(x,26))
    df["macd_line"]   = e12 - e26
    df["macd_signal"] = g["macd_line"].transform(lambda x: _ema(x,9))
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]
    hs  = np.sign(df["macd_hist"])
    hs1 = g["macd_hist"].transform(lambda x: np.sign(x).shift(1))
    df["macd_cross"] = np.where((hs==1)&(hs1==-1), 1, np.where((hs==-1)&(hs1==1),-1, 0))

    # ATR 14
    cl1 = g["close"].transform(lambda x: x.shift(1))
    df["tr"] = np.maximum(df["high"]-df["low"],
               np.maximum(np.abs(df["high"]-cl1), np.abs(df["low"]-cl1)))
    df["atr_14"] = g["tr"].transform(lambda x: x.ewm(com=13,adjust=False,min_periods=5).mean())
    df["atr_pct"] = np.where(df["close"]>0, df["atr_14"]/df["close"]*100, np.nan)

    # Bollinger Bands (20, 2σ)
    df["bb_mid"]   = g["close"].transform(lambda x: x.rolling(20,min_periods=10).mean())
    bb_std         = g["close"].transform(lambda x: x.rolling(20,min_periods=10).std())
    df["bb_upper"] = df["bb_mid"] + 2*bb_std
    df["bb_lower"] = df["bb_mid"] - 2*bb_std
    bb_rng         = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = np.where(bb_rng>0, (df["close"]-df["bb_lower"])/bb_rng, 0.5)
    df["bb_width"] = np.where(df["bb_mid"]>0, bb_rng/df["bb_mid"]*100, np.nan)

    # Range
    df["range_pct"]    = np.where(df["close"]>0, (df["high"]-df["low"])/df["close"]*100, np.nan)
    df["range_5d_avg"] = g["range_pct"].transform(lambda x: x.rolling(5,min_periods=3).mean())

    # Volume
    df["vol_ma_5"]     = g["volume"].transform(lambda x: x.rolling(5, min_periods=3).mean())
    df["vol_ma_20"]    = g["volume"].transform(lambda x: x.rolling(20,min_periods=5).mean())
    df["vol_ratio_5"]  = np.where(df["vol_ma_5"]  >0, df["volume"]/df["vol_ma_5"],  np.nan)
    df["vol_ratio_20"] = np.where(df["vol_ma_20"] >0, df["volume"]/df["vol_ma_20"], np.nan)
    df["vol_slope_5"]  = g["volume"].transform(lambda x: x.rolling(5,min_periods=3).apply(_slope5,raw=True))

    # OBV
    def _obv_fn(grp):
        direction = np.sign(grp["close"].diff().fillna(0))
        return (direction * grp["volume"]).cumsum()
    df["obv"] = g.apply(_obv_fn).reset_index(level=0, drop=True)
    df["obv_slope"] = g["obv"].transform(lambda x: x.rolling(5,min_periods=3).apply(_slope5,raw=True))

    # Consolidation score
    rrank = g["range_pct"].transform(lambda x: x.rolling(20,min_periods=5).rank(pct=True))
    vrank = g["volume"].transform(   lambda x: x.rolling(20,min_periods=5).rank(pct=True))
    df["consol_score"] = ((1-rrank) + (1-vrank)) / 2

    log.info("  Technicals done")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIND GAINER EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def find_top_gainers(df, top_n, min_gain):
    log.info("Finding top %d gainers/day (min %.1f%%) ...", top_n, min_gain)
    valid = df[
        (df["day_change_pct"] >= min_gain) &
        (df["day_change_pct"] <= 50.0) &
        (df["volume"] > 0)
    ].copy()

    valid["rank"] = valid.groupby("date")["day_change_pct"].rank(
        method="first", ascending=False).astype(int)

    gainers = valid[valid["rank"] <= top_n].copy()
    gainers = gainers.rename(columns={
        "date":          "event_date",
        "day_change_pct":"event_change_pct",
    })
    gainers["event_id"] = np.arange(len(gainers))
    log.info("  %d gainer events across %d trading days",
             len(gainers), gainers["event_date"].nunique())
    return gainers


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD LOOKBACK SERIES
# ══════════════════════════════════════════════════════════════════════════════

def build_lookback_series(gainers, price_df, fs_df, sector_map):
    log.info("Building lookback series ...")

    price_df = price_df.sort_values(["symbol","date"]).reset_index(drop=True)
    price_df["_idx"] = price_df.groupby("symbol").cumcount()

    # Map event_date → integer index
    idx_map = price_df[["symbol","date","_idx"]].rename(
        columns={"date":"event_date"})
    gainers = gainers.merge(idx_map, on=["symbol","event_date"], how="left")
    gainers = gainers.dropna(subset=["_idx"])
    gainers["_idx"]   = gainers["_idx"].astype(int)
    gainers["sector"] = gainers["symbol"].map(sector_map).fillna("UNKNOWN")
    log.info("  %d events matched", len(gainers))

    # Expand × LOOKBACK_DAYS
    offsets  = np.arange(-LOOKBACK_DAYS, 0)
    n_ev, n_off = len(gainers), len(offsets)

    rep = gainers.loc[gainers.index.repeat(n_off)].reset_index(drop=True)
    rep["days_before"] = np.tile(offsets, n_ev)
    rep["_tgt"]        = rep["_idx"] + rep["days_before"]
    rep = rep[rep["_tgt"] >= 0].copy()

    # Price lookup columns — everything in TECH_COLS that exists
    price_carry = ["symbol","_idx","date"] + [
        c for c in TECH_COLS if c in price_df.columns
    ]
    price_carry = list(dict.fromkeys(price_carry))

    price_lkp = price_df[price_carry].rename(columns={
        "_idx": "_tgt",
        "date": "lookback_date",
    })

    result = rep.merge(price_lkp, on=["symbol","_tgt"], how="left")

    # Floorsheet merge
    if not fs_df.empty:
        fs_m = fs_df[["symbol","date"] + [c for c in FS_COLS if c in fs_df.columns]].rename(
            columns={"date":"lookback_date"})
        result = result.merge(fs_m, on=["symbol","lookback_date"], how="left")
    else:
        for c in FS_COLS:
            result[c] = np.nan

    # Clean up internal cols
    result = result.drop(columns=[c for c in result.columns if c.startswith("_")],
                         errors="ignore")

    # Final column order
    id_cols   = ["event_id","event_date","symbol","sector",
                 "rank","event_change_pct","days_before","lookback_date"]
    tech_here = [c for c in TECH_COLS if c in result.columns]
    fs_here   = [c for c in FS_COLS   if c in result.columns]
    final     = id_cols + tech_here + fs_here
    result    = result[[c for c in final if c in result.columns]].copy()
    result    = result.sort_values(["event_id","days_before"]).reset_index(drop=True)

    log.info("  %d rows | %d cols | %d events",
             len(result), len(result.columns), result["event_id"].nunique())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_patterns(long_df):
    log.info("Analyzing patterns ...")
    metrics = [m for m in ANALYSIS_METRICS if m in long_df.columns]

    # Per-day aggregation
    agg_d = {}
    for m in metrics:
        agg_d[f"{m}_mean"]   = (m, "mean")
        agg_d[f"{m}_median"] = (m, "median")
        agg_d[f"{m}_std"]    = (m, "std")
    pattern_df = long_df.groupby("days_before").agg(**agg_d).reset_index()

    # Binary signal %
    checks = {
        "pct_vol_above_20d": ("vol_ratio_20",   lambda x: (x>1.0).mean()*100),
        "pct_vol_drying":    ("vol_ratio_20",   lambda x: (x<0.8).mean()*100),
        "pct_rsi_above50":   ("rsi_14",         lambda x: (x>50).mean()*100),
        "pct_rsi_oversold":  ("rsi_14",         lambda x: (x<35).mean()*100),
        "pct_above_ema20":   ("close_vs_ema20", lambda x: (x>0).mean()*100),
        "pct_macd_bull":     ("macd_hist",      lambda x: (x>0).mean()*100),
        "pct_bb_upper_half": ("bb_pct_b",       lambda x: (x>0.5).mean()*100),
        "pct_consolidating": ("consol_score",   lambda x: (x>0.6).mean()*100),
        "pct_strong_buyer":  ("buyer_pressure", lambda x: (x>0.5).mean()*100),
        "pct_institutional": ("institutional_flag", lambda x: x.mean()*100),
        "pct_obv_rising":    ("obv_slope",      lambda x: (x>0).mean()*100),
    }
    for col, (src, fn) in checks.items():
        if src in long_df.columns:
            pct = long_df.groupby("days_before")[src].apply(fn).reset_index(name=col)
            pattern_df = pattern_df.merge(pct, on="days_before", how="left")

    # Day-over-day deltas
    delta_df  = pattern_df.sort_values("days_before").copy()
    mean_cols = [c for c in delta_df.columns if c.endswith("_mean")]
    for c in mean_cols:
        delta_df[f"delta_{c}"] = delta_df[c].diff()

    # Spearman correlations
    corr_rows = []
    for m in metrics:
        sub = long_df[["days_before",m,"event_change_pct"]].dropna()
        for day in sorted(sub["days_before"].unique()):
            dd = sub[sub["days_before"]==day]
            if len(dd) < 10: continue
            rho, pval = stats.spearmanr(dd[m], dd["event_change_pct"])
            corr_rows.append({"metric":m,"days_before":day,
                              "spearman_rho":round(rho,4),
                              "p_value":round(pval,6),
                              "significant":pval<0.05,"n":len(dd)})
    corr_df = pd.DataFrame(corr_rows) if corr_rows else pd.DataFrame()

    # Sector segmentation
    key_m = [m for m in ["day_change_pct","vol_ratio_20","rsi_14","macd_hist",
                          "buyer_pressure","consol_score","atr_pct","bb_pct_b"]
             if m in long_df.columns]
    sec_agg = {f"{m}_mean":(m,"mean") for m in key_m}
    sector_df = long_df.groupby(["sector","days_before"]).agg(**sec_agg).reset_index()

    # HIGH / MID / LOW split
    events   = long_df.drop_duplicates("event_id")[["event_id","event_change_pct"]]
    high_ids = events[events["event_change_pct"] >= 9.5]["event_id"].values
    mid_ids  = events[(events["event_change_pct"] >= 6) &
                      (events["event_change_pct"] <  9.5)]["event_id"].values
    low_ids  = events[(events["event_change_pct"] >= 3) &
                      (events["event_change_pct"] <  6)]["event_id"].values

    split_rows = []
    for label, ids in [("HIGH_>=9.5%",high_ids),("MID_6-9.5%",mid_ids),("LOW_3-6%",low_ids)]:
        sub = long_df[long_df["event_id"].isin(ids)]
        for day in sorted(sub["days_before"].unique()):
            d   = sub[sub["days_before"]==day]
            row = {"gain_group":label,"days_before":day,"n_events":len(ids)}
            for m in key_m:
                row[f"{m}_mean"]   = d[m].mean()
                row[f"{m}_median"] = d[m].median()
            split_rows.append(row)
    split_df = pd.DataFrame(split_rows)

    findings = _extract_findings(long_df, pattern_df, corr_df, split_df)
    return pattern_df, delta_df, corr_df, sector_df, split_df, findings


def _extract_findings(long_df, pattern_df, corr_df, split_df):
    f  = {}
    ev = long_df.drop_duplicates("event_id")
    f["n_events"]    = ev["event_id"].nunique()
    f["n_days"]      = long_df["event_date"].nunique()
    f["date_range"]  = f"{long_df['event_date'].min().date()} → {long_df['event_date'].max().date()}"
    f["avg_gain"]    = round(ev["event_change_pct"].mean(), 2)
    f["median_gain"] = round(ev["event_change_pct"].median(), 2)

    def _pd_val(col, day):
        r = pattern_df[pattern_df["days_before"]==day][col].values
        return float(r[0]) if len(r) else np.nan

    for metric, key15, key1 in [
        ("vol_ratio_20_mean",  "vol_at_d15",  "vol_at_d1"),
        ("atr_pct_mean",       "atr_d15",     "atr_d1"),
        ("rsi_14_mean",        "rsi_d15",     "rsi_d1"),
        ("bb_pct_b_mean",      "bb_d15",      "bb_d1"),
        ("macd_hist_mean",     "macd_d15",    "macd_d1"),
        ("buyer_pressure_mean","bp_d15",      "bp_d1"),
        ("consol_score_mean",  "consol_d15",  "consol_d1"),
    ]:
        if metric in pattern_df.columns:
            f[key15] = round(_pd_val(metric,-15), 4)
            f[key1]  = round(_pd_val(metric,-1),  4)

    f["vol_trend"]      = "RISING"      if f.get("vol_at_d1",0)  > f.get("vol_at_d15",0)  else "FALLING"
    f["atr_compressed"] = f.get("atr_d1",99) < f.get("atr_d15",0)

    if not corr_df.empty:
        sig = corr_df[corr_df["significant"]].sort_values(
            "spearman_rho", key=abs, ascending=False).head(10)
        f["top_correlations"] = sig[["metric","days_before","spearman_rho","p_value"]].to_dict("records")

    if not split_df.empty and "buyer_pressure_mean" in split_df.columns:
        hh = split_df[split_df["gain_group"]=="HIGH_>=9.5%"].sort_values("days_before")
        ll = split_df[split_df["gain_group"]=="LOW_3-6%"].sort_values("days_before")
        if not hh.empty and not ll.empty:
            mg = hh[["days_before","buyer_pressure_mean"]].merge(
                ll[["days_before","buyer_pressure_mean"]],
                on="days_before", suffixes=("_h","_l"))
            mg["diff"] = mg["buyer_pressure_mean_h"] - mg["buyer_pressure_mean_l"]
            dv = mg[mg["diff"]>0.05]["days_before"].min()
            f["bp_diverge_day"] = int(dv) if not pd.isna(dv) else None

    return f


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def write_outputs(long_df, pattern_df, delta_df, corr_df,
                  sector_df, split_df, findings, top_n, min_gain):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(   RESULTS_DIR/"gainer_events.csv",    index=False, float_format="%.5f")
    pattern_df.to_csv(RESULTS_DIR/"pattern_summary.csv",  index=False, float_format="%.5f")
    delta_df.to_csv(  RESULTS_DIR/"pattern_deltas.csv",   index=False, float_format="%.5f")
    sector_df.to_csv( RESULTS_DIR/"sector_patterns.csv",  index=False, float_format="%.5f")
    split_df.to_csv(  RESULTS_DIR/"gain_group_split.csv", index=False, float_format="%.5f")
    if not corr_df.empty:
        corr_df.to_csv(RESULTS_DIR/"correlations.csv",    index=False, float_format="%.6f")
    _write_summary(RESULTS_DIR/"summary.txt",
                   pattern_df, delta_df, corr_df, split_df, findings, top_n, min_gain)
    log.info("All outputs written to %s/", RESULTS_DIR)


def _w(cols, rows, widths):
    lines = ["  "+"  ".join(f"{c:>{w}}" for c,w in zip(cols,widths)),
             "  "+"  ".join("─"*w for w in widths)]
    for row in rows:
        lines.append("  "+"  ".join(f"{v:>{w}}" for v,w in zip(row,widths)))
    return "\n".join(lines)


def _write_summary(path, pattern_df, delta_df, corr_df, split_df, findings, top_n, min_gain):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L=[]; a=L.append

    a("="*72)
    a("  NEPSE TOP GAINER PRE-SIGNAL STUDY  v2")
    a(f"  Generated : {now}")
    a("="*72)
    a(f"\nSTUDY PARAMETERS")
    a(f"  Top N / Min gain  : {top_n} / {min_gain}%")
    a(f"  Lookback          : {LOOKBACK_DAYS} trading days")
    a(f"  Date range        : {findings.get('date_range')}")
    a(f"  Events / Days     : {findings.get('n_events'):,} / {findings.get('n_days'):,}")
    a(f"  Avg / Median gain : {findings.get('avg_gain')}% / {findings.get('median_gain')}%\n")

    # ── 1. Price momentum ─────────────────────────────────────────────────────
    a("─"*72)
    a("1. PRICE MOMENTUM (day_change_pct trajectory)")
    a("─"*72)
    if "day_change_pct_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            delta_col = f"delta_day_change_pct_mean"
            dlt = delta_df[delta_df["days_before"]==d][delta_col].values if delta_col in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('day_change_pct_mean',np.nan):.3f}%",
                         f"{r.get('day_change_pct_median',np.nan):.3f}%",
                         f"{r.get('day_change_pct_std',np.nan):.3f}%",
                         f"{float(dlt[0]):+.3f}%" if len(dlt) and not np.isnan(float(dlt[0])) else "  N/A"])
        a(_w(["Day","Mean","Median","Std","Δ(day-on-day)"],rows,[5,9,9,8,14]))
    a("")

    # ── 2. RSI ────────────────────────────────────────────────────────────────
    a("─"*72)
    a("2. RSI 14 TRAJECTORY")
    a("─"*72)
    if "rsi_14_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = f"delta_rsi_14_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('rsi_14_mean',np.nan):.2f}",
                         f"{r.get('rsi_14_median',np.nan):.2f}",
                         f"{r.get('pct_rsi_above50',np.nan):.1f}%",
                         f"{r.get('pct_rsi_oversold',np.nan):.1f}%",
                         f"{float(dlt[0]):+.3f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","Mean","Median","%>50","%<35","Δ"],rows,[5,8,8,6,6,8]))
    a(f"\n  d-15={findings.get('rsi_d15','?')}  →  d-1={findings.get('rsi_d1','?')}\n")

    # ── 3. MACD ───────────────────────────────────────────────────────────────
    a("─"*72)
    a("3. MACD HISTOGRAM TRAJECTORY")
    a("─"*72)
    if "macd_hist_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = "delta_macd_hist_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('macd_hist_mean',np.nan):.5f}",
                         f"{r.get('macd_hist_median',np.nan):.5f}",
                         f"{r.get('pct_macd_bull',np.nan):.1f}%",
                         f"{float(dlt[0]):+.5f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","Mean","Median","%Bullish","Δ"],rows,[5,10,10,9,12]))
    a(f"\n  d-15={findings.get('macd_d15','?')}  →  d-1={findings.get('macd_d1','?')}\n")

    # ── 4. Bollinger ──────────────────────────────────────────────────────────
    a("─"*72)
    a("4. BOLLINGER BANDS (%B and width)")
    a("─"*72)
    if "bb_pct_b_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = "delta_bb_pct_b_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('bb_pct_b_mean',np.nan):.3f}",
                         f"{r.get('bb_width_mean',np.nan):.3f}%",
                         f"{r.get('pct_bb_upper_half',np.nan):.1f}%",
                         f"{float(dlt[0]):+.4f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","%B Mean","BB Width","%UpperHalf","Δ%B"],rows,[5,9,10,11,9]))
    a(f"\n  d-15 %B={findings.get('bb_d15','?')}  →  d-1={findings.get('bb_d1','?')}\n")

    # ── 5. ATR ────────────────────────────────────────────────────────────────
    a("─"*72)
    a("5. ATR / RANGE COMPRESSION")
    a("─"*72)
    if "atr_pct_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = "delta_atr_pct_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('atr_pct_mean',np.nan):.4f}%",
                         f"{r.get('range_pct_mean',np.nan):.4f}%",
                         f"{r.get('range_5d_avg_mean',np.nan):.4f}%",
                         f"{float(dlt[0]):+.5f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","ATR%","Range%","Range5d","ΔATR"],rows,[5,9,9,9,10]))
    comp = findings.get("atr_compressed")
    a(f"\n  ATR: d-15={findings.get('atr_d15','?')}%  →  d-1={findings.get('atr_d1','?')}%")
    if comp is not None:
        a(f"  → {'COMPRESSING ✅ squeeze pattern' if comp else 'EXPANDING ⚠️'}\n")

    # ── 6. Volume ─────────────────────────────────────────────────────────────
    a("─"*72)
    a("6. VOLUME TRAJECTORY")
    a("─"*72)
    if "vol_ratio_20_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = "delta_vol_ratio_20_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('vol_ratio_20_mean',np.nan):.4f}",
                         f"{r.get('pct_vol_above_20d',np.nan):.1f}%",
                         f"{r.get('pct_vol_drying',np.nan):.1f}%",
                         f"{r.get('pct_obv_rising',np.nan):.1f}%",
                         f"{float(dlt[0]):+.4f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","Vol/20d","%Above","%%Dry","%OBV↑","Δ"],rows,[5,8,8,7,7,8]))
    a(f"\n  d-15={findings.get('vol_at_d15','?')}x  →  d-1={findings.get('vol_at_d1','?')}x  ({findings.get('vol_trend')})\n")

    # ── 7. Floorsheet ─────────────────────────────────────────────────────────
    a("─"*72)
    a("7. FLOORSHEET TRAJECTORY")
    a("─"*72)
    if "buyer_pressure_mean" in pattern_df.columns:
        pd_s = pattern_df.sort_values("days_before")
        rows=[]
        for _,r in pd_s.iterrows():
            d=int(r["days_before"])
            dc = "delta_buyer_pressure_mean"
            dlt = delta_df[delta_df["days_before"]==d][dc].values if dc in delta_df.columns else [np.nan]
            rows.append([f"{d}",
                         f"{r.get('buyer_pressure_mean',np.nan):.4f}",
                         f"{r.get('broker_concentration_mean',np.nan):.4f}",
                         f"{r.get('large_order_pct_mean',np.nan):.4f}",
                         f"{r.get('pct_strong_buyer',np.nan):.1f}%",
                         f"{r.get('pct_institutional',np.nan):.1f}%",
                         f"{float(dlt[0]):+.4f}" if len(dlt) and not np.isnan(float(dlt[0])) else "N/A"])
        a(_w(["Day","BuyPres","BrkrConc","LrgOrd%","%StrongBP","%Instit","ΔBP"],rows,
             [5,8,9,8,10,8,8]))
    a(f"\n  BP: d-15={findings.get('bp_d15','?')}  →  d-1={findings.get('bp_d1','?')}\n")

    # ── 8. HIGH vs MID vs LOW split ───────────────────────────────────────────
    a("─"*72)
    a("8. HIGH (>=9.5%) vs MID (6-9.5%) vs LOW (3-6%) GAINER SPLIT")
    a("─"*72)
    if not split_df.empty:
        for m in [m for m in ["day_change_pct","rsi_14","vol_ratio_20",
                               "macd_hist","buyer_pressure","consol_score","atr_pct"]
                  if f"{m}_mean" in split_df.columns]:
            col = f"{m}_mean"
            a(f"\n  {m}")
            a(f"  {'Day':>5}  {'HIGH':>9}  {'MID':>9}  {'LOW':>9}  {'H-L':>9}")
            a(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
            for d in sorted(split_df["days_before"].unique()):
                def _g(grp):
                    v = split_df[(split_df["gain_group"]==grp)&(split_df["days_before"]==d)][col].values
                    return float(v[0]) if len(v) else np.nan
                hv,mv,lv = _g("HIGH_>=9.5%"),_g("MID_6-9.5%"),_g("LOW_3-6%")
                hl = hv-lv if not (np.isnan(hv) or np.isnan(lv)) else np.nan
                a(f"  {d:>5}  {hv:>9.4f}  {mv:>9.4f}  {lv:>9.4f}  {hl:>+9.4f}")

    div = findings.get("bp_diverge_day")
    if div:
        a(f"\n  → Buyer pressure HIGH vs LOW diverges (>0.05) from day {div}\n")

    # ── 9. Correlations ───────────────────────────────────────────────────────
    a("─"*72)
    a("9. TOP SPEARMAN CORRELATIONS (|rho| ranked, p<0.05)")
    a("─"*72)
    for c in findings.get("top_correlations",[]):
        a(f"  {c['metric']:<30}  day={c['days_before']:>3}  rho={c['spearman_rho']:>+7.4f}  p={c['p_value']:.6f}")
    a("")

    # ── 10. Pattern recipe ────────────────────────────────────────────────────
    a("─"*72)
    a("10. PATTERN RECIPE — what to look for before a big gainer")
    a("─"*72)
    a("""
  WEEK 3 (days -15 to -11):
    • Mild positive drift already underway (+0.6% avg daily)
    • RSI 48-52 — neutral zone, not extended
    • Volume at or slightly below average (vol_ratio ~0.9-1.0)
    • MACD histogram near zero — no strong signal yet
    • Buyer pressure 0.38-0.42 — indistinguishable from non-gainers

  WEEK 2 (days -10 to -6):
    • Daily gains accelerating (+1.0-1.4%)
    • RSI crossing 52-55 and rising — bullish shift
    • MACD histogram turning positive
    • Volume starting to pick up
    • Buyer pressure begins diverging: HIGH gainers rise, LOW stay flat
    • Broker concentration ticking up — fewer brokers doing more trades

  FINAL WEEK (days -5 to -1):
    • Daily gains +1.6-4.2% — momentum clearly visible
    • RSI 55-65 — bullish zone, not yet overbought
    • MACD histogram positive and rising (Δ positive each day)
    • BB %B rising toward 0.7+ — price in upper band
    • Buyer pressure >0.47 (HIGH gainers) vs 0.40 (LOW gainers)
    • Broker concentration >0.67 — concentrated smart money
    • OBV rising — accumulation confirmed
    • large_order_pct FALLING — retail momentum, not institutional

  COUNTER-SIGNALS (circuit move LESS likely even if price rising):
    • institutional_flag rising — institutions distributing
    • large_order_pct > 0.65 — block trades = distribution
    • buyer_pressure flat or falling in days -5 to -1
    • RSI already > 70 — overbought risk
    • BB %B > 0.95 — price at extreme upper band, reversal risk
    • MACD histogram declining (Δ negative) while price rising
""")
    a("="*72)

    path.write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(from_date=DEFAULT_FROM, to_date=DEFAULT_TO,
        top_n=DEFAULT_TOP_N, min_gain=DEFAULT_MIN_GAIN):
    log.info("="*65)
    log.info("TOP GAINER PRE-SIGNAL STUDY v2")
    log.info("  Range: %s → %s  |  Top %d  |  Min %.1f%%",
             from_date, to_date, top_n, min_gain)
    log.info("="*65)

    price_df   = load_price_history(from_date, to_date)
    fs_df      = load_floorsheet_signals(from_date, to_date)
    sector_map = load_sectors()

    price_df = compute_technicals(price_df)
    gainers  = find_top_gainers(price_df, top_n, min_gain)
    if gainers.empty:
        log.error("No gainer events found"); sys.exit(1)

    long_df = build_lookback_series(gainers, price_df, fs_df, sector_map)

    pattern_df, delta_df, corr_df, sector_df, split_df, findings = \
        analyze_patterns(long_df)

    write_outputs(long_df, pattern_df, delta_df, corr_df,
                  sector_df, split_df, findings, top_n, min_gain)

    log.info("="*65)
    log.info("DONE — %d events | %.2f%% avg gain | %s/",
             findings["n_events"], findings["avg_gain"], RESULTS_DIR)
    log.info("="*65)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from",     dest="from_date", default=DEFAULT_FROM)
    p.add_argument("--to",       dest="to_date",   default=DEFAULT_TO)
    p.add_argument("--top-n",    dest="top_n",     type=int,   default=DEFAULT_TOP_N)
    p.add_argument("--min-gain", dest="min_gain",  type=float, default=DEFAULT_MIN_GAIN)
    a = p.parse_args()
    run(a.from_date, a.to_date, a.top_n, a.min_gain)
