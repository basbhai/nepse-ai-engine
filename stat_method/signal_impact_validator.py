"""
signal_impact_validator_v2.py
─────────────────────────────────────────────────────────────────────────────
ENHANCED: Backfill international prices to database, then test correlation.

Two modes:
  1. --backfill: Fetch VIX/Crude/Nifty/DXY/Gold from yfinance → store in international_prices
  2. --test: Read from international_prices table, test correlation vs NEPSE
  
Usage:
  python signal_impact_validator_v2.py --backfill
  python signal_impact_validator_v2.py --test
  python signal_impact_validator_v2.py --full (both)

Output:
  - Spearman rank correlation ρ at lags 0-7 days
  - Bonferroni-corrected p-values
  - Recommendation: KEEP | REMOVE | TIEBREAKER
  - Token savings estimate
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import yfinance as yf
import sys
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TEST_START_DATE = "2020-01-01"
TEST_END_DATE = "2026-03-28"

# Your NEPSE composite index ID
NEPSE_INDEX_ID = "58"

# INTERNATIONAL variables only
INTERNATIONAL_VARIABLES = {
    "VIX": "^VIX",           # US volatility
    "Crude": "CL=F",         # WTI Crude Oil
    "Nifty": "^NSEI",        # India Nifty50
    "DXY": "DX-Y.NYB",       # US Dollar index
    "Gold": "GC=F",          # Gold futures
}

# Bonferroni correction: 5 vars × 8 lags = 40 tests
BONFERRONI_ALPHA = 0.05 / 40  # ≈ 0.00125

RHO_THRESHOLD = 0.15
P_THRESHOLD = BONFERRONI_ALPHA
LAGS = [0, 1, 2, 3, 4, 5, 6, 7]

# ─────────────────────────────────────────────────────────────────────────────
# BACKFILL TO DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def backfill_international_prices():
    """Fetch from yfinance and store in international_prices table."""
    try:
        from sheets import write_row
        
        print("\n" + "="*80)
        print("BACKFILLING INTERNATIONAL PRICES")
        print("="*80 + "\n")
        
        for var_name, ticker in INTERNATIONAL_VARIABLES.items():
            log.info(f"Fetching {var_name} ({ticker})...")
            
            try:
                df = yf.download(ticker, start=TEST_START_DATE, end=TEST_END_DATE, 
                                progress=False)
                if df.empty:
                    log.warning(f"  No data for {ticker}")
                    continue
                
                # Extract date and close price
                df = df[['Close']].reset_index()
                df.columns = ['date', 'close_price']
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                df['variable_name'] = var_name
                df['source'] = 'yfinance'
                
                # Write each row
                count = 0
                for _, row in df.iterrows():
                    success = write_row("international_prices", {
                        'date': row['date'],
                        'variable_name': var_name,
                        'close_price': str(round(float(row['close_price']), 4)),
                        'source': 'yfinance'
                    })
                    if success:
                        count += 1
                
                log.info(f"  → {count} rows written for {var_name}")
                
            except Exception as e:
                log.error(f"  Error fetching {ticker}: {e}")
                continue
        
        print("\n✅ Backfill complete")
        
    except Exception as e:
        log.error(f"Backfill failed: {e}")
        return False
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FROM DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def load_nepse_from_db() -> pd.DataFrame:
    """Load NEPSE composite index from database."""
    try:
        from sheets import read_tab_where
        
        log.info("Loading NEPSE (index_id=58) from database...")
        nepse_rows = read_tab_where(
            "nepse_indices",
            {"index_id": NEPSE_INDEX_ID},
            order_by="date",
            desc=False
        )
        
        if not nepse_rows:
            log.error("  No NEPSE data in database")
            return pd.DataFrame()
        
        df = pd.DataFrame(nepse_rows)
        print(df)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['Close'] = df['current_value'].astype(float)
        df = df[['Close']].sort_index()
        log.info(f"  → {len(df)} NEPSE rows")
        return df
        
    except Exception as e:
        log.error(f"  Error loading from DB: {e}")
        return pd.DataFrame()


def load_international_from_db(variable_name: str) -> pd.DataFrame:
    """Load international prices from database."""
    try:
        from sheets import read_tab_where
        
        rows = read_tab_where(
            "international_prices",
            {"variable_name": variable_name},
            order_by="date",
            desc=False
        )
        
        if not rows:
            log.warning(f"  No {variable_name} data in database")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df['Close'] = df['close_price'].astype(float)
        df = df[['Close']].sort_index()
        log.info(f"  {variable_name}: {len(df)} rows from DB")
        return df
        
    except Exception as e:
        log.error(f"  Error loading {variable_name}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def align_series(nepse: pd.Series, geo: pd.Series, var_name: str) -> tuple:
    """Align two price series on common dates."""
    common = nepse.index.intersection(geo.index)
    n1 = nepse[common].dropna()
    n2 = geo[common].dropna()
    
    common = n1.index.intersection(n2.index)
    n1 = n1[common]
    n2 = n2[common]
    
    log.info(f"  {var_name}: {len(n1)} aligned trading days")
    return n1, n2


def compute_returns(prices: pd.Series, lag: int = 1) -> pd.Series:
    """Compute pct returns over lag days."""
    return prices.pct_change(lag).dropna() * 100


def test_correlation(nepse_ret: pd.Series, geo_ret: pd.Series, 
                     var_name: str, lag: int) -> dict:
    """Spearman rank correlation test."""
    common = nepse_ret.index.intersection(geo_ret.index)
    n = nepse_ret[common]
    g = geo_ret[common]
    
    if len(n) < 20:
        return {
            'variable': var_name,
            'lag': lag,
            'rho': np.nan,
            'p_value': np.nan,
            'n': len(n),
            'recommendation': 'INSUFFICIENT_DATA'
        }
    
    # Check for constant arrays (no variance)
    if n.std() == 0 or g.std() == 0:
        return {
            'variable': var_name,
            'lag': lag,
            'rho': 0.0,
            'p_value': 1.0,
            'n': len(n),
            'recommendation': 'REMOVE'  # Constant = no signal
        }
    
    try:
        rho, p_value = spearmanr(n, g)
    except Exception as e:
        log.warning(f"  Correlation error at lag {lag}: {e}")
        return {
            'variable': var_name,
            'lag': lag,
            'rho': np.nan,
            'p_value': np.nan,
            'n': len(n),
            'recommendation': 'SKIP'
        }
    
    # Handle NaN result
    if np.isnan(rho):
        return {
            'variable': var_name,
            'lag': lag,
            'rho': 0.0,
            'p_value': 1.0,
            'n': len(n),
            'recommendation': 'REMOVE'
        }
    
    # Recommendation logic
    if abs(rho) < 0.05:
        rec = 'REMOVE'
    elif (abs(rho) > RHO_THRESHOLD) and (p_value < P_THRESHOLD):
        rec = 'KEEP'
    elif abs(rho) > 0.08:
        rec = 'TIEBREAKER'
    else:
        rec = 'REMOVE'
    
    return {
        'variable': var_name,
        'lag': lag,
        'rho': round(rho, 4),
        'p_value': round(p_value, 6),
        'n': len(n),
        'recommendation': rec
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    """Run correlation tests using DB data."""
    
    print("\n" + "="*80)
    print("NEPSE INTERNATIONAL VARIABLE IMPACT TEST")
    print("="*80)
    print(f"Test period: {TEST_START_DATE} to {TEST_END_DATE}")
    print(f"NEPSE index_id: {NEPSE_INDEX_ID} (composite)")
    print(f"Bonferroni α: {BONFERRONI_ALPHA:.6f}")
    print()
    
    # Load NEPSE
    nepse_df = load_nepse_from_db()
    if nepse_df.empty:
        log.error("Cannot proceed")
        return
    
    nepse_close = nepse_df['Close']
    nepse_close = nepse_close[
        (nepse_close.index >= TEST_START_DATE) & 
        (nepse_close.index <= TEST_END_DATE)
    ]
    
    log.info(f"NEPSE rows in period: {len(nepse_close)}\n")
    
    # Load international variables from DB
    geo_data = {}
    for var_name in INTERNATIONAL_VARIABLES.keys():
        df = load_international_from_db(var_name)
        if not df.empty:
            geo_data[var_name] = df['Close']
    
    print()
    
    if not geo_data:
        log.error("No international data in database. Run --backfill first.")
        return
    
    # Run correlation tests
    results = []
    for var_name, geo_series in geo_data.items():
        log.info(f"\nTesting {var_name}...")
        
        nepse_aligned, geo_aligned = align_series(nepse_close, geo_series, var_name)
        
        if len(nepse_aligned) < 20:
            log.warning(f"  Insufficient data")
            continue
        
        for lag in LAGS:
            nepse_ret = compute_returns(nepse_aligned, lag)
            geo_ret = compute_returns(geo_aligned, lag)
            
            result = test_correlation(nepse_ret, geo_ret, var_name, lag)
            results.append(result)
    
    # Results table
    print("\n" + "="*80)
    print("ALL CORRELATION RESULTS")
    print("="*80 + "\n")
    
    df_results = pd.DataFrame(results)
    print(df_results[['variable', 'lag', 'rho', 'p_value', 'n', 'recommendation']].to_string(index=False))
    
    # Best per variable
    print("\n" + "="*80)
    print("BEST CORRELATION PER VARIABLE")
    print("="*80 + "\n")
    
    for var in INTERNATIONAL_VARIABLES.keys():
        var_data = df_results[df_results['variable'] == var]
        if var_data.empty:
            continue
        
        var_data = var_data.copy()
        var_data['abs_rho'] = var_data['rho'].abs()
        best = var_data.loc[var_data['abs_rho'].idxmax()]
        
        print(f"{var:8} | ρ={best['rho']:+.4f} | lag={int(best['lag'])}d | p={best['p_value']:.6f} | {best['recommendation']}")
    
    # Summary
    print("\n" + "="*80)
    print("RECOMMENDATION SUMMARY")
    print("="*80 + "\n")
    
    for var in INTERNATIONAL_VARIABLES.keys():
        var_data = df_results[df_results['variable'] == var]
        
        keep = (var_data['recommendation'] == 'KEEP').sum()
        tie = (var_data['recommendation'] == 'TIEBREAKER').sum()
        remove = (var_data['recommendation'] == 'REMOVE').sum()
        
        final = 'KEEP' if keep > 0 else ('TIEBREAKER' if tie >= 3 else 'REMOVE')
        
        print(f"{var:10} | KEEP:{keep} | TIE:{tie} | REMOVE:{remove} | → {final}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80 + """
    
If variable → REMOVE:
  1. Strip from geo_sentiment.py scoring formula
  2. Remove from claude_analyst.py context prompt
  3. Remove from filter_engine.py weighting
  
If variable → KEEP:
  1. Validate logic in geo_sentiment.py
  2. Add to claude_analyst.py reasoning
  
If variable → TIEBREAKER:
  1. Use only as secondary confirmation
  2. Never primary reason to enter/exit
  
Expected token savings: ~50 tokens per removed variable per call
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--test"
    
    if mode == "--backfill":
        backfill_international_prices()
    elif mode == "--test":
        run_test()
    elif mode == "--full":
        backfill_international_prices()
        run_test()
    else:
        print("""
Usage:
  python signal_impact_validator_v2.py --backfill   # Fetch & store prices
  python signal_impact_validator_v2.py --test       # Run correlation tests
  python signal_impact_validator_v2.py --full       # Both
        """)