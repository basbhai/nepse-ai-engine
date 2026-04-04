

"""
scripts/load_fundamental_beta.py
Load fundamental_beta.csv into the fundamental_beta table.
Run once after each quarterly fundamental_study run.

Usage:
    python -m modules.load_fundamental_beta
    python -m modules.load_fundamental_beta --csv path/to/fundamental_beta.csv
"""

import csv
import sys
import os
from pathlib import Path
from datetime import date

def load_beta_csv(csv_path: str = "outputs/fundamentals/fundamental_beta.csv"):
    """Load fundamental_beta.csv into DB."""

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Import DB connection (adjust to your project's pattern)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.connection import _db

    rows_loaded = 0
    rows_skipped = 0

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        with _db() as cur:
            for row in reader:
                symbol = row.get("symbol", "").strip().upper()
                if not symbol:
                    rows_skipped += 1
                    continue

                try:
                    beta         = float(row["beta"])
                    market_corr  = float(row["market_corr"])   if row.get("market_corr")  not in ("", None) else None
                    market_corr_p= float(row["market_corr_p"]) if row.get("market_corr_p") not in ("", None) else None
                    n_months     = int(float(row["n_months"])) if row.get("n_months")      not in ("", None) else None
                except (ValueError, KeyError) as e:
                    print(f"  SKIP {symbol}: {e}")
                    rows_skipped += 1
                    continue

                cur.execute(
                    """
                    INSERT INTO fundamental_beta
                        (id, symbol, beta, market_corr, market_corr_p, n_months, computed_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, computed_date) DO UPDATE SET
                        beta          = EXCLUDED.beta,
                        market_corr   = EXCLUDED.market_corr,
                        market_corr_p = EXCLUDED.market_corr_p,
                        n_months      = EXCLUDED.n_months
                    """,
                    (f"{symbol}_{date.today()}", symbol, beta, market_corr, market_corr_p, n_months, date.today())
                )
                rows_loaded += 1

    print(f"fundamental_beta loaded: {rows_loaded} symbols, {rows_skipped} skipped")
    print(f"Table now contains data computed: {date.today()}")


if __name__ == "__main__":
    args = sys.argv[1:]
    csv_path = "outputs/fundamentals/fundamental_beta.csv"

    for arg in args:
        if arg.startswith("--csv"):
            parts = arg.split("=", 1)
            if len(parts) == 2:
                csv_path = parts[1]
            elif args.index(arg) + 1 < len(args):
                csv_path = args[args.index(arg) + 1]

    print(f"Loading: {csv_path}")
    load_beta_csv(csv_path)