from dotenv import load_dotenv
import os
import csv

load_dotenv()

from db.connection import _db

# Get 7-10 symbols per sector
with _db() as cur:
    cur.execute("""
        WITH ranked AS (
            SELECT symbol, sectorname,
                   ROW_NUMBER() OVER (PARTITION BY sectorname ORDER BY RANDOM()) as rn
            FROM share_sectors
            WHERE status = 'A'
        )
        SELECT symbol, sectorname
        FROM ranked
        WHERE rn <= 10
        ORDER BY sectorname, symbol;
    """)
    sectors = cur.fetchall()

symbols = [r['symbol'] for r in sectors]
print(f"Selected {len(symbols)} symbols across {len(set(r['sectorname'] for r in sectors))} sectors")

# Get full price history
placeholders = ','.join([f"'{s}'" for s in symbols])
with _db() as cur:
    cur.execute(f"""
        SELECT ph.date, ph.symbol, ss.sectorname, ph.conf_score,
               ph.open, ph.high, ph.low, ph.close, ph.ltp,
               ph.volume, ph.turnover, ph.prev_close
        FROM price_history ph
        JOIN share_sectors ss ON ss.symbol = ph.symbol
        WHERE ph.symbol IN ({placeholders})
          AND ph.close IS NOT NULL AND ph.close != ''
        ORDER BY ph.symbol, ph.date;
    """)
    rows = cur.fetchall()

print(f"Fetched {len(rows)} rows")

output_path = "nepse_sector_sample.csv"
with open(output_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {output_path}")