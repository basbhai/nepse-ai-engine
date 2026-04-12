import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv('DATABASE_URL'))
cur = conn.cursor()

cur.execute("""
    SELECT
        COUNT(DISTINCT symbol) FILTER (WHERE date < '2024-01-01') as pre_2024_symbols,
        COUNT(DISTINCT symbol) FILTER (WHERE date < '2022-01-01') as pre_2022_symbols,
        COUNT(DISTINCT symbol) FILTER (WHERE date < '2020-01-01') as pre_2020_symbols,
        COUNT(DISTINCT symbol)                                     as total_symbols
    FROM price_history
""")
row = cur.fetchone()
print(f"Total symbols       : {row[3]}")
print(f"With pre-2024 data  : {row[0]}")
print(f"With pre-2022 data  : {row[1]}")
print(f"With pre-2020 data  : {row[2]}")

# Average days per symbol
cur.execute("""
    SELECT 
        AVG(cnt) as avg_days,
        MIN(cnt) as min_days,
        MAX(cnt) as max_days
    FROM (
        SELECT symbol, COUNT(*) as cnt
        FROM price_history
        GROUP BY symbol
    ) t
""")
row = cur.fetchone()
print(f"\nAvg days per symbol : {row[0]:.0f}")
print(f"Min days per symbol : {row[1]}")
print(f"Max days per symbol : {row[2]}")

conn.close()