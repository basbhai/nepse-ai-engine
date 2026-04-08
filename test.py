# test_market_log_write.py
# Run: python test_market_log_write.py
# Shows exactly what claude_analyst would write to market_log

from dotenv import load_dotenv
load_dotenv()

from sheets import run_raw_sql

# Get the latest market_log row written by claude_analyst
rows = run_raw_sql("""
    SELECT *
    FROM market_log
    WHERE source = 'claude_analyst'
       OR timestamp IS NOT NULL
    ORDER BY id DESC
    LIMIT 1
""", ())

if not rows:
    print("NO ROWS in market_log at all")
else:
    row = rows[0]
    print(f"=== Latest market_log row (id={row.get('id')}) ===")
    for col, val in row.items():
        filled = "✅" if val not in (None, "", "PENDING") else "❌"
        print(f"  {filled} {col:<30} = {repr(val)}")