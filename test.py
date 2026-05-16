from modules.atrad_scraper import login, fetch_order_book

symbol = "USHEC"

if not login():
    print("Login failed")
    exit()

print("Login OK")

r = fetch_order_book(symbol)
print("\nfetch_order_book result:")
print(r)

if r:
    tb  = r.get("total_bid_qty", 0)
    ta  = r.get("total_ask_qty", 0)
    imb = r.get("imbalance", 0)
    print(f"\ntotal_bid_qty: {tb}")
    print(f"total_ask_qty: {ta}")
    print(f"imbalance:     {imb}  ({'buy pressure' if imb > 0.5 else 'sell pressure'})")
    print(f"top-of-book ratio would have been: {round(tb / (tb + ta), 4) if tb + ta else 'n/a'}")
else:
    print("Empty result")