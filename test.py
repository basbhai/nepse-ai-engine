from modules.indicators import HistoryCache
cache = HistoryCache()
cache.load()
sym = "NABIL"
print(f"Closes length: {len(cache.get_closes(sym))}")
print(f"Volumes length: {len(cache.get_volumes(sym))}")
print(f"First 5 volumes: {cache.get_volumes(sym)[:5]}")