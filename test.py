from modules.indicators import compute_indicators, HistoryCache
from modules.scraper import PriceRow

cache = HistoryCache()
cache.load()

# Simulate ATrad returning volume=0 for SHIVM
price_row = PriceRow(
    symbol='SHIVM',
    ltp=677.0,
    open_price=667.0,
    high=680.0,
    low=667.0,
    close=677.0,
    prev_close=666.9,
    volume=0,
)

result = compute_indicators('SHIVM', price_row, cache)
print('volume in result :', result.volume)
print('rsi_14           :', result.rsi_14)
print('macd_histogram   :', result.macd_histogram)
print('bb_signal        :', result.bb_signal)