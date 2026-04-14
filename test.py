import requests

url = "https://sharehubnepal.com/live/api/v2/floorsheet?Size=10&date=2026-4-8"
response = requests.get(url)
data = response.json()

print(data)