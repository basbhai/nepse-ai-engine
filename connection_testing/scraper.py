import requests
headers = {"User-Agent": "Mozilla/5.0"}

sites = {
    "ShareSansar": "https://www.sharesansar.com/today-share-price",
    "Merolagani": "https://merolagani.com/LatestMarket.aspx"
}

for name, url in sites.items():
    r = requests.get(url, headers=headers)
    status = "✅ OK" if r.status_code == 200 else "❌ Failed"
    print(f"{status} - {name}: {r.status_code}")
