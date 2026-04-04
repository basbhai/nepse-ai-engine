import requests
data=requests.get("https://chukul.com/api/stock/").json()
for s in data:
    n=s.get("name","").lower()
    if "debenture" in n or "mutual" in n: continue
    r=requests.get(f"https://chukul.com/api/stock/{s['id']}/report/").json()
    print(r)
    break


