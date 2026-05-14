import requests, json

s = requests.Session()
r = s.post(
    "https://sharehubnepal.com/account/api/v1/auth/login/email",
    json={"email": "basbhai2026@gmail.com", "password": "Mahanatma@021"},
    headers={"Content-Type": "application/json", "referer": "https://sharehubnepal.com"},
    timeout=10,
)
token = r.json()["data"]["accessToken"]

r2 = s.get(
    "https://sharehubnepal.com/data/api/v1/floorsheet-analysis/broker-distribution",
    params={"duration": "1D"},
    headers={"Authorization": f"Bearer {token}", "referer": "https://sharehubnepal.com"},
    timeout=20,
)
data = r2.json()["data"]["content"]
print("Total rows:", len(data))
print("First item:", json.dumps(data[0], indent=2))