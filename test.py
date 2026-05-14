import requests

s = requests.Session()
r = s.post(
    "https://sharehubnepal.com/account/api/v1/auth/login/email",
    json={"email": "basbhai2026@gmail.com", "password": "Mahanatma@021"},
    headers={"Content-Type": "application/json", "referer": "https://sharehubnepal.com"},
    timeout=10,
)
token = r.json()["data"]["accessToken"]

# # r2 = s.get(
# #     "https://sharehubnepal.com/data/api/v1/floorsheet-analysis/broker-aggressive-holdings",
# #     params={"EquityOnly": "true"},
# #     headers={"Authorization": f"Bearer {token}", "referer": "https://sharehubnepal.com"},
# #     timeout=20,
# )
r3 = s.get(
    "https://sharehubnepal.com/data/api/v1/floorsheet-analysis/broker-aggressive-holdings",
    params={"EquityOnly": "true", "from": "2026-05-04", "to": "2026-05-07"},
    headers={"Authorization": f"Bearer {token}", "referer": "https://sharehubnepal.com"},
    timeout=20,
)
import json
print("Status:", r3.status_code)
print("Count:", len(r3.json()["data"]["content"]))
print("First symbol:", r3.json()["data"]["content"][0]["symbol"])
print("Sample:", json.dumps(r3.json()["data"]["content"][0], indent=2)[:800])