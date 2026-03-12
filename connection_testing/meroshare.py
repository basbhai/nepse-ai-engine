from dotenv import load_dotenv
import os, requests
load_dotenv()

username = os.getenv("MEROSHARE_USERNAME")
password = os.getenv("MEROSHARE_PASSWORD")
dp_id = os.getenv("MEROSHARE_DP_ID")

url = "https://webbackend.cdsc.com.np/api/meroShare/auth/"
payload = {
    "clientId": dp_id,
    "username": username,
    "password": password
}
login_headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://meroshare.cdsc.com.np",
    "Referer": "https://meroshare.cdsc.com.np/",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

r = requests.post(url, json=payload, headers=login_headers)
print(f"Status: {r.status_code}")

if r.status_code == 200:
    # Token is in Authorization HEADER not body
    token = r.headers.get("Authorization")
    
    
    if token:
        print(f"✅ Test 4 OK - Meroshare token received")
        print(f"Token preview: {token[:30]}...")
        
        # Save token for API 2 and API 3
        print(f"\nToken works for next API calls")
    else:
        print(f"❌ Token not found in headers")
        print(f"Headers received: {dict(r.headers)}")
else:
    print(f"❌ Failed: {r.text}")