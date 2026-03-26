import requests
from bs4 import BeautifulSoup

session = requests.Session()
r = session.get("https://www.sharesansar.com/today-share-price")
soup = BeautifulSoup(r.text, 'html.parser')

token = soup.find('input', {'name': '_token'})['value']

data = {
    "_token": token,
    "sector": "all_sec",
    "date": "2026-03-23"
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.sharesansar.com/today-share-price"
}

resp = session.post("https://www.sharesansar.com/ajaxtodayshareprice", data=data, headers=headers)
print(resp.text)