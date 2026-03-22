import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get("https://www.sharesansar.com/live-trading", headers=headers, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")
table = soup.find("table")
first_row = table.find("tr")
cols = [th.get_text(strip=True) for th in first_row.find_all(["th", "td"])]
print("HEADERS:", cols)
rows = table.find_all("tr")
if len(rows) > 1:
    cells = [td.get_text(strip=True) for td in rows[1].find_all("td")]
    print("ROW1:", cells)