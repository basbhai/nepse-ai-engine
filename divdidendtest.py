import requests
import pandas as pd
from bs4 import BeautifulSoup

session = requests.Session()

year_to_fy = {
    31: "2081/2082",
    30: "2080/2081",
    29: "2079/2080",
    28: "2078/2079",
    27: "2077/2078",
    26: "2076/2077",
    24: "2075/2076",
    5:  "2074/2075",
    4:  "2073/2074",
    3:  "2072/2073",
    2:  "2071/2072",
    1:  "2070/2071",
    16: "2069/2070",
    15: "2068/2069",
    14: "2067/2068",
    13: "2066/2067",
    12: "2065/2066",
    11: "2064/2065",
    23: "2063/2064",
    17: "2062/2063",
    18: "2061/2062",
    19: "2060/2061",
    20: "2059/2060",
    21: "2058/2059",
    22: "2057/2058",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.sharesansar.com/proposed-dividend"
}

session.get("https://www.sharesansar.com/proposed-dividend", headers=headers)

params = {
    "draw": "1",
    "columns[0][data]": "DT_Row_Index","columns[0][name]": "","columns[0][searchable]": "false","columns[0][orderable]": "false","columns[0][search][value]": "","columns[0][search][regex]": "false",
    "columns[1][data]": "symbol","columns[1][name]": "tbl_company_list.symbol","columns[1][searchable]": "true","columns[1][orderable]": "true","columns[1][search][value]": "","columns[1][search][regex]": "false",
    "columns[2][data]": "companyname","columns[2][name]": "tbl_company_list.companyname","columns[2][searchable]": "true","columns[2][orderable]": "true","columns[2][search][value]": "","columns[2][search][regex]": "false",
    "columns[3][data]": "bonus_share","columns[3][name]": "","columns[3][searchable]": "true","columns[3][orderable]": "true","columns[3][search][value]": "","columns[3][search][regex]": "false",
    "columns[4][data]": "cash_dividend","columns[4][name]": "","columns[4][searchable]": "true","columns[4][orderable]": "true","columns[4][search][value]": "","columns[4][search][regex]": "false",
    "columns[5][data]": "total_dividend","columns[5][name]": "","columns[5][searchable]": "true","columns[5][orderable]": "true","columns[5][search][value]": "","columns[5][search][regex]": "false",
    "columns[6][data]": "announcement_date","columns[6][name]": "","columns[6][searchable]": "true","columns[6][orderable]": "true","columns[6][search][value]": "","columns[6][search][regex]": "false",
    "columns[7][data]": "bookclose_date","columns[7][name]": "","columns[7][searchable]": "true","columns[7][orderable]": "true","columns[7][search][value]": "","columns[7][search][regex]": "false",
    "columns[8][data]": "distribution_date","columns[8][name]": "","columns[8][searchable]": "true","columns[8][orderable]": "true","columns[8][search][value]": "","columns[8][search][regex]": "false",
    "columns[9][data]": "bonus_listing_date","columns[9][name]": "","columns[9][searchable]": "true","columns[9][orderable]": "true","columns[9][search][value]": "","columns[9][search][regex]": "false",
    "columns[10][data]": "year","columns[10][name]": "tbl_macro_year.year","columns[10][searchable]": "true","columns[10][orderable]": "true","columns[10][search][value]": "","columns[10][search][regex]": "false",
    "order[0][column]": "6","order[0][dir]": "desc",
    "start": "0","length": "20",
    "search[value]": "","search[regex]": "false",
    "type": "YEARWISE","year": "30","sector": "0"
}

r = session.get("https://www.sharesansar.com/proposed-dividend", params=params, headers=headers)
json_data = r.json()

# Extract the list of dividend records
records = json_data['data']

# Convert to DataFrame
df = pd.DataFrame(records)

# ------------------------------------------------------------------
# Clean the HTML in 'symbol' and 'companyname' columns
# ------------------------------------------------------------------
def extract_text_from_html(html):
    """Extract the text content from an HTML snippet."""
    if pd.isna(html) or not isinstance(html, str):
        return html
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(strip=True)

df['symbol'] = df['symbol'].apply(extract_text_from_html)
df['companyname'] = df['companyname'].apply(extract_text_from_html)

# ------------------------------------------------------------------
# Clean the 'bookclose_date' column: remove '[Closed]' and convert to datetime
# ------------------------------------------------------------------
df['bookclose_date'] = df['bookclose_date'].str.replace(r' \[Closed\]', '', regex=True)
# Convert to datetime, errors='coerce' will turn invalid dates into NaT
df['bookclose_date'] = pd.to_datetime(df['bookclose_date'], errors='coerce')

# ------------------------------------------------------------------
# Convert numeric columns to appropriate types
# ------------------------------------------------------------------
numeric_cols = ['bonus_share', 'cash_dividend', 'total_dividend', 'close']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# (Optional) Convert date columns to datetime
date_cols = ['announcement_date', 'distribution_date', 'bonus_listing_date', 'published_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# ------------------------------------------------------------------
# Display the cleaned DataFrame
# ------------------------------------------------------------------
print(df.head())
print(df.info())
