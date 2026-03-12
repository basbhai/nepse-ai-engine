from dotenv import load_dotenv
import os, gspread
from google.oauth2.service_account import Credentials
load_dotenv()

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_file(
    "../service_account.json", scopes=scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(os.getenv("GOOGLE_SHEETS_ID"))
settings = sheet.worksheet("SETTINGS")
data = settings.get_all_records()
print(f"✅ Test 2 OK - Connected to sheet")
print(f"Rows found in SETTINGS: {len(data)}")
