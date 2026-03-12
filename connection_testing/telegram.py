from dotenv import load_dotenv
import os, requests
load_dotenv()

token = os.getenv("TELEGRAM_BOT_TOKEN")
chat_id = os.getenv("TELEGRAM_CHAT_ID")

url = f"https://api.telegram.org/bot{token}/sendMessage"
r = requests.post(url, data={
    "chat_id": chat_id,
    "text": "✅ Test 1 OK - Telegram working"
})
print(r.json())