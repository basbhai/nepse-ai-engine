import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPEN_ROUTER_KEY')

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {OPENROUTER_API_KEY}", # FIXED: Added Bearer and f-string
    "Content-Type": "application/json",             # Recommended: explicitly set type
    "HTTP-Referer": "http://localhost:3000",       # Optional: for OpenRouter rankings
    "X-Title": "My Local Test Script",             # Optional: for OpenRouter rankings
  },
  data=json.dumps({
    "model": "google/gemma-3-4b-it:free",                      # Note: gpt-5.2 doesn't exist yet (use gpt-4o)
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)

# Print the actual text response instead of just the <Response [200]> object
if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error {response.status_code}: {response.text}")