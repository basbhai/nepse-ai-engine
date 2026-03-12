import os, requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

# 2026 Active Free Tier Models
models = [
    "gemini-2.5-flash-lite", 
    "gemini-2.5-flash",
    "gemini-3.1-flash-preview"
]

data = {"contents": [{"parts": [{"text": "Hello! reply anything, do you know my name"}]}]}

for model in models:
    # Note: Use the stable /v1/ for 2.5 models if v1beta fails
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={key}"
    r = requests.post(url, json=data)
    response = r.json()
    
    if "candidates" in response:
        reply = response["candidates"][0]["content"]["parts"][0]["text"]
        print(f"✅ Test 3 OK - {reply}")
        print(f"✅ Working model: {model}")
        
    else:
        error_msg = response.get("error", {}).get("message", "Unknown Error")
        print(f"❌ {model}: {error_msg}")