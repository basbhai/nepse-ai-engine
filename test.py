
import requests
import json
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file





OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
     "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "liquid/lfm-2.5-1.2b-thinking:free",
    "messages": [
        {
          "role": "user",
          "content": {"How many r's are in the word 'strawberry'?"}
        }
      ],
    "reasoning": {"enabled": False}
  })
)

# Extract the assistant message with reasoning_details
response = response.json()

print(response)