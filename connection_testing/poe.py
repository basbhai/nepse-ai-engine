import openai
from dotenv import load_dotenv
import os
load_dotenv()

client = openai.OpenAI(
    api_key = os.getenv("POE_API_KEY"),
    base_url = "https://api.poe.com/v1",
)

chat = client.chat.completions.create(
    model = "gpt-5.3-codex-spark",
    messages = [{
      "role": "user",
      "content": "give me js function for anyting"
    }]
)

print(chat.choices[0].message.content)