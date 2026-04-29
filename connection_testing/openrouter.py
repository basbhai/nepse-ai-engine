from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= os.getenv("OPENROUTER_API_KEY"),
)

completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/basbhai/nepse-ai-engine",
            "X-Title":      "NEPSE AI Engine",
        },
  extra_body={},
  model="qwen/qwen3-coder:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)