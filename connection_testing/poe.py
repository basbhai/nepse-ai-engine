import os
import base64
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(
    api_key = os.getenv("POE_API_KEY"),
    base_url = "https://api.poe.com/v1",
)

file_path = "tms_captcha_8a634dd3.png"

# Check if file exists to prevent FileNotFoundError
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        # FIX: Ensure 'base64' is imported (added at top)
        base64_data = base64.b64encode(f.read()).decode("utf-8")

chat = client.chat.completions.create(
    model = "deep-ai-search",
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Please solve this captcha and return only the characters."
                    },
                    {
                        "type": "file",
                        "file": {
                            "filename": file_path,
                            "file_data": f"data:image/png;base64,{base64_data}"
                        }
                    },
                ]
            }
        ],
)

print(chat.choices[0].message.content)