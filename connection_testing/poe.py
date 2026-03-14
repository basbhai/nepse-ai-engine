import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_and_solve(captcha_file="tms_captcha_8a634dd3.png"):
    # 1. Check if file exists
    if not os.path.exists(captcha_file):
        print(f"Error: {captcha_file} not found.")
        return

    # 2. Read the file as binary bytes
    with open(captcha_file, "rb") as f:
        image_bytes = f.read()

    # 3. Initialize Client
    client = openai.OpenAI(
        api_key=os.getenv("POE_API_KEY"),
        base_url="https://api.poe.com/v1",
    )

    # 4. Create Chat Completion
    # Note: Ensure the model 'gpt-5.3-codex-spark' supports vision/binary inputs via this method
    try:
        chat = client.chat.completions.create(
            model="gpt-5.3-codex-spark",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Solve this captcha. Only return the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_bytes}"}} # Standard way to pass images
                ]
            }]
        )
        
        result = chat.choices[0].message.content
        print(f"Solved Captcha: {result.strip()}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Execution ---

# 1. Solve the Captcha
read_and_solve()

# 2. General JS Function Request
client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY"),
    base_url="https://api.poe.com/v1",
)

chat = client.chat.completions.create(
    model="gpt-5.3-codex-spark",
    messages=[{
        "role": "user",
        "content": "hello."
    }]
)

print("\nJS Function Example:")
print(chat.choices[0].message.content)