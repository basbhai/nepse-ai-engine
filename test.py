import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# 1. Path to your saved captcha
captcha_file = "tms_captcha_24a0fb82.png"

def read_and_solve():
    # 2. Read the file as binary bytes
    if not os.path.exists(captcha_file):
        print(f"Error: {captcha_file} not found.")
        return
        
    with open(captcha_file, "rb") as f:
        image_bytes = f.read()

    # 3. Initialize Gemini Client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 4. Solve using Gemini 1.5 Flash (best for OCR)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            "Solve this captcha. Output only the characters, no spaces.",
            types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        ]
    )

    print(f"Solved Captcha: {response.text.strip()}")

read_and_solve()