import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

import json
import pandas as pd
from openai import OpenAI

# Initialize your client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def talk_to_hunter(df_input, prompt="Analyze these news headlines and provide sentiment."):
    """
    Takes a DataFrame (or list of DataFrames), converts to text, 
    and returns a JSON response from hunter-alpha.
    """
    
    # 1. Convert DataFrame(s) to a string format for the AI
    if isinstance(df_input, list):
        # Merge list of DFs into one for context
        combined_df = pd.concat(df_input, ignore_index=True)
    else:
        combined_df = df_input

    # Clean the data for the prompt
    data_str = combined_df.to_string(index=False)

    # 2. Build the message with a JSON enforcement prompt
    system_prompt = (
        "You are a financial analyst. Return your output strictly in JSON format. "
        "Analyze the provided stock news/announcements."
    )
    
    user_content = f"{prompt}\n\nData:\n{data_str}"

    try:
        # 3. Call Hunter Alpha with reasoning enabled
        response = client.chat.completions.create(
            model="openrouter/hunter-alpha",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            extra_body={
                "reasoning": {"enabled": True},
                "response_format": {"type": "json_object"} # Force JSON
            }
        )

        # 4. Extract content
        raw_output = response.choices[0].message.content
        
        # Parse string to JSON object
        return json.loads(raw_output)

    except Exception as e:
        return {"error": str(e), "raw_response": raw_output if 'raw_output' in locals() else None}

# --- Example Usage with your Scraper Output ---
# Assuming 'df' is the result from your scraping script:
# output = talk_to_hunter(df, prompt="Identify which companies have IPOs or Dividends mentioned.")
# print(json.dumps(output, indent=2))