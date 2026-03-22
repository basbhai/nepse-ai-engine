
import openai

client = openai.OpenAI(
    api_key = "YOUR_POE_API_KEY",  # or os.getenv("POE_API_KEY")
    base_url = "https://api.poe.com/v1",
)

chat = client.chat.completions.create(
    model = "deepseek-r1",
    messages = [{
      "role": "user",
      "content": "Write a simple function that generates the nth element of the Fibonacci sequence for an input n and then optimize its implementation."
    }]
)
