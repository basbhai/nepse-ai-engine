"""
AI/__init__.py
==============
NEPSE AI Engine — central AI gateway.
 
Import everything from here:
 
    from AI import ask_gemini_json, ask_gemini_text
    from AI import ask_claude, ask_gpt, ask_deepseek
 
Provider routing:
    Gemini Flash  → AI/gemini.py      (google.genai SDK)
    Claude Sonnet → AI/openrouter.py  (OpenRouter HTTP)
    GPT-4o        → AI/openrouter.py  (OpenRouter HTTP)
    DeepSeek R1   → AI/openrouter.py  (OpenRouter HTTP)
"""
 
from AI.gemini import ask_gemini_json, ask_gemini_text
from AI.openrouter import ask_claude, ask_gpt, ask_deepseek, ask_free, ask_deepseek_review
from AI.deepseek import ask_deepseek_text

__all__ = [
    "ask_gemini_json",
    "ask_gemini_text",
    "ask_claude",
    "ask_gpt",
    "ask_deepseek",
    "ask_free",
    "ask_deepseek_text",
    "ask_deepseek_review"
]