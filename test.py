import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "gemma3:1b"

WAIT_CONDITION = (
    'BUY becomes valid if: (1) tech_score rises to ≥75 via MACD bullish cross or BB breakout above upper band, '
    'AND (2) Volume/OS Ratio exceeds 1% confirming genuine smart money participation, '
    'AND (3) LTP holds above 350 with OBV turning upward — ideally on a day when NEPSE turnover exceeds Rs 4B and IPO drain subsides.'
)

PROMPT = f"""Parse this NEPSE WAIT condition text into structured JSON.

WAIT CONDITION TEXT:
"{WAIT_CONDITION}"

OUTPUT RULES:
- type "indicator": field exists in indicators table (macd_cross, bb_signal, obv_trend, ema_trend, rsi_14, tech_score, macd_histogram, bb_pct_b)
- type "price": requires checking LTP/price level against price_history.ltp
- type "market": field from market state (nepal_score, geo_score, market_state, breadth_signal)
- type "ambiguous": cannot be evaluated from a single field (e.g. ratio calculations, multi-step conditions, narrative conditions like IPO drain, turnover thresholds, confidence score, political conditions)
- ops allowed: eq, neq, gt, gte, lt, lte, in
- logic: ALL (every requirement must pass)

STRICT FORMATTING RULES:
- Price ranges MUST be two separate requirements: one with op "gte" for lower bound, one with op "lte" for upper bound
- Single-sided price conditions (e.g. "LTP holds above 350") are ONE requirement only — do NOT add a second bound
- OR conditions (e.g. "MACD bullish cross OR BB breakout") must each become a separate requirement
- ALL numeric values MUST be numbers not strings: 75 not "75", 350 not "350"
- type spelling is exactly "ambiguous" — never "ambiguity"
- No trailing commas. No markdown. No explanation. JSON only.

Respond ONLY with valid JSON."""

def parse_condition(wait_condition: str, market_log_id: int, symbol: str) -> dict | None:
    prompt = PROMPT

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        # sanitize common model typo
        for req in parsed.get("requirements", []):
            if req.get("type") == "ambiguity":
                req["type"] = "ambiguous"

        parsed["market_log_id"] = market_log_id
        parsed["symbol"]        = symbol

        return parsed

    except requests.exceptions.ConnectionError:
        print("ERROR: Ollama not reachable at localhost:11434 — is it running?")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed: {e}")
        print(f"Raw output: {raw}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


if __name__ == "__main__":
    result = parse_condition(WAIT_CONDITION, market_log_id=180, symbol="API")
    if result:
        print("SUCCESS:")
        print(json.dumps(result, indent=2))
    else:
        print("FAILED")