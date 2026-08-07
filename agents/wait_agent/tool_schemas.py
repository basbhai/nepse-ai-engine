"""
agent/tool_schemas.py
─────────────────────────────────────────────────────────────────────────────
NEPSE AI Engine — Phase 1 Agentic WAIT Monitor
JSON schemas for DeepSeek function calling API.

These schemas define the tools the orchestrator can call.
Format: OpenAI-compatible function calling (supported by OpenRouter).

Usage:
    from agent.tool_schemas import TOOL_SCHEMAS
    payload["tools"] = TOOL_SCHEMAS
    payload["tool_choice"] = "auto"
─────────────────────────────────────────────────────────────────────────────
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_state",
            "description": (
                "Get current NEPSE market state, geo/nepal sentiment scores, "
                "NRB rate decision, and paper mode flag. "
                "Call this first every cycle to decide if any action is warranted."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_open_waits",
            "description": (
                "Get all WAIT signals from market_log where outcome=PENDING "
                "and wait_condition is set. Returns the list of symbols waiting "
                "for a condition to be met before converting to BUY."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": (
                "Get current portfolio state: open positions, slots remaining, "
                "and available capital. Check this before escalating to Claude "
                "— if portfolio is full, no escalation needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fresh_indicators",
            "description": (
                "Get the latest technical indicators for a specific symbol: "
                "RSI, MACD cross, Bollinger Bands, volume ratio, EMA trend. "
                "Returns boolean flags (macd_cross_just_bullish, bb_breakout_up, "
                "volume_above_avg, etc.) — use these flags, not raw numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol e.g. 'NABIL', 'NTC', 'UPPER'",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_wait_condition",
            "description": (
                "Python pre-check: does the wait_condition text match any "
                "observable boolean flags from fresh indicators? "
                "Returns python_says_met=True if condition appears met. "
                "If confidence=UNKNOWN, condition is too narrative for Python — "
                "use your own judgment on whether to escalate to Claude."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "wait_condition": {
                        "type": "string",
                        "description": "The wait_condition text from market_log",
                    },
                    "indicators": {
                        "type": "object",
                        "description": (
                            "The full indicators dict returned by get_fresh_indicators. "
                            "Pass the entire object."
                        ),
                    },
                },
                "required": ["symbol", "wait_condition", "indicators"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_skip",
            "description": (
                "Log a silent skip to agent_trace. No Telegram. No market_log write. "
                "Call this when you decide NOT to escalate a symbol this cycle. "
                "Always call this for symbols you skip — the audit trail is important."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol being skipped",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why this symbol is being skipped this cycle. "
                            "e.g. 'MACD not bullish', 'portfolio full', "
                            "'market adverse', 'condition not met'"
                        ),
                    },
                    "cycle_ts": {
                        "type": "string",
                        "description": "Cycle timestamp (NST) passed from orchestrator",
                    },
                    "step": {
                        "type": "integer",
                        "description": "Iteration step number within this cycle",
                    },
                },
                "required": ["symbol", "reason", "cycle_ts", "step"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_claude",
            "description": (
                "Escalate a WAIT symbol to Claude Sonnet for final decision: "
                "BUY, STILL_WAIT, or NOW_AVOID. "
                "Only call this when you are confident the wait_condition may be met. "
                "Maximum 2 escalations per cycle — enforced externally. "
                "If shadow_mode=true (from settings), logs the escalation without "
                "calling Claude — safe for testing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol to escalate",
                    },
                    "wait_log_id": {
                        "type": "integer",
                        "description": "The market_log id of the original WAIT row",
                    },
                    "wait_condition": {
                        "type": "string",
                        "description": "The wait_condition text from market_log",
                    },
                    "indicators": {
                        "type": "object",
                        "description": "Fresh indicators dict from get_fresh_indicators",
                    },
                    "market_state": {
                        "type": "object",
                        "description": "Market state dict from get_market_state",
                    },
                    "cycle_ts": {
                        "type": "string",
                        "description": "Cycle timestamp (NST) from orchestrator",
                    },
                    "step": {
                        "type": "integer",
                        "description": "Iteration step number within this cycle",
                    },
                    "shadow_mode": {
                        "type": "boolean",
                        "description": (
                            "If true, log the escalation without calling Claude. "
                            "Read from AGENT_SHADOW_MODE setting."
                        ),
                    },
                },
                "required": [
                    "symbol", "wait_log_id", "wait_condition",
                    "indicators", "market_state", "cycle_ts", "step", "shadow_mode",
                ],
            },
        },
    },
]
