"""
prompt_viewer.py — NEPSE AI Engine
====================================
Inspect AI prompts from any module without calling external APIs.
Fetches real data from Neon DB, builds the prompt, prints it with token stats.

Usage:
    python -m helper.prompt_viewer daily_context                    # today
    python -m helper.prompt_viewer daily_context --date 2026-04-01  # specific date
    python -m helper.prompt_viewer learning_hub                     # current week
    python -m helper.prompt_viewer learning_hub --stats             # token stats only
    python -m helper.prompt_viewer --list                           # list all modules

Called by: developers for debugging prompt quality and token costs.
No side effects — reads DB only, never writes.
"""

import argparse
import sys
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.WARNING,  # suppress module info logs during prompt viewing
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

NST = ZoneInfo("Asia/Kathmandu")

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str, model_type: str = "gemini") -> int:
    """
    Estimate token count.
    - Gemini: ~1 token per 4 chars (conservative)
    - GPT: try tiktoken, fallback to ~1 token per 4 chars
    """
    if model_type == "gpt":
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o")
            return len(enc.encode(text))
        except (ImportError, KeyError):
            pass
    return len(text) // 4


def format_token_stats(text: str, label: str, model_type: str = "gemini") -> str:
    """Format token/char stats for a prompt section."""
    tokens = estimate_tokens(text, model_type)
    chars  = len(text)
    lines  = text.count("\n") + 1
    return f"  {label:<25} {tokens:>7} tokens | {chars:>8} chars | {lines:>5} lines"

# ─────────────────────────────────────────────────────────────────────────────
# MODULE HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def view_daily_context(args):
    """View the Gemini prompt for daily_context_summarizer."""
    from analysis.daily_context_summarizer import get_prompt_for_date

    target = args.date or datetime.now(NST).strftime("%Y-%m-%d")
    prompt = get_prompt_for_date(target)

    if not prompt:
        print(f"\n  No data found for {target} — cannot build prompt.\n")
        return

    if args.stats:
        print(f"\n  Daily Context Summarizer — {target}")
        print(f"  {'─' * 50}")
        print(format_token_stats(prompt, "Gemini prompt", "gemini"))
        print()
        return

    print(f"\n{'═' * 70}")
    print(f"  DAILY CONTEXT SUMMARIZER — Gemini Flash prompt for {target}")
    print(f"{'═' * 70}\n")
    print(prompt)
    print(f"\n{'─' * 70}")
    tokens = estimate_tokens(prompt, "gemini")
    print(f"  Estimated tokens: ~{tokens}")
    print(f"  Characters: {len(prompt)}")
    print(f"  Approx cost: ~${tokens * 0.000001:.4f} (Gemini Flash input)")
    print()


def view_learning_hub(args):
    """View the GPT prompt for learning_hub weekly review."""
    from analysis.learning_hub import get_review_prompts

    system_prompt, user_prompt = get_review_prompts()

    if args.stats:
        sys_tok  = estimate_tokens(system_prompt, "gpt")
        usr_tok  = estimate_tokens(user_prompt, "gpt")
        total    = sys_tok + usr_tok
        print(f"\n  Learning Hub — GPT Weekly Review Prompt")
        print(f"  {'─' * 50}")
        print(format_token_stats(system_prompt, "System prompt", "gpt"))
        print(format_token_stats(user_prompt, "User prompt", "gpt"))
        print(f"  {'─' * 50}")
        print(f"  {'TOTAL':<25} {total:>7} tokens")
        print(f"  Approx cost: ~${total * 0.0000025:.4f} (GPT-4o input) / ~${total * 0.000005:.4f} (GPT-5o est.)")
        print()
        return

    print(f"\n{'═' * 70}")
    print(f"  LEARNING HUB — GPT System Prompt")
    print(f"{'═' * 70}\n")
    print(system_prompt)

    print(f"\n{'═' * 70}")
    print(f"  LEARNING HUB — GPT User Prompt")
    print(f"{'═' * 70}\n")
    print(user_prompt)

    sys_tok  = estimate_tokens(system_prompt, "gpt")
    usr_tok  = estimate_tokens(user_prompt, "gpt")
    total    = sys_tok + usr_tok
    print(f"\n{'─' * 70}")
    print(f"  System: ~{sys_tok} tokens | User: ~{usr_tok} tokens | Total: ~{total} tokens")
    print(f"  Approx cost: ~${total * 0.0000025:.4f} (GPT-4o input)")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# MODULE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODULES = {
    "daily_context": {
        "handler":     view_daily_context,
        "description": "Gemini Flash nightly summary prompt",
        "model":       "Gemini Flash",
        "schedule":    "Nightly ~9 PM NST (Sun-Thu)",
    },
    "learning_hub": {
        "handler":     view_learning_hub,
        "description": "GPT-5o weekly learning review prompt",
        "model":       "GPT-5o via OpenRouter",
        "schedule":    "Sunday ~5:45 PM NST",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE AI Engine — Prompt Viewer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m helper.prompt_viewer daily_context
  python -m helper.prompt_viewer daily_context --date 2026-04-03
  python -m helper.prompt_viewer daily_context --stats
  python -m helper.prompt_viewer learning_hub
  python -m helper.prompt_viewer learning_hub --stats
  python -m helper.prompt_viewer --list
        """,
    )
    parser.add_argument("module", nargs="?", choices=list(MODULES.keys()),
                        help="Module whose prompt to view")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date for daily modules (YYYY-MM-DD)")
    parser.add_argument("--stats", action="store_true",
                        help="Show token stats only, not full prompt")
    parser.add_argument("--list", action="store_true",
                        help="List all available modules")
    args = parser.parse_args()

    if args.list or not args.module:
        print(f"\n  Available prompt modules:")
        print(f"  {'─' * 60}")
        for name, info in MODULES.items():
            print(f"  {name:<20} {info['description']}")
            print(f"  {'':20} Model: {info['model']} | Schedule: {info['schedule']}")
        print()
        if not args.list:
            parser.print_help()
        return

    module_info = MODULES[args.module]
    module_info["handler"](args)


if __name__ == "__main__":
    main()