from AI.openrouter import _call
# Pass a bad API key to force a 401 — confirms retry loop runs
result = _call('anthropic/claude-sonnet-4-6', [{'role':'user','content':'hi'}], 100, 0.2, 'retry_test')
print('Result:', result)