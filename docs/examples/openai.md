# OpenAI Example

End-to-end demo using the OpenAI API directly (no framework adapter needed). Shows triage catching a `WRONG_TOOL_CALLED` failure and recovering with `retry_with_tool_manifest`.

**Source:** [`examples/raw_openai.py`](https://github.com/mattekudacy/triage/blob/main/examples/raw_openai.py)

## Requirements

```bash
pip install triage-agent openai
export OPENAI_API_KEY=sk-...
```

## What it demonstrates

1. **Attempt 1** — the agent deliberately sends a non-existent tool name (`"nonexistent_calculator"`) to the model. The handler detects the unknown tool and raises `RuntimeError("no tool named 'nonexistent_calculator'")`.
2. triage classifies the error as `WRONG_TOOL_CALLED` (matched by `RulesClassifier` rule 2).
3. `retry_with_tool_manifest` returns `RecoveryAction.RETRY` with a hint.
4. **Attempt 2** — `_triage_hint` is injected into kwargs. The agent switches to the correct tool manifest and succeeds.

## Code

```python
import asyncio
import json
import triage
from triage.strategies.retry import retry_with_tool_manifest
from triage.taxonomy import Step
import openai

CORRECT_TOOLS = [{"type": "function", "function": {
    "name": "calculator",
    "description": "Evaluate a simple arithmetic expression.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string"}
    }, "required": ["expression"]},
}}]

BAD_TOOLS = [{"type": "function", "function": {
    "name": "nonexistent_calculator",
    "description": "Does not exist.",
    "parameters": {"type": "object", "properties": {}},
}}]

async def openai_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    client = openai.AsyncOpenAI()
    messages = [{"role": "user", "content": task}]

    if _triage_hint:
        messages.insert(0, {"role": "system", "content": f"Hint: {_triage_hint}"})

    # Use bad tools on first call to trigger WRONG_TOOL_CALLED
    tools = CORRECT_TOOLS if _triage_hint else BAD_TOOLS

    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    message = response.choices[0].message

    if not message.tool_calls:
        record_step(Step(index=0, action="no tool call", llm_output=message.content))
        return message.content or ""

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    valid_names = {t["function"]["name"] for t in CORRECT_TOOLS}

    if tool_name not in valid_names:
        err = f"no tool named {tool_name!r}"
        record_step(Step(index=0, action=f"called {tool_name!r}", tool_called=tool_name, error=err))
        raise RuntimeError(err)

    args = json.loads(tool_call.function.arguments)
    result = str(eval(args["expression"], {"__builtins__": {}}))
    record_step(Step(index=0, action="calculator", tool_called=tool_name,
                     tool_input=args, tool_output=result))
    return f"Result: {result}"

policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED=retry_with_tool_manifest(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(openai_agent, policy=policy)

async def main():
    result = await agent.run("What is 42 * 17?")
    print(result)

asyncio.run(main())
```

## Run

```bash
python examples/raw_openai.py
```
