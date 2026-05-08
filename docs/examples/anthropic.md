# Anthropic Example

Demo using the Anthropic API directly. Shows triage detecting a `LOOP_DETECTED` failure and recovering with `replan`.

**Source:** [`examples/anthropic_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/anthropic_agent.py)

## Requirements

```bash
pip install "triage-agent[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## What it demonstrates

1. A synthetic agent records the same tool call three times in a row, which `RulesClassifier` matches as `LOOP_DETECTED`.
2. `replan` returns `RecoveryAction.REPLAN` with a hint.
3. On retry, `_triage_hint` is injected. The agent checks for the hint and breaks out of the loop.

## Code

```python
import asyncio
import triage
from triage.strategies.replan import replan
from triage.taxonomy import Step
import anthropic

_attempt = [0]

async def claude_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    _attempt[0] += 1
    client = anthropic.Anthropic()

    # First attempt: simulate a loop (same tool call three times)
    if _attempt[0] == 1 and not _triage_hint:
        for i in range(3):
            record_step(Step(
                index=i,
                action="search",
                tool_called="web_search",
                tool_input={"q": task},
                tool_output="no results",
            ))
        raise RuntimeError("Agent is stuck in a search loop.")

    # Retry with hint: take a different approach
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {"role": "user", "content": f"{task}\n\nHint: {_triage_hint}"}
        ],
    )
    result = message.content[0].text
    record_step(Step(index=0, action="llm_response", llm_output=result))
    return result

policy = triage.FailurePolicy(
    LOOP_DETECTED=replan(hint="You are stuck in a loop. Try a different approach."),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(claude_agent, policy=policy)

async def main():
    result = await agent.run("Summarise the latest AI research.")
    print(result)

asyncio.run(main())
```

## Run

```bash
python examples/anthropic_agent.py
```
