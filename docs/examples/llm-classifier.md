# LLM Classifier Example

Demo: using `LLMClassifier` to detect `GOAL_DRIFT` — a failure that pattern-matching cannot catch.

**Source:** [`examples/llm_classifier.py`](https://github.com/mattekudacy/triage/blob/main/examples/llm_classifier.py)

## Requirements

```bash
pip install "triage-agent[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## What it demonstrates

- `RulesClassifier` cannot detect `GOAL_DRIFT` — the error message is semantically ambiguous
- `LLMClassifier` reads the trajectory and correctly identifies the failure type
- `replan` routes the recovery with a focused hint

## Code

```python
import asyncio
import triage
from triage.classifier.llm import LLMClassifier
from triage.strategies.replan import replan
from triage.strategies.retry import backoff_and_retry
from triage.taxonomy import Step

_attempt = [0]

async def research_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    _attempt[0] += 1

    if _triage_hint:
        print(f"Recovery hint: {_triage_hint!r}")

    # Attempt 1: simulate goal drift
    if _attempt[0] == 1:
        record_step(Step(
            index=0,
            action="web_search",
            tool_called="search",
            tool_input={"q": "unrelated topic"},
            llm_output="I got distracted and started researching something else entirely.",
        ))
        raise RuntimeError(
            "The agent appears to have deviated from the original objective "
            "and is now pursuing an unrelated sub-task."
        )

    # Attempt 2 succeeds
    record_step(Step(index=0, action="web_search", tool_called="search",
                     tool_input={"q": task}, tool_output="relevant results"))
    return f"Completed: {task}"

classifier = LLMClassifier(
    model="claude-haiku-4-5-20251001",
    max_trajectory_steps=10,
)

policy = triage.FailurePolicy(
    GOAL_DRIFT=replan(hint="Stay focused on the original task. Do not pursue sub-topics."),
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(research_agent, policy=policy, classifier=classifier)

async def main():
    result = await agent.run("Summarise the latest research on transformer architectures.")
    print(result)

asyncio.run(main())
```

## Expected output

```
[triage] goal_drift detected at step 0
[triage] Dispatching: RecoveryAction.REPLAN(hint='Stay focused on the original task...')
Recovery hint: 'Stay focused on the original task. Do not pursue sub-topics.'
Completed: Summarise the latest research on transformer architectures.
```

## Run

```bash
python examples/llm_classifier.py
```

## Why not just use RulesClassifier?

The error message `"The agent appears to have deviated from the original objective..."` contains no HTTP codes, no tool-not-found string, no JSON error. `RulesClassifier` returns `UNKNOWN`. With `UNKNOWN → escalate` as the default, the agent would be handed off to a human every time.

`LLMClassifier` reads the trajectory, sees the `llm_output` saying the agent "got distracted," and returns `GOAL_DRIFT`. The `replan` strategy fires and the agent corrects itself.

For most production agents, `HybridClassifier` is the right choice — rules for free, LLM only for the ambiguous cases:

```python
from triage.classifier.hybrid import HybridClassifier
classifier = HybridClassifier(llm=LLMClassifier())
```
