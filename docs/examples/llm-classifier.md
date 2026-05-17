# LLM Classifier Example

Demo: using `LLMClassifier` to detect `CONSTRAINT_IGNORED` — a failure where the agent violates an explicit task constraint in a way that pattern-matching cannot reliably catch.

**Source:** [`examples/llm_classifier.py`](https://github.com/mattekudacy/triage/blob/main/examples/llm_classifier.py)

## Requirements

```bash
pip install "triage-agent[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## What it demonstrates

- `RulesClassifier` requires explicit `constraints=` config to detect `CONSTRAINT_IGNORED` — it won't fire without it
- `LLMClassifier` reads the trajectory and correctly identifies the failure type from context
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

    # Attempt 1: simulate constraint violation
    if _attempt[0] == 1:
        record_step(Step(
            index=0,
            action="web_search",
            tool_called="search",
            tool_input={"q": "unrelated topic"},
            llm_output="I answered in Spanish, ignoring the English-only constraint.",
        ))
        raise RuntimeError(
            "Response language constraint violated — output was not in English."
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
    CONSTRAINT_IGNORED=replan(hint="You must respond in English only."),
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
[triage] constraint_ignored detected at step 0
[triage] Dispatching: RecoveryAction.REPLAN(hint='You must respond in English only.')
Recovery hint: 'You must respond in English only.'
Completed: Summarise the latest research on transformer architectures.
```

## Run

```bash
python examples/llm_classifier.py
```

## Why not just use RulesClassifier?

`RulesClassifier` supports `CONSTRAINT_IGNORED` but requires explicit `constraints=["must respond in English"]` configuration. When constraints come from the task description itself rather than a fixed list, `RulesClassifier` returns `UNKNOWN`.

`LLMClassifier` reads the trajectory, sees the `llm_output` admitting a constraint violation, and returns `CONSTRAINT_IGNORED`. The `replan` strategy fires and the agent corrects itself.

For most production agents, `HybridClassifier` is the right choice — rules for free, LLM only for the ambiguous cases:

```python
from triage.classifier.hybrid import HybridClassifier
classifier = HybridClassifier(llm=LLMClassifier())
```
