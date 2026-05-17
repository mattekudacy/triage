# Groq Example

Demo using the Groq API (Llama 3.1) with the OpenAI-compatible `LLMClassifier`. Very fast inference.

**Source:** [`examples/groq_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/groq_agent.py)

## Requirements

```bash
pip install triage-agent openai
export GROQ_API_KEY=gsk_...
```

## What it demonstrates

- `LLMClassifier` configured for Groq's OpenAI-compatible endpoint
- Fast semantic failure classification via Llama 3.1 on Groq's accelerated hardware
- `HybridClassifier` to minimize API calls: rules handle common failures, Groq handles ambiguous ones

## Code

```python
import asyncio
import os
import triage
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan
from triage.taxonomy import Step

classifier = HybridClassifier(
    llm=LLMClassifier(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.1-8b-instant",
    )
)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    LOOP_DETECTED=replan(hint="Try a different approach."),
    CONSTRAINT_IGNORED=replan(hint="Stay focused on the original task."),
    default=triage.FailurePolicy.escalate_by_default(),
)

async def my_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    record_step(Step(index=0, action="process", tool_output="result"))
    return f"Completed: {task}"

agent = triage.Agent(my_agent, policy=policy, classifier=classifier)

async def main():
    result = await agent.run("Analyse the quarterly sales data.")
    print(result)

asyncio.run(main())
```

## Environment variable alternative

```bash
TRIAGE_LLM_BASE_URL=https://api.groq.com/openai/v1 \
TRIAGE_LLM_API_KEY=gsk_... \
TRIAGE_LLM_MODEL=llama-3.1-8b-instant \
python your_agent.py
```

## Run

```bash
python examples/groq_agent.py
```
