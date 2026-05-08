# Ollama Example

Demo using a local Ollama model (Llama 3.2) with the OpenAI-compatible `LLMClassifier`. No API key required.

**Source:** [`examples/ollama_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/ollama_agent.py)

## Requirements

```bash
pip install triage-agent openai

# Install and start Ollama
brew install ollama          # macOS
ollama pull llama3.2
ollama serve                 # starts on http://localhost:11434
```

## What it demonstrates

- `LLMClassifier` pointed at a local Ollama endpoint via `base_url`
- No API key required — Ollama runs entirely locally
- `HybridClassifier`: `RulesClassifier` handles common failures; Ollama/Llama handles the ambiguous ones

## Code

```python
import asyncio
import triage
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier
from triage.strategies.retry import backoff_and_retry
from triage.strategies.replan import replan
from triage.taxonomy import Step

classifier = HybridClassifier(
    llm=LLMClassifier(
        base_url="http://localhost:11434/v1",
        model="llama3.2",
    )
)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=backoff_and_retry(max_attempts=3),
    LOOP_DETECTED=replan(hint="Try a different approach."),
    GOAL_DRIFT=replan(hint="Stay focused on the original task."),
    default=triage.FailurePolicy.escalate_by_default(),
)

async def my_agent(task: str, *, record_step, _triage_hint=None, **_kwargs) -> str:
    # Your agent logic here
    record_step(Step(index=0, action="process", tool_output="result"))
    return f"Completed: {task}"

agent = triage.Agent(my_agent, policy=policy, classifier=classifier)

async def main():
    result = await agent.run("Analyse the quarterly sales data.")
    print(result)

asyncio.run(main())
```

## Environment variable alternative

Instead of passing `base_url` and `model` in code, you can use environment variables:

```bash
TRIAGE_LLM_BASE_URL=http://localhost:11434/v1 \
TRIAGE_LLM_MODEL=llama3.2 \
python your_agent.py
```

And construct the classifier with no arguments:

```python
from triage.classifier.llm import LLMClassifier
classifier = LLMClassifier()  # reads TRIAGE_LLM_BASE_URL and TRIAGE_LLM_MODEL
```

## Run

```bash
python examples/ollama_agent.py
```
