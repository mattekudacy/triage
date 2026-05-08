# HuggingFace Example

Demo using the HuggingFace Inference API with the OpenAI-compatible `LLMClassifier`. Shows triage recovering from a `SCHEMA_MISMATCH` failure.

**Source:** [`examples/huggingface_agent.py`](https://github.com/mattekudacy/triage/blob/main/examples/huggingface_agent.py)

## Requirements

```bash
pip install triage-agent openai
export HF_TOKEN=hf_...
```

## What it demonstrates

- `LLMClassifier` configured for HuggingFace's OpenAI-compatible Inference API
- `SCHEMA_MISMATCH` recovery: agent fails with a JSON parse error on attempt 1, triage retries with a schema hint
- No local GPU needed — inference runs on HuggingFace's servers

## Code

```python
import asyncio
import json
import os
import triage
from triage.classifier.llm import LLMClassifier
from triage.strategies.retry import retry_with_tool_manifest
from triage.taxonomy import Step

classifier = LLMClassifier(
    base_url="https://api-inference.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    model="mistralai/Mistral-7B-Instruct-v0.3",
)

_attempt = [0]

async def structured_agent(
    task: str, *, record_step, _triage_hint=None, **_kwargs
) -> dict:
    _attempt[0] += 1

    # Simulate a schema mismatch on the first attempt
    if _attempt[0] == 1:
        bad_output = "Here is the result: { missing closing brace"
        record_step(Step(
            index=0,
            action="llm_structured_output",
            llm_output=bad_output,
            error="JSONDecodeError: Expecting property name or '}' at column 35",
        ))
        raise ValueError(
            "JSONDecodeError: Expecting property name or '}' at column 35"
        )

    # Retry succeeds with valid JSON
    good_output = json.dumps({"result": "success", "task": task})
    record_step(Step(index=0, action="llm_structured_output", llm_output=good_output))
    return json.loads(good_output)

policy = triage.FailurePolicy(
    SCHEMA_MISMATCH=retry_with_tool_manifest(max_attempts=2),
    default=triage.FailurePolicy.escalate_by_default(),
)

agent = triage.Agent(structured_agent, policy=policy, classifier=classifier)

async def main():
    result = await agent.run("Extract key information from the document.")
    print(result)

asyncio.run(main())
```

## Run

```bash
python examples/huggingface_agent.py
```
