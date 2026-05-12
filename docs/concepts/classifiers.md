# Classifiers

A classifier answers one question: **given a trajectory and a task, which `FailureType` best describes what went wrong?**

triage ships three classifiers. They all satisfy the same `Classifier` protocol, so you can swap them in `Agent.__init__` without touching anything else.

## The Classifier protocol

```python
class Classifier(Protocol):
    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...
```

`classify()` is **synchronous** — it must not be `async def`. triage runs it via `anyio.to_thread.run_sync()` inside the async recovery loop, keeping the event loop unblocked even for the LLM-backed classifier.

---

## RulesClassifier

The default. Pattern-based, zero API calls, completes in microseconds.

```python
from triage.classifier.rules import RulesClassifier

clf = RulesClassifier()
clf = RulesClassifier(constraints=["do not use SQL", "output must be JSON"])
```

### Rules (evaluated in priority order)

| Priority | Failure type | Condition |
|---|---|---|
| 1 | `LOOP_DETECTED` | Last 3 steps share identical `tool_called` + canonical `tool_input` |
| 2 | `WRONG_TOOL_CALLED` | Any step's `error` matches `tool.{0,30}not found\|no tool named` |
| 3 | `SCHEMA_MISMATCH` | Any step's `error` matches `validation error\|json.*parse\|jsondecodeerror` |
| 4 | `EXTERNAL_FAULT` | Any step's `error` contains `"429"`, `"500"`, `"502"`, or `"503"` |
| 5 | `CONSTRAINT_IGNORED` | Any step's `llm_output` contains a string from `self.constraints` |
| 6 | `UNKNOWN` | No rule matched |

First match wins. If you need `CONSTRAINT_IGNORED` detection, pass constraint strings when constructing:

```python
clf = RulesClassifier(constraints=[
    "do not hallucinate",
    "must cite sources",
])
```

### Configuring the loop window

The default loop window is 3 steps. If your agent legitimately repeats the same tool call twice in a row (e.g. polling), raise the threshold:

```python
clf = RulesClassifier(loop_window=5)
```

A loop is only declared when `loop_window` consecutive steps share **both** the same `tool_called` **and** the same canonical `tool_input`. Steps with `tool_called=None` are never matched.

### What RulesClassifier cannot detect

`HALLUCINATED_STATE`, `PLAN_INCOMPLETE`, `CONTEXT_OVERFLOW`, and `GOAL_DRIFT` require semantic understanding of the trajectory. Pattern-matching physically cannot detect these. For these failure types, use `LLMClassifier` or `HybridClassifier`. If you use `RulesClassifier` alone and these failure types occur, they will be classified as `UNKNOWN` and routed to your `UNKNOWN` strategy (or escalated if none is set).

| Failure type | RulesClassifier | LLMClassifier / HybridClassifier |
|---|---|---|
| `WRONG_TOOL_CALLED` | ✓ | ✓ |
| `SCHEMA_MISMATCH` | ✓ | ✓ |
| `EXTERNAL_FAULT` | ✓ | ✓ |
| `LOOP_DETECTED` | ✓ | ✓ |
| `CONSTRAINT_IGNORED` | ✓ (with `constraints=`) | ✓ |
| `HALLUCINATED_STATE` | ✗ → `UNKNOWN` | ✓ |
| `PLAN_INCOMPLETE` | ✗ → `UNKNOWN` | ✓ |
| `CONTEXT_OVERFLOW` | ✗ → `UNKNOWN` | ✓ |
| `GOAL_DRIFT` | ✗ → `UNKNOWN` | ✓ |

---

## LLMClassifier

Semantic classifier that asks an LLM to read the trajectory and name the failure type.

```python
from triage.classifier.llm import LLMClassifier
```

### Installation

```bash
# Anthropic backend (Claude)
pip install "triage-agent[anthropic]"

# OpenAI-compatible backend (Ollama, Groq, OpenAI, HuggingFace, etc.)
pip install openai
```

### Anthropic backend (default)

```python
clf = LLMClassifier()  # reads ANTHROPIC_API_KEY from env

clf = LLMClassifier(
    api_key="sk-ant-...",
    model="claude-haiku-4-5-20251001",
    max_trajectory_steps=10,
)
```

### OpenAI-compatible backend

Pass `base_url` to switch to any OpenAI-compatible API:

```python
# Ollama — local, no key needed
clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")

# Groq
clf = LLMClassifier(
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
    model="llama-3.1-8b-instant",
)

# Standard OpenAI
clf = LLMClassifier(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini",
)
```

### BYOK — environment variables

All parameters can be set via environment variables so you can switch providers without changing code:

| Variable | Purpose |
|---|---|
| `TRIAGE_LLM_BASE_URL` | Base URL for any OpenAI-compatible API |
| `TRIAGE_LLM_MODEL` | Model name override |
| `TRIAGE_LLM_API_KEY` | API key fallback |

```bash
# Ollama — no config change needed
TRIAGE_LLM_BASE_URL=http://localhost:11434/v1 \
TRIAGE_LLM_MODEL=llama3.2 \
python agent.py
```

Explicit constructor arguments take precedence over environment variables.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` → env var | API key for the backend |
| `model` | `claude-haiku-4-5-20251001` (Anthropic) or `llama3.2` (OpenAI-compat) | Model name |
| `max_trajectory_steps` | `10` | How many recent steps to include in the prompt |
| `base_url` | `None` | If set, uses OpenAI-compatible backend |

### Fallback behavior

`LLMClassifier` returns `FailureType.UNKNOWN` silently on any error — network failure, rate limit, parse error. This means a degraded LLM classifier degrades gracefully to your `UNKNOWN` strategy rather than crashing the recovery loop.

---

## HybridClassifier

Runs `RulesClassifier` first. Only calls the LLM when rules return `UNKNOWN`.

```python
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier

classifier = HybridClassifier(llm=LLMClassifier())
```

This is the recommended production configuration:

- Rules handle the common cases (loops, HTTP errors, schema failures) for free
- LLM handles the semantically ambiguous cases (`HALLUCINATED_STATE`, `GOAL_DRIFT`, `CONTEXT_OVERFLOW`, `PLAN_INCOMPLETE`)
- LLM is only called when necessary — API cost stays low

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    classifier=HybridClassifier(llm=LLMClassifier()),
)
```

---

## Choosing a classifier

| Classifier | Cost | Accuracy | Use when |
|---|---|---|---|
| `RulesClassifier` | Free | Catches ~60% of failures | Default; production with mostly structural failures |
| `LLMClassifier` | API calls on every failure | Handles all 10 types | Agents with complex reasoning failures |
| `HybridClassifier` | API calls only for `UNKNOWN` | Best of both | Most production agents |

---

## Writing a custom classifier

Any object with a synchronous `classify(trajectory, task) -> FailureType` method satisfies the protocol:

```python
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        # inspect trajectory.steps, task, return a FailureType
        if any("budget exceeded" in (s.error or "") for s in trajectory.steps):
            return FailureType.CONSTRAINT_IGNORED
        return FailureType.UNKNOWN

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```
