# Classifiers

All classifiers satisfy the `Classifier` protocol:

```python
class Classifier(Protocol):
    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...
```

`classify()` is always `def` (not `async def`). triage runs it via `anyio.to_thread.run_sync()` to keep the event loop unblocked.

Classifiers may optionally define `async def aclassify(self, trajectory, task) -> FailureType`. Duck-typed, not part of the protocol; when present, `agent.py` awaits it directly instead of using the thread path. `LLMClassifier` and `HybridClassifier` both define it.

---

## RulesClassifier

```python
from triage.classifier.rules import RulesClassifier
```

Pattern-based, zero API calls, microseconds per call.

```python
clf = RulesClassifier()
clf = RulesClassifier(constraints=["must return JSON", "no markdown"])
clf = RulesClassifier(loop_similarity_threshold=0.9)  # fuzzy loop detection
```

**Rules (priority order):**

1. `LOOP_DETECTED` — last 3 steps: same `tool_called` + identical canonical `tool_input` (or, with `loop_similarity_threshold` set, consecutively similar `tool_input` per `difflib.SequenceMatcher.ratio()`)
2. `WRONG_TOOL_CALLED` — error matches `tool.{0,30}not found|no tool named`
3. `SCHEMA_MISMATCH` — error matches `validation error|json.*parse|jsondecodeerror`
4. `EXTERNAL_FAULT` — error contains `"429"`, `"500"`, `"502"`, or `"503"`
5. `CONSTRAINT_IGNORED` — any step's `llm_output` contains a constraint string
6. `UNKNOWN` — no rule matched

---

## LLMClassifier

```python
from triage.classifier.llm import LLMClassifier
```

Semantic classifier. Calls an LLM to read the trajectory and name the failure type.

```python
# Anthropic (default)
clf = LLMClassifier()
clf = LLMClassifier(api_key="sk-ant-...", model="claude-haiku-4-5-20251001")

# OpenAI-compatible
clf = LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
clf = LLMClassifier(base_url="https://api.groq.com/openai/v1",
                    api_key="gsk_...", model="llama-3.1-8b-instant")
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `None` → `TRIAGE_LLM_API_KEY` env var | API key |
| `model` | `claude-haiku-4-5-20251001` (Anthropic) or `llama3.2` (OpenAI-compat) | Model name |
| `max_trajectory_steps` | `10` | Steps included in the prompt |
| `base_url` | `None` → `TRIAGE_LLM_BASE_URL` env var | If set, uses OpenAI-compatible client |
| `max_retries` | `1` | Retries for transient errors (429/5xx/timeout/connection) before falling back to `UNKNOWN` |
| `retry_backoff_base` | `0.5` | Seconds; backoff is `retry_backoff_base * 2 ** attempt` |

Falls back to `UNKNOWN` silently on any error.

Also defines `async def aclassify(trajectory, task) -> FailureType`, backed by `AsyncAnthropic`/`AsyncOpenAI`. `agent.py` prefers this over `classify()` when present — no thread hop.

**Install:**
```bash
pip install "triage-agent[anthropic]"   # Anthropic backend
pip install openai                       # OpenAI-compatible backend
```

---

## HybridClassifier

```python
from triage.classifier.hybrid import HybridClassifier
```

Runs `RulesClassifier` first; calls the LLM only when rules return `UNKNOWN`.

```python
clf = HybridClassifier(llm=LLMClassifier())
```

Recommended for production: free for structural failures, semantic fallback for ambiguous ones.

Also defines `aclassify()` — calls `llm.aclassify()` when the wrapped LLM classifier has one, else falls back to `llm.classify()`.

**Cost cap:** `HybridClassifier(llm=..., max_llm_calls_per_run=N)` bounds LLM calls per `Agent.run()` call; the counter resets via `reset_call_count()` (duck-typed, called by `agent.py`) at the start of each run.

---

## Custom classifiers

Any class with a synchronous `classify()` method satisfies the protocol:

```python
class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        if any("budget" in (s.error or "") for s in trajectory.steps):
            return FailureType.CONSTRAINT_IGNORED
        return FailureType.UNKNOWN

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```
