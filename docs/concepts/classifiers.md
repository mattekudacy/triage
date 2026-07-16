# Classifiers

A classifier answers one question: **given a trajectory and a task, which `FailureType` best describes what went wrong?**

triage ships three classifiers. They all satisfy the same `Classifier` protocol, so you can swap them in `Agent.__init__` without touching anything else.

## The Classifier protocol

```python
class Classifier(Protocol):
    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...
```

`classify()` is **synchronous** — it must not be `async def`. triage runs it via `anyio.to_thread.run_sync()` inside the async recovery loop, keeping the event loop unblocked even for classifiers that make a blocking HTTP call.

### Optional: aclassify() for native-async classification

Classifiers that talk to an LLM API may additionally define:

```python
async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType: ...
```

This is not part of the `Classifier` protocol itself — it's duck-typed. When present, `agent.py` awaits it directly instead of dispatching `classify()` to a thread, skipping that hop entirely. `LLMClassifier` and `HybridClassifier` both define `aclassify()` using the native async Anthropic/OpenAI client; `RulesClassifier` has no I/O and doesn't need one. You get this automatically — no configuration required, just use `LLMClassifier`/`HybridClassifier` as your `classifier=`.

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

### Fuzzy loop detection

By default, `tool_input` must match **exactly** across the window. Real agents often rework a query slightly on each retry — `{"q": "revenue Q1"}` then `{"q": "revenue for Q1"}` — which the exact-match rule misses even though it's still a loop. Set `loop_similarity_threshold` (in `(0.0, 1.0]`) to catch these:

```python
clf = RulesClassifier(loop_similarity_threshold=0.9)
```

Similarity is computed with `difflib.SequenceMatcher.ratio()` on the canonical JSON string form of `tool_input`, comparing each step **against the previous step** in the window (not all steps against the first) — so a query that drifts gradually across the window is still caught, even if the first and last steps have drifted far apart from each other. `tool_called` must still match exactly across the whole window; only `tool_input` gets the fuzzy comparison.

Default is `None` — exact match only, unchanged from pre-v0.12 behavior. This is opt-in: existing code that doesn't pass `loop_similarity_threshold` sees no behavior change.

A threshold around `0.85`–`0.95` is a reasonable starting point; lower values risk false-positiving on genuinely different queries that happen to share a lot of characters (e.g. two searches with the same long boilerplate prefix).

### Accuracy on the synthetic suite

The benchmark in `examples/benchmark.py` runs 26 trajectories covering all structurally-detectable failure types plus known negative cases (inputs that should **not** match). Results as of v0.14:

| Failure type | Cases | Pass | Notes |
|---|---|---|---|
| `LOOP_DETECTED` | 2 | 2 | Exact-match window = 3 |
| `WRONG_TOOL_CALLED` | 4 | 4 | OpenAI, Anthropic, generic patterns |
| `SCHEMA_MISMATCH` | 4 | 4 | JSONDecodeError, pydantic, invalid json |
| `EXTERNAL_FAULT` | 4 | 4 | 429, 500, 502, 503 |
| `CONSTRAINT_IGNORED` | 2 | 2 | With `constraints=` set |
| `UNKNOWN` (negatives) | 10 | 10 | No false positives |
| **Total** | **26** | **26** | **100%** |

These numbers reflect the synthetic test cases, not production data. Real-world accuracy depends on your framework's error message format, language, and SDK version. Run `python examples/benchmark.py` to test against the same suite locally, or add your own cases to the `CASES` list.

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` are intentionally absent from the table — they require semantic understanding and are never detected by `RulesClassifier` regardless of trajectory content.

### What RulesClassifier cannot detect

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` require semantic understanding of the trajectory. Pattern-matching physically cannot detect these. For these failure types, use `LLMClassifier` or `HybridClassifier`. If you use `RulesClassifier` alone and these failure types occur, they will be classified as `UNKNOWN` and routed to your `UNKNOWN` strategy (or escalated if none is set).

| Failure type | RulesClassifier | LLMClassifier / HybridClassifier |
|---|---|---|
| `WRONG_TOOL_CALLED` | ✓ | ✓ |
| `SCHEMA_MISMATCH` | ✓ | ✓ |
| `EXTERNAL_FAULT` | ✓ | ✓ |
| `TIMEOUT` | ✓ | ✓ |
| `LOOP_DETECTED` | ✓ | ✓ |
| `CONSTRAINT_IGNORED` | ✓ (with `constraints=`) | ✓ |
| `PLAN_INCOMPLETE` | ✗ → `UNKNOWN` | ✓ |
| `CONTEXT_OVERFLOW` | ✗ → `UNKNOWN` | ✓ |

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
| `max_retries` | `1` | Retries for transient errors before falling back to `UNKNOWN` |
| `retry_backoff_base` | `0.5` | Backoff seconds; doubles each retry (`0.5s`, `1s`, ...) |

### Retrying transient errors

As of v0.11, both `classify()` and `aclassify()` retry on errors that look transient — HTTP `429`/`500`/`502`/`503`/`529`, or an exception class named `RateLimitError`, `APITimeoutError`, `APIConnectionError`, or `InternalServerError` (matched by name, not by importing the real SDK exception classes, so this works regardless of which backend you're on). Non-transient errors (bad request, auth failure, parse error) are never retried — they fall straight to `UNKNOWN` on the first attempt.

```python
clf = LLMClassifier(max_retries=2, retry_backoff_base=1.0)  # up to 3 attempts total
clf = LLMClassifier(max_retries=0)                          # disable retries entirely
```

Keep this budget small — classification runs on the failure path, and every retry adds latency on top of an agent that's already failing. The default (`max_retries=1`, `retry_backoff_base=0.5`) adds at most ~0.5s before falling back to `UNKNOWN`.

### Fallback behavior

`LLMClassifier` returns `FailureType.UNKNOWN` silently on any error — network failure, rate limit, parse error — once the retry budget (if any) is exhausted. This means a degraded LLM classifier degrades gracefully to your `UNKNOWN` strategy rather than crashing the recovery loop. Both `classify()` and `aclassify()` share this fallback and retry behavior.

### Native async via aclassify()

`LLMClassifier` defines `async def aclassify(trajectory, task) -> FailureType`, backed by `AsyncAnthropic`/`AsyncOpenAI` instead of the sync client. `agent.py` detects and awaits this directly, avoiding the `anyio.to_thread.run_sync()` hop that `classify()` still needs. The sync and async clients are built and cached independently — calling both `classify()` and `aclassify()` on the same `LLMClassifier` instance creates one of each, not a shared client.

---

## HybridClassifier

Runs `RulesClassifier` first. Only calls the LLM when rules return `UNKNOWN`.

```python
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier

classifier = HybridClassifier(llm=LLMClassifier())
```

`HybridClassifier` also defines `aclassify()`: it runs `RulesClassifier` synchronously (as before), and on `UNKNOWN` calls `self._llm.aclassify()` if the wrapped LLM classifier defines one — falling back to `self._llm.classify()` otherwise. No extra config needed; passing `HybridClassifier(llm=LLMClassifier())` as `classifier=` gets the async path automatically.

This is the recommended production configuration:

- Rules handle the common cases (loops, HTTP errors, schema failures) for free
- LLM handles the semantically ambiguous cases (`CONTEXT_OVERFLOW`, `PLAN_INCOMPLETE`)
- LLM is only called when necessary — API cost stays low

### Capping LLM calls per run

An agent that keeps failing ambiguously within one `Agent.run()` call — retry, fail, retry, fail — will call the LLM once per recovery attempt by default. `max_llm_calls_per_run` caps that:

```python
classifier = HybridClassifier(llm=LLMClassifier(), max_llm_calls_per_run=2)
```

Once the cap is hit, `HybridClassifier` returns `UNKNOWN` for any further rules-ambiguous failure in that run, without calling the LLM. `Agent.run()` resets the counter (via `reset_call_count()`, duck-typed — checked with `getattr`) at the start of every run, so the budget applies per run, not per classifier lifetime.

If you share one `HybridClassifier` instance across multiple agents or concurrent `run()` calls, the reset is best-effort rather than strictly isolated per task — a concurrent run's reset can zero out a budget another run was still counting against. Note that `agent.clone()` shares the *same* classifier instance as the original, so it does not give you an independent budget either. Construct a separate `HybridClassifier(llm=...)` per concurrent task if you need a precise, independent budget per task.

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

If your custom classifier makes an async API call, add an optional `aclassify()` method — `agent.py` will detect and await it directly instead of running `classify()` in a thread:

```python
class MyAsyncClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...  # sync fallback path, e.g. anyio.from_thread or a blocking client

    async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType:
        # inspect trajectory.steps, task, await your async client, return a FailureType
        ...
```
