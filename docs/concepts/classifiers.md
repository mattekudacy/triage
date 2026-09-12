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

### Structured error codes

Every rule above also checks a caller-supplied structured code in `Step.metadata`, in addition to its message-text pattern — a second, independent signal for the same failure type, not a separate rule. It's opt-in: nothing in `triage` populates `Step.metadata` automatically, so if you never set it, behavior is unchanged.

```python
async def my_agent(task: str, *, record_step, **kwargs) -> Any:
    try:
        result = await call_tool(...)
    except httpx.HTTPStatusError as e:
        record_step(Step(
            index=0, action="call_tool",
            error=str(e),
            metadata={"http_status": e.response.status_code},
        ))
        raise
```

Two keys are recognized:

| `metadata` key | Type | Source | Mapped `FailureType`s |
|---|---|---|---|
| `"http_status"` | `int` | `anthropic`/`openai` `APIStatusError.status_code`, `httpx.HTTPStatusError.response.status_code`, Ollama `ResponseError.status_code`, etc. | `429/500/502/503` → `EXTERNAL_FAULT`; `408/504` → `TIMEOUT` |
| `"json_rpc_code"` | `int` | An MCP `McpError`'s `error.code` (JSON-RPC 2.0) | `-32601` → `WRONG_TOOL_CALLED`; `-32700` → `SCHEMA_MISMATCH`; `-32603` → `EXTERNAL_FAULT` |

Only codes with an **unambiguous** single-`FailureType` mapping are matched. `404`/`400` HTTP statuses and JSON-RPC `-32602` ("Invalid params" — shared by both a bad tool name and a malformed argument shape) are deliberately excluded: a code that maps to more than one failure type would turn `RulesClassifier`'s zero-false-positive guarantee into a coin flip. An excluded or absent code simply falls through to the message-text rules, same as if `metadata` carried nothing at all.

`-32600` ("Invalid Request") is excluded too, despite the JSON-RPC spec describing it as an unambiguous "malformed request" code — corpus E (see `docs/known-limitations.md`'s "Corpus E scoping") found a real MCP server reusing it for a session/auth-lifecycle condition, not a malformed request. A code the spec defines cleanly and real servers use loosely is not safe to match on; only `-32700` (Parse error — can only mean the request body failed to parse as JSON) stayed in the `SCHEMA_MISMATCH` mapping.

This exists to test whether a structural signal — a stable field or protocol code, rather than free-text SDK wording — generalizes across SDKs better than pattern tuning does. See `docs/known-limitations.md`'s "Corpus E scoping" section for the full rationale and what would still need to happen (a corpus built with real codes, and a per-framework extraction helper) before this changes measured accuracy on unfamiliar stacks.

### Accuracy

`RulesClassifier` scores 100% on its own regression suite and on corpora A–C (`tests/`,
`tests/data/error_corpus_{a,b,c}.json`) — expected, since those cases guided the patterns
`rules.py` matches on. That number is training data, not evidence of generalization.

The number that means something is the held-out one, scored once against sources the
patterns were never tuned against: **69% recall, 100% precision on corpus E** (corpus D,
its predecessor, sits at 40%). It splits sharply by failure type — self-healing types
(`external_fault`, `timeout`) are easy because any retry recovers them regardless of
classification; routing-sensitive types (`wrong_tool_called`, `schema_mismatch`) are the
ones classification actually has to get right, and that recall is far lower. Reproduce it:

```bash
PYTHONPATH=. python scripts/classifier_accuracy.py
```

See the README's "RulesClassifier accuracy" section for the headline chart, and
[Known Limitations](../known-limitations.md#accuracy-is-corpus-dependent) for the full
per-corpus, per-type breakdown and what it implies about pattern-matching as an approach.

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` are intentionally absent from any of this — they
require semantic understanding and are never detected by `RulesClassifier` regardless of
trajectory content.

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
| `TRIAGE_LLM_MAX_TOKENS` | Output token budget fallback (default 32 — see [Output budget and reasoning models](#output-budget-and-reasoning-models)) |

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
| `max_tokens` | `32` (or `TRIAGE_LLM_MAX_TOKENS`) | Output token budget for the classification call |

### Multi-agent trajectories: `agent_id` in the prompt

Each step's prompt line includes `agent: <id>` when `Step.agent_id` is set, so a trajectory that interleaves steps from more than one agent doesn't look identical to a single-agent one. `None` (the default) omits the line — zero prompt change for existing single-agent callers. This is a prerequisite for, not itself an instance of, MAST-mode detection: `LLMClassifier` still only ever returns one of the 9 stable `FailureType` members — see `docs/concepts/multi-agent-failures.md`'s phase 3 scoping section for what a real MAST-mode classification prompt would need beyond this.

### Output budget and reasoning models

As of v1.2, `max_tokens` is configurable — previously hardcoded to `32`. That default is enough for a plain instruct model to emit one category word (`"external_fault"`), but a *reasoning* model (gpt-oss, o1/o3-style, DeepSeek-R1, Qwen3 "thinking" mode, ...) can spend the entire budget on hidden reasoning tokens before ever emitting the answer. The failure is silent and easy to misdiagnose: the API call succeeds, `finish_reason` comes back `"length"`, content is `""`, and `classify()` returns `UNKNOWN` — indistinguishable from a real auth or network failure unless you inspect the raw response yourself.

```python
clf = LLMClassifier(base_url="https://ollama.com/v1",
                    model="gpt-oss:120b-cloud", max_tokens=500)
```

Or via env var: `TRIAGE_LLM_MAX_TOKENS=500`. A few hundred tokens is usually enough headroom for a reasoning model's hidden thinking plus the final word. If you're not sure whether your model needs this, test it directly: a `finish_reason` of `"length"` with empty `content` on a plain "say ok" prompt at the default budget is the tell.

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

### Accuracy

On the seven structural types corpora A–E cover, `LLMClassifier` closes the recall gap
`RulesClassifier`'s pattern-tuning couldn't (8% → 83% routing-sensitive recall on corpus D
with a real model) at the cost of `RulesClassifier`'s 100%-precision guarantee — every rules
miss falls safely to `UNKNOWN`; the LLM sometimes guesses wrong instead. See the README's
"Does LLMClassifier/HybridClassifier actually close the gap?" section for the measured
numbers (`scripts/llm_classifier_accuracy.py`), and
[Known Limitations](../known-limitations.md#llmclassifierhybridclassifier-close-the-recall-gap-but-not-the-precision-gap)
for the misroute breakdown.

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` are the two types `RulesClassifier` cannot detect at
all — `LLMClassifier`/`HybridClassifier` are the only options for them. There is no held-out
corpus for these two yet (corpora A–E only cover the seven structural types), so — consistent
with this project's own rule against claiming a number without held-out evidence — no
accuracy figure is claimed for them here. `tests/test_classifier_llm.py` covers the parsing
and dispatch mechanism with mocked responses; it does not measure real-model recall.

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

### Accuracy

`HybridClassifier` matches `RulesClassifier` exactly wherever rules don't return `UNKNOWN`
(zero API calls), and falls to the LLM path everywhere else — so its accuracy is the LLM
section above wherever rules are `UNKNOWN`, and `RulesClassifier`'s 100%-precision floor
everywhere else. Measured on corpus D: 83% routing-sensitive recall, 3/20 misroutes — one of
them a structural risk worth knowing about, not just noise: `HybridClassifier` can't tell
"rules doesn't recognize this wording" from "this genuinely has no answer," so it can
overturn a correctly-conservative rules `UNKNOWN` into a confident wrong guess. See
[Known Limitations](../known-limitations.md#llmclassifierhybridclassifier-close-the-recall-gap-but-not-the-precision-gap)
for the full mechanism and reproduce with `scripts/llm_classifier_accuracy.py`.

---

## Choosing a classifier

| Classifier | Cost | Types covered | Use when |
|---|---|---|---|
| `RulesClassifier` | Free | 7 of 9 (structural only) | Default; production with mostly structural failures |
| `LLMClassifier` | API call on every failure | All 9 | Agents with complex reasoning failures |
| `HybridClassifier` | API call only for `UNKNOWN` | All 9 | Most production agents — best cost/coverage tradeoff |

`RulesClassifier` cannot detect `PLAN_INCOMPLETE` or `CONTEXT_OVERFLOW` — these always return `UNKNOWN`. If your agents produce these failure types, use `LLMClassifier` or `HybridClassifier`, and test against your own trajectories before deploying — there is no held-out corpus for these two types yet (see the "Accuracy" section above).

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
