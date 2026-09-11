# triage

**Classify why your agent failed. Recover intelligently.**

```
pip install triage-agent
```

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The problem

Current agent frameworks know *that* your agent failed. They don't know *why* — and without knowing why, every failure gets the same blunt response: retry from scratch or give up.

`triage` adds a classification-and-routing layer between the failure and the recovery:

```
agent fails → classify failure type → route to matching strategy → recover
```

It works with any async agent callable — OpenAI, LangGraph, raw LLM loops — without requiring you to change your framework.

## Results

### Routing demo (synthetic)

Routing correctness across three failure modes (`RulesClassifier`, zero API calls):

| Task | Failure type | No-recovery baseline | Triage |
|---|---|---|---|
| fetch_weather, call_payments, send_email | `external_fault` (transient 503) | ✓ heals on any retry | ✓ |
| lookup_user, create_ticket | `wrong_tool_called` | ✗ no hint → fails again | ✓ routes to manifest hint |
| parse_response | `schema_mismatch` | ✗ no hint → fails again | ✓ routes to schema hint |

| | no-recovery | triage |
|---|---|---|
| success rate | **50%** | **100%** |
| recoveries | — | 6 |

The 50 % gap is attributable to the two types where classification changes the outcome.
`external_fault` heals on any retry — triage gives no advantage there, and this is
intentional: conceding the transient case makes the comparison honest.

**This demo assumes correct classification.** The error strings here are ones `rules.py`
recognises, so it measures routing, not detection. On held-out error strings
`RulesClassifier` currently detects these same two types at 1/12 — see
[per-type recall](#read-the-recall-number-per-type-not-in-aggregate) below, and read the
two sections together before drawing a conclusion.

Reproduce: `PYTHONPATH=. python scripts/bench_synthetic.py`

### RulesClassifier accuracy

Nine-block measurement, `RulesClassifier` default configuration:

| Block | What it measures | Score |
|---|---|---|
| Regression | In-corpus positives from `test_classifier_rules.py` | 100% (17/17) |
| False-positive resistance | Near-miss strings that must NOT fire a rule | 100% (12/12) |
| Corpus A (training, v0.25) | SDK exceptions used to guide v0.25 fixes | 100% (30/30) |
| Corpus B (training, v0.26 + v1.1) | Guided v0.26 fixes; last 2 misses closed in v1.1 | 100% (20/20) |
| Corpus C (training, v1.1) | Held-out through v1.0 (52%); v1.1 tuned against its misses | 100% (27/27) |
| **Corpus D (held-out, v1.1)** | **Scored once, right after tuning against C** | **40% recall, 100% precision** |

**Corpus C is no longer held-out.** It was genuinely held-out through v1.0 — scored once,
100% precision, 52% recall. The v1.1 release tuned `rules.py` directly against its 13 misses,
which is the exact act that converts a corpus to training data (the same thing that happened
to corpus A in v0.25 and corpus B in v0.26). Its 100% here reflects tuning, not generalization.
**Corpus D is the current held-out measurement**, built from sources disjoint from A, B, and C
(`huggingface_hub`, Ollama, OpenRouter, MCP, CrewAI, Semantic Kernel, novel phrasings) and
scored exactly once, immediately after the v1.1 tuning pass, before any further edit to
`rules.py`. 100% precision — the one misroute this pass found (a name collision between two
unrelated SDKs' same-named exception class) was fixed as a precision bug before freezing this
number, not used to chase recall.

#### Read the recall number per type, not in aggregate — and read across corpora

The 40% aggregate on corpus D averages two groups of failure types whose value to you is
opposite:

| Failure type | Corpus D held-out recall | Does classification change the outcome? |
|---|---|---|
| `timeout` | 3/3 — **100%** | No — any retry heals it |
| `external_fault` | 3/4 — **75%** | No — any retry heals it |
| `schema_mismatch` | 1/4 — **25%** | **Yes — recovery needs the schema hint** |
| `wrong_tool_called` | 0/8 — **0%** | **Yes — recovery needs the manifest hint** |
| | | |
| **Self-healing types** | **6/7 — 86%** | classification buys nothing over blind retry |
| **Routing-sensitive types** | **1/12 — 8%** | classification is the entire value proposition |

**The headline finding of the v1.1 cycle:** routing-sensitive recall on corpus D (1/12 = 8%)
is statistically unchanged from corpus C's *pre-tuning* number — also 1/12 = 8%. The v1.1 pass
added ~15 new patterns, closed all 13 of corpus C's misses, and it did not move the needle on
fresh data at all. Those patterns were narrow string literals keyed close to corpus C's exact
wording (e.g. `"tool with name 'x' was not found"`, an Azure resource-path shape) and simply
didn't overlap with how Ollama, MCP, CrewAI, Semantic Kernel, HuggingFace, and OpenRouter phrase
the same failures. Self-healing recall held at 86% both before and after, for the same reason
it's easy to get right: that group clusters around a small, largely SDK-independent vocabulary
(HTTP status codes, the words "timeout" and "rate limit") that routing-sensitive failures do
not share — every SDK invents its own wording for "no such tool" and "bad request shape."

Put plainly: **pattern-tuning against one held-out corpus's specific misses does not
generalize to a different corpus of the same failure types.** This is evidence about the
regex-pattern-matching approach itself, not just this particular pattern set — see
[Known Limitations](docs/known-limitations.md) for the full writeup and what it implies for
where classifier effort goes next.

**What this means if you are evaluating triage today.** On a stack whose error formats
happen to match the specific strings in `rules.py` (OpenAI, Anthropic, LangChain, botocore,
azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI, LlamaIndex — corpora A, B, and C),
routing works as demonstrated in the synthetic benchmark above. On any other stack — which,
per corpus D, is most of them — expect most tool and schema failures to land in `UNKNOWN`
and fall through to your `default` policy: safe, but no better than the retry loop you would
have written yourself.

#### Does LLMClassifier/HybridClassifier actually close the gap? Measured, not assumed.

`LLMClassifier`/`HybridClassifier` don't have `RulesClassifier`'s wording ceiling by
construction — semantic classification reads the meaning, not a literal string. That claim
sat in this README untested for most of the v1.1 cycle. It's now measured: corpus D scored
with `HybridClassifier(llm=LLMClassifier(model="gpt-oss:120b-cloud"))`, a real reasoning
model via Ollama Cloud (reproduce with `scripts/llm_classifier_accuracy.py`):

| Classifier | Routing-sensitive recall | Misroutes (of 20) |
|---|---|---|
| `RulesClassifier` | 1/12 — 8% | 0 |
| `LLMClassifier` alone | 9–10/12 — 75–83% (two runs) | 4/20 — 20% |
| `HybridClassifier` (recommended) | 10/12 — 83% | 3/20 — 15% |

The recall claim holds: 8% → 83%. But it's not free — `RulesClassifier`'s 100%-precision
guarantee (every miss falls to safe `UNKNOWN`, never a wrong guess) does not carry over.
`HybridClassifier` still misrouted 3 of 20 entries. One misroute is structural, not just LLM
noise: corpus D's one genuinely unclassifiable entry (true label `unknown`) was correctly
left as `UNKNOWN` by `RulesClassifier` — and `HybridClassifier` overturned that correct,
conservative answer into a confident wrong guess anyway, because its fallback rule is
`if rules_result is UNKNOWN: ask the LLM`, with no way to distinguish "rules doesn't
recognize this wording but there's a real answer" from "this genuinely has no answer." Every
LLM-involving run in this measurement misrouted that same entry. (n=1 in corpus D — a
real, reproducible mechanism, not yet a measured rate.) The remaining 2 misroutes were both
in `wrong_tool_called`, at a consistent 6/8 across runs — some tool-not-found phrasings
apparently read as ambiguous to this model even semantically.

Results vary run to run (reasoning-model sampling, not a bug) — these are representative
runs, not a frozen benchmark the way `RulesClassifier`'s corpus D floor is. If you adopt
`HybridClassifier` for routing-sensitive types, budget for occasional confident misroutes,
not just occasional `UNKNOWN`s — a stronger or more expensive model, or a stricter
classification prompt, may trade some recall back for precision if that matters more for
your recovery strategies.

Supply `constraints=`, a `framework=` hint, or please
[open an issue](https://github.com/mattekudacy/triage/issues) with strings that miss.

Corpus D stays frozen. The structural fix that finding pointed at — matching a caller-supplied
structured code in `Step.metadata` (`"http_status"`, `"json_rpc_code"`) rather than message
text — shipped and was scored against a fresh corpus E: routing-sensitive recall rose to 44%
(4/9), but almost entirely via MCP's spec-guaranteed JSON-RPC codes, not HTTP status. Every
fresh HTTP-only vendor's "wrong tool"/"bad schema" failure used a code (`404`/`400`/`422`)
deliberately excluded from the mapping as too ambiguous to resolve safely. See Known
Limitations' "Corpus E scoping" for the full breakdown.

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` are not scored — `RulesClassifier` returns
`UNKNOWN` for them by design; use `LLMClassifier` or `HybridClassifier` for those.

Reproduce:
```
PYTHONPATH=. python scripts/classifier_accuracy.py
```

---

## Installation

```bash
# Core only
pip install triage-agent

# With framework adapters
pip install "triage-agent[langgraph]"
pip install "triage-agent[langchain]"

# With LLM-based classifier
pip install "triage-agent[anthropic]"

# With durable checkpoint storage
pip install "triage-agent[sqlite]"
pip install "triage-agent[redis]"

# With OpenTelemetry spans and metrics
pip install "triage-agent[otel]"

# With YAML/TOML policy config
pip install "triage-agent[yaml]"
```

Python 3.10+ required. Core dependencies: `anyio>=4.0`, `pydantic>=2.0`.

---

## Quick start

```python
import triage
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan
from triage.taxonomy import Step

# 1. Define your agent — it receives record_step and update_state callbacks
async def my_agent(task: str, *, record_step, update_state, _triage_hint=None, **kwargs):
    data = fetch_data(task)
    record_step(Step(index=0, action="called search", tool_called="search",
                     tool_input={"q": task}, tool_output=data))
    update_state({"data": data})   # persisted into checkpoints; restored on rollback
    return "done"

# 2. Declare a recovery policy
policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    default            = triage.FailurePolicy.escalate_by_default(),
)

# 3. Wrap and run
agent = triage.Agent(my_agent, policy=policy)
result = await agent.run("search for recent AI papers")
```

Or use the decorator form:

```python
@triage.agent(policy=policy)
async def my_agent(task: str, *, record_step, **kwargs):
    ...
```

---

## Framework adapters

Drop-in wrappers let you add triage to an existing agent without changing its internals.

### LangGraph

```python
from triage.adapters.langgraph import wrap_langgraph

agent = wrap_langgraph(compiled_graph, policy=policy)
result = await agent.run("your task")
```

Streams events via `graph.astream_events(..., version="v2")` to capture tool calls and LLM turns.

### LangChain

```python
from triage.adapters.langchain import wrap_langchain

agent = wrap_langchain(executor, policy=policy)
result = await agent.run("your task")
```

Injects a fresh `BaseCallbackHandler` per call via `config={"callbacks": [...]}`.

All adapters accept the same optional kwargs as `triage.Agent`: `classifier`, `checkpoint_store`, `max_recovery_attempts`, `auto_checkpoint`.

---

## How it works

### 1. Record steps

Your agent calls `record_step(Step(...))` for each observable action. `triage` injects the callback — you don't need to import or construct anything:

```python
async def my_agent(task: str, *, record_step, **kwargs):
    result = call_tool("search", {"q": task})
    record_step(Step(
        index=0,
        action="called search tool",
        tool_called="search",
        tool_input={"q": task},
        tool_output=result,
    ))
```

Alternatively, avoid signature changes entirely using context-var injection:

```python
from triage.agent import get_recorder, get_state_updater, get_usage_recorder

async def my_agent(task: str, **kwargs):
    record_step = get_recorder()
    update_state = get_state_updater()
    record_usage = get_usage_recorder()
    ...
```

**Already have OpenTelemetry spans?** If your framework emits its own spans for tool calls and
errors (openllmetry/traceloop-style auto-instrumentation, or the OTel GenAI semantic
conventions), you don't have to hand-write `Step`s at all — convert the spans instead:

```python
from triage.observability.otel_ingest import trajectory_from_spans

async def my_agent(task: str, *, record_step, **kwargs):
    try:
        return await already_instrumented_framework.run(task)
    finally:
        for step in trajectory_from_spans(exporter.get_finished_spans()).steps:
            record_step(step)
```

See `examples/otel_trajectory.py` for a full runnable version, including which fields this
extracts reliably (error info, from OTel's stable exception-event convention) versus
best-effort (tool name/input, from the still-Development-stability GenAI conventions).

### 2. Classify the failure

When your agent raises an exception, `triage` runs the classifier over the recorded trajectory and returns one of 9 `FailureType` values:

| FailureType | Trigger | Default recovery |
|---|---|---|
| `WRONG_TOOL_CALLED` | Error matches `"tool not found"` / `"no tool named"` | Retry with correct manifest |
| `CONSTRAINT_IGNORED` | LLM output contains a forbidden string | Replan with constraint reminder |
| `LOOP_DETECTED` | Last 3 steps identical tool + input | Replan or rollback |
| `PLAN_INCOMPLETE` | Success declared but sub-goals incomplete | Resume from subgoal |
| `SCHEMA_MISMATCH` | Error matches `"validation error"` / JSON parse failure | Retry with schema hint |
| `CONTEXT_OVERFLOW` | Agent lost earlier context | Replan with compressed context |
| `EXTERNAL_FAULT` | HTTP 429 / 500 / 502 / 503 in error | Exponential backoff + retry |
| `TIMEOUT` | `timeout` / `timed out` / `deadline exceeded` in error | Backoff and retry |
| `UNKNOWN` | None of the above | Escalate to human |

The default `RulesClassifier` is pattern-based and makes zero API calls. For semantic classification use `LLMClassifier`, or use `HybridClassifier` to get the best of both:

```python
from triage.classifier.llm import LLMClassifier
from triage.classifier.hybrid import HybridClassifier

# LLM only — every failure classified by Claude
agent = triage.Agent(
    my_agent,
    policy=policy,
    classifier=LLMClassifier(model="claude-haiku-4-5-20251001"),
)

# Hybrid — rules first, LLM only when rules return UNKNOWN (~20% of failures)
agent = triage.Agent(
    my_agent,
    policy=policy,
    classifier=HybridClassifier(llm=LLMClassifier()),
)
```

`LLMClassifier` supports Anthropic and any OpenAI-compatible provider. Configure via constructor args or env vars:

```bash
# Anthropic (default)
ANTHROPIC_API_KEY=sk-ant-... python my_agent.py

# Ollama (local, no key)
TRIAGE_LLM_BASE_URL=http://localhost:11434/v1 TRIAGE_LLM_MODEL=llama3.2 python my_agent.py

# Groq
TRIAGE_LLM_BASE_URL=https://api.groq.com/openai/v1 TRIAGE_LLM_API_KEY=gsk_... TRIAGE_LLM_MODEL=llama-3.1-8b-instant python my_agent.py
```

### 3. Dispatch to a strategy

The policy maps each `FailureType` to a strategy callable. The strategy returns a `RecoveryAction` that tells `triage` what to do next.

### 4. Execute the recovery

`triage` executes the action and re-runs your agent with injected context:

| Action | What happens |
|---|---|
| `RETRY` | Re-runs the agent; injects `_triage_hint` into kwargs |
| `REPLAN` | Re-runs the agent; injects `_triage_hint` with new plan instruction |
| `ROLLBACK` | Restores trajectory from checkpoint, re-runs agent |
| `RESUME` | Re-runs agent; injects `_triage_subgoal` pointing at incomplete subgoal |
| `SUSPEND` | Serializes run state to `SuspensionStore`; raises `TriageSuspendedError` |
| `ESCALATE` | Raises `TriageEscalationError(message, context)` |
| `ABORT` | Raises `TriageAbortError(reason, context)` |

---

## Failure policy

`FailurePolicy` is a plain dataclass — one field per `FailureType`:

```python
policy = triage.FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    CONSTRAINT_IGNORED = replan(hint="Re-read the task constraints carefully."),
    LOOP_DETECTED      = replan(max_replans=2),
    PLAN_INCOMPLETE    = resume_from_subgoal(),
    SCHEMA_MISMATCH    = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    default            = triage.FailurePolicy.escalate_by_default(),
)
```

Any `FailureType` not explicitly listed falls through to `default`. If `default` is also unset, triage escalates automatically.

### Sequencing strategies

Step through strategies in order across successive failures of the same type:

```python
policy = triage.FailurePolicy(
    EXTERNAL_FAULT=triage.FailurePolicy.sequence(
        backoff_and_retry(max_attempts=2),
        replan(hint="External service may be down. Try a different approach."),
    ),
)
```

### Loading from config

```python
policy = triage.FailurePolicy.from_yaml("policy.yaml")
policy = triage.FailurePolicy.from_yaml("policy.toml")
```

---

## Built-in strategies

### `triage.strategies.retry`

```python
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry

retry_with_tool_manifest(max_attempts=3)   # retry with hint to use correct manifest
backoff_and_retry(max_attempts=5)          # exponential backoff (2^attempt seconds)
```

### `triage.strategies.replan`

```python
from triage.strategies.replan import replan, resume_from_subgoal

replan(hint="The previous approach used the wrong API endpoint.")
resume_from_subgoal()
```

### `triage.strategies.rollback`

```python
from triage.strategies.rollback import rollback_to_checkpoint

rollback_to_checkpoint()                            # latest checkpoint
rollback_to_checkpoint(checkpoint_id="before-api-call")
```

### `triage.strategies.circuit_breaker`

Wrap any strategy with a cross-run failure-rate guard:

```python
from triage.breaker import CircuitBreaker
from triage.strategies.circuit_breaker import circuit_breaker

breaker = CircuitBreaker(failure_threshold=5, window_seconds=60, cooldown_seconds=30)

policy = triage.FailurePolicy(
    EXTERNAL_FAULT=circuit_breaker(breaker, backoff_and_retry(max_attempts=3)),
)

# Notify the breaker when a run completes cleanly (closes HALF_OPEN state)
agent = triage.Agent(my_agent, policy=policy, circuit_breakers=[breaker])
```

States: `CLOSED` → `OPEN` (threshold reached) → `HALF_OPEN` (cooldown elapsed) → `CLOSED` (probe succeeds). When `OPEN`, recovery is skipped and `TriageEscalationError` is raised immediately.

---

## Checkpoints

Save agent state at key points so triage can roll back to them on failure.

### In-memory (default)

```python
store = triage.InMemoryCheckpointStore()
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### SQLite (persistent, single-process)

```python
from triage.checkpoint.sqlite import SQLiteCheckpointStore

store = SQLiteCheckpointStore("runs/checkpoints.db")
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### Redis (distributed)

```python
import redis.asyncio as aioredis
from triage.checkpoint.redis import RedisCheckpointStore

client = aioredis.Redis.from_url("redis://localhost:6379")
store = RedisCheckpointStore(client)
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store)
```

### Auto-checkpoint

Enable automatic checkpointing after every successful step:

```python
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

---

## Human-in-the-loop pause/resume

When a failure needs a human decision, use `SUSPEND` instead of `ESCALATE`. The run pauses, serializes its state, and returns a token. Call `agent.resume(token, action=...)` with the human's decision to continue from the exact point of suspension.

```python
from triage.suspension import InMemorySuspensionStore

suspension_store = InMemorySuspensionStore()

async def needs_approval(ctx: triage.FailureContext) -> triage.RecoveryAction:
    return triage.RecoveryAction.SUSPEND(
        message=f"Agent hit {ctx.failure_type.value} — approve recovery?",
        metadata={"channel": "#ops"},   # routing hints for your notification layer
    )

policy = triage.FailurePolicy(EXTERNAL_FAULT=needs_approval)
agent = triage.Agent(my_agent, policy=policy, suspension_store=suspension_store)

try:
    result = await agent.run("task")
except triage.TriageSuspendedError as e:
    token = e.token
    # route token to human (Slack, webhook, CLI — your code, not triage's)
    notify_human(token, e.run.message, e.run.metadata)

# ... later, after human responds ...
result = await agent.resume(token, action=triage.RecoveryAction.RETRY())
# or: action=triage.RecoveryAction.ABORT(reason="rejected")
# or: action=triage.RecoveryAction.REPLAN(hint="try a different approach")
```

The core stores and reloads state; routing the token to Slack, an HTTP callback, or a CLI prompt is userland. Swap `InMemorySuspensionStore` for `RedisSuspensionStore` in production so tokens survive process restarts — **experimental**: tested, but with no known production users as of this writing; see the module docstring in `triage/suspension_redis.py`.

---

## Recovery context in your agent

Three callbacks are always injected, plus recovery context on retry:

```python
async def my_agent(
    task: str,
    *,
    record_step,
    update_state,
    record_usage,          # report token/cost usage for budget tracking
    _triage_hint=None,
    _triage_subgoal=None,
    _triage_state=None,
    **kwargs,
):
    if _triage_state:
        data = _triage_state["data"]   # restored from checkpoint on rollback
    else:
        data = fetch_data(task)

    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})

    response = await call_llm(prompt)
    record_usage(triage.Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=0.0001,
    ))
```

| Key | Set when |
|---|---|
| `record_step` | Always |
| `update_state` | Always |
| `record_usage` | Always |
| `_triage_hint` | `RETRY`, `REPLAN`, or `ROLLBACK` action |
| `_triage_subgoal` | `RESUME` action |
| `_triage_state` | `ROLLBACK` action, when checkpoint has non-empty state |
| `_triage_context` | All recovery actions — typed `TriageContext` object |

---

## Token and cost budgets

Cap the total tokens or dollars spent per `run()` call. The check fires at each failure point — if the budget is already exceeded when the agent raises, triage escalates instead of attempting recovery.

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    max_tokens=50_000,      # escalate after 50k tokens total
    max_cost_usd=0.10,      # escalate after $0.10 total
)
```

`LLMClassifier` automatically reports its own token usage to the meter. For agent LLM calls, report via `record_usage(triage.Usage(...))` in the agent body or via `get_usage_recorder()`.

---

## Attempt history

Strategies can inspect everything that was tried before they were called:

```python
async def smart_strategy(ctx: triage.FailureContext) -> triage.RecoveryAction:
    replan_count = sum(1 for _, kind in ctx.attempt_history if kind == "replan")
    if replan_count >= 2:
        return triage.RecoveryAction.ESCALATE(message="Replanned twice, still failing.")
    return triage.RecoveryAction.REPLAN(hint="Try a different approach.")
```

`attempt_history` is empty on the first failure and grows by one entry per recovery attempt. Each entry is `(failure_type, action_kind)` where `action_kind` is one of `"retry"`, `"replan"`, `"rollback"`, `"resume"`, `"suspend"`, `"escalate"`, `"abort"`.

---

## Handling escalation, abort, and suspension

```python
try:
    result = await agent.run(task)
except triage.TriageSuspendedError as e:
    # Run paused — route e.token to a human for a decision
    # Call agent.resume(e.token, action=...) to continue
    notify_human(e.token, e.run.message)
except triage.TriageEscalationError as e:
    # Automatic recovery exhausted — needs human review
    print(f"Failure type: {e.context.failure_type.value}")
    print(f"Trajectory: {[s.action for s in e.context.trajectory]}")
except triage.TriageAbortError as e:
    print(f"Hard stop: {e}")
```

### `on_escalate` hook

Intercept escalations before they raise — useful for last-chance recovery or routing:

```python
async def on_escalate(ctx: triage.FailureContext) -> triage.RecoveryAction | None:
    if ctx.failure_type == triage.FailureType.EXTERNAL_FAULT:
        await notify_oncall(ctx)
        return triage.RecoveryAction.SUSPEND(message="on-call notified")
    return None  # proceed with escalation

agent = triage.Agent(my_agent, policy=policy, on_escalate=on_escalate)
```

---

## Lifecycle hooks

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    on_step=lambda step: print(f"step {step.index}: {step.action}"),
    on_failure=lambda ctx: metrics.increment(f"failure.{ctx.failure_type.value}"),
    on_recovery=lambda ctx, action: print(f"recovering via {action.kind}"),
)
```

Hook exceptions are swallowed with a warning so they never interrupt a run.

---

## Observability

### OpenTelemetry spans

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace

trace.set_tracer_provider(TracerProvider(...))

# triage auto-detects the configured provider — no explicit tracer needed
agent = triage.Agent(my_agent, policy=policy)
```

Three spans per `run()` call: `triage.run` (root), `triage.classify` (per failure), `triage.dispatch` (per recovery). Pass `Agent(tracer=my_tracer)` to override. Install: `pip install "triage-agent[otel]"`.

### OpenTelemetry metrics

Five instruments emitted automatically when a `MeterProvider` is configured:

| Instrument | Type | Attributes |
|---|---|---|
| `triage.runs` | Counter | `outcome` |
| `triage.failures` | Counter | `failure_type` |
| `triage.recoveries` | Counter | `failure_type`, `action_kind` |
| `triage.run.duration` | Histogram | `outcome` |
| `triage.recovery.attempts` | UpDownCounter | `failure_type` |

Pass `Agent(meter=my_meter)` to override the auto-detected meter.

### Structured log events

All triage decisions emit structured log records via the `"triage"` logger:

```python
import logging
logging.getLogger("triage").setLevel(logging.INFO)
```

Events: `failure_classified`, `action_dispatched`, `retry_backoff`, `attempt_start`, `run_suspended`, `hook_error`. Each includes `extra={"triage_event": ..., ...}`.

---

## Recovery caps

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    max_recovery_attempts=3,       # per-run attempt cap (default 3)
    max_total_attempts=10,         # cross-type global cap
    max_recovery_seconds=30.0,     # wall-clock budget for recovery
    max_tokens=50_000,             # token budget
    max_cost_usd=0.10,             # cost budget
    strict_idempotency=True,       # escalate instead of retrying non-idempotent steps
)
```

---

## Concurrent runs

A single `Agent` instance is safe for concurrent `run()` calls — per-run state (trajectory, checkpoints, usage meter) is isolated via `contextvars.ContextVar`. Use `agent.clone()` when you need independent lifecycle hooks or a dedicated checkpoint store per task:

```python
agents = [agent.clone() for _ in tasks]
results = await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks)])
```

---

## Custom classifier

Any class implementing `classify(trajectory, task) -> FailureType` satisfies the protocol:

```python
from triage.classifier.base import Classifier
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```

---

## Example: OpenAI tool-calling loop

See [`examples/raw_openai.py`](examples/raw_openai.py) for a full working example that deliberately triggers a `WRONG_TOOL_CALLED` failure on the first attempt:

```bash
OPENAI_API_KEY=sk-... python examples/raw_openai.py
```

---

## Project layout

```
triage/
  taxonomy.py          FailureType enum (9 types), Step, FailureContext, TriageContext
  trajectory.py        Trajectory (append / replay_from / last_n_steps)
  checkpoint/
    base.py            Checkpoint, CheckpointStore protocol, serialization helpers
    memory.py          InMemoryCheckpointStore
    sqlite.py          SQLiteCheckpointStore (requires aiosqlite)
    redis.py           RedisCheckpointStore (requires redis[asyncio])
  policy.py            RecoveryAction (7 constructors), FailurePolicy
  agent.py             Agent, TriageEscalationError, TriageAbortError,
                         TriageSuspendedError, @agent decorator
  suspension.py        SuspendedRun, SuspensionStore protocol,
                         InMemorySuspensionStore
  breaker.py           CircuitBreaker, BreakerState
  usage.py             Usage, UsageMeter
  classifier/
    base.py            Classifier protocol
    rules.py           RulesClassifier — 6 rules, sync, zero API calls
    llm.py             LLMClassifier — Anthropic or OpenAI-compatible backend
    hybrid.py          HybridClassifier — rules first, LLM fallback on UNKNOWN
  strategies/
    retry.py           retry_with_tool_manifest(), backoff_and_retry()
    replan.py          replan(), resume_from_subgoal()
    rollback.py        rollback_to_checkpoint()
    circuit_breaker.py circuit_breaker()
  adapters/
    langgraph.py       wrap_langgraph() (requires langgraph)
    langchain.py       wrap_langchain() (requires langchain)
  observability/
    otel.py            Span helpers (lazy OTel import)
    metrics.py         Metric helpers (lazy OTel import)
  scorer/
    base.py            StepRiskScorer protocol, RiskScore
    rules.py           RulesRiskScorer — destructive pattern detection
  bench.py             run_benchmark(), BenchReport, BenchResult
  feedback.py          Correction, record_correction(), load_corrections()
  testing.py           make_step(), RecordingAgent, assert_classifies_as()
```

---

## License

MIT
