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

Reproduce: `PYTHONPATH=. python scripts/bench_synthetic.py`

### RulesClassifier accuracy

Three-block measurement, `RulesClassifier` default configuration:

| Block | What it measures | Score |
|---|---|---|
| **Regression** | In-corpus positive examples from `test_classifier_rules.py` | 100% (17/17) |
| **False-positive resistance** | Near-miss strings that must NOT fire a rule | 100% (12/12) |
| **Held-out (corpus A)** | Real SDK exceptions not used to write the rules | 100% (30/30) |

Held-out corpus sources: `json.JSONDecodeError`, `asyncio.TimeoutError`, `httpx`
status errors, `pydantic.ValidationError`, `openai` / `anthropic` / `langchain` /
`langgraph` exception strings — provoked and captured, not hand-written.

The regression block is tautological (regexes were written against those strings) —
it detects regression, not generalization. The held-out block is the number to improve
against across releases.

`PLAN_INCOMPLETE`, `CONTEXT_OVERFLOW`, and `CONSTRAINT_IGNORED` are not in the corpus —
`RulesClassifier` returns `UNKNOWN` for the first two by design (semantic types); use
`LLMClassifier` or `HybridClassifier` for those.

Reproduce:
```
PYTHONPATH=. python scripts/classifier_accuracy.py
PYTHONPATH=. python scripts/gen_error_corpus.py  # regenerate corpus
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

The core stores and reloads state; routing the token to Slack, an HTTP callback, or a CLI prompt is userland. Swap `InMemorySuspensionStore` for a Redis-backed store in production so tokens survive process restarts.

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
