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

**Does the routing mechanism work?** A synthetic demo confirms the core claim in principle:
given correct classification, routing to the matching strategy beats blind retry on the two
failure types where the hint actually changes the outcome. `external_fault` heals on any
retry either way — conceding it keeps the comparison honest.

| | no-recovery | triage |
|---|---|---|
| success rate | 50% | 100% |
| recoveries | — | 6 |

Reproduce: `PYTHONPATH=. python scripts/bench_synthetic.py`

**Does classification actually detect those types on error strings it hasn't seen?** That's
the harder, more important question — the demo above assumes correct classification, so it
measures routing, not detection. Scored once against each of three held-out corpora, never
tuned against:

![Recall on three successive held-out corpora: self-healing types hold at 86%, 86%, then 100%,
while routing-sensitive types sit at 8% on corpus C, 8% on corpus D after a full regex-tuning
cycle, and rise to 44% on corpus E only after structured error-code matching
shipped](https://raw.githubusercontent.com/mattekudacy/triage/main/docs/assets/charts/heldout-recall-by-group.png)

Self-healing types (`external_fault`, `timeout`) hold at 86–100% because any retry recovers
them regardless of classification. Routing-sensitive types (`wrong_tool_called`,
`schema_mismatch`) — the ones classification actually has to get right — sat flat at 8%
through an entire tuning cycle (~15 new regex patterns, every one of corpus C's misses closed,
zero movement on the next held-out corpus), then rose to 44% only once matching moved from
message text to a caller-supplied structured error code. Read that as evidence about
regex-based pattern matching as an approach, not just this pattern set: it generalizes only as
far as vendors share a literal vocabulary, which they don't for "no such tool" or "bad schema"
the way they share HTTP status codes.

`LLMClassifier`/`HybridClassifier` close most of that gap — 8% → 83% recall on the same
held-out corpus — but not for free: `RulesClassifier`'s 100%-precision guarantee (every miss
falls safely to `UNKNOWN`) doesn't carry over, and `HybridClassifier` misrouted 3 of 20 entries
in the same measurement.

**Full corpus-by-corpus numbers, four more charts, and what all of this implies for where
classifier effort goes next: see [Known Limitations](docs/known-limitations.md).** Reproduce
everything yourself:

```
PYTHONPATH=. python scripts/classifier_accuracy.py    # the numbers
PYTHONPATH=. python scripts/gen_readme_charts.py      # this chart and 4 more, redrawn from them
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

Prefer not to change your agent's signature? `get_recorder()`/`get_state_updater()`/
`get_usage_recorder()` give you the same callbacks via `contextvars` instead. Already have
OpenTelemetry spans for your tool calls (openllmetry/traceloop, or the OTel GenAI semantic
conventions)? `trajectory_from_spans()` converts them straight into `Step`s instead of you
hand-writing any — see the [OTel Trajectory example](docs/examples/otel-trajectory.md).

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

The default `RulesClassifier` is pattern-based and makes zero API calls. For semantic classification use `LLMClassifier`, or `HybridClassifier` to get the best of both — rules first, LLM only when rules return `UNKNOWN` (see [Results](#results) above for what that trades off):

```python
from triage.classifier.hybrid import HybridClassifier
from triage.classifier.llm import LLMClassifier

agent = triage.Agent(my_agent, policy=policy, classifier=HybridClassifier(llm=LLMClassifier()))
```

`LLMClassifier` supports Anthropic and any OpenAI-compatible provider (Ollama, Groq, OpenAI, HuggingFace) via constructor args or env vars — see [Concepts → Classifiers](docs/concepts/classifiers.md) for the full configuration reference and [Custom classifiers](#custom-classifier) below to bring your own.

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

Any `FailureType` not explicitly listed falls through to `default`. If `default` is also unset, triage escalates automatically. Two composition helpers step through multiple strategies for the same type — `FailurePolicy.chain()` falls through to a fallback within one attempt, `FailurePolicy.sequence()` advances one step per failure across attempts (see the [Policy Chain](docs/examples/policy-chain.md) and [Policy Sequence](docs/examples/policy-sequence.md) examples) — and `FailurePolicy.from_yaml("policy.yaml")` loads a policy from YAML/TOML (needs `triage-agent[yaml]`). Full reference: [Concepts → Policies & Actions](docs/concepts/policies.md).

---

## Built-in strategies

| Module | Provides |
|---|---|
| `triage.strategies.retry` | `retry_with_tool_manifest()`, `backoff_and_retry()` (exponential backoff) |
| `triage.strategies.replan` | `replan(hint=...)`, `resume_from_subgoal()` |
| `triage.strategies.rollback` | `rollback_to_checkpoint(checkpoint_id=None)` |
| `triage.strategies.circuit_breaker` | `circuit_breaker(breaker, strategy)` — wraps any strategy with a cross-run failure-rate guard (`CLOSED → OPEN → HALF_OPEN → CLOSED`); pass `circuit_breakers=[breaker]` to `Agent` so a clean run can close it |

Full signatures and examples: [API Reference → Strategies](docs/api/strategies.md).

---

## Checkpoints

Save agent state at key points so triage can roll back to them on failure. `InMemoryCheckpointStore` is the default; swap in `SQLiteCheckpointStore` (persistent, single-process, `triage-agent[sqlite]`) or `RedisCheckpointStore` (distributed, `triage-agent[redis]`) via `checkpoint_store=`, and pass `auto_checkpoint=True` to checkpoint after every successful step automatically instead of calling a checkpoint API yourself.

```python
from triage.checkpoint.sqlite import SQLiteCheckpointStore

store = SQLiteCheckpointStore("runs/checkpoints.db")
agent = triage.Agent(my_agent, policy=policy, checkpoint_store=store, auto_checkpoint=True)
```

Full backend reference and a custom-store protocol: [Concepts → Checkpoints](docs/concepts/checkpoints.md).

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

`record_step`, `update_state`, and `record_usage` are always injected; recovery actions add `_triage_hint` (`RETRY`/`REPLAN`/`ROLLBACK`), `_triage_subgoal` (`RESUME`), `_triage_state` (`ROLLBACK`, when the checkpoint has state), or the canonical typed `_triage_context: TriageContext` on any action:

```python
async def my_agent(task: str, *, record_step, update_state, _triage_state=None, **kwargs):
    data = _triage_state["data"] if _triage_state else fetch_data(task)   # restored on rollback
    record_step(Step(index=0, action="fetch", tool_output=data))
    update_state({"data": data})
    ...
```

Full contract, including the `contextvars` alternative for agents that shouldn't take new kwargs: [API Reference → Agent](docs/api/agent.md).

Strategies can also inspect everything already tried, via `ctx.attempt_history` — a list of `(failure_type, action_kind)` pairs, empty on the first failure. See [Concepts → Attempt History](docs/concepts/attempt-history.md) for patterns like escalating after N failures or detecting oscillation between two strategies.

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

Install `triage-agent[otel]` and configure a `TracerProvider`/`MeterProvider` — triage auto-detects them, no code change needed. Three spans per `run()` (`triage.run`, `triage.classify`, `triage.dispatch`) and five metric instruments (`triage.runs`, `triage.failures`, `triage.recoveries`, `triage.run.duration`, `triage.recovery.attempts`) are emitted automatically; pass `Agent(tracer=..., meter=...)` to override the auto-detected ones. Every decision also emits a structured log record via the `"triage"` logger regardless of whether OTel is configured — `logging.getLogger("triage").setLevel(logging.INFO)` to see them.

Full span/metric/event reference: [API Reference → Agent](docs/api/agent.md#observability). Runnable demos: [OTel Tracing](docs/examples/otel-tracing.md), [OTel Trajectory](docs/examples/otel-trajectory.md) (the inverse direction — building a `Trajectory` from spans a framework already emits).

---

## Recovery caps and budgets

```python
agent = triage.Agent(
    my_agent,
    policy=policy,
    max_recovery_attempts=3,       # per-run attempt cap (default 3)
    max_total_attempts=10,         # cross-type global cap
    max_recovery_seconds=30.0,     # wall-clock budget for recovery
    max_tokens=50_000,             # escalate instead of recovering past 50k tokens
    max_cost_usd=0.10,             # ...or past $0.10 — report usage via record_usage()
    strict_idempotency=True,       # escalate instead of retrying non-idempotent steps
)
```

Every cap is checked at the failure boundary, not preemptively — an agent that burns budget but never raises runs to completion regardless. `LLMClassifier` reports its own token usage automatically; for agent LLM calls, report via `record_usage(triage.Usage(...))` or `get_usage_recorder()`.

---

## Concurrent runs

A single `Agent` instance is safe for concurrent `run()` calls — per-run state is isolated via `contextvars.ContextVar`. Use `agent.clone()` when you need independent lifecycle hooks or a dedicated checkpoint store per task:

```python
agents = [agent.clone() for _ in tasks]
results = await asyncio.gather(*[ag.run(t) for ag, t in zip(agents, tasks)])
```

---

## Custom classifier

Any class implementing `classify(trajectory, task) -> FailureType` satisfies the `Classifier` protocol — no base class to inherit:

```python
class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```

See [Concepts → Classifiers](docs/concepts/classifiers.md#writing-a-custom-classifier) for the full protocol, including the optional async `aclassify()`.

---

## Learn more

The full documentation site covers everything above in more depth, plus what's not repeated here:

| | |
|---|---|
| [Getting Started](https://mattekudacy.github.io/triage/getting-started/installation/) | Installation, quick start, how it works |
| [Concepts](https://mattekudacy.github.io/triage/concepts/failure-types/) | Failure types, classifiers, policies, checkpoints, attempt history, multi-agent failures |
| [Adapters](https://mattekudacy.github.io/triage/adapters/) | LangGraph, LangChain |
| [Examples](https://mattekudacy.github.io/triage/examples/openai/) | 14 runnable demos — OpenAI, Anthropic, Ollama, Groq, HuggingFace, LangGraph, multi-agent, policy composition, checkpoints, OTel |
| [Known Limitations](docs/known-limitations.md) | Every honest caveat, corpus-by-corpus |
| [API Reference](https://mattekudacy.github.io/triage/api/agent/) | Full signatures for every public class and function |

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
