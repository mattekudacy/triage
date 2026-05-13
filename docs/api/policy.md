# FailurePolicy

`FailurePolicy` is a dataclass that maps each `FailureType` to a strategy callable. It is the user-facing configuration object.

## Declaration

```python
from triage import FailurePolicy
from triage.strategies.retry import retry_with_tool_manifest, backoff_and_retry
from triage.strategies.replan import replan, resume_from_subgoal
from triage.strategies.rollback import rollback_to_checkpoint

policy = FailurePolicy(
    WRONG_TOOL_CALLED  = retry_with_tool_manifest(max_attempts=3),
    SCHEMA_MISMATCH    = retry_with_tool_manifest(max_attempts=2),
    EXTERNAL_FAULT     = backoff_and_retry(max_attempts=5),
    LOOP_DETECTED      = replan(hint="Try a different approach."),
    CONSTRAINT_IGNORED = replan(hint="Re-read the constraints carefully."),
    HALLUCINATED_STATE = rollback_to_checkpoint(),
    PLAN_INCOMPLETE    = resume_from_subgoal(),
    default            = FailurePolicy.escalate_by_default(),
)
```

## Fields

Each field corresponds to a `FailureType` member. All are optional (`None` by default). Any type without a registered strategy falls through to `default`.

| Field | Type |
|---|---|
| `WRONG_TOOL_CALLED` | `StrategyFn \| None` |
| `CONSTRAINT_IGNORED` | `StrategyFn \| None` |
| `LOOP_DETECTED` | `StrategyFn \| None` |
| `HALLUCINATED_STATE` | `StrategyFn \| None` |
| `PLAN_INCOMPLETE` | `StrategyFn \| None` |
| `SCHEMA_MISMATCH` | `StrategyFn \| None` |
| `CONTEXT_OVERFLOW` | `StrategyFn \| None` |
| `GOAL_DRIFT` | `StrategyFn \| None` |
| `EXTERNAL_FAULT` | `StrategyFn \| None` |
| `UNKNOWN` | `StrategyFn \| None` |
| `default` | `StrategyFn \| None` |

`StrategyFn = Callable[[FailureContext], Awaitable[RecoveryAction]]`

## Methods

### resolve(failure_type)

```python
def resolve(self, failure_type: FailureType) -> StrategyFn | None
```

Returns the strategy for the given type, falling back to `default` if not set. Returns `None` if neither is set.

### dispatch(ctx)

```python
async def dispatch(self, ctx: FailureContext) -> RecoveryAction
```

Calls `resolve(ctx.failure_type)` and awaits the strategy. If no strategy is found, returns `RecoveryAction.ESCALATE(...)`.

### chain(primary, fallback, after_kinds=("escalate",))

```python
@staticmethod
def chain(primary: StrategyFn, fallback: StrategyFn, after_kinds: tuple[str, ...] = ("escalate",)) -> StrategyFn
```

Returns a strategy that tries `primary` and falls through to `fallback` when `primary` returns an action whose `kind` is in `after_kinds`. Default is to fall through when primary escalates.

```python
from triage.strategies.replan import replan
from triage.strategies.rollback import rollback_to_checkpoint

# Replan first; if replan escalates (e.g. max_replans reached), rollback instead
policy = triage.FailurePolicy(
    LOOP_DETECTED=FailurePolicy.chain(
        replan(hint="Try a different approach."),
        rollback_to_checkpoint(),
    ),
)
```

```python
# Retry up to 2 times, then replan
policy = triage.FailurePolicy(
    EXTERNAL_FAULT=FailurePolicy.chain(
        backoff_and_retry(max_attempts=2),
        replan(hint="External service is down. Try a different approach."),
        after_kinds=("escalate",),
    ),
)
```

### escalate_by_default()

```python
@staticmethod
def escalate_by_default() -> StrategyFn
```

Returns a strategy that always returns `RecoveryAction.ESCALATE`. Use as `default=` to ensure every unhandled failure is escalated.

### abort_by_default()

```python
@staticmethod
def abort_by_default() -> StrategyFn
```

Returns a strategy that always returns `RecoveryAction.ABORT`. Use as `default=` for a hard-stop on any unhandled failure.
