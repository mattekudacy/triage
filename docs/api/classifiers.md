# Classifiers

All classifiers satisfy the `Classifier` protocol. `Agent` uses `RulesClassifier` by
default; pass `classifier=` to swap it out.

## Classifier protocol

::: triage.classifier.base.Classifier

## RulesClassifier

::: triage.classifier.rules.RulesClassifier
    options:
      members:
        - __init__
        - classify
        - fit

### Rule priority

Rules fire in order; first match wins:

1. `LOOP_DETECTED` — last `loop_window` steps share identical `tool_called` and equal (or fuzzy-similar) `tool_input`
2. `WRONG_TOOL_CALLED` — error matches tool-not-found patterns across OpenAI / Anthropic / LangGraph SDKs
3. `SCHEMA_MISMATCH` — error matches validation / JSON parse patterns
4. `EXTERNAL_FAULT` — error contains an HTTP status code (`429`, `500`, `502`, `503`) as a whole token, not in a quantity context
5. `TIMEOUT` — error matches timeout / deadline patterns
6. `CONSTRAINT_IGNORED` — `llm_output` contains a forbidden constraint string
7. `UNKNOWN` — default

`PLAN_INCOMPLETE` and `CONTEXT_OVERFLOW` require semantic understanding and always return
`UNKNOWN` from `RulesClassifier`. Use `LLMClassifier` or `HybridClassifier` for those.

## LLMClassifier

::: triage.classifier.llm.LLMClassifier
    options:
      members:
        - __init__
        - classify
        - aclassify

## HybridClassifier

::: triage.classifier.hybrid.HybridClassifier
    options:
      members:
        - __init__
        - classify
        - aclassify
        - reset_call_count

---

## Custom classifier

Any object with a synchronous `classify(trajectory, task) -> FailureType` method satisfies
the protocol:

```python
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory

class MyClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType: ...

agent = triage.Agent(my_agent, policy=policy, classifier=MyClassifier())
```

For async classifiers, add an `aclassify` method (duck-typed, not part of the protocol):

```python
class MyAsyncClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...  # sync fallback

    async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...  # async path — used by Agent when present
```
