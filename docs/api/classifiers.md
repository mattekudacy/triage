# Classifiers

All classifiers satisfy the `Classifier` protocol. `Agent` uses `RulesClassifier` by
default; pass `classifier=` to swap it out.

## Classifier protocol

::: triage.classifier.base.Classifier

## ClassificationResult

::: triage.classifier.base.ClassificationResult

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
        - classify_with_confidence
        - aclassify_with_confidence

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

A classifier may additionally define `classify_with_confidence(trajectory, task) ->
ClassificationResult` (and/or its async counterpart `aclassify_with_confidence`) — also
duck-typed, not part of the protocol. `HybridClassifier(llm=my_classifier,
confidence_threshold=0.7)` checks for these methods and, when present, only trusts a
non-`UNKNOWN` answer whose confidence meets the threshold; a classifier without them
(like `RulesClassifier`, which has nothing meaningful to report beyond its
100%-precision-by-construction rules) is unaffected — `confidence_threshold` on a
classifier that can't report confidence is a no-op, not an error:

```python
from triage.classifier.base import ClassificationResult

class MyConfidentClassifier:
    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        ...

    def classify_with_confidence(self, trajectory: Trajectory, task: str) -> ClassificationResult:
        ...  # e.g. ask the model for a confidence score alongside the category
```
