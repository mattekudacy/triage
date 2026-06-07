# Step Risk Scoring — Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/skill.md` to implement this plan task-by-task.

**Goal:** Add a `StepRiskScorer` that scores each step in real time via the `on_step` hook and raises `TriageAbortError` before the agent reaches an unrecoverable state — intercepting non-idempotent side effects *before* they happen rather than classifying failures after.

**Architecture:** A `StepRiskScorer` is a callable that receives a `Step` and the current `Trajectory` and returns a `RiskScore` (float 0–1 + optional reason). `Agent` accepts a new `risk_scorer` init param; when set, `_record_step` calls it after appending the step, and aborts if score exceeds `risk_threshold` (default 0.9). A `RulesRiskScorer` provides a zero-API default that flags destructive patterns (email send, payment charge, DELETE/DROP, external writes) in `step.action` and `step.tool_called`.

**Tech Stack:** Python stdlib only. No new dependencies. `triage/scorer/` new sub-package. All existing tests unaffected.

---

## Task 1: `RiskScore` dataclass and `StepRiskScorer` protocol

**Files:**
- Create: `triage/scorer/__init__.py`
- Create: `triage/scorer/base.py`
- Create: `tests/test_scorer_base.py`

**Step 1: Write the failing test**

```python
# tests/test_scorer_base.py
from triage.scorer.base import RiskScore, StepRiskScorer
from triage.taxonomy import Step
from triage.trajectory import Trajectory

def test_risk_score_fields():
    rs = RiskScore(score=0.8, reason="destructive action")
    assert rs.score == 0.8
    assert rs.reason == "destructive action"

def test_risk_score_defaults():
    rs = RiskScore(score=0.1)
    assert rs.reason is None

def test_risk_score_clamps_above_one():
    rs = RiskScore(score=1.5)
    assert rs.score == 1.0

def test_risk_score_clamps_below_zero():
    rs = RiskScore(score=-0.1)
    assert rs.score == 0.0

def test_step_risk_scorer_is_callable_protocol():
    from triage.scorer.base import StepRiskScorer
    from typing import runtime_checkable
    # Any callable with the right signature satisfies the protocol
    def my_scorer(step, trajectory):
        return RiskScore(score=0.0)
    assert callable(my_scorer)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_scorer_base.py -v
```
Expected: ImportError / ModuleNotFoundError

**Step 3: Create `triage/scorer/__init__.py`**

```python
from __future__ import annotations
```

**Step 4: Create `triage/scorer/base.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from triage.taxonomy import Step
    from triage.trajectory import Trajectory


@dataclass
class RiskScore:
    """Risk assessment for a single recorded step."""

    score: float
    reason: str | None = None

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))


@runtime_checkable
class StepRiskScorer(Protocol):
    """Synchronous per-step risk scorer. Must not make API calls."""

    def __call__(self, step: "Step", trajectory: "Trajectory") -> RiskScore:
        ...
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_scorer_base.py -v
```
Expected: 5 PASSED

**Step 6: Commit**

```bash
git add triage/scorer/__init__.py triage/scorer/base.py tests/test_scorer_base.py
git commit -m "feat: add RiskScore dataclass and StepRiskScorer protocol"
```

---

## Task 2: `RulesRiskScorer` — zero-API pattern-based scorer

**Files:**
- Create: `triage/scorer/rules.py`
- Create: `tests/test_scorer_rules.py`

**Step 1: Write the failing tests**

```python
# tests/test_scorer_rules.py
import pytest
from triage.scorer.rules import RulesRiskScorer
from triage.taxonomy import Step
from triage.trajectory import Trajectory

def make_step(action="test step", tool_called=None, error=None, idempotent=False):
    return Step(index=0, action=action, tool_called=tool_called,
                error=error, idempotent=idempotent)

def traj(*steps):
    t = Trajectory()
    for s in steps:
        t.append(s)
    return t

def test_low_risk_read_only_step():
    scorer = RulesRiskScorer()
    step = make_step(action="search results", tool_called="search", idempotent=True)
    result = scorer(step, traj(step))
    assert result.score < 0.5

def test_high_risk_email_action():
    scorer = RulesRiskScorer()
    step = make_step(action="send_email to user@example.com")
    result = scorer(step, traj(step))
    assert result.score >= 0.9
    assert result.reason is not None

def test_high_risk_payment_tool():
    scorer = RulesRiskScorer()
    step = make_step(tool_called="charge_card")
    result = scorer(step, traj(step))
    assert result.score >= 0.9

def test_high_risk_delete_action():
    scorer = RulesRiskScorer()
    step = make_step(action="DELETE FROM users WHERE id=1")
    result = scorer(step, traj(step))
    assert result.score >= 0.9

def test_high_risk_drop_table():
    scorer = RulesRiskScorer()
    step = make_step(action="DROP TABLE payments")
    result = scorer(step, traj(step))
    assert result.score >= 0.9

def test_medium_risk_external_write():
    scorer = RulesRiskScorer()
    step = make_step(action="write file to disk", tool_called="file_write")
    result = scorer(step, traj(step))
    assert result.score >= 0.6

def test_non_idempotent_flag_raises_base_score():
    scorer = RulesRiskScorer()
    step = make_step(action="update_record", idempotent=False)
    result = scorer(step, traj(step))
    assert result.score > 0.0

def test_idempotent_flag_keeps_score_low():
    scorer = RulesRiskScorer()
    step = make_step(action="get_data", idempotent=True)
    result = scorer(step, traj(step))
    assert result.score < 0.5

def test_custom_high_risk_patterns():
    scorer = RulesRiskScorer(high_risk_patterns=["nuke_db", "wipe_all"])
    step = make_step(action="nuke_db")
    result = scorer(step, traj(step))
    assert result.score >= 0.9

def test_score_returns_risk_score_instance():
    from triage.scorer.base import RiskScore
    scorer = RulesRiskScorer()
    step = make_step()
    result = scorer(step, traj(step))
    assert isinstance(result, RiskScore)
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_scorer_rules.py -v
```
Expected: ImportError

**Step 3: Create `triage/scorer/rules.py`**

```python
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from triage.scorer.base import RiskScore

if TYPE_CHECKING:
    from triage.taxonomy import Step
    from triage.trajectory import Trajectory

_HIGH_RISK_RE = re.compile(
    r"\b(send_email|send_message|charge_card|payment|delete\b|drop\s+table"
    r"|rm\s+-rf|truncate|destroy|nuke|wipe|purge)\b",
    re.IGNORECASE,
)
_MEDIUM_RISK_RE = re.compile(
    r"\b(write|upload|post|put|patch|update|insert|create|deploy|publish)\b",
    re.IGNORECASE,
)


class RulesRiskScorer:
    """Pattern-based step risk scorer. Zero API calls, synchronous.

    Parameters
    ----------
    high_risk_patterns:
        Additional patterns to treat as high-risk (score 0.95). ORed with
        built-in patterns.
    medium_risk_patterns:
        Additional patterns to treat as medium-risk (score 0.65).
    """

    def __init__(
        self,
        high_risk_patterns: list[str] | None = None,
        medium_risk_patterns: list[str] | None = None,
    ) -> None:
        extra_high = "|".join(re.escape(p) for p in (high_risk_patterns or []))
        extra_med = "|".join(re.escape(p) for p in (medium_risk_patterns or []))
        self._high_re = (
            re.compile(f"{_HIGH_RISK_RE.pattern}|{extra_high}", re.IGNORECASE)
            if extra_high else _HIGH_RISK_RE
        )
        self._med_re = (
            re.compile(f"{_MEDIUM_RISK_RE.pattern}|{extra_med}", re.IGNORECASE)
            if extra_med else _MEDIUM_RISK_RE
        )

    def __call__(self, step: "Step", trajectory: "Trajectory") -> RiskScore:
        text = " ".join(filter(None, [step.action, step.tool_called]))

        if self._high_re.search(text):
            return RiskScore(score=0.95, reason=f"high-risk pattern in: {text!r}")

        if self._med_re.search(text):
            base = 0.65
            if not step.idempotent:
                base = min(1.0, base + 0.1)
            return RiskScore(score=base, reason=f"medium-risk pattern in: {text!r}")

        if not step.idempotent:
            return RiskScore(score=0.2, reason="step marked non-idempotent")

        return RiskScore(score=0.0)
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_scorer_rules.py -v
```
Expected: 10 PASSED

**Step 5: Commit**

```bash
git add triage/scorer/rules.py tests/test_scorer_rules.py
git commit -m "feat: add RulesRiskScorer with pattern-based step risk scoring"
```

---

## Task 3: Wire `risk_scorer` and `risk_threshold` into `Agent`

**Files:**
- Modify: `triage/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: Write the failing tests** (add at end of `tests/test_agent.py`)

```python
# ── step risk scoring ─────────────────────────────────────────────────────────

async def test_risk_scorer_aborts_on_high_score():
    from triage.scorer.base import RiskScore

    def always_risky(step, trajectory):
        return RiskScore(score=1.0, reason="always abort")

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="dangerous action"))
        return "should not reach"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=always_risky, risk_threshold=0.9)
    with pytest.raises(TriageAbortError) as exc_info:
        await ag.run("task")
    assert "always abort" in str(exc_info.value)


async def test_risk_scorer_does_not_abort_below_threshold():
    from triage.scorer.base import RiskScore

    def low_risk(step, trajectory):
        return RiskScore(score=0.5)

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="safe step"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=low_risk, risk_threshold=0.9)
    result = await ag.run("task")
    assert result == "ok"


async def test_risk_scorer_none_does_not_affect_run():
    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="step"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy)  # no risk_scorer
    result = await ag.run("task")
    assert result == "ok"


async def test_risk_scorer_receives_trajectory():
    received = []

    def capture_scorer(step, trajectory):
        from triage.scorer.base import RiskScore
        received.append(len(trajectory.steps))
        return RiskScore(score=0.0)

    async def agent_fn(task, *, record_step, update_state, **kwargs):
        record_step(Step(index=0, action="step 1"))
        record_step(Step(index=1, action="step 2"))
        return "ok"

    policy = FailurePolicy()
    ag = Agent(agent_fn, policy, risk_scorer=capture_scorer)
    await ag.run("task")
    assert received == [1, 2]  # trajectory grows with each step


def test_clone_copies_risk_scorer_and_threshold():
    from triage.scorer.base import RiskScore

    def scorer(step, trajectory):
        return RiskScore(score=0.0)

    policy = FailurePolicy()
    ag = Agent(lambda t, **kw: None, policy, risk_scorer=scorer, risk_threshold=0.8)
    cloned = ag.clone()
    assert cloned._risk_scorer is scorer
    assert cloned._risk_threshold == 0.8
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_agent.py -k "risk_scor" -v
```
Expected: errors about unknown keyword argument `risk_scorer`

**Step 3: Update `Agent.__init__`** — add after `strict_idempotency` param:

```python
risk_scorer: "StepRiskScorer | None" = None,
risk_threshold: float = 0.9,
```

Store:
```python
self._risk_scorer = risk_scorer
self._risk_threshold = risk_threshold
```

Add to the lazy import block at top of agent.py:
```python
# StepRiskScorer is imported lazily to avoid circular import at module level
from triage.scorer.base import StepRiskScorer as _StepRiskScorer
```
(Or use `TYPE_CHECKING` guard — whichever matches the existing pattern in agent.py.)

**Step 4: Update `_record_step`** — after `_safe_hook(self._on_step, step)`:

```python
if self._risk_scorer is not None:
    risk = self._risk_scorer(step, self._trajectory)
    if risk.score >= self._risk_threshold:
        raise TriageAbortError(
            f"step risk score {risk.score:.2f} >= threshold {self._risk_threshold}"
            + (f": {risk.reason}" if risk.reason else ""),
            None,
        )
```

Note: `TriageAbortError` takes `(message, context)` — pass `None` for context since we don't have a `FailureContext` yet (no failure has occurred).

**Step 5: Update `clone()`** — add:

```python
risk_scorer=self._risk_scorer,
risk_threshold=self._risk_threshold,
```

**Step 6: Run tests to verify they pass**

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_agent.py -k "risk_scor" -v
```
Expected: 5 PASSED

**Step 7: Run full suite**

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q --tb=short
```
Expected: all pass, 2 skipped

**Step 8: Commit**

```bash
git add triage/agent.py tests/test_agent.py
git commit -m "feat: wire risk_scorer and risk_threshold into Agent"
```

---

## Task 4: Exports and CLAUDE.md

**Files:**
- Modify: `triage/__init__.py`
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml` (version bump to 0.9.0)

**Step 1: Update `triage/__init__.py`**

Add to imports:
```python
from triage.scorer.base import RiskScore, StepRiskScorer
from triage.scorer.rules import RulesRiskScorer
```

Add to `__all__`:
```python
"RiskScore",
"RulesRiskScorer",
"StepRiskScorer",
```

**Step 2: Update `pyproject.toml`**

```toml
version = "0.9.0"
```

Also update `triage/__init__.py`:
```python
__version__ = "0.9.0"
```

**Step 3: Update `CLAUDE.md`**

Add to repo layout under `scorer/`:
```
  scorer/
    base.py             — RiskScore dataclass, StepRiskScorer protocol
    rules.py            — RulesRiskScorer — pattern-based, zero API calls
```

Add to v0.9 changes section:
```
- **Step risk scoring** — `RulesRiskScorer` scores each step as it's recorded;
  `Agent(risk_scorer=..., risk_threshold=0.9)` raises `TriageAbortError` before
  the agent can execute a step scoring >= threshold; `RulesRiskScorer` detects
  destructive patterns (email, payment, DELETE, DROP TABLE) with zero API calls.
```

**Step 4: Run full suite one final time**

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q --tb=short
```
Expected: all pass, 2 skipped

**Step 5: Commit**

```bash
git add triage/__init__.py CLAUDE.md pyproject.toml
git commit -m "feat: export RiskScore, StepRiskScorer, RulesRiskScorer; bump to v0.9.0"
```
