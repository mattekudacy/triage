# Contributing to triage

Thanks for your interest. This document covers everything you need to get a change merged.

---

## Setup

```bash
git clone https://github.com/mattekudacy/triage
cd triage
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,langgraph,langchain,sqlite,redis]"
```

Python 3.10–3.13 are all supported. The dev extra pulls in pytest, pytest-asyncio, ruff, mypy, and all optional deps.

---

## Running tests

```bash
pytest tests/ -x --tb=short
```

The `-x` flag stops on first failure. **Do not open a PR with failing tests.**

The full matrix (Python 3.10–3.13) runs in CI on every push. You only need to run it locally against your working Python version.

---

## Linting

```bash
ruff check triage/
```

CI enforces this. Fix all violations before pushing — auto-fix handles most of them:

```bash
ruff check triage/ --fix
```

We don't run mypy in CI yet (the codebase uses `Any` in several adapter/OTel paths where strict typing adds noise), but the `py.typed` marker is present and PRs that add new public APIs should be typed.

---

## Project structure

```
triage/           — core library (no framework imports; see rules below)
  taxonomy.py     — FailureType enum, Step, FailureContext
  trajectory.py   — Trajectory
  checkpoint/     — Checkpoint stores (in-memory, SQLite, Redis)
  policy.py       — RecoveryAction, FailurePolicy
  agent.py        — Agent, lifecycle, recovery loop
  classifier/     — RulesClassifier, LLMClassifier, HybridClassifier
  strategies/     — retry, replan, rollback helpers
  adapters/       — LangGraph, LangChain wrappers (framework deps here only)
  observability/  — OTel spans (lazy import, safe without opentelemetry-sdk)
  scorer/         — StepRiskScorer protocol, RulesRiskScorer
tests/            — one file per source module, pytest-asyncio auto mode
examples/         — runnable demos
docs/             — MkDocs documentation
```

### Core import rules

Files inside `triage/` (except `adapters/` and `observability/`) may only import from:
- Python stdlib
- `anyio`
- `pydantic`
- Other `triage.*` sibling modules

No `openai`, `anthropic`, `langchain`, `langgraph`, or `opentelemetry` imports in core. Adapters and optional integrations live in their own packages.

---

## Adding a new FailureType

1. Add the member to `FailureType` in `triage/taxonomy.py` — before `UNKNOWN` (which must stay last).
2. Add a corresponding `StrategyFn | None = None` field to `FailurePolicy` in `triage/policy.py`.
3. Add an entry to `FailurePolicy._FIELD_MAP`.
4. Update `test_failure_type_count` in `tests/test_taxonomy.py` (the `assert len(FailureType) == N` canary).
5. Add classifier coverage in `tests/test_classifier_rules.py` (positive + negative + priority test).

---

## Adding a new RecoveryAction

1. Add a classmethod constructor to `RecoveryAction` in `triage/policy.py` — uppercase name, `None` params excluded from `self.params`.
2. Add handling in `Agent._execute_action()` in `triage/agent.py`.
3. Add tests.

---

## Test conventions

- `asyncio_mode = "auto"` is set in `pyproject.toml`. Never add `@pytest.mark.asyncio`.
- Each test file defines its own local `make_step()` helper — no shared conftest fixtures for triage types.
- Use real triage objects, not mocks (`Step`, `Trajectory`, `FailureContext`, `RecoveryAction`, `FailurePolicy`).
- Test names describe the scenario: `test_loop_not_detected_two_steps`, not `test_classify_returns_unknown_when_len_lt_3`.
- Priority tests use the `_over_` suffix: `test_priority_loop_over_external`.

---

## Opening a PR

- One logical change per PR. Bug fixes don't need surrounding cleanup.
- PRs that add public API should update `docs/api/` and `CHANGELOG.md` (under an `[Unreleased]` heading).
- CI must pass (tests on 3.10–3.13, lint, package build check).

---

## Reporting a bug

Open an issue at https://github.com/mattekudacy/triage/issues. Include:
- The `triage-agent` version (`pip show triage-agent`)
- A minimal reproduction (agent function + policy + the error you see)
- What you expected to happen
