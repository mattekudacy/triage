<!--
Thanks for opening a PR. Keep it to one logical change — bug fixes don't need
surrounding cleanup, and unrelated refactors should be their own PR.
See CONTRIBUTING.md for the full guide.
-->

**Ownership.** AI-assisted or AI-authored commits are welcome here — human contributor using
an assistant to draft the change, or an automated agent session opening the PR directly — as
long as CI passes and a human reviews the diff before it merges. That review is what makes a
PR accountable, not a footer declaring which tool typed it. Do not add `Co-Authored-By:`
trailers, "Generated with [tool]" badges, session links, or any other AI-attribution footer to
commit messages or this description — no exceptions, including for PRs opened directly by an
automated agent session. If your tooling adds one of these by default, strip it before
pushing/opening; "the platform requires it" is not a reason to leave it in.

**What does this change and why?**
<!-- The failure scenario, bug, or gap this addresses. Link an issue if one exists. -->

**Related issue**
<!-- Fixes #123, or "None" -->

---

**Type of change**
<!-- Check one. -->

- [ ] Bug fix
- [ ] New feature (failure type, recovery strategy, classifier, adapter, checkpoint store, ...)
- [ ] Breaking change (touches anything covered by the [stability commitment](../CHANGELOG.md#stability-commitment) —
      `FailureType` members/values, `Step`/`FailureContext` fields, `RecoveryAction`/`FailurePolicy`
      constructor and kwarg names, `Classifier.classify()`, `Agent.__init__`, `CheckpointStore`, adapter signatures)
- [ ] Documentation
- [ ] Classifier / corpus change (`triage/classifier/rules.py`, `triage/scorer/`, `tests/data/error_corpus_*.json`)

---

**Checklist**

- [ ] No AI-attribution trailers or badges anywhere in this PR — commit messages and this
      description are clean, including if this PR was opened directly by an automated agent
      session. I am the author and I am responsible for this change.
- [ ] `pytest tests/ -x --tb=short` passes locally
- [ ] `ruff check .`, `ruff format --check .`, and `mypy triage/ --strict` are all clean
- [ ] No new imports of `openai`/`anthropic`/`langchain`/`langgraph`/`opentelemetry`/etc. inside
      `triage/` core — only `triage/adapters/` and `triage/observability/` may import optional deps
- [ ] If this adds or changes public API: `docs/api/` and `CHANGELOG.md` (under `[Unreleased]`) are updated
- [ ] If this adds a `FailureType` or `RecoveryAction`: followed the steps in
      [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-failuretype)

**If this touches `rules.py` or an error corpus** — read
[`scripts/README.md`'s "Corpus discipline"](../scripts/README.md#corpus-discipline) first:

- [ ] I did not tune `rules.py` against the current held-out corpus's misses (check
      `scripts/README.md` for which corpus is currently frozen/held-out vs. training data)
- [ ] `scripts/classifier_accuracy.py` was re-run and any changed numbers are reflected in
      README / `CLAUDE.md` / `docs/known-limitations.md` — quote the held-out number together
      with its self-healing/routing-sensitive split, never the aggregate alone

---

**Anything reviewers should look at closely?**
<!-- Tricky regex, a tradeoff you're unsure about, a test you couldn't write, etc. -->
