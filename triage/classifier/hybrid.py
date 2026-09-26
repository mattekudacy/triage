"""
triage.classifier.hybrid
~~~~~~~~~~~~~~~~~~~~~~~~
HybridClassifier: runs RulesClassifier first (free, zero API calls), falls
back to LLMClassifier only when the result is UNKNOWN.

Usage::

    from triage.classifier.hybrid import HybridClassifier
    from triage.classifier.llm import LLMClassifier

    clf = HybridClassifier(llm=LLMClassifier())

    # Or with an OpenAI-compatible backend (e.g. Ollama):
    clf = HybridClassifier(
        llm=LLMClassifier(base_url="http://localhost:11434/v1", model="llama3.2")
    )

    # Cap LLM calls within a single Agent.run() call (cost control):
    clf = HybridClassifier(llm=LLMClassifier(), max_llm_calls_per_run=2)

    # Decline a low-confidence LLM guess instead of trusting it (see
    # confidence_threshold's docstring below for why this exists and what
    # it requires of the wrapped classifier):
    clf = HybridClassifier(llm=LLMClassifier(), confidence_threshold=0.7)

The LLM call is only made when rules cannot determine the failure type —
typically < 20% of failures in practice.
"""

from __future__ import annotations

import threading
from typing import Any, cast

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory


class HybridClassifier:
    """Rules-first classifier with LLM fallback on UNKNOWN.

    Satisfies the ``Classifier`` protocol (synchronous ``classify`` method).
    ``LLMClassifier.classify()`` is called in a thread by ``agent.py``
    (via ``anyio.to_thread.run_sync``), so blocking HTTP is safe here.

    ``max_llm_calls_per_run`` bounds how many times the wrapped LLM classifier
    is actually called within one ``Agent.run()`` call — protecting against an
    agent that fails repeatedly from burning an LLM call on every recovery
    attempt. Once the cap is reached, ambiguous (rules-UNKNOWN) failures fall
    straight to ``UNKNOWN`` without touching the LLM. ``None`` (default) means
    unlimited.

    The call counter is a plain, ``threading.Lock``-guarded instance attribute
    rather than a ``ContextVar`` — deliberately, because ``classify()`` may run
    inside ``anyio.to_thread.run_sync()`` (when a classifier has no
    ``aclassify()``, or is called directly outside ``Agent``), and a
    ``ContextVar`` mutated inside a thread dispatch is invisible to the caller
    once that dispatch returns (each ``to_thread.run_sync`` call runs in its
    own copy of the current context). A real shared counter is required for
    the cap to work through *both* the sync (``classify()``-in-a-thread) and
    async (``aclassify()``) dispatch paths.

    ``Agent.run()`` calls ``reset_call_count()`` (if present, duck-typed) once
    at the very start of every ``run()`` call, so the cap is scoped to a single
    run rather than this instance's whole lifetime. If you share one
    ``HybridClassifier`` across multiple *concurrently running* ``Agent``
    instances (or concurrent ``run()`` calls sharing one ``Agent``), the reset
    is best-effort, not isolated — one run's reset can zero out a budget
    another concurrent run was still counting against. Use a separate
    ``HybridClassifier`` (or ``agent.clone()``, which does not share this
    counter) per concurrent task if you need a precise, independent budget.

    ``confidence_threshold`` (default ``None``) gates the LLM fallback on a
    confidence score instead of trusting any non-``UNKNOWN`` answer outright.
    ``None`` (the default) is the original, unconditional behavior — every
    existing caller's behavior is unchanged unless this is explicitly set.
    When set, the wrapped classifier's ``classify_with_confidence()`` /
    ``aclassify_with_confidence()`` method (duck-typed — see
    ``triage.classifier.base.ClassificationResult``) is used instead of
    ``classify()`` / ``aclassify()``, and the fallback only returns a
    non-``UNKNOWN`` answer whose confidence is ``>= confidence_threshold``;
    otherwise it returns ``UNKNOWN``, same as a rules miss with no LLM fallback
    at all. If the wrapped classifier doesn't define a ``*_with_confidence``
    method, this silently falls back to the unconditional behavior — setting
    ``confidence_threshold`` on a classifier that can't report confidence is a
    no-op, not an error, since ``Classifier`` implementations aren't required
    to support it.

    This exists because prompt-level guidance alone, tried first, measurably
    failed to reduce ``HybridClassifier``'s override rate on genuinely
    out-of-taxonomy trajectories — see ``docs/known-limitations.md``'s "close
    the recall gap, but not the precision gap" section for the measurement.
    Pick ``confidence_threshold`` by calibrating against your own labeled
    data, not a default guessed here — a wrapped classifier's self-reported
    confidence has not been independently validated as calibrated for your
    traffic distribution. ``scripts/hybrid_ambiguity_accuracy.py --confidence-
    threshold`` scores this trade-off against the ambiguous corpus.
    """

    def __init__(
        self,
        llm: Any,
        max_llm_calls_per_run: int | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        self._rules = RulesClassifier()
        self._llm = llm
        self._max_llm_calls_per_run = max_llm_calls_per_run
        self._confidence_threshold = confidence_threshold
        self._llm_call_count = 0
        self._count_lock = threading.Lock()

    def reset_call_count(self) -> None:
        """Reset the per-run LLM call counter. Called by ``Agent.run()`` at
        the start of every run — see the class docstring for the concurrency
        caveat when sharing one instance across concurrent runs."""
        with self._count_lock:
            self._llm_call_count = 0

    def _consume_call_budget(self) -> bool:
        """Atomically check-and-increment. Returns True if the call is allowed
        (budget not yet exhausted), False if the cap has been reached."""
        with self._count_lock:
            if (
                self._max_llm_calls_per_run is not None
                and self._llm_call_count >= self._max_llm_calls_per_run
            ):
                return False
            self._llm_call_count += 1
            return True

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        result = self._rules.classify(trajectory, task)
        if result is not FailureType.UNKNOWN:
            return result
        if not self._consume_call_budget():
            return FailureType.UNKNOWN
        if self._confidence_threshold is not None:
            confidence_fn = getattr(self._llm, "classify_with_confidence", None)
            if confidence_fn is not None:
                gated = confidence_fn(trajectory, task)
                if gated.confidence < self._confidence_threshold:
                    return FailureType.UNKNOWN
                return cast(FailureType, gated.failure_type)
        return cast(FailureType, self._llm.classify(trajectory, task))

    async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType:
        """Async counterpart to ``classify()``. Uses ``self._llm.aclassify()``
        when the configured LLM classifier defines one (e.g. ``LLMClassifier``),
        avoiding the sync-client-in-a-thread hop on the failure path. When
        ``confidence_threshold`` is set, prefers ``aclassify_with_confidence()``
        over ``aclassify()``/``classify()`` — see ``__init__``'s docstring.
        """
        result = self._rules.classify(trajectory, task)
        if result is not FailureType.UNKNOWN:
            return result
        if not self._consume_call_budget():
            return FailureType.UNKNOWN
        if self._confidence_threshold is not None:
            aconfidence_fn = getattr(self._llm, "aclassify_with_confidence", None)
            if aconfidence_fn is not None:
                gated = await aconfidence_fn(trajectory, task)
                if gated.confidence < self._confidence_threshold:
                    return FailureType.UNKNOWN
                return cast(FailureType, gated.failure_type)
        aclassify = getattr(self._llm, "aclassify", None)
        if aclassify is not None:
            return cast(FailureType, await aclassify(trajectory, task))
        return cast(FailureType, self._llm.classify(trajectory, task))
