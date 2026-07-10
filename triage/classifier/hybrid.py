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

The LLM call is only made when rules cannot determine the failure type —
typically < 20% of failures in practice.
"""

from __future__ import annotations

from triage.classifier.rules import RulesClassifier
from triage.taxonomy import FailureType
from triage.trajectory import Trajectory


class HybridClassifier:
    """Rules-first classifier with LLM fallback on UNKNOWN.

    Satisfies the ``Classifier`` protocol (synchronous ``classify`` method).
    ``LLMClassifier.classify()`` is called in a thread by ``agent.py``
    (via ``anyio.to_thread.run_sync``), so blocking HTTP is safe here.
    """

    def __init__(self, llm: object) -> None:
        self._rules = RulesClassifier()
        self._llm = llm

    def classify(self, trajectory: Trajectory, task: str) -> FailureType:
        result = self._rules.classify(trajectory, task)
        if result is FailureType.UNKNOWN:
            return self._llm.classify(trajectory, task)  # type: ignore[union-attr]
        return result

    async def aclassify(self, trajectory: Trajectory, task: str) -> FailureType:
        """Async counterpart to ``classify()``. Uses ``self._llm.aclassify()``
        when the configured LLM classifier defines one (e.g. ``LLMClassifier``),
        avoiding the sync-client-in-a-thread hop on the failure path.
        """
        result = self._rules.classify(trajectory, task)
        if result is not FailureType.UNKNOWN:
            return result
        aclassify = getattr(self._llm, "aclassify", None)
        if aclassify is not None:
            return await aclassify(trajectory, task)
        return self._llm.classify(trajectory, task)  # type: ignore[union-attr]
