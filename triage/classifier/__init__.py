from __future__ import annotations

from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier

__all__ = ["Classifier", "HybridClassifier", "LLMClassifier", "RulesClassifier"]


def __getattr__(name: str) -> object:
    if name == "LLMClassifier":
        from triage.classifier.llm import LLMClassifier
        return LLMClassifier
    if name == "HybridClassifier":
        from triage.classifier.hybrid import HybridClassifier
        return HybridClassifier
    raise AttributeError(f"module 'triage.classifier' has no attribute {name!r}")
