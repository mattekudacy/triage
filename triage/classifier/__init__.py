from __future__ import annotations

from triage.classifier.base import Classifier
from triage.classifier.rules import RulesClassifier

__all__ = ["Classifier", "LLMClassifier", "RulesClassifier"]


def __getattr__(name: str) -> object:
    if name == "LLMClassifier":
        from triage.classifier.llm import LLMClassifier
        return LLMClassifier
    raise AttributeError(f"module 'triage.classifier' has no attribute {name!r}")
