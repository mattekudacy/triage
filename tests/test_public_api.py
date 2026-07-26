"""
tests/test_public_api.py
~~~~~~~~~~~~~~~~~~~~~~~~
Verify that every name in triage.__all__ and triage.classifier.__all__
is importable — including names served by the lazy __getattr__ path.
A typo in a lazily-exported name would pass all other tests but fail
for any user who does `from triage import X`.
"""

from __future__ import annotations

import triage
import triage.classifier


def test_triage_all_names_importable():
    for name in triage.__all__:
        obj = getattr(triage, name)
        assert obj is not None, f"triage.{name} resolved to None"


def test_classifier_all_names_importable():
    for name in triage.classifier.__all__:
        obj = getattr(triage.classifier, name)
        assert obj is not None, f"triage.classifier.{name} resolved to None"
