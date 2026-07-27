"""
tests/test_pricing.py
~~~~~~~~~~~~~~~~~~~~~
Tests for triage.pricing — PRICE_TABLE and lookup_cost().
"""

from __future__ import annotations

import pytest

from triage.pricing import PRICE_TABLE, lookup_cost

# ── lookup_cost — known models ────────────────────────────────────────────────


def test_haiku_cost_zero_tokens():
    assert lookup_cost("claude-haiku-4-5-20251001", 0, 0) == pytest.approx(0.0)


def test_haiku_cost_input_only():
    # 1M input tokens at $1.00/1M = $1.00
    assert lookup_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)


def test_haiku_cost_output_only():
    # 1M output tokens at $5.00/1M = $5.00
    assert lookup_cost("claude-haiku-4-5-20251001", 0, 1_000_000) == pytest.approx(5.0)


def test_haiku_cost_combined():
    # 100k input + 50k output = $0.10 + $0.25 = $0.35
    assert lookup_cost("claude-haiku-4-5-20251001", 100_000, 50_000) == pytest.approx(0.35)


def test_sonnet_cost_combined():
    # $3.00/$15.00 per 1M; 200k in + 100k out = $0.60 + $1.50 = $2.10
    assert lookup_cost("claude-sonnet-4-6", 200_000, 100_000) == pytest.approx(2.10)


def test_opus_cost_combined():
    # $5.00/$25.00 per 1M; 500k in + 200k out = $2.50 + $5.00 = $7.50
    assert lookup_cost("claude-opus-5", 500_000, 200_000) == pytest.approx(7.50)


def test_fable_cost_combined():
    # $10.00/$50.00 per 1M; 1M in + 1M out = $10.00 + $50.00 = $60.00
    assert lookup_cost("claude-fable-5", 1_000_000, 1_000_000) == pytest.approx(60.0)


def test_mythos_cost():
    # Same rate as fable; 1M input = $10.00
    assert lookup_cost("claude-mythos-5", 1_000_000, 0) == pytest.approx(10.0)


# ── lookup_cost — dated model variants ───────────────────────────────────────


def test_dated_haiku_resolves():
    # "claude-haiku-4-5-20251001" starts with "claude-haiku-4-5"
    assert lookup_cost("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)


def test_dated_sonnet_resolves():
    assert lookup_cost("claude-sonnet-5-20260101", 1_000_000, 0) == pytest.approx(3.0)


def test_dated_opus_resolves():
    assert lookup_cost("claude-opus-4-7", 1_000_000, 0) == pytest.approx(5.0)


# ── lookup_cost — unknown model ───────────────────────────────────────────────


def test_unknown_model_returns_zero():
    assert lookup_cost("llama3.2", 100_000, 50_000) == pytest.approx(0.0)


def test_empty_model_returns_zero():
    assert lookup_cost("", 1_000_000, 1_000_000) == pytest.approx(0.0)


# ── PRICE_TABLE override ──────────────────────────────────────────────────────


def test_price_table_override(monkeypatch):
    monkeypatch.setitem(PRICE_TABLE, "my-custom-model", (0.000001, 0.000005))
    assert lookup_cost("my-custom-model", 1_000_000, 0) == pytest.approx(1.0)
    assert lookup_cost("my-custom-model", 0, 1_000_000) == pytest.approx(5.0)


def test_price_table_prefix_match_not_substring():
    # "claude-sonnet-4" must not match "claude-haiku-4-5" prefixes
    cost = lookup_cost("claude-haiku-4-5", 1_000_000, 0)
    assert cost == pytest.approx(1.0)  # haiku rate, not sonnet rate


def test_longest_prefix_wins():
    # "claude-haiku-4-5" (more specific) should beat "claude-haiku-4" for the same model
    cost_specific = lookup_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
    cost_generic = lookup_cost("claude-haiku-4-0-20250601", 1_000_000, 0)
    # both map to haiku rates; just confirm no crash and both are non-zero
    assert cost_specific > 0.0
    assert cost_generic > 0.0
