"""
triage.pricing
~~~~~~~~~~~~~~
Per-model price table for automatic cost_usd computation in LLMClassifier.

Rates are stored as USD per token (not per 1 000 000 tokens).
The table covers Anthropic-hosted models; OpenAI-compatible backends
default to 0.0 since triage has no authoritative price data for them.

Override or extend the table at runtime::

    from triage.pricing import PRICE_TABLE
    PRICE_TABLE["my-custom-model"] = (0.000001, 0.000005)  # input, output per token

Rates source: Anthropic pricing, 2026-06-24.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger("triage")
_unknown_model_warned: set[str] = set()

# Keys are model ID prefixes; longest match wins.
# Values are (input_per_token_usd, output_per_token_usd).
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0 / 1_000_000, 50.0 / 1_000_000),
    "claude-mythos-5": (10.0 / 1_000_000, 50.0 / 1_000_000),
    "claude-opus-5": (5.0 / 1_000_000, 25.0 / 1_000_000),
    "claude-opus-4": (5.0 / 1_000_000, 25.0 / 1_000_000),
    "claude-sonnet-5": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "claude-sonnet-4": (3.0 / 1_000_000, 15.0 / 1_000_000),
    "claude-haiku-4-5": (1.0 / 1_000_000, 5.0 / 1_000_000),
    "claude-haiku-4": (1.0 / 1_000_000, 5.0 / 1_000_000),
}


def lookup_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD for the given model and token counts.

    Matches ``model`` by longest prefix, so both the canonical short IDs
    (``"claude-haiku-4-5"``) and dated variants
    (``"claude-haiku-4-5-20251001"``) resolve correctly.
    Returns ``0.0`` for unrecognised models — cost reporting is best-effort.
    """
    for key in sorted(PRICE_TABLE, key=len, reverse=True):
        if model.startswith(key):
            in_rate, out_rate = PRICE_TABLE[key]
            return input_tokens * in_rate + output_tokens * out_rate
    if model and model not in _unknown_model_warned:
        _unknown_model_warned.add(model)
        _logger.warning(
            "[triage] unknown model for cost estimation — cost_usd will be 0.0; "
            "add an entry to triage.pricing.PRICE_TABLE to enable cost tracking",
            extra={"triage_event": "unknown_model_cost", "model": model},
        )
    return 0.0
