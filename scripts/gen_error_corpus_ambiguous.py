"""
scripts/gen_error_corpus_ambiguous.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate tests/data/error_corpus_ambiguous.json — a corpus built to answer one
question the corpus D measurement raised but couldn't settle: how often does
HybridClassifier overturn a correct, conservative RulesClassifier UNKNOWN into
a confident wrong guess?

Corpus D had exactly one entry with true label `unknown` (a genuinely
out-of-taxonomy IAM/permission string). RulesClassifier correctly classified
it UNKNOWN; HybridClassifier overturned it into a wrong concrete guess in
every LLM-involving run. That's a confirmed *mechanism*
(HybridClassifier.classify() escalates to the LLM on ANY rules UNKNOWN, with
no way to tell "unrecognized wording, real answer exists" from "genuinely no
answer") but n=1 is not a measured *rate*. This corpus exists to measure it.

THIS IS NOT "corpus E". Corpus E (see docs/known-limitations.md and
scripts/README.md's Corpus discipline) is reserved for the next
RulesClassifier held-out generalization test — same axis as A/B/C/D, testing
whether a *structural* rules.py fix (vs. more literal pattern tuning)
generalizes. This corpus tests a different, orthogonal question about
LLMClassifier/HybridClassifier's *precision* on ambiguous inputs, not
RulesClassifier's recall on unseen SDK wording. Scoring this corpus does not
touch rules.py and has no bearing on corpus E's discipline.

Two groups, by design:

  UNKNOWN_LABELED (majority) — genuinely out-of-taxonomy failures: none of
  the 9 FailureType members describe them (permission/IAM denial, content
  moderation blocks, geographic/export restrictions, account suspension,
  billing lapse, compliance holds, ...). RulesClassifier should say UNKNOWN
  for all of these, correctly. The question is what HybridClassifier does
  next. Sourced from real, cited SDK/API error formats where possible.

  TRICKY_BUT_CLASSIFIABLE (minority) — real concrete FailureTypes, phrased
  conversationally/obliquely rather than in typical SDK-exception shape, to
  check the measurement isn't one-sided: an LLM/Hybrid combo that gets
  cautious about ambiguity shouldn't start missing genuinely answerable
  cases either. Provenance is "constructed" (deliberately written to stress
  this boundary), not transcribed from a real source — same convention
  corpus D used for its "novel phrasings" entries.

Provenance:
  _entry(label, exc_type, error, source) — "transcribed:<source>" for real,
  cited error strings; "constructed:<description>" for deliberately written
  oblique phrasings (the TRICKY_BUT_CLASSIFIABLE group, entirely).

Not frozen the way corpus D is. This corpus can be extended over time — the
question it answers (a rate, not a single held-out score) benefits from more
data, and extending it doesn't create the same "tuning against the answer
key" risk that touching rules.py would, since nothing here feeds rules.py.

Run:
    PYTHONPATH=. .venv/bin/python scripts/gen_error_corpus_ambiguous.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _entry(label: str, exc_type: str, error: str, source: str, group: str) -> dict:
    return {
        "exception_type": exc_type,
        "error": error,
        "label": label,
        "provenance": (
            f"transcribed:{source}" if group == "unknown_labeled" else f"constructed:{source}"
        ),
        "group": group,
    }


def build_corpus() -> list[dict]:
    cases: list[dict] = []

    # ── UNKNOWN_LABELED: genuinely out-of-taxonomy failures ────────────────

    # AWS Bedrock — IAM permission denial.
    # Source: https://repost.aws/knowledge-center/bedrock-access-denied-exception
    cases.append(
        _entry(
            "unknown",
            "AccessDeniedException",
            "User: arn:aws:iam::123456789012:user/agent-runner is not authorized "
            "to perform: bedrock:InvokeModel on resource: "
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3",
            "aws-bedrock-repost-knowledge-center",
            "unknown_labeled",
        )
    )
    # OpenAI — content policy violation.
    # Source: https://portkey.ai/error-library/content-violation-error-10069
    cases.append(
        _entry(
            "unknown",
            "BadRequestError",
            "Invalid prompt: your prompt was flagged as potentially violating "
            "our usage policy. Please try again with a different prompt.",
            "openai-content-policy-error-library",
            "unknown_labeled",
        )
    )
    # Anthropic — geographic/export restriction.
    # Source: https://anthropic.com/supported-countries (referenced error text
    # documented in community reports, e.g. anthropics/claude-code#2656)
    cases.append(
        _entry(
            "unknown",
            "PermissionError",
            "Access to Anthropic models is not allowed from unsupported "
            "countries, regions, or territories. See "
            "https://www.anthropic.com/supported-countries for details.",
            "anthropic-supported-countries",
            "unknown_labeled",
        )
    )
    # Azure OpenAI — Responsible AI content filter.
    # Source: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
    cases.append(
        _entry(
            "unknown",
            "BadRequestError",
            "The response was filtered due to the prompt triggering Azure "
            "OpenAI's content management policy. Please modify your prompt "
            "and retry. To learn more: ResponsibleAIPolicyViolation.",
            "azure-openai-content-filter-docs",
            "unknown_labeled",
        )
    )
    # Account suspension / ToS violation — common SaaS pattern, not tied to
    # one SDK's exact wording.
    cases.append(
        _entry(
            "unknown",
            "ForbiddenError",
            "Your account has been suspended for violating our Terms of "
            "Service. Contact support to appeal this decision.",
            "generic-saas-account-suspension-pattern",
            "unknown_labeled",
        )
    )
    # Billing lapse — distinct from corpus D's insufficient-credits (a
    # per-request balance check); this is account-level deactivation.
    cases.append(
        _entry(
            "unknown",
            "SubscriptionExpiredError",
            "Your subscription expired on 2026-08-15. Renew your plan to "
            "continue making API requests.",
            "generic-saas-billing-lapse-pattern",
            "unknown_labeled",
        )
    )
    # Google Cloud IAM — same failure class as the Bedrock entry above but a
    # distinct vendor and phrasing, for diversity within the group.
    cases.append(
        _entry(
            "unknown",
            "PermissionDenied",
            "403 The caller does not have permission to access resource "
            "projects/my-project/locations/us-central1/publishers/google/"
            "models/gemini-pro:predict",
            "google-cloud-iam-error-shape",
            "unknown_labeled",
        )
    )
    # GDPR / data-residency compliance block.
    cases.append(
        _entry(
            "unknown",
            "ComplianceError",
            "Request blocked: EU data residency requirements are not met "
            "for this operation. Route this request through an EU region "
            "endpoint.",
            "generic-data-residency-compliance-pattern",
            "unknown_labeled",
        )
    )
    # Export control / sanctions restriction — OFAC-style, distinct from the
    # general region-restriction entry above (legal/sanctions basis, not
    # product availability).
    cases.append(
        _entry(
            "unknown",
            "ExportComplianceError",
            "This request cannot be fulfilled: the destination account is "
            "associated with a region subject to U.S. export control "
            "restrictions (OFAC).",
            "anthropic-ofac-restriction-pattern",
            "unknown_labeled",
        )
    )
    # Permanent deprecation — not transient (unlike a maintenance window,
    # which would arguably fit EXTERNAL_FAULT's retry intent), and not a
    # wrong-tool-reference error since the agent's original call was valid
    # when written; the service itself was withdrawn out-of-band.
    cases.append(
        _entry(
            "unknown",
            "DeprecatedEndpointError",
            "This API version has been permanently discontinued as of "
            "2026-03-01 and will not be reinstated. See the migration guide "
            "for v2.",
            "generic-permanent-deprecation-pattern",
            "unknown_labeled",
        )
    )
    # Vague internal failure — no discriminating detail of any kind.
    cases.append(
        _entry(
            "unknown",
            "InternalError",
            "Internal error: policy engine returned no verdict for this request.",
            "generic-vague-internal-error-pattern",
            "unknown_labeled",
        )
    )
    # Trial-ended billing block — distinct phrasing from both the
    # subscription-expired and insufficient-credits entries.
    cases.append(
        _entry(
            "unknown",
            "PaymentRequiredError",
            "Your free trial has ended. Add a payment method to continue using this API.",
            "generic-trial-ended-billing-pattern",
            "unknown_labeled",
        )
    )

    # ── TRICKY_BUT_CLASSIFIABLE: real types, oblique/conversational phrasing ─
    # Deliberately constructed, not transcribed — see module docstring.

    cases.append(
        _entry(
            "wrong_tool_called",
            "RuntimeError",
            "The function you tried to call doesn't correspond to any of "
            "the tools I currently have available.",
            "oblique-wrong-tool-phrasing",
            "tricky_but_classifiable",
        )
    )
    cases.append(
        _entry(
            "schema_mismatch",
            "RuntimeError",
            "I couldn't make sense of what came back from that call — it "
            "wasn't in the shape I was expecting at all.",
            "oblique-schema-mismatch-phrasing",
            "tricky_but_classifiable",
        )
    )
    cases.append(
        _entry(
            "external_fault",
            "RuntimeError",
            "Something went wrong on our end while handling that request. "
            "Please try again in a little while.",
            "oblique-external-fault-phrasing",
            "tricky_but_classifiable",
        )
    )
    cases.append(
        _entry(
            "timeout",
            "RuntimeError",
            "That operation was taking far too long to finish, so we gave "
            "up waiting and aborted it.",
            "oblique-timeout-phrasing",
            "tricky_but_classifiable",
        )
    )

    return cases


if __name__ == "__main__":
    corpus = build_corpus()
    out = Path("tests/data/error_corpus_ambiguous.json")
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} entries to {out}")
    for group in ("unknown_labeled", "tricky_but_classifiable"):
        n = sum(1 for c in corpus if c["group"] == group)
        print(f"  {group}: {n}")
