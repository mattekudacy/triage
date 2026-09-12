# scripts/

Measurement harnesses. None of these are part of the shipped package — they are not
included in the wheel or sdist, and nothing under `triage/` imports them.

All are run from the repo root with `PYTHONPATH=.`.

## Benchmarks

| Script | What it measures |
|---|---|
| `bench_synthetic.py` | Routing demo — triage vs. a no-recovery baseline across three failure modes |
| `classifier_accuracy.py` | Ten-block precision/recall report for `RulesClassifier` — zero API calls |
| `llm_classifier_accuracy.py` | Scores corpus D with `LLMClassifier`/`HybridClassifier` alongside `RulesClassifier` — **requires an LLM backend**, makes real calls |
| `hybrid_ambiguity_accuracy.py` | Measures `HybridClassifier`'s override rate on genuinely-ambiguous inputs — **requires an LLM backend**, makes real calls |
| `mast_mode_pilot_accuracy.py` | Pilot: can an LLM tell apart 3 of the 12 semantic-only MAST modes (2.5, 2.4, 1.4)? Separate experimental prompt, does not touch `LLMClassifier` — **requires an LLM backend**, makes real calls |

All three "requires an LLM backend" scripts are **open by default, no key needed**: with
nothing configured they talk to a local Ollama server (`http://localhost:11434/v1`,
`llama3.2`) via each script's own `_resolve_backend()` — set an Anthropic credential (or
`TRIAGE_LLM_BASE_URL` for any other provider) to use something else instead. See
`docs/concepts/classifiers.md`'s "Open by default" note.

```bash
PYTHONPATH=. python scripts/bench_synthetic.py
PYTHONPATH=. python scripts/classifier_accuracy.py

# Local Ollama (default — no key)
ollama pull llama3.2   # once
PYTHONPATH=. python scripts/llm_classifier_accuracy.py
PYTHONPATH=. python scripts/hybrid_ambiguity_accuracy.py
PYTHONPATH=. python scripts/mast_mode_pilot_accuracy.py

# Or Anthropic, if you'd rather use a key
ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/llm_classifier_accuracy.py
ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/hybrid_ambiguity_accuracy.py
ANTHROPIC_API_KEY=sk-ant-... PYTHONPATH=. python scripts/mast_mode_pilot_accuracy.py
```

`llm_classifier_accuracy.py` tests a claim the docs make but had never measured: that
`LLMClassifier`/`HybridClassifier` close the routing-sensitive gap `RulesClassifier`'s v1.1
pattern-tuning pass could not (see corpus D above). It only *reads* corpus D — it never
edits `rules.py` or a corpus file, so it's safe to re-run any number of times and cannot
convert corpus D to training data the way tuning against its misses would. Runs a pre-flight
sanity check first and refuses to print a report if it fails, since `LLMClassifier.classify()`
silently returns `UNKNOWN` on any error (bad key, missing dependency, unreachable endpoint) —
without that guard, a broken credential would look identical to "the LLM doesn't help either."

**Now measured** (`gpt-oss:120b-cloud` via Ollama Cloud): routing-sensitive recall goes
1/12 (8%, `RulesClassifier`) → 10/12 (83%, `HybridClassifier`) — the recall claim holds — but
`HybridClassifier` misrouted 3/20 entries where `RulesClassifier` misroutes zero by
construction. One misroute is structural: `HybridClassifier` cannot tell "rules doesn't
recognize this wording" from "this genuinely has no answer," so it can overturn a correctly-
conservative rules `UNKNOWN` into a confident wrong LLM guess — observed on corpus D's one
`unknown`-labeled entry in every LLM-involving run. See `README.md`'s "Does
LLMClassifier/HybridClassifier actually close the gap?" and `docs/known-limitations.md` for
the full table and writeup. Results vary run to run (reasoning-model sampling) — this is a
representative measurement, not a frozen, CI-enforced benchmark like `RulesClassifier`'s
corpus D floor.

`hybrid_ambiguity_accuracy.py` follows up on a finding `llm_classifier_accuracy.py` could only
show with `n=1`: `HybridClassifier` overturned corpus D's one genuinely out-of-taxonomy entry
(correctly classified `UNKNOWN` by `RulesClassifier`) into a confident wrong guess, in every
LLM-involving run. That's a confirmed *mechanism* — `HybridClassifier.classify()` escalates to
the LLM on any rules `UNKNOWN`, with no way to tell "unrecognized wording, real answer exists"
from "genuinely no answer" — but not yet a measured *rate*. This script scores
`tests/data/error_corpus_ambiguous.json` (16 entries: 12 genuinely out-of-taxonomy sourced
from real, cited SDK/API errors — IAM/permission denial, content-policy blocks, region/export
restrictions, account suspension, billing lapse — plus 4 real `FailureType`s phrased
obliquely, to check the measurement isn't one-sided) to get one. Same read-only discipline as
`llm_classifier_accuracy.py`: never touches `rules.py` or edits a corpus file. Unlike corpus
D, this corpus isn't frozen — it's designed to grow, since a rate benefits from more data and
nothing here feeds back into `rules.py` the way tuning would. `tests/test_classifier_ambiguity.py`
guards its precondition with zero API calls: every entry must be a `RulesClassifier` `UNKNOWN`,
or the escalation-to-LLM premise silently breaks.

`bench_synthetic.py` is a *mechanism* demo, not an accuracy measurement: the tasks are
constructed so the correct recovery hint changes the outcome. It shows that routing works,
not how often classification is right.

`mast_mode_pilot_accuracy.py` is a different kind of measurement from the four above: it scores
`tests/data/mast_pilot_corpus.json` (6 entries, 2 each real-cited for 2.5 Ignored Other Agent's
Input and 1.4 Loss of Conversation History, 1 for 2.4 Information Withholding plus 1 genuinely
ambiguous between 2.4/2.5 — see `gen_mast_pilot_corpus.py`'s docstring) against an
**experimental prompt and label set that live only in this script** — not `LLMClassifier`,
which still only ever returns one of the 9 stable `FailureType`s. This is Step 1 of
`docs/concepts/multi-agent-failures.md`'s "Phase 3 scoping": measure whether an LLM can tell
MAST modes apart at all before considering any change to the stable classifier contract.
Recall-only (no negative examples exist yet, so no false-positive rate) and not yet run against
a real model in this environment — same API-key-blocked status as `hybrid_ambiguity_accuracy.py`.
`tests/test_mast_pilot_corpus.py` guards the corpus's own shape with zero API calls.

## Error corpora

`gen_error_corpus{,_b,_c,_d,_e}.py` regenerate `tests/data/error_corpus_{a,b,c,d,e}.json` by
provoking real exceptions from installed SDKs and transcribing published error strings. The
JSON files are checked in, so scoring is reproducible without re-running the generators or
installing every SDK. Re-run a generator only when adding cases. `_e.py` additionally captures
a `"metadata"` field per entry (a real `http_status` or `json_rpc_code`, not invented) — see
`triage/taxonomy.py`'s `Step` docstring for the convention this tests.

### Corpus discipline

This is the part that's easy to destroy by accident:

| Corpus | Sources | Status | Score |
|---|---|---|---|
| A | stdlib json/asyncio, httpx, pydantic, openai, anthropic | **Training** — guided v0.25 fixes | 100% (30/30) |
| B | boto3/botocore, google-genai/grpc, aiohttp, requests/urllib3 | **Training** — guided v0.26 + v1.1 fixes | 100% (20/20) |
| C | azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI, LlamaIndex | **Training** — guided v1.1 fixes | 100% (27/27) |
| D | huggingface_hub, Ollama, OpenRouter, MCP, CrewAI, Semantic Kernel | **Held out — frozen** | 40% recall, 100% precision |
| E | MCP (json_rpc_code), Together AI, Fireworks AI, Replicate, Cerebras, Perplexity, DeepSeek, NVIDIA NIM, xAI | **Held out — frozen** | 69% recall, 100% precision |

Corpus C was held out through v1.0 (52% recall, 100% precision) and became training data in
v1.1 the same way A and B did before it: `rules.py` was tuned directly against its 13 misses.
Corpus D replaced it and carries the held-out claim now.

Corpus D's 40% is an average over two groups that must not be collapsed: 86% (6/7) on the
self-healing types (`external_fault`, `timeout`) where any retry recovers regardless of
classification, and **8% (1/12) on the routing-sensitive types** (`wrong_tool_called`,
`schema_mismatch`) where the matched hint is the only thing that makes recovery work — the
*same* 8% corpus C measured before the v1.1 tuning pass. Blocks 7-8 of `classifier_accuracy.py`
print this split, plus a comparison to corpus C's pre-tuning number. Quote both whenever you
quote the aggregate — the fact that they match is the finding: the v1.1 pattern pass improved
corpus C from 8% to 100% on the routing-sensitive types and improved corpus D not at all,
because the new patterns were string literals keyed to corpus C's specific wording. See
`docs/known-limitations.md` for the full writeup and what it implies for corpus E.

A corpus becomes training data the moment its misses inform a `rules.py` edit. A, B, and C
already have; their scores prove the patterns fit the data they were written against and say
nothing about generalization.

**Corpus E — what actually happened instead of another tuning cycle.** `docs/known-
limitations.md`'s "Corpus E scoping" section proposed a *structural* change instead of another
literal-pattern-tuning pass: `RulesClassifier` learned to check a caller-supplied structured
code in `Step.metadata` (`"http_status"`, `"json_rpc_code"`) alongside its message-text
patterns — built and unit-tested first, entirely without reading corpus D's misses, then
corpus E was generated from fresh sources (disjoint from A-D) *with* real codes captured
alongside message and exception type, to test whether the structural signal generalizes where
message-text tuning didn't.

It partly does. Routing-sensitive recall on corpus E is 44% (4/9), well above corpus D's 8% —
but nearly all of that gain is the two MCP `json_rpc_code` entries (`-32601`/`-32700`), not
HTTP status codes. Every fresh HTTP-only vendor's "wrong tool"/"bad schema" failure in corpus E
(Together AI, Fireworks AI, Replicate, Cerebras, Perplexity) used `404`/`400`/`422` — codes
deliberately excluded from the HTTP tables as too ambiguous to map safely (see
`triage/classifier/rules.py`'s module docstring). The structural signal generalizes where a
protocol *spec* guarantees a code's meaning (JSON-RPC); it buys nothing for vendors whose
"wrong tool"/"bad schema" signal is just an HTTP status shared with a dozen unrelated failure
causes. See `docs/known-limitations.md`'s "Corpus E scoping" for the full breakdown.

One adversarial case earned its keep: a real MCP server (`langgenius/dify#22675`) reused
`-32600` for a session/auth condition, not the malformed-request meaning the JSON-RPC spec
assigns it — corpus E's one `unknown`-labeled entry was built specifically to test this, caught
a genuine misroute, and `-32600` was dropped from the mapping before this floor was frozen. The
same "generic code reused for an unrelated failure" pattern that dropped `OutputParserError`
from corpus D — protocol-level codes are not immune to it either. 100% precision on corpus E as
a result: every remaining miss returns `UNKNOWN`, zero misroutes.

**Both D and E are frozen.** Do not tune `rules.py` against either corpus's remaining misses —
same rule as before, applied twice now. If you want to test a further structural idea (a
different protocol's error codes, a broader signal), generate a corpus F from sources disjoint
from A-E.

When quoting accuracy anywhere — README, docs, issues — quote the held-out number and label
the training ones as training.
