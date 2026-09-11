# scripts/

Measurement harnesses. None of these are part of the shipped package — they are not
included in the wheel or sdist, and nothing under `triage/` imports them.

All are run from the repo root with `PYTHONPATH=.`.

## Benchmarks

| Script | What it measures |
|---|---|
| `bench_synthetic.py` | Routing demo — triage vs. a no-recovery baseline across three failure modes |
| `classifier_accuracy.py` | Nine-block precision/recall report for `RulesClassifier` |

```bash
PYTHONPATH=. python scripts/bench_synthetic.py
PYTHONPATH=. python scripts/classifier_accuracy.py
```

`bench_synthetic.py` is a *mechanism* demo, not an accuracy measurement: the tasks are
constructed so the correct recovery hint changes the outcome. It shows that routing works,
not how often classification is right.

## Error corpora

`gen_error_corpus{,_b,_c,_d}.py` regenerate `tests/data/error_corpus_{a,b,c,d}.json` by
provoking real exceptions from installed SDKs and transcribing published error strings. The
JSON files are checked in, so scoring is reproducible without re-running the generators or
installing every SDK. Re-run a generator only when adding cases.

### Corpus discipline

This is the part that's easy to destroy by accident:

| Corpus | Sources | Status | Score |
|---|---|---|---|
| A | stdlib json/asyncio, httpx, pydantic, openai, anthropic | **Training** — guided v0.25 fixes | 100% (30/30) |
| B | boto3/botocore, google-genai/grpc, aiohttp, requests/urllib3 | **Training** — guided v0.26 + v1.1 fixes | 100% (20/20) |
| C | azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI, LlamaIndex | **Training** — guided v1.1 fixes | 100% (27/27) |
| D | huggingface_hub, Ollama, OpenRouter, MCP, CrewAI, Semantic Kernel | **Held out — frozen** | 40% recall, 100% precision |

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

**Corpus D is frozen.** Do not tune `rules.py` against D's misses. If you do, the only
held-out measurement in the repo is gone and there is no way to get it back — you cannot
un-see the data. Before spending a corpus E on another pattern-tuning cycle, read the
"what this means for where effort goes next" note in `docs/known-limitations.md` — corpus D's
result suggests literal-pattern tuning has a ceiling that another round of it is unlikely to
clear, and a structural change may be the better use of the next cycle. If tuning does
proceed: generate corpus E from fresh, disjoint sources without consulting `rules.py`, tune
against D's misses, then score E once. D then joins the training set and E becomes the new
frozen benchmark.

When quoting accuracy anywhere — README, docs, issues — quote the held-out number and label
the training ones as training.
