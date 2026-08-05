# scripts/

Measurement harnesses. None of these are part of the shipped package — they are not
included in the wheel or sdist, and nothing under `triage/` imports them.

All are run from the repo root with `PYTHONPATH=.`.

## Benchmarks

| Script | What it measures |
|---|---|
| `bench_synthetic.py` | Routing demo — triage vs. a no-recovery baseline across three failure modes |
| `classifier_accuracy.py` | Four-block precision/recall report for `RulesClassifier` |

```bash
PYTHONPATH=. python scripts/bench_synthetic.py
PYTHONPATH=. python scripts/classifier_accuracy.py
```

`bench_synthetic.py` is a *mechanism* demo, not an accuracy measurement: the tasks are
constructed so the correct recovery hint changes the outcome. It shows that routing works,
not how often classification is right.

## Error corpora

`gen_error_corpus{,_b,_c}.py` regenerate `tests/data/error_corpus_{a,b,c}.json` by provoking
real exceptions from installed SDKs and transcribing published error strings. The JSON files
are checked in, so scoring is reproducible without re-running the generators or installing
every SDK. Re-run a generator only when adding cases.

### Corpus discipline

This is the part that's easy to destroy by accident:

| Corpus | Sources | Status | Score |
|---|---|---|---|
| A | stdlib json/asyncio, httpx, pydantic, openai, anthropic | **Training** — guided v0.25 fixes | 100% (30/30) |
| B | boto3/botocore, google-genai/grpc, aiohttp, requests/urllib3 | **Training** — guided v0.26 fixes | 90% (18/20) |
| C | azure-core, Mistral, Cohere, Groq, LiteLLM, Vertex AI, LlamaIndex | **Held out — frozen** | 52% recall, 100% precision |

A corpus becomes training data the moment its misses inform a `rules.py` edit. A and B
already have; their scores prove the patterns fit the data they were written against and say
nothing about generalization.

**Corpus C is frozen.** Do not tune `rules.py` against C's misses. If you do, the only
held-out measurement in the repo is gone and there is no way to get it back — you cannot
un-see the data. To improve the classifier: generate corpus D from fresh, disjoint sources
without consulting `rules.py`, tune against C's misses, then score D once. C then joins the
training set and D becomes the new frozen benchmark.

When quoting accuracy anywhere — README, docs, issues — quote the held-out number and label
the training ones as training.
