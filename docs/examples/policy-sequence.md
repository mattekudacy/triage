# Policy Sequence Example

Demo of `FailurePolicy.sequence()` — stepping through an ordered list of strategies across *successive* failures of the same type, escalating once the list is exhausted.

**Source:** [`examples/policy_sequence.py`](https://github.com/mattekudacy/triage/blob/main/examples/policy_sequence.py)

## Requirements

None — fully synthetic, no API key needed.

## What it demonstrates

Two scenarios, run side by side:

**Scenario A — `EXTERNAL_FAULT`: retry → replan → escalate.**
Attempt 1 hits a 503 and `sequence()` picks strategy 0 (`backoff_and_retry`). Attempt 2 is still failing, so `sequence()` advances to strategy 1 (`replan`). Attempt 3 reads the hint, switches to a backup API, and succeeds.

**Scenario B — `LOOP_DETECTED`: replan → rollback → escalate.**
Attempt 1 loops, `sequence()` picks strategy 0 (`replan` with a hint). Attempt 2 loops again despite the hint, `sequence()` advances to strategy 1 (`rollback_to_checkpoint`). Attempt 3 resumes from the restored checkpoint state and succeeds.

`sequence()` advances one step *per failure*, across attempts — contrast with `FailurePolicy.chain()` ([Policy Chain example](policy-chain.md)), which falls through to a fallback strategy within a *single* attempt once the primary is exhausted. See [`docs/api/policy.md`](../api/policy.md) for both methods' full signatures.

## Run

```bash
python examples/policy_sequence.py
```
