# Policy Chain Example

Demo of `FailurePolicy.chain()` — falling through to a fallback strategy once the primary one exhausts itself, within a single failure type.

**Source:** [`examples/policy_chain.py`](https://github.com/mattekudacy/triage/blob/main/examples/policy_chain.py)

## Requirements

None — fully synthetic, no API key needed.

## What it demonstrates

Two scenarios, run side by side:

**Scenario A — `LOOP_DETECTED`: replan, then rollback.**
`FailurePolicy.chain(replan(max_replans=1), rollback_to_checkpoint())` tries `replan` first. The agent keeps looping despite the hint, `replan` exhausts its one allowed attempt, and the chain falls through to `rollback_to_checkpoint()` — restoring the state `update_state()` saved before the loop started (`auto_checkpoint=True` saves after every `record_step()` call).

**Scenario B — `EXTERNAL_FAULT`: retry, then replan.**
`FailurePolicy.chain(backoff_and_retry(max_attempts=2), replan(...))` retries the primary API twice, then falls through to a replan hint pointing the agent at a backup data source.

`chain()` composes strategies for the *same* failure type in one attempt — contrast with `FailurePolicy.sequence()` ([Policy Sequence example](policy-sequence.md)), which advances one step *per failure* across attempts instead. See [`docs/api/policy.md`](../api/policy.md) for both methods' full signatures.

## Run

```bash
python examples/policy_chain.py
```
