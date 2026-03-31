# Streaming Contract (Mechanism-Auditing Mode)

This document defines the exact streaming semantics used by the single-GPU mechanism-auditing path.

## Terms

- `sequence`: one tokenized training example of length `T` used for next-token LM loss.
- `segment`: one externally provided slice of a longer document (used in eval/inference workflows).
- `chunk`: one training-time online slice used inside `train.online_updates=true`.
- `batch`: the `B` sequences processed together by the dataloader.
- `fast-state context`: per-context mutable memory state (CMS/TITAN/self-mod) used for online updates.

## State Scope and Lifetime

- Base model parameters (meta params): persistent across all steps.
- Fast-state: initialized per batch in training when `train.use_fast_state=true`.
- With `data.batch_size>1`, fast-state is currently shared across examples in the same batch.
- In mechanism-auditing presets we set `data.batch_size=1` to preserve per-context semantics.

## CMS Buffer Lifecycle

For each CMS level `level_name`:

1. `initialize`: create empty buffer with `inputs`, `teach`, `active`, `count=0`.
2. `accumulate`: append current tokens and increment `count`.
3. `boundary update`: while `count >= update_period`, pop exactly `update_period` tokens and apply one update.
4. `finalize`:
   - if `cms_flush_partial_at_end=true`, flush remaining partial tokens once.
   - clear buffer contents and reset count to zero.
5. `reset`: equivalent to finalize + clear, used at sequence end.

## `finalize_updates` Contract

- `finalize_updates=false`:
  - accumulate/update only full `update_period` boundaries.
  - do not partial-flush.
  - keep pending tokens for the next chunk call.
- `finalize_updates=true`:
  - apply normal boundary updates.
  - optional partial flush (`cms_flush_partial_at_end=true`).
  - clear per-level CMS buffers after finalize.

Training uses `finalize_updates=true` only on the last chunk of the sequence.

## Chunk-Boundary Objective Semantics

Two training modes are supported:

1. **Overlap mode (default)**: one-token overlap between neighboring chunks.
2. **Boundary-target mode**: no overlap; each chunk receives explicit `next_tokens` boundary targets.

Example for tokens `[t0 t1 t2 t3 t4]` and `chunk_size=2`:

- Overlap mode:
  - chunk 1: `[t0 t1]` contributes pair `t0->t1`
  - chunk 2: `[t1 t2 t3]` contributes pairs `t1->t2`, `t2->t3`
  - chunk 3: `[t3 t4]` contributes pair `t3->t4`
- Boundary-target mode:
  - chunk 1: `[t0 t1]` + boundary target `t2`
  - chunk 2: `[t2 t3]` + boundary target `t4`
  - chunk 3: `[t4]` (no boundary target)

Total supervised pairs remain `T-1`.

Boundary-target mode is enabled with:
- `train.online_boundary_targets=true`
- `train.online_carry_attention_cache=true` is the canonical paper-faithful setting for
  transformer-backed chunked runs in this repo.

## Segment Semantics for Long Documents

- A segment is external input partitioning, not the same as training chunking.
- Optional attention-state carry is available via model attention cache APIs:
  - `model.init_attention_cache()`
  - `model(..., attention_cache=..., return_attention_cache=True)`
- Training can carry attention state across chunk calls when:
  - `train.online_boundary_targets=true`
  - `train.online_carry_attention_cache=true`
- Fast-memory updates can persist across steps when the caller reuses fast-state.

## Strict Mode

Set `train.strict_streaming_contract=true` to fail fast on known semantics violations:

- distributed training with unsupported paper-auditing features,
- fast-state with `data.batch_size>1`,
- `train.online_updates=true` with `train.use_fast_state=false`,
- non paper-defined variant under strict paper-auditing expectations,
- invalid boundary/carry combinations for online chunking.

## Cadence Verification Example

After a run that emits JSON metrics, validate a CMS level cadence:

```bash
uv run python scripts/checks/verify_update_cadence.py \
  --log-path logs/mechanism_audit_smoke.json \
  --metric-prefix layer0.cms.cms_mid \
  --total-tokens 8 \
  --update-period 4 \
  --output reports/cadence_mechanism_audit_smoke.json
```

Expected report keys:
- `ok`
- `metric_prefix`
- `expected`
- `observed`
- `checks`
