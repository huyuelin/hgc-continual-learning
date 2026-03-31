# Paper Compliance / Fidelity Guide (Nested Learning / HOPE)

This doc explains the **fidelity‑critical behaviors** (what the paper relies on) and how they map to this repo’s code, flags, and tests.

It is deliberately **mechanism‑focused**: you can use it to answer “did we implement the architecture/update rules correctly?” without requiring full‑scale training reproduction.

For exact chunk/segment/buffer semantics, see `docs/STREAMING_CONTRACT.md`.

## Paper Reference Pin

All compliance/equation references in this repo are pinned to:

- Source: `google_papers/Nested_Learning_Full_Paper/Nested_Learning_Full_Paper.md`
- SHA-256: `7524af0724ac8e3bad9163bf0e79c85b490a26bc30b92d96b0bdf17a27f9febc`

## Scope

**In scope**
- HOPE blocks (attention + CMS + TITAN/self‑mod paths) and the *nested/online* update mechanism.
- Correct teach‑signal alignment (LM head vs embedding), per‑layer local error signals (δℓ), and chunk‑accumulated CMS updates.
- A paper‑style optimizer option (M3) alongside practical defaults.

**Out of scope (today)**
- Full bi‑level meta‑learning experiments over explicit task episodes (outer objective over tasks + inner adaptation per task).
- Results parity at the original paper’s compute scale.

## Semantic contract (important)

This repo focuses on **mechanism-level fidelity** (update rules + dataflow) with explicit tests.

- **Differentiable reads:** the forward pass used to compute the outer LM loss is standard autograd.
- **Stop‑grad writes:** online memory updates are applied in an explicit update pass (typically under `torch.no_grad()`), so we do **not** backprop through online writes.
- **Algorithm mode:** `train.algorithm_mode=two_pass_stopgrad_updates` is the stable default.  
  `train.algorithm_mode=boundary_state_grad_through_write` is available as an **experimental single-process mechanism path** with strict runtime constraints (`online_updates=true`, `per_layer_teach_signal=true`, `use_fast_state=true`, non-DDP). It is not yet treated as full paper-training reproduction.
- **Boundary-target mode:** we support explicit boundary-token supervision (`train.online_boundary_targets=true`) and optional attention-state carry across chunks (`train.online_carry_attention_cache=true`) for stronger streaming equivalence, while keeping stop-grad write semantics.
- **Fast-state guardrail:** `train.online_updates=true` with `train.use_fast_state=false` now emits a warning in non-strict mode and raises in strict/paper-faithful mode.
- **Meta initialization (fast-state mode):** when `train.use_fast_state=true`, meta parameters are not mutated by online updates, but the *read-path* meta parameters still receive outer gradients:
  - CMS/TITAN fast state uses **meta+delta** (forward uses `meta + delta`; updates write deltas only).
  - HOPE‑SelfMod uses a detached per‑context state, but the read path uses a **straight‑through meta gradient** link so the meta initialization remains trainable.

## Quick start: mechanism-auditing presets (single GPU)

The highest-fidelity execution path in this repo is **single‑GPU** `train.py`, because it supports both:
1) **per‑layer δℓ** teach signals and  
2) **online chunked training** where later tokens’ loss/gradients can see earlier memory updates.

Minimal smoke:

```bash
uv run python train.py --config-name pilot_paper_faithful train.steps=5
```

Note: these presets set `data.batch_size=1` to avoid cross-sample memory sharing
when `train.use_fast_state=true`.

Optional: select the paper optimizer variant for the *outer* step:

```bash
uv run python train.py --config-name pilot_paper_faithful train.steps=5 optim.type=m3
```

Mechanism-auditing HOPE self-mod variant:

```bash
uv run python train.py --config-name pilot_selfmod_paper_faithful train.steps=5
```

Boundary-state experimental smoke (single process only):

```bash
uv run python train.py --config-name pilot_paper_faithful \
  train.algorithm_mode=boundary_state_grad_through_write \
  train.steps=5
```

Boundary-state mode tradeoffs:
- Keeps cross-chunk write paths differentiable, which increases activation retention and memory pressure.
- Usually runs slower than `two_pass_stopgrad_updates` due to larger autograd graphs.
- Intended for mechanism probing and diagnostics, not for long production runs in this repo yet.

## Mechanism-Auditing vs Practical Mode (Matrix)

This repo supports both mechanism-auditing presets (for correctness checks) and practical defaults (for running pilots quickly).

| Mechanism | Paper intent | This repo (single GPU) | Notes / Tests |
|---|---|---|---|
| Teach‑signal alignment | δ uses LM head weights | `compute_teach_signal()` matches autograd grad | `tests/test_teach_signal.py` |
| Per‑layer δℓ | block‑local error signals | `train.per_layer_teach_signal=true` | `tests/test_teach_signal.py` |
| Online chunked training | later tokens can “see” earlier inner updates | `train.online_updates=true` with either overlap mode or explicit boundary-target mode + end-of-sequence finalize | `src/nested_learning/training.py`, `tests/test_online_chunking.py` |
| CMS chunk accumulation | sum over token deltas per chunk | `cms_chunk_reduction="sum"` default | `tests/test_cms.py`, `tests/test_cms_delta_rule.py` |
| CMS partial-chunk flush | update on final partial chunk | `model.cms_flush_partial_at_end=true` | `tests/test_cms_flush_partial.py` |
| CMS cadence across chunked calls | `update_period` accumulation must survive multiple update-pass calls | fast-state CMS buffers persist until `finalize_updates=true` | `tests/test_cms_cross_call.py` |
| CMS LayerNorm | paper is architecture-light; norm is optional | `model.cms_use_layernorm=true` (default) | `tests/test_cms.py` |
| HOPE‑SelfMod local conv | local conv window=4 (paper HOPE module) | `SelfModifyingTitansConfig.local_conv_window=4` default (causal depthwise) | `tests/test_selfmod_local_conv.py` |
| HOPE‑SelfMod fixed q | paper: `q_t = x_t W_q` non‑adaptive | `SelfModifyingTitansConfig.adaptive_q=false` default | `tests/test_selfmod_adaptive_q.py` |
| HOPE‑SelfMod Eq. (91) skip | no projection skip term (`w_skip`) | `model.self_mod_use_skip=false` (mechanism-auditing presets) | `tests/test_residual_mlp_memory.py` |
| HOPE‑SelfMod read/write separation | differentiable read; stopgrad through writes | forward uses differentiable read; updates occur only in explicit update pass | `tests/test_selfmod_grad_flow.py`, `tests/test_hope_selfmod_update_pass.py` |
| Fast‑state isolation | per‑context inner updates without mutating meta params, while read‑path meta init remains learnable | `train.use_fast_state=true` | CMS/TITAN use **meta+delta**; HOPE‑SelfMod read path uses straight‑through meta gradients. Meta params remain unchanged during updates and still receive outer grads (`tests/test_hope_selfmod_fast_state_meta_unchanged.py`, `tests/test_fast_state_meta_grads.py`, `tests/test_fast_state_selfmod_meta_grads.py`, `tests/test_fast_state_forward_equivalence.py`, `tests/test_fast_state_batch_semantics.py`) |
| Surprise metric | paper “surprise” trigger | `model.surprise_metric=l2` (default); also `loss`, `logit_entropy` | `tests/test_surprise_metric.py`, `tests/test_faithfulness_harness.py` |
| Outer optimizer | M3 option exists | `optim.type=m3` | `tests/test_m3.py` |
| Outer param policy | include memory initial states in meta-update | `optim.param_policy=all` | `tests/test_optimizer_param_policy.py` |
| DDP fail-fast | avoid silent paper-divergent fallbacks | `train.fail_if_paper_faithful_disabled=true` | `tests/test_distributed_fail_fast.py` |
| Multi‑GPU | (not required by paper) | DDP disables `online_updates` + `per_layer_teach_signal`; FSDP uses offline updates | documented below |

Surprise-gating note: for `model.surprise_metric=l2`, the current implementation applies a
chunk-level gate from mean teach-signal norm, then applies token-level masking inside TITAN/CMS
updates. This behavior is intentionally tested (`tests/test_surprise_metric.py`).

## Claims Boundary (What We Claim vs What We Do Not)

| Claim category | Status | Notes |
|---|---|---|
| CMS/TITAN/self-mod mechanism wiring | Implemented | Unit tests cover teach-signal, chunking, cadence primitives, and update-path invariants. |
| Mechanism-auditing single-GPU path | Implemented | Uses per-layer teach signals + explicit stop-grad update pass. |
| Full paper boundary-state gradient training through online writes | Partially implemented (experimental) | `train.algorithm_mode=boundary_state_grad_through_write` enables a constrained single-process differentiable write path; still not treated as production/full-scale reproduction. |
| Cross-chunk attention-state continuity (KV cache) | Partially implemented | Optional cache-carry path is available in model APIs and training boundary-target mode; distributed faithful path remains deferred. |
| Full paper-scale result reproduction | Not implemented | Compute/data scale parity is intentionally deferred. |

## Implementation Fidelity vs Result Fidelity

- **Implementation fidelity (this repo target):** architecture/update-path correctness, teach-signal alignment, cadence, chunking semantics, and guardrails.
- **Result fidelity (deferred):** matching full-paper training scale, data budget, and final benchmark curves.
- This repo treats implementation fidelity as complete only when mechanism checks/tests pass; result parity is explicitly a separate track.

## Scale Statement (Current vs Paper)

- Current mechanism-auditing and pilot runs are intentionally below the full paper scale.
- This repo does **not** claim paper-scale result reproduction at current compute/data settings.
- Maintainer stance: prioritize faithful implementation and auditable behavior first; scale-up remains optional contributor work.

## Paper-Faithful Configs (Usage + Caveats)

| Config | Purpose | Default Algorithm Mode | Caveats |
|---|---|---|---|
| `configs/pilot_paper_faithful.yaml` | HOPE-attention mechanism-auditing baseline | `two_pass_stopgrad_updates` | Single-process intended; sets `data.batch_size=1`, `strict_streaming_contract=true`, boundary-target + cache-carry enabled |
| `configs/pilot_selfmod_paper_faithful.yaml` | HOPE self-mod mechanism-auditing baseline | `two_pass_stopgrad_updates` | Same constraints as above; self-mod paper knobs forced (`self_mod_use_skip=false`, fixed `q`, local conv) |

Boundary-state experimental override:
- `train.algorithm_mode=boundary_state_grad_through_write`
- Requires: `online_updates=true`, `per_layer_teach_signal=true`, `use_fast_state=true`, single-process (non-DDP).

## Equation / Mechanism Code Pointers (file:line)

| Paper mechanism | Code pointer |
|---|---|
| Teach-signal proxy `dL/dh` via LM head weights | `src/nested_learning/training.py:225` |
| Per-layer teach signals (`δℓ`) from block outputs | `src/nested_learning/training.py:295` |
| Online chunk iterators (overlap / boundary-target) | `src/nested_learning/training.py:352`, `src/nested_learning/training.py:369` |
| Algorithm-mode constraints (including boundary-state experimental mode) | `src/nested_learning/training.py:606` |
| Online cache/chunk constraint guards | `src/nested_learning/training.py:650` |
| Online chunked train loop + update pass wiring | `src/nested_learning/training.py:685` |
| Run-feature telemetry (algorithm + online flags) | `src/nested_learning/training.py:1418` |
| Checkpoint metadata with algorithm/online flags | `src/nested_learning/training.py:1492` |
| Tied embedding / LM head weight contract | `src/nested_learning/model.py:156` |
| Block output capture for δℓ path | `src/nested_learning/model.py:317` |
| Fast-state init + attention-cache init | `src/nested_learning/model.py:531`, `src/nested_learning/model.py:578` |
| CMS chunk accumulation + cadence telemetry | `src/nested_learning/hope/block.py:297`, `src/nested_learning/hope/block.py:341`, `src/nested_learning/hope/block.py:365` |
| CMS partial flush on final chunk | `src/nested_learning/hope/block.py:342`, `src/nested_learning/hope/block.py:941`, `src/nested_learning/hope/block.py:1493` |
| Surprise gating threshold logic | `src/nested_learning/hope/block.py:567`, `src/nested_learning/hope/block.py:1676` |
| Differentiable inner-update path toggle | `src/nested_learning/optim/manager.py:109`, `src/nested_learning/optim/manager.py:125` |
| Test-time memorization with path/threshold controls | `src/nested_learning/memorize.py:169`, `src/nested_learning/memorize.py:292`, `src/nested_learning/memorize.py:366` |

## Reproducibility Protocol (Mechanism Track)

1. Environment:
   - `uv sync --all-extras --dev`
   - PyTorch `2.9.0`
2. Determinism:
   - set `train.seed=<int>`
   - set `train.deterministic=true` for deterministic smoke runs
3. Minimal mechanism run:
   - `uv run python train.py --config-name pilot_paper_faithful train.steps=5`
4. Optional boundary-state mechanism probe:
   - `uv run python train.py --config-name pilot_paper_faithful train.algorithm_mode=boundary_state_grad_through_write train.steps=5`
5. Validation gates:
   - `uv run ruff check .`
   - `uv run mypy src`
   - `bash scripts/checks/run_fidelity_ci_subset.sh`
   - `uv run pytest -q`

## Community-Reported Remediation Map

- Data split fallback robustness: `docs/data_pipeline.md` + `scripts/data/{train_tokenizer,shard_corpus,filter_corpus}.py`
- Missing tokenizer/help ergonomics: `scripts/data/run_sample.sh`, `scripts/checks/check_data_script_help.sh`, CI workflow
- Boundary-state mode guardrails + visibility: `src/nested_learning/training.py` + `tests/test_strict_streaming_contract.py` + `tests/test_boundary_state_training_loop.py`
- Packaging metadata completeness: `src/nested_learning/training.py` + `scripts/package_pilot_release.sh` + `tests/test_package_release_script.py`

## Acceptance Checklist (Mechanism Fidelity)

- [x] Teach signal uses LM head weights with tied embedding head.
- [x] Per-layer teach signals (`δℓ`) are available and tested.
- [x] Online chunked updates support overlap + boundary-target semantics.
- [x] CMS chunk accumulation/cadence is audited with machine-readable reports.
- [x] Surprise gating behavior is tested (loss/entropy/l2 paths).
- [x] Test-time memorization path controls (`paths`, `surprise_threshold`) are implemented and tested.
- [x] Algorithm mode + online flags are emitted in run telemetry and checkpoint metadata.
- [x] Data scripts have deterministic split fallback and CI help-smoke coverage.
- [x] Security/release gates block large/binary artifact leakage.
- [ ] Full paper-scale result reproduction (explicitly out of current scope).

## Concepts → implementation mapping

### 1) Outer parameters vs inner (“fast”) procedure

In this codebase:
- **Outer update** = the standard optimizer step (`optimizer.step()`) on the model parameters after backprop.
- **Inner update** = memory/fast updates applied *outside* the gradient graph using teach signals (δ), e.g. CMS updates and self‑modifying TITAN updates.

Where:
- Outer loop: `src/nested_learning/training.py` (`run_training_loop`)
- Inner update calls: inside the training loop after backward:
  - `base_model(tokens, teach_signal=...)` or `base_model(tokens, teach_signals=[...])`
- The update logic lives in the block implementations:
  - `src/nested_learning/hope/block.py`

### 2) “Levels” and update frequencies

Levels are represented explicitly as `LevelSpec` entries with independent `update_period`s.

Where:
- Specs: `src/nested_learning/levels.py`
- Config surface (Hydra): `model.titan_level` and `model.cms_levels` in `configs/*.yaml`
- Enforcement:
  - Online CMS buffering + update‑period gating in `src/nested_learning/hope/block.py`
  - Level optimizer tick/step orchestration in `src/nested_learning/optim/manager.py`

### 3) Teach signal alignment (LM head gradient proxy)

The global teach signal is an approximation to **dL/dh**, where `h` is the hidden state **before** the LM head. This approximation must align to the LM head weights.

In this repo, `h` is explicitly the **post-LayerNorm hidden** (the exact input to `lm_head`), and tests pin this contract.

Where:
- Weight tying is explicit: `src/nested_learning/model.py` (`self.lm_head.weight = self.embed.weight`)
- Teach signal implementation: `src/nested_learning/training.py` (`compute_teach_signal`)
- Unit coverage: `tests/test_teach_signal.py`

### 4) Per‑layer local error signals (δℓ)

When enabled, we compute a teach signal **per block output** (δℓ) via autograd and route it into each block’s update path.

Where:
- Block output capture: `src/nested_learning/model.py` (`forward_with_block_outputs`)
- δℓ computation: `src/nested_learning/training.py` (`_compute_layer_teach_signals`)
- Routing to blocks: `src/nested_learning/model.py` (`teach_signals=[...]`)
- Unit coverage: `tests/test_teach_signal.py` (shape + matching expectations)

Flag:
- `train.per_layer_teach_signal=true`

### 5) Chunked online training (read‑after‑write for *loss*, not just updates)

This is the core “gradient propagation across frequencies” concern:

If you compute the LM loss on a full sequence **once**, and only apply memory updates after the backward pass, then later tokens’ loss does not reflect earlier inner updates.

To make later tokens “see” earlier inner updates during training, we support an **online chunked training** mode:
- Split the sequence into chunks.
- For each chunk:
  1) forward → loss  
  2) `loss.backward()` **accumulating** gradients across chunks (we do not zero grads per chunk)  
  3) apply inner updates in `torch.no_grad()`  
  4) proceed to the next chunk with updated memory
- At the end, we do a single outer `optimizer.step()`.
- Chunking supports **one-token overlap** mode and **explicit boundary-target** mode.
- In fast-state mode, CMS accumulation buffers persist across calls and are finalized (optional partial flush + reset) only when `finalize_updates=true` for the sequence end.

Where:
- `src/nested_learning/training.py` (search for `online_updates`)

Flags:
- `train.online_updates=true`
- `train.online_chunk_size=0` (auto‑infer a chunk size from the minimum CMS update period)

### 6) CMS update semantics (per‑token δ + sum‑over‑chunk accumulation)

CMS updates are applied using:
- **per‑token δ targets** (no chunk‑mean broadcast), and
- **sum‑over‑chunk reduction** for the CMS update loss (rather than mean), which preserves the “accumulate over C tokens” semantics.

We implement the CMS local objective via a **gradient-shaping construction**:
- `_chunk_loss()` chooses a target `t = stopgrad(prediction − δ)` so that `∂loss/∂prediction ∝ δ` under the chosen mask and reduction.
- This matches the paper’s δ-based local learning rule while letting us implement the update via standard autograd.
- Verified by `tests/test_cms_delta_rule.py`.

Where:
- Chunk loss reduction: `src/nested_learning/hope/block.py` (`_chunk_loss`, `cms_chunk_reduction="sum"`)
- Online buffering by update_period and “pop exactly C tokens”: `src/nested_learning/hope/block.py` (`_CmsBuffer`, `_pop_buffer_chunk`, `_cms_forward_online`)
- Unit coverage:
  - `tests/test_cms.py` (online updates affect later tokens; update_period gating)

Notes:
- In the Hydra configs, CMS chunk reduction / online toggles are mechanism-auditing defaults inside the HOPE block configs. They are not currently exposed as top-level YAML keys; changing them requires a small code change.
- `model.cms_flush_partial_at_end` is exposed because it affects correctness when sequence lengths are not exact multiples of update periods.

### 7) Self‑modifying TITAN path (always‑on)

Self‑modifying TITAN updates run in the update pass; they do not require the teach signal to be nonzero, but they **do** require an explicit update call (i.e., passing `teach_signal`/`teach_signals` to trigger the update pass).

Where:
- `src/nested_learning/hope/block.py` (self‑mod update path)
- Unit coverage: `tests/test_selfmod_online.py`

### 8) Outer optimizer options (including paper M3)

Default outer optimizer in configs is practical and reproducible (`optim.type=muon` hybrid with AdamW fallback for 1D params).

Paper option:
- `optim.type=m3` selects the M3 optimizer (multi‑scale momentum + Newton‑Schulz orthogonalization).

Where:
- `src/nested_learning/optim/m3.py`
- Unit coverage: `tests/test_m3.py`

### 8.1 `nl_l2_precond` mapping assumptions (best-effort)

The inner deep optimizer variant `nl_l2_precond` is implemented as a rank-1 projection-style preconditioner:

- Context vector `x_t`: repo uses the provided level context (typically mean hidden state over batch/sequence for that update event).
- Projector: update is projected orthogonal to context via `g - (g·u)u` where `u = x_t / ||x_t||`.
- This is a best-effort mechanism mapping, not a formal proof of exact paper-equation equivalence under all normalizations/objective variants.

Code + tests:
- `src/nested_learning/optim/deep.py` (`_nl_precondition`)
- `tests/test_optim.py`

## Distributed training caveats (important)

Mechanism-auditing mode is currently focused on `train.py` (single‑GPU).

- **DDP (`train_dist.py`)**: calls the shared training loop, but explicitly disables:
  - per‑layer teach signals (`train.per_layer_teach_signal`)
  - online chunked training (`train.online_updates`)
  because these require capturing block outputs and applying sequential inner updates in a way that is not yet DDP‑safe in this repo.
  - If you want to avoid silent fallback behavior, set `train.fail_if_paper_faithful_disabled=true` to raise instead of disabling.

- **FSDP (`train_fsdp.py`)**: currently uses a simpler “offline” update pass with a global teach signal after each outer step. It does not yet implement per‑layer δℓ or online chunked training.

If you need mechanism-auditing semantics at multi-GPU scale, the next engineering task is to port the `train.py` online/per-layer flow to FSDP (or a custom DDP scheme) while keeping correctness tests.

## Verification checklist (fast)

Run the fidelity tests:

```bash
uv run python scripts/checks/verify_docs_refs.py

uv run pytest \
  tests/test_teach_signal.py \
  tests/test_cms.py \
  tests/test_cms_cross_call.py \
  tests/test_cms_flush_partial.py \
  tests/test_online_chunking.py \
  tests/test_attention_cache.py \
  tests/test_eval_state.py \
  tests/test_selfmod_online.py \
  tests/test_m3.py \
  tests/test_residual_mlp_memory.py \
  tests/test_selfmod_local_conv.py \
  tests/test_selfmod_adaptive_q.py \
  tests/test_selfmod_grad_flow.py \
  tests/test_hope_selfmod_update_pass.py \
  tests/test_cms_delta_rule.py \
  tests/test_selfmod_dgd_linear.py \
  tests/test_optimizer_param_policy.py \
  tests/test_distributed_fail_fast.py \
  tests/test_strict_streaming_contract.py \
  tests/test_verify_docs_refs.py
```

Confirm you’re running with the intended features:
- startup `run_features` should include `train.algorithm_mode=two_pass_stopgrad_updates` and `train.backprop_through_online_writes=false`.
- training logs include `teach_signal_norm` and per‑layer update telemetry (e.g. `layer0.cms.cms_fast.grad_norm`) when an update pass runs.
- streaming semantics match `docs/STREAMING_CONTRACT.md` for the selected config mode.

## Known gaps / intentionally deferred work

- Full task‑episode meta‑learning evaluation loops are not implemented.
- Multi‑GPU mechanism-auditing training (online + per-layer δℓ) is not yet implemented.
- Full distributed mechanism-auditing path with boundary-target + attention-cache carry remains deferred.
- Large‑scale results reproduction is not a requirement for claiming mechanism fidelity in this repo.
