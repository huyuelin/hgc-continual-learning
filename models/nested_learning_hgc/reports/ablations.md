# Planned Ablations – Pilot Run

This document tracks the ablation studies we intend to run once the 3 B-token pilot checkpoint is available. The goal is to isolate the contributions of teach-signal scaling, CMS chunk accumulation, self-modifiers, and optimizer choices (AdamW vs Muon) before moving to larger configs.

## 1. Teach-signal schedule
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| Baseline | Warmup 2 k → decay 120 k→140 k (current pilot config) | ✅ (step 230 k) | Metrics in `eval/zeroshot_pilot_step230000.json` + friends. |
| Low scale (0.05) | Reduce teach_scale to 0.05, 2 k-step pilot ablation | ✅ | `artifacts/checkpoints/pilot_teach05/step_002000.pt`, JSON log `logs/pilot-teach05-20251114010549.json`, evals under `eval/*_pilot_teach05_step2000.json`. |
| High scale (0.15) | Increase teach_scale to 0.15, runs at 2 k and 25 k steps | ✅ | Short run: `artifacts/checkpoints/pilot_teach15/step_002000.pt`; long run: `artifacts/checkpoints/pilot_teach15_long/step_025000.pt`; logs/evals `logs/pilot-teach15-20251114012109.json`, `logs/pilot-teach15-long-20251114185448.json`, `eval/*_pilot_teach15_long_step25000.json`. |
| No decay | Warmup only, no decay | ⏳ | Expect higher plasticity, risk of instability |
| Per-level scale | Different teach_scale per CMS level | ⏳ | Requires config changes |

## 2. CMS chunk accumulation
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| Full CMS | Chunk accumulation + telemetry (default) | ✅ smoke-tested | Verified via `tests/test_cms.py`. Baseline checkpoint: `artifacts/checkpoints/pilot/step_230000.pt`. |
| No chunking | Update each token (Transformer-like) | ✅ | Run `pilot-cms-nochunk` (5 k steps) with overrides `model.cms_levels.*.update_period=1`. Outputs: `logs/pilot-cms-nochunk-20251114124501.json`, eval JSONs `eval/*_pilot_cms_nochunk_step5000.json`. |
| Sparse chunks | Update every 512 tokens only | ✅ | Config `configs/ablations/cms_sparse.yaml` (dim 384, layers 8, seq 1024, batch 2, chunk periods 8/32/128/512). Run `pilot-cms-sparse` (5 k steps) w/ resolved config `configs/resolved/cms_sparse_eval.yaml`. Metrics: PIQA 0.516, BoolQ 0.367, continual CE ≈25 across segments (see `eval/*_pilot_cms_sparse_step5000.json`). |

To keep chunk buffers tractable we reduced the CMS-hidden multiplier to 2, halved the batch size, and exported a resolved Hydra config at `configs/resolved/cms_sparse_eval.yaml` so that evaluation scripts can load the composed settings without Hydra. The highest-frequency buffer now tops out at ~3 GB (inputs + targets) instead of the 12 GB spikes we observed during the initial 2048-token attempt.

## 3. Self-modifier toggles
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| Enabled | SelfModifier active (default) | ✅ | Baseline pilot + long-run checkpoints. |
| Disabled | Freeze self-modifier params | ✅ | Run `pilot-selfmod-off` (5 k steps). Continual CE jumped to ~45; see `eval/*_pilot_selfmod_off_step5000.json`. |
| Teach-only | Teach signal applied but self-mod not updated | ⏳ | Planned follow-up once optimizer ablation finishes. |

## 4. Optimizer swaps
| Variant | Description | Status | Notes |
|---------|-------------|--------|-------|
| AdamW fused (control) | `optim.type=adamw` with fused kernels (override the Muon default) | ✅ | Run `pilot-opt-adamw-20251115173858` (5 k steps, batch 6, seq 2048) → checkpoint `artifacts/checkpoints/pilot-opt-adamw-20251115173858/step_005000.pt`. Eval highlights: PIQA 0.559, HellaSwag 0.273, Winogrande 0.500, BoolQ 0.367 (`eval/zeroshot_pilot_opt_adamw_step5000.json`); NIAH accuracies {0.75, 1.0, 0.5, 0.75, 0.5, 0.25}; continual CE ≈50/43/39/39 across segments. |
| Muon hybrid | `optim.type=muon` for ≥2D params, AdamW for embeddings/bias | ✅ | Run `pilot-opt-muon-20251115180139` (identical setup) → checkpoint `artifacts/checkpoints/pilot-opt-muon-20251115180139/step_005000.pt`. Eval highlights: PIQA 0.531, HellaSwag 0.313, Winogrande 0.484, BoolQ 0.570 (`eval/zeroshot_pilot_opt_muon_step5000.json`); NIAH {0.5, 0.5, 0.25, 0.75, 0.75, 0.75}; continual CE ≈11 across all segments (`eval/continual_pilot_opt_muon_step5000.json`). |
| Full Muon | Force Muon everywhere | ⏳ | Pending stability run; expect to require per-layer LR tuning. |

At 5 k pilot steps the hybrid Muon optimizer trades a small PIQA drop (0.559→0.531) for markedly better BoolQ (0.37→0.57) and 4× lower continual losses. Muon also cuts training loss faster (final CE ≈6.8 vs 8.5). Based on this we plan to adopt Muon for the resumed long HOPE run while keeping AdamW checkpoints for baseline comparisons.

## 5. Automation hooks
| Tool | Purpose | Status | Notes |
|------|---------|--------|-------|
| `scripts/package_pilot_release.sh` | Copies latest pilot checkpoint/config/logs into `artifacts/pilot_release/` and updates metadata | ✅ | Use after every significant checkpoint (e.g., 1k-step milestones) so collaborators can download a coherent bundle. |
| `scripts/eval/run_pilot_suite.sh` | Runs zero-shot, NIAH (up to 64k), and continual harnesses (plus optional TITAN baseline) with memorization flags enabled | ✅ | Set `HOPE_CHECKPOINT`, `TITAN_*`, etc., to reuse for each ablation. Outputs land under `eval/`. |

## 5. Evaluation checklist per ablation
1. Run zero-shot suite (`scripts/eval/zeroshot.py --tasks all --memorize ...`).
2. Run extended NIAH (`--context-lengths 2048 --context-lengths 4096 --context-lengths 8192 --context-lengths 16384 --context-lengths 32768 --context-lengths 65536`).
3. Run continual-learning harness with memorization toggles (`--memorize --memorize-steps 2 --memorize-no-reset` and baseline run without memorization).
4. Record metrics in `artifacts/pilot_release/` (JSON/CSV) and summarize deltas here.
5. Long-context extras: `scripts/eval/passkey.py` (default 64 prompts, memorize on) and `scripts/eval/pg19_perplexity.py` (streaming PG-19, 2 048-token truncation). Archive outputs alongside zero-shot/NIAH JSONs.
6. Continual plot: for multi-checkpoint evals, run `scripts/eval/plot_forgetting.py` and stash PNGs under `reports/plots/`.

_Status legend:_ ✅ complete, ⏳ pending, 🔄 running, ⚠️ blocked.

## 6. Reference snapshot – Pilot step 230k (HOPE)
| Eval | Command | Output | Notes |
|------|---------|--------|-------|
| Zero-shot (full suite, memorize on) | `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/zeroshot.py --config configs/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --tasks all --max-samples 256 --device cuda:1 --output eval/zeroshot_pilot_step230000.json --memorize --memorize-steps 2 --memorize-use-correct-answer` | `eval/zeroshot_pilot_step230000.json` | PIQA 0.496, HellaSwag 0.297, Winogrande 0.473, ARC-E/C 0.285/0.234, BoolQ 0.367, SIQA 0.316, CSQA 0.180, OpenBookQA 0.113. |
| NIAH (2k→65k, memorize on) | `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/niah.py --config configs/pilot.yaml --checkpoint artifacts/checkpoints/pilot/step_230000.pt --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --context-lengths 2048 --context-lengths 4096 --context-lengths 8192 --context-lengths 16384 --context-lengths 32768 --context-lengths 65536 --samples-per-length 8 --device cuda:1 --output eval/niah_pilot_step230000.json --memorize --memorize-steps 2 --memorize-use-correct-answer` | `eval/niah_pilot_step230000.json` | Accuracies 0.625 / 0.50 / 0.375 / 0.50 / 0.75 / 0.50 (2k→65k contexts). |
| Continual segments (memorize 1 step) | `UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/continual.py --config configs/pilot.yaml --checkpoints artifacts/checkpoints/pilot/step_230000.pt --segments-yaml configs/data/continual_segments_sample.yaml --tokenizer-path artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model --batch-size 4 --max-batches 20 --device cuda:1 --output eval/continual_pilot_step230000.json --memorize --memorize-steps 1` | `eval/continual_pilot_step230000.json` | CE ≈8.06 / 7.79 / 7.68 / 7.95 for RefinedWeb / Wikipedia / C4 / RedPajama sample segments. |

All outputs are copied to `artifacts/pilot_release/` via `scripts/package_pilot_release.sh` for reproducibility.

### Teach-scale=0.05 short-run notes
- **Run:** `uv run python train.py --config-name pilot model.teach_scale=0.05 train.steps=2000 ...` on GPU 0, checkpoints in `artifacts/checkpoints/pilot_teach05/`.
- **Training log:** `logs/pilot-teach05-20251114010549.json` (40 records, final loss 10.49, teach_signal_norm 7.8e‑3 at step 1950).
- **Zeroshot (128 samples, memorize on):** `eval/zeroshot_pilot_teach05_step2000.json` → PIQA 0.453, HellaSwag 0.273, Winogrande 0.508, ARC-E 0.250, ARC-C 0.227, BoolQ 0.664, SIQA 0.289, CSQA 0.188, OBQA 0.180.
- **NIAH:** `eval/niah_pilot_teach05_step2000.json` → 0.50 / 0.75 / 1.00 / 0.75 / 0.25 / 1.00 at 2k→65k.
- **Continual:** `eval/continual_pilot_teach05_step2000.json` → CE ≈37.4 / 33.2 / 35.9 / 32.9 on RefinedWeb/Wiki/C4/RedPajama segments (expectedly high because the run saw only 2 k steps).

### Teach-scale=0.15 short-run notes
- **Run:** `uv run python train.py --config-name pilot model.teach_scale=0.15 train.steps=2000 ...` on GPU 1, checkpoints at `artifacts/checkpoints/pilot_teach15/step_{001000,002000}.pt`.
- **Training log:** `logs/pilot-teach15-20251114012109.json` (40 records, final loss 8.70, ppl ≈6.0e3, teach_signal_norm ≈7.3e‑3).
- **Zeroshot (128 samples, memorize on):** `eval/zeroshot_pilot_teach15_step2000.json` → PIQA 0.484, HellaSwag 0.258, Winogrande 0.461, ARC-E 0.203, ARC-C 0.219, BoolQ 0.336, SIQA 0.344, CSQA 0.211, OBQA 0.148.
- **NIAH:** `eval/niah_pilot_teach15_step2000.json` → 0.75 / 0.75 / 0.75 / 0.50 / 0.25 / 0.50 (2k→65k).
- **Continual:** `eval/continual_pilot_teach15_step2000.json` → CE ≈69.4 / 66.6 / 66.5 / 68.6 (substantially higher than baseline because the run barely saw data; will normalize once longer steps are run).

### Teach-scale=0.05 long-run notes
- **Run:** 25 k-step job (`pilot-teach05-long-20251114155521`) on GPU 1. Checkpoints under `artifacts/checkpoints/pilot_teach05_long/step_*.pt`.
- **Training log:** `logs/pilot-teach05-long-20251114155521.json` (500 records; final loss ≈7.76, teach_sig_norm ≈7.3e‑3).
- **Zeroshot:** `eval/zeroshot_pilot_teach05_long_step25000.json` → PIQA 0.508, HellaSwag 0.285, Winogrande 0.477, ARC-E 0.320, ARC-C 0.238, BoolQ 0.367, SIQA 0.328, CSQA 0.199, OBQA 0.145.
- **NIAH:** `eval/niah_pilot_teach05_long_step25000.json` → 0.25 / 0.50 / 0.375 / 0.75 / 0.75 / 0.75.
- **Continual:** `eval/continual_pilot_teach05_long_step25000.json` → CE ≈52.1 / 49.4 / 48.9 / 50.9 (much higher than baseline despite the long run).

### Teach-scale=0.15 long-run notes
- **Run:** 25 k-step job (`pilot-teach15-long-20251114185448`) on GPU 1. Checkpoints under `artifacts/checkpoints/pilot_teach15_long/step_*.pt`.
- **Training log:** `logs/pilot-teach15-long-20251114185448.json` (500 records; final loss ≈7.76, teach_sig_norm ≈7.3e‑3).
- **Zeroshot:** `eval/zeroshot_pilot_teach15_long_step25000.json` → PIQA 0.496, HellaSwag 0.305, Winogrande 0.500, ARC-E 0.301, ARC-C 0.238, BoolQ 0.367, SIQA 0.316, CSQA 0.176, OBQA 0.125.
- **NIAH:** `eval/niah_pilot_teach15_long_step25000.json` → 0.75 / 0.625 / 0.375 / 0.75 / 0.50 / 0.75.
- **Continual:** `eval/continual_pilot_teach15_long_step25000.json` → CE ≈7.91 / 7.63 / 7.56 / 7.79 (comparable to the HOPE baseline).

### CMS chunk ablation – update_period=1
- **Run:** `pilot-cms-nochunk-20251114232720` on GPU 1 (5 k steps) with all CMS levels set to `update_period=1`.
- **Training log:** `logs/pilot-cms-nochunk-20251114232720.json` (100 records, final loss 8.65, teach_signal_norm ≈7.3e‑3).
- **Zeroshot:** `eval/zeroshot_pilot_cms_nochunk_step5000.json` → PIQA 0.520, HellaSwag 0.277, Winogrande 0.473, ARC-E 0.320, ARC-C 0.242, BoolQ 0.633, SIQA 0.301, CSQA 0.191, OpenBookQA 0.148.
- **NIAH:** `eval/niah_pilot_cms_nochunk_step5000.json` → 0.75 / 0.25 / 0.25 / 0.25 / 0.75 / 0.50.
- **Continual:** `eval/continual_pilot_cms_nochunk_step5000.json` → CE ≈46.6 / 47.8 / 49.7 / 52.1 (forgetting worsens without chunk accumulation).

### Self-modifier off (self_mod_lr=0)
- **Run:** `pilot-selfmod-off-20251115132848` on GPU 1 (5 k steps) with `model.self_mod_lr=0`.
- **Training log:** `logs/pilot-selfmod-off-20251115132848.json` (100 records, final loss 8.14, teach_signal_norm ≈7.3e‑3).
- **Zeroshot:** `eval/zeroshot_pilot_selfmod_off_step5000.json` → PIQA 0.516, HellaSwag 0.266, Winogrande 0.465, ARC-E 0.289, ARC-C 0.207, BoolQ 0.633, SIQA 0.332, CSQA 0.164, OpenBookQA 0.164.
- **NIAH:** `eval/niah_pilot_selfmod_off_step5000.json` → 0.75 / 0.75 / 0.50 / 0.75 / 0.25 / 0.75.
- **Continual:** `eval/continual_pilot_selfmod_off_step5000.json` → CE ≈45.7 / 44.9 / 44.4 / 45.5 (self-modifier appears critical for continual learning even at short horizons).

## 7. Upcoming experiments queue
| ID | Variant | Command seed | Notes |
|----|---------|--------------|-------|
| Q1 | TITAN baseline (9k steps) | `uv run python train.py --config-name mid_titan_baseline ... train.steps=9000` | ✅ W&B `titan-short-20251112195149`; metrics stored as `eval/*_titan.json`. |
| Q2 | Pilot long run (3 B tokens) | `tmux new -s pilot_full "... train.steps=246667 train.checkpoint.save_interval=1000"` | 🔄 Paused at step 246 667; release bundle now tracks step 230 000 (`artifacts/pilot_release/`). Resume after TITAN catches up. |
| Q3 | Teach-scale ablation | `+model.teach_scale=0.05/0.15` (pilot config) | Run 2 k-step jobs to quantify stability vs accuracy. |
| Q4 | CMS chunk toggle | `+model.cms_levels[].update_period=1` (Transformer-like) | Compare zero-shot/NIAH vs default chunking. |
| Q5 | Muon vs AdamW | `optim.type=muon` vs `adamw` | Use 5 k-step runs, document speed/quality in `docs/experiments_report.md`. |
| Q6 | TITAN long run (25 k steps) | `TMPDIR=/mnt/drive_4/tmp_titan UV_CACHE_DIR=/tmp/uv-cache uv run python train.py --config-name mid_titan_baseline ... train.steps=25000 train.checkpoint.save_interval=1000` | 🔄 Running on `cuda:0` (W&B `titan-long-20251113192738`); monitor `logs/titan_long.log` + wandb for checkpoints every 1 000 steps. |

Mark each queue item ✅/⏳/⚠️ as it progresses so we know which ablations have data ready for reporting.

## 9. Phase 3 – Self-modifying Titans (paper HOPE scaffold)

The paper-defined `hope_selfmod` scaffold (`Self-modifying Titans → CMS`) has its own knobs and
ablation-ready configs. These are intended for **implementation validation** and small-scale
experiments (not paper-scale reproduction).

| Config | Variant | Key override(s) |
|--------|---------|-----------------|
| `configs/hope/pilot_selfmod.yaml` | Pilot defaults | `model.block_variant=hope_selfmod`, `self_mod_chunk_size=8`, `self_mod_chunk_size_memory=64` |
| `configs/ablations/selfmod_rank1_precond_off.yaml` | No DGD preconditioner | `model.self_mod_use_rank1_precond=false` |
| `configs/ablations/selfmod_no_alpha.yaml` | No alpha/decay | `model.self_mod_use_alpha=false` |
| `configs/ablations/selfmod_chunked_8_64.yaml` | Explicit chunking | `model.self_mod_chunk_size=8`, `model.self_mod_chunk_size_memory=64` |
| `configs/ablations/selfmod_no_cms.yaml` | Selfmod-only | `model.cms_levels=[]` |
| `configs/ablations/selfmod_momentum_on.yaml` | Momentum on | `model.self_mod_momentum=0.9` |
| `configs/ablations/selfmod_momentum_off.yaml` | Momentum off | `model.self_mod_momentum=0.0` |

These configs require `src/nested_learning/training.py:build_model_from_cfg()` to plumb the self-mod
fields through `ModelConfig`; this is covered by `tests/test_build_model_from_cfg_selfmod.py`.

## 8. Baseline comparison (HOPE step 230k vs TITAN step 25k)
| Metric | HOPE | TITAN | Notes |
|--------|------|-------|-------|
| PIQA / HellaSwag / Winogrande | 0.496 / 0.297 / 0.473 | 0.484 / 0.293 / 0.480 | `eval/zeroshot_pilot_step230000.json` vs `eval/zeroshot_titan_step25000.json`. |
| ARC-E / ARC-C / BoolQ / SIQA / CSQA / OpenBookQA | 0.285 / 0.234 / 0.367 / 0.316 / 0.180 / 0.113 | 0.281 / 0.250 / 0.398 / 0.293 / 0.188 / 0.145 | Same zero-shot outputs as above. |
| NIAH (2k / 4k / 8k / 16k / 32k / 65k) | 0.625 / 0.50 / 0.375 / 0.50 / 0.75 / 0.50 | 0.50 / 0.625 / 0.125 / 0.75 / 0.50 / 0.125 | `eval/niah_pilot_step230000.json` vs `eval/niah_titan_step25000.json`. |
| Continual CE (RefinedWeb / Wiki / C4 / RedPajama) | 8.06 / 7.79 / 7.68 / 7.95 | 8.36 / 8.12 / 7.85 / 8.11 | `eval/continual_pilot_step230000.json` vs `eval/continual_titan_step25000.json`. |

Use these values as the reference when logging ablations; refresh the table whenever a new HOPE or TITAN checkpoint is evaluated.

### Additional snapshots (surprise-gated relaunch)
- **HOPE pilot relaunch (step 477k):** `reports/checkpoints/pilot_relaunch_step477000.md` with eval outputs `eval/*_pilot.json`.
- **TITAN long relaunch (step 32k):** `reports/checkpoints/titan_long_step32000.md` with eval outputs `eval/*_titan.json`.

These runs use `model.surprise_threshold=0.02` and therefore show ≈0 memorization deltas on short eval prompts (the gate did not trigger update events). Treat them as “pipeline / plumbing” validations rather than evidence of long-context advantages.
