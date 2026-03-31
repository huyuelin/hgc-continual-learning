# Changelog

All notable changes to this project will be documented here. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning once tagged releases begin.

## [Unreleased]
### Added
- Optional attention KV-cache path for continuous streaming inference (`init_attention_cache`, `attention_cache`, `return_attention_cache`) across HOPE/TITAN/Transformer blocks.
- Boundary-target online chunking mode (`train.online_boundary_targets`) and optional training-time attention-cache carry (`train.online_carry_attention_cache`) for stronger chunk-boundary semantics.
- Evaluation streaming-state utilities (`src/nested_learning/eval_state.py`) plus continual-eval controls (`--eval-state-mode`, `--eval-use-fast-state`, `--eval-use-attention-cache`).
- Compliance report automation (`scripts/checks/compliance_report.py`) with CI subset + mechanism smoke integration.
- Flash/SDPA-backed self-attention path with safe fallbacks, unlocking PyTorch 2.9 SDPA kernels by default.
- Hydra toggles for bf16 autocast (`train.mixed_precision.*`), `torch.compile` (`train.compile.*`), and fused optimizers.
- Muon + AdamW hybrid optimizer option exposed via `optim.type=muon`, routing ≥2D matrices through `torch.optim.Muon`.
- Test-time memorization flags (`--memorize*`) documented in README + `docs/guide.md`, matching TITAN eval behavior.
- Automation helpers: `scripts/run_e2e_smoke.sh` documented in Quickstart, plus new `scripts/run_cpu_ddp_smoke.sh` for CPU-only DDP/gloo smoke coverage.
- Streaming contract doc (`docs/STREAMING_CONTRACT.md`) defining sequence/segment/chunk semantics and fast-state lifecycle.
- Cadence verification utility (`scripts/checks/verify_update_cadence.py`) with synthetic tests and release-checklist integration.
- Fidelity CI subset runner (`scripts/checks/run_fidelity_ci_subset.sh`) and mechanism-auditing smoke runner (`scripts/run_mechanism_audit_smoke.sh`).
- Progress/status docs for P7 execution (`docs/PLAN_PROGRESS_P7.md`, `docs/IMPLEMENTATION_STATUS.md`).
- Bug-report reproducibility checklist (`docs/BUG_REPORT_CHECKLIST.md`).
- Boundary-state training-loop regression coverage (`tests/test_boundary_state_training_loop.py`) plus eval-loader/metadata roundtrip coverage (`tests/test_checkpoint_metadata_and_eval_loaders.py`).
- `scripts/checks/check_data_script_help.sh` to guarantee `scripts/data/* --help` exits cleanly; wired into CI.
- Markdown anchor verification in `scripts/checks/verify_docs_refs.py` with dedicated unit coverage.
- Tag release automation now creates GitHub Release entries with attached wheel/sdist artifacts plus `SHA256SUMS.txt`.
- Added GHCR package publishing workflow (`.github/workflows/packages.yml`) so the Packages tab contains a versioned `nested-learning-dist` OCI bundle.

### Changed
- README / compliance / streaming docs now reflect boundary-target mode, optional KV-cache carry, and explicit scope boundaries.
- CPU DDP smoke now includes strict-mode fail-fast verification.
- Repository license metadata now matches the shipped Apache-2.0 text; badges updated accordingly.
- README and guide refreshed with performance knobs, optimizer guidance, and memorization instructions so release consumers have a single source of truth.
- Release checklist tracks the new CPU DDP smoke script to keep packaging instructions aligned with available tooling.
- Training loop strict-mode guardrails: `train.strict_streaming_contract` now fail-fasts on known semantics violations (DDP feature downgrades, shared-batch fast-state, non paper-defined variant in strict mode).
- CMS telemetry now includes cadence metrics (`updates_applied`, `tokens_flushed`, `pending_tokens`, `gate_hits`) to make update-frequency behavior auditable.
- Paper-auditing preset now explicitly enables strict streaming contract checks.
- `configs/pilot_paper_faithful.yaml` now explicitly sets `train.online_updates=true` and tests verify no implicit algorithm-mode fallback.
- Boundary-state mode now emits an explicit startup warning code (`experimental_boundary_state_mode`) and validates cache/chunk constraints early.
- Checkpoint metadata now records algorithm/online flags (`algorithm_mode`, `online_updates`, `online_boundary_targets`, `online_carry_attention_cache`, `use_fast_state`), and release manifest includes those flags.
- Data split fallback policy is deterministic across data scripts (`train -> validation -> test -> first available`) with explicit available-splits logging.

### Upcoming
- GitHub Actions workflow covering `ruff`, `mypy`, and `pytest`.
- End-to-end release dry-run ahead of the `v0.1.0` tag.

## [0.1.0] - 2025-11-09
### Added
- PyTorch **2.9.0** / torchvision **0.24.0** environment managed via `uv` with reproducible `pyproject.toml` + `uv.lock`.
- HOPE block implementation (attention → TITAN memory → CMS + deep optimizers) with configurable level clocks and self-modifier wiring.
- Hydrated Hydra config tree for pilot, mid, target, and CPU-only smoke runs plus DDP/FSDP/DeepSpeed entrypoints.
- Data tooling: tokenizer trainer, corpus filtering, mixture processing, and `scripts/data/run_sample.sh` shortcut emitting stats under `data/mixtures/`.
- Evaluation suite: zero-shot benchmark CLI (PIQA/HellaSwag/WinoGrande/ARC/BoolQ/SIQA), Needle-in-a-Haystack generator, continual-learning forgetting analyzer.
- Sample artifacts (`artifacts/examples/pilot_dummy.pt`, `logs/pilot_smoke.json`, `logs/mid_smoke.json`) for reproducing eval commands without lengthy training.
- Documentation set (`docs/stage1_plan.md`, `docs/stage2_plan.md`, `docs/data_pipeline.md`, `docs/guide.md`) outlining architecture, scaling strategy, and onboarding.

### Changed
- README rewritten with badges, quickstart commands, and references to the new guide + release checklist.
- Logging defaults clarified (`logging.backend=json|wandb`), with instructions for saving structured metrics under `logs/`.

### Known gaps
- Release automation and CI are tracked in `docs/release_plan.md`.
- Scaling guidance for >100 B token corpora pending additional storage + GPU availability.
