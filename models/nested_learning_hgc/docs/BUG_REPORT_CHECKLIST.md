# Bug Report Checklist

Use this checklist when filing reproducibility or correctness issues.

## Required Context

- Commit SHA (`git rev-parse --short HEAD`)
- Exact command line used
- Config name and CLI overrides
- Device/runtime details (`python --version`, `uv --version`, `nvidia-smi` if CUDA)

## Required Artifacts

- JSON training log path (if training path involved)
- Full traceback/error output
- Minimal failing input or dataset pointer
- If streaming/cadence related: include `scripts/checks/verify_update_cadence.py` output
- Include `scripts/checks/compliance_report.py` output (or note why unavailable)

## Fast Reproduction Path

1. Run `uv run bash scripts/checks/run_fidelity_ci_subset.sh`.
2. Run `uv run bash scripts/run_mechanism_audit_smoke.sh`.
3. Attach outputs and note which step failed.

## Streaming/Cadence Issues

- Specify `train.strict_streaming_contract` value.
- Specify `train.online_updates`, `train.online_chunk_size`, `train.online_boundary_targets`, `train.online_carry_attention_cache`, `model.cms_flush_partial_at_end`.
- Include expected vs observed update counts per level.
