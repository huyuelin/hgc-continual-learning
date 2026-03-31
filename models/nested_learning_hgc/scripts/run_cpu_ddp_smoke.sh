#!/usr/bin/env bash

set -euo pipefail

# Force CPU execution so torchrun selects the gloo backend.
export CUDA_VISIBLE_DEVICES=""

uv run torchrun --standalone --nproc_per_node=2 train_dist.py --config-name pilot_smoke "$@"

# Strict mechanism-auditing guardrails should fail fast under DDP.
if uv run torchrun --standalone --nproc_per_node=2 train_dist.py \
  --config-name pilot_smoke \
  train.strict_streaming_contract=true \
  train.fail_if_paper_faithful_disabled=true \
  >/tmp/ddp_strict_expected_fail.log 2>&1; then
  echo "[cpu-ddp-smoke] expected strict mode failure did not occur"
  exit 1
fi
