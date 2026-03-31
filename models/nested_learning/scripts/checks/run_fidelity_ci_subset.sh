#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

uv run python scripts/checks/verify_docs_refs.py
bash scripts/checks/check_git_tracked_sizes.sh
bash scripts/checks/check_data_script_help.sh

uv run pytest \
  tests/test_algorithm_mode_grad.py \
  tests/test_boundary_state_mode.py \
  tests/test_attention_cache.py \
  tests/test_teach_signal.py \
  tests/test_cms.py \
  tests/test_cms_cross_call.py \
  tests/test_cms_flush_partial.py \
  tests/test_online_chunking.py \
  tests/test_surprise_override.py \
  tests/test_model_streaming_cadence.py \
  tests/test_verify_update_cadence.py \
  tests/test_eval_state.py \
  tests/test_optim.py \
  tests/test_distributed_fail_fast.py \
  tests/test_fast_state_batch_semantics.py \
  tests/test_strict_streaming_contract.py \
  tests/test_tied_weight_guard.py \
  tests/test_verify_docs_refs.py \
  tests/test_paper_faithful_configs.py \
  tests/test_compliance_report.py \
  tests/test_compile_toggle.py

uv run python scripts/checks/compliance_report.py \
  --config configs/pilot.yaml \
  --output /tmp/compliance_report_ci.json

bash scripts/run_mechanism_audit_smoke.sh
