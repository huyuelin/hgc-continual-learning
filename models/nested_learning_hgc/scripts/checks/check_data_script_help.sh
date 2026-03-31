#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

tmp_out="$(mktemp)"
trap 'rm -f "${tmp_out}"' EXIT

fail=0

for script in scripts/data/*.py; do
  base="$(basename "${script}")"
  if [[ "${base}" == "__init__.py" ]]; then
    continue
  fi
  if ! uv run python "${script}" --help >"${tmp_out}" 2>&1; then
    echo "[help-check] FAIL ${script}"
    cat "${tmp_out}"
    fail=1
  fi
done

for script in scripts/data/*.sh; do
  if ! bash "${script}" --help >"${tmp_out}" 2>&1; then
    echo "[help-check] FAIL ${script}"
    cat "${tmp_out}"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "[help-check] OK scripts/data --help"
