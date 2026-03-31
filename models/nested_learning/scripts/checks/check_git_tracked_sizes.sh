#!/usr/bin/env bash
set -euo pipefail

MAX_BYTES="${MAX_TRACKED_FILE_BYTES:-5242880}"  # 5 MiB default
FORBIDDEN_EXT_REGEX='(\.pt|\.ckpt|\.safetensors|\.npy|\.npz|\.zip)$'

fail=0

while IFS= read -r path; do
  [[ -f "${path}" ]] || continue
  if [[ "${path}" =~ ${FORBIDDEN_EXT_REGEX} ]]; then
    echo "[size-check] forbidden tracked artifact extension: ${path}"
    fail=1
  fi
  size=$(wc -c < "${path}")
  if (( size > MAX_BYTES )); then
    echo "[size-check] tracked file exceeds ${MAX_BYTES} bytes: ${path} (${size})"
    fail=1
  fi
done < <(git ls-files)

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "[size-check] OK (max_bytes=${MAX_BYTES})"
