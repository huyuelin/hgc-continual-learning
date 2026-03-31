# Security / Release Gate Log

Executed at: `2026-02-24T00:40:32Z` (UTC)

## Commands Run
- `rg -n --hidden --glob '!.git' --glob '!*.pt' --glob '!*.bin' "(AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{20,}|BEGIN PRIVATE KEY|SECRET_KEY|API_KEY|PASSWORD=|token=)" .`
- `git ls-files | rg -n "(\\.pt$|\\.ckpt$|\\.safetensors$|\\.npy$|\\.zip$|git\\.env|docs_tmp|artifacts/|data/raw/)"`
- `git ls-files -s | awk '{print $4}' | xargs -r du -h | sort -h | tail -n 30`
- `git check-ignore -v artifacts/checkpoints/pilot/step_000001.pt logs/mechanism_audit_smoke.json docs_tmp/placeholder.txt data/raw/example.txt git.env docs/POSTS.md`
- `bash scripts/checks/check_git_tracked_sizes.sh`

## Findings
- Secret-pattern scan: no credentials/tokens detected in tracked content.
- Tracked artifact scan: no forbidden checkpoint/artifact extensions are tracked.
- `.gitignore` coverage confirmed for:
  - `artifacts/`
  - `logs/`
  - `data/`
  - `docs_tmp/`
  - `git.env`
  - `docs/POSTS.md`
- Largest tracked files are paper/reference artifacts and lockfile; all below size gate threshold (`5 MiB`) enforced by `scripts/checks/check_git_tracked_sizes.sh`.

## Remediation / Guardrails Added
- Added `scripts/checks/check_git_tracked_sizes.sh`:
  - fails CI for forbidden tracked artifact extensions (`.pt`, `.ckpt`, `.safetensors`, `.npy`, `.npz`, `.zip`)
  - fails CI for tracked files above `MAX_TRACKED_FILE_BYTES` (default `5 MiB`)
- Added CI step in `.github/workflows/ci.yml` to run size/artifact gate.
- Added test `tests/test_git_tracked_sizes_check.py`.
- Added package-script test `tests/test_package_release_script.py` validating manifest train flags and that raw data is not included in release bundle.

## Status
- Gate result: `PASS`
