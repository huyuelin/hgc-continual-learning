# PyPI Trusted Publishing Setup

This repository ships `.github/workflows/release.yml` for OIDC-based publishing.
Use this checklist once per repository to activate it.
The same workflow also publishes a GitHub Release entry (Releases tab) with wheel/sdist/checksum assets for each tag.

## 1) Configure TestPyPI trusted publisher

In TestPyPI project settings (`nested-learning`):
- Publisher: **GitHub**
- Owner: `kmccleary3301`
- Repository: `nested_learning`
- Workflow name: `release.yml`
- Environment: `testpypi`

## 2) Configure PyPI trusted publisher

In PyPI project settings (`nested-learning`):
- Publisher: **GitHub**
- Owner: `kmccleary3301`
- Repository: `nested_learning`
- Workflow name: `release.yml`
- Environment: `pypi`

## 3) Validate release tags

- RC tags (publish to TestPyPI): `vX.Y.ZrcN`
- Stable tags (publish to PyPI): `vX.Y.Z`

Example:
```bash
git tag v0.2.0rc1
git push origin v0.2.0rc1
```

## 4) Verify workflow permissions

`release.yml` requires:
- `id-token: write` (for OIDC)
- `contents: write`

No long-lived PyPI API tokens are required.

## 5) Recommended first dry-run

1. Create RC tag and publish to TestPyPI.
2. Create clean virtualenv.
3. Install package from TestPyPI.
4. Run:
   - `python -m nested_learning --help`
   - `python -m nested_learning doctor --json`
   - `python -m nested_learning smoke --config-name pilot_smoke --device cpu --batch-size 1 --seq-len 8`

## 6) Verify GitHub release assets

After the tag workflow completes, confirm the Releases tab entry for that tag contains:
- `nested_learning-<version>-py3-none-any.whl`
- `nested_learning-<version>.tar.gz`
- `SHA256SUMS.txt`

## 7) Verify GitHub Packages tab (GHCR)

The repository also ships `.github/workflows/packages.yml`, which publishes:
- `ghcr.io/<owner>/nested-learning-dist:<tag>`

This is an OCI artifact bundle for distribution files (`dist/*`) and appears in the GitHub Packages tab.
Use PyPI for normal `pip install` workflows.
