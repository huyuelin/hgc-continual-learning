# Next Backlog (Scoped, Non-Feature-Creep)

1. Stabilize boundary-state mode for longer single-GPU runs (memory profiling + guardrail docs).
2. Add optional `require_strict` compliance gate job in CI for paper-faithful config only.
3. Expand packaging tests to assert required sidecars for both HOPE and TITAN checkpoints.
4. Add deterministic mini-run harness that stores and compares two short metric traces.
5. Add doc automation that validates CLI flags mentioned in README against `--help` output snapshots.
6. Keep result-scale work explicitly optional (no default 100B+ reproduction commitments).
