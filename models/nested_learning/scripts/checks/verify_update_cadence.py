#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _expected_counts(
    *,
    total_tokens: int,
    update_period: int,
    flush_partial: bool,
) -> dict[str, float]:
    if update_period <= 0:
        raise ValueError("update_period must be > 0")
    if total_tokens < 0:
        raise ValueError("total_tokens must be >= 0")
    full_updates = total_tokens // update_period
    remainder = total_tokens % update_period
    updates = full_updates + (1 if flush_partial and remainder > 0 else 0)
    chunk_tokens = float(total_tokens - (0 if flush_partial else remainder))
    return {
        "updates_applied": float(updates),
        "chunk_tokens": chunk_tokens,
        "tokens_flushed": float(remainder if flush_partial else 0),
        "pending_tokens": float(0 if flush_partial else remainder),
        "remainder_tokens": float(remainder),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("JSON log must be a list of records")
    return [rec for rec in records if isinstance(rec, dict)]


def _find_last_with_prefix(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    suffixes = ("updates_applied", "chunk_tokens", "tokens_flushed", "pending_tokens", "gate_hits")
    for record in reversed(records):
        for suffix in suffixes:
            if f"{prefix}.{suffix}" in record:
                return record
    raise ValueError(f"No record found for metric prefix {prefix!r}")


def verify_cadence(
    *,
    log_path: Path,
    metric_prefix: str,
    total_tokens: int,
    update_period: int,
    flush_partial: bool,
    atol: float = 1e-6,
) -> dict[str, Any]:
    records = _load_records(log_path)
    record = _find_last_with_prefix(records, metric_prefix)
    expected = _expected_counts(
        total_tokens=total_tokens,
        update_period=update_period,
        flush_partial=flush_partial,
    )
    observed = {
        "updates_applied": float(record.get(f"{metric_prefix}.updates_applied", 0.0)),
        "chunk_tokens": float(record.get(f"{metric_prefix}.chunk_tokens", 0.0)),
        "tokens_flushed": float(record.get(f"{metric_prefix}.tokens_flushed", 0.0)),
        "pending_tokens": float(record.get(f"{metric_prefix}.pending_tokens", 0.0)),
    }
    checks = {
        key: abs(observed[key] - expected[key]) <= atol
        for key in ("updates_applied", "chunk_tokens", "tokens_flushed", "pending_tokens")
    }
    ok = all(checks.values())
    return {
        "ok": ok,
        "metric_prefix": metric_prefix,
        "log_path": str(log_path),
        "flush_partial": flush_partial,
        "total_tokens": total_tokens,
        "update_period": update_period,
        "expected": expected,
        "observed": observed,
        "checks": checks,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify CMS update cadence from JSON logs.")
    parser.add_argument("--log-path", required=True, type=Path, help="Path to JSON metrics log.")
    parser.add_argument(
        "--metric-prefix",
        required=True,
        help="Metric prefix, e.g. layer0.cms.cms_fast",
    )
    parser.add_argument("--total-tokens", required=True, type=int)
    parser.add_argument("--update-period", required=True, type=int)
    parser.add_argument(
        "--flush-partial",
        action="store_true",
        help="Use ceil(T/C) expectation and zero pending tokens.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used for float comparisons.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON report.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = verify_cadence(
        log_path=args.log_path,
        metric_prefix=args.metric_prefix,
        total_tokens=args.total_tokens,
        update_period=args.update_period,
        flush_partial=args.flush_partial,
        atol=args.atol,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
