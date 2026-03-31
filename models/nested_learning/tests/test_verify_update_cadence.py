import importlib.util
import json
from pathlib import Path


def _load_verify_cadence():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "checks" / "verify_update_cadence.py"
    )
    spec = importlib.util.spec_from_file_location("verify_update_cadence", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load verify_update_cadence script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.verify_cadence


def _write_log(path: Path, payload: dict[str, float]) -> None:
    path.write_text(json.dumps([{"step": 0, **payload}], indent=2))


def test_verify_update_cadence_no_flush(tmp_path: Path) -> None:
    verify_cadence = _load_verify_cadence()
    log_path = tmp_path / "metrics.json"
    _write_log(
        log_path,
        {
            "layer0.cms.cms_fast.updates_applied": 2.0,
            "layer0.cms.cms_fast.chunk_tokens": 8.0,
            "layer0.cms.cms_fast.tokens_flushed": 0.0,
            "layer0.cms.cms_fast.pending_tokens": 2.0,
        },
    )
    report = verify_cadence(
        log_path=log_path,
        metric_prefix="layer0.cms.cms_fast",
        total_tokens=10,
        update_period=4,
        flush_partial=False,
    )
    assert report["ok"] is True


def test_verify_update_cadence_with_flush(tmp_path: Path) -> None:
    verify_cadence = _load_verify_cadence()
    log_path = tmp_path / "metrics.json"
    _write_log(
        log_path,
        {
            "layer0.cms.cms_fast.updates_applied": 3.0,
            "layer0.cms.cms_fast.chunk_tokens": 10.0,
            "layer0.cms.cms_fast.tokens_flushed": 2.0,
            "layer0.cms.cms_fast.pending_tokens": 0.0,
        },
    )
    report = verify_cadence(
        log_path=log_path,
        metric_prefix="layer0.cms.cms_fast",
        total_tokens=10,
        update_period=4,
        flush_partial=True,
    )
    assert report["ok"] is True


def test_verify_update_cadence_detects_mismatch(tmp_path: Path) -> None:
    verify_cadence = _load_verify_cadence()
    log_path = tmp_path / "metrics.json"
    _write_log(
        log_path,
        {
            "layer0.cms.cms_fast.updates_applied": 1.0,
            "layer0.cms.cms_fast.chunk_tokens": 4.0,
            "layer0.cms.cms_fast.tokens_flushed": 0.0,
            "layer0.cms.cms_fast.pending_tokens": 0.0,
        },
    )
    report = verify_cadence(
        log_path=log_path,
        metric_prefix="layer0.cms.cms_fast",
        total_tokens=10,
        update_period=4,
        flush_partial=False,
    )
    assert report["ok"] is False


def test_verify_update_cadence_report_schema_is_non_empty(tmp_path: Path) -> None:
    verify_cadence = _load_verify_cadence()
    log_path = tmp_path / "metrics.json"
    _write_log(
        log_path,
        {
            "layer0.cms.cms_fast.updates_applied": 2.0,
            "layer0.cms.cms_fast.chunk_tokens": 8.0,
            "layer0.cms.cms_fast.tokens_flushed": 0.0,
            "layer0.cms.cms_fast.pending_tokens": 2.0,
        },
    )
    report = verify_cadence(
        log_path=log_path,
        metric_prefix="layer0.cms.cms_fast",
        total_tokens=10,
        update_period=4,
        flush_partial=False,
    )
    for key in (
        "ok",
        "metric_prefix",
        "log_path",
        "expected",
        "observed",
        "checks",
    ):
        assert key in report
    assert report["metric_prefix"] == "layer0.cms.cms_fast"
    assert isinstance(report["checks"], dict)
    assert report["checks"]
