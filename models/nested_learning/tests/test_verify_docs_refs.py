import importlib.util
from pathlib import Path


def _load_verify_docs_refs():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "checks" / "verify_docs_refs.py"
    spec = importlib.util.spec_from_file_location("verify_docs_refs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load verify_docs_refs script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_parse_referenced_paths_extracts_repo_paths() -> None:
    module = _load_verify_docs_refs()
    text = """
Use `scripts/checks/run_fidelity_ci_subset.sh` and `uv run pytest`.
Also see [status](docs/IMPLEMENTATION_STATUS.md) and `README.md`.
Code pointer `src/nested_learning/training.py:225`.
Ignore `https://example.com` and `--flag`.
"""
    refs = module.parse_referenced_paths(text)
    assert "scripts/checks/run_fidelity_ci_subset.sh" in refs
    assert "docs/IMPLEMENTATION_STATUS.md" in refs
    assert "README.md" in refs
    assert "src/nested_learning/training.py" in refs
    assert "https://example.com" not in refs


def test_verify_docs_refs_reports_missing_paths(tmp_path: Path) -> None:
    module = _load_verify_docs_refs()
    (tmp_path / "scripts" / "checks").mkdir(parents=True)
    existing = tmp_path / "scripts" / "checks" / "ok.sh"
    existing.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text(
        "`scripts/checks/ok.sh` `scripts/checks/missing.sh`",
        encoding="utf-8",
    )
    missing, missing_anchors = module.verify_docs_refs(repo_root=tmp_path, docs=[doc])
    assert str(doc) in missing
    assert missing[str(doc)] == ["scripts/checks/missing.sh"]
    assert missing_anchors == {}


def test_verify_docs_refs_validates_markdown_anchors(tmp_path: Path) -> None:
    module = _load_verify_docs_refs()
    target = tmp_path / "docs" / "guide.md"
    target.parent.mkdir(parents=True)
    target.write_text(
        "# Overview\n\n## Streaming Contract\n",
        encoding="utf-8",
    )
    doc = tmp_path / "doc.md"
    doc.write_text(
        "[ok](docs/guide.md#overview) [bad](docs/guide.md#missing-anchor)",
        encoding="utf-8",
    )
    missing, missing_anchors = module.verify_docs_refs(repo_root=tmp_path, docs=[doc])
    assert missing == {}
    assert str(doc) in missing_anchors
    assert missing_anchors[str(doc)] == ["docs/guide.md#missing-anchor"]
