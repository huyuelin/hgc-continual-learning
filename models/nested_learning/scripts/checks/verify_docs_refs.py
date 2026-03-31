#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

DEFAULT_DOCS = (
    "README.md",
    "docs/PAPER_COMPLIANCE.md",
    "docs/STREAMING_CONTRACT.md",
    "docs/IMPLEMENTATION_STATUS.md",
    "docs/release_checklist.md",
    "docs/BUG_REPORT_CHECKLIST.md",
)

PATH_PREFIXES = (
    "src/",
    "scripts/",
    "tests/",
    "configs/",
    "docs/",
    ".github/",
)

SCHEME_PREFIXES = ("http://", "https://", "mailto:")


def _iter_code_spans(text: str) -> Iterable[str]:
    for match in re.finditer(r"`([^`\n]+)`", text):
        yield match.group(1)


def _iter_link_targets(text: str) -> Iterable[str]:
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        yield match.group(1)


def _normalize_path_candidate(token: str) -> str | None:
    token = token.strip().strip("`")
    token = token.strip("()[]{}'\".,;:")
    if not token:
        return None
    if token.startswith(SCHEME_PREFIXES):
        return None
    if token.startswith(("-", "--")):
        return None
    if any(ch in token for ch in ("<", ">", "{", "}", "*", "|", "$")):
        return None
    token = re.sub(r":\d+(?::\d+)?$", "", token)
    if "#" in token and not token.startswith("#"):
        token = token.split("#", 1)[0]
    if token.startswith("./"):
        token = token[2:]
    if token.startswith("../"):
        return None
    if token == "README.md":
        return token
    if "/" not in token:
        return None
    if not token.startswith(PATH_PREFIXES):
        return None
    return token


def parse_referenced_paths(doc_text: str) -> set[str]:
    refs: set[str] = set()
    for span in _iter_code_spans(doc_text):
        for piece in span.split():
            normalized = _normalize_path_candidate(piece)
            if normalized is not None:
                refs.add(normalized)
    for target in _iter_link_targets(doc_text):
        normalized = _normalize_path_candidate(target)
        if normalized is not None:
            refs.add(normalized)
    return refs


def _slugify_heading(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[`*_~]", "", slug)
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _extract_markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*)\s*$", line)
        if not match:
            continue
        base = _slugify_heading(match.group(1))
        if not base:
            continue
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        anchor = base if idx == 0 else f"{base}-{idx}"
        anchors.add(anchor)
    return anchors


def parse_anchor_references(doc_text: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for target in _iter_link_targets(doc_text):
        cleaned = target.strip()
        if cleaned.startswith(SCHEME_PREFIXES) or cleaned.startswith("#"):
            continue
        if "#" not in cleaned:
            continue
        path_part, anchor = cleaned.split("#", 1)
        if not anchor:
            continue
        normalized = _normalize_path_candidate(path_part)
        if normalized is None:
            continue
        refs.append((normalized, anchor.strip()))
    return refs


def verify_docs_refs(
    *,
    repo_root: Path,
    docs: list[Path],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    missing: dict[str, list[str]] = {}
    missing_anchors: dict[str, list[str]] = {}
    anchor_cache: dict[Path, set[str]] = {}
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        refs = sorted(parse_referenced_paths(text))
        missing_for_doc = [ref for ref in refs if not (repo_root / ref).exists()]
        if missing_for_doc:
            missing[str(doc)] = missing_for_doc
        bad_anchors: list[str] = []
        for ref_path, anchor in parse_anchor_references(text):
            candidate = repo_root / ref_path
            if not candidate.exists() or candidate.suffix.lower() != ".md":
                continue
            anchors = anchor_cache.get(candidate)
            if anchors is None:
                anchors = _extract_markdown_anchors(candidate)
                anchor_cache[candidate] = anchors
            if anchor not in anchors:
                bad_anchors.append(f"{ref_path}#{anchor}")
        if bad_anchors:
            missing_anchors[str(doc)] = sorted(set(bad_anchors))
    return missing, missing_anchors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that code/documentation references in docs resolve to existing files."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=list(DEFAULT_DOCS),
        help="Docs to scan (repo-relative paths).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    docs = [repo_root / Path(p) for p in args.docs]
    missing, missing_anchors = verify_docs_refs(repo_root=repo_root, docs=docs)
    payload = {
        "ok": len(missing) == 0 and len(missing_anchors) == 0,
        "docs_checked": [str(d.relative_to(repo_root)) for d in docs],
        "missing": missing,
        "missing_anchors": missing_anchors,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if missing or missing_anchors:
        print(json.dumps(payload, indent=2))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "docs_checked": payload["docs_checked"],
                "message": "all referenced repo paths and markdown anchors exist",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
