"""Build an atomic evidence database from pinned Scripture and lexicon packages."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_store import EvidenceStore, file_sha256
from .lexicon_import import PreparedLexicon, prepare_lexicon
from .registry import load_json
from .source_import import PreparedSource, prepare_source, verify_source_approval


@dataclass(frozen=True, slots=True)
class EvidenceBuildResult:
    database_path: Path
    citation_corpus_path: Path
    manifest_path: Path
    inventory: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": str(self.database_path),
            "database_sha256": file_sha256(self.database_path),
            "citation_corpus_path": str(self.citation_corpus_path),
            "citation_corpus_sha256": file_sha256(self.citation_corpus_path),
            "manifest_path": str(self.manifest_path),
            "inventory": self.inventory,
        }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def build_approved_evidence(
    *,
    root: str | Path,
    cache_dir: str | Path,
    artifact_dir: str | Path,
    database_path: str | Path,
    citation_corpus_path: str | Path,
    manifest_path: str | Path,
    fetch: bool = False,
) -> EvidenceBuildResult:
    project = Path(root).resolve()
    cache = Path(cache_dir).resolve()
    artifacts = Path(artifact_dir).resolve()
    database = Path(database_path).resolve()
    citation_corpus = Path(citation_corpus_path).resolve()
    manifest = Path(manifest_path).resolve()
    registry_path = project / "configs/data/source_registry.json"
    canon_path = project / "configs/canon.json"
    source_lock_path = project / "configs/data/source_packages.json"
    lexicon_lock_path = project / "configs/data/lexicon_packages.json"

    scripture_packages = load_json(source_lock_path).get("packages", [])
    lexicon_packages = load_json(lexicon_lock_path).get("packages", [])
    prepared_sources: list[PreparedSource] = []
    prepared_lexicons: list[PreparedLexicon] = []
    for package in scripture_packages:
        prepared = prepare_source(
            str(package["source_id"]),
            lock_path=source_lock_path,
            canon_path=canon_path,
            output_dir=artifacts,
            cache_dir=cache,
            fetch=fetch,
        )
        verify_source_approval(
            prepared,
            registry_path=registry_path,
            lock_path=source_lock_path,
        )
        prepared_sources.append(prepared)
    for package in lexicon_packages:
        prepared_lexicons.append(
            prepare_lexicon(
                str(package["source_id"]),
                lock_path=lexicon_lock_path,
                output_dir=artifacts,
                cache_dir=cache,
                fetch=fetch,
            )
        )

    database.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{database.name}.", suffix=".tmp", dir=database.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        with EvidenceStore(temporary) as store:
            for item in prepared_sources:
                store.import_artifact(item.artifact_path, registry_path, canon_path)
            for item in prepared_lexicons:
                store.import_lexicon_artifact(item.artifact_path, registry_path)
            corpora = store.citation_corpora()
            inventory = tuple(store.source_inventory())
        temporary.replace(database)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    _write_json(
        citation_corpus,
        {
            "schema_version": "1.0",
            "sources": corpora,
            "organizational_source_ids": [],
            "excludes": ["lexicon definitions", "organizational documents"],
        },
    )
    _write_json(
        manifest,
        {
            "schema_version": "1.0",
            "database_sha256": file_sha256(database),
            "citation_corpus_sha256": file_sha256(citation_corpus),
            "inventory": inventory,
        },
    )
    return EvidenceBuildResult(database, citation_corpus, manifest, inventory)
