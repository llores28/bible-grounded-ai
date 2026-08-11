"""License-locked Hebrew, Aramaic, and Koine Greek lexicon preparation."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_store import file_sha256
from .registry import load_json
from .source_import import SourceImportError, _verified_file, _write_canonical


@dataclass(frozen=True, slots=True)
class PreparedLexicon:
    source_id: str
    revision: str
    artifact_path: Path
    artifact_sha256: str
    entry_count: int
    language_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "revision": self.revision,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "entry_count": self.entry_count,
            "language_counts": self.language_counts,
        }


_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def _space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _unique_text(elements: list[ET.Element]) -> str:
    values: list[str] = []
    for element in elements:
        value = _space("".join(element.itertext()))
        if value and value not in values:
            values.append(value)
    return "; ".join(values)


def _entry(
    *,
    entry_id: str,
    language: str,
    lemma: str,
    transliteration: str,
    gloss: str,
    definition: str,
    source_ref: str,
) -> dict[str, str]:
    if not entry_id or not lemma or not definition:
        raise SourceImportError(f"lexicon entry is incomplete: {entry_id or '<unknown>'}")
    return {
        "entry_id": entry_id,
        "language": language,
        "lemma": lemma,
        "transliteration": transliteration,
        "gloss": gloss,
        "definition": definition,
        "source_ref": source_ref,
    }


def _parse_hebrew_lexicon(files: dict[str, Path]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    strong_root = ET.parse(files["HebrewStrong.xml"]).getroot()
    for item in strong_root.findall(".//{*}entry"):
        word = item.find("./{*}w")
        if word is None:
            continue
        language = {
            "heb": "Hebrew",
            "arc": "Aramaic",
            "x-pn": "Hebrew/Aramaic proper name",
        }.get(word.get(_XML_LANG, ""), word.get(_XML_LANG, "Unknown"))
        gloss = _unique_text(item.findall(".//{*}def"))
        entries.append(
            _entry(
                entry_id=f"STRONG:{item.get('id', '')}",
                language=language,
                lemma=_space("".join(word.itertext())),
                transliteration=word.get("xlit", ""),
                gloss=gloss,
                definition=_space("".join(item.itertext())),
                source_ref=f"HebrewStrong.xml#{item.get('id', '')}",
            )
        )

    bdb_root = ET.parse(files["BrownDriverBriggs.xml"]).getroot()
    for part in bdb_root.findall(".//{*}part"):
        language = {"heb": "Hebrew", "arc": "Aramaic"}.get(
            part.get(_XML_LANG, ""), part.get(_XML_LANG, "Unknown")
        )
        for item in part.findall(".//{*}entry"):
            word = item.find("./{*}w")
            if word is None:
                continue
            entry_id = str(item.get("id", ""))
            entries.append(
                _entry(
                    entry_id=f"BDB:{entry_id}",
                    language=language,
                    lemma=_space("".join(word.itertext())),
                    transliteration=word.get("xlit", ""),
                    gloss=_unique_text(item.findall(".//{*}def")),
                    definition=_space("".join(item.itertext())),
                    source_ref=f"BrownDriverBriggs.xml#{entry_id}",
                )
            )
    return entries


def _parse_abbott_smith(files: dict[str, Path]) -> list[dict[str, str]]:
    root = ET.parse(files["abbott-smith.tei.xml"]).getroot()
    entries: list[dict[str, str]] = []
    for position, item in enumerate(root.findall(".//{*}entry"), start=1):
        label = str(item.get("n", ""))
        lemma, separator, strong = label.partition("|")
        orth = item.find("./{*}form/{*}orth")
        if orth is None:
            orth = item.find("./{*}form/{*}foreign")
        if orth is None:
            orth = item.find("./{*}form/{*}form[@type='lemma']/{*}foreign")
        headword = _space("".join(orth.itertext())) if orth is not None else _space(lemma)
        number = strong if separator and strong else "UNNUMBERED"
        stable_id = f"{number}-{position:04d}"
        entries.append(
            _entry(
                entry_id=f"ABBOTT_SMITH:{stable_id}",
                language="Koine Greek",
                lemma=headword,
                transliteration="",
                gloss=_unique_text(item.findall(".//{*}gloss")),
                definition=_space("".join(item.itertext())),
                source_ref=f"abbott-smith.tei.xml#{stable_id}",
            )
        )
    return entries


_PARSERS: dict[str, Callable[[dict[str, Path]], list[dict[str, str]]]] = {
    "openscriptures_hebrew_lexicon": _parse_hebrew_lexicon,
    "abbott_smith_tei": _parse_abbott_smith,
}


def prepare_lexicon(
    source_id: str,
    *,
    lock_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    fetch: bool = False,
) -> PreparedLexicon:
    lock = load_json(lock_path)
    packages = [item for item in lock.get("packages", []) if item.get("source_id") == source_id]
    if len(packages) != 1:
        raise SourceImportError(f"lexicon lock must contain exactly one package for {source_id}")
    package = packages[0]
    parser_name = str(package.get("parser", ""))
    if parser_name not in _PARSERS:
        raise SourceImportError(f"unsupported lexicon parser: {parser_name}")
    files = {
        str(item["filename"]): _verified_file(item, Path(cache_dir), fetch=fetch)
        for item in package.get("downloads", [])
    }
    entries = _PARSERS[parser_name](files)
    entries.sort(key=lambda item: (item["language"], item["lemma"], item["entry_id"]))
    ids = [item["entry_id"] for item in entries]
    if len(ids) != len(set(ids)):
        raise SourceImportError(f"duplicate lexicon entry IDs in {source_id}")
    expected = int(package["expected_entry_count"])
    if len(entries) != expected:
        raise SourceImportError(
            f"lexicon entry count mismatch for {source_id}: {len(entries)}/{expected}"
        )
    language_counts: dict[str, int] = {}
    for item in entries:
        language_counts[item["language"]] = language_counts.get(item["language"], 0) + 1
    expected_languages = package.get("expected_language_counts")
    if expected_languages is not None and language_counts != expected_languages:
        raise SourceImportError(
            f"lexicon language counts mismatch for {source_id}: "
            f"{language_counts}/{expected_languages}"
        )
    artifact = Path(output_dir) / f"{source_id}.json"
    digest = _write_canonical(
        artifact,
        {
            "schema_version": "1.0",
            "source_id": source_id,
            "revision": package["revision"],
            "entries": entries,
        },
    )
    expected_digest = str(package.get("canonical_artifact_sha256") or "").lower()
    if expected_digest and digest != expected_digest:
        artifact.unlink(missing_ok=True)
        raise SourceImportError(
            f"canonical lexicon digest mismatch for {source_id}: "
            f"expected {expected_digest}, got {digest}"
        )
    return PreparedLexicon(
        source_id=source_id,
        revision=str(package["revision"]),
        artifact_path=artifact,
        artifact_sha256=file_sha256(artifact),
        entry_count=len(entries),
        language_counts=language_counts,
    )
