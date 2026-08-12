"""Reproducible, license-locked biblical source preparation."""

from __future__ import annotations

import json
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .canon import CanonRegistry
from .evidence_store import file_sha256
from .registry import load_json


class SourceImportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PreparedSource:
    source_id: str
    revision: str
    artifact_path: Path
    upstream_sha256: str
    artifact_sha256: str
    passage_count: int
    book_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "revision": self.revision,
            "artifact_path": str(self.artifact_path),
            "upstream_sha256": self.upstream_sha256,
            "artifact_sha256": self.artifact_sha256,
            "passage_count": self.passage_count,
            "book_count": self.book_count,
        }


_USFM_BOOKS = {
    "GEN": "Genesis",
    "EXO": "Exodus",
    "LEV": "Leviticus",
    "NUM": "Numbers",
    "DEU": "Deuteronomy",
    "JOS": "Joshua",
    "JDG": "Judges",
    "RUT": "Ruth",
    "1SA": "1 Samuel",
    "2SA": "2 Samuel",
    "1KI": "1 Kings",
    "2KI": "2 Kings",
    "1CH": "1 Chronicles",
    "2CH": "2 Chronicles",
    "EZR": "Ezra",
    "NEH": "Nehemiah",
    "EST": "Esther",
    "JOB": "Job",
    "PSA": "Psalms",
    "PRO": "Proverbs",
    "ECC": "Ecclesiastes",
    "SNG": "Song of Solomon",
    "ISA": "Isaiah",
    "JER": "Jeremiah",
    "LAM": "Lamentations",
    "EZK": "Ezekiel",
    "DAN": "Daniel",
    "HOS": "Hosea",
    "JOL": "Joel",
    "AMO": "Amos",
    "OBA": "Obadiah",
    "JON": "Jonah",
    "MIC": "Micah",
    "NAM": "Nahum",
    "HAB": "Habakkuk",
    "ZEP": "Zephaniah",
    "HAG": "Haggai",
    "ZEC": "Zechariah",
    "MAL": "Malachi",
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "1 Corinthians",
    "2CO": "2 Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians",
    "1TI": "1 Timothy",
    "2TI": "2 Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "1 Peter",
    "2PE": "2 Peter",
    "1JN": "1 John",
    "2JN": "2 John",
    "3JN": "3 John",
    "JUD": "Jude",
    "REV": "Revelation",
}

_COMPACT_BOOKS = {
    "Matt": "Matthew",
    "Mark": "Mark",
    "Luke": "Luke",
    "John": "John",
    "Acts": "Acts",
    "Rom": "Romans",
    "1Cor": "1 Corinthians",
    "2Cor": "2 Corinthians",
    "Gal": "Galatians",
    "Eph": "Ephesians",
    "Phil": "Philippians",
    "Col": "Colossians",
    "1Thess": "1 Thessalonians",
    "2Thess": "2 Thessalonians",
    "1Tim": "1 Timothy",
    "2Tim": "2 Timothy",
    "Titus": "Titus",
    "Phlm": "Philemon",
    "Heb": "Hebrews",
    "Jas": "James",
    "1Pet": "1 Peter",
    "2Pet": "2 Peter",
    "1John": "1 John",
    "2John": "2 John",
    "3John": "3 John",
    "Jude": "Jude",
    "Rev": "Revelation",
}

_MORPH_BOOKS = {
    "Mt": "Matthew",
    "Mk": "Mark",
    "Lk": "Luke",
    "Jn": "John",
    "Ac": "Acts",
    "Ro": "Romans",
    "1Co": "1 Corinthians",
    "2Co": "2 Corinthians",
    "Ga": "Galatians",
    "Eph": "Ephesians",
    "Php": "Philippians",
    "Col": "Colossians",
    "1Th": "1 Thessalonians",
    "2Th": "2 Thessalonians",
    "1Ti": "1 Timothy",
    "2Ti": "2 Timothy",
    "Tit": "Titus",
    "Phm": "Philemon",
    "Heb": "Hebrews",
    "Jas": "James",
    "1Pe": "1 Peter",
    "2Pe": "2 Peter",
    "1Jn": "1 John",
    "2Jn": "2 John",
    "3Jn": "3 John",
    "Jud": "Jude",
    "Re": "Revelation",
}


def _download(url: str, destination: Path) -> None:
    if not url.startswith("https://"):
        raise SourceImportError("source downloads must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "biblical-moral-ai-source-lock/1"})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            final_url = response.geturl()
            if not final_url.startswith("https://"):
                raise SourceImportError("source download redirected away from HTTPS")
            with temporary.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        temporary.replace(destination)
    except OSError as exc:
        raise SourceImportError(f"source download failed: {url}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verified_file(spec: dict[str, Any], cache_dir: Path, *, fetch: bool) -> Path:
    expected = str(spec.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SourceImportError("every locked download requires a SHA-256 digest")
    filename = str(spec.get("filename", "")).strip()
    if not filename or Path(filename).name != filename:
        raise SourceImportError("locked download filename must be a plain filename")
    path = cache_dir / filename
    if not path.is_file():
        if not fetch:
            raise SourceImportError(f"locked download is not cached: {path}")
        _download(str(spec["url"]), path)
    actual = file_sha256(path)
    if actual != expected:
        raise SourceImportError(
            f"locked download digest mismatch for {filename}: expected {expected}, got {actual}"
        )
    return path


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    max_files = 20_000
    max_uncompressed = 768 * 1024 * 1024
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > max_files:
            raise SourceImportError("source archive contains too many files")
        if sum(item.file_size for item in members) > max_uncompressed:
            raise SourceImportError("source archive exceeds the uncompressed size limit")
        for item in members:
            member = PurePosixPath(item.filename)
            if (
                member.is_absolute()
                or ".." in member.parts
                or "\\" in item.filename
                or ":" in item.filename
            ):
                raise SourceImportError(f"unsafe source archive member: {item.filename}")
        bundle.extractall(destination)


def _clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _strip_usfm(value: str) -> str:
    value = re.sub(r"\\(?:f|x)\s.*?\\(?:f|x)\*", " ", value)
    value = re.sub(r"\\w\s+([^|\\]+?)(?:\|[^\\]*?)?\\w\*", r"\1", value)
    value = re.sub(r"\\add\s*", "", value)
    value = value.replace("\\add*", "")
    value = re.sub(r"\\[A-Za-z0-9-]+\*?(?:\s+)?", " ", value)
    value = re.sub(r"\|[A-Za-z0-9_-]+=(?:\"[^\"]*\"|'[^']*')", "", value)
    return _clean_space(value)


def _passage(
    source_id: str,
    book: str,
    chapter: int,
    verse: int,
    language: str,
    text: str,
    context: str,
) -> dict[str, Any]:
    return {
        "reference": f"{book} {chapter}:{verse}",
        "book": book,
        "chapter": chapter,
        "verse_start": verse,
        "verse_end": verse,
        "language": language,
        "text": text,
        "context": context,
    }


def _parse_usfm(root: Path, source_id: str) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.usfm")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        id_line = next((line for line in lines if line.startswith("\\id ")), "")
        match = re.match(r"\\id\s+([A-Z0-9]{3})\b", id_line)
        if match is None or match.group(1) not in _USFM_BOOKS:
            continue
        book = _USFM_BOOKS[match.group(1)]
        chapter = 0
        current_verse: int | None = None
        buffer: list[str] = []

        def flush(book_name: str, chapter_number: int) -> None:
            nonlocal buffer, current_verse
            if current_verse is None:
                return
            text = _strip_usfm(" ".join(buffer))
            if not text:
                raise SourceImportError(
                    f"empty USFM verse: {book_name} {chapter_number}:{current_verse}"
                )
            passages.append(
                _passage(
                    source_id,
                    book_name,
                    chapter_number,
                    current_verse,
                    "English",
                    text,
                    "KJV standardized 1769 text; Strong's markup removed deterministically.",
                )
            )
            buffer = []
            current_verse = None

        for line in lines:
            chapter_match = re.match(r"\\c\s+(\d+)", line)
            if chapter_match:
                flush(book, chapter)
                chapter = int(chapter_match.group(1))
                continue
            verse_match = re.match(r"\\v\s+(\d+)(?:-\d+)?\s*(.*)", line)
            if verse_match:
                flush(book, chapter)
                if chapter < 1:
                    raise SourceImportError(f"USFM verse appears before chapter in {path}")
                current_verse = int(verse_match.group(1))
                buffer = [verse_match.group(2)]
                continue
            if current_verse is not None and re.match(r"\\(?:q\d*|m|mi|p|pi\d*)\b", line):
                buffer.append(line)
        flush(book, chapter)
    return passages


def _parse_osis(root: Path, source_id: str) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.xml")):
        if path.stem == "VerseMap" or "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise SourceImportError(f"invalid OSIS XML: {path}: {exc}") from exc
        for verse in tree.findall(".//{*}verse"):
            osis_id = verse.get("osisID", "")
            match = re.fullmatch(r"([A-Za-z0-9]+)\.(\d+)\.(\d+)", osis_id)
            if match is None:
                continue
            source_book, chapter, verse_number = match.groups()
            book = {
                "Ps": "Psalms",
                "1Sam": "1 Samuel",
                "2Sam": "2 Samuel",
                "1Kgs": "1 Kings",
                "2Kgs": "2 Kings",
                "1Chr": "1 Chronicles",
                "2Chr": "2 Chronicles",
            }.get(source_book, source_book)
            text = _clean_space("".join(verse.itertext()))
            if not text:
                raise SourceImportError(f"empty OSIS verse: {osis_id}")
            passages.append(
                _passage(
                    source_id,
                    book,
                    int(chapter),
                    int(verse_number),
                    "Hebrew/Aramaic",
                    text,
                    "OSHB v2.2 morphology-preserving OSIS text; Unicode is not normalized.",
                )
            )
    return passages


def _parse_sblgnt(root: Path, source_id: str) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    candidates = [
        path
        for path in root.rglob("*.txt")
        if "/data/sblgnt/text/" in path.as_posix()
    ]
    for path in sorted(candidates):
        for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
            if not line.strip():
                continue
            try:
                reference, text = line.split("\t", 1)
            except ValueError as exc:
                raise SourceImportError(f"invalid SBLGNT text row in {path}: {line}") from exc
            match = re.fullmatch(r"(.+?)\s+(\d+):(\d+)", reference.strip())
            if match is None or match.group(1) not in _COMPACT_BOOKS:
                raise SourceImportError(f"invalid SBLGNT reference: {reference}")
            book = _COMPACT_BOOKS[match.group(1)]
            passages.append(
                _passage(
                    source_id,
                    book,
                    int(match.group(2)),
                    int(match.group(3)),
                    "Greek",
                    _clean_space(text),
                    "Faithlife SBLGNT v1.2 text at the pinned repository revision.",
                )
            )
    return passages


def _parse_morphgnt(root: Path, source_id: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*-morphgnt.txt")):
        match = re.fullmatch(r"\d{2}-(.+)-morphgnt", path.stem)
        if match is None or match.group(1) not in _MORPH_BOOKS:
            raise SourceImportError(f"unknown MorphGNT book filename: {path.name}")
        book = _MORPH_BOOKS[match.group(1)]
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if len(parts) != 7 or not re.fullmatch(r"\d{6}", parts[0]):
                raise SourceImportError(f"invalid MorphGNT row in {path}: {line}")
            chapter = int(parts[0][2:4])
            verse = int(parts[0][4:6])
            grouped[(book, chapter, verse)].append(parts[3])
    passages: list[dict[str, Any]] = []
    for (book, chapter, verse), tokens in grouped.items():
        text = " ".join(tokens)
        text = re.sub(r"\s+([,.;··:!?])", r"\1", text)
        passages.append(
            _passage(
                source_id,
                book,
                chapter,
                verse,
                "Greek",
                text,
                "MorphGNT 6.12 surface text with morphology available in the locked upstream.",
            )
        )
    return passages


_PARSERS: dict[str, Callable[[Path, str], list[dict[str, Any]]]] = {
    "usfm_zip": _parse_usfm,
    "osis_zip": _parse_osis,
    "sblgnt_text_zip": _parse_sblgnt,
    "morphgnt_zip": _parse_morphgnt,
}


def _sort_and_validate(
    passages: list[dict[str, Any]], canon: CanonRegistry, package: dict[str, Any]
) -> list[dict[str, Any]]:
    order = {book.name: book.order for book in canon.books}
    seen: set[str] = set()
    books: set[str] = set()
    for item in passages:
        book = canon.normalize_book(str(item["book"]))
        item["book"] = book
        item["reference"] = f"{book} {item['chapter']}:{item['verse_start']}"
        if item["reference"] in seen:
            raise SourceImportError(f"duplicate source reference: {item['reference']}")
        seen.add(item["reference"])
        books.add(book)
    passages.sort(
        key=lambda item: (order[item["book"]], item["chapter"], item["verse_start"])
    )
    expected_books = int(package["expected_book_count"])
    expected_passages = int(package["expected_passage_count"])
    if len(books) != expected_books or len(passages) != expected_passages:
        raise SourceImportError(
            f"source cardinality mismatch: books={len(books)}/{expected_books}, "
            f"passages={len(passages)}/{expected_passages}"
        )
    return passages


def _write_canonical(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The approved artifact digests were established with CRLF line endings.
    # Write bytes explicitly so Linux training hosts and Windows review hosts
    # produce the same canonical artifact instead of inheriting platform text
    # newline translation.
    serialized = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\r\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(path)
    return file_sha256(path)


def prepare_source(
    source_id: str,
    *,
    lock_path: str | Path,
    canon_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    fetch: bool = False,
) -> PreparedSource:
    lock = load_json(lock_path)
    packages = [item for item in lock.get("packages", []) if item.get("source_id") == source_id]
    if len(packages) != 1:
        raise SourceImportError(f"source lock must contain exactly one package for {source_id}")
    package = packages[0]
    parser_name = str(package.get("parser", ""))
    if parser_name not in _PARSERS:
        raise SourceImportError(f"unsupported source parser: {parser_name}")
    cache = Path(cache_dir)
    archive = _verified_file(package["download"], cache, fetch=fetch)
    _verified_file(package["license_evidence"], cache, fetch=fetch)
    with tempfile.TemporaryDirectory(prefix=f"{source_id}-") as directory:
        extracted = Path(directory)
        _safe_extract_zip(archive, extracted)
        passages = _PARSERS[parser_name](extracted, source_id)
    canon = CanonRegistry.load(canon_path)
    passages = _sort_and_validate(passages, canon, package)
    artifact_path = Path(output_dir) / f"{source_id}.json"
    artifact_hash = _write_canonical(
        artifact_path,
        {
            "schema_version": "1.0",
            "source_id": source_id,
            "revision": package["revision"],
            "passages": passages,
        },
    )
    expected_artifact = str(package.get("canonical_artifact_sha256") or "").lower()
    if expected_artifact and artifact_hash != expected_artifact:
        artifact_path.unlink(missing_ok=True)
        raise SourceImportError(
            f"canonical artifact digest mismatch for {source_id}: "
            f"expected {expected_artifact}, got {artifact_hash}"
        )
    return PreparedSource(
        source_id=source_id,
        revision=str(package["revision"]),
        artifact_path=artifact_path,
        upstream_sha256=file_sha256(archive),
        artifact_sha256=artifact_hash,
        passage_count=len(passages),
        book_count=len({item["book"] for item in passages}),
    )


def verify_source_approval(
    prepared: PreparedSource, *, registry_path: str | Path, lock_path: str | Path
) -> None:
    registry = load_json(registry_path)
    matches = [
        item for item in registry.get("sources", []) if item.get("source_id") == prepared.source_id
    ]
    if len(matches) != 1:
        raise SourceImportError(f"source registry entry missing or duplicated: {prepared.source_id}")
    entry = matches[0]
    if entry.get("status") != "approved":
        raise SourceImportError(f"source is not approved: {prepared.source_id}")
    if entry.get("revision") != prepared.revision:
        raise SourceImportError(f"approved revision mismatch: {prepared.source_id}")
    if str(entry.get("sha256", "")).lower() != prepared.artifact_sha256:
        raise SourceImportError(f"approved canonical digest mismatch: {prepared.source_id}")
    lock = load_json(lock_path)
    package = next(
        item for item in lock.get("packages", []) if item.get("source_id") == prepared.source_id
    )
    if package["download"]["sha256"] != prepared.upstream_sha256:
        raise SourceImportError(f"approved upstream digest mismatch: {prepared.source_id}")
    approval = entry.get("approval", {})
    if not approval.get("decision_id") or not approval.get("approved_by"):
        raise SourceImportError(f"source approval decision is incomplete: {prepared.source_id}")
