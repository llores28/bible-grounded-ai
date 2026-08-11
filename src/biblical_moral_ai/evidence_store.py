"""Approved-source passage retrieval and reviewed canonical graph storage."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .canon import CanonRegistry
from .registry import load_json
from .schemas import Confidence, EvidenceClass, ReviewStatus


class EvidenceStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source_id: str
    title: str
    role: str
    revision: str
    sha256: str
    status: str = "approved"
    organizational: bool = False


@dataclass(frozen=True, slots=True)
class Passage:
    source_id: str
    reference: str
    book: str
    chapter: int
    verse_start: int
    verse_end: int
    language: str
    text: str
    context: str = ""


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    source_id: str
    entry_id: str
    language: str
    lemma: str
    transliteration: str
    gloss: str
    definition: str
    source_ref: str


class EdgeType(str):
    ALLOWED = {
        "explicit_quotation",
        "formula_introduction",
        "fulfillment_claim",
        "allusion",
        "type_antitype",
        "symbol_defined_by_text",
        "shared_duration",
        "lexical_parallel",
        "thematic_parallel",
        "proposed_historical_fulfillment",
    }


@dataclass(frozen=True, slots=True)
class CanonicalEdge:
    edge_id: str
    from_source_id: str
    from_reference: str
    to_source_id: str
    to_reference: str
    edge_type: str
    evidence_class: EvidenceClass
    rationale: str
    reviewer_status: ReviewStatus
    confidence: Confidence
    reviewer_ids: tuple[str, ...]
    adjudication_id: str | None = None

    def __post_init__(self) -> None:
        if self.edge_type not in EdgeType.ALLOWED:
            raise EvidenceStoreError(f"unsupported canonical edge type: {self.edge_type}")
        if self.reviewer_status not in {ReviewStatus.APPROVED, ReviewStatus.DISPUTED}:
            raise EvidenceStoreError("canonical edges must be approved or explicitly disputed")
        if not self.rationale.strip():
            raise EvidenceStoreError("canonical edge rationale is required")
        if len(set(self.reviewer_ids)) != len(self.reviewer_ids) or not self.reviewer_ids:
            raise EvidenceStoreError("canonical edges require unique reviewer IDs")
        if (
            self.edge_type in {"fulfillment_claim", "proposed_historical_fulfillment"}
            and len(self.reviewer_ids) < 2
        ):
            raise EvidenceStoreError("fulfillment edges require two independent reviewers")
        if self.reviewer_status is ReviewStatus.DISPUTED and not self.adjudication_id:
            raise EvidenceStoreError("disputed canonical edges require an adjudication ID")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceStore:
    """SQLite store that excludes organizational material from biblical retrieval by default."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                role TEXT NOT NULL,
                revision TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status = 'approved'),
                organizational INTEGER NOT NULL CHECK (organizational IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS passages (
                passage_id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                reference TEXT NOT NULL,
                book TEXT NOT NULL,
                chapter INTEGER NOT NULL CHECK (chapter > 0),
                verse_start INTEGER NOT NULL CHECK (verse_start > 0),
                verse_end INTEGER NOT NULL CHECK (verse_end >= verse_start),
                language TEXT NOT NULL,
                text TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                UNIQUE(source_id, reference)
            );

            CREATE TABLE IF NOT EXISTS lexicon_entries (
                lexicon_row_id INTEGER PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                entry_id TEXT NOT NULL,
                language TEXT NOT NULL,
                lemma TEXT NOT NULL,
                transliteration TEXT NOT NULL,
                gloss TEXT NOT NULL,
                definition TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                UNIQUE(source_id, entry_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS lexicon_search USING fts5(
                lemma,
                transliteration,
                gloss,
                definition,
                content='lexicon_entries',
                content_rowid='lexicon_row_id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS lexicon_after_insert AFTER INSERT ON lexicon_entries BEGIN
                INSERT INTO lexicon_search(rowid, lemma, transliteration, gloss, definition)
                VALUES (new.lexicon_row_id, new.lemma, new.transliteration, new.gloss, new.definition);
            END;
            CREATE TRIGGER IF NOT EXISTS lexicon_after_delete AFTER DELETE ON lexicon_entries BEGIN
                INSERT INTO lexicon_search(lexicon_search, rowid, lemma, transliteration, gloss, definition)
                VALUES ('delete', old.lexicon_row_id, old.lemma, old.transliteration, old.gloss, old.definition);
            END;

            CREATE VIRTUAL TABLE IF NOT EXISTS passage_search USING fts5(
                text,
                context,
                content='passages',
                content_rowid='passage_id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS passages_after_insert AFTER INSERT ON passages BEGIN
                INSERT INTO passage_search(rowid, text, context) VALUES (new.passage_id, new.text, new.context);
            END;
            CREATE TRIGGER IF NOT EXISTS passages_after_delete AFTER DELETE ON passages BEGIN
                INSERT INTO passage_search(passage_search, rowid, text, context)
                VALUES ('delete', old.passage_id, old.text, old.context);
            END;
            CREATE TRIGGER IF NOT EXISTS passages_after_update AFTER UPDATE ON passages BEGIN
                INSERT INTO passage_search(passage_search, rowid, text, context)
                VALUES ('delete', old.passage_id, old.text, old.context);
                INSERT INTO passage_search(rowid, text, context) VALUES (new.passage_id, new.text, new.context);
            END;

            CREATE TABLE IF NOT EXISTS canonical_edges (
                edge_id TEXT PRIMARY KEY,
                from_source_id TEXT NOT NULL,
                from_reference TEXT NOT NULL,
                to_source_id TEXT NOT NULL,
                to_reference TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                evidence_class TEXT NOT NULL,
                rationale TEXT NOT NULL,
                reviewer_status TEXT NOT NULL,
                confidence TEXT NOT NULL,
                reviewer_ids TEXT NOT NULL,
                adjudication_id TEXT,
                FOREIGN KEY(from_source_id, from_reference) REFERENCES passages(source_id, reference),
                FOREIGN KEY(to_source_id, to_reference) REFERENCES passages(source_id, reference)
            );
            """
        )
        self.connection.commit()

    def add_source(self, metadata: SourceMetadata, passages: list[Passage]) -> None:
        if metadata.status != "approved":
            raise EvidenceStoreError("only approved sources may be ingested")
        if not metadata.revision or not re.fullmatch(r"[0-9a-fA-F]{64}", metadata.sha256):
            raise EvidenceStoreError("approved sources require a revision and 64-character SHA-256")
        if not passages:
            raise EvidenceStoreError("a source artifact must contain passages")
        if any(item.source_id != metadata.source_id for item in passages):
            raise EvidenceStoreError("every passage source_id must match its source metadata")

        with self.connection:
            self.connection.execute(
                "INSERT INTO sources(source_id, title, role, revision, sha256, status, organizational) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    metadata.source_id,
                    metadata.title,
                    metadata.role,
                    metadata.revision,
                    metadata.sha256.lower(),
                    metadata.status,
                    int(metadata.organizational),
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO passages(source_id, reference, book, chapter, verse_start, verse_end, language, text, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.source_id,
                        item.reference,
                        item.book,
                        item.chapter,
                        item.verse_start,
                        item.verse_end,
                        item.language,
                        item.text,
                        item.context,
                    )
                    for item in passages
                ],
            )

    def import_artifact(
        self,
        artifact_path: str | Path,
        source_registry_path: str | Path,
        canon_path: str | Path,
    ) -> None:
        path = Path(artifact_path)
        registry = load_json(source_registry_path)
        payload = load_json(path)
        source_id = payload.get("source_id")
        matches = [
            item for item in registry.get("sources", []) if item.get("source_id") == source_id
        ]
        if len(matches) != 1:
            raise EvidenceStoreError(
                f"source registry must contain exactly one entry for {source_id}"
            )
        entry = matches[0]
        actual_hash = file_sha256(path)
        if entry.get("status") != "approved":
            raise EvidenceStoreError(f"source is not approved: {source_id}")
        if entry.get("sha256", "").lower() != actual_hash:
            raise EvidenceStoreError(f"artifact digest does not match source registry: {source_id}")
        if entry.get("revision") != payload.get("revision"):
            raise EvidenceStoreError(
                f"artifact revision does not match source registry: {source_id}"
            )

        metadata = SourceMetadata(
            source_id=source_id,
            title=str(entry["title"]),
            role=str(entry["role"]),
            revision=str(entry["revision"]),
            sha256=actual_hash,
            organizational=bool(entry.get("organizational", False)),
        )
        canon = CanonRegistry.load(canon_path)
        passages: list[Passage] = []
        for item in payload.get("passages", []):
            reference, book, chapter, verse_start, verse_end = canon.normalize_reference(
                str(item["reference"])
            )
            declared = (
                canon.normalize_book(str(item["book"])),
                int(item["chapter"]),
                int(item["verse_start"]),
                int(item.get("verse_end", item["verse_start"])),
            )
            if declared != (book, chapter, verse_start, verse_end):
                raise EvidenceStoreError(
                    f"passage metadata does not match normalized reference: {item['reference']}"
                )
            passages.append(
                Passage(
                    source_id=source_id,
                    reference=reference,
                    book=book,
                    chapter=chapter,
                    verse_start=verse_start,
                    verse_end=verse_end,
                    language=str(item["language"]),
                    text=str(item["text"]),
                    context=str(item.get("context", "")),
                )
            )
        self.add_source(metadata, passages)

    def import_lexicon_artifact(
        self,
        artifact_path: str | Path,
        source_registry_path: str | Path,
    ) -> None:
        path = Path(artifact_path)
        registry = load_json(source_registry_path)
        payload = load_json(path)
        source_id = str(payload.get("source_id", ""))
        matches = [
            item for item in registry.get("sources", []) if item.get("source_id") == source_id
        ]
        if len(matches) != 1:
            raise EvidenceStoreError(
                f"source registry must contain exactly one entry for {source_id}"
            )
        source = matches[0]
        actual_hash = file_sha256(path)
        if source.get("status") != "approved":
            raise EvidenceStoreError(f"source is not approved: {source_id}")
        if str(source.get("sha256", "")).lower() != actual_hash:
            raise EvidenceStoreError(f"artifact digest does not match source registry: {source_id}")
        if source.get("revision") != payload.get("revision"):
            raise EvidenceStoreError(
                f"artifact revision does not match source registry: {source_id}"
            )
        entries = [
            LexiconEntry(
                source_id=source_id,
                entry_id=str(item["entry_id"]),
                language=str(item["language"]),
                lemma=str(item["lemma"]),
                transliteration=str(item.get("transliteration", "")),
                gloss=str(item.get("gloss", "")),
                definition=str(item["definition"]),
                source_ref=str(item["source_ref"]),
            )
            for item in payload.get("entries", [])
        ]
        if not entries:
            raise EvidenceStoreError("a lexicon artifact must contain entries")
        metadata = SourceMetadata(
            source_id=source_id,
            title=str(source["title"]),
            role=str(source["role"]),
            revision=str(source["revision"]),
            sha256=actual_hash,
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO sources(source_id, title, role, revision, sha256, status, organizational) VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    metadata.source_id,
                    metadata.title,
                    metadata.role,
                    metadata.revision,
                    metadata.sha256,
                    metadata.status,
                ),
            )
            self.connection.executemany(
                """
                INSERT INTO lexicon_entries(source_id, entry_id, language, lemma, transliteration, gloss, definition, source_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.source_id,
                        item.entry_id,
                        item.language,
                        item.lemma,
                        item.transliteration,
                        item.gloss,
                        item.definition,
                        item.source_ref,
                    )
                    for item in entries
                ],
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 12,
        include_organizational: bool = False,
    ) -> list[Passage]:
        if not 1 <= limit <= 100:
            raise EvidenceStoreError("search limit must be between 1 and 100")
        terms = list(dict.fromkeys(re.findall(r"[\w]+", query.casefold(), flags=re.UNICODE)))
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:30])
        rows = self.connection.execute(
            """
            SELECT p.*
            FROM passage_search
            JOIN passages p ON p.passage_id = passage_search.rowid
            JOIN sources s ON s.source_id = p.source_id
            WHERE passage_search MATCH ? AND (? OR s.organizational = 0)
            ORDER BY bm25(passage_search), p.source_id, p.book, p.chapter, p.verse_start
            LIMIT ?
            """,
            (fts_query, int(include_organizational), limit),
        ).fetchall()
        return [self._row_to_passage(row) for row in rows]

    def get_passage(self, source_id: str, reference: str) -> Passage | None:
        row = self.connection.execute(
            "SELECT * FROM passages WHERE source_id = ? AND reference = ?",
            (source_id, reference),
        ).fetchone()
        return self._row_to_passage(row) if row else None

    def search_lexicon(
        self,
        query: str,
        *,
        language: str | None = None,
        limit: int = 12,
    ) -> list[LexiconEntry]:
        if not 1 <= limit <= 100:
            raise EvidenceStoreError("search limit must be between 1 and 100")
        terms = list(dict.fromkeys(re.findall(r"[\w]+", query.casefold(), flags=re.UNICODE)))
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:30])
        rows = self.connection.execute(
            """
            SELECT e.*
            FROM lexicon_search
            JOIN lexicon_entries e ON e.lexicon_row_id = lexicon_search.rowid
            WHERE lexicon_search MATCH ? AND (? IS NULL OR e.language = ?)
            ORDER BY bm25(lexicon_search), e.source_id, e.entry_id
            LIMIT ?
            """,
            (fts_query, language, language, limit),
        ).fetchall()
        return [self._row_to_lexicon_entry(row) for row in rows]

    def export_corpora(self, passages: list[Passage]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for passage in passages:
            result.setdefault(passage.source_id, {})[passage.reference] = passage.text
        return result

    def citation_corpora(self) -> dict[str, dict[str, str]]:
        """Export Scripture passages only; lexicon definitions are never citation corpora."""
        rows = self.connection.execute(
            """
            SELECT p.* FROM passages p
            JOIN sources s ON s.source_id = p.source_id
            WHERE s.organizational = 0
            ORDER BY p.source_id, p.book, p.chapter, p.verse_start
            """
        ).fetchall()
        return self.export_corpora([self._row_to_passage(row) for row in rows])

    def source_inventory(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT s.source_id, s.role, s.revision, s.sha256,
                   COUNT(DISTINCT p.passage_id) AS passage_count,
                   COUNT(DISTINCT l.lexicon_row_id) AS lexicon_entry_count
            FROM sources s
            LEFT JOIN passages p ON p.source_id = s.source_id
            LEFT JOIN lexicon_entries l ON l.source_id = s.source_id
            GROUP BY s.source_id, s.role, s.revision, s.sha256
            ORDER BY s.source_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def approved_source_ids(self, *, include_organizational: bool = False) -> set[str]:
        rows = self.connection.execute(
            "SELECT source_id FROM sources WHERE ? OR organizational = 0",
            (int(include_organizational),),
        ).fetchall()
        return {str(row["source_id"]) for row in rows}

    def organizational_source_ids(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT source_id FROM sources WHERE organizational = 1"
        ).fetchall()
        return {str(row["source_id"]) for row in rows}

    def add_edge(self, edge: CanonicalEdge) -> None:
        for source_id, reference in (
            (edge.from_source_id, edge.from_reference),
            (edge.to_source_id, edge.to_reference),
        ):
            row = self.connection.execute(
                """
                SELECT s.organizational
                FROM passages p JOIN sources s ON s.source_id = p.source_id
                WHERE p.source_id = ? AND p.reference = ?
                """,
                (source_id, reference),
            ).fetchone()
            if row is None:
                raise EvidenceStoreError(
                    f"canonical edge endpoint does not exist: {source_id} {reference}"
                )
            if row["organizational"]:
                raise EvidenceStoreError(
                    "organizational documents cannot be canonical graph endpoints"
                )
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO canonical_edges(edge_id, from_source_id, from_reference, to_source_id, to_reference, edge_type, evidence_class, rationale, reviewer_status, confidence, reviewer_ids, adjudication_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.edge_id,
                    edge.from_source_id,
                    edge.from_reference,
                    edge.to_source_id,
                    edge.to_reference,
                    edge.edge_type,
                    edge.evidence_class.value,
                    edge.rationale,
                    edge.reviewer_status.value,
                    edge.confidence.value,
                    json.dumps(edge.reviewer_ids),
                    edge.adjudication_id,
                ),
            )

    def neighbors(self, source_id: str, reference: str) -> list[CanonicalEdge]:
        rows = self.connection.execute(
            """
            SELECT * FROM canonical_edges
            WHERE (from_source_id = ? AND from_reference = ?)
               OR (to_source_id = ? AND to_reference = ?)
            ORDER BY edge_id
            """,
            (source_id, reference, source_id, reference),
        ).fetchall()
        return [
            CanonicalEdge(
                edge_id=row["edge_id"],
                from_source_id=row["from_source_id"],
                from_reference=row["from_reference"],
                to_source_id=row["to_source_id"],
                to_reference=row["to_reference"],
                edge_type=row["edge_type"],
                evidence_class=EvidenceClass(row["evidence_class"]),
                rationale=row["rationale"],
                reviewer_status=ReviewStatus(row["reviewer_status"]),
                confidence=Confidence(row["confidence"]),
                reviewer_ids=tuple(json.loads(row["reviewer_ids"])),
                adjudication_id=row["adjudication_id"],
            )
            for row in rows
        ]

    def expand_with_neighbors(
        self,
        passages: list[Passage],
        *,
        additional_limit: int = 8,
    ) -> list[Passage]:
        if not 0 <= additional_limit <= 100:
            raise EvidenceStoreError("graph expansion limit must be between 0 and 100")
        expanded = list(passages)
        seen = {(item.source_id, item.reference) for item in passages}
        for passage in passages:
            for edge in self.neighbors(passage.source_id, passage.reference):
                endpoints = (
                    (edge.from_source_id, edge.from_reference),
                    (edge.to_source_id, edge.to_reference),
                )
                for endpoint in endpoints:
                    if endpoint in seen:
                        continue
                    related = self.get_passage(*endpoint)
                    if related is not None:
                        expanded.append(related)
                        seen.add(endpoint)
                    if len(expanded) - len(passages) >= additional_limit:
                        return expanded
        return expanded

    @staticmethod
    def _row_to_passage(row: sqlite3.Row) -> Passage:
        return Passage(
            source_id=row["source_id"],
            reference=row["reference"],
            book=row["book"],
            chapter=row["chapter"],
            verse_start=row["verse_start"],
            verse_end=row["verse_end"],
            language=row["language"],
            text=row["text"],
            context=row["context"],
        )

    @staticmethod
    def _row_to_lexicon_entry(row: sqlite3.Row) -> LexiconEntry:
        return LexiconEntry(
            source_id=row["source_id"],
            entry_id=row["entry_id"],
            language=row["language"],
            lemma=row["lemma"],
            transliteration=row["transliteration"],
            gloss=row["gloss"],
            definition=row["definition"],
            source_ref=row["source_ref"],
        )
