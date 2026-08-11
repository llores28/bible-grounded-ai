"""Machine-enforced 66-book Protestant canon registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .registry import RegistryError, load_json


@dataclass(frozen=True, slots=True)
class CanonBook:
    order: int
    name: str
    testament: str
    aliases: tuple[str, ...]


class CanonRegistry:
    def __init__(self, books: tuple[CanonBook, ...]) -> None:
        if len(books) != 66 or [book.order for book in books] != list(range(1, 67)):
            raise RegistryError(
                "canon registry must contain exactly 66 consecutively ordered books"
            )
        if sum(book.testament == "OT" for book in books) != 39:
            raise RegistryError("canon registry must contain 39 Old Testament books")
        if sum(book.testament == "NT" for book in books) != 27:
            raise RegistryError("canon registry must contain 27 New Testament books")
        names: dict[str, str] = {}
        for book in books:
            if book.testament not in {"OT", "NT"} or not book.name.strip():
                raise RegistryError(f"invalid canon book record: {book}")
            for candidate in (book.name, *book.aliases):
                key = candidate.casefold()
                if key in names:
                    raise RegistryError(f"duplicate canon name or alias: {candidate}")
                names[key] = book.name
        self.books = books
        self._names = names

    @classmethod
    def load(cls, path: str | Path) -> CanonRegistry:
        payload = load_json(path)
        if payload.get("schema_version") != "1.0":
            raise RegistryError("unsupported canon registry schema_version")
        books = tuple(
            CanonBook(
                order=int(item["order"]),
                name=str(item["name"]),
                testament=str(item["testament"]),
                aliases=tuple(item.get("aliases", [])),
            )
            for item in payload.get("books", [])
        )
        return cls(books)

    def normalize_book(self, value: str) -> str:
        try:
            return self._names[value.strip().casefold()]
        except KeyError as exc:
            raise RegistryError(f"book is outside the configured 66-book canon: {value}") from exc

    def normalize_reference(self, value: str) -> tuple[str, str, int, int, int]:
        match = re.fullmatch(r"(.+?)\s+(\d+):(\d+)(?:-(\d+))?", value.strip())
        if match is None:
            raise RegistryError(
                f"reference must use a single-chapter Book chapter:verse[-verse] form: {value}"
            )
        book = self.normalize_book(match.group(1))
        chapter = int(match.group(2))
        verse_start = int(match.group(3))
        verse_end = int(match.group(4) or verse_start)
        if chapter < 1 or verse_start < 1 or verse_end < verse_start:
            raise RegistryError(f"invalid chapter or verse range: {value}")
        reference = f"{book} {chapter}:{verse_start}"
        if verse_end != verse_start:
            reference += f"-{verse_end}"
        return reference, book, chapter, verse_start, verse_end
