"""Read nro's BibTeX bibliography and format author-year citations.

The reader accepts the controlled subset of BibTeX used in ``references.bib``:
``@type{key, field = {value}, ...}`` entries with braced, quoted, or bare values.
It is not a general BibTeX implementation. Unsupported syntax raises
``BibliographyError`` so a malformed entry cannot silently change a citation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from nro.methods.document import Run

BIBLIOGRAPHY_PATH = Path(__file__).with_name("references.bib")

_ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,")
_FIELD_NAME = re.compile(r"\s*([A-Za-z][\w-]*)\s*=\s*")
_LATEX_REPLACEMENTS = (
    ("---", "—"),
    ("--", "–"),
    ("\\&", "&"),
    ("\\%", "%"),
    ("\\_", "_"),
    ("~", " "),
)


_ACCENT = re.compile(r"\\([\"'`^~=.])\{?([A-Za-z])\}?|\\([cvu])\{([A-Za-z])\}")
_COMBINING = {
    '"': "\u0308",
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "c": "\u0327",
    "v": "\u030c",
    "u": "\u0306",
}


def _accent(match: re.Match[str]) -> str:
    mark = match.group(1) or match.group(3)
    letter = match.group(2) or match.group(4)
    return unicodedata.normalize("NFC", letter + _COMBINING[mark])


class BibliographyError(ValueError):
    """Raised when the bibliography is malformed or a citation key is unknown."""


@dataclass(frozen=True)
class Person:
    """One author name split into family and given parts."""

    family: str
    given: str
    institutional: bool = False

    def initials(self) -> str:
        """Return APA-style initials such as ``T. E.`` for the given names."""
        parts = re.split(r"[\s]+", self.given.strip())
        result = []
        for part in parts:
            if not part:
                continue
            hyphenated = [piece for piece in part.split("-") if piece]
            result.append("-".join(f"{piece[0]}." for piece in hyphenated))
        return " ".join(result)


@dataclass(frozen=True)
class Entry:
    """One parsed bibliography entry with raw and display-ready field values."""

    key: str
    entry_type: str
    fields: Mapping[str, str]
    raw: str

    def field(self, name: str) -> str | None:
        """Return a display-ready field value with LaTeX markup removed."""
        value = self.fields.get(name)
        return None if value is None else plain_text(value)

    @property
    def authors(self) -> tuple[tuple[Person, ...], bool]:
        """Return authors and whether the list ends with BibTeX ``others``."""
        value = self.fields.get("author")
        if value is None:
            raise BibliographyError(f"Bibliography entry {self.key!r} has no author")
        return parse_people(value)

    @property
    def year(self) -> str:
        """Return the publication year, or ``n.d.`` for an undated resource."""
        return self.field("year") or "n.d."


def plain_text(value: str) -> str:
    """Convert the LaTeX subset used in field values to display text."""
    text = _ACCENT.sub(_accent, value)
    for old, new in _LATEX_REPLACEMENTS:
        text = text.replace(old, new)
    if "\\" in text:
        raise BibliographyError(f"Unsupported LaTeX command in bibliography value: {value!r}")
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def _split_top_level(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
        if depth == 0 and value.startswith(separator, index):
            parts.append("".join(current))
            current = []
            index += len(separator)
            continue
        current.append(character)
        index += 1
    parts.append("".join(current))
    return parts


def parse_people(value: str) -> tuple[tuple[Person, ...], bool]:
    """Parse a BibTeX name list into people and a trailing-``others`` flag."""
    names = [name.strip() for name in _split_top_level(value, " and ")]
    truncated = bool(names) and names[-1] == "others"
    if truncated:
        names = names[:-1]
    people = []
    for name in names:
        if name.startswith("{") and name.endswith("}"):
            people.append(Person(family=plain_text(name), given="", institutional=True))
            continue
        pieces = [piece.strip() for piece in _split_top_level(name, ",")]
        if len(pieces) == 2:
            family, given = pieces
        elif len(pieces) == 1:
            words = pieces[0].split()
            family, given = words[-1], " ".join(words[:-1])
        else:
            raise BibliographyError(f"Unsupported author name form: {name!r}")
        people.append(Person(family=plain_text(family), given=plain_text(given)))
    if not people:
        raise BibliographyError(f"Empty author list: {value!r}")
    return tuple(people), truncated


def _read_value(text: str, index: int) -> tuple[str, int]:
    if text[index] == "{":
        depth = 0
        start = index + 1
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index], index + 1
            index += 1
        raise BibliographyError("Unterminated braced bibliography value")
    if text[index] == '"':
        end = text.find('"', index + 1)
        if end < 0:
            raise BibliographyError("Unterminated quoted bibliography value")
        return text[index + 1 : end], end + 1
    match = re.compile(r"[\w.-]+").match(text, index)
    if not match:
        raise BibliographyError(
            f"Unsupported bibliography value near: {text[index : index + 40]!r}"
        )
    return match.group(0), match.end()


def parse_bibliography(text: str) -> dict[str, Entry]:
    """Parse BibTeX text into entries keyed by citation key."""
    entries: dict[str, Entry] = {}
    position = 0
    while True:
        match = _ENTRY_START.search(text, position)
        if match is None:
            remainder = text[position:].strip()
            if "@" in remainder:
                raise BibliographyError(f"Unparsed bibliography text: {remainder[:60]!r}")
            break
        entry_type, key = match.group(1).lower(), match.group(2)
        index = match.end()
        fields: dict[str, str] = {}
        while True:
            while index < len(text) and text[index] in " \t\r\n,":
                index += 1
            if index >= len(text):
                raise BibliographyError(f"Unterminated bibliography entry {key!r}")
            if text[index] == "}":
                index += 1
                break
            field_match = _FIELD_NAME.match(text, index)
            if field_match is None:
                raise BibliographyError(f"Malformed field in bibliography entry {key!r}")
            name = field_match.group(1).lower()
            value, index = _read_value(text, field_match.end())
            if name in fields:
                raise BibliographyError(f"Duplicate field {name!r} in entry {key!r}")
            fields[name] = value.strip()
        if key in entries:
            raise BibliographyError(f"Duplicate bibliography key {key!r}")
        entries[key] = Entry(key, entry_type, fields, text[match.start() : index].strip())
        position = index
    return entries


def load_bibliography(path: Path = BIBLIOGRAPHY_PATH) -> dict[str, Entry]:
    """Read and parse the packaged bibliography or an explicit BibTeX file."""
    return parse_bibliography(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CitationLabels:
    """Author-year labels for one cited set, with year suffixes resolved."""

    parenthetical: Mapping[str, str]
    narrative: Mapping[str, str]
    years: Mapping[str, str]


def _author_label(entry: Entry, conjunction: str) -> str:
    people, truncated = entry.authors
    if len(people) == 1 and not truncated:
        return people[0].family
    if len(people) == 2 and not truncated:
        return f"{people[0].family} {conjunction} {people[1].family}"
    return f"{people[0].family} et al."


def citation_labels(entries: Mapping[str, Entry], keys: Iterable[str]) -> CitationLabels:
    """Build author-year labels, adding ``a``/``b`` suffixes to same-label years."""
    cited = sorted(set(keys))
    missing = [key for key in cited if key not in entries]
    if missing:
        raise BibliographyError("Unknown citation key(s): " + ", ".join(missing))
    groups: dict[tuple[str, str], list[str]] = {}
    for key in cited:
        entry = entries[key]
        groups.setdefault((_author_label(entry, "&"), entry.year), []).append(key)
    years: dict[str, str] = {}
    for (_, year), members in groups.items():
        if len(members) == 1:
            years[members[0]] = year
            continue
        ordered = sorted(members, key=lambda key: (entries[key].field("title") or "", key))
        for offset, key in enumerate(ordered):
            years[key] = f"{year}{chr(ord('a') + offset)}"
    return CitationLabels(
        parenthetical={key: f"{_author_label(entries[key], '&')}, {years[key]}" for key in cited},
        narrative={key: f"{_author_label(entries[key], 'and')} ({years[key]})" for key in cited},
        years=years,
    )


def _reference_authors(entry: Entry) -> str:
    people, truncated = entry.authors
    names = [
        person.family if person.institutional else f"{person.family}, {person.initials()}".strip()
        for person in people
    ]
    if truncated:
        return ", ".join(names) + ", et al."
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def _sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def reference_runs(entry: Entry, year: str) -> tuple[Run, ...]:
    """Format one APA-like reference-list item as inline runs."""
    runs: list[Run] = [Run.text(f"{_reference_authors(entry)} ({year}). ")]
    title = entry.field("title") or entry.key
    container = entry.field("journal") or entry.field("booktitle")
    if container:
        runs.append(Run.text(_sentence(title) + " "))
        runs.append(Run.emphasis(container))
        volume = entry.field("volume")
        number = entry.field("number")
        pages = entry.field("pages")
        if volume:
            runs.append(Run.text(", "))
            runs.append(Run.emphasis(volume))
            if number:
                runs.append(Run.text(f"({number})"))
        if pages:
            runs.append(Run.text(f", {pages}"))
        runs.append(Run.text(". "))
    else:
        runs.append(Run.emphasis(_sentence(title)))
        publisher = entry.field("organization") or entry.field("publisher")
        runs.append(Run.text(f" {_sentence(publisher)} " if publisher else " "))
    doi = entry.field("doi")
    url = entry.field("url")
    if doi:
        runs.append(Run.text(f"https://doi.org/{doi}"))
    elif url:
        runs.append(Run.text(url))
    return tuple(_merge_text(runs))


def _merge_text(runs: Sequence[Run]) -> list[Run]:
    merged: list[Run] = []
    for run in runs:
        if merged and run.kind == "text" and merged[-1].kind == "text":
            merged[-1] = Run.text(merged[-1].value + run.value)
        else:
            merged.append(run)
    if merged and merged[-1].kind == "text":
        merged[-1] = Run.text(merged[-1].value.rstrip())
    return merged


def reference_order(entries: Mapping[str, Entry], keys: Iterable[str]) -> list[str]:
    """Sort cited keys by first author, year, and title for the reference list."""

    def sort_key(key: str) -> tuple[str, str, str, str]:
        entry = entries[key]
        people, _ = entry.authors
        return (
            people[0].family.casefold(),
            entry.year,
            (entry.field("title") or "").casefold(),
            key,
        )

    return sorted(set(keys), key=sort_key)


def subset_bibtex(entries: Mapping[str, Entry], keys: Iterable[str]) -> str:
    """Return the original BibTeX text of the cited entries in key order."""
    return "\n\n".join(entries[key].raw for key in sorted(set(keys))) + "\n"
