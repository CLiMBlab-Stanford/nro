"""Format-independent document model for generated methods sections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Run:
    """One inline span: plain text, emphasized text, or a citation group.

    A citation stores its keys in ``keys``. ``narrative`` selects the in-sentence
    form, such as "Afyouni and Nichols (2018)", instead of the parenthetical form.
    """

    kind: str
    value: str = ""
    keys: tuple[str, ...] = ()
    narrative: bool = False

    @classmethod
    def text(cls, value: str) -> "Run":
        """Return a plain-text run."""
        return cls("text", value)

    @classmethod
    def emphasis(cls, value: str) -> "Run":
        """Return an italic run."""
        return cls("emphasis", value)

    @classmethod
    def citation(cls, keys: tuple[str, ...], *, narrative: bool = False) -> "Run":
        """Return a citation run for one or more bibliography keys."""
        if not keys or (narrative and len(keys) != 1):
            raise ValueError("A citation needs keys, and a narrative citation needs exactly one")
        return cls("citation", keys=keys, narrative=narrative)


Paragraph = tuple[Run, ...]


@dataclass(frozen=True)
class Section:
    """One rendered template section with its source template path."""

    module: str
    title: str
    paragraphs: tuple[Paragraph, ...]
    template: str


@dataclass(frozen=True)
class Table:
    """A supplementary table with string cells in display order."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MethodsDocument:
    """A complete methods report before format-specific writing."""

    sections: tuple[Section, ...]
    module_titles: Mapping[str, str]
    references: tuple[Paragraph, ...]
    tables: tuple[Table, ...]
    footer: str
    cited_keys: tuple[str, ...] = field(default=())
