"""Bind a variable-length value list to one SQLite parameter.

SQLite limits how many parameters one statement may bind. The limit is 32766
in current builds and 999 in older ones. A query that writes one ``?`` per
work item fails with ``too many SQL variables`` on large sites.

Use :data:`JSON_VALUES` inside ``IN (...)`` and bind :func:`json_values` as
its single parameter::

    database.execute(
        f"SELECT id FROM attempts WHERE work_item_id IN ({JSON_VALUES})",
        (json_values(work_item_ids),),
    )

``json_each`` returns JSON integers as SQLite integers and JSON strings as
text, so comparisons match the old one-parameter-per-value binding. An empty
collection produces no rows: ``IN`` is false and ``NOT IN`` is true, as with
an empty literal list.

Queries over large lists still run in chunks of :data:`QUERY_CHUNK_SIZE`
values so that each statement and its parameter stay bounded. Use
:func:`chunks` to split a list, or :func:`rows_for_values` to run one query per
chunk.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

JSON_VALUES = "SELECT value FROM json_each(?)"
QUERY_CHUNK_SIZE = 500

T = TypeVar("T")


def json_values(values: Iterable[int | str]) -> str:
    """Return the JSON array parameter that matches :data:`JSON_VALUES`."""
    items = list(values)
    for value in items:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError(f"SQL list values must be int or str, not {type(value).__name__}")
    return json.dumps(items)


def chunks(values: Sequence[T], size: int = QUERY_CHUNK_SIZE) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of at most `size` values, in order."""
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def rows_for_values(
    database: sqlite3.Connection,
    statement: str,
    values: Sequence[int | str],
    *,
    chunk_size: int = QUERY_CHUNK_SIZE,
) -> Iterator[sqlite3.Row]:
    """Yield the rows of `statement` for each chunk of `values`.

    `statement` contains a `{placeholders}` field, which is replaced by
    :data:`JSON_VALUES` and bound to one chunk at a time.
    """
    for chunk in chunks(values, chunk_size):
        yield from database.execute(
            statement.format(placeholders=JSON_VALUES), (json_values(chunk),)
        )
