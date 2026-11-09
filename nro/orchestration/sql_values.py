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
"""

from __future__ import annotations

import json
from collections.abc import Iterable

JSON_VALUES = "SELECT value FROM json_each(?)"


def json_values(values: Iterable[int | str]) -> str:
    """Return the JSON array parameter that matches :data:`JSON_VALUES`."""
    items = list(values)
    for value in items:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError(f"SQL list values must be int or str, not {type(value).__name__}")
    return json.dumps(items)
