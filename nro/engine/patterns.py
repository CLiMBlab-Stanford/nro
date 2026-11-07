"""Glob and regular-expression matching for data selectors.

These rules apply to project, participant, task, and ``-r`` run-entity
selector values. A value that starts with ``re:`` is a regular expression that
must match the whole value. A value containing ``*``, ``?``, or ``[`` is a
case-sensitive shell-style glob. Any other value matches only itself.
"""

import re
from fnmatch import fnmatchcase
from functools import lru_cache
from typing import Callable, Iterable, Sequence

REGEX_PREFIX = "re:"
_GLOB_CHARACTERS = frozenset("*?[")


def is_pattern(value: str) -> bool:
    """Return whether a selector value is a glob or regular expression."""
    return value.startswith(REGEX_PREFIX) or not _GLOB_CHARACTERS.isdisjoint(value)


@lru_cache(maxsize=1024)
def _compile(value: str) -> Callable[[str], bool]:
    if value.startswith(REGEX_PREFIX):
        expression = value[len(REGEX_PREFIX) :]
        try:
            compiled = re.compile(expression)
        except re.error as error:
            raise ValueError(f"Invalid regular expression {expression!r}: {error}") from error
        return lambda candidate: compiled.fullmatch(candidate) is not None
    if is_pattern(value):
        return lambda candidate: fnmatchcase(candidate, value)
    return lambda candidate: candidate == value


def validate_patterns(values: Iterable[str]) -> None:
    """Raise ValueError for a selector value that is not a valid pattern."""
    for value in values:
        _compile(value)


def reject_patterns(values: Iterable[str], option: str) -> None:
    """Raise ValueError when an exact-only option receives a pattern."""
    patterns = [str(value) for value in values if is_pattern(str(value))]
    if patterns:
        raise ValueError(f"{option} accepts exact values only, got: {', '.join(patterns)}")


def matches_any(candidate: str, values: Iterable[str]) -> bool:
    """Return whether a candidate matches any exact value or pattern."""
    return any(_compile(str(value))(candidate) for value in values)


def split_alternatives(value: str) -> tuple[str, ...]:
    """Split comma-separated alternatives, keeping commas inside brackets.

    Commas inside ``(...)``, ``[...]``, and ``{...}`` belong to the pattern, so
    ``re:0{1,2}`` and ``[1,2]`` stay whole. A backslash escapes the next
    character. Empty alternatives are dropped.
    """
    alternatives: list[str] = []
    current: list[str] = []
    closing: list[str] = []
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}"}
    for character in value:
        if escaped:
            current.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character in pairs:
            closing.append(pairs[character])
        elif closing and character == closing[-1]:
            closing.pop()
        if character == "," and not closing:
            alternatives.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    alternatives.append("".join(current).strip())
    return tuple(item for item in alternatives if item)


def expand_patterns(values: Sequence[str], available: Iterable[str]) -> tuple[str, ...]:
    """Return available identifiers matched by any selector value, sorted.

    Exact values are kept even when absent so that callers can report them.
    """
    candidates = tuple(dict.fromkeys(available))
    exact = [value for value in values if not is_pattern(value)]
    patterns = [_compile(value) for value in values if is_pattern(value)]
    matched = [item for item in candidates if any(match(item) for match in patterns)]
    return tuple(sorted(dict.fromkeys([*exact, *matched])))


def unmatched_patterns(values: Sequence[str], available: Iterable[str]) -> tuple[str, ...]:
    """Return pattern values that match none of the available identifiers."""
    candidates = tuple(available)
    return tuple(
        value
        for value in values
        if is_pattern(value) and not any(_compile(value)(item) for item in candidates)
    )
