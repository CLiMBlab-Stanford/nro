"""Strict YAML parsing shared by configuration consumers and authoring tools."""

from copy import deepcopy
from functools import lru_cache
import re

import yaml


class DefinitionError(ValueError):
    """A user-authored definition has invalid syntax or unsupported settings."""


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys):
            raise DefinitionError(f"Mapping keys must be strings at line {node.start_mark.line + 1}")
        if len(set(keys)) != len(keys):
            raise DefinitionError(f"Duplicate YAML mapping keys at line {node.start_mark.line + 1}")
        return super().construct_mapping(node, deep=deep)


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


@lru_cache(maxsize=256)
def _parse(text: str) -> dict:
    value = yaml.load(text, Loader=_Loader)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DefinitionError("Definition must be a YAML mapping")

    def check_cycles(item, ancestors):
        if not isinstance(item, (dict, list)):
            return
        if id(item) in ancestors:
            raise DefinitionError("Recursive YAML aliases are not supported")
        for child in item.values() if isinstance(item, dict) else item:
            check_cycles(child, ancestors | {id(item)})

    check_cycles(value, set())
    return value


def parse_mapping(text: str, *, source: str = "definition") -> dict:
    """Parse strict YAML using a bounded text cache; return an independent copy."""
    try:
        return deepcopy(_parse(text))
    except (yaml.YAMLError, DefinitionError) as error:
        raise DefinitionError(f"{source}: {error}") from error
