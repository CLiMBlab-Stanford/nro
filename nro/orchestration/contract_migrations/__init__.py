"""Module-specific scientific artifact-contract migration chains."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nro.modules import MODULE_NAMES

from .anat import CHAIN as ANAT_CHAIN
from .core import (
    INDETERMINATE,
    AddField,
    ContractMigration,
    ContractMigrationChain,
    MapValues,
    RemoveField,
    RenameField,
)

CHAINS = {
    module: ANAT_CHAIN if module == "anat" else ContractMigrationChain(module=module)
    for module in MODULE_NAMES
}


def current_contract_schema(module: str) -> int:
    """Return a built-in schema, or the baseline for a branch extension."""
    return CHAINS.get(module, ContractMigrationChain(module=module)).current_version


def migrate_contract(
    contract: Mapping[str, Any],
    configuration: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Normalize a recorded contract and optional configuration snapshot."""
    module = contract.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError(f"Unknown artifact contract module: {module!r}")
    chain = CHAINS.get(module, ContractMigrationChain(module=module))
    return chain.normalize(contract, configuration)


__all__ = [
    "INDETERMINATE",
    "AddField",
    "ContractMigration",
    "ContractMigrationChain",
    "MapValues",
    "RemoveField",
    "RenameField",
    "current_contract_schema",
    "migrate_contract",
]
