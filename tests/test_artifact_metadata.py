"""Compatibility rules for evolving public metadata schemas."""

import math
from dataclasses import replace

import pytest

from nro.engine.artifact_metadata import (
    metadata_contract_compatible,
    metadata_value,
    validate_metadata_fields,
)
from nro.orchestration import catalog


def test_metadata_validation_accepts_defaults_and_ignores_unknown_fields() -> None:
    fields = {
        "Required": "string",
        "Added": {"kind": "integer", "default": 3},
    }
    document = {"Required": "present", "Retired": "ignored"}

    validate_metadata_fields(document, fields)

    assert metadata_value(document, "Added", fields["Added"]) == 3


def test_metadata_contract_accepts_removed_fields_and_defaulted_additions() -> None:
    recorded = {"manifest_fields": {"Required": "string", "Retired": "integer"}}
    current = {
        "manifest_fields": {
            "Required": "string",
            "Added": {"kind": "boolean", "default": False},
        }
    }

    assert metadata_contract_compatible(recorded, current)
    assert not metadata_contract_compatible(
        recorded,
        {"manifest_fields": {"Required": "string", "Added": "boolean"}},
    )
    assert not metadata_contract_compatible(
        recorded,
        {"manifest_fields": {"Required": "integer"}},
    )


def test_metadata_contract_rejects_changed_defaults_and_new_requirements() -> None:
    recorded = {"manifest_fields": {"Optional": {"kind": "integer", "default": 1}}}

    assert not metadata_contract_compatible(
        recorded,
        {"manifest_fields": {"Optional": {"kind": "integer", "default": 2}}},
    )
    assert not metadata_contract_compatible(
        recorded,
        {"manifest_fields": {"Optional": "integer"}},
    )


def test_metadata_numbers_must_be_finite() -> None:
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="must have type number"):
            validate_metadata_fields({"Metric": value}, {"Metric": "number"})


def test_legacy_required_field_lists_normalize_to_typed_schemas() -> None:
    recorded = {"required_manifest_fields": ["kept", "removed"]}
    current = {
        "required_manifest_fields": {
            "kept": "string",
            "added": {"kind": "boolean", "default": False},
        }
    }

    assert metadata_contract_compatible(recorded, current)


def test_contract_canonicalization_uses_compatible_current_metadata(monkeypatch) -> None:
    descriptor = catalog.module_descriptor("clean")
    current = {
        "manifest_fields": {
            "Required": "string",
            "Added": {"kind": "boolean", "default": False},
        }
    }
    monkeypatch.setitem(
        catalog.MODULE_CATALOG,
        "clean",
        replace(descriptor, processing_contract=lambda: {"output_metadata": current}),
    )
    recorded = {
        "module": "clean",
        "processing": {
            "output_metadata": {"manifest_fields": {"Required": "string", "Retired": "number"}},
            "estimator": "unchanged",
        },
    }

    assert catalog.canonical_contract(recorded)["processing"] == {
        "output_metadata": current,
        "estimator": "unchanged",
    }


def test_invalid_default_is_rejected() -> None:
    with pytest.raises(ValueError, match="must have type integer"):
        validate_metadata_fields({}, {"Added": {"kind": "integer", "default": "three"}})
