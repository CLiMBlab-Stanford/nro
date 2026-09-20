from __future__ import annotations

import pytest

from nro.configuration.schema import scientific_values
from nro.configuration.store import ConfigStore, configuration_fingerprint, fingerprint
from nro.orchestration.catalog import canonical_contract
from nro.orchestration.contract_migrations import (
    INDETERMINATE,
    AddField,
    ContractMigration,
    ContractMigrationChain,
    MapValues,
    RemoveField,
    RenameField,
    current_contract_schema,
    migrate_contract,
)


def _anat_contract(*, version: int | None = None, lesion: bool | None = None) -> dict:
    source_markup = {"id": "main"}
    if lesion is not None:
        source_markup["lesion"] = lesion
    result = {
        "module": "anat",
        "configuration": "fingerprint",
        "processing": {"source_markup": source_markup},
    }
    if version is not None:
        result["contract_schema"] = version
    return result


def test_unversioned_anatomy_contract_records_historical_nonlesion_meaning() -> None:
    migrated, configuration = migrate_contract(_anat_contract(), {})

    assert configuration is not None
    assert configuration["lesion"]["masker_command"] is None
    assert configuration["lesion"]["fastsurfer_image"] is None
    assert migrated["contract_schema"] == current_contract_schema("anat") == 3
    assert migrated["processing"]["source_markup"]["lesion"] is False


def test_current_anatomy_contract_uses_current_nonlesion_default() -> None:
    migrated, _ = migrate_contract(_anat_contract(version=3))

    assert migrated["processing"]["source_markup"]["lesion"] is False


def test_anatomy_configuration_migration_preserves_ordinary_scientific_identity() -> None:
    current = ConfigStore().load_configuration("anat", "main").values
    historical = {key: value for key, value in current.items() if key != "lesion"}
    _, migrated = migrate_contract(_anat_contract(), historical)

    assert migrated is not None
    assert configuration_fingerprint(
        "anat", "main", migrated, scientific=True, complete_snapshot=True
    ) == configuration_fingerprint("anat", "main", current, scientific=True)
    historical_lineage = fingerprint(
        {
            "module": "anat",
            "config_id": "main",
            "values": scientific_values("anat", historical),
        }
    )
    assert historical_lineage == configuration_fingerprint("anat", "main", current, scientific=True)


def test_canonical_contract_distinguishes_revised_anatomical_methods() -> None:
    current = ConfigStore().load_configuration("anat", "main").values
    historical = {key: value for key, value in current.items() if key != "lesion"}
    source_markup = {
        "id": "main",
        "project": "demo",
        "subject_dir": "/bids/demo/sub-1",
        "T1w": [],
        "T2w": [],
        "exclude": [],
    }
    old_fingerprint = configuration_fingerprint("anat", "main", historical)
    new_fingerprint = configuration_fingerprint("anat", "main", current)
    old_contract = {
        "module": "anat",
        "configuration": old_fingerprint,
        "processing": {"source_markup": source_markup},
    }
    new_contract = {
        "contract_schema": 3,
        "module": "anat",
        "configuration": new_fingerprint,
        "processing": {
            "source_markup": {**source_markup, "lesion": False},
            "bias_correction": {
                "method": "N4BiasFieldCorrection",
                "mask_source": "SynthStrip",
                "mask_application": "hard_mask",
                "bias_field_retained": True,
            },
            "surface_reconstruction": {
                "backend": "FreeSurfer",
                "version": "7.4.1",
                "build": "freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460",
                "skull_stripping": "SynthStrip_external_mask",
                "recon_all_stages": ["autorecon1", "autorecon2", "autorecon3"],
            },
        },
    }

    assert canonical_contract(
        old_contract,
        {"id": "main", "fingerprint": old_fingerprint, "resolved": historical},
    ) != canonical_contract(
        new_contract,
        {"id": "main", "fingerprint": new_fingerprint, "resolved": current},
    )


def test_lesion_true_remains_scientifically_distinct() -> None:
    ordinary, _ = migrate_contract(_anat_contract())
    lesioned, _ = migrate_contract(_anat_contract(version=3, lesion=True))

    assert ordinary != lesioned
    assert lesioned["processing"]["source_markup"]["lesion"] is True


def test_add_field_distinguishes_current_default_from_historical_value() -> None:
    chain = ContractMigrationChain(
        module="example",
        migrations=(
            ContractMigration(
                destination=2,
                summary="Change method default",
                configuration=(AddField("method", default="new", historical="legacy"),),
            ),
        ),
    )

    old_contract, old_configuration = chain.normalize(
        {"module": "example"},
        {},
    )
    new_contract, new_configuration = chain.normalize(
        {"module": "example", "contract_schema": 2},
        {},
    )

    assert old_contract["contract_schema"] == new_contract["contract_schema"] == 2
    assert old_configuration == {"method": "legacy"}
    assert new_configuration == {"method": "new"}


def test_indeterminate_historical_value_cannot_equal_current_default() -> None:
    chain = ContractMigrationChain(
        module="example",
        migrations=(
            ContractMigration(
                destination=2,
                summary="Add an unrecoverable method",
                contract=(AddField("method", default="new", historical=INDETERMINATE),),
            ),
        ),
    )

    old, _ = chain.normalize({"module": "example"})
    current, _ = chain.normalize({"module": "example", "contract_schema": 2})

    assert old["method"] == INDETERMINATE
    assert current["method"] == "new"
    assert old != current


def test_restricted_contract_operations_transform_historical_syntax() -> None:
    chain = ContractMigrationChain(
        module="example",
        migrations=(
            ContractMigration(
                destination=2,
                summary="Normalize method vocabulary",
                contract=(
                    RenameField("old_name", "method"),
                    MapValues("method", {"a": "alpha", "b": "beta"}),
                    RemoveField("temporary", reconstructible=True),
                ),
            ),
        ),
    )

    migrated, _ = chain.normalize({"module": "example", "old_name": "a", "temporary": 1})

    assert migrated == {"module": "example", "method": "alpha", "contract_schema": 2}


def test_contract_chain_rejects_gaps_and_future_versions() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        ContractMigrationChain(
            module="example",
            migrations=(ContractMigration(destination=3, summary="gap"),),
        )

    chain = ContractMigrationChain(module="example")
    with pytest.raises(ValueError, match="Unsupported"):
        chain.normalize({"module": "example", "contract_schema": 2})
