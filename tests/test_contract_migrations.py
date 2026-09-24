from __future__ import annotations

import pytest

from nro.configuration.store import ConfigStore, configuration_fingerprint
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
    assert configuration["surface_reconstruction_engine"] == "freesurfer"
    assert migrated["contract_schema"] == current_contract_schema("anat") == 6
    assert migrated["processing"]["source_markup"]["lesion"] is False


def test_current_anatomy_contract_uses_current_nonlesion_default() -> None:
    migrated, configuration = migrate_contract(_anat_contract(version=6), {})

    assert migrated["processing"]["source_markup"]["lesion"] is False
    assert configuration is not None
    assert configuration["surface_reconstruction_engine"] == "freesurfer"


def test_anatomy_configuration_migration_preserves_ordinary_scientific_identity() -> None:
    current = ConfigStore().load_configuration("anat", "main").values
    historical = {
        key: value
        for key, value in current.items()
        if key not in {"lesion", "surface_reconstruction_engine"}
    }
    _, migrated = migrate_contract(_anat_contract(), historical)

    assert migrated is not None
    assert configuration_fingerprint(
        "anat", "main", migrated, scientific=True, complete_snapshot=True
    ) == configuration_fingerprint("anat", "main", current, scientific=True)
    assert migrated["surface_reconstruction_engine"] == "freesurfer"


def test_freesurfer_contract_survives_introduction_of_engine_selector() -> None:
    resolved = ConfigStore().load_configuration("anat", "main")
    current = resolved.values
    historical = {
        key: value
        for key, value in current.items()
        if key not in {"fastsurfer_container", "surface_reconstruction_engine"}
    }
    historical_full_fingerprint = configuration_fingerprint("anat", "main", historical)
    old_fingerprint = "historical-anat-scientific-fingerprint"
    new_fingerprint = resolved.module_fingerprint("anat")
    processing = {
        "source_markup": {"id": "main", "lesion": False},
        "bias_correction": {
            "method": "N4BiasFieldCorrection",
            "mask_source": "SynthStrip",
            "mask_application": "hard_mask",
            "bias_field_retained": True,
        },
        "surface_reconstruction": {
            "backend": "FreeSurfer",
            "version": "7.4.1",
            "build": "freesurfer-linux-centos8_x86_64-7.4.1-20230613-7eb8460",
            "skull_stripping": "SynthStrip_external_mask",
            "recon_all_stages": ["autorecon1", "autorecon2", "autorecon3"],
            "external_mask_resampling": "nearest_neighbor",
            "brainmask_intensity_source": "FreeSurfer_normalized_T1",
        },
    }
    old_contract = {
        "contract_schema": 4,
        "module": "anat",
        "configuration": old_fingerprint,
        "processing": processing,
    }
    new_contract = {
        "contract_schema": 6,
        "module": "anat",
        "configuration": new_fingerprint,
        "processing": processing,
    }

    assert canonical_contract(
        old_contract,
        {
            "id": "main",
            "fingerprint": historical_full_fingerprint,
            "resolved": historical,
        },
    ) == canonical_contract(
        new_contract,
        {
            "id": "main",
            "fingerprint": configuration_fingerprint("anat", "main", current),
            "resolved": current,
        },
    )


def test_canonical_contract_distinguishes_revised_anatomical_methods() -> None:
    current = ConfigStore().load_configuration("anat", "main").values
    historical = {
        key: value
        for key, value in current.items()
        if key not in {"lesion", "surface_reconstruction_engine"}
    }
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
    lesioned, _ = migrate_contract(_anat_contract(version=4, lesion=True))

    assert ordinary != lesioned
    assert lesioned["processing"]["source_markup"]["lesion"] is True


def test_legacy_lesion_backend_fields_retire_after_pipeline_refactor() -> None:
    historical = _anat_contract(version=3, lesion=True)
    historical["processing"]["lesion_reconstruction"] = {"surface_backend": "FastSurfer-LIT"}
    current = _anat_contract(version=4, lesion=True)
    current["processing"]["lesion_reconstruction"] = {"surface_backend": "FastSurfer-LIT"}

    migrated_historical, _ = migrate_contract(historical)
    migrated_current, _ = migrate_contract(current)

    for migrated in (migrated_historical, migrated_current):
        lesion = migrated["processing"]["lesion_reconstruction"]
        assert lesion["pipeline"] == INDETERMINATE
        assert "surface_backend" not in lesion
        assert "fastsurfer_voxel_size_mm" not in lesion


def test_lesion_resolution_migration_leaves_ordinary_anatomy_unchanged() -> None:
    historical = _anat_contract(version=3, lesion=False)
    historical["processing"]["surface_reconstruction"] = {"backend": "FreeSurfer"}
    current = _anat_contract(version=4, lesion=False)
    current["processing"]["surface_reconstruction"] = {"backend": "FreeSurfer"}

    migrated_historical, _ = migrate_contract(historical)
    migrated_current, _ = migrate_contract(current)

    assert migrated_historical == migrated_current


def test_decoupled_lesion_reconstruction_invalidates_only_lesion_anatomy() -> None:
    ordinary_v5 = _anat_contract(version=5, lesion=False)
    ordinary_v5["processing"]["surface_reconstruction"] = {"backend": "FreeSurfer"}
    ordinary_v6 = _anat_contract(version=6, lesion=False)
    ordinary_v6["processing"]["surface_reconstruction"] = {"backend": "FreeSurfer"}
    lesion_v5 = _anat_contract(version=5, lesion=True)
    lesion_v5["processing"]["lesion_reconstruction"] = {"surface_backend": "FastSurfer-LIT"}
    lesion_v6 = _anat_contract(version=6, lesion=True)
    lesion_v6["processing"]["lesion_reconstruction"] = {
        "pipeline": "inpainting_surface_reconstruction_excision"
    }

    migrated_ordinary_v5, _ = migrate_contract(ordinary_v5)
    migrated_ordinary_v6, _ = migrate_contract(ordinary_v6)
    migrated_lesion_v5, _ = migrate_contract(lesion_v5)
    migrated_lesion_v6, _ = migrate_contract(lesion_v6)

    assert migrated_ordinary_v5 == migrated_ordinary_v6
    assert migrated_lesion_v5 != migrated_lesion_v6
    assert migrated_lesion_v5["processing"]["lesion_reconstruction"]["pipeline"] == (INDETERMINATE)


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
