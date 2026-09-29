from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nro.engine.io import read_public_json, write_public_json
from nro.engine.references import (
    ReferenceRoots,
    absolute_path_values,
    bids_uri,
    ensure_derivative_dataset,
    omit_private_path_values,
    portable_path,
    resolve_reference,
)
from nro.orchestration.ownership import (
    convert_legacy_ownership_record,
    normalize_portable_ownership_record,
)


def _roots(tmp_path: Path) -> ReferenceRoots:
    project = tmp_path / "BIDS" / "demo"
    return ReferenceRoots.for_project(
        project,
        site_roots={
            "definitions": tmp_path / "definitions",
            "templates": tmp_path / "templateflow",
        },
    )


def test_bids_references_round_trip_after_dataset_move(tmp_path: Path) -> None:
    first = _roots(tmp_path / "first")
    source = first.project / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    output = first.derivative / "anat" / "main-abc" / "sub-01" / "anat" / "result.nii.gz"

    source_reference = bids_uri(source, first)
    output_reference = bids_uri(output, first)

    second = _roots(tmp_path / "second")
    assert resolve_reference(source_reference, second) == (
        second.project / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
    )
    assert resolve_reference(output_reference, second) == (
        second.derivative / "anat" / "main-abc" / "sub-01" / "anat" / "result.nii.gz"
    )


def test_reference_resolution_rejects_traversal_and_unknown_links(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    with pytest.raises(ValueError, match="Invalid BIDS URI"):
        resolve_reference("bids:raw:../outside", roots)
    with pytest.raises(ValueError, match="Unknown BIDS dataset link"):
        resolve_reference("bids:other:sub-01/file.nii.gz", roots)


def test_absolute_path_inventory_ignores_embedded_commands() -> None:
    assert absolute_path_values(
        {"path": "/host/data/file.nii.gz", "command": "tool /host/data/file.nii.gz"}
    ) == ("/host/data/file.nii.gz",)


def test_private_receipts_can_name_configured_site_resources(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    template = tmp_path / "templateflow" / "tpl-demo" / "template.nii.gz"
    reference = portable_path(template, roots)

    assert reference == "nro-site:templates:tpl-demo/template.nii.gz"
    assert resolve_reference(reference, roots) == template.resolve()
    assert portable_path(template, roots, public=True) == reference


def test_public_metadata_omits_private_execution_paths(tmp_path: Path) -> None:
    project = tmp_path / "BIDS/demo"
    work = tmp_path / "WORK"
    roots = ReferenceRoots.for_project(project, site_roots={"work": work})

    converted = omit_private_path_values(
        {
            "registration": {
                "static_warp": str(work / "warp.nii.gz"),
                "details": {
                    "method": "topup",
                    "transform": str(work / "transform.mat"),
                },
                "inputs": [str(project / "sub-01/func/bold.nii.gz"), str(work / "ref.nii.gz")],
            }
        },
        roots,
    )

    assert converted == {
        "registration": {
            "static_warp": "BOLDToT1wComposite",
            "details": {"method": "topup"},
            "inputs": [str(project / "sub-01/func/bold.nii.gz")],
        }
    }


def test_public_metadata_omits_private_path_mapping_keys(tmp_path: Path) -> None:
    project = tmp_path / "BIDS/demo"
    work = tmp_path / "WORK"
    roots = ReferenceRoots.for_project(project, site_roots={"work": work})

    converted = omit_private_path_values(
        {
            str(work / "private.nii.gz"): {"state": "private"},
            str(project / "sub-01/anat/sub-01_T1w.nii.gz"): {"state": "public"},
        },
        roots,
    )

    assert converted == {str(project / "sub-01/anat/sub-01_T1w.nii.gz"): {"state": "public"}}


def test_bids_reference_preserves_a_logical_symlink_member(tmp_path: Path) -> None:
    external = tmp_path / "external/source.nii.gz"
    external.parent.mkdir()
    external.write_bytes(b"image")
    project = tmp_path / "bids/demo"
    linked = project / "sub-01/anat/sub-01_T1w.nii.gz"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(external)
    roots = ReferenceRoots.for_project(project)

    reference = bids_uri(linked, roots)

    assert reference == "bids:raw:sub-01/anat/sub-01_T1w.nii.gz"
    assert resolve_reference(reference, roots) == linked


def test_derivative_dataset_description_links_raw_project(tmp_path: Path) -> None:
    project = tmp_path / "BIDS" / "demo"
    path = ensure_derivative_dataset(project, version="1.2.3")
    value = json.loads(path.read_text(encoding="utf-8"))

    assert path == project / "derivatives" / "nro" / "dataset_description.json"
    assert value["DatasetType"] == "derivative"
    assert value["DatasetLinks"] == {"raw": "../.."}
    assert value["GeneratedBy"][0] == {"Name": "nro", "Version": "1.2.3"}


def test_legacy_ownership_conversion_removes_launcher_and_host_paths(tmp_path: Path) -> None:
    project = tmp_path / "bids" / "demo"
    derivative = project / "derivatives" / "nro"
    roots = ReferenceRoots.for_project(
        project,
        derivative_root=derivative,
        site_roots={"checkout": tmp_path / "checkout"},
    )
    legacy = {
        "record_version": 4,
        "owner": "nro",
        "artifact_contract": {
            "inputs": [str(project / "sub-01/anat/sub-01_T1w.nii.gz")],
            "output": {"root": str(derivative / "anat/main/sub-01")},
        },
        "scientific_contract": {},
        "execution": {
            "command": [
                "/usr/bin/python3",
                str(tmp_path / "checkout/nro/orchestration/source_launcher.py"),
                "digest",
                "-",
                "-",
                "nro.modules.anat",
                "--participant",
                "01",
            ],
            "runtime_configuration": {},
        },
    }

    converted = convert_legacy_ownership_record(legacy, roots=roots)

    assert converted["record_version"] == 5
    assert converted["artifact_contract"]["inputs"] == ["bids:raw:sub-01/anat/sub-01_T1w.nii.gz"]
    assert converted["artifact_contract"]["output"]["root"] == "bids::anat/main/sub-01"
    assert converted["execution"] == {
        "module_argv": ["nro.modules.anat", "--participant", "01"],
        "runtime_configuration": {},
    }
    assert str(tmp_path) not in json.dumps(converted)


def test_public_document_resolves_against_moved_project(tmp_path: Path) -> None:
    first = tmp_path / "first/BIDS/demo"
    output = first / "derivatives/nro/anat/main/sub-01/anat/result.nii.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"image")
    source = first / "sub-01/anat/sub-01_T1w.nii.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    manifest = output.with_name("manifest.json")
    write_public_json(manifest, {"source": str(source), "output": str(output)})

    second = tmp_path / "second/BIDS/demo"
    shutil.copytree(first, second)
    moved = second / manifest.relative_to(first)
    document = read_public_json(moved)

    assert document == {
        "source": str(second / source.relative_to(first)),
        "output": str(second / output.relative_to(first)),
    }


def test_public_document_converts_path_valued_mapping_keys(tmp_path: Path) -> None:
    project = tmp_path / "BIDS/demo"
    source = project / "sub-01/anat/sub-01_T1w.nii.gz"
    manifest = project / "derivatives/nro/anat/main/sub-01/manifest.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")

    write_public_json(manifest, {str(source): {"action": "already_corrected"}})
    stored = json.loads(manifest.read_text(encoding="utf-8"))

    assert stored == {"bids:raw:sub-01/anat/sub-01_T1w.nii.gz": {"action": "already_corrected"}}
    assert read_public_json(manifest) == {str(source): {"action": "already_corrected"}}


def test_portable_ownership_uses_site_setting_for_container_binds(tmp_path: Path) -> None:
    roots = _roots(tmp_path)
    record = {
        "record_version": 5,
        "owner": "nro",
        "configuration_class": "anat",
        "configuration": {
            "id": "main",
            "fingerprint": "historical",
            "resolved": {
                "container": {
                    "engine": "singularity",
                    "bind": ["/host:/container"],
                }
            },
        },
        "execution": {
            "runtime_configuration": {
                "container": {
                    "engine": "singularity",
                    "bind": ["/host:/container"],
                }
            }
        },
    }

    converted = normalize_portable_ownership_record(record, roots=roots)

    expected = {"$nro_site_setting": "binds"}
    assert converted["configuration"]["resolved"]["container"]["bind"] == expected
    assert converted["execution"]["runtime_configuration"]["container"]["bind"] == expected
    assert "/host" not in json.dumps(converted)
