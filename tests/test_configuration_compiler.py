"""Shared configuration validation, normalization, and scientific freshness."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from nro.configuration.authoring import definition_target, validate_definition
from nro.configuration.parsing import parse_mapping
from nro.configuration.runtime import load_runtime_configuration
from nro.configuration.schema import SCHEMAS, compile_configuration, scientific_values
from nro.configuration.store import ConfigStore, configuration_fingerprint, fingerprint


@pytest.fixture
def store(tmp_path):
    store = ConfigStore()
    shutil.copytree(store.root, tmp_path / "store")
    store.root = tmp_path / "store"
    return store


@pytest.mark.parametrize("kind", SCHEMAS)
def test_defaults_compile_completely_and_cached_results_are_independent(store, kind):
    from nro.configuration import schema

    config = store.load_configuration(kind, "main")
    source = deepcopy(config.values)
    first = compile_configuration(kind, source)
    info = schema._compile.cache_info()
    assert compile_configuration(kind, source) == first
    assert schema._compile.cache_info().hits == info.hits + 1
    first.clear()
    assert compile_configuration(kind, source) == config.values
    assert source == config.values
    assert compile_configuration(kind, compile_configuration(kind, source)) == source


@pytest.mark.parametrize(
    "text", ["verbose: false\nverbose: true", "coarsening:\n  iterations: 2\n  iterations: 3"]
)
def test_duplicate_keys_rejected_on_every_read_path(store, tmp_path, text):
    kind = "clean" if text.startswith("verbose") else "microparcellation"
    target = definition_target(store, "config", f"{kind}/bad")
    target.path.write_text(text)
    with pytest.raises(ValueError, match="Duplicate YAML.*line"):
        validate_definition(store, target, text)
    with pytest.raises(ValueError, match=str(target.path)):
        store.load_configuration(kind, "bad")
    runtime = tmp_path / f"bad_{kind}.yml"
    runtime.write_text(text)
    with pytest.raises(ValueError, match="Duplicate YAML"):
        load_runtime_configuration(runtime, kind)


@pytest.mark.parametrize(
    "kind,values,field",
    [
        ("clean", {"min_trs": True}, "min_trs"),
        ("clean", {"min_trs": 3.5}, "min_trs"),
        ("clean", {"verbose": "false"}, "verbose"),
        ("clean", {"nuisance_variance_explained": float("nan")}, "nuisance_variance_explained"),
        ("clean", {"minimum_temporal_rank_fraction": 2}, "minimum_temporal_rank_fraction"),
        ("clean", {"confounds_regex": "["}, "confounds_regex"),
        ("clean", {"high_pass": 0.2}, "high_pass"),
        ("clean", {"min_trs": "50"}, "min_trs"),
        ("networks", {"parcellation_strategy": "typo"}, "parcellation_strategy"),
        ("networks", {"ica": {"n_networks": 0}}, "n_networks"),
        ("networks", {"oslom": {"directed": True}}, "directed"),
        (
            "networks",
            {"parcellation_strategy": "oslom", "oslom": {"initialization": "file"}},
            "initial_partition",
        ),
        ("microparcellation", {"mask": 42}, "mask"),
        ("microparcellation", {"input_filter": {"task": False}}, "task"),
        (
            "microparcellation",
            {"connectivity": {"minimum_retained_frames": 4}},
            "minimum_retained_frames",
        ),
        ("firstlevels", {"ar_grid": [0, 0]}, "ar_grid"),
        ("firstlevels", {"ar_grid": [1]}, "ar_grid"),
        ("firstlevels", {"aggregation_weighting": "unknown"}, "aggregation_weighting"),
        ("firstlevels", {"low_pass": 0.1}, "low_pass"),
        ("preprocessing", {"func": {"bbregister_dof": 5}}, "bbregister_dof"),
        ("preprocessing", {"func": {"output_spaces": []}}, "output_spaces"),
    ],
)
def test_store_and_authoring_share_semantic_errors(store, kind, values, field):
    target = definition_target(store, "config", f"{kind}/bad")
    text = yaml.safe_dump(values)
    target.path.write_text(text)
    with pytest.raises(ValueError, match=field):
        store.load_configuration(kind, "bad")
    with pytest.raises(ValueError, match=field):
        validate_definition(store, target, text)


def test_external_main_is_a_validated_partial_override(store):
    path = store.configs / "clean/main_clean.yml"
    for value in ({"verbose": "yes"}, {"unknown": 1}):
        path.write_text(yaml.safe_dump(value))
        with pytest.raises(ValueError):
            store.load_configuration("clean", "main")
    path.write_text("minimum_temporal_rank: 25\n")
    values = store.load_configuration("clean", "main").values
    assert values["minimum_temporal_rank"] == 25
    assert values["min_trs"] == 50


def test_equivalent_values_defaults_and_site_references(store):
    default = store.load_configuration("clean", "main")
    assert (
        store.load_configuration("clean", "main", document=default.values).fingerprint
        == default.fingerprint
    )
    source = yaml.safe_load(default.path.read_text())
    source["min_trs"] = float(source["min_trs"])
    source["minimum_temporal_rank"] = float(source["minimum_temporal_rank"])
    source["gm_mask_threshold"] = 0.2
    source = dict(reversed(list(source.items())))
    assert (
        store.load_configuration("clean", "main", document=source).fingerprint
        == default.fingerprint
    )
    override = store.load_configuration("clean", "other", document={})
    explicit = store.load_configuration("clean", "other", document={"min_trs": 50.0})
    assert override == explicit
    assert override.scientific_fingerprint != default.scientific_fingerprint
    main = store.resolve("main")
    assert store.resolve("main", document={key: "main" for key in SCHEMAS}) == main
    assert parse_mapping("a: []") is not parse_mapping("a: []")
    assert parse_mapping("low_pass: 1e-1") == {"low_pass": 0.1}
    assert parse_mapping("low_pass: '1e-1'") == {"low_pass": "1e-1"}


def test_content_cache_detects_edits_without_timestamp_changes(store):
    path = store.configs / "clean/dev_clean.yml"
    path.write_text("min_trs: 50\n")
    original = store.load_configuration("clean", "dev")
    import os

    stamp = path.stat()
    path.write_text("min_trs: 60\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert (
        store.load_configuration("clean", "dev").scientific_fingerprint
        != original.scientific_fingerprint
    )


def test_execution_roles_are_explicit_and_scientific_order_is_preserved(store):
    original = store.load_configuration("networks", "main")
    updated = store.load_configuration(
        "networks",
        "main",
        document={
            **original.values,
            "overwrite": True,
            "oslom": {**original.values["oslom"], "timeout_seconds": 60},
        },
    )
    assert original.fingerprint != updated.fingerprint
    assert original.scientific_fingerprint == updated.scientific_fingerprint
    # Clustering batch size changes its stochastic fit; it is not an I/O block size.
    changed = store.load_configuration(
        "networks",
        "main",
        document={
            **original.values,
            "clustering": {**original.values["clustering"], "batch_size": 512},
        },
    )
    assert original.scientific_fingerprint != changed.scientific_fingerprint
    values = store.load_configuration("firstlevels", "main").values
    assert values["aggregation_weighting"] == "precision"
    assert scientific_values("firstlevels", values) != scientific_values(
        "firstlevels", {**values, "ar_grid": values["ar_grid"][::-1]}
    )
    assert scientific_values("firstlevels", values) != scientific_values(
        "firstlevels", {**values, "aggregation_weighting": "equal"}
    )


def test_bids_filter_sets_normalize_without_losing_absence_semantics(store):
    from nro.engine.bids import matches_filter

    first = store.load_configuration(
        "microparcellation", "dev", document={"input_filter": {"run": 1, "ses": None}}
    )
    second = store.load_configuration(
        "microparcellation", "dev", document={"input_filter": {"ses": None, "run": ["1", "1"]}}
    )
    assert first.fingerprint == second.fingerprint
    assert matches_filter({"run": "1"}, first.values["input_filter"])
    assert not matches_filter({"run": "1", "ses": "a"}, first.values["input_filter"])


def test_workflow_errors_and_runtime_snapshot_validation(store, tmp_path):
    target = definition_target(store, "workflow", "bad")
    for text in ("clean: main\nclean: other", "clean: missing", "unknown: main", "clean: 5"):
        target.path.write_text(text)
        with pytest.raises(ValueError):
            store.resolve("bad")
        with pytest.raises(ValueError):
            validate_definition(store, target, text)
    runtime = tmp_path / "main_firstlevels.yml"
    values = store.load_configuration("firstlevels", "main").values
    runtime.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="preprocessing_directory"):
        load_runtime_configuration(runtime, "firstlevels")
    values.update(preprocessing_directory="main", preprocessing_aroma=True)
    runtime.write_text(yaml.safe_dump(values))
    assert load_runtime_configuration(runtime, "firstlevels")[1] == values
    with pytest.raises(ValueError, match="Recursive YAML"):
        parse_mapping("x: &x {y: *x}")


@pytest.mark.parametrize("record_full_snapshot", [False, True])
def test_execution_edit_preserves_completed_registry_artifacts(
    store, tmp_path, record_full_snapshot
):
    from nro.orchestration.catalog import module_descriptor
    from nro.orchestration.contracts import InstanceSpec
    from nro.orchestration.manifests import (
        MANIFEST_VERSION,
        assess_registry,
        file_record,
        preview_registry,
    )
    from nro.orchestration.registry import Registry

    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = store.resolve("main")
    registered = registry.register_workflow(workflow)
    config = workflow.configuration("clean")
    output = tmp_path / "result.txt"
    output.write_text("completed science")
    spec = InstanceSpec.create(
        key="clean:" + "f" * 64,
        module="clean",
        project="demo",
        participant="01",
        entities={"space": "T1w", "smoothing": "2"},
        scope="run",
        directory_label="main",
        configuration_lineage_id=registered.lineages["clean"],
        config_fingerprint=config.scientific_fingerprint,
        runtime_config=registry.runtime_config_path(registered, "clean"),
        command=("true",),
        dependencies=(),
        input_paths=(),
        output_root=tmp_path,
        output_prefix=None,
        expected_outputs=(output,),
        resource_class="large",
        processing=module_descriptor("clean").processing_contract(),
    )
    instance_id = registry.register_instances((spec,))[spec.key]
    row = registry.instance_rows()[0]
    contract = spec.instance_contract
    snapshot = deepcopy(config.values)
    if record_full_snapshot:
        snapshot["min_trs"] = 50.0
        contract["configuration"] = configuration_fingerprint("clean", "main", snapshot)
    contract_hash = fingerprint(contract)
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE instances SET artifact_state='fresh', artifact_contract_json=?, artifact_fingerprint=? WHERE id=?",
            (json.dumps(contract), contract_hash, instance_id),
        )
    manifest = Path(row["manifest_path"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": MANIFEST_VERSION,
                "artifact_contract": contract,
                "artifact_fingerprint": contract_hash,
                "revision_fingerprint": row["revision_fingerprint"],
                "configuration": {
                    "id": "main",
                    "resolved": snapshot,
                    "fingerprint": configuration_fingerprint("clean", "main", snapshot),
                },
                "inputs": [],
                "upstream": [],
                "public_outputs": [file_record(output)],
            }
        )
    )
    assert preview_registry(registry)[instance_id][0] == "fresh"
    assert assess_registry(registry)[instance_id][0] == "fresh"
    stamp = output.stat().st_mtime_ns
    external_main = store.configs / "clean" / "main_clean.yml"
    external_main.write_text(yaml.safe_dump({"verbose": True}))
    updated = store.resolve("main")
    selected = registry.register_workflow(updated)
    assert selected.revision != registered.revision
    assert selected.lineages == registered.lineages
    replacement = spec.evolve(
        config_fingerprint=updated.configuration("clean").scientific_fingerprint,
        runtime_config=registry.runtime_config_path(selected, "clean"),
    )
    registry.register_instances((replacement,))
    assert registry.instance_rows()[0]["artifact_state"] == "fresh"
    assert preview_registry(registry)[instance_id][0] == "fresh"
    assert assess_registry(registry)[instance_id][0] == "fresh"
    assert output.stat().st_mtime_ns == stamp
    changed = spec.evolve(
        config_fingerprint=store.load_configuration(
            "clean", "main", document={**config.values, "nuisance_variance_explained": 0.9}
        ).scientific_fingerprint
    )
    registry.register_instances((changed,))
    assert assess_registry(registry)[instance_id][0] == "stale"
