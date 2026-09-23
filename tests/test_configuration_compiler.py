"""Shared configuration validation, normalization, and scientific freshness."""

import json
import shutil
from copy import deepcopy

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
    return ConfigStore(tmp_path / "store")


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
        (
            "networks",
            {"connectivity_source": "dynconn", "parcellation_strategy": "oslom"},
            "microparcellation",
        ),
        ("microparcellation", {"mask": 42}, "mask"),
        ("microparcellation", {"input_filter": {"task": False}}, "task"),
        ("firstlevels", {"ar_grid": [0, 0]}, "ar_grid"),
        ("firstlevels", {"ar_grid": [1]}, "ar_grid"),
        ("firstlevels", {"aggregation_weighting": "unknown"}, "aggregation_weighting"),
        ("firstlevels", {"low_pass": 0.1}, "low_pass"),
        ("func", {"bbregister_dof": 5}, "bbregister_dof"),
        ("func", {"output_spaces": []}, "output_spaces"),
        ("func", {"fsaverage_template": "fsaverage"}, "fsaverage_template"),
        ("clean", {"space": "T1w"}, "space"),
        ("clean", {"smoothing": 4}, "smoothing"),
        ("microparcellation", {"output_dir": "/tmp/other"}, "output_dir"),
        ("dynconn", {"prefix": "custom"}, "prefix"),
        ("networks", {"output_dir": "/tmp/other"}, "output_dir"),
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


def test_scientific_fingerprint_tolerates_safe_schema_evolution(store):
    current = store.load_configuration("dynconn", "main").values
    historical = deepcopy(current)
    historical["retired_setting"] = "no longer scientific"
    historical["low_rank_options"].pop("power_iterations")
    historical["low_rank_options"]["retired_nested_setting"] = 1

    assert configuration_fingerprint(
        "dynconn", "main", historical, scientific=True
    ) == configuration_fingerprint("dynconn", "main", current, scientific=True)

    changed = deepcopy(current)
    changed["low_rank_options"]["power_iterations"] += 1
    assert configuration_fingerprint(
        "dynconn", "main", historical, scientific=True
    ) != configuration_fingerprint("dynconn", "main", changed, scientific=True)


def test_func_classifier_vocabulary_ignores_unselected_backend_settings(store):
    values = store.load_configuration("func", "main").values
    current = scientific_values("func", values)
    assert current["ica_classifier"] == "none"
    assert "ica_regression" not in current
    assert "ica_aroma_cmd" not in current
    assert "cicada_cmd" not in current
    assert "cicada_tolerance" not in current
    assert "cicada_smoothing_retention_mode" not in current

    aroma = scientific_values("func", {**values, "ica_classifier": "ica_aroma"})
    assert aroma["ica_classifier"] == "ica_aroma"
    assert aroma["ica_regression"] == "aggressive"
    assert "cicada_cmd" not in aroma
    cicada = scientific_values("func", {**values, "ica_classifier": "cicada"})
    assert cicada["ica_classifier"] == "cicada"
    assert cicada["ica_regression"] == "aggressive"
    assert "ica_aroma_cmd" not in cicada
    no_classifier = {**values, "ica_classifier": "none"}
    assert scientific_values("func", no_classifier)["ica_classifier"] == "none"
    assert scientific_values("func", no_classifier) == scientific_values(
        "func",
        {
            **no_classifier,
            "ica_regression": "nonaggressive",
            "ica_aroma_cmd": "/unused/ICA_AROMA.py",
        },
    )


def test_networks_scientific_vocabulary_ignores_unselected_source_and_estimators(store):
    values = store.load_configuration("networks", "main").values
    micro_ica = scientific_values("networks", values)
    assert "connectivity" in micro_ica
    assert "feature_reduction" in micro_ica
    assert "ica" in micro_ica
    assert "clustering" not in micro_ica
    assert "oslom" not in micro_ica
    assert "consensus" not in micro_ica

    dynconn_ica = scientific_values("networks", {**values, "connectivity_source": "dynconn"})
    assert "connectivity" not in dynconn_ica
    changed_unused = {
        **values,
        "connectivity_source": "dynconn",
        "connectivity": {**values["connectivity"], "minimum_weight": 0.5},
    }
    assert scientific_values("networks", changed_unused) == dynconn_ica

    oslom = scientific_values("networks", {**values, "parcellation_strategy": "oslom"})
    assert "feature_reduction" not in oslom
    assert "ica" not in oslom
    assert "clustering" not in oslom
    assert "oslom" in oslom
    assert "consensus" in oslom


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
    clustering = {**original.values, "parcellation_strategy": "clustering"}
    changed_clustering = {
        **clustering,
        "clustering": {**original.values["clustering"], "batch_size": 512},
    }
    assert scientific_values("networks", clustering) != scientific_values(
        "networks", changed_clustering
    )
    values = store.load_configuration("firstlevels", "main").values
    assert values["input_filter"] == {}
    assert values["aggregation_weighting"] == "precision"
    assert scientific_values("firstlevels", values) != scientific_values(
        "firstlevels", {**values, "ar_grid": values["ar_grid"][::-1]}
    )
    assert scientific_values("firstlevels", values) != scientific_values(
        "firstlevels", {**values, "aggregation_weighting": "equal"}
    )
    dynconn = store.load_configuration("dynconn", "main").values
    changed_options = {
        **dynconn,
        "low_rank_options": {
            **dynconn["low_rank_options"],
            "dimensions": dynconn["low_rank_options"]["dimensions"] + 1,
        },
    }
    assert dynconn["low_rank"] is True
    assert dynconn["weighting"] == "precision"
    assert dynconn["low_rank_options"]["dimensions"] == 1000
    assert "random_seed" not in dynconn["low_rank_options"]
    assert scientific_values("dynconn", dynconn) != scientific_values("dynconn", changed_options)
    assert scientific_values("dynconn", dynconn) != scientific_values(
        "dynconn", {**dynconn, "weighting": "equal"}
    )
    assert scientific_values("dynconn", {**dynconn, "low_rank": False}) == scientific_values(
        "dynconn", {**changed_options, "low_rank": False}
    )
    microparcellation = store.load_configuration("microparcellation", "main").values
    assert microparcellation["connectivity"]["weighting"] == "precision"
    equal_microparcellation = {
        **microparcellation,
        "connectivity": {
            **microparcellation["connectivity"],
            "weighting": "equal",
        },
    }
    assert scientific_values("microparcellation", microparcellation) != scientific_values(
        "microparcellation", equal_microparcellation
    )


@pytest.mark.parametrize("kind", SCHEMAS)
def test_every_module_uses_execution_only_overwrite(store, kind):
    config = store.load_configuration(kind, "main")
    assert config.values["overwrite"] is False
    changed = store.load_configuration(
        kind,
        "main",
        document={**config.values, "overwrite": True},
    )
    assert changed.fingerprint != config.fingerprint
    assert changed.scientific_fingerprint == config.scientific_fingerprint


@pytest.mark.parametrize("kind", ("anat", "func", "clean"))
def test_legacy_execution_and_flat_container_fields_are_rejected(store, kind):
    with pytest.raises(ValueError, match="force"):
        store.load_configuration(kind, "legacy", document={"force": False})
    with pytest.raises(ValueError, match="container_engine"):
        store.load_configuration(kind, "legacy", document={"container_engine": "apptainer"})


@pytest.mark.parametrize("kind", ("anat", "func", "clean"))
def test_nested_container_changes_remain_scientific(store, kind):
    config = store.load_configuration(kind, "main")
    changed = store.load_configuration(
        kind,
        "main",
        document={
            **config.values,
            "container": {**config.values["container"], "cleanenv": False},
        },
    )
    assert changed.scientific_fingerprint != config.scientific_fingerprint


def test_marss_cutoff_is_scientific_only_when_auto_mode_uses_it(store):
    defaults = store.load_configuration("func", "main")
    changed_auto = store.load_configuration(
        "func",
        "main",
        document={**defaults.values, "marss_min_multiband_factor": 5},
    )
    assert defaults.scientific_fingerprint != changed_auto.scientific_fingerprint

    diagnose = store.load_configuration(
        "func",
        "main",
        document={**defaults.values, "marss_mode": "diagnose"},
    )
    changed_diagnose = store.load_configuration(
        "func",
        "main",
        document={**diagnose.values, "marss_min_multiband_factor": 5},
    )
    assert diagnose.fingerprint != changed_diagnose.fingerprint
    assert diagnose.scientific_fingerprint == changed_diagnose.scientific_fingerprint


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
    with pytest.raises(ValueError, match="anat_directory"):
        load_runtime_configuration(runtime, "firstlevels")
    values.update(
        anat_directory="main",
        func_directory="main",
    )
    runtime.write_text(yaml.safe_dump(values))
    assert load_runtime_configuration(runtime, "firstlevels")[1] == values
    with pytest.raises(ValueError, match="Recursive YAML"):
        parse_mapping("x: &x {y: *x}")


@pytest.mark.parametrize("record_full_snapshot", [False, True])
def test_execution_edit_preserves_completed_registry_artifacts(
    store, tmp_path, record_full_snapshot
):
    from nro.orchestration.artifact_records import file_record
    from nro.orchestration.catalog import module_descriptor
    from nro.orchestration.contracts import WorkItemSpec
    from nro.orchestration.manifests import assess_registry, preview_registry
    from nro.orchestration.registry import Registry

    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = store.resolve("main")
    registered = registry.register_workflow(workflow)
    config = workflow.configuration("clean")
    output = tmp_path / "result.txt"
    output.write_text("completed science")
    spec = WorkItemSpec.create(
        key="clean:" + "f" * 64,
        module="clean",
        project="demo",
        participant="01",
        entities={"space": "T1w", "smoothing": "2"},
        scope="run",
        directory_label=registered.directory_for("clean"),
        module_lineage_id=registered.lineages["clean"],
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
    work_item_id = registry.register_work_items((spec,))[spec.key]
    row = registry.work_item_rows()[0]
    contract = spec.work_item_contract
    snapshot = deepcopy(config.values)
    if record_full_snapshot:
        snapshot["min_trs"] = 50.0
        contract["configuration"] = configuration_fingerprint("clean", "main", snapshot)
    contract_hash = fingerprint(contract)
    with registry.connection(write=True) as db:
        completed = file_record(output)
        lineage_fingerprint = db.execute(
            "SELECT lineage_fingerprint FROM module_lineages WHERE id=?",
            (row["module_lineage_id"],),
        ).fetchone()[0]
        db.execute(
            "UPDATE work_items SET artifact_state='fresh', artifact_contract_json=?, artifact_fingerprint=? WHERE id=?",
            (json.dumps(contract), contract_hash, work_item_id),
        )
        db.execute(
            "UPDATE module_lineages SET resolved_yaml=?,config_fingerprint=? WHERE id=?",
            (
                yaml.safe_dump(snapshot),
                configuration_fingerprint("clean", "main", snapshot),
                row["module_lineage_id"],
            ),
        )
        db.execute(
            """INSERT INTO completions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                work_item_id,
                None,
                1,
                "2026-01-01T00:00:00+00:00",
                row["revision_fingerprint"],
                json.dumps(contract),
                contract_hash,
                "main",
                configuration_fingerprint("clean", "main", snapshot),
                lineage_fingerprint,
                yaml.safe_dump(snapshot),
                "{}",
                "[]",
            ),
        )
        db.execute(
            """INSERT INTO artifacts(
                   work_item_id,attempt_id,direction,path,size,mtime_ns,
                   digest_algorithm,digest,metadata_json
               ) VALUES (?,NULL,'output',?,?,?,?,?, '{}')""",
            (
                work_item_id,
                completed["path"],
                completed["size"],
                completed["mtime_ns"],
                "sha256",
                completed["sha256"],
            ),
        )
    assert preview_registry(registry)[work_item_id][0] == "fresh"
    assert assess_registry(registry)[work_item_id][0] == "fresh"
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
    registry.register_work_items((replacement,))
    assert registry.work_item_rows()[0]["artifact_state"] == "fresh"
    assert preview_registry(registry)[work_item_id][0] == "fresh"
    assert assess_registry(registry)[work_item_id][0] == "fresh"
    assert output.stat().st_mtime_ns == stamp
    changed = spec.evolve(
        config_fingerprint=store.load_configuration(
            "clean", "main", document={**config.values, "nuisance_variance_explained": 0.9}
        ).scientific_fingerprint
    )
    registry.register_work_items((changed,))
    assert assess_registry(registry)[work_item_id][0] == "stale"
