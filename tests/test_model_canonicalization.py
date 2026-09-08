"""Compiled model equivalence and the boundaries of freshness normalization."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from nro.configuration.store import ConfigStore, fingerprint
from nro.firstlevels.compiler import compile_model
from nro.firstlevels.contract import _matches_definition, definition_fingerprint
from nro.firstlevels.task_models import scientific_model
from nro.orchestration.catalog import canonical_contract
from nro.orchestration.manifests import _certificate_matches_contract


def _source():
    return {"conditions": "trial_type", "contrasts": {"E": {"E": 1}, "EvH": {"E": "1/2", "H": "-1/2"}}}


def test_canonical_model_is_cached_data_free_and_mutation_safe(monkeypatch):
    from nro.firstlevels import task_models

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Canonicalization must not read files")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    task_models._canonical_model.cache_clear()
    source = _source()
    result = scientific_model(source)
    cached = task_models._canonical_model.cache_info()
    assert scientific_model({**source, "model_set": ["dev"], "description": "A note"}) == result
    assert task_models._canonical_model.cache_info().hits == cached.hits + 1
    assert scientific_model(result) == result
    result["statsmodels"]["Contrasts"].clear()
    assert len(scientific_model(source)["statsmodels"]["Contrasts"]) == 2
    assert source == _source()


@pytest.mark.parametrize("form", ["qualified", "explicit_defaults", "advanced", "mapping_order"])
def test_authoring_forms_compile_identically(form):
    source = _source()
    variant = deepcopy(source)
    if form == "qualified":
        variant["contrasts"] = {"E": {"trial_type.E": 1.0}, "EvH": {"trial_type.E": .5, "trial_type.H": -.5}}
    elif form == "explicit_defaults":
        variant.update(hrf="spm", hrf_overrides={}, aggregation={"weighting": "equal"})
    elif form == "advanced":
        variant = scientific_model(source)
    else:
        variant["contrasts"] = dict(reversed(list(variant["contrasts"].items())))
    assert fingerprint(scientific_model(source)) == fingerprint(scientific_model(variant))
    config = ConfigStore().load_configuration("firstlevels", "main").values
    assert compile_model(source, "task/main", config) == compile_model(variant, "task/main", config)


@pytest.mark.parametrize("field,value", [
    ("hrf", "glover"), ("hrf_overrides", {"trial_type.E": None}),
    ("aggregation", {"weighting": "precision"}),
    ("contrasts", {"E": {"E": 2}, "EvH": {"E": 1, "H": -1}}),
])
def test_scientific_changes_remain_distinct(field, value):
    source = _source()
    assert scientific_model(source) != scientific_model({**source, field: value})


def test_explicit_sequence_order_is_preserved():
    source = {"predictors": ["a", "b"], "transformations": [
        {"Name": "Demean", "Input": "a"}, {"Name": "Scale", "Input": "a"}],
        "contrasts": {"a": {"a": 1}, "b": {"b": 1}}}
    for key in ("predictors", "transformations"):
        assert scientific_model(source) != scientific_model({**source, key: source[key][::-1]})
    canonical = scientific_model(source)
    reordered = deepcopy(canonical)
    reordered["statsmodels"]["Contrasts"].reverse()
    assert scientific_model(reordered) != canonical
    explicit = deepcopy(source)
    explicit["transformations"] = [
        {"Name": "Demean", "Input": ["a"], "Groupby": []},
        {"Name": "Scale", "Input": ["a"], "Groupby": [], "Demean": True, "Rescale": True}]
    assert scientific_model(explicit) == canonical


def test_recorded_contracts_normalize_without_loading_current_model():
    contract = {"module": "firstlevels", "processing": {"task_model": _source(), "other_policy": "unchanged"}}
    certificate = {"artifact_contract": contract, "artifact_fingerprint": fingerprint(contract)}
    expected = fingerprint(canonical_contract(contract))
    assert certificate["artifact_fingerprint"] != expected
    assert _certificate_matches_contract(certificate, expected)
    changed = deepcopy(contract)
    changed["processing"]["task_model"]["hrf"] = "glover"
    assert not _certificate_matches_contract(certificate, fingerprint(canonical_contract(changed)))
    certificate["artifact_fingerprint"] = "corrupt"
    assert not _certificate_matches_contract(certificate, expected)
    assert not _certificate_matches_contract({}, expected)


def test_registry_preview_assessment_and_registration_accept_equivalent_recorded_model(tmp_path, monkeypatch):
    import yaml
    from nro.firstlevels import task_models
    from nro.orchestration.catalog import module_descriptor
    from nro.orchestration.contracts import InstanceSpec
    from nro.orchestration.manifests import MANIFEST_VERSION, assess_registry, file_record, preview_registry
    from nro.orchestration.registry import Registry

    model_path = tmp_path / "model.yml"
    model_path.write_text(yaml.safe_dump(_source()))
    monkeypatch.setattr(task_models, "model_path", lambda identifier: model_path)
    (tmp_path / "bids/demo/sub-01").mkdir(parents=True)
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    output = tmp_path / "output.txt"
    output.write_text("synthetic completed derivative")
    entities = {"task": "task", "model": "main", "space": "T1w", "smoothing": "0"}
    spec = InstanceSpec.create(
        key="firstlevels:" + "a" * 64, module="firstlevels", project="demo", participant="01",
        entities=entities, scope="subject", configuration_lineage_id=registered.lineages["firstlevels"],
        config_fingerprint=workflow.configuration("firstlevels").fingerprint, directory_label="main",
        runtime_config=registry.runtime_config_path(registered, "firstlevels"), command=("true",),
        dependencies=(), input_paths=(), output_root=tmp_path, output_prefix=None, resource_class="large",
        expected_outputs=(output,), processing=module_descriptor("firstlevels").processing_for(entities))
    instance_id = registry.register_instances((spec,))[spec.key]
    row = registry.instance_rows()[0]
    recorded = spec.instance_contract
    recorded["processing"]["task_model"] = _source()
    old_fingerprint = fingerprint(recorded)

    def record_source_contract():
        with registry.connection(write=True) as db:
            db.execute("UPDATE instances SET artifact_contract_json=?, artifact_fingerprint=?, artifact_state='fresh' WHERE id=?",
                       (json.dumps(recorded), old_fingerprint, instance_id))

    record_source_contract()
    certificate = Path(row["manifest_path"])
    certificate.parent.mkdir(parents=True, exist_ok=True)
    certificate.write_text(json.dumps({"manifest_version": MANIFEST_VERSION,
        "artifact_contract": recorded, "artifact_fingerprint": old_fingerprint,
        "revision_fingerprint": row["revision_fingerprint"], "inputs": [], "upstream": [],
        "public_outputs": [file_record(output)]}))
    stamp = output.stat().st_mtime_ns
    assert preview_registry(registry)[instance_id][0] == "fresh"
    assert registry.instance_rows()[0]["artifact_fingerprint"] == old_fingerprint
    assert assess_registry(registry)[instance_id][0] == "fresh"
    assert registry.instance_rows()[0]["artifact_fingerprint"] == spec.contract_fingerprint
    record_source_contract()
    registry.register_instances((spec,))
    assert registry.instance_rows()[0]["artifact_state"] == "fresh"
    assert assess_registry(registry)[instance_id][0] == "fresh"
    assert output.stat().st_mtime_ns == stamp
    model_path.write_text(yaml.safe_dump({**_source(), "hrf": "glover"}))
    assert preview_registry(registry)[instance_id][0] == "stale"
    assert assess_registry(registry)[instance_id][0] == "stale"


@pytest.mark.parametrize("runs", [None, ["run-01", "run-02"]])
def test_saved_compiled_definitions_preserve_run_set_and_configuration(runs):
    config = ConfigStore().load_configuration("firstlevels", "main").values
    expected = {"model": compile_model(_source(), "task/main", config), "config": config}
    if runs is not None:
        expected["runs"] = runs
    recorded = deepcopy(expected)
    factor = recorded["model"]["Nodes"][0]["Transformations"]["Instructions"][0]
    factor["Input"] = "trial_type"
    del factor["Constraint"]
    manifest = {"model_document": recorded["model"], "configuration": config,
                "definition_fingerprint": definition_fingerprint(recorded)}
    assert _matches_definition(manifest, expected)
    if runs is not None:
        assert not _matches_definition(manifest, {**expected, "runs": ["run-01"]})
    assert not _matches_definition(manifest, {**expected, "config": {**config, "noise_model": "ols" if config["noise_model"] == "ar1" else "ar1"}})
    changed = deepcopy(expected)
    changed["model"]["Nodes"][0]["Contrasts"][0]["Weights"] = [2]
    assert not _matches_definition(manifest, changed)
    assert json.loads(json.dumps(manifest)) == manifest
