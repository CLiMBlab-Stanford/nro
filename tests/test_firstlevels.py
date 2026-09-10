"""Statistical identities, missing conditions and firstlevels publication."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from nro.configuration.store import ConfigStore
from nro.engine.bids import discover_raw_runs
from nro.modules.firstlevels.compiler import compile_model, realize_run_node
from nro.modules.firstlevels.contract import validate_completion
from nro.modules.firstlevels.design import build_design as fit_design
from nro.modules.firstlevels.estimation import meta_records
from nro.modules.firstlevels.io import load_fit, save_fit
from nro.modules.firstlevels.models import validate_model
from nro.modules.firstlevels.module import run_module
from nro.modules.firstlevels.statistics import (
    Estimate,
    aggregate,
    fit_glm,
    linear_combination,
)
from nro.modules.firstlevels.task_models import load_task_model, register_model


def load_model(identifier, config=None):
    """Compile the shipped task fixture for low-level estimator tests."""
    return compile_model(
        load_task_model(identifier),
        identifier,
        config or ConfigStore().load_configuration("firstlevels", "main").values,
    )


def build_design(node, events, confounds, tr, config):
    """Resolve a test run's metadata before constructing its numerical design."""
    return fit_design(
        realize_run_node(node, events, confounds, config), events, confounds, tr, config
    )


def _fits():
    rng = np.random.default_rng(109)
    result = {}
    for index, n in enumerate((90, 110, 130)):
        design = np.column_stack((np.ones(n), rng.normal(size=(n, 3))))
        # Correlated predictors make dropping coefficient covariance detectable.
        design[:, 2] += 0.7 * design[:, 1]
        y = design @ rng.normal(size=(4, 9)) + rng.normal(size=(n, 9)) * (index + 1)
        result[str(index)] = fit_glm(y, design, retained=np.ones(n, bool), ar_grid=np.array([0.0]))
    return result


@pytest.mark.parametrize("weights", [[1 / 3] * 3, [0.1, 0.3, 0.6]])
def test_general_contrast_commutes_with_common_run_aggregation(weights):
    fits = _fits()
    c = np.array([0, 1.2, -0.7, 0.25])
    per_run = [Estimate({key: c}) for key in fits]
    contrast_first = linear_combination(per_run, weights).evaluate(fits)
    pooled_conditions = [
        linear_combination([Estimate({key: np.eye(4)[j]}) for key in fits], weights)
        for j in range(4)
    ]
    aggregate_first = linear_combination(pooled_conditions, list(c)).evaluate(fits)
    for statistic in ("effect", "variance", "t", "dof"):
        assert_allclose(contrast_first[statistic], aggregate_first[statistic], rtol=1e-12)
    expected_variance = sum(w * w * fit.variance(c) for w, fit in zip(weights, fits.values()))
    expected_dof = expected_variance**2 / sum(
        (w * w * fit.variance(c)) ** 2 / fit.dof for w, fit in zip(weights, fits.values())
    )
    assert_allclose(aggregate_first["variance"], expected_variance)
    assert_allclose(aggregate_first["dof"], expected_dof)


def test_missing_conditions_and_overlapping_runs_preserve_covariance():
    fits = _fits()
    a = aggregate([Estimate({key: np.array([0, 1, 0, 0])}) for key in ("0", "1")], fits)
    b = aggregate([Estimate({key: np.array([0, 0, 1, 0])}) for key in ("1", "2")], fits)
    result = linear_combination([a, b], [1, -1]).evaluate(fits)
    cross = fits["1"].covariance[0, 1, 2] * fits["1"].residual_variance / 4
    independent = a.evaluate(fits)["variance"] + b.evaluate(fits)["variance"]
    assert_allclose(result["variance"], independent - 2 * cross, rtol=1e-6)
    assert not np.allclose(result["variance"], independent)


def test_disjoint_conditions_recover_welch_satterthwaite():
    fits = _fits()
    a = Estimate({"0": np.array([0, 1, 0, 0])})
    b = Estimate({"2": np.array([0, 0, 1, 0])})
    result = linear_combination([a, b], [1, -1]).evaluate(fits)
    va, vb = a.evaluate(fits)["variance"], b.evaluate(fits)["variance"]
    assert_allclose(result["variance"], va + vb)
    assert_allclose(result["dof"], (va + vb) ** 2 / (va**2 / fits["0"].dof + vb**2 / fits["2"].dof))


def test_duplicate_run_pool_is_rejected():
    fits = _fits()
    estimate = Estimate({"0": np.ones(4)})
    with pytest.raises(ValueError, match="more than once"):
        aggregate([estimate, estimate], fits)


def test_gap_aware_gls_matches_direct_matrix_calculation_and_compact_roundtrip(tmp_path):
    rng = np.random.default_rng(91)
    n, rho = 75, 0.6
    retained = np.ones(n, bool)
    retained[[2, 3, 5, 10, 18, 19]] = False
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    y = rng.normal(size=(n, 7))
    fit = fit_glm(y, x[retained], retained=retained, ar_grid=np.array([rho]), block_size=3)
    indices = np.flatnonzero(retained)
    covariance = rho ** np.abs(indices[:, None] - indices[None, :])
    inverse = np.linalg.inv(covariance)
    coefficient_covariance = np.linalg.inv(x[retained].T @ inverse @ x[retained])
    beta = coefficient_covariance @ x[retained].T @ inverse @ y[retained]
    residual = y[retained] - x[retained] @ beta
    scale = np.einsum("tv,ts,sv->v", residual, inverse, residual) / (retained.sum() - 2)
    assert_allclose(fit.beta, beta, rtol=1e-6, atol=1e-7)
    assert_allclose(fit.residual_variance, scale, rtol=1e-6)
    assert_allclose(fit.covariance[0], coefficient_covariance)
    record, paths = save_fit(tmp_path / "fit", fit)
    assert len(paths) == 5
    restored = load_fit(record, slice(2, 5))
    assert_allclose(restored.variance(np.array([0, 1])), fit.variance(np.array([0, 1]))[2:5])


def test_gls_preserves_unfiltered_observations_and_counts_retained_frames():
    rng = np.random.default_rng(23)
    n = 100
    retained = np.ones(n, bool)
    retained[::11] = False
    times = np.arange(n)
    x = np.column_stack((np.ones(n), np.sin(2 * np.pi * 0.3 * times)))
    y = rng.normal(size=(n, 4)) + 20 * x[:, 1, None]
    fitted = fit_glm(y, x[retained], retained=retained, ar_grid=np.array([0.0]))
    assert fitted.dof == retained.sum() - 2
    expected = np.linalg.lstsq(x[retained], y[retained], rcond=None)[0]
    assert_allclose(fitted.beta, expected, rtol=1e-6, atol=1e-7)
    residual = y[retained] - x[retained] @ expected
    assert_allclose(fitted.residual_variance, (residual**2).sum(axis=0) / fitted.dof, rtol=1e-6)


def test_nested_summaries_keep_original_run_variance_contributions():
    fits = _fits()
    effects = [Estimate({key: np.array([0, 1, 0, 0])}) for key in fits]
    session = aggregate(effects[:2], fits)
    nested = aggregate([session, effects[2]], fits).evaluate(fits)
    direct = linear_combination(effects, [0.25, 0.25, 0.5]).evaluate(fits)
    for key in direct:
        assert_allclose(nested[key], direct[key])


def test_precision_weights_need_not_commute_and_ignore_zero_variance_locations():
    fits = _fits()
    a = [Estimate({key: np.array([0, 1, 0, 0])}) for key in fits]
    b = [Estimate({key: np.array([0, 0, 1, 0])}) for key in fits]
    pooled = linear_combination(
        [aggregate(a, fits, weighting="precision"), aggregate(b, fits, weighting="precision")],
        [1, -1],
    ).evaluate(fits)
    contrast = aggregate(
        [linear_combination([x, y], [1, -1]) for x, y in zip(a, b)], fits, weighting="precision"
    ).evaluate(fits)
    assert not np.allclose(pooled["effect"], contrast["effect"])
    for fit in fits.values():
        fit.residual_variance[0] = 0
    invalid = aggregate(a, fits, weighting="precision").evaluate(fits)
    assert np.isnan(invalid["t"][0])
    assert np.isfinite(invalid["t"][1:]).all()


def test_task_coefficients_remain_nuisance_adjusted_with_rank_protection():
    from nro.modules.firstlevels.models import event_design

    rng = np.random.default_rng(63)
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config.update(nuisance_variance_explained=1.0)
    node = load_model("langlocSN/main", config)["Nodes"][0]
    events = pd.DataFrame(
        {"onset": [5, 35, 65, 95], "duration": [6] * 4, "trial_type": ["S", "N"] * 2}
    )
    confounds = pd.DataFrame({"csf": rng.normal(size=130), "motion_outlier00": [1] + [0] * 129})
    raw_node = realize_run_node(node, events, confounds, config)
    raw_node["Model"]["X"] = ["trial_type.S", "trial_type.N", "csf", 1]
    raw = event_design(raw_node, events, confounds, 1)
    confounds["csf"] += 2 * raw["trial_type.S"]
    raw = event_design(raw_node, events, confounds, 1)
    data = raw.to_numpy() @ np.array([2, -1, 3, 100])[:, None] + rng.normal(size=(130, 10))
    design = build_design(node, events, confounds, 1, config)
    fit = fit_glm(data, design.matrix, retained=design.retained, ar_grid=np.array([0.0]))
    assert design.metadata["ObservationDimension"] == design.retained.sum()
    assert design.metadata["TemporalFiltering"] == "none"
    expected = np.linalg.lstsq(raw.to_numpy()[design.retained], data[design.retained], rcond=None)[
        0
    ]
    indices = [list(raw).index(name) for name in design.names]
    assert_allclose(design.coefficient_map @ fit.beta, expected[indices], rtol=1e-5, atol=1e-5)
    assert design.metadata["NuisancePCs"] == 1
    config["minimum_temporal_rank"] = 200
    node = load_model("langlocSN/main", config)["Nodes"][0]
    assert build_design(node, events, confounds, 1, config).metadata["NuisancePCs"] == 0


@pytest.mark.parametrize("estimated", [False, True])
def test_gap_aware_ar1_null_calibration(estimated):
    from scipy.linalg import cholesky
    from scipy.stats import t

    rng = np.random.default_rng(655)
    n = 160
    retained = np.ones(n, bool)
    retained[::13] = False
    times = np.arange(n)
    x = np.column_stack([np.sin(times * 0.23), rng.normal(size=n)])
    y = cholesky(0.6 ** abs(times[:, None] - times[None, :]), lower=True) @ rng.normal(
        size=(n, 3000)
    )
    grid = np.array([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]) if estimated else np.array([0.6])
    fit = fit_glm(y, x[retained], retained=retained, ar_grid=grid)
    values = Estimate({"r": np.array([1, 0])}).evaluate({"r": fit})
    rejection = np.mean(2 * t.sf(abs(values["t"]), values["dof"]) < 0.05)
    # Estimated AR inference is conditional, not exact. Its wider bound guards
    # against large regressions; the observed inflation is reported in methods.
    assert 0.03 < rejection < (0.08 if estimated else 0.07)


def test_ols_null_calibration():
    from scipy.stats import t

    rng = np.random.default_rng(433)
    n = 80
    x = np.column_stack((np.ones(n), rng.normal(size=n)))
    fit = fit_glm(rng.normal(size=(n, 2000)), x, retained=np.ones(n, bool), ar_grid=np.array([0.0]))
    result = Estimate({"r": np.array([0, 1])}).evaluate({"r": fit})
    rejected = np.mean(2 * t.sf(np.abs(result["t"]), result["dof"]) < 0.05)
    assert 0.03 < rejected < 0.07


@pytest.mark.parametrize("change", ["dataset", "meta-glm", "formula", "unknown-transform"])
def test_unsupported_statsmodels_fail_before_execution(change):
    model = load_model("langlocSN/main")
    if change == "dataset":
        model["Nodes"][-1]["Level"] = "Dataset"
    elif change == "meta-glm":
        model["Nodes"][-1]["Model"]["Type"] = "glm"
    elif change == "formula":
        model["Nodes"][0]["Model"]["Formula"] = "1 + trial_type"
    else:
        model["Nodes"][0]["Transformations"]["Instructions"][0]["Name"] = "MadeUp"
    with pytest.raises(ValueError):
        validate_model(model, task="langlocSN")


@pytest.mark.parametrize("option", ["HighPassFilterCutoffHz", "LowPassFilterCutoffHz"])
def test_statsmodels_cannot_enable_temporal_filtering(option):
    model = load_model("langlocSN/main")
    model["Nodes"][0]["Model"]["Options"] = {option: 0.1}
    with pytest.raises(ValueError, match="Unsupported Model fields"):
        validate_model(model, task="langlocSN")


@pytest.mark.parametrize("option", ["high_pass", "low_pass"])
def test_filter_configuration_is_removed_and_rejected(option):
    config = ConfigStore().load_configuration("firstlevels", "main").values
    assert option not in config
    config[option] = None
    with pytest.raises(ValueError, match="does not support temporal filtering"):
        fit_design(
            load_model("langlocSN/main")["Nodes"][0], pd.DataFrame(), pd.DataFrame(), 1, config
        )


def test_contract_declares_unfiltered_fit():
    from nro.modules.firstlevels.contract import firstlevels_output_contract

    assert firstlevels_output_contract()["temporal_filtering"] == "none"


def test_registration_unique_task_variant(tmp_path):
    import yaml

    source = tmp_path / "model.yml"
    source.write_text(yaml.safe_dump(load_task_model("langlocSN/main")))
    destination = register_model("langlocSN/alternative", source, root=tmp_path / "models")
    assert destination == tmp_path / "models/langlocSN/alternative.yml"
    with pytest.raises(FileExistsError):
        register_model("langlocSN/alternative", source, root=tmp_path / "models")


def _synthetic_project(tmp_path, domain):
    root = tmp_path / "demo"
    rng = np.random.default_rng(19)
    for i in (1, 2):
        stem = f"sub-01_ses-01_task-langlocSN_run-{i:02}"
        raw = root / "sub-01/ses-01/func"
        functional = root / "derivatives/preprocessing/main/sub-01/ses-01/func"
        raw.mkdir(parents=True, exist_ok=True)
        functional.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.ones((2, 2, 2, 120), np.float32), np.eye(4)),
            raw / f"{stem}_bold.nii.gz",
        )
        (raw / f"{stem}_bold.json").write_text('{"RepetitionTime": 1}')
        # Run 2 has S but no N.
        pd.DataFrame(
            {
                "onset": [5, 30, 60, 90],
                "duration": [5] * 4,
                "trial_type": ["S", "N", "S", "N"] if i == 1 else ["S"] * 4,
            }
        ).to_csv(raw / f"{stem}_events.tsv", sep="\t", index=False)
        pd.DataFrame(
            {
                "csf": rng.normal(size=120),
                "white_matter": rng.normal(size=120),
                "rot_x": rng.normal(size=120),
                "trans_x": rng.normal(size=120),
                "motion_outlier00": [1] + [0] * 119,
            }
        ).to_csv(functional / f"{stem}_desc-confounds_timeseries.tsv", sep="\t", index=False)
        if domain == "volume":
            path = functional / f"{stem}_space-T1w_desc-preproc_bold.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    rng.normal(size=(2, 2, 2, 120)).astype(np.float32) + 100, np.eye(4)
                ),
                path,
            )
            sidecar = path.with_name(path.name.removesuffix(".nii.gz") + ".json")
            sidecar.write_text('{"RepetitionTime": 1}')
        else:
            for hemi in ("L", "R"):
                path = functional / f"{stem}_space-fsnative_hemi-{hemi}_desc-preproc_bold.func.gii"
                nib.save(
                    nib.GiftiImage(
                        darrays=[
                            nib.gifti.GiftiDataArray(rng.normal(size=8).astype(np.float32))
                            for _ in range(120)
                        ]
                    ),
                    path,
                )
                path.with_suffix(".json").write_text('{"RepetitionTime": 1}')
    return root


@pytest.mark.parametrize("domain,smoothing", [("volume", 0), ("volume", 2), ("surface", 0)])
def test_module_outputs_omissions_and_resumption(tmp_path, domain, smoothing):
    root = _synthetic_project(tmp_path, domain)
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config.update(noise_model="ols")
    model = load_task_model("langlocSN/main")
    kwargs = dict(
        runs=discover_raw_runs(root / "sub-01"),
        participant="01",
        project_root=root,
        preprocessing_id="main",
        config_id="main",
        model_id="langlocSN/main",
        model=model,
        config=config,
        space="T1w" if domain == "volume" else "fsnative",
        smoothing=smoothing,
        work_root=tmp_path / "work",
    )
    path = run_module(**kwargs)
    assert validate_completion(path)[0]
    document = json.loads(path.read_text())
    assert any(
        o["contrast"] == "N" and o["reason"] == "missing_condition" for o in document["omissions"]
    )
    assert not list(path.parent.glob("*run-02_contrast-N_*"))
    assert (
        list(path.parents[2].glob("node-subject/sub-01/*contrast-SvN_stat-t_*.nii.gz"))
        if domain == "volume"
        else list(
            path.parents[2].glob("node-subject/sub-01/*contrast-SvN_hemi-L_stat-t_*.shape.gii")
        )
    )
    before = path.stat().st_mtime_ns
    run_module(**kwargs)
    assert path.stat().st_mtime_ns == before
    missing = next(Path(p) for p in document["public_outputs"] if p.endswith(".svg"))
    missing.unlink()
    assert not validate_completion(path)[0]
    run_module(**kwargs)
    assert missing.exists()
    old_definition = json.loads(path.read_text())["definition_fingerprint"]
    config["minimum_temporal_rank"] = 100
    run_module(**kwargs)
    assert json.loads(path.read_text())["definition_fingerprint"] != old_definition


@pytest.mark.parametrize("domain,smoothing", [("volume", 0), ("volume", 2), ("surface", 0)])
def test_branch_firstlevels_reads_mixed_owners(tmp_path, domain, smoothing):
    import shutil

    from nro.orchestration.branches import BranchPaths
    from nro.orchestration.execution_context import ExecutionContext, InputBinding

    root = _synthetic_project(tmp_path / "BIDS", domain)
    paths = BranchPaths("feature/glm", root.parent, tmp_path / "WORK", tmp_path / "NRO_DEV")
    dev = BranchPaths("dev", paths.bids, paths.work, paths.development)
    runs = discover_raw_runs(root / "sub-01")
    directory = root / "derivatives/preprocessing/main/sub-01/ses-01/func"
    other = dev.output_project("demo") / "derivatives/preprocessing/main/sub-01/ses-01/func"
    other.mkdir(parents=True)
    for path in directory.glob(runs[1].stem + "_*"):
        shutil.move(path, other / path.name)
    context = ExecutionContext(
        paths,
        "demo",
        "firstlevels:test",
        (
            InputBinding("main", "func:1", 1, directory, directory, runs[0].stem),
            InputBinding("dev", "func:2", 1, directory, other, runs[1].stem),
        ),
    )
    before = {p: p.read_bytes() for folder in (directory, other) for p in folder.iterdir()}
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config["noise_model"] = "ols"
    output = run_module(
        runs=runs,
        participant="01",
        project_root=root,
        preprocessing_id="main",
        config_id="main",
        model_id="langlocSN/main",
        model=load_task_model("langlocSN/main"),
        config=config,
        space="T1w" if domain == "volume" else "fsnative",
        smoothing=smoothing,
        work_root=paths.work / "demo/derivatives/firstlevels/main",
        execution_context=context,
    )
    assert validate_completion(output)[0]
    context.require_output(output)
    for path in json.loads(output.read_text())["public_outputs"]:
        context.require_output(Path(path))
    assert {p: p.read_bytes() for folder in (directory, other) for p in folder.iterdir()} == before


def test_all_censored_run_is_an_explicit_omission(tmp_path):
    root = _synthetic_project(tmp_path, "volume")
    config = ConfigStore().load_configuration("firstlevels", "main").values
    for path in root.glob(
        "derivatives/preprocessing/main/sub-01/ses-01/func/*confounds_timeseries.tsv"
    ):
        frame = pd.read_csv(path, sep="\t")
        frame["motion_outlier00"] = 1
        frame.to_csv(path, sep="\t", index=False)
    path = run_module(
        runs=discover_raw_runs(root / "sub-01"),
        participant="01",
        project_root=root,
        preprocessing_id="main",
        config_id="main",
        model_id="langlocSN/main",
        model=load_task_model("langlocSN/main"),
        config=config,
        space="T1w",
        smoothing=0,
        work_root=tmp_path / "work",
    )
    result = json.loads(path.read_text())
    assert validate_completion(path)[0]
    assert not result["records"]
    assert len(result["omissions"]) == 2
    assert all(
        record["reason"] == "unidentifiable_temporal_model" for record in result["omissions"]
    )


def test_run_groupby_and_inherited_events(tmp_path):
    from nro.engine.bids import resolve_bids_table
    from nro.modules.firstlevels.models import validate_run_groups

    root = _synthetic_project(tmp_path, "volume")
    runs = discover_raw_runs(root / "sub-01")
    node = load_model("langlocSN/main")["Nodes"][0]
    node["GroupBy"] = ["subject", "task"]
    with pytest.raises(ValueError, match="uniquely identify"):
        validate_run_groups(runs, node, "01")
    shared = root / "task-langlocSN_events.tsv"
    shared.write_text("onset\tduration\n0\t1\n")
    local = resolve_bids_table(runs[0].path, suffix="events")
    assert local != shared
    local.unlink()
    assert resolve_bids_table(runs[0].path, suffix="events") == shared


def test_functional_inputs_select_non_aroma_from_workflow(tmp_path):
    from nro.modules.firstlevels.module import functional_paths

    root = _synthetic_project(tmp_path, "volume")
    run = discover_raw_runs(root / "sub-01")[0]
    assert (
        "desc-preprocNoAROMA"
        in functional_paths(root, "main", run, "T1w", aroma_enabled=True)[0].name
    )
    assert (
        "desc-preproc_bold"
        in functional_paths(root, "main", run, "T1w", aroma_enabled=False)[0].name
    )


def test_absent_named_dummy_contrast_is_recorded():
    from nro.modules.firstlevels.estimation import run_records

    config = ConfigStore().load_configuration("firstlevels", "main").values
    node = load_model("langlocSN/main")["Nodes"][0]
    node["Model"]["X"] = ["trial_type.N"]
    node.pop("Contrasts")
    node["DummyContrasts"] = {"Test": "t", "Contrasts": ["trial_type.N"]}
    events = pd.DataFrame({"onset": [1], "duration": [5], "trial_type": ["S"]})
    design = build_design(node, events, pd.DataFrame({"csf": np.zeros(90)}), 1, config)
    records, omissions = run_records(node, design.names, design, "run", {"subject": "01"})
    assert not records
    assert omissions[0]["contrast"] == "trial_type.N"
    assert omissions[0]["reason"] == "missing_condition"


def test_planner_targets_dependencies_runtime_and_purge_isolation(tmp_path):
    import yaml

    from nro.bin.purge import _instance_paths
    from nro.orchestration.planner import build_subject_instances
    from nro.orchestration.registry import Registry

    root = _synthetic_project(tmp_path / "bids", "volume")
    anatomy = root / "sub-01/anat/sub-01_T1w.nii.gz"
    anatomy.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), anatomy)
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=root.parent)
    registered = registry.register_workflow(workflow)
    specs = build_subject_instances(
        project="demo",
        participant="01",
        module="firstlevels",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=root.parent,
        spaces=("fsnative", "T1w"),
        smoothing_levels=(0, 2),
    )
    targets = [s for s in specs if s.module == "firstlevels"]
    assert len(targets) == 4
    assert {s.module for s in specs} == {"anat", "func", "firstlevels"}
    assert all(len(s.dependencies) == 3 for s in targets)
    assert {(s.entities["space"], s.entities["smoothing"]) for s in targets} == {
        ("fsnative", "0"),
        ("fsnative", "2"),
        ("T1w", "0"),
        ("T1w", "2"),
    }
    target = targets[0]
    runtime = yaml.safe_load(target.runtime_config.read_text())
    assert "preprocessing_aroma" in str(runtime)
    ids = registry.register_instances(specs)
    row = next(row for row in registry.instance_rows() if row["id"] == ids[target.key])
    output = target.expected_outputs[0]
    output.parent.mkdir(parents=True)
    output.write_text("{}")
    other = output.with_name(output.name.replace("model-main", "model-other"))
    other.write_text("{}")
    controlled, _ = _instance_paths(row, registry=registry, work_root=tmp_path / "work")
    assert output in controlled
    assert other not in controlled


def test_bootstrap_discovers_existing_firstlevels_without_demand(tmp_path):
    from nro.modules.firstlevels.paths import artifact_root, instance_prefix
    from nro.orchestration.discovery import register_existing_artifacts
    from nro.orchestration.registry import Registry

    root = _synthetic_project(tmp_path / "bids", "volume")
    anatomy = root / "sub-01/anat/sub-01_T1w.nii.gz"
    anatomy.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), anatomy)
    directory = artifact_root(root, "main", "langlocSN/main", "T1w", 0) / "node-run/sub-01"
    directory.mkdir(parents=True)
    prefix = instance_prefix("01", "langlocSN/main", "T1w", 0)
    (directory / f"{prefix}_partial.txt").write_text("partial artifact")
    registry = Registry.for_project("demo", bids_root=root.parent)
    register_existing_artifacts(registry, bids_root=root.parent, inventory={"demo": ("01",)})
    assert any(row["module"] == "firstlevels" for row in registry.instance_rows())
    with registry.connection() as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 0


@pytest.fixture
def task_store(tmp_path, monkeypatch):
    import yaml

    from nro.modules.firstlevels import task_models

    source = load_task_model("langlocSN/main")
    root = tmp_path / "configuration"
    monkeypatch.setattr(task_models, "definitions_root", lambda: root)
    for task, variant, membership in (
        ("langlocSN", "main", ["main"]),
        ("langlocSN", "dev", ["dev", "experiment"]),
        ("other", "main", ["experiment"]),
        ("other", "unassigned", []),
    ):
        path = root / "models" / task / f"{variant}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({**source, "model_set": membership}))
    return root


def test_model_set_default_and_explicit_selection(task_store):
    from nro.modules.firstlevels.task_models import select_models

    assert set(select_models()) == {"langlocSN/main"}
    assert set(select_models(models=("dev",))) == {"langlocSN/dev"}
    assert set(select_models(models=("main",))) == {"langlocSN/main", "other/main"}
    assert set(select_models(model_sets=("experiment",))) == {"langlocSN/dev", "other/main"}
    assert set(select_models(tasks=("langlocSN",), model_sets=("experiment",))) == {"langlocSN/dev"}
    assert not select_models(models=("dev",), model_sets=("main",))
    assert len(select_models(model_sets=())) == 4


def test_shared_model_selectors_preserve_task_variant_pairs(task_store):
    from nro.bin.run import build_parser
    from nro.engine.cli import core_selection, matches_instance_selectors

    selection = core_selection(
        build_parser().parse_args(
            [
                "-m",
                "firstlevels",
                "--task",
                "langlocSN",
                "--model-set",
                "experiment",
                "--run",
                "task=langlocSN,other",
                "-s",
                "T1w",
                "-S",
                "0",
            ]
        )
    )
    assert selection.runs == {"task": ("langlocSN",)}
    entities = {"task": "langlocSN", "model": "dev", "space": "T1w", "smoothing": "0"}
    assert matches_instance_selectors(entities, selection.instance_entities)
    assert not matches_instance_selectors(
        {**entities, "model": "main"}, selection.instance_entities
    )
    with pytest.raises(ValueError, match="disjoint"):
        core_selection(build_parser().parse_args(["--task", "other", "--run", "task=langlocSN"]))
    assert matches_instance_selectors(entities, {"model": ("langlocSN/dev",)})


def test_model_membership_does_not_change_contract_or_configuration(task_store, tmp_path):
    import yaml

    from nro.modules.firstlevels.task_models import model_contract, scientific_model
    from nro.orchestration.manifests import _current_contract, assess_registry
    from nro.orchestration.planner import build_subject_instances
    from nro.orchestration.registry import Registry

    root = _synthetic_project(tmp_path / "bids", "volume")
    anatomy = root / "sub-01/anat/sub-01_T1w.nii.gz"
    anatomy.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.float32), np.eye(4)), anatomy)
    workflow = ConfigStore().resolve("main")
    assert not {"models", "task", "model_documents"}.intersection(
        workflow.configuration("firstlevels").values
    )
    registry = Registry.for_project("demo", bids_root=root.parent)
    registered = registry.register_workflow(workflow)
    kwargs = dict(
        project="demo",
        participant="01",
        module="firstlevels",
        workflow=workflow,
        registered=registered,
        registry=registry,
        bids_root=root.parent,
        spaces=("T1w",),
        smoothing_levels=(0,),
        models=("main", "dev"),
    )
    specs = build_subject_instances(**kwargs)
    targets = {s.entities["model"]: s for s in specs if s.module == "firstlevels"}
    assert set(targets) == {"main", "dev"}
    ids = registry.register_instances(specs)
    with registry.connection(write=True) as db:
        db.execute("UPDATE instances SET artifact_state='fresh'")
    model = load_task_model("langlocSN/main")
    scientific = scientific_model(model)
    path = task_store / "models/langlocSN/main.yml"
    path.write_text(
        yaml.safe_dump({**model, "model_set": ["experiment"], "description": "New description"})
    )
    assert model_contract(targets["main"].entities)["task_model"] == scientific
    updated = build_subject_instances(**kwargs)
    assert [s.as_record() for s in specs] == [s.as_record() for s in updated]
    registry.register_instances(updated)
    assert all(row["artifact_state"] == "fresh" for row in registry.instance_rows())
    for spec in targets.values():
        assert not _current_contract(spec.as_record())[2]
        assert path not in spec.input_paths
    changed = load_task_model("langlocSN/main")
    changed["contrasts"]["SvN"]["S"] = 1
    path.write_text(yaml.safe_dump(changed))
    assert _current_contract(targets["main"].as_record())[2]
    assert not _current_contract(targets["dev"].as_record())[2]
    assert ConfigStore().resolve("main").fingerprint == workflow.fingerprint
    assess_registry(registry, projects=["demo"])
    row = next(r for r in registry.instance_rows() if r["id"] == ids[targets["main"].key])
    command = json.loads(row["command_json"])
    snapshot = json.loads(command[command.index("--model-definition") + 1])
    assert snapshot == scientific_model(changed)


@pytest.mark.parametrize("edit", ["membership", "qualified", "defaults", "advanced", "execution"])
def test_equivalent_model_edit_does_not_rerun_completed_module(tmp_path, edit):
    root = _synthetic_project(tmp_path, "volume")
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config["noise_model"] = "ols"
    source = load_task_model("langlocSN/main")
    kwargs = dict(
        runs=discover_raw_runs(root / "sub-01"),
        participant="01",
        project_root=root,
        preprocessing_id="main",
        config_id="main",
        model_id="langlocSN/main",
        model=source,
        config=config,
        space="T1w",
        smoothing=0,
        work_root=tmp_path / "work",
    )
    output = run_module(**kwargs)
    before = {
        p: Path(p).stat().st_mtime_ns for p in json.loads(output.read_text())["public_outputs"]
    }
    if edit == "membership":
        source["model_set"] = ["development"]
        source["description"] = "Edited without changing the analysis"
    elif edit == "qualified":
        source["contrasts"] = {
            name: {f"trial_type.{key}": value for key, value in weights.items()}
            for name, weights in source["contrasts"].items()
        }
    elif edit == "defaults":
        source.update(hrf="spm", hrf_overrides={})
    elif edit == "execution":
        config["spatial_block_size"] = 4
    else:
        from nro.modules.firstlevels.task_models import scientific_model

        kwargs["model"] = scientific_model(source)
    run_module(**kwargs)
    assert before == {p: Path(p).stat().st_mtime_ns for p in before}
    compiled_path = next(
        Path(p) for p in before if p.endswith("_statsmodel.json") and "run-01" in p
    )
    compiled = json.loads(compiled_path.read_text())
    rules = compiled["Nodes"][0]["Model"]["Software"]["nro"]
    assert rules["outlier_columns"] == ["motion_outlier00"]
    assert "csf" in compiled["Nodes"][0]["Model"]["X"]
    assert not any("model_set" in json.dumps(d) for d in (compiled, json.loads(output.read_text())))


def test_event_transforms_and_hrf_are_source_aware():
    config = ConfigStore().load_configuration("firstlevels", "main").values
    source = {
        "predictors": ["trial_type.*", "modulator"],
        "hrf": "spm",
        "hrf_overrides": {"modulator": None},
        "transformations": [
            {"Name": "Factor", "Input": "trial_type"},
            {
                "Name": "Scale",
                "Input": "response",
                "Demean": True,
                "Rescale": False,
                "Groupby": ["trial_type"],
                "Output": "centered",
            },
            {"Name": "Product", "Input": ["trial_type.S", "centered"], "Output": "modulator"},
        ],
        "contrasts": {"S": {"trial_type.S": 1}, "modulation": {"modulator": 1}},
    }
    events = pd.DataFrame(
        {
            "onset": [5, 25, 45, 65],
            "duration": [5] * 4,
            "trial_type": ["S", "N", "S", "N"],
            "response": [1.0, 10.0, 3.0, 20.0],
        }
    )
    confounds = pd.DataFrame(
        {"csf": np.arange(100, dtype=float), "global_signal": np.arange(100, dtype=float)}
    )
    model = compile_model(source, "task/main", config, sessions=False)
    assert [n["Level"] for n in model["Nodes"]] == ["Run", "Subject"]
    design = build_design(model["Nodes"][0], events, confounds, 1, config)
    resolved = design.metadata["StatsModelNode"]
    from nro.modules.firstlevels.models import event_design

    sampled = event_design(resolved, events, confounds, 1)
    assert_allclose(sampled["csf"], confounds["csf"])
    assert_allclose(sampled["modulator"].iloc[5:10], -1)
    assert sampled["trial_type.S"].iloc[10] != 0
    assert "global_signal" not in sampled
    assert design.metadata["PredictorSources"]["modulator"] == "events.tsv"
    assert design.metadata["PredictorSources"]["csf"] == "confounds.tsv"


@pytest.mark.parametrize(
    "field", ["confounds_regex", "models", "task", "levels", "noise_model", "Nodes"]
)
def test_task_yaml_rejects_wrong_ownership(field):
    from nro.modules.firstlevels.task_models import validate_task_model

    with pytest.raises(ValueError, match="Unsupported task-model"):
        validate_task_model({**load_task_model("langlocSN/main"), field: "anything"})


def test_advanced_blocks_are_validated_and_cannot_select_confounds():
    config = ConfigStore().load_configuration("firstlevels", "main").values
    source = {
        "predictors": ["trial_type.*"],
        "statsmodels": {
            "Transformations": {
                "Transformer": "pybids-transforms-v1",
                "Instructions": [{"Name": "Factor", "Input": "trial_type"}],
            },
            "Contrasts": [
                {"Name": "S", "ConditionList": ["trial_type.S"], "Weights": [1], "Test": "t"}
            ],
        },
    }
    assert compile_model(source, "task/main", config)["Nodes"][0]["Contrasts"][0]["Name"] == "S"
    from nro.modules.firstlevels.task_models import validate_task_model

    with pytest.raises(ValueError, match="not both"):
        validate_task_model({**source, "conditions": "trial_type"})
    source = {"predictors": ["csf"], "contrasts": {"csf": {"csf": 1}}}
    node = compile_model(source, "task/main", config)["Nodes"][0]
    events = pd.DataFrame({"onset": [0], "duration": [1], "trial_type": ["A"]})
    with pytest.raises(ValueError, match="does not match"):
        build_design(node, events, pd.DataFrame({"csf": np.ones(100)}), 1, config)


def test_exact_outliers_bypass_nuisance_rank_cap():
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config["minimum_temporal_rank"] = 1000
    model = load_model("langlocSN/main", config)
    events = pd.DataFrame({"onset": [0, 20], "duration": [2, 2], "trial_type": ["S", "N"]})
    confounds = pd.DataFrame(
        {"csf": np.arange(80, dtype=float), "motion_outlier00": [1] + [0] * 79}
    )
    design = build_design(model["Nodes"][0], events, confounds, 1, config)
    assert design.metadata["NuisancePCs"] == 0
    assert design.retained.sum() == 79
    assert "motion_outlier00" not in design.metadata["NuisanceCandidates"]


def test_compiled_contrasts_commute_when_all_conditions_share_runs():
    from types import SimpleNamespace

    from nro.modules.firstlevels.estimation import evaluate_recipe, run_records

    model = load_model("langlocSN/main")
    fits = _fits()
    records = []
    for key in fits:
        run, _ = run_records(
            model["Nodes"][0],
            ["intercept", "trial_type.S", "trial_type.N", "other"],
            SimpleNamespace(is_estimable=lambda _: True),
            key,
            {"subject": "01"},
        )
        records.extend(run)
    pooled, omitted = meta_records(
        model["Nodes"][-1], records, model["Edges"][-1], weighting="equal"
    )
    assert not omitted
    result = evaluate_recipe(
        next(r["recipe"] for r in pooled if r["name"] == "SvN"), fits
    ).evaluate(fits)
    per_run = [evaluate_recipe(r["recipe"], fits) for r in records if r["name"] == "SvN"]
    expected = aggregate(per_run, fits, weighting="equal").evaluate(fits)
    for statistic in result:
        assert_allclose(result[statistic], expected[statistic], rtol=1e-12)


def test_disjoint_conditions_produce_subject_contrast_but_no_run_contrasts(tmp_path):
    root = _synthetic_project(tmp_path, "volume")
    config = ConfigStore().load_configuration("firstlevels", "main").values
    config["noise_model"] = "ols"
    for index, path in enumerate(sorted(root.glob("sub-01/ses-01/func/*events.tsv"))):
        events = pd.read_csv(path, sep="\t")
        events["trial_type"] = "S" if index == 0 else "N"
        events.to_csv(path, sep="\t", index=False)
    model = load_task_model("langlocSN/main")
    model["contrasts"] = {"SvN": {"S": 0.5, "N": -0.5}}
    output = run_module(
        runs=discover_raw_runs(root / "sub-01"),
        participant="01",
        project_root=root,
        preprocessing_id="main",
        config_id="main",
        model_id="langlocSN/main",
        model=model,
        config=config,
        space="T1w",
        smoothing=0,
        work_root=tmp_path / "work",
    )
    result = json.loads(output.read_text())
    assert not list(output.parent.glob("*contrast-*_stat-t_*.nii.gz"))
    assert list(output.parents[2].glob("node-subject/sub-01/*contrast-SvN_stat-t_*.nii.gz"))
    assert not list(output.parents[2].rglob("*contrast-nroEffect*"))
    assert sum(o["contrast"] == "SvN" for o in result["omissions"]) == 2


def test_public_adoption_checks_scientific_model_not_set(tmp_path):
    from nro.modules.firstlevels.contract import firstlevels_output_contract
    from nro.modules.firstlevels.task_models import scientific_model
    from nro.orchestration.manifests import (
        _public_derivative_completion,
        _PublicDerivativeContractMismatch,
    )

    source = load_task_model("langlocSN/main")
    scientific = scientific_model(source)
    path = tmp_path / "test_manifest.json"
    path.write_text(
        json.dumps(
            {
                "complete": True,
                "task_model": source,
                "output_metadata_contract": firstlevels_output_contract(),
                "public_outputs": [],
            }
        )
    )
    row = {
        "module": "firstlevels",
        "output_root": str(tmp_path),
        "expected_outputs_json": json.dumps([str(path)]),
        "artifact_contract_json": json.dumps({"processing": {"task_model": scientific}}),
    }
    assert _public_derivative_completion(row, None)[0] is not None
    source["model_set"] = ["development"]
    row["artifact_contract_json"] = json.dumps(
        {"processing": {"task_model": scientific_model(source)}}
    )
    assert _public_derivative_completion(row, None)[0] is not None
    source["contrasts"]["SvN"]["S"] = 1
    row["artifact_contract_json"] = json.dumps(
        {"processing": {"task_model": scientific_model(source)}}
    )
    with pytest.raises(_PublicDerivativeContractMismatch, match="task model differs"):
        _public_derivative_completion(row, None)


def test_invalid_unselected_development_model_does_not_block_main(task_store):
    import yaml

    from nro.modules.firstlevels.task_models import select_models

    path = task_store / "models/langlocSN/dev.yml"
    path.write_text(yaml.safe_dump({"model_set": "dev", "not_implemented_yet": True}))
    assert set(select_models()) == {"langlocSN/main"}
    with pytest.raises(ValueError, match="Unsupported task-model"):
        select_models(model_sets=("dev",))


def test_reference_levels_and_constant_group_handling():
    from nro.modules.firstlevels.transforms import event_variables

    events = pd.DataFrame({"trial_type": ["A", "B", "A"], "x": [2.0, 2.0, 2.0]})
    transformed, _, _ = event_variables(
        events,
        {
            "Instructions": [
                {
                    "Name": "Factor",
                    "Input": "trial_type",
                    "Constraint": "drop_one",
                    "RefLevel": "A",
                },
                {"Name": "Demean", "Input": "x"},
            ]
        },
    )
    assert "trial_type.A" not in transformed
    assert_allclose(transformed["trial_type.B"], [0, 1, 0])
    assert_allclose(transformed["x"], 0)
    with pytest.raises(ValueError, match="constant"):
        event_variables(events, {"Instructions": [{"Name": "Scale", "Input": "x"}]})
    with pytest.raises(ValueError, match="explicit RefLevel"):
        event_variables(
            events,
            {"Instructions": [{"Name": "Factor", "Input": "trial_type", "Constraint": "drop_one"}]},
        )
