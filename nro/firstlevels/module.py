"""Construct a complete firstlevels runner from a task model and source runs."""

from functools import partial
from copy import deepcopy
from itertools import count
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

from nro.engine.bids import BidsRun, resolve_bids_table
from nro.engine.images import sidecar_json_path
from nro.engine.io import atomic_output_path, atomic_write_json, atomic_write_text
from nro.engine.templates import find_fsaverage_surface
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from .contract import definition_fingerprint, firstlevels_output_contract, validate_completion
from .design import build_design
from .estimation import evaluate_maps, meta_records, run_records
from .io import read_timeseries, save_fit, write_statmaps
from .paths import artifact_root, completion_path, instance_prefix, node_prefix
from .report import write_design
from .models import validate_run_groups
from .compiler import compile_model, realize_run_node
from .task_models import scientific_model
from .statistics import RunFit, UnidentifiableDesignError, fit_glm


LOG = logging.getLogger(__name__)


def functional_paths(project_root: Path, preprocessing_id: str, run: BidsRun, space: str,
                     *, aroma_enabled: bool = False) -> tuple[Path, ...]:
    """Select non-AROMA func outputs from the upstream workflow's denoising choice."""
    relative = run.path.parent.relative_to(project_root)
    directory = project_root / "derivatives" / "preprocessing" / preprocessing_id / relative
    description = "preprocNoAROMA" if aroma_enabled else "preproc"
    if space in {"fsnative", "fsaverage"}:
        return tuple(directory / f"{run.stem}_space-{space}_hemi-{h}_desc-{description}_bold.func.gii" for h in ("L", "R"))
    return (directory / f"{run.stem}_space-{space}_desc-{description}_bold.nii.gz",)


def _write_manifest(path: Path, base: dict, *, outputs: list[Path], records: list,
                    omissions: list, sources: dict) -> None:
    previous = json.loads(path.read_text()) if path.is_file() else {}
    value = {**base, "complete": True, "public_outputs": sorted(set(str(p) for p in outputs)),
             "records": records, "omissions": omissions, "source_fits": sources,
             "output_metadata_contract": firstlevels_output_contract()}
    atomic_write_json(path, value)
    # Only superseded files explicitly owned by this node's prior inventory may
    # disappear when an effect becomes unavailable. Never scan/delete by suffix.
    for old in set(previous.get("public_outputs", [])) - set(value["public_outputs"]):
        candidate = Path(old)
        if candidate.parent == path.parent and candidate.name.startswith(base["prefix"] + "_"):
            candidate.unlink(missing_ok=True)


def _publish_records(records: list, sources: dict, prefix: Path, geometry: dict, base: dict, config: dict) -> list[Path]:
    outputs = []
    for record in records:
        if record.get("internal"):
            continue
        entities = {{"session": "ses", "direction": "dir", "acquisition": "acq"}.get(k, k): v
                    for k, v in record["entities"].items()}
        extra = "".join(f"_{key}-{value}" for key, value in entities.items()
                        if key not in {"subject", "task"} and value is not None
                        and f"_{key}-{value}" not in prefix.name)
        destination = prefix.with_name(f"{prefix.name}{extra}_contrast-{record['name']}")
        maps = evaluate_maps(record["recipe"], sources, block_size=config["spatial_block_size"])
        if record["test"] == "pass":
            maps = {key: maps[key] for key in ("effect", "variance", "dof")}
        metadata = {"Model": base["model"], "Node": base["node"], "Contrast": record["name"],
                    "Space": base["space"], "SmoothingFWHMmm": base["smoothing"],
                    "DegreesOfFreedomMethod": "conditional-GLS; independent-run Satterthwaite",
                    "NoiseModel": config["noise_model"], "AggregationWeighting": base["aggregation_weighting"],
                    "ResponseScaling": "none", "InvalidLocations": "NaN t/DOF where variance is zero",
                    "InputDenoising": "without ICA-AROMA",
                    "InferenceLimitations": "Conditional on estimated AR groups and, if selected, precision weights; approximate t reference",
                    "LinearRecipe": record["recipe"]}
        outputs.extend(write_statmaps(destination, maps, geometry, metadata))
    return outputs


def create_fit_step(*, run: BidsRun, images: tuple[Path, ...], original_images: tuple[Path, ...],
                    confounds_path: Path, events_path: Path, node: dict, prefix: Path,
                    base: dict, config: dict) -> Step:
    """Declare one independent run fit, with runtime omissions recorded in its manifest."""
    manifest = prefix.with_name(prefix.name + "_manifest.json")

    def action() -> None:
        confounds = pd.read_csv(confounds_path, sep="\t")
        events = pd.read_csv(events_path, sep="\t")
        metadata = json.loads(sidecar_json_path(original_images[0]).read_text())
        if "_desc-preproc_bold" in original_images[0].name and metadata.get("Denoising", {}).get("applied"):
            raise ValueError("Firstlevels requires non-AROMA input; check the upstream workflow selection")
        tr = float(metadata["RepetitionTime"])
        events["onset"] -= float(metadata.get("StartTime", 0))
        compiled_path = prefix.with_name(prefix.name + "_statsmodel.json")
        compiled = deepcopy(base["model_document"])
        resolved_node = realize_run_node(node, events, confounds, config)
        compiled["Nodes"][0] = resolved_node
        atomic_write_json(compiled_path, compiled)
        try:
            design = build_design(resolved_node, events, confounds, tr, config)
        except UnidentifiableDesignError as error:
            LOG.warning("Omitting %s: %s", run.stem, error)
            _write_manifest(manifest, base, outputs=[compiled_path], records=[], sources={},
                            omissions=[{"run": run.stem, "contrast": None,
                                        "reason": "unidentifiable_temporal_model", "detail": str(error)}])
            return
        design_outputs = [compiled_path, *write_design(prefix, design)]
        entities = {"subject": base["participant"]}
        entities.update({{"ses": "session", "dir": "direction", "acq": "acquisition"}.get(k, k): v
                         for k, v in run.entities.items() if k != "sub"})
        key = run.stem
        records, omissions = run_records(node, design.names, design, key, entities)
        if not records:
            _write_manifest(manifest, {**base, "design_metadata": design.metadata},
                            outputs=design_outputs, records=[], omissions=omissions, sources={})
            return
        data, geometry = read_timeseries(images)
        if len(data) != len(confounds):
            raise ValueError("Functional frame count differs from confounds")
        grid = np.array([0.0]) if config["noise_model"] == "ols" else np.asarray(config["ar_grid"])
        LOG.info("Fitting %s: %d retained frames, %d observation dimensions, %d predictors, %d locations",
                 run.stem, design.retained.sum(), design.matrix.shape[0], design.matrix.shape[1], data.shape[1])
        fitted = fit_glm(data, design.matrix, retained=design.retained,
                         ar_grid=grid, block_size=config["spatial_block_size"])
        mapping = design.coefficient_map
        fit = RunFit((mapping @ fitted.beta).astype(np.float32), fitted.residual_variance, fitted.groups,
                     np.array([mapping @ covariance @ mapping.T for covariance in fitted.covariance]),
                     fitted.dof, fitted.ar_coefficients)
        fit_record, outputs = save_fit(prefix, fit)
        outputs.extend(design_outputs)
        sources = {key: fit_record}
        outputs.extend(_publish_records(records, sources, prefix, geometry, base, config))
        spatial = {key: value for key, value in geometry.items() if key != "reference"}
        if geometry["domain"] == "volume":
            spatial["affine"] = geometry["reference"].affine.tolist()
            spatial["reference_path"] = str(original_images[0])
        _write_manifest(manifest, {**base, "design_metadata": design.metadata, "geometry": spatial}, outputs=outputs,
                        records=records, omissions=omissions, sources=sources)

    return Step.python(name=f"Fit {run.stem}", outputs=(manifest,),
                       inputs=(*images, confounds_path, events_path, sidecar_json_path(original_images[0])),
                       action=action, validate=partial(validate_completion, manifest,
                           definition={"model": base["model_document"], "config": config}))


def create_meta_step(*, inputs: tuple[Path, ...], edge: dict, node: dict,
                     prefix: Path, base: dict, config: dict) -> Step:
    """Declare a session or subject summary without refitting time series."""
    manifest = prefix.with_name(prefix.name + "_manifest.json")

    def action() -> None:
        parents = [json.loads(path.read_text()) for path in inputs]
        records, omissions = meta_records(node, [r for p in parents for r in p["records"]], edge,
                                         weighting=node["Model"]["Software"]["nro"]["aggregation_weighting"])
        sources = {key: value for p in parents for key, value in p["source_fits"].items()}
        spatial = [p["geometry"] for p in parents if p["records"]]
        geometry = spatial[0] if spatial else None
        for candidate in spatial[1:]:
            if any(candidate.get(key) != geometry.get(key) for key in ("domain", "shape", "counts")):
                raise ValueError("Run geometries differ within one firstlevels target")
            if geometry["domain"] == "volume" and not np.allclose(candidate["affine"], geometry["affine"]):
                raise ValueError("Run affines differ; pooled voxel correspondence is undefined")
        if geometry and geometry["domain"] == "volume":
            geometry = {**geometry, "reference": nib.load(geometry["reference_path"])}
        outputs = _publish_records(records, sources, prefix, geometry, base, config) if records else []
        _write_manifest(manifest, {**base, "geometry": spatial[0] if spatial else None},
                        outputs=outputs, records=records, omissions=omissions, sources=sources)

    return Step.python(name=f"Aggregate {node['Name']}", inputs=inputs, outputs=(manifest,), action=action,
                       validate=partial(validate_completion, manifest,
                           definition={"model": base["model_document"], "config": config, "runs": base["runs"]}))


def create_smoothing_step(source: Path, output: Path, *, smoothing: int,
                          surface: Path | None, wb_command: str, run_command) -> Step:
    """Declare Euclidean volume or Workbench geodesic surface smoothing."""
    def action() -> None:
        with atomic_output_path(output) as staged:
            if surface is not None:
                run_command([wb_command, "-metric-smoothing", str(surface), str(source), str(smoothing), str(staged), "-fwhm"])
            else:
                image = nib.load(str(source))
                axes = image.affine[:3, :3]
                spacing = np.linalg.norm(axes, axis=0)
                if not np.allclose((axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-4):
                    raise ValueError("Euclidean smoothing requires an orthogonal voxel grid")
                sigma = smoothing / np.sqrt(8 * np.log(2)) / spacing
                values = gaussian_filter(np.asarray(image.dataobj, dtype=np.float32), (*sigma, 0), mode="constant")
                nib.save(nib.Nifti1Image(values, image.affine, image.header), staged)
    return Step.python(name=f"Smooth {source.name}", inputs=(source, *((surface,) if surface else ())),
                       outputs=(output,), action=action)


def run_module(*, runs: tuple[BidsRun, ...], participant: str, project_root: Path,
               preprocessing_id: str, config_id: str, model_id: str, model: dict,
               config: dict, space: str, smoothing: int, work_root: Path,
               output_root: Path | None = None, definition_inputs: tuple[Path, ...] = ()) -> Path:
    """Construct and execute one participant/task/model/space/smoothing DAG."""
    if not runs or smoothing < 0 or config["noise_model"] not in {"ols", "ar1"}:
        raise ValueError("Firstlevels requires selected runs, nonnegative smoothing and ols/ar1 noise")
    if config["spatial_block_size"] < 1:
        raise ValueError("Invalid block-size configuration")
    task_definition = scientific_model(model)
    model = compile_model(task_definition, model_id, config, sessions=any("ses" in run.entities for run in runs))
    validate_run_groups(runs, model["Nodes"][0], participant)
    root = output_root or artifact_root(project_root, config_id, model_id, space, smoothing)
    prefix = instance_prefix(participant, model_id, space, smoothing)
    runner = Runner(module_name="firstlevels", container=None, binds=(), logger=LOG, next_step=count(1).__next__)
    runner.set_definition_inputs(definition_inputs)
    definition = {"model": model, "config": config}
    base = {"model": model_id, "participant": participant, "space": space, "smoothing": smoothing,
            "configuration": config, "model_document": model, "task_model": task_definition, "prefix": prefix,
            "aggregation_weighting": task_definition["aggregation"]["weighting"],
            "definition_fingerprint": definition_fingerprint(definition)}
    manifests = {}
    root_node = model["Nodes"][0]
    for run in runs:
        original = functional_paths(project_root, preprocessing_id, run, space,
                                    aroma_enabled=config.get("preprocessing_aroma", False))
        images = original
        if smoothing:
            smoothed = []
            for index, source in enumerate(original):
                surface = None
                if source.name.endswith(".gii"):
                    if space == "fsnative":
                        anatomy = project_root / "derivatives" / "preprocessing" / preprocessing_id / f"sub-{participant}" / "anat"
                        anatomy_manifest = anatomy / f"sub-{participant}_desc-preprocessAnat_manifest.json"
                        surfaces = json.loads(anatomy_manifest.read_text())["outputs"]["surfaces"]
                        surface = Path(surfaces[f"{'lh' if index == 0 else 'rh'}.midthickness"])
                    else:
                        count_vertices = len(nib.load(str(source)).darrays[0].data)
                        surface = find_fsaverage_surface(hemi="L" if index == 0 else "R", surface="midthickness", n_vertices=count_vertices)
                output = work_root / prefix / source.name
                runner.add_step(create_smoothing_step(source, output, smoothing=smoothing, surface=surface,
                                                     wb_command=config["wb_command"], run_command=runner.run_child))
                smoothed.append(output)
            images = tuple(smoothed)
        destination = node_prefix(root, prefix, root_node, run_stem=run.stem)
        confounds = original[0].parent / f"{run.stem}_desc-confounds_timeseries.tsv"
        events = resolve_bids_table(run.path, suffix="events")
        step = create_fit_step(run=run, images=images, original_images=original, confounds_path=confounds,
                               events_path=events, node=root_node, prefix=destination,
                               base={**base, "node": root_node["Name"]}, config=config)
        runner.add_step(step)
        manifests.setdefault(root_node["Name"], []).extend(step.outputs)
    definition = {**definition, "runs": [run.stem for run in runs]}
    base = {**base, "runs": definition["runs"], "definition_fingerprint": definition_fingerprint(definition)}
    for node in model["Nodes"][1:]:
        edge = next(e for e in model["Edges"] if e["Destination"] == node["Name"])
        step = create_meta_step(inputs=tuple(manifests[edge["Source"]]), edge=edge, node=node,
                                prefix=node_prefix(root, prefix, node),
                                base={**base, "node": node["Name"]}, config=config)
        runner.add_step(step)
        manifests[node["Name"]] = list(step.outputs)
    completion = completion_path(root, prefix)
    inputs = tuple(path for paths in manifests.values() for path in paths)

    def finalize() -> None:
        documents = [json.loads(path.read_text()) for path in inputs]
        outputs = list(inputs) + [Path(p) for document in documents for p in document["public_outputs"]]
        source_path = completion.with_name(prefix + "_model.yml")
        config_path = completion.with_name(prefix + "_configuration.json")
        compiled_path = completion.with_name(prefix + "_statsmodel.json")
        atomic_write_text(source_path, yaml.safe_dump(task_definition, sort_keys=False))
        atomic_write_json(config_path, config)
        atomic_write_json(compiled_path, model)
        outputs.extend((source_path, config_path, compiled_path))
        _write_manifest(completion, {**base, "node": "module"}, outputs=outputs,
                        records=[r for d in documents for r in d["records"]],
                        omissions=[o for d in documents for o in d["omissions"]],
                        sources={k: v for d in documents for k, v in d["source_fits"].items()})
    runner.add_step(Step.python(name="Publish firstlevels", inputs=inputs, outputs=(completion,), action=finalize,
                               validate=partial(validate_completion, completion, definition=definition), completion_boundary=True))
    with runner.run_context():
        runner.execute()
    return completion
