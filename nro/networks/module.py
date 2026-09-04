from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import asdict, replace
from itertools import count
from pathlib import Path

import numpy as np
import yaml

from nro.orchestration.runtime import selected_configuration_fingerprint
from nro.microparcellation.cifti import load_dlabel
from nro.orchestration.runner import Runner, write_completion_breadcrumb
from nro.orchestration.runner_graph import Step
from nro.engine.images import write_cifti_dense_scalar
from nro.engine.io import (
    atomic_save_npz,
    atomic_write_text,
    json_path_default,
    manifest_value,
)
from .adjacency import load_adjacency, pconn_to_adjacency, save_adjacency
from .config import ModuleConfig, validate_config
from .consensus import membership_stability
from .leiden import leiden_partition, write_hint
from .oslom import parse_tp, resolve_oslom_executable, run_oslom, write_oslom_graph
from .labeling import (
    REFERENCE_ATLASES,
    network_map_names,
    project_references_to_cifti,
    rank_reference_candidates,
    reference_paths,
)
from .scene import write_network_scene

LOG = logging.getLogger(__name__)


def _load_inputs(cfg: ModuleConfig):
    import nibabel as nib

    labels, vertex_counts = load_dlabel(cfg.inputs.microparcels)
    dlabel = nib.load(str(cfg.inputs.microparcels))
    pconn = nib.load(str(cfg.inputs.connectivity))
    brain_axis = dlabel.header.get_axis(1)
    parcel_axes = (pconn.header.get_axis(0), pconn.header.get_axis(1))
    if parcel_axes[0] != parcel_axes[1]:
        raise ValueError("Microparcel connectivity row and column mappings differ")
    parcel_axis = parcel_axes[0]
    n_microparcels = int(pconn.shape[0])
    active = labels >= 0
    represented = np.unique(labels[active])
    expected = np.arange(n_microparcels)
    if not np.array_equal(represented, expected):
        raise ValueError(
            "Microparcel assignments do not match adjacency indices "
            f"0..{n_microparcels - 1}"
        )
    label_axis = dlabel.header.get_axis(0)
    label_table = label_axis.label[0]
    expected_names = np.asarray([label_table[index + 1][0] for index in expected])
    if not np.array_equal(parcel_axis.name, expected_names):
        raise ValueError("Microparcellation dlabel and pconn parcel names or order differ")
    expected_voxels: list[np.ndarray] = [
        np.empty((0, 3), dtype=np.int64) for _ in expected
    ]
    expected_vertices: list[dict[str, np.ndarray]] = [{} for _ in expected]
    for structure, structure_slice, _ in brain_axis.iter_structures():
        local_labels = labels[structure_slice]
        active_local = local_labels >= 0
        if not np.any(active_local):
            continue
        parcel_ids = local_labels[active_local]
        order = np.argsort(parcel_ids, kind="stable")
        parcel_ids = parcel_ids[order]
        starts = np.r_[0, np.flatnonzero(np.diff(parcel_ids)) + 1]
        stops = np.r_[starts[1:], len(parcel_ids)]
        voxels = np.asarray(brain_axis.voxel[structure_slice], dtype=np.int64)[active_local][order]
        vertices = np.asarray(brain_axis.vertex[structure_slice], dtype=np.int64)[active_local][order]
        for start, stop in zip(starts, stops):
            parcel = int(parcel_ids[start])
            parcel_voxels = voxels[start:stop]
            parcel_voxels = parcel_voxels[parcel_voxels[:, 0] >= 0]
            if len(parcel_voxels):
                expected_voxels[parcel] = parcel_voxels
            parcel_vertices = vertices[start:stop]
            parcel_vertices = parcel_vertices[parcel_vertices >= 0]
            if len(parcel_vertices):
                expected_vertices[parcel][str(structure)] = parcel_vertices

    for parcel in expected:
        actual_voxels = np.asarray(
            parcel_axis.voxels[parcel], dtype=np.int64
        ).reshape(-1, 3)
        if not np.array_equal(actual_voxels, expected_voxels[parcel]):
            raise ValueError(
                f"Microparcellation dlabel and pconn voxel mapping differ for parcel {parcel + 1}"
            )
        actual_vertices = parcel_axis.vertices[parcel]
        if set(actual_vertices) != set(expected_vertices[parcel]) or any(
            not np.array_equal(np.asarray(actual_vertices[key]), value)
            for key, value in expected_vertices[parcel].items()
        ):
            raise ValueError(
                f"Microparcellation dlabel and pconn surface mapping differ for parcel {parcel + 1}"
            )
    return labels, active, vertex_counts, n_microparcels


NetworkOutputs = dict[
    str, Path | tuple[Path, Path] | list[Path | tuple[Path, Path]]
]


def build_module(
    cfg: ModuleConfig,
    runner: Runner,
    *,
    completion_boundary: bool = True,
) -> NetworkOutputs:
    """Build the complete networks DAG without checking freshness or running work."""
    validate_config(cfg)
    out = cfg.output.directory
    work = cfg.output.work_directory
    manifest_path = out / f"{cfg.output.prefix}_manifest.yaml"
    publication_breadcrumb = out / f".{cfg.output.prefix}_complete"
    labeling_inputs = reference_paths() if cfg.labeling.enabled else ()
    source_inputs = tuple(
        dict.fromkeys(
            (
                cfg.inputs.microparcels,
                cfg.inputs.connectivity,
                *cfg.inputs.source_surfaces,
                *cfg.inputs.scene_surfaces,
                *(
                    (cfg.inputs.label_volume,)
                    if cfg.inputs.label_volume is not None
                    else ()
                ),
                *(
                    (cfg.inputs.anatomical_manifest,)
                    if cfg.inputs.anatomical_manifest is not None
                    else ()
                ),
                *(
                    (cfg.inputs.anatomical_reference,)
                    if cfg.inputs.anatomical_reference is not None
                    else ()
                ),
                *(
                    (cfg.inputs.mni_to_t1_transform,)
                    if cfg.inputs.mni_to_t1_transform is not None
                    else ()
                ),
                *labeling_inputs,
            )
        )
    )
    def recorded_paths(value):
        if isinstance(value, str):
            yield Path(value)
        elif isinstance(value, dict):
            for item in value.values():
                yield from recorded_paths(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from recorded_paths(item)

    def validate_recorded_publication(
        manifest: dict[str, object], *, check_completion_time: bool
    ) -> tuple[bool, str, dict[str, object]]:
        recorded_outputs = manifest.get("outputs") or {}
        if not isinstance(recorded_outputs, dict) or not recorded_outputs:
            return False, "Networks publication manifest has no output inventory.", {}
        recorded = tuple(recorded_paths(recorded_outputs))
        if not recorded:
            return False, "Networks publication manifest has no output inventory.", recorded_outputs
        escaped: list[str] = []
        for path in recorded:
            try:
                path.resolve(strict=False).relative_to(out.resolve(strict=False))
            except ValueError:
                escaped.append(str(path))
        if escaped:
            return False, "Networks publication inventory escapes its owned directory: " + ", ".join(escaped), recorded_outputs
        missing = [
            str(path)
            for path in recorded
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            return False, "Networks publication is missing recorded output(s): " + ", ".join(missing), recorded_outputs
        if check_completion_time:
            completion_ns = publication_breadcrumb.stat().st_mtime_ns
            changed = [str(path) for path in recorded if path.stat().st_mtime_ns > completion_ns]
            if changed:
                return False, "Networks publication member changed after completion: " + ", ".join(changed), recorded_outputs
        return True, "Networks publication and its complete output inventory are valid.", recorded_outputs

    graph_path = work / f"{cfg.output.prefix}_graph.dat"
    adjacency_path = work / f"{cfg.output.prefix}_adjacency.npz"
    input_state_path = work / f"{cfg.output.prefix}_input_state.npz"
    graph_validation_path = work / f"{cfg.output.prefix}_graph_validation.json"
    executable_path = work / f"{cfg.output.prefix}_oslom_executable.txt"
    consensus_path = work / f"{cfg.output.prefix}_consensus.npz"
    consensus_assignments_path = work / f"{cfg.output.prefix}_consensus_assignments.json"
    labeling_path = work / f"{cfg.output.prefix}_network_labels.json"
    leiden_hint_path = (
        work / f"{cfg.output.prefix}_leiden_hint.dat"
        if cfg.oslom.initialization == "leiden"
        else None
    )
    run_root = work / "oslom_runs"
    repetition_breadcrumbs = tuple(
        run_root / f"run_{i + 1:03d}" / ".nro_complete"
        for i in range(cfg.oslom.repetitions)
    )
    private_outputs = (
        graph_path,
        adjacency_path,
        input_state_path,
        graph_validation_path,
        executable_path,
        consensus_path,
        consensus_assignments_path,
        labeling_path,
        *((leiden_hint_path,) if leiden_hint_path is not None else ()),
        *repetition_breadcrumbs,
    )
    publication_inputs = (*source_inputs, *private_outputs)

    initialization_breadcrumb = work / "initialized.complete"
    validation_path = work / f"{cfg.output.prefix}_validated_config.json"
    validation_text = json.dumps(asdict(cfg), default=json_path_default, sort_keys=True) + "\n"
    def validate_and_record_config() -> None:
        validate_config(cfg)
        atomic_write_text(validation_path, validation_text)

    def validate_recorded_config() -> tuple[bool, str]:
        try:
            matches = validation_path.read_text(encoding="utf-8") == validation_text
        except OSError:
            matches = False
        return matches, (
            "Recorded networks configuration matches the requested workflow."
            if matches
            else "Recorded networks configuration differs from the requested workflow."
        )

    runner.add_step(Step.python(
        name="Validate Networks Configuration",
        outputs=(validation_path,),
        inputs=source_inputs,
        force=bool(cfg.output.overwrite),
        action=validate_and_record_config,
        validate=validate_recorded_config,
    ))
    def initialize_outputs() -> None:
        work.mkdir(parents=True, exist_ok=True)
        write_completion_breadcrumb(
            initialization_breadcrumb, "Networks output initialized\n"
        )

    runner.add_step(Step.python(
        name="Initialize Networks Outputs",
        outputs=(initialization_breadcrumb,),
        action=initialize_outputs,
    ))

    def transform_inputs() -> None:
        labels, mask, vertex_counts, n_microparcels_local = _load_inputs(cfg)
        adjacency_local, percentile_weight_local = pconn_to_adjacency(
            cfg.inputs.connectivity,
            transform=cfg.connectivity.transform,
            minimum_weight=cfg.connectivity.minimum_weight,
            percentile_cutoff=cfg.connectivity.percentile_cutoff,
        )
        LOG.info("Saving sparse network adjacency to %s", adjacency_path)
        save_adjacency(adjacency_path, adjacency_local)
        atomic_save_npz(
            input_state_path,
            compressed=True,
            labels=np.asarray(labels, dtype=np.int64),
            mask=np.asarray(mask, dtype=np.uint8),
            vertex_counts=np.asarray(vertex_counts, dtype=np.int64),
            n_microparcels=np.asarray([n_microparcels_local], dtype=np.int64),
            percentile_weight=np.asarray([
                np.nan if percentile_weight_local is None else percentile_weight_local
            ], dtype=np.float64),
        )

    runner.add_step(Step.python(
        name="Load and Transform Network Inputs",
        outputs=(adjacency_path, input_state_path),
        inputs=source_inputs + (validation_path, initialization_breadcrumb),
        force=bool(cfg.output.overwrite),
        action=transform_inputs,
    ))

    def write_graph() -> None:
        adjacency = load_adjacency(adjacency_path)
        write_oslom_graph(graph_path, adjacency)

    runner.add_step(Step.python(
        name="Construct Network Adjacency and Write OSLOM Graph",
        outputs=(graph_path,),
        inputs=(adjacency_path, input_state_path, validation_path),
        force=bool(cfg.output.overwrite),
        action=write_graph,
    ))

    def validate_graph() -> None:
        with np.load(input_state_path, allow_pickle=False) as state:
            n_microparcels = int(state["n_microparcels"][0])
            percentile_value = float(state["percentile_weight"][0])
        percentile_weight = None if np.isnan(percentile_value) else percentile_value
        possible_edges = n_microparcels * (n_microparcels - 1) // 2
        edge_count = sum(1 for line in graph_path.read_text().splitlines() if line.strip())
        if percentile_weight is not None:
            LOG.info(
                "Connectivity percentile cutoff P%g corresponds to weight >= %.9g",
                cfg.connectivity.percentile_cutoff,
                percentile_weight,
            )
        LOG.info(
            "Sparsified OSLOM graph has %d nodes and %d edges (%.2f%% density)",
            n_microparcels,
            edge_count,
            100.0 * edge_count / possible_edges if possible_edges else 0.0,
        )
        if edge_count == 0:
            raise ValueError("No graph edges survived connectivity thresholding")
        atomic_write_text(
            graph_validation_path,
            json.dumps({"edge_count": edge_count, "possible_edges": possible_edges})
            + "\n",
        )

    runner.add_step(Step.python(
        name="Validate OSLOM Graph",
        outputs=(graph_validation_path,),
        inputs=(graph_path, input_state_path),
        force=bool(cfg.output.overwrite),
        action=validate_graph,
    ))
    oslom_initialization_breadcrumb = work / "oslom_initialization.complete"

    def resolve_executable() -> None:
        executable = resolve_oslom_executable(cfg.oslom.executable)
        atomic_write_text(executable_path, str(executable) + "\n")

    runner.add_step(Step.python(
        name="Resolve OSLOM Executable",
        outputs=(executable_path,),
        inputs=(validation_path,),
        force=bool(cfg.output.overwrite),
        action=resolve_executable,
    ))
    hint_path: Path | None = cfg.oslom.initial_partition
    if cfg.oslom.initialization == "leiden":
        hint_path = leiden_hint_path
        assert hint_path is not None

        def write_initial_partition() -> None:
            adjacency = load_adjacency(adjacency_path)
            write_hint(
                hint_path,
                leiden_partition(
                    adjacency,
                    resolution=cfg.oslom.leiden_resolution,
                    iterations=cfg.oslom.leiden_iterations,
                    seed=cfg.oslom.leiden_seed,
                ),
            )
            write_completion_breadcrumb(
                oslom_initialization_breadcrumb, "Network initialization complete\n"
            )

        runner.add_step(Step.python(
            name="Resolve Network Initialization",
            outputs=(hint_path, oslom_initialization_breadcrumb),
            inputs=(graph_path, adjacency_path, validation_path),
            force=bool(cfg.output.overwrite),
            action=write_initial_partition,
        ))
    else:
        hint_path = None
        runner.add_step(Step.python(
            name="Resolve Network Initialization",
            outputs=(oslom_initialization_breadcrumb,),
            inputs=(graph_path, validation_path),
            force=bool(cfg.output.overwrite),
            action=lambda: write_completion_breadcrumb(
                oslom_initialization_breadcrumb, "Network initialization complete\n"
            ),
        ))

    for i in range(cfg.oslom.repetitions):
        repetition_name = f"OSLOM Repetition {i + 1}/{cfg.oslom.repetitions}"
        repetition_dir = run_root / f"run_{i + 1:03d}"
        repetition_breadcrumb = repetition_dir / ".nro_complete"
        def execute_repetition(
            repetition_dir: Path = repetition_dir,
            hint_path: Path | None = hint_path,
        ) -> None:
            executable = Path(executable_path.read_text(encoding="utf-8").strip())
            oslom_cfg = replace(
                cfg.oslom,
                executable=executable,
                initial_partition=hint_path,
            )
            run_oslom(graph_path, repetition_dir, oslom_cfg, runner=runner)

        def validate_repetition(
            repetition_dir: Path = repetition_dir,
        ) -> tuple[bool, str]:
            candidates = (
                repetition_dir / "graph.dat_oslo_files" / "tp_without_singletons",
                repetition_dir / "graph.dat_oslo_files" / "tp",
            )
            if any(path.is_file() and path.stat().st_size > 0 for path in candidates):
                return True, "OSLOM repetition contains a nonempty partition output."
            return False, f"OSLOM repetition has no nonempty tp output: {repetition_dir}"

        runner.add_step(Step.directory_step(
            name=repetition_name,
            directory=repetition_dir,
            breadcrumb=repetition_breadcrumb,
            inputs=(
                graph_path,
                executable_path,
                oslom_initialization_breadcrumb,
                *((hint_path,) if hint_path is not None else ()),
            ),
            force=bool(cfg.output.overwrite),
            action=execute_repetition,
            validate=validate_repetition,
            breadcrumb_text="OSLOM repetition complete\n",
        ))

    def compute_consensus() -> None:
        with np.load(input_state_path, allow_pickle=False) as state:
            labels = np.asarray(state["labels"], dtype=np.int64)
            mask = np.asarray(state["mask"], dtype=bool)
            n_microparcels = int(state["n_microparcels"][0])
        n_vertices = len(labels)
        runs: list[list[set[int]]] = []
        for index in range(cfg.oslom.repetitions):
            repetition_dir = run_root / f"run_{index + 1:03d}"
            candidates = (
                repetition_dir / "graph.dat_oslo_files" / "tp_without_singletons",
                repetition_dir / "graph.dat_oslo_files" / "tp",
            )
            existing = next((path for path in candidates if path.is_file()), None)
            if existing is None:
                raise RuntimeError(f"Completed OSLOM repetition has no tp output: {repetition_dir}")
            runs.append(parse_tp(existing))
        stability, homeless, overlap, ref_idx = membership_stability(
            runs,
            n_microparcels,
            cfg.consensus.minimum_match_jaccard,
        )
        vertex_stability = np.zeros((n_vertices, stability.shape[1]), dtype=np.float32)
        vertex_stability[mask] = stability[labels[mask]]
        vertex_homeless = np.ones(n_vertices, dtype=np.float32)
        vertex_homeless[mask] = homeless[labels[mask]]
        vertex_overlap = np.zeros(n_vertices, dtype=np.float32)
        vertex_overlap[mask] = overlap[labels[mask]]
        atomic_save_npz(
            consensus_path,
            compressed=True,
            vertex_stability=vertex_stability,
            vertex_homeless=vertex_homeless,
            vertex_overlap=vertex_overlap,
            ref_idx=np.asarray([ref_idx], dtype=np.int64),
        )
        atomic_write_text(
            consensus_assignments_path,
            json.dumps(
                [
                    [[int(vertex) for vertex in sorted(group)] for group in fit]
                    for fit in runs
                ]
            )
            + "\n",
        )

    runner.add_step(Step.python(
        name="Compute Network Consensus",
        outputs=(consensus_path, consensus_assignments_path),
        inputs=tuple(repetition_breadcrumbs) + (input_state_path, validation_path),
        force=bool(cfg.output.overwrite),
        action=compute_consensus,
    ))

    def compute_labels() -> None:
        with np.load(consensus_path, allow_pickle=False) as consensus:
            vertex_stability = np.asarray(
                consensus["vertex_stability"], dtype=np.float32
            )
        if cfg.labeling.enabled:
            references = project_references_to_cifti(
                cfg.inputs.microparcels,
                space=cfg.inputs.space,
                source_surfaces=cfg.inputs.source_surfaces,
                anatomical_reference=cfg.inputs.anatomical_reference,
                mni_to_t1_transform=cfg.inputs.mni_to_t1_transform,
            )
            records = rank_reference_candidates(
                vertex_stability.T,
                references,
                cfg.labeling.candidates_per_reference,
            )
        else:
            records = []
        atomic_write_text(
            labeling_path,
            json.dumps(
                {
                    "map_names": network_map_names(vertex_stability.shape[1], records),
                    "candidates": records,
                },
                indent=2,
            )
            + "\n",
        )

    runner.add_step(Step.python(
        name="Assign Heuristic Network Labels",
        outputs=(labeling_path,),
        inputs=tuple(
            dict.fromkeys(
                (
                    consensus_path,
                    cfg.inputs.microparcels,
                    *cfg.inputs.source_surfaces,
                    *((cfg.inputs.anatomical_manifest,) if cfg.inputs.anatomical_manifest else ()),
                    *((cfg.inputs.anatomical_reference,) if cfg.inputs.anatomical_reference else ()),
                    *((cfg.inputs.mni_to_t1_transform,) if cfg.inputs.mni_to_t1_transform else ()),
                    *labeling_inputs,
                )
            )
        ),
        force=bool(cfg.output.overwrite),
        action=compute_labels,
    ))

    membership_path = out / f"{cfg.output.prefix}_desc-membership_network.dscalar.nii"
    stability_path = out / f"{cfg.output.prefix}_desc-stability_network.dscalar.nii"
    homeless_path = out / f"{cfg.output.prefix}_desc-homeless_network.dscalar.nii"
    overlap_path = out / f"{cfg.output.prefix}_desc-overlap_network.dscalar.nii"
    labels_tsv_path = out / f"{cfg.output.prefix}_network-labels.tsv"
    labels_json_path = out / f"{cfg.output.prefix}_network-labels.json"
    scene_path = out / f"{cfg.output.prefix}_networks.scene"
    scene_connectivity_path = out / f"{cfg.output.prefix}_connectivity.pconn.nii"
    if cfg.inputs.domain == "surface":
        scene_assets = (
            scene_connectivity_path,
            *(
                out / f"{cfg.output.prefix}_hemi-{hemi}_{kind}.surf.gii"
                for hemi in ("L", "R")
                for kind in ("pial", "midthickness", "white", "inflated")
            ),
        )
    else:
        scene_assets = (
            scene_connectivity_path,
            out / f"{cfg.output.prefix}_microparcels.nii.gz",
        )

    public_graph_path = out / f"{cfg.output.prefix}_graph.dat"
    public_adjacency_path = (
        out / f"{cfg.output.prefix}_adjacency.npz"
        if cfg.connectivity.write_matrix
        else None
    )
    public_hint_path = (
        out / f"{cfg.output.prefix}_leiden_hint.dat"
        if leiden_hint_path is not None
        else None
    )
    assignment_path = out / f"{cfg.output.prefix}_oslom_assignments.json"
    outputs: NetworkOutputs = {
        "oslom_graph": public_graph_path,
        **({"adjacency": public_adjacency_path} if public_adjacency_path is not None else {}),
        **({"leiden_hint": public_hint_path} if public_hint_path is not None else {}),
        "membership": membership_path,
        **({"stability": stability_path} if cfg.oslom.repetitions > 1 else {}),
        **({"homeless": homeless_path} if cfg.oslom.repetitions > 1 else {}),
        **({"overlap": overlap_path} if cfg.oslom.repetitions > 1 else {}),
        "network_labels": labels_tsv_path,
        "network_labels_metadata": labels_json_path,
        "scene": scene_path,
        "scene_assets": list(scene_assets),
        "oslom_assignments": assignment_path,
    }

    def write_publication() -> None:
        with np.load(input_state_path, allow_pickle=False) as state:
            labels = np.asarray(state["labels"], dtype=np.int64)
            mask = np.asarray(state["mask"], dtype=bool)
            vertex_counts = tuple(int(value) for value in state["vertex_counts"])
            n_microparcels = int(state["n_microparcels"][0])
        with np.load(consensus_path, allow_pickle=False) as consensus:
            vertex_stability = np.asarray(consensus["vertex_stability"], dtype=np.float32)
            vertex_homeless = np.asarray(consensus["vertex_homeless"], dtype=np.float32)
            vertex_overlap = np.asarray(consensus["vertex_overlap"], dtype=np.float32)
            ref_idx = int(consensus["ref_idx"][0])
        graph_validation = json.loads(graph_validation_path.read_text(encoding="utf-8"))
        edge_count = int(graph_validation["edge_count"])
        assignments = json.loads(consensus_assignments_path.read_text(encoding="utf-8"))
        labeling = json.loads(labeling_path.read_text(encoding="utf-8"))
        map_names = [str(value) for value in labeling["map_names"]]
        label_records = list(labeling["candidates"])
        n_vertices = len(labels)
        binary_arrays = [
            (vertex_stability[:, i] > cfg.consensus.assignment_threshold).astype(np.float32)
            for i in range(vertex_stability.shape[1])
        ]

        out.mkdir(parents=True, exist_ok=True)
        for old_path in out.iterdir():
            if not old_path.name.startswith(f"{cfg.output.prefix}_"):
                continue
            if old_path.is_file() or old_path.is_symlink():
                old_path.unlink()
        shutil.copy2(graph_path, public_graph_path)
        if adjacency_path is not None and public_adjacency_path is not None:
            shutil.copy2(adjacency_path, public_adjacency_path)
        if leiden_hint_path is not None and public_hint_path is not None:
            shutil.copy2(leiden_hint_path, public_hint_path)
        write_cifti_dense_scalar(
            membership_path, cfg.inputs.microparcels, binary_arrays, map_names
        )
        if cfg.oslom.repetitions > 1:
            write_cifti_dense_scalar(
                stability_path,
                cfg.inputs.microparcels,
                [vertex_stability[:, index] for index in range(vertex_stability.shape[1])],
                map_names,
            )
            write_cifti_dense_scalar(
                homeless_path,
                cfg.inputs.microparcels,
                [
                    vertex_homeless,
                    (vertex_homeless > cfg.consensus.homeless_threshold).astype(np.float32),
                ],
                ["Homeless stability", "Homeless binary"],
            )
            write_cifti_dense_scalar(
                overlap_path,
                cfg.inputs.microparcels,
                [vertex_overlap],
                ["Overlap stability"],
            )
        atomic_write_text(assignment_path, json.dumps(assignments) + "\n")
        with labels_tsv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "reference",
                    "candidate",
                    "network",
                    "similarity_rank",
                    "similarity_score",
                ),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(label_records)
        atomic_write_text(
            labels_json_path,
            json.dumps(
                {
                    "method": "independent spatial correlation with population reference maps",
                    "candidates_per_reference": cfg.labeling.candidates_per_reference,
                    "references": [atlas.identifier for atlas in REFERENCE_ATLASES]
                    if cfg.labeling.enabled
                    else [],
                    "map_names": map_names,
                },
                indent=2,
            )
            + "\n",
        )
        _scene_path, written_scene_assets = write_network_scene(
            scene_path,
            domain=cfg.inputs.domain,
            membership=membership_path,
            connectivity=cfg.inputs.connectivity,
            scene_surfaces=cfg.inputs.scene_surfaces,
            label_volume=cfg.inputs.label_volume,
        )
        if written_scene_assets != scene_assets:
            raise RuntimeError("Network scene asset inventory is inconsistent")
        manifest = {
            "domain": cfg.inputs.domain,
            "space": cfg.inputs.space,
            "smoothing_fwhm_mm": cfg.inputs.smoothing_mm,
            "n_spatial_nodes": n_vertices,
            "n_surface_vertices": n_vertices if cfg.inputs.domain == "surface" else None,
            "hemisphere_vertex_counts": list(vertex_counts) if cfg.inputs.domain == "surface" else [],
            "n_active_vertices": int(mask.sum()) if cfg.inputs.domain == "surface" else None,
            "n_gray_matter_voxels": n_vertices if cfg.inputs.domain == "volume" else None,
            "n_microparcels": n_microparcels,
            "n_edges": edge_count,
            "reference_run": ref_idx + 1,
            "n_reference_networks": vertex_stability.shape[1],
            "source_surfaces": [str(path) for path in cfg.inputs.source_surfaces],
            "anatomical_labeling_provenance": (
                {
                    "manifest": str(cfg.inputs.anatomical_manifest),
                    "reference": str(cfg.inputs.anatomical_reference),
                    "mni_to_t1_transform": str(cfg.inputs.mni_to_t1_transform),
                }
                if cfg.labeling.enabled and cfg.inputs.space in {"T1w", "fsnative"}
                else None
            ),
            "outputs": {name: manifest_value(path) for name, path in outputs.items()},
            "config": asdict(cfg),
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "interpretation": (
                "Network stability values are repeated-fit assignment frequencies, not posterior probabilities."
                if cfg.oslom.repetitions > 1
                else "Single-repetition run: stability, homeless, and overlap metrics were not written."
            ),
        }
        atomic_write_text(
            manifest_path,
            yaml.safe_dump(
                json.loads(json.dumps(manifest, default=json_path_default)),
                sort_keys=False,
            ),
        )

    def validate_publication() -> tuple[bool, str]:
        try:
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
        except (OSError, yaml.YAMLError, TypeError):
            return False, f"Networks publication manifest is unreadable: {manifest_path}"
        valid, reason, _recorded_outputs = validate_recorded_publication(
            manifest,
            check_completion_time=publication_breadcrumb.is_file(),
        )
        if valid:
            current_config = json.loads(json.dumps(asdict(cfg), default=json_path_default))
            if manifest.get("config") != current_config:
                return False, "Published networks configuration differs from the requested workflow."
            current_fingerprint = selected_configuration_fingerprint()
            if (
                current_fingerprint is not None
                and manifest.get("configuration_fingerprint") != current_fingerprint
            ):
                return False, "Published networks configuration fingerprint is stale."
        return valid, reason

    runner.add_step(Step.directory_step(
        name="Publish Networks Directory",
        directory=out,
        breadcrumb=publication_breadcrumb,
        outputs=(manifest_path,),
        inputs=publication_inputs,
        force=bool(cfg.output.overwrite),
        action=write_publication,
        validate=validate_publication,
        breadcrumb_text="Networks directory publication complete\n",
        reset_directory=False,
        completion_boundary=completion_boundary,
    ))
    outputs["manifest"] = manifest_path
    return outputs


def run(
    cfg: ModuleConfig,
    *,
    runner: Runner | None = None,
) -> NetworkOutputs:
    runner_started = time.perf_counter()
    active_runner = runner or Runner(
        module_name="Networks Module",
        container=None, binds=(), logger=LOG, next_step=count(1).__next__
    )
    outputs = build_module(cfg, active_runner)
    with active_runner.run_context(started_at=runner_started):
        active_runner.execute()
    return outputs
