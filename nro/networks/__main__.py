import argparse
import json
import logging
import time
from itertools import count
from pathlib import Path

from nro.configuration.paths import BIDS_PATH, WORK_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.engine.bids import discover_raw_runs, matches_filter
from nro.networks.config import (
    ClusteringConfig,
    ConnectivityConfig,
    ConsensusConfig,
    IcaConfig,
    InputsConfig,
    OslomConfig,
    OutputConfig,
    ModuleConfig,
    LabelingConfig,
)
from nro.networks.module import build_module
from nro.networks.paths import fixed_output_paths
from nro.networks.targets import (
    MicroparcellationTarget,
    discover_microparcellation_targets,
)
from nro.orchestration.runtime import (
    load_runtime_workflow_snapshot,
    select_runtime_config,
    selected_configuration_fingerprint,
)
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step
from nro.engine.publication import write_json_atomic
from nro.engine.paths import anatomical_manifest_path, optional_path
from nro.engine.cli import stderr
from nro.engine.targets import (
    DEFAULT_SMOOTHING_MM,
    DEFAULT_SPACE,
    smoothing_entity_value,
    target_directory_name,
)


def target_output_names(
    base_prefix: str, space: str, smoothing_mm: int
) -> tuple[str, str]:
    target = target_directory_name(space, smoothing_mm)
    prefix = (
        f"{base_prefix}_space-{space}_"
        f"smoothing-{smoothing_entity_value(smoothing_mm)}"
    )
    return target, prefix


def _target_module_config(
    target: MicroparcellationTarget,
    output: Path,
    work: Path,
    prefix: str,
    config: dict,
    *,
    overwrite: bool | None,
    anatomical_manifest: Path | None,
    anatomical_reference: Path | None,
    mni_to_t1_transform: Path | None,
) -> ModuleConfig:
    oslom = config["oslom"].copy()
    oslom["executable"] = optional_path(oslom.get("executable"))
    oslom["initial_partition"] = optional_path(oslom.get("initial_partition"))
    oslom["extra_args"] = tuple(oslom["extra_args"])
    return ModuleConfig(
        inputs=InputsConfig(
            microparcellation_manifest=target.manifest,
            microparcels=target.microparcels,
            connectivity=target.connectivity,
            domain=target.domain,
            space=target.space,
            smoothing_mm=target.smoothing_mm,
            source_surfaces=target.source_surfaces,
            anatomical_manifest=anatomical_manifest,
            anatomical_reference=anatomical_reference,
            mni_to_t1_transform=mni_to_t1_transform,
        ),
        output=OutputConfig(
            directory=output,
            work_directory=work,
            prefix=prefix,
            overwrite=config["overwrite"] if overwrite is None else overwrite,
        ),
        connectivity=ConnectivityConfig(**config["connectivity"]),
        parcellation_strategy=str(config["parcellation_strategy"]),
        ica=IcaConfig(**config["ica"]),
        clustering=ClusteringConfig(**config["clustering"]),
        oslom=OslomConfig(**oslom),
        consensus=ConsensusConfig(**config["consensus"]),
        labeling=LabelingConfig(**config["labeling"]),
    )


def make_target_config(
    project,
    participant,
    networks_id,
    config,
    *,
    overwrite=None,
    micro_manifest=None,
    anatomical_reference=None,
    mni_to_t1_transform=None,
    anatomical_manifest=None,
):
    """Construct a networks config for one requested space and smoothing level."""
    subject_id = f"sub-{participant}"
    if micro_manifest is None:
        raise ValueError(
            "micro_manifest must be derived from the requested space and smoothing"
        )
    targets = discover_microparcellation_targets((micro_manifest,))
    if len(targets) != 1:
        raise ValueError(f"Expected one microparcellation target, found {len(targets)}")
    target = targets[0]
    output_base = Path(config.get("output_dir") or (
        Path(BIDS_PATH) / project / "derivatives" / "networks" / networks_id
    ))
    work_base = (
        Path(WORK_PATH)
        / project
        / "derivatives"
        / "networks"
        / networks_id
    )
    target_dir, target_prefix = target_output_names(
        config.get("prefix") or subject_id,
        target.space,
        target.smoothing_mm,
    )
    return (
        target,
        _target_module_config(
            target,
            output_base / target_dir / subject_id,
            work_base / target_dir / subject_id,
            target_prefix,
            config,
            overwrite=overwrite,
            anatomical_manifest=anatomical_manifest,
            anatomical_reference=anatomical_reference,
            mni_to_t1_transform=mni_to_t1_transform,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Estimate networks for one space and smoothing level"
    )
    parser.add_argument("-p", "--participant", required=True, help="BIDS participant ID")
    parser.add_argument("-P", "--project", required=True)
    parser.add_argument("-s", "--space", default=DEFAULT_SPACE)
    parser.add_argument(
        "-S", "--smoothing", type=int, default=DEFAULT_SMOOTHING_MM, metavar="MM"
    )
    parser.add_argument("-w", "--workflow", default="main")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.smoothing < 0:
        raise SystemExit("--smoothing must be a nonnegative integer FWHM in mm")
    smoothing_mm = args.smoothing

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    participant = args.participant.replace("sub-", "")
    runtime_config = select_runtime_config(
        project=args.project,
        workflow_id=args.workflow,
        derivative_class="networks",
    )
    networks_id, config = load_runtime_configuration(runtime_config, "networks")
    snapshot = load_runtime_workflow_snapshot(runtime_config)
    configurations = snapshot["configurations"]
    micro_configuration = configurations["microparcellation"]
    micro_config = micro_configuration.get("resolved") or {}
    participant_id = f"sub-{participant}"
    output_spaces = tuple(
        str(value)
        for value in configurations["preprocessing"]["resolved"]["func"]["output_spaces"]
    )
    if args.space not in output_spaces:
        raise SystemExit(
            f"space-{args.space} is not published by preprocessing; "
            f"choose from {', '.join(output_spaces)}"
        )
    preprocessing_id = str(configurations["preprocessing"]["directory"])
    anat_manifest_path = anatomical_manifest_path(
        participant_id,
        project=args.project,
        preprocessing_id=preprocessing_id,
    )
    if not anat_manifest_path.is_file():
        raise FileNotFoundError(f"Missing anatomical publication manifest: {anat_manifest_path}")
    anat_manifest = json.loads(anat_manifest_path.read_text(encoding="utf-8"))
    anat_outputs = anat_manifest.get("outputs") or {}
    anatomical_reference_value = anat_outputs.get("brain_image")
    mni_to_t1_value = (anat_outputs.get("xfms") or {}).get("mni_to_t1")
    if not anatomical_reference_value or not mni_to_t1_value:
        raise ValueError(
            f"Anatomical manifest lacks outputs.brain_image or outputs.xfms.mni_to_t1: "
            f"{anat_manifest_path}"
        )
    anatomical_reference = Path(str(anatomical_reference_value))
    mni_to_t1_transform = Path(str(mni_to_t1_value))
    missing_anatomical_inputs = [
        str(path)
        for path in (anatomical_reference, mni_to_t1_transform)
        if not path.is_file()
    ]
    if missing_anatomical_inputs:
        raise FileNotFoundError(
            "Anatomical manifest records missing labeling input(s): "
            + ", ".join(missing_anatomical_inputs)
        )
    source_runs = tuple(
        run
        for run in discover_raw_runs(Path(BIDS_PATH) / args.project / participant_id)
        if matches_filter(run.entities, micro_config.get("input_filter"))
    )
    micro_base = Path(
        micro_config.get("output_dir")
        or Path(BIDS_PATH) / args.project / "derivatives" / "microparcellation"
        / config["microparcellation_directory"]
    ).expanduser().resolve()
    micro_prefix = str(micro_config.get("prefix") or participant_id)
    micro_subject = (
        micro_base
        / target_directory_name(args.space, smoothing_mm)
        / participant_id
    )
    micro_target_prefix = (
        f"{micro_prefix}_space-{args.space}_"
        f"smoothing-{smoothing_entity_value(smoothing_mm)}"
    )
    from nro.microparcellation.paths import output_paths as micro_output_paths

    micro_paths = micro_output_paths(micro_subject, micro_target_prefix)
    micro_manifest = micro_paths["manifest"]
    publication_index = (
        micro_paths["index"]
    )
    if not publication_index.is_file():
        raise FileNotFoundError(
            f"Missing microparcellation publication index: {publication_index}"
        )
    index = json.loads(publication_index.read_text(encoding="utf-8"))
    if (
        index.get("space") != args.space
        or index.get("smoothing_fwhm_mm") != smoothing_mm
        or str(Path(str(index.get("target_manifest", ""))).resolve())
        != str(micro_manifest.resolve())
    ):
        raise ValueError(
            "Microparcellation publication manifest does not match the requested "
            f"space-{args.space} smoothing-{smoothing_mm}mm target: {publication_index}"
        )
    target, cfg = make_target_config(
        args.project,
        participant,
        networks_id,
        config,
        overwrite=True if args.overwrite else None,
        micro_manifest=micro_manifest,
        anatomical_manifest=anat_manifest_path,
        anatomical_reference=anatomical_reference,
        mni_to_t1_transform=mni_to_t1_transform,
    )
    stderr(
        f"Planned networks for {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm\n"
    )
    runner = Runner(
        module_name="Subject Networks Module",
        container=None,
        binds=(),
        logger=logging.getLogger("networks"),
        next_step=count(1).__next__,
    )
    result = build_module(cfg, runner, completion_boundary=False)
    manifest = Path(result["manifest"])
    output_index = fixed_output_paths(cfg.output.directory, cfg.output.prefix)["index"]

    def published_paths(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from published_paths(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from published_paths(item)

    def index_payload() -> dict[str, object]:
        import yaml

        publication = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        public_outputs = [
            path
            for value in (publication.get("outputs") or {}).values()
            for path in published_paths(value)
        ]
        return {
            "manifest_version": 1,
            "module": "networks",
            "participant": participant_id,
            "domain": target.domain,
            "space": target.space,
            "smoothing_fwhm_mm": target.smoothing_mm,
            "source_runs": [source.stem for source in source_runs],
            "target_manifest": str(manifest),
            "public_outputs": [*public_outputs, str(manifest)],
            "configuration_fingerprint": selected_configuration_fingerprint(),
            "complete": True,
        }

    def validate_index() -> tuple[bool, str]:
        try:
            current = json.loads(output_index.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False, "Networks publication index is missing or unreadable."
        if current != index_payload():
            return False, "Networks publication index differs from the requested module."
        return True, "Networks publication index is complete and current."

    runner.add_step(
        Step.python(
            name="Write Networks Publication Index",
            outputs=(output_index,),
            inputs=(manifest,),
            force=bool(args.overwrite),
            action=lambda: write_json_atomic(output_index, index_payload()),
            validate=validate_index,
            completion_boundary=True,
        )
    )
    started = time.perf_counter()
    stderr(
        f"Running networks for {target.domain} space-{target.space} "
        f"smoothing-{target.smoothing_mm}mm\n"
    )
    with runner.run_context(started_at=started):
        runner.execute()
    for name, path in result.items():
        print(
            f"{target.domain}/space-{target.space}/"
            f"smoothing-{target.smoothing_mm}mm/{name}: {path}"
        )


if __name__ == "__main__":
    main()
