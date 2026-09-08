"""Task/node layout for named first-level model variants."""

from pathlib import Path
from nro.engine.targets import target_directory_name


def artifact_root(project_root: Path, config_id: str, model: str, space: str, smoothing: int) -> Path:
    """Return the task root shared by its variants and analysis levels."""
    return Path(project_root) / "derivatives" / "firstlevels" / config_id / target_directory_name(space, smoothing) / model.split("/")[0]


def instance_prefix(participant: str, model: str, space: str, smoothing: int) -> str:
    """Return a prefix unique to participant, task, variant and spatial target."""
    task, variant = model.split("/")
    return f"sub-{participant.removeprefix('sub-')}_model-{variant}_task-{task}_space-{space}_smoothing-{smoothing}mm"


def node_prefix(root: Path, prefix: str, node: dict, *, run_stem: str | None = None) -> Path:
    """Place node artifacts at node-LEVEL/sub-ID with collision-free filenames."""
    subject = prefix.split("_", 1)[0]
    name = f"{prefix}_node-{node['Name']}"
    if run_stem:
        from nro.engine.bids import parse_bids_entities
        entities = parse_bids_entities(run_stem)
        name += "".join(f"_{key}-{value}" for key, value in entities.items() if key not in {"sub", "task"})
    return Path(root) / f"node-{node['Level'].lower()}" / subject / name


def completion_path(root: Path, prefix: str) -> Path:
    """Return the fixed module manifest independent of which effects are estimable."""
    return Path(root) / "node-run" / prefix.split("_", 1)[0] / f"{prefix}_desc-firstlevels_manifest.json"
