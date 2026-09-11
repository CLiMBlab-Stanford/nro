"""Task/node layout for named first-level model variants."""

from pathlib import Path


def artifact_root(
    project_root: Path,
    config_id: str,
    participant: str,
) -> Path:
    """Return one participant's shared firstlevels directory."""
    return (
        Path(project_root)
        / "derivatives"
        / "firstlevels"
        / config_id
        / f"sub-{participant.removeprefix('sub-')}"
    )


def instance_prefix(participant: str, model: str, space: str, smoothing: int) -> str:
    """Return a prefix unique to participant, task, variant and spatial target."""
    task, variant = model.split("/")
    return f"sub-{participant.removeprefix('sub-')}_model-{variant}_task-{task}_space-{space}_smoothing-{smoothing}mm"


def _task_directory(root: Path, prefix: str) -> Path:
    from nro.engine.bids import parse_bids_entities

    task = parse_bids_entities(prefix).get("task")
    if not task:
        raise ValueError(f"Firstlevels prefix lacks a task entity: {prefix}")
    return Path(root) / f"task-{task}"


def node_prefix(root: Path, prefix: str, node: dict, *, run_stem: str | None = None) -> Path:
    """Place node artifacts below a participant task with collision-free filenames."""
    name = f"{prefix}_node-{node['Name']}"
    task_directory = _task_directory(root, prefix)
    directory = task_directory / f"node-{node['Level'].lower()}"
    if run_stem:
        from nro.engine.bids import parse_bids_entities

        entities = parse_bids_entities(run_stem)
        if session := entities.get("ses"):
            directory = (
                Path(root)
                / f"ses-{session}"
                / task_directory.name
                / f"node-{node['Level'].lower()}"
            )
        name += "".join(
            f"_{key}-{value}" for key, value in entities.items() if key not in {"sub", "task"}
        )
    return directory / name


def completion_path(root: Path, prefix: str) -> Path:
    """Return the fixed module manifest independent of which effects are estimable."""
    return _task_directory(root, prefix) / "node-run" / f"{prefix}_desc-firstlevels_manifest.json"
