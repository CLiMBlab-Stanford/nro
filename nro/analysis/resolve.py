"""Resolve analysis inputs to files. Planning and ``query`` share this resolver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from nro.analysis.artifacts import Artifact, InputSet, file_entities
from nro.analysis.manifest import InputSpec
from nro.engine.bids import matches_filter

_RUN_ENTITY_EXCLUDED = {"sub", "suffix", "extension"}


class ResolutionError(ValueError):
    """An input cannot be resolved to the files it declares."""


class UpstreamIndex(Protocol):
    """Read-only access to recorded upstream module lineages and work items."""

    def lineages(
        self, project: str, module: str, *, workflow: str | None, config: str | None
    ) -> list[dict]:
        """Return rows with ``directory_label``, ``config_id``, and ``workflow_id``."""

    def work_items(
        self, project: str, module: str, directory_label: str, participant: str
    ) -> list[dict]:
        """Return one participant's work items for one module lineage."""


class RegistryIndex:
    """Upstream index backed by the shared scheduler registry."""

    def __init__(self, registry_path: str | Path | None = None):
        """Use the site registry, or the registry at ``registry_path``."""
        self.registry_path = registry_path
        self._registries: dict[str, Any] = {}

    def _registry(self, project: str):
        from nro.orchestration.registry import Registry

        if project not in self._registries:
            self._registries[project] = Registry.for_project(
                project, registry_path=self.registry_path
            )
        return self._registries[project]

    def lineages(self, project, module, *, workflow, config):
        """Return the lineages that a workflow or configuration ID selects."""
        return self._registry(project).module_lineage_directories(
            module, workflow_id=workflow, config_id=config
        )

    def work_items(self, project, module, directory_label, participant):
        """Return one participant's recorded work items for one lineage."""
        return self._registry(project).lineage_work_items(
            project, module, directory_label, participant
        )


class SnapshotIndex:
    """Upstream index backed by the scheduler's cached status snapshot.

    Development installations may not open the production registry, so they
    read the last read model the scheduler published for the ``main`` branch.
    The snapshot can trail the registry by one scheduler update. A workflow
    selects the lineages of work items that one of its requests demanded.
    """

    def __init__(self, rows: list[dict] | None = None, *, branch: str = "main"):
        """Use the given rows, or read the site snapshot on first use."""
        self._rows = rows
        self.branch = branch

    @property
    def rows(self) -> list[dict]:
        """Return the snapshot's work-item rows for the branch."""
        if self._rows is None:
            from nro.orchestration.scheduler_bus import read_snapshot
            from nro.site.configuration import settings

            snapshot = read_snapshot(Path(settings()[0]["registry"]))
            if snapshot is None:
                raise ResolutionError("The scheduler has not published a status snapshot yet")
            branch = snapshot["branches"].get(self.branch) or {"rows": []}
            self._rows = list(branch["rows"])
        return self._rows

    def lineages(self, project, module, *, workflow, config):
        """Return the lineages that a workflow or configuration ID selects."""
        found = {}
        for row in self.rows:
            if row.get("module") != module or row.get("project") != project:
                continue
            workflows = [item for item in str(row.get("workflow_ids") or "").split(",") if item]
            if workflow is not None and workflow not in workflows:
                continue
            if config is not None and row.get("config_id") != config:
                continue
            for name in workflows or [""]:
                key = (str(row["directory_label"]), name)
                found[key] = {
                    "directory_label": key[0],
                    "config_id": str(row.get("config_id")),
                    "workflow_id": name,
                }
        return [found[key] for key in sorted(found)]

    def work_items(self, project, module, directory_label, participant):
        """Return one participant's work items for one lineage."""
        result = []
        for row in self.rows:
            if (
                row.get("project") != project
                or row.get("module") != module
                or row.get("directory_label") != directory_label
                or str(row.get("participant")) != participant
            ):
                continue
            entities = row.get("entities")
            if entities is None:
                entities = json.loads(row.get("entities_json") or "{}")
            result.append(
                {
                    "entities": entities,
                    "artifact_state": str(row["artifact_state"]),
                    "artifact_fingerprint": str(row["artifact_fingerprint"]),
                    "generation": int(row["current_generation"]),
                    "output_root": str(row["output_root"]),
                    "output_prefix": row.get("output_prefix"),
                    "expected_outputs": [],
                    "key": str(row.get("work_item_key")),
                }
            )
        return sorted(result, key=lambda item: item["key"])


def default_index() -> UpstreamIndex:
    """Return the registry index when direct access is allowed, else the snapshot."""
    from nro.site.configuration import require_execution_support

    try:
        require_execution_support()
    except ValueError:
        return SnapshotIndex()
    return RegistryIndex()


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one input for one work item.

    ``status`` is ``ok``, ``not_applicable`` (fewer matches than ``min``),
    ``waiting`` (required upstream work items are not fresh), or ``error``.
    """

    name: str
    status: str
    value: Artifact | InputSet | None = None
    message: str = ""
    lineage: str | None = None


def _signature(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _format(reference: str, parameters: Mapping[str, Any]) -> str:
    return reference.format(**{key: value for key, value in parameters.items()})


def resolve_lineage(
    spec: InputSpec, *, project: str, parameters: Mapping[str, Any], index: UpstreamIndex
) -> str:
    """Return the lineage directory that an input's workflow or config selects."""
    module = spec.module
    assert module is not None
    workflow = _format(spec.workflow, parameters) if spec.workflow is not None else None
    config = _format(spec.config, parameters) if spec.config is not None else None
    rows = index.lineages(project, module, workflow=workflow, config=config)
    if not rows:
        named = f"workflow {workflow!r}" if workflow is not None else f"configuration {config!r}"
        raise ResolutionError(
            f"input {spec.name}: no registered {module} lineage for {named} in project {project}; "
            f"run nro run {module} for that workflow first"
        )
    labels = sorted({row["directory_label"] for row in rows})
    if len(labels) > 1:
        workflows = ", ".join(sorted({row["workflow_id"] for row in rows}))
        raise ResolutionError(
            f"input {spec.name}: {module} configuration {config!r} leads to {len(labels)} "
            f"lineages through workflows {workflows}; name one with workflow:"
        )
    return labels[0]


def _upstream_files(item: Mapping[str, Any]) -> list[Path]:
    root = Path(item["output_root"])
    prefix = item.get("output_prefix")
    if not root.is_dir():
        return []
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and (prefix is None or path.name.startswith(str(prefix)))
    ]
    return paths


def _drop_sidecars(paths: Sequence[Path], select: Mapping[str, Any]) -> list[Path]:
    """Drop JSON sidecars of data files unless the selector names an extension."""
    if "extension" in select:
        return list(paths)
    from nro.analysis.artifacts import file_extension

    sidecars = set()
    for path in paths:
        extension = file_extension(path)
        if extension == ".json":
            continue
        # A NIfTI sidecar replaces the whole extension; a CIFTI sidecar
        # (``.dscalar.json``) replaces only the final ``.nii``.
        sidecars.add(str(path)[: -len(extension)] + ".json")
        if str(path).endswith(".nii"):
            sidecars.add(str(path).removesuffix(".nii") + ".json")
    return [path for path in paths if str(path) not in sidecars]


def _matches_run(entities: Mapping[str, str], run_entities: Mapping[str, str] | None) -> bool:
    if not run_entities:
        return True
    return all(
        entities.get(key) == value
        for key, value in run_entities.items()
        if key not in _RUN_ENTITY_EXCLUDED and key in entities
    )


def _module_candidates(
    spec: InputSpec,
    *,
    project: str,
    participant: str,
    parameters: Mapping[str, Any],
    run_entities: Mapping[str, str] | None,
    index: UpstreamIndex,
) -> tuple[list[Artifact], list[str], str]:
    lineage = resolve_lineage(spec, project=project, parameters=parameters, index=index)
    assert spec.module is not None
    candidates: list[Artifact] = []
    pending: list[str] = []
    for item in index.work_items(project, spec.module, lineage, participant):
        recorded = {str(key): str(value) for key, value in item["entities"].items()}
        if item["artifact_state"] != "fresh":
            # Filenames are unknown until the item runs, so any unfinished work
            # item of the lineage may still supply a match.
            pending.append(f"{item.get('key', '?')} ({item['artifact_state']})")
            continue
        for path in _drop_sidecars(_upstream_files(item), spec.select):
            entities = {**recorded, **file_entities(path)}
            if not matches_filter(entities, spec.select) or not _matches_run(
                entities, run_entities
            ):
                continue
            candidates.append(
                Artifact(
                    path=path,
                    entities=entities,
                    module=spec.module,
                    signature=_signature(
                        {
                            "lineage": lineage,
                            "fingerprint": item["artifact_fingerprint"],
                            "generation": item["generation"],
                            "file": path.name,
                        }
                    ),
                    project=project,
                    participant=participant,
                    lineage=lineage,
                )
            )
    return candidates, pending, lineage


def _raw_candidates(
    spec: InputSpec,
    *,
    project: str,
    participant: str,
    run_entities: Mapping[str, str] | None,
    bids_root: Path,
) -> list[Artifact]:
    project_root = bids_root / project
    subject = project_root / f"sub-{participant}"
    if spec.path is not None:
        pattern = spec.path.format(participant=participant, project=project)
        paths = sorted(path for path in project_root.glob(pattern) if path.is_file())
    else:
        from nro.engine.bids import discover_raw_runs

        paths = [run.path for run in discover_raw_runs(subject)] if subject.is_dir() else []
    result = []
    for path in _drop_sidecars(paths, spec.select):
        entities = file_entities(path)
        if not matches_filter(entities, spec.select) or not _matches_run(entities, run_entities):
            continue
        status = path.stat()
        result.append(
            Artifact(
                path=path,
                entities=entities,
                module=None,
                signature=_signature(
                    {"path": str(path), "size": status.st_size, "mtime_ns": status.st_mtime_ns}
                ),
                project=project,
                participant=participant,
            )
        )
    return result


def resolve_input(
    spec: InputSpec,
    *,
    project: str,
    participant: str,
    parameters: Mapping[str, Any],
    index: UpstreamIndex,
    bids_root: Path,
    run_entities: Mapping[str, str] | None = None,
) -> Resolution:
    """Resolve one input for one work item without raising for data conditions."""
    participant = participant.removeprefix("sub-")
    pending: list[str] = []
    lineage = None
    try:
        if spec.module is None:
            candidates = _raw_candidates(
                spec,
                project=project,
                participant=participant,
                run_entities=run_entities,
                bids_root=bids_root,
            )
        else:
            candidates, pending, lineage = _module_candidates(
                spec,
                project=project,
                participant=participant,
                parameters=parameters,
                run_entities=run_entities,
                index=index,
            )
    except ResolutionError as error:
        return Resolution(spec.name, "error", message=str(error))
    candidates.sort(key=lambda item: str(item.path))
    listed = "".join(f"\n  {item.path}" for item in candidates)
    where = f"sub-{participant} in project {project}"
    if spec.many:
        if spec.maximum is not None and len(candidates) > spec.maximum:
            return Resolution(
                spec.name,
                "error",
                message=f"input {spec.name}: {len(candidates)} files match {where}, "
                f"more than max {spec.maximum}:{listed}",
                lineage=lineage,
            )
        if len(candidates) < spec.minimum:
            if pending:
                return _waiting(spec, where, pending, lineage)
            if spec.optional and not candidates:
                return Resolution(spec.name, "ok", None, lineage=lineage)
            return Resolution(
                spec.name,
                "not_applicable",
                message=f"input {spec.name}: {len(candidates)} files match {where}, "
                f"fewer than min {spec.minimum}",
                lineage=lineage,
            )
        return Resolution(spec.name, "ok", InputSet(tuple(candidates)), lineage=lineage)
    if len(candidates) > 1:
        return Resolution(
            spec.name,
            "error",
            message=f"input {spec.name}: {len(candidates)} files match {where}; "
            f"narrow select or declare many: true:{listed}",
            lineage=lineage,
        )
    if not candidates:
        if pending:
            return _waiting(spec, where, pending, lineage)
        if spec.optional:
            return Resolution(spec.name, "ok", None, lineage=lineage)
        return Resolution(
            spec.name,
            "error",
            message=f"input {spec.name}: no file matches {where} with select {dict(spec.select)}",
            lineage=lineage,
        )
    return Resolution(spec.name, "ok", candidates[0], lineage=lineage)


def _waiting(spec: InputSpec, where: str, pending: list[str], lineage: str | None) -> Resolution:
    return Resolution(
        spec.name,
        "waiting",
        message=f"input {spec.name}: upstream work items for {where} are not fresh: "
        + ", ".join(pending[:5])
        + (" ..." if len(pending) > 5 else ""),
        lineage=lineage,
    )


def query(
    *,
    project: str,
    participant: str,
    module: str,
    workflow: str | None = None,
    config: str | None = None,
    select: Mapping[str, Any] | None = None,
    many: bool = False,
    optional: bool = False,
    min: int = 1,  # noqa: A002 - mirrors the manifest key
    max: int | None = None,  # noqa: A002 - mirrors the manifest key
    index: UpstreamIndex | None = None,
    bids_root: Path | None = None,
) -> Artifact | InputSet | None:
    """Resolve files the way a planned work item would, without planning anything.

    ``module`` is an nro module name or ``"raw"``. Raises ``ResolutionError`` for a
    missing, ambiguous, insufficient, or not-yet-fresh match.
    """
    from nro.analysis.manifest import _input

    value: dict[str, Any] = {"from": module, "select": dict(select or {}), "many": many}
    if optional:
        value["optional"] = True
    if many:
        value["min"] = min
        if max is not None:
            value["max"] = max
    if workflow is not None:
        value["workflow"] = workflow
    if config is not None:
        value["config"] = config
    spec = _input("query", value, parameters={})
    if bids_root is None:
        from nro.site.configuration import bids_root as site_bids_root

        bids_root = site_bids_root()
    resolution = resolve_input(
        spec,
        project=project,
        participant=participant,
        parameters={},
        index=index or default_index(),
        bids_root=Path(bids_root),
    )
    if resolution.status != "ok":
        raise ResolutionError(resolution.message)
    return resolution.value
