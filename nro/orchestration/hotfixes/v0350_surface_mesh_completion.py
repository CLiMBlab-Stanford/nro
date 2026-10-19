"""Invalidate completed dynconn artifacts whose CIFTI mesh contradicts anatomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nro.engine.scenes import incompatible_surface_data
from nro.orchestration.dependency_state import invalidate
from nro.orchestration.hotfixes import HotfixReport
from nro.orchestration.registry import utcnow

HOTFIX_ID = "v0350-surface-mesh-completion"
SUMMARY = "Rebuild dynconn outputs whose CIFTI mesh predates their published anatomy."


@dataclass(frozen=True)
class _Candidate:
    """Identify one completed dynconn item with incompatible display geometry."""

    work_item_id: int
    paths: tuple[Path, ...]


def _paths(root: Path, pattern: str) -> tuple[Path, ...]:
    """Return regular files matching one artifact pattern below a managed root."""
    try:
        return tuple(path for path in root.rglob(pattern) if path.is_file() and not path.is_symlink())
    except OSError:
        return ()


def _candidate(database, row: dict) -> _Candidate | None:
    """Return a candidate only when actual published meshes disagree."""
    dynconn_root = Path(str(row["output_root"]))
    ciftis = _paths(dynconn_root, "*.dtseries.nii")
    if not ciftis:
        return None
    anatomy_roots = database.execute(
        """WITH RECURSIVE upstream(id) AS (
               SELECT upstream_work_item_id FROM work_item_dependencies WHERE work_item_id=?
               UNION
               SELECT edge.upstream_work_item_id FROM work_item_dependencies edge
               JOIN upstream parent ON edge.work_item_id=parent.id
           )
           SELECT output_root FROM work_items
           WHERE id IN upstream AND module='anat' AND artifact_state='fresh'""",
        (int(row["id"]),),
    ).fetchall()
    surfaces = tuple(
        path
        for anatomy in anatomy_roots
        for path in _paths(Path(str(anatomy["output_root"])), "*_inflated.surf.gii")
    )
    mismatched = incompatible_surface_data(ciftis, surfaces)
    if not mismatched:
        return None
    return _Candidate(
        work_item_id=int(row["id"]),
        paths=tuple(sorted(mismatched)),
    )


def _candidates(database, projects: tuple[str, ...]) -> tuple[_Candidate, ...]:
    """Find only fresh dynconn artifacts with objectively incompatible meshes."""
    placeholders = ",".join("?" for _project in projects)
    rows = database.execute(
        f"""SELECT id,output_root FROM work_items
            WHERE project IN ({placeholders}) AND module='dynconn' AND artifact_state='fresh'
            ORDER BY id""",
        projects,
    ).fetchall()
    return tuple(
        candidate for row in rows if (candidate := _candidate(database, dict(row))) is not None
    )


def run(registry, *, projects: tuple[str, ...], execute: bool) -> HotfixReport:
    """Invalidate only dynconn products with a demonstrated mesh inconsistency."""
    selected = tuple(sorted(set(projects)))
    if not selected:
        raise ValueError("Hotfix requires at least one BIDS project")
    missing = [project for project in selected if not (registry.paths.bids_root / project).is_dir()]
    if missing:
        raise ValueError("Unknown BIDS project(s): " + ", ".join(missing))
    with registry.connection(write=execute) as database:
        candidates = _candidates(database, selected)
        if execute and candidates:
            invalidate(
                database,
                (candidate.work_item_id for candidate in candidates),
                now=utcnow(),
                reason="Published CIFTI mesh is incompatible with its current anatomical surfaces",
                error_type="IncompatibleSurfaceMesh",
                include_roots=True,
            )
    return HotfixReport(
        identifier=HOTFIX_ID,
        summary=SUMMARY,
        projects=selected,
        paths=tuple(path for candidate in candidates for path in candidate.paths),
        records=len(candidates),
        applied=execute,
    )
