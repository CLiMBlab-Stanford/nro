"""Index completed derivatives that the browser viewer can display.

The catalog reads one branch's cached registry snapshot and the publication
manifests it names. It exposes files through opaque identifiers so the server
never resolves a client-supplied path.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from nro.engine.bids import parse_bids_entities
from nro.engine.scenes import cortical_vertex_counts, surface_vertex_counts
from nro.engine.workbench import SURFACE_TYPES, surface_inventory

HEMISPHERES = ("L", "R")
GIFTI_METRICS = (".shape.gii", ".func.gii")
GIFTI_LABELS = (".label.gii",)


def _identifier(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SurfaceFamily:
    """One bilateral set of white, pial, midthickness, and inflated surfaces."""

    id: str
    label: str
    module: str
    space: str
    paths: dict[tuple[str, str], Path]
    vertex_counts: dict[str, int]

    def public(self) -> dict[str, object]:
        """Describe the family for the browser without exposing file paths."""
        return {
            "id": self.id,
            "label": self.label,
            "module": self.module,
            "space": self.space,
            "kinds": [kind for kind in SURFACE_TYPES if ("L", kind) in self.paths],
            "vertex_counts": self.vertex_counts,
        }


@dataclass(frozen=True)
class Layer:
    """One displayable cortical file.

    ``kind`` is ``metric`` for continuous maps, ``label`` for parcellations, and
    ``connectivity`` for dense time series that support seed correlation. GIFTI
    layers address each hemisphere through its own file; CIFTI layers use one
    file for both hemispheres.
    """

    id: str
    kind: str
    label: str
    module: str
    space: str | None
    paths: dict[str, Path]
    maps: tuple[str, ...]
    vertex_counts: dict[str, int]
    size_bytes: int = 0

    def public(self) -> dict[str, object]:
        """Describe the layer for the browser without exposing file paths."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "module": self.module,
            "space": self.space,
            "maps": list(self.maps),
            "vertex_counts": self.vertex_counts,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ParticipantView:
    """Everything the viewer can show for one project participant."""

    project: str
    participant: str
    surfaces: tuple[SurfaceFamily, ...]
    layers: tuple[Layer, ...]

    def public(self) -> dict[str, object]:
        """Describe the participant's surfaces and layers for the browser."""
        return {
            "project": self.project,
            "participant": self.participant,
            "surfaces": [surface.public() for surface in self.surfaces],
            "layers": [layer.public() for layer in self.layers],
        }


def _display_label(path: Path, participant: str) -> str:
    name = path.name
    for suffix in (".dtseries.nii", ".dscalar.nii", ".dlabel.nii", *GIFTI_METRICS, *GIFTI_LABELS):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    tokens = [
        token
        for token in name.split("_")
        if token != f"sub-{participant}" and not token.startswith(("hemi-", "space-"))
    ]
    return " ".join(tokens) or path.name


def _gifti_maps(path: Path) -> tuple[str, ...]:
    import nibabel as nib

    image = nib.load(str(path))
    names = []
    for index, array in enumerate(image.darrays):
        meta = array.meta or {}
        names.append(str(meta.get("Name") or f"map {index + 1}"))
    return tuple(names)


def _cifti_maps(path: Path) -> tuple[str, ...]:
    import nibabel as nib

    axis = nib.load(str(path)).header.get_axis(0)
    names = getattr(axis, "name", None)
    if names is None:
        return ("",)
    return tuple(str(name) or f"map {index + 1}" for index, name in enumerate(names))


def _layer_kind(path: Path) -> str | None:
    name = path.name
    if name.endswith(GIFTI_METRICS) or name.endswith(".dscalar.nii"):
        return "metric"
    if name.endswith(GIFTI_LABELS) or name.endswith(".dlabel.nii"):
        return "label"
    if name.endswith(".dtseries.nii"):
        return "connectivity"
    return None


def _hemisphere(path: Path) -> str | None:
    if "_hemi-L_" in path.name:
        return "L"
    if "_hemi-R_" in path.name:
        return "R"
    return None


def build_layers(
    sources: Iterable[tuple[str, Path, str | None]], participant: str
) -> tuple[Layer, ...]:
    """Group derivative files into displayable layers.

    Each source is ``(module, path, space)``. Left and right GIFTI files with the
    same name apart from their hemisphere entity form one layer. CIFTI files
    without cortical vertices and files of other types are omitted.
    """
    gifti: dict[tuple[str, Path, str], dict[str, Path]] = {}
    gifti_space: dict[tuple[str, Path, str], str | None] = {}
    layers: list[Layer] = []
    for module, path, space in sources:
        kind = _layer_kind(path)
        if kind is None:
            continue
        if path.name.endswith(".gii"):
            hemisphere = _hemisphere(path)
            if hemisphere is None:
                continue
            key = (module, path.parent, path.name.replace(f"_hemi-{hemisphere}_", "_hemi-X_"))
            gifti.setdefault(key, {})[hemisphere] = path
            gifti_space[key] = space
            continue
        counts = cortical_vertex_counts(path)
        if not counts:
            continue
        try:
            maps = ("",) if kind == "connectivity" else _cifti_maps(path)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        layers.append(
            Layer(
                id=_identifier("layer", path),
                kind=kind,
                label=_display_label(path, participant),
                module=module,
                space=space,
                paths={"LR": path},
                maps=maps,
                vertex_counts=counts,
                size_bytes=path.stat().st_size,
            )
        )
    for key, paths in gifti.items():
        module = key[0]
        first = paths.get("L") or paths["R"]
        try:
            maps = _gifti_maps(first)
            counts = {
                hemisphere: cortical_vertex_counts(path)[hemisphere]
                for hemisphere, path in paths.items()
            }
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            continue
        layers.append(
            Layer(
                id=_identifier("layer", *sorted(map(str, paths.values()))),
                kind=_layer_kind(first) or "metric",
                label=_display_label(first, participant),
                module=module,
                space=gifti_space[key],
                paths=dict(sorted(paths.items())),
                maps=maps,
                vertex_counts=counts,
                size_bytes=sum(path.stat().st_size for path in paths.values()),
            )
        )
    return tuple(sorted(layers, key=lambda layer: (layer.module, layer.label)))


def build_surfaces(
    sources: Iterable[tuple[str, Path, str | None, Path]],
) -> tuple[SurfaceFamily, ...]:
    """Group display-surface files into complete bilateral families.

    Each source is ``(module, path, space, output_root)``. Surfaces without a
    space entity are native-space surfaces. Incomplete families are omitted.
    """
    grouped: dict[tuple[str, str, Path], list[Path]] = {}
    for module, path, space, root in sources:
        grouped.setdefault((module, space or "fsnative", root), []).append(path)
    families = []
    for (module, space, root), paths in grouped.items():
        try:
            inventory = surface_inventory(paths)
        except ValueError:
            continue
        label = f"{module} {space}"
        if any(family.label == label for family in families):
            label = f"{label} ({root.parent.name if root.name == 'anat' else root.name})"
        families.append(
            SurfaceFamily(
                id=_identifier("surface", *sorted(map(str, paths))),
                label=label,
                module=module,
                space=space,
                paths=inventory,
                vertex_counts=surface_vertex_counts(tuple(inventory.values())),
            )
        )
    return tuple(families)


@dataclass
class Catalog:
    """Lazily index participants from cached registry rows."""

    rows: list[dict]
    _views: dict[tuple[str, str], ParticipantView] = field(default_factory=dict)
    _surfaces: dict[str, SurfaceFamily] = field(default_factory=dict)
    _layers: dict[str, Layer] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def participants(self) -> list[dict[str, object]]:
        """List projects and participants with at least one completed derivative."""
        found: dict[tuple[str, str], set[str]] = {}
        for row in self._completed():
            found.setdefault((str(row["project"]), str(row["participant"])), set()).add(
                str(row["module"])
            )
        return [
            {"project": project, "participant": participant, "modules": sorted(modules)}
            for (project, participant), modules in sorted(found.items())
        ]

    def view(self, project: str, participant: str) -> ParticipantView:
        """Return the cached view for one participant, reading manifests on first use.

        Raises KeyError when the participant has no completed derivatives.
        """
        key = (project, participant)
        with self._lock:
            if key in self._views:
                return self._views[key]
        rows = [
            row
            for row in self._completed()
            if row.get("project") == project and row.get("participant") == participant
        ]
        if not rows:
            raise KeyError(f"No completed derivatives for {project} sub-{participant}")
        from nro.bin.scene import _collect

        data, surface_sources, _lesion = _collect(rows, include_heavy=True)
        layers = build_layers(
            (
                (source.module, source.path, entities.get("space") or source.space)
                for source, entities in data
                if not source.path.name.endswith(".pconn.nii")
            ),
            participant,
        )
        surfaces = build_surfaces(
            (source.module, source.path, source.space, source.output_root)
            for source in surface_sources
        )
        view = ParticipantView(project, participant, surfaces, layers)
        with self._lock:
            self._views[key] = view
            self._surfaces.update((surface.id, surface) for surface in surfaces)
            self._layers.update((layer.id, layer) for layer in layers)
        return view

    def refresh(self, rows: list[dict]) -> None:
        """Replace the registry rows and forget every cached participant view.

        Identifiers from earlier views stop resolving, so clients must reload
        the participant they are showing.
        """
        with self._lock:
            self.rows = rows
            self._views.clear()
            self._surfaces.clear()
            self._layers.clear()

    def register(self, layer: Layer) -> None:
        """Make a layer resolvable by identifier, as listed layers are."""
        with self._lock:
            self._layers[layer.id] = layer

    def surface(self, identifier: str) -> SurfaceFamily:
        """Return a surface family from a previously listed view."""
        with self._lock:
            return self._surfaces[identifier]

    def layer(self, identifier: str) -> Layer:
        """Return a layer from a previously listed view."""
        with self._lock:
            return self._layers[identifier]

    def _completed(self) -> Iterable[dict]:
        return (row for row in self.rows if row.get("status") in {"Success", "success"})


class Catalogs:
    """Hold one lazily built catalog per branch for a shared server.

    ``rows_for`` reads registry rows for a branch name, or for the server
    checkout's own branch when the name is empty.
    """

    def __init__(
        self,
        rows_for: Callable[[str], list[dict]],
        project_root: Callable[[str], Path] | None = None,
    ) -> None:
        """Start with no catalogs; each is built on first use.

        ``project_root`` maps a project name to its BIDS directory; by default
        it is ``<site bids>/<project>``.
        """
        self._rows_for = rows_for
        self._project_root = project_root or _site_project_root
        self._catalogs: dict[str, Catalog] = {}
        self._lock = threading.Lock()

    def get(self, branch: str = "") -> Catalog:
        """Return the catalog for a branch, reading its registry rows on first use."""
        with self._lock:
            if branch in self._catalogs:
                return self._catalogs[branch]
        catalog = Catalog(self._rows_for(branch))
        with self._lock:
            return self._catalogs.setdefault(branch, catalog)

    def refresh(self, branch: str = "") -> Catalog:
        """Reread one branch's registry rows and forget its cached views."""
        catalog = self.get(branch)
        catalog.refresh(self._rows_for(branch))
        return catalog

    def open_file(self, project: str, relative: str, *, branch: str = "") -> Layer:
        """Index one displayable file named relative to a project's BIDS directory.

        A GIFTI file with a hemisphere entity is paired with its other
        hemisphere when that file exists. The resolved path, after following
        symbolic links, must lie inside the project directory.

        Raises PermissionError for a path outside the project, FileNotFoundError
        for a missing file, and ValueError for a file the viewer cannot show.
        """
        root = Path(self._project_root(project)).resolve()
        candidate = (root / relative.strip().lstrip("/")).resolve()
        if not candidate.is_relative_to(root):
            raise PermissionError(f"{relative} is outside the {project} BIDS directory")
        if not candidate.is_file():
            raise FileNotFoundError(f"No file {relative} in the {project} BIDS directory")
        paths = [candidate]
        hemisphere = _hemisphere(candidate)
        if hemisphere and candidate.name.endswith(".gii"):
            other = "R" if hemisphere == "L" else "L"
            partner = candidate.with_name(
                candidate.name.replace(f"_hemi-{hemisphere}_", f"_hemi-{other}_")
            )
            if partner.is_file() and partner.resolve().is_relative_to(root):
                paths.append(partner)
        entities = parse_bids_entities(candidate.name)
        participant = entities.get("sub", "")
        layers = build_layers(
            (("file", path, entities.get("space")) for path in paths), participant
        )
        if not layers:
            raise ValueError(
                f"{relative} is not a cortical GIFTI metric or label, or a CIFTI file "
                "with cortical vertices"
            )
        layer = layers[0]
        self.get(branch).register(layer)
        return layer

    def surface(self, identifier: str) -> SurfaceFamily:
        """Find a surface family listed by any branch's catalog."""
        return self._find("surface", identifier)

    def layer(self, identifier: str) -> Layer:
        """Find a layer listed by any branch's catalog."""
        return self._find("layer", identifier)

    def _find(self, kind: str, identifier: str):
        with self._lock:
            catalogs = tuple(self._catalogs.values())
        for catalog in catalogs:
            try:
                return getattr(catalog, kind)(identifier)
            except KeyError:
                continue
        raise KeyError(f"Unknown {kind} {identifier}; reload the page")


def _site_project_root(project: str) -> Path:
    from nro.site.configuration import settings

    return Path(settings()[0]["bids"]) / project


def branch_names() -> dict[str, object]:
    """List registered branches and name the one the server checkout belongs to."""
    from nro.orchestration.branch_store import BranchStore
    from nro.site.configuration import CHECKOUT, settings

    records = BranchStore(Path(settings()[0]["registry"])).read().topology.records
    current = next(
        (
            name
            for name, record in records.items()
            if Path(CHECKOUT).absolute() in {Path(path).absolute() for path in record.checkouts}
        ),
        None,
    )
    names = sorted(name for name, record in records.items() if not record.retired)
    return {"branches": names, "current": current}


def branch_rows(branch: str | None) -> list[dict]:
    """Read cached registry rows for a named branch, or for the current checkout.

    Raises SystemExit when the branch is unknown or the registry is not active.
    """
    from nro.orchestration.branch_store import BranchStore
    from nro.orchestration.scheduler_bus import read_branch_snapshot
    from nro.site.configuration import CHECKOUT, settings

    control = Path(settings()[0]["registry"])
    checkout = CHECKOUT
    if branch:
        records = BranchStore(control).read().topology.records
        if branch not in records:
            raise SystemExit(
                f"Unknown branch {branch!r}; choose from " + ", ".join(sorted(records))
            )
        checkouts = records[branch].checkouts
        if not checkouts:
            raise SystemExit(f"Branch {branch!r} has no registered checkout")
        checkout = checkouts[0]
    try:
        report = read_branch_snapshot(control, checkout)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"Cannot read the registry snapshot for {checkout}: {error}") from None
    visible = set(report["visible_ids"])
    return [row for row in report["rows"] if row["id"] in visible]
