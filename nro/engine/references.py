"""Portable references for public derivatives and durable ownership records."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from nro.engine.io import atomic_write_json

_BIDS_URI = re.compile(r"^bids:(?P<dataset>[^:]*):(?P<path>.*)$")
_SITE_URI = re.compile(r"^nro-site:(?P<root>[^:]+):(?P<path>.*)$")


def _absolute(path: str | Path) -> Path:
    """Normalize a host path lexically without dereferencing symlinks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid {label} path: {value!r}")
    return path


def nro_derivative_root(project_root: Path) -> Path:
    """Return the BIDS-Derivatives dataset owned by nro for one project."""
    return _absolute(project_root) / "derivatives" / "nro"


@dataclass(frozen=True)
class ReferenceRoots:
    """Roots needed to translate persistent references and runtime paths."""

    project: Path
    derivative: Path
    site: Mapping[str, Path]

    @classmethod
    def for_project(
        cls,
        project_root: Path,
        *,
        derivative_root: Path | None = None,
        site_roots: Mapping[str, str | Path] | None = None,
    ) -> "ReferenceRoots":
        """Construct normalized roots without reading global configuration."""
        project = _absolute(project_root)
        derivative = (
            nro_derivative_root(project)
            if derivative_root is None
            else _absolute(derivative_root)
        )
        return cls(
            project=project,
            derivative=derivative,
            site={
                str(name): _absolute(path)
                for name, path in (site_roots or {}).items()
            },
        )


def configured_reference_roots(
    project_root: Path, *, derivative_root: Path | None = None
) -> ReferenceRoots:
    """Build reference roots from the active immutable site configuration."""
    from nro.configuration.site import CHECKOUT, PATH_KEYS, settings

    site, _ = settings()
    roots = {
        key: Path(str(site[key]))
        for key in sorted(PATH_KEYS)
        if key in site and str(site[key]).strip()
    }
    roots["checkout"] = CHECKOUT
    return ReferenceRoots.for_project(
        project_root,
        derivative_root=derivative_root,
        site_roots=roots,
    )


def bids_uri(path: str | Path, roots: ReferenceRoots) -> str:
    """Encode a raw or nro-derivative file as a BIDS URI."""
    resolved = _absolute(path)
    try:
        relative = resolved.relative_to(roots.derivative)
    except ValueError:
        try:
            relative = resolved.relative_to(roots.project)
        except ValueError as error:
            raise ValueError(
                f"Path is outside the project and nro derivative roots: {resolved}"
            ) from error
        return f"bids:raw:{relative.as_posix()}"
    return f"bids::{relative.as_posix()}"


def resolve_bids_uri(value: str, roots: ReferenceRoots) -> Path:
    """Resolve a local or raw-project BIDS URI against explicit dataset roots."""
    match = _BIDS_URI.fullmatch(value)
    if match is None:
        raise ValueError(f"Not a BIDS URI: {value!r}")
    dataset = match.group("dataset")
    relative = _relative(match.group("path"), label="BIDS URI")
    if dataset == "":
        root = roots.derivative
    elif dataset == "raw":
        root = roots.project
    else:
        raise ValueError(f"Unknown BIDS dataset link {dataset!r}")
    path = root.joinpath(*relative.parts)
    path.relative_to(root)
    return path


def site_uri(path: str | Path, roots: ReferenceRoots) -> str:
    """Encode a site resource using the most specific configured root."""
    resolved = _absolute(path)
    matches: list[tuple[int, str, Path]] = []
    for name, root in roots.site.items():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), name, relative))
    if not matches:
        raise ValueError(f"Path is outside configured site roots: {resolved}")
    _length, name, relative = max(matches)
    member = relative.as_posix() if relative.parts else "."
    return f"nro-site:{name}:{member}"


def resolve_site_uri(value: str, roots: ReferenceRoots) -> Path:
    """Resolve one configured site-resource URI."""
    match = _SITE_URI.fullmatch(value)
    if match is None:
        raise ValueError(f"Not an nro site URI: {value!r}")
    name = match.group("root")
    if name not in roots.site:
        raise ValueError(f"Unknown nro site root {name!r}")
    member = match.group("path")
    relative = PurePosixPath() if member == "." else _relative(member, label="site URI")
    root = roots.site[name]
    path = root.joinpath(*relative.parts)
    path.relative_to(root)
    return path


def portable_path(path: str | Path, roots: ReferenceRoots, *, public: bool = False) -> str:
    """Encode a path for a public document or a private recovery receipt."""
    try:
        return bids_uri(path, roots)
    except ValueError:
        # Public records use the stable site-resource name, never its host path.
        # Consumers outside nro may retain the URI without resolving it.
        _ = public
    return site_uri(path, roots)


def resolve_reference(value: str, roots: ReferenceRoots) -> Path:
    """Resolve a supported portable file reference."""
    if value.startswith("bids:"):
        return resolve_bids_uri(value, roots)
    if value.startswith("nro-site:"):
        return resolve_site_uri(value, roots)
    raise ValueError(f"Unsupported portable reference: {value!r}")


def is_portable_reference(value: object) -> bool:
    """Return whether a value uses a supported portable file-reference scheme."""
    return isinstance(value, str) and value.startswith(("bids:", "nro-site:"))


def absolute_path_values(value: Any) -> tuple[str, ...]:
    """Return exact absolute host-path strings found in a structured value."""
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for member in item.values():
                visit(member)
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        elif (
            isinstance(item, str)
            and Path(item).is_absolute()
            and ":" not in item
            and "\n" not in item
        ):
            found.append(item)

    visit(value)
    return tuple(found)


_OMIT = object()


def _private_public_value(value: object, roots: ReferenceRoots) -> bool:
    """Return whether one path points into private orchestration storage."""
    if isinstance(value, str) and value.startswith(("nro-site:work:", "nro-site:registry:")):
        return True
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        return False
    path = _absolute(value)
    for name in ("work", "registry"):
        root = roots.site.get(name)
        if root is not None and path.is_relative_to(root):
            return True
    return False


def omit_private_path_values(value: Any, roots: ReferenceRoots) -> Any:
    """Remove exact WORK and registry path leaves from public metadata."""

    def visit(item: Any) -> Any:
        if _private_public_value(item, roots):
            return _OMIT
        if isinstance(item, dict):
            converted = {}
            for key, member in item.items():
                if key == "static_warp" and _private_public_value(member, roots):
                    converted[key] = "BOLDToT1wComposite"
                    continue
                value = visit(member)
                if value is not _OMIT:
                    converted[key] = value
            return converted
        if isinstance(item, (list, tuple)):
            return [converted for member in item if (converted := visit(member)) is not _OMIT]
        return item

    converted = visit(value)
    return None if converted is _OMIT else converted


def encode_path_values(value: Any, roots: ReferenceRoots, *, public: bool = False) -> Any:
    """Encode exact path-valued leaves in a typed payload.

    Path objects are always references. Absolute strings are references only when
    the entire string is a path; embedded shell fragments and descriptive text are
    left unchanged.
    """
    if isinstance(value, Path):
        return portable_path(value, roots, public=public)
    if isinstance(value, tuple):
        return [encode_path_values(item, roots, public=public) for item in value]
    if isinstance(value, list):
        return [encode_path_values(item, roots, public=public) for item in value]
    if isinstance(value, dict):
        return {
            key: encode_path_values(item, roots, public=public) for key, item in value.items()
        }
    if (
        isinstance(value, str)
        and Path(value).is_absolute()
        and ":" not in value
        and "\n" not in value
    ):
        return portable_path(value, roots, public=public)
    return value


def resolve_path_values(value: Any, roots: ReferenceRoots) -> Any:
    """Resolve portable-reference leaves while preserving non-reference values."""
    if is_portable_reference(value):
        return str(resolve_reference(value, roots))
    if isinstance(value, list):
        return [resolve_path_values(item, roots) for item in value]
    if isinstance(value, dict):
        return {key: resolve_path_values(item, roots) for key, item in value.items()}
    return value


def public_document_roots(path: Path) -> ReferenceRoots:
    """Infer reference roots for a document inside an nro derivative dataset."""
    resolved = _absolute(path)
    derivative = next(
        (
            parent
            for parent in (resolved.parent, *resolved.parents)
            if parent.name == "nro" and parent.parent.name == "derivatives"
        ),
        None,
    )
    if derivative is None:
        raise ValueError(f"Public nro document is outside derivatives/nro: {path}")
    output_project = derivative.parent.parent
    raw_project = output_project
    parts = output_project.parts
    inferred_site: dict[str, Path] = {}
    if "NRO_DEV" in parts:
        index = parts.index("NRO_DEV")
        if len(parts) >= index + 4 and parts[index + 2] == "BIDS":
            site_root = Path(*parts[:index])
            raw_project = site_root / "BIDS" / output_project.name
            inferred_site["development"] = site_root / "NRO_DEV"
            inferred_site["work"] = site_root / "WORK"
            inferred_site["bids"] = site_root / "BIDS"
    roots = configured_reference_roots(raw_project, derivative_root=derivative)
    if not inferred_site:
        return roots
    return ReferenceRoots.for_project(
        roots.project,
        derivative_root=roots.derivative,
        site_roots={**roots.site, **inferred_site},
    )


def portable_public_payload(path: Path, value: Any) -> Any:
    """Encode path-valued leaves for one nro-owned public document."""
    try:
        roots = public_document_roots(path)
    except ValueError:
        # Unit-level runners and third-party callers may construct manifests in
        # ordinary temporary directories. Portability is a property of the
        # controlled derivatives/nro namespace, not of arbitrary JSON/YAML.
        return value
    return encode_path_values(omit_private_path_values(value, roots), roots, public=True)


def resolve_public_payload(path: Path, value: Any) -> Any:
    """Resolve path-valued leaves read from one nro-owned public document."""
    try:
        roots = public_document_roots(path)
    except ValueError:
        return value
    return resolve_path_values(value, roots)


def derivative_dataset_description(project_root: Path, *, version: str) -> dict:
    """Return the canonical description of one nro derivative dataset."""
    path = nro_derivative_root(project_root) / "dataset_description.json"
    expected = {
        "Name": "nro derivatives",
        "BIDSVersion": "1.11.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "nro", "Version": str(version)}],
        "DatasetLinks": {"raw": "../.."},
    }
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid nro derivative dataset description: {path}") from error
        if not isinstance(current, dict):
            raise ValueError(f"Invalid nro derivative dataset description: {path}")
        generated = current.get("GeneratedBy")
        if isinstance(generated, list):
            retained = [
                item
                for item in generated
                if not isinstance(item, dict) or item.get("Name") != "nro"
            ]
        else:
            retained = []
        expected = {**current, **expected, "GeneratedBy": [expected["GeneratedBy"][0], *retained]}
    return expected


def ensure_derivative_dataset(project_root: Path, *, version: str) -> Path:
    """Create or validate the dataset description for nro derivatives."""
    root = nro_derivative_root(project_root)
    path = root / "dataset_description.json"
    expected = derivative_dataset_description(project_root, version=version)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) == expected:
        return path
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, expected, sort_keys=True, mode=0o664, durable=True)
    return path
