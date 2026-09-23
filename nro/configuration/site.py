"""Resolve protected site definitions and immutable execution snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Callable

import yaml

from nro.engine.io import atomic_write_text

_READ_CACHE: ContextVar[dict[tuple[str, str], object] | None] = ContextVar(
    "nro_site_read_cache", default=None
)


def with_site_read_cache(function: Callable) -> Callable:
    """Reuse immutable site-definition reads within one planning operation."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _READ_CACHE.get() is not None:
            return function(*args, **kwargs)
        token = _READ_CACHE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _READ_CACHE.reset(token)

    return wrapped


def _checkout() -> Path:
    """Resolve the checkout independently of editable package placement."""
    explicit = os.environ.get("NRO_CHECKOUT")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise ValueError("NRO_CHECKOUT must be an absolute path")
        return path.resolve()
    if "NRO_EXECUTION_SOURCE_ROOT" not in os.environ:
        marker = Path(sys.prefix) / ".nro-checkout"
        if marker.is_file():
            path = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
            if not path.is_absolute():
                raise ValueError(f"Invalid checkout marker: {marker}")
            return path.resolve()
    return Path(__file__).resolve().parents[2]


CHECKOUT = _checkout()
RECORD_NAME = ".nro-installation.json"
SITE_DEFINITION_VERSION = 1
SITE_DEFINITION = Path("site/site.yml")
LAB = Path("/juice6/u/nlp/climblab")
DEFAULTS = {
    "definitions": str(LAB / "nro-definitions"),
    "bids": str(LAB / "BIDS"),
    "work": str(LAB / "WORK"),
    "development": str(LAB / "NRO_DEV"),
    "registry": str(LAB / ".nro"),
    "images": str(LAB / "apptainer/images"),
    "gradient_coefficients": str(LAB / "shared/gradient-coefficients"),
    "templates": str(LAB / "templateflow"),
    "workbench": str(LAB / "shared/workbench/bin_linux64/wb_command"),
    "oslom": str(LAB / "shared/oslom/oslom_undir"),
    "pycicada": str(LAB / "shared/pycicada/bin/cicada-python"),
    "license": str(LAB / "freesurfer/license.txt"),
    "runtime": "singularity",
    "partition": "sphinx",
    "viewing_partition": "john",
    "account": "nlp",
    "flywheel_server": "",
    "flywheel_project": "",
    "binds": ["/juice6:/juice6"],
}
ENVIRONMENT_KEYS = {
    "NRO_WORK_PATH": "work",
    "NRO_WB_COMMAND": "workbench",
    "TEMPLATEFLOW_HOME": "templates",
    "FS_LICENSE": "license",
}
DERIVED = {
    "qunex": ("images", "qunex_suite-1.5.1.sif"),
    "synthstrip": ("images", "synthstrip_1.7.sif"),
    "synbold": ("images", "synbold-disco_v1.4.sif"),
    "gradient_unwarp": ("images", "hcp-base_1.0.3_4.3.0.sif"),
    "freesurfer": ("images", "freesurfer_7.4.1.sif"),
    "fastsurfer": ("images", "fastsurfer-cu118_2.5.4.sif"),
    "fastsurfer_data": ("images", "fastsurfer-lit-0.6.1"),
    "synthstroke_data": ("images", "synthstroke-synth-plus-e9774354"),
    "mni_template": (
        "templates",
        "tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
    ),
}
PATH_KEYS = (
    set(DEFAULTS)
    - {
        "runtime",
        "partition",
        "viewing_partition",
        "account",
        "flywheel_server",
        "flywheel_project",
        "binds",
    }
) | set(DERIVED)

SITE_SECTIONS = {
    "storage": ("bids", "work", "development", "registry"),
    "resources": (
        "images",
        "gradient_coefficients",
        "templates",
        "workbench",
        "oslom",
        "pycicada",
        "license",
        "qunex",
        "synthstrip",
        "synbold",
        "gradient_unwarp",
        "freesurfer",
        "fastsurfer",
        "fastsurfer_data",
        "synthstroke_data",
        "mni_template",
    ),
    "execution": ("runtime", "partition", "viewing_partition", "account", "binds"),
}
REQUIRED_SITE_KEYS = {
    "storage": SITE_SECTIONS["storage"],
    "resources": ("images", "gradient_coefficients", "templates", "workbench", "oslom", "license"),
    "execution": SITE_SECTIONS["execution"],
}
SITE_BIDSIFY_KEYS = {
    "default_server",
    "default_project",
    "servers",
    "project_sources",
    "scanplans",
    "session_rules",
    "event_rules",
}


def generic_defaults() -> dict:
    """Return checkout-independent path proposals for a new standalone site."""
    root = Path.home().resolve() / "nro"
    return {
        **{
            key: str(root / suffix)
            for key, suffix in {
                "definitions": "definitions",
                "bids": "bids",
                "work": "work",
                "development": "development",
                "registry": ".nro",
                "images": "images",
                "gradient_coefficients": "gradient-coefficients",
                "templates": "templateflow",
                "workbench": "workbench/bin_linux64/wb_command",
                "oslom": "oslom/oslom_undir",
                "pycicada": "pycicada/bin/cicada-python",
                "license": "freesurfer/license.txt",
            }.items()
        },
        "binds": [],
        "viewing_partition": "interactive",
    }


def installation_record(root: Path = CHECKOUT) -> dict:
    """Read the checkout role record without creating it; reject records from another checkout."""
    path = root / RECORD_NAME
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get("mode") not in {"personal", "shared", "branch"}:
        raise ValueError(f"Invalid installation mode in {path}")
    if value.get("checkout") != str(root):
        raise ValueError(f"Installation record belongs to another checkout: {path}")
    return value


def _require_no_pending_conversion(record: dict) -> None:
    if (CHECKOUT / ".nro-installation-transition.json").exists() and record.get("mode") != "branch":
        raise ValueError(
            "Shared-to-branch conversion is incomplete; resume ./install --convert-to-branch"
        )


def site_file() -> Path:
    """Resolve the site TOML path, enforcing the recorded path for shared installations."""
    record = installation_record()
    _require_no_pending_conversion(record)
    selected = os.environ.get("NRO_SITE_CONFIG")
    # A verified source launcher replaces the mutable site locator with the
    # content-addressed execution snapshot whose digest it checked before
    # importing nro. This is internal execution state, not a user override of
    # the shared installation's protected site configuration.
    if selected and "NRO_EXECUTION_SOURCE_ROOT" in os.environ:
        return Path(selected).expanduser().resolve()
    if record.get("mode") in {"shared", "branch"}:
        required = Path(record["site"]).expanduser().resolve()
        if selected and Path(selected).expanduser().resolve() != required:
            raise ValueError(f"This {record['mode']} installation uses {required}")
        return required
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file() and (not record or str(path) != record.get("site")):
            raise ValueError(f"Selected site configuration does not exist: {path}")
        return path
    if record:
        return Path(record["site"])
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nro/site.toml"


def read_overrides(path: Path) -> dict:
    """Read a locator, resolved execution snapshot, or legacy site TOML file."""
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    unknown = set(values) - set(DEFAULTS) - set(DERIVED)
    if unknown:
        raise ValueError(f"Unknown site settings in {path}: {', '.join(sorted(unknown))}")
    for key, value in values.items():
        validate_setting(key, value)
    return values


def _definitions_from_locator(path: Path, overrides: dict) -> Path:
    """Resolve the authoritative definitions root without reading its contents."""
    value = overrides.get("definitions", DEFAULTS["definitions"])
    if not LAB.is_dir() or not os.access(LAB, os.R_OK | os.X_OK):
        value = overrides.get("definitions", generic_defaults()["definitions"])
    return Path(value).expanduser().resolve()


def site_definition_path(definitions: Path) -> Path:
    """Return the protected site document in a definitions store."""
    return Path(definitions).expanduser().resolve() / SITE_DEFINITION


def _validate_bidsify_site(value: object) -> dict:
    """Validate protected ingestion routing without importing bidsification code."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - SITE_BIDSIFY_KEYS:
        raise ValueError(
            "site.bidsify accepts only default_server, default_project, servers, "
            "project_sources, scanplans, session_rules, and event_rules"
        )
    defaults = {
        "default_server": None,
        "default_project": None,
        "servers": {},
        "project_sources": {},
        "scanplans": {"location": None, "credential_env": None},
        "session_rules": [],
        "event_rules": [],
    }
    result = {**defaults, **value}
    supplied_scanplans = value.get("scanplans")
    if supplied_scanplans is not None and not isinstance(supplied_scanplans, dict):
        raise ValueError("site.bidsify.scanplans must be a mapping")
    result["scanplans"] = {**defaults["scanplans"], **(supplied_scanplans or {})}
    for key in ("default_server", "default_project"):
        if result[key] is not None and not isinstance(result[key], str):
            raise ValueError(f"site.bidsify.{key} must be a string or null")
    if not isinstance(result["servers"], dict):
        raise ValueError("site.bidsify.servers must be a mapping")
    if not isinstance(result["project_sources"], dict):
        raise ValueError("site.bidsify.project_sources must be a mapping")
    scanplans = result["scanplans"]
    if not isinstance(scanplans, dict) or set(scanplans) != {"location", "credential_env"}:
        raise ValueError("site.bidsify.scanplans requires location and credential_env")
    for key, item in scanplans.items():
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"site.bidsify.scanplans.{key} must be a string or null")
    for key in ("session_rules", "event_rules"):
        if not isinstance(result[key], list):
            raise ValueError(f"site.bidsify.{key} must be a list")
    return result


def validate_site_document(value: object) -> tuple[dict, dict]:
    """Validate a protected site document and return flat settings and ingestion facts."""
    if not isinstance(value, dict) or set(value) != {
        "version",
        "storage",
        "resources",
        "execution",
        "bidsify",
    }:
        raise ValueError(
            "site/site.yml requires version, storage, resources, execution, and bidsify"
        )
    if value["version"] != SITE_DEFINITION_VERSION:
        raise ValueError(f"Unsupported site definition version: {value['version']!r}")
    settings_values: dict = {}
    for section, keys in SITE_SECTIONS.items():
        section_value = value[section]
        if not isinstance(section_value, dict) or set(section_value) - set(keys):
            raise ValueError(f"site.{section} contains unknown settings")
        missing = set(REQUIRED_SITE_KEYS[section]) - set(section_value)
        if missing:
            raise ValueError(f"site.{section} is missing: {', '.join(sorted(missing))}")
        settings_values.update(section_value)
    for key, item in settings_values.items():
        validate_setting(key, item)
    return settings_values, _validate_bidsify_site(value["bidsify"])


def read_site_definition(definitions: Path, *, required: bool = True) -> tuple[dict, dict]:
    """Read the protected site document from an authoritative definitions store."""
    path = site_definition_path(definitions)
    if not path.is_file():
        if required:
            raise ValueError(f"Definitions store has no protected site definition: {path}")
        return {}, _validate_bidsify_site({})
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read protected site definition {path}: {error}") from error
    return validate_site_document(value)


def make_site_document(settings_values: dict, *, bidsify: dict | None = None) -> dict:
    """Build the canonical protected document from resolved deployment values."""
    settings_values = dict(settings_values)
    for key, (parent, suffix) in DERIVED.items():
        if settings_values.get(key) == str(Path(settings_values.get(parent, "")) / suffix):
            settings_values.pop(key)
    document = {
        "version": SITE_DEFINITION_VERSION,
        **{
            section: {key: settings_values[key] for key in keys if key in settings_values}
            for section, keys in SITE_SECTIONS.items()
        },
        "bidsify": _validate_bidsify_site(bidsify or {}),
    }
    validate_site_document(document)
    return document


def write_site_definition(
    definitions: Path, settings_values: dict, *, bidsify: dict | None = None
) -> Path:
    """Atomically publish protected site settings inside a definitions store."""
    path = site_definition_path(definitions)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = make_site_document(settings_values, bidsify=bidsify)
    from nro.configuration.definition_migrations import (
        MANIFEST,
        normalize_managed_text,
        update_store,
    )

    text = normalize_managed_text(path, yaml.safe_dump(document, sort_keys=False))
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    if (definitions / MANIFEST).is_file():
        update_store(definitions, {path.relative_to(definitions): text.encode("utf-8")})
        return path
    atomic_write_text(
        path,
        text,
        mode=0o644,
        durable=True,
    )
    return path


def protected_site(definitions: Path | None = None) -> tuple[dict, dict]:
    """Return centrally governed settings and ingestion facts."""
    if definitions is None:
        path = site_file()
        overrides = read_overrides(path)
        definitions = _definitions_from_locator(path, overrides)
    return read_site_definition(Path(definitions))


def protected_site_fingerprint(definitions: Path | None = None) -> str:
    """Identify the canonical site-wide settings admitted by the scheduler."""
    definitions = (
        Path(settings()[0]["definitions"])
        if definitions is None
        else Path(definitions).expanduser().resolve()
    )
    site_settings, bidsify = protected_site(definitions)
    hardware = definitions / "hardware/gradient_unwarping.yml"
    payload = json.dumps(
        {
            "settings": site_settings,
            "bidsify": bidsify,
            "hardware_sha256": (
                hashlib.sha256(hardware.read_bytes()).hexdigest() if hardware.is_file() else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_setting(key: str, value: object) -> None:
    """Reject unknown settings, malformed scalar values, and nonabsolute resource paths."""
    if key not in set(DEFAULTS) | set(DERIVED):
        raise ValueError(f"Unknown site setting {key!r}")
    if key == "binds":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError("binds must be a list of strings")
        return
    if not isinstance(value, str) or "\n" in value or "\x00" in value:
        raise ValueError(f"{key} must be a single-line string")
    if key == "flywheel_server" and value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("flywheel_server must be a configured server name")
    if key == "flywheel_project" and value and not re.fullmatch(r"[^/\s]+/[^/\s]+", value):
        raise ValueError("flywheel_project must use GROUP/PROJECT")
    if key in PATH_KEYS and not Path(value).expanduser().is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    if key not in {"account", "flywheel_server", "flywheel_project"} and not value.strip():
        raise ValueError(f"{key} cannot be empty")


def settings(*, path: Path | None = None) -> tuple[dict, dict]:
    """Return resolved site values and their provenance labels.

    Explicit overrides precede derived resource paths. Personal environment
    overrides are ignored by shared installations. This function performs no writes.
    """
    path = site_file() if path is None else path
    cache = _READ_CACHE.get()
    cache_key = ("settings", str(path))
    if cache is not None and cache_key in cache:
        cached_values, cached_sources = cache[cache_key]
        return dict(cached_values), dict(cached_sources)
    values = dict(DEFAULTS)
    sources = {key: "lab default" for key in values}
    if not LAB.is_dir() or not os.access(LAB, os.R_OK | os.X_OK):
        proposals = generic_defaults()
        values.update(proposals)
        sources.update({key: "home default" for key in proposals})
    overrides = read_overrides(path)
    locator_only = set(overrides) <= {"definitions"}
    if locator_only:
        definitions = _definitions_from_locator(path, overrides)
        values["definitions"] = str(definitions)
        sources["definitions"] = str(path) if "definitions" in overrides else sources["definitions"]
        protected_path = site_definition_path(definitions)
        if protected_path.is_file():
            protected, bidsify = read_site_definition(definitions)
            values.update(protected)
            values["flywheel_server"] = bidsify["default_server"] or ""
            values["flywheel_project"] = bidsify["default_project"] or ""
            sources.update({key: str(protected_path) for key in protected})
            sources["flywheel_server"] = str(protected_path)
            sources["flywheel_project"] = str(protected_path)
    else:
        # Full TOML files are legacy installation state or immutable execution
        # snapshots. Installation migrates the former; workers must keep using
        # the latter without consulting mutable definitions.
        values.update(overrides)
        sources.update({key: str(path) for key in overrides})
    if installation_record().get("mode") not in {"shared", "branch"}:
        for variable, key in ENVIRONMENT_KEYS.items():
            if os.environ.get(variable):
                values[key] = os.environ[variable]
                sources[key] = variable
    for key, (parent, suffix) in DERIVED.items():
        if key not in values:
            values[key] = str(Path(values[parent]) / suffix)
            sources[key] = f"derived from {parent}"
    for key in PATH_KEYS:
        values[key] = str(Path(values[key]).expanduser())
    if cache is not None:
        cache[cache_key] = (dict(values), dict(sources))
    return values, sources


def bids_root() -> Path:
    """Return the BIDS root selected by the global site configuration."""
    return Path(settings()[0]["bids"]).resolve()


def definitions_root() -> Path:
    """Return the nearest writable or shared definitions directory."""
    return definitions_roots()[0]


def definitions_roots() -> tuple[Path, ...]:
    """Return definition stores in nearest-first lookup order.

    Development branches inherit private stores from their registered parents,
    followed by the shared site store. Shared and standalone installations use
    one store.
    """
    cache = _READ_CACHE.get()
    cache_key = ("definitions", "active-roots")
    if cache is not None and cache_key in cache:
        return tuple(Path(path) for path in cache[cache_key])
    values = settings()[0]
    shared = Path(values["definitions"]).resolve()
    roots = (shared,)
    record = installation_record()
    if record.get("mode") == "branch" and record.get("ready"):
        from nro.configuration.branch_definitions import inherited_definitions

        roots = inherited_definitions(Path(values["registry"]), record, shared)
    for root in roots:
        if (root / ".nro-incomplete").exists():
            raise ValueError(
                f"Definitions publication is incomplete: {root}; inspect or recreate the store"
            )
    if cache is not None:
        cache[cache_key] = tuple(roots)
    return tuple(roots)


def resolve_definition(relative: str | Path) -> Path | None:
    """Return the nearest definition matching a safe relative path."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Definition path must be relative: {relative}")
    for root in definitions_roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def require_execution_support(
    *, scientific: bool = False, installation_maintenance: bool = False
) -> None:
    """Require branch execution to pass through the central admission boundary."""
    record = installation_record()
    _require_no_pending_conversion(record)
    if record.get("mode") == "shared" and not record.get("ready") and not installation_maintenance:
        raise ValueError("The shared installation is undergoing setup or maintenance")
    if record.get("mode") == "branch":
        raise ValueError(
            "Development installations must use nro run and the central scheduler; "
            "direct production registry access and unbound execution are not permitted."
        )
    if scientific:
        from nro.orchestration.branch_store import BranchStore
        from nro.orchestration.releases import ReleaseStore
        from nro.orchestration.scheduler_implementation import implementation_path

        control = Path(settings()[0]["registry"])
        if implementation_path(control).exists():
            ReleaseStore(BranchStore(control)).require_installed(CHECKOUT, installation_record())


def require_definition_write(path: Path | None = None, *, creating_store: bool = False) -> None:
    """Limit development writes to a selected private store or a new isolated store."""
    record = installation_record()
    _require_no_pending_conversion(record)
    if record.get("mode") == "branch":
        if not record.get("ready"):
            raise ValueError(
                "Incomplete development installations cannot edit definitions; finish setup first"
            )
        from nro.configuration.branch_definitions import require_private_store, selected_definitions

        values = settings()[0]
        control, shared = Path(values["registry"]), Path(values["definitions"])
        selected = selected_definitions(control, record, shared)
        if creating_store and path is not None:
            require_private_store(control, shared, path, owner=record["branch"])
            return
        if selected is None:
            raise ValueError(
                "Development installations cannot edit shared definitions; "
                "use nro branch definitions --definitions PATH to select a private store"
            )
        require_private_store(control, shared, selected, owner=record["branch"])
        if path is not None and not path.expanduser().resolve().is_relative_to(selected):
            raise ValueError("Definition is outside the selected development store")


@contextmanager
def definition_write(path: Path):
    """Keep branch selection fixed while publishing or deleting one definition.

    This lock serializes branch selection with nro writers. The store manifest
    rejects direct filesystem edits made outside this interface.
    """
    record = installation_record()
    if record.get("mode") == "branch" and record.get("ready"):
        from nro.orchestration.branch_store import BranchStore

        store = BranchStore(Path(settings()[0]["registry"]))
        with store._lock():
            require_definition_write(path)
            yield
    else:
        require_definition_write(path)
        yield


def resolve_resources(value: object) -> object:
    """Replace explicit site references before validating workflow options."""
    values, _ = settings()

    def resolve(item: object) -> object:
        if isinstance(item, str) and item.startswith("site:"):
            key = item[5:]
            if key not in values:
                raise ValueError(f"Unknown site resource {key!r}")
            return values[key]
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        return item

    return resolve(value)
