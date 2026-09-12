"""Resolve installation settings independently of scientific configurations."""

from __future__ import annotations

import json
import os
import re
import tomllib
from contextlib import contextmanager
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2]
RECORD_NAME = ".nro-installation.json"
LAB = Path("/juice6/u/nlp/climblab")
DEFAULTS = {
    "definitions": str(LAB / "nro-definitions"),
    "bids": str(LAB / "BIDS"),
    "work": str(LAB / "WORK"),
    "development": str(LAB / "NRO_DEV"),
    "registry": str(LAB / ".nro"),
    "images": str(LAB / "apptainer/images"),
    "templates": str(LAB / "templateflow"),
    "workbench": str(LAB / "shared/workbench/bin_linux64/wb_command"),
    "oslom": str(LAB / "shared/oslom/oslom_undir"),
    "license": str(LAB / "freesurfer/license.txt"),
    "runtime": "singularity",
    "partition": "sphinx",
    "viewing_partition": "dev-interactive",
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
                "templates": "templateflow",
                "workbench": "workbench/bin_linux64/wb_command",
                "oslom": "oslom/oslom_undir",
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
    """Read and validate site overrides; return an empty mapping for an absent file."""
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
    values = dict(DEFAULTS)
    sources = {key: "lab default" for key in values}
    if not LAB.is_dir() or not os.access(LAB, os.R_OK | os.X_OK):
        proposals = generic_defaults()
        values.update(proposals)
        sources.update({key: "home default" for key in proposals})
    overrides = read_overrides(path)
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
    return values, sources


def bids_root() -> Path:
    """Return the BIDS root selected by the global site configuration."""
    return Path(settings()[0]["bids"]).resolve()


def definitions_root() -> Path:
    """Resolve the selected definitions directory without creating it or reading its files."""
    values = settings()[0]
    root = Path(values["definitions"]).resolve()
    record = installation_record()
    if record.get("mode") == "branch" and record.get("ready"):
        from nro.configuration.branch_definitions import selected_definitions

        root = selected_definitions(Path(values["registry"]), record, root) or root
    if (root / ".nro-incomplete").exists():
        raise ValueError(
            f"Definitions publication is incomplete: {root}; inspect or recreate the store"
        )
    return root


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
            ReleaseStore(BranchStore(control)).require_approved(CHECKOUT)


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

    This lock serializes branch selection with cooperating nro writers. Direct
    filesystem edits do not take the lock and remain the developer's responsibility.
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
