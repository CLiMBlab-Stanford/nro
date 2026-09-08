"""Resolve installation settings independently of scientific configurations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib


CHECKOUT = Path(__file__).resolve().parents[2]
RECORD_NAME = ".nro-installation.json"
LAB = Path("/juice6/u/nlp/climblab")
DEFAULTS = {
    "definitions": str(LAB / "nro-definitions"),
    "bids": str(LAB / "BIDS"),
    "work": str(LAB / "WORK"),
    "registry": str(LAB / ".nro"),
    "images": str(LAB / "apptainer/images"),
    "templates": str(LAB / "templateflow"),
    "workbench": str(LAB / "shared/workbench/bin_linux64/wb_command"),
    "oslom": str(LAB / "shared/oslom/oslom_undir"),
    "license": str(LAB / "freesurfer/license.txt"),
    "runtime": "singularity",
    "partition": "sphinx",
    "account": "nlp",
    "binds": ["/juice6:/juice6"],
}
ENVIRONMENT_KEYS = {
    "NRO_BIDS_PATH": "bids", "NRO_WORK_PATH": "work",
    "NRO_WB_COMMAND": "workbench", "TEMPLATEFLOW_HOME": "templates",
    "FS_LICENSE": "license",
}
DERIVED = {
    "qunex": ("images", "qunex_suite-1.5.1.sif"),
    "synthstrip": ("images", "synthstrip_1.7.sif"),
    "synbold": ("images", "synbold-disco_v1.4.sif"),
    "mni_template": (
        "templates", "tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
    ),
}
PATH_KEYS = (set(DEFAULTS) - {"runtime", "partition", "account", "binds"}) | set(DERIVED)


def installation_record(root: Path = CHECKOUT) -> dict:
    """Read the checkout role record without creating it; reject records from another checkout."""
    path = root / RECORD_NAME
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get("mode") not in {"personal", "shared"}:
        raise ValueError(f"Invalid installation mode in {path}")
    if value.get("checkout") != str(root):
        raise ValueError(f"Installation record belongs to another checkout: {path}")
    return value


def site_file() -> Path:
    """Resolve the site TOML path, enforcing the recorded path for shared installations."""
    record = installation_record()
    selected = os.environ.get("NRO_SITE_CONFIG")
    if record.get("mode") == "shared":
        required = Path(record["site"]).expanduser().resolve()
        if selected and Path(selected).expanduser().resolve() != required:
            raise ValueError(f"This shared installation uses {required}")
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
    if key in PATH_KEYS and not Path(value).expanduser().is_absolute():
        raise ValueError(f"{key} must be an absolute path")
    if key not in {"account"} and not value.strip():
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
        values["definitions"] = str(Path.home().resolve() / "nro/definitions")
        sources["definitions"] = "home default"
    overrides = read_overrides(path)
    values.update(overrides)
    sources.update({key: str(path) for key in overrides})
    if installation_record().get("mode") != "shared":
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


def definitions_root() -> Path:
    """Resolve the selected definitions directory without creating it or reading its files."""
    root = Path(settings()[0]["definitions"]).resolve()
    if (root / '.nro-incomplete').exists():
        raise ValueError(f'Definitions publication is incomplete: {root}; inspect or recreate the store')
    return root


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
