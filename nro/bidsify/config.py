"""Combine protected ingestion sources with branchable conversion profiles."""

import re
from copy import deepcopy
from pathlib import Path

from nro.configuration.parsing import parse_mapping
from nro.configuration.site import (
    definitions_root,
    read_site_definition,
    settings,
    site_definition_path,
)


def identifier(value: str) -> str:
    """Validate an opaque path component; reject traversal and separators."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Identifiers must contain letters, digits, underscores, or hyphens")
    return value


def bids_label(value: str) -> str:
    """Validate a BIDS entity label without silently changing its identity."""
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        raise ValueError("BIDS labels must contain only letters and digits")
    return value


def load_config(
    path: Path | None = None, *, root: Path | None = None, site_root: Path | None = None
) -> dict:
    """Validate an ingestion profile and resolve runtime paths from the selected store.

    An explicit root supports validating a store before selecting it. Empty
    server mappings are valid for new stores but cannot submit ingestion work.
    """
    root = Path(root) if root is not None else definitions_root()
    source = path or root / "bidsify/main.yml"
    value = parse_mapping(source.read_text(), source=str(source))
    expected = {
        "staging",
        "dcm2niix",
        "synthstrip",
        "validator",
        "memory_gb",
        "cpus",
        "hours",
        "concurrency",
        "protocols",
    }
    optional = {"scanplans"}
    if not expected <= set(value) or set(value) - expected - optional:
        raise ValueError(
            f"Bidsification configuration requires: {sorted(expected)}; optional: {sorted(optional)}"
        )
    value = deepcopy(value)
    if site_definition_path(root).is_file():
        protected_root = root
    elif site_root is not None:
        protected_root = Path(site_root)
    else:
        protected_root = Path(settings()[0]["definitions"])
    site_settings, site = read_site_definition(protected_root)
    value.update(
        servers=deepcopy(site["servers"]),
        project_sources=deepcopy(site["project_sources"]),
        session_rules=deepcopy(site["session_rules"]),
        event_rules=deepcopy(site["event_rules"]),
    )
    scanplan_policy = deepcopy(site["scanplans"])
    profile_scanplans = value.setdefault("scanplans", {"parser": None})
    if not isinstance(profile_scanplans, dict) or set(profile_scanplans) != {"parser"}:
        raise ValueError("bidsify profile scanplans requires only parser")
    value["scanplans"] = {**scanplan_policy, **profile_scanplans}
    value["event_store"] = str(root / "events")
    value["staging"] = (
        str(Path(site_settings["work"]) / "bidsify")
        if value["staging"] is None
        else value["staging"]
    )
    if not Path(value["staging"]).is_absolute():
        raise ValueError("staging must be an absolute shared path")
    for key in ("memory_gb", "cpus", "hours", "concurrency"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    scanplans = value["scanplans"]
    if not isinstance(scanplans, dict) or set(scanplans) != {
        "location",
        "parser",
        "credential_env",
    }:
        raise ValueError("scanplans requires location, parser, and credential_env")
    location = scanplans["location"]
    if location is not None and (not isinstance(location, str) or not location.strip()):
        raise ValueError("scanplans.location must be a directory path, Google Drive URL, or null")
    if isinstance(location, str) and "://" not in location:
        path = Path(location).expanduser()
        if not path.is_absolute():
            path = root / path
        scanplans["location"] = str(path.resolve())
    elif isinstance(location, str) and not re.match(
        r"^https://drive\.google\.com/drive/(?:u/\d+/)?folders/[A-Za-z0-9_-]+",
        location,
    ):
        raise ValueError("scanplans supports local directories and Google Drive folder URLs")
    parser = scanplans["parser"]
    if parser is not None:
        if not isinstance(parser, str) or not parser.strip():
            raise ValueError("scanplans.parser must be a Python file path or null")
        path = Path(parser).expanduser()
        if not path.is_absolute():
            path = root / path
        scanplans["parser"] = str(path.resolve())
    credential_env = scanplans["credential_env"]
    if credential_env is not None and (
        not isinstance(credential_env, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", credential_env)
    ):
        raise ValueError("scanplans.credential_env must name an environment variable or be null")
    for key in ("dcm2niix", "synthstrip", "validator"):
        if key != "validator" and value[key] is None:
            resolved_site = dict(site_settings)
            images = Path(resolved_site["images"])
            resolved_site.setdefault("qunex", str(images / "qunex_suite-1.5.1.sif"))
            resolved_site.setdefault("synthstrip", str(images / "synthstrip_1.7.sif"))
            operation = "exec" if key == "dcm2niix" else "run"
            image = resolved_site["qunex" if key == "dcm2niix" else "synthstrip"]
            value[key] = [
                resolved_site["runtime"],
                operation,
                "--cleanenv",
                "--bind",
                "{staging}",
                image,
            ]
            if key == "dcm2niix":
                value[key].append("dcm2niix")
        if (
            not isinstance(value[key], list)
            or not value[key]
            or any(not isinstance(v, str) or not v for v in value[key])
        ):
            raise ValueError(f"{key} must be a nonempty argument list")
    if not isinstance(value["servers"], dict):
        raise ValueError("Servers must be a mapping")
    for name, server in value["servers"].items():
        identifier(name)
        if set(server) != {"host", "credential_env", "projects"}:
            raise ValueError(f"Invalid server fields: {name}")
        if not re.fullmatch(r"[A-Za-z0-9.-]+", server["host"]):
            raise ValueError("Server host must be a hostname, not a URL or credential")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", server["credential_env"]):
            raise ValueError("credential_env must name an environment variable")
        if not isinstance(server["projects"], list) or any(
            not isinstance(p, str) or len(p.split("/")) != 2 for p in server["projects"]
        ):
            raise ValueError("Remote projects must be GROUP/PROJECT names")
    if not isinstance(value["project_sources"], dict):
        raise ValueError("project_sources must map BIDS projects to Flywheel sources")
    for project, sources in value["project_sources"].items():
        identifier(project)
        if not isinstance(sources, list) or not sources:
            raise ValueError("Each project_sources entry must be a nonempty list of sources")
        seen = set()
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"server", "project"}:
                raise ValueError("Each source requires server and project")
            if not isinstance(source["server"], str) or source["server"] not in value["servers"]:
                raise ValueError("Project source names an unknown server")
            if source["project"] not in value["servers"][source["server"]]["projects"]:
                raise ValueError("Project source names an unconfigured Flywheel project")
            key = source["server"], source["project"]
            if key in seen:
                raise ValueError("Duplicate source in project_sources")
            seen.add(key)
    for rule in value["protocols"]:
        if set(rule) != {"pattern", "datatype", "suffix"}:
            raise ValueError("Protocol rules require pattern, datatype, suffix")
        re.compile(rule["pattern"])
        if (rule["datatype"], rule["suffix"]) not in ALLOWED_TYPES:
            raise ValueError("Unsupported protocol output")
    for rule in value["event_rules"]:
        if set(rule) != {"task", "pattern"} or not Path(rule["pattern"]).is_absolute():
            raise ValueError("Event rules require task and an absolute glob pattern")
        bids_label(rule["task"])
    from nro.bidsify.discovery import validate_session_rules

    validate_session_rules(value["session_rules"], value["servers"])
    return value


ALLOWED_TYPES = {
    ("anat", "T1w"),
    ("anat", "T2w"),
    ("func", "bold"),
    ("func", "sbref"),
    ("fmap", "epi"),
    ("ignore", "ignore"),
}
