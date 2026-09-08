"""Validated ingestion profiles without stored credentials."""

from copy import deepcopy
from pathlib import Path
import re

from nro.configuration.parsing import parse_mapping
from nro.configuration.paths import WORK_PATH
from nro.configuration.site import definitions_root


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


def load_config(path: Path | None = None, *, root: Path | None = None) -> dict:
    """Validate an ingestion profile and resolve runtime paths from the selected store.

    An explicit root supports validating a store before selecting it. Empty
    server mappings are valid for new stores but cannot submit ingestion work.
    """
    root = Path(root) if root is not None else definitions_root()
    source = path or root / "bidsify/main.yml"
    value = parse_mapping(source.read_text(), source=str(source))
    expected = {"servers", "staging", "dcm2niix", "synthstrip", "validator",
                "memory_gb", "cpus", "hours", "concurrency", "protocols", "event_rules", "session_rules"}
    if not expected <= set(value) or set(value) - expected - {'project_sources'}:
        raise ValueError(f"Bidsification configuration requires exactly: {sorted(expected)}; optional: project_sources")
    value = deepcopy(value)
    value.setdefault('project_sources', {})
    value['event_store'] = str(root / 'events')
    value["staging"] = str(WORK_PATH / "bidsify") if value["staging"] is None else value["staging"]
    if not Path(value["staging"]).is_absolute():
        raise ValueError("staging must be an absolute shared path")
    for key in ("memory_gb", "cpus", "hours", "concurrency"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("dcm2niix", "synthstrip", "validator"):
        if key != 'validator' and value[key] is None:
            from nro.configuration.site import settings
            site, _ = settings()
            operation = 'exec' if key == 'dcm2niix' else 'run'
            image = site['qunex' if key == 'dcm2niix' else 'synthstrip']
            value[key] = [site['runtime'], operation, '--cleanenv', '--bind', '{staging}', image]
            if key == 'dcm2niix':
                value[key].append('dcm2niix')
        if not isinstance(value[key], list) or not value[key] or any(not isinstance(v, str) or not v for v in value[key]):
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
        if not isinstance(server["projects"], list) or any(not isinstance(p, str) or len(p.split('/')) != 2 for p in server["projects"]):
            raise ValueError("Remote projects must be GROUP/PROJECT names")
    if not isinstance(value['project_sources'], dict):
        raise ValueError('project_sources must map BIDS projects to Flywheel sources')
    for project, sources in value['project_sources'].items():
        identifier(project)
        if not isinstance(sources, list) or not sources:
            raise ValueError('Each project_sources entry must be a nonempty list of sources')
        seen = set()
        for source in sources:
            if not isinstance(source, dict) or set(source) != {'server', 'project'}:
                raise ValueError('Each source requires server and project')
            if not isinstance(source['server'], str) or source['server'] not in value['servers']:
                raise ValueError('Project source names an unknown server')
            if source['project'] not in value['servers'][source['server']]['projects']:
                raise ValueError('Project source names an unconfigured Flywheel project')
            key = source['server'], source['project']
            if key in seen:
                raise ValueError('Duplicate source in project_sources')
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
    validate_session_rules(value['session_rules'], value['servers'])
    return value


ALLOWED_TYPES = {("anat", "T1w"), ("anat", "T2w"), ("func", "bold"),
                 ("func", "sbref"), ("fmap", "epi"), ("ignore", "ignore")}
