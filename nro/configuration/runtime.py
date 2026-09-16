"""Load resolved, immutable configurations selected by the orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parsing import parse_mapping
from .schema import compile_configuration

_RUNTIME_SUFFIX = {
    "anat": "_anat.yml",
    "func": "_func.yml",
    "clean": "_clean.yml",
    "dynconn": "_dynconn.yml",
    "microparcellation": "_microparcellation.yml",
    "networks": "_networks.yml",
    "firstlevels": "_firstlevels.yml",
}


class ConfigNode:
    """Provide attribute access to one resolved runtime configuration mapping."""

    def __init__(self, data: dict[str, Any]) -> None:
        """Wrap a resolved mapping for attribute and item access."""
        self.replace(data)

    def replace(self, data: dict[str, Any]) -> None:
        """Replace the underlying mapping in place so existing imports retain this wrapper."""
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self._data[name])
        except KeyError as error:
            raise AttributeError(name) from error

    def __getitem__(self, key: str) -> Any:
        return _wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        """Return a value or default, wrapping nested mappings as ConfigNode objects."""
        return _wrap(self._data.get(key, default))


def _wrap(value: Any) -> Any:
    return ConfigNode(value) if isinstance(value, dict) else value


SETTINGS = ConfigNode({})


def configure(settings: dict[str, Any]) -> None:
    """Replace the process-local resolved settings without invalidating imports."""
    SETTINGS.replace(settings)


def load_runtime_configuration(
    path: str | Path, configuration_class: str
) -> tuple[str, dict[str, Any]]:
    """Read a fully resolved private configuration without applying defaults."""
    from nro.configuration.site import require_execution_support

    require_execution_support()
    try:
        suffix = _RUNTIME_SUFFIX[configuration_class]
    except KeyError as error:
        raise ValueError(f"Unknown configuration class: {configuration_class}") from error
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.name.endswith(suffix):
        raise ValueError(
            f"{configuration_class} runtime config must be named <ID>{suffix}: {resolved_path}"
        )
    if not resolved_path.is_file():
        raise ValueError(f"Runtime config does not exist: {resolved_path}")
    try:
        values = parse_mapping(resolved_path.read_text(encoding="utf-8"), source=str(resolved_path))
        values = compile_configuration(configuration_class, values, runtime=True)
    except ValueError as error:
        raise ValueError(f"Invalid runtime config {resolved_path}: {error}") from error
    return resolved_path.name[: -len(suffix)], values


def configure_anat(project: str, anat_id: str, values: dict[str, Any]) -> None:
    """Adapt a resolved anatomical configuration to module settings."""
    configure(
        {
            "common": {
                "project": project,
                "anat_id": anat_id,
                "qunex_container": values["container"]["image"],
                "multi_session_label": "ses-multi",
            },
            "anat": {
                **{
                    key: value
                    for key, value in values.items()
                    if key not in {"container", "fsaverage_template"}
                },
                "fsaverage_template": values["fsaverage_template"],
                "container": values["container"],
                "out_dir": None,
                "work_dir": None,
            },
        }
    )


def configure_func(project: str, func_id: str, values: dict[str, Any]) -> None:
    """Adapt a resolved functional configuration to module settings."""
    configure(
        {
            "common": {
                "project": project,
                "func_id": func_id,
                "anat_id": values["anat_directory"],
                "qunex_container": values["container"]["image"],
                "multi_session_label": "ses-multi",
            },
            "func": {
                **{
                    key: value
                    for key, value in values.items()
                    if key
                    not in {
                        "anat_directory",
                        "confounds",
                        "container",
                    }
                },
                "container": values["container"],
                "work_dir": None,
            },
            "func_confounds": values["confounds"],
        }
    )


def configure_clean(project: str, clean_id: str, values: dict[str, Any]) -> None:
    """Adapt a resolved cleaning configuration to module settings."""
    configure(
        {
            "common": {
                "project": project,
                "func_id": values["func_directory"],
                "anat_id": values["anat_directory"],
                "clean_id": clean_id,
                "qunex_home_dirname": "_qunex_home",
                "wb_command": values["wb_command"],
            },
            "clean": values,
        }
    )
