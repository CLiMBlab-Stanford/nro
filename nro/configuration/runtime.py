"""Load resolved, immutable configurations selected by the orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parsing import parse_mapping
from .schema import compile_configuration

_RUNTIME_SUFFIX = {
    "preprocessing": "_preprocess.yml",
    "clean": "_clean.yml",
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
    path: str | Path, derivative_class: str
) -> tuple[str, dict[str, Any]]:
    """Read a fully resolved private configuration without applying defaults."""
    try:
        suffix = _RUNTIME_SUFFIX[derivative_class]
    except KeyError as error:
        raise ValueError(f"Unknown derivative class: {derivative_class}") from error
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.name.endswith(suffix):
        raise ValueError(
            f"{derivative_class} runtime config must be named <ID>{suffix}: "
            f"{resolved_path}"
        )
    if not resolved_path.is_file():
        raise ValueError(f"Runtime config does not exist: {resolved_path}")
    try:
        values = parse_mapping(resolved_path.read_text(encoding="utf-8"), source=str(resolved_path))
        values = compile_configuration(derivative_class, values, runtime=True)
    except ValueError as error:
        raise ValueError(f"Invalid runtime config {resolved_path}: {error}") from error
    return resolved_path.name[: -len(suffix)], values


def configure_preprocessing(
    project: str, preprocessing_id: str, values: dict[str, Any]
) -> None:
    """Adapt a resolved preprocessing configuration to module settings."""
    container = values["container"]
    container_args = {
        "no_container": container["no_container"],
        "container_engine": container["engine"],
        "container_cleanenv": container["cleanenv"],
        "container_bind": container["bind"],
        "container_home": container["home"],
        "container_inner_setup": container["inner_setup"],
    }
    configure(
        {
            "common": {
                "project": project,
                "preprocessing_id": preprocessing_id,
                "qunex_container": container["image"],
                "multi_session_label": "ses-multi",
            },
            "preprocess_anat": {
                **values["anat"],
                **container_args,
                "out_dir": None,
                "work_dir": None,
            },
            "preprocess": {
                **values["func"],
                **container_args,
                "work_dir": None,
            },
            "get_confounds": values["confounds"],
        }
    )


def configure_clean(project: str, clean_id: str, values: dict[str, Any]) -> None:
    """Adapt a resolved cleaning configuration to module settings."""
    configure(
        {
            "common": {
                "project": project,
                "preprocessing_id": values["preprocessing_directory"],
                "clean_id": clean_id,
                "qunex_container": values["container"],
                "default_container_engine": values["container_engine"],
                "default_bind": values["container_bind"][0],
                "qunex_home_dirname": "_qunex_home",
                "wb_command": values["wb_command"],
            },
            "clean": values,
        }
    )
