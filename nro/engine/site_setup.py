"""Edit site settings without moving existing data."""

import glob
import json
import os
import readline
import sys
from pathlib import Path

from nro.configuration.site import (
    LAB,
    PATH_KEYS,
    generic_defaults,
    installation_record,
    read_overrides,
    settings,
    site_file,
    validate_setting,
)
from nro.engine.io import atomic_write_text

DESCRIPTIONS = {
    "definitions": "Configurations, workflows, models, events, and ingestion profiles",
    "bids": "Directory containing BIDS projects",
    "work": "Intermediate files",
    "development": "Branch-owned derivatives, intermediate files, and debug BIDS",
    "registry": "Shared registry and logs",
    "images": "Container images",
    "templates": "TemplateFlow data",
    "workbench": "wb_command executable",
    "license": "Existing FreeSurfer license",
    "runtime": "Singularity or Apptainer executable",
    "oslom": "oslom_undir executable",
    "partition": "Slurm partition",
    "viewing_partition": "Slurm partition for interactive scene viewing",
    "account": "Slurm account (- for none)",
    "flywheel_server": "Default Flywheel server (- for none)",
    "flywheel_project": "Default Flywheel GROUP/PROJECT (- for none)",
}


def interactive_defaults(sources: dict) -> dict:
    """Suggest generic storage when lab storage is unavailable; retain explicit settings."""
    if LAB.is_dir() and os.access(LAB, os.R_OK | os.X_OK):
        return {}
    return {
        key: value for key, value in generic_defaults().items() if sources[key] == "lab default"
    }


def save_settings(path: Path, overrides: dict) -> None:
    """Validate all overrides and atomically replace the site TOML file."""
    for key, value in overrides.items():
        validate_setting(key, value)
    text = "# nro site settings; changing paths does not move data.\n"
    text += "".join(f"{key} = {json.dumps(value)}\n" for key, value in sorted(overrides.items()))
    atomic_write_text(path, text, mode=0o644, durable=True)


def edit_settings(assignments=None, *, maintain=False, path=None) -> None:
    """Edit independent site values interactively or from key=value assignments.

    Shared edits require maintenance authorization and an inactive worker pool.
    Cancellation before save leaves the old file intact. No data are relocated.
    """
    if installation_record().get("mode") == "branch":
        raise ValueError(
            "Branch installations cannot edit the shared site; use its maintainer installation"
        )
    if installation_record().get("mode") == "shared" and not maintain:
        raise ValueError("Shared settings require --maintain and maintainer write access.")
    path = site_file() if path is None else path
    if installation_record().get("mode") == "shared":
        from nro.engine.bootstrap import check_workers

        check_workers(path)
    overrides = read_overrides(path)
    values, sources = settings(path=path)
    if assignments:
        for assignment in assignments:
            key, separator, value = assignment.partition("=")
            if not separator:
                raise ValueError(f"Expected key=value: {assignment!r}")
            if key == "binds":
                value = json.loads(value)
            elif key in PATH_KEYS:
                value = str(Path(value).expanduser())
            validate_setting(key, value)
            overrides[key] = value
    else:
        if not sys.stdin.isatty():
            raise ValueError("Interactive setup needs a terminal; use paths set key=value.")

        def complete(text, state):
            matches = glob.glob(os.path.expanduser(text) + "*")
            return matches[state] if state < len(matches) else None

        previous = readline.get_completer()
        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")
        try:
            proposals = interactive_defaults(sources)
            values.update(proposals)
            if "binds" in proposals:
                overrides["binds"] = proposals["binds"]
            print(f"Site configuration: {path}\nEnter keeps a value; Tab completes paths.")
            print("Proposed settings:")
            for key in DESCRIPTIONS:
                print(f"  {key}: {values[key]}")
            accept_all = input("Accept all defaults? [Y/n]: ").strip().lower() in {"", "y", "yes"}
            for key, description in DESCRIPTIONS.items():
                entered = (
                    "" if accept_all else input(f"{description}\n  {key} [{values[key]}]: ").strip()
                )
                value = entered or values[key]
                if key in {"account", "flywheel_server", "flywheel_project"} and entered == "-":
                    value = ""
                if key in PATH_KEYS:
                    value = str(Path(value).expanduser())
                validate_setting(key, value)
                overrides[key] = value
            for key, value in sorted(overrides.items()):
                print(f"  {key} = {value}")
            if input("Save? [Y/n]: ").strip().lower() not in {"", "y", "yes"}:
                return
        finally:
            readline.set_completer(previous)
    save_settings(path, overrides)
    print(f"Saved {path}. New commands use these settings; existing data were not moved.")
