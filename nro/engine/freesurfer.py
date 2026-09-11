"""FreeSurfer installation and shared-template discovery."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_FSAVERAGE_MARKER = "__NRO_FSAVERAGE__="


def find_fsaverage_directory(
    runner: Any,
    *,
    environment: dict[str, str],
    subjects_directory: Path,
    template: str,
) -> Path:
    """Find the selected fsaverage template on the host or in the active container."""
    if template not in {"fsaverage", "fsaverage6"}:
        raise ValueError(f"Unsupported fsaverage template: {template}")
    local_subject = Path(subjects_directory) / template
    if local_subject.exists():
        return local_subject
    freesurfer_home = os.environ.get("FREESURFER_HOME", "").strip()
    if freesurfer_home and not runner.using_container():
        installed = Path(freesurfer_home) / "subjects" / template
        if installed.exists():
            return installed
    if runner.using_container():
        raw = runner.run_out(
            [
                "bash",
                "-lc",
                f'if [ -d "$SUBJECTS_DIR/{template}" ]; then printf "{_FSAVERAGE_MARKER}%s" "$SUBJECTS_DIR/{template}"; '
                f'elif [ -n "${{FREESURFER_HOME:-}}" ] && [ -d "$FREESURFER_HOME/subjects/{template}" ]; then '
                f'printf "{_FSAVERAGE_MARKER}%s" "$FREESURFER_HOME/subjects/{template}"; fi',
            ],
            env=environment,
            quiet=True,
        )
        cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw or "")
        marker_index = cleaned.rfind(_FSAVERAGE_MARKER)
        if marker_index >= 0:
            discovered = cleaned[marker_index + len(_FSAVERAGE_MARKER) :].strip()
            if discovered:
                return Path(discovered)
    raise SystemExit(f"Could not locate {template} under SUBJECTS_DIR or FREESURFER_HOME.")
