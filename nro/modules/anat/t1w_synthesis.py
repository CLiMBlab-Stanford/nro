"""Construct a synthetic T1w reference when only T2w anatomy is available."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from nro.orchestration.runner_graph import Step

from .policy import t1w_synthesis_contract


def create_t1w_synthesis_step(
    *,
    run_child: Callable[..., Any],
    runtime: str,
    image: Path,
    license_file: Path,
    t2w: Path,
    output: Path,
    threads: int,
    env: Mapping[str, str],
    force: bool,
) -> Step:
    """Create a CPU SynthSR step with explicit input and software provenance."""

    def action() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        work = output.parent.resolve()
        source = t2w.resolve()
        if not source.is_relative_to(work):
            raise ValueError("T1w synthesis input and output must share a private work directory")
        command = [
            runtime,
            "exec",
            "--cleanenv",
            "--bind",
            f"{work}:/work,{license_file}:/license.txt:ro",
            str(image),
            "bash",
            "-lc",
            (
                "export FREESURFER_HOME=/usr/local/freesurfer; "
                "export FS_LICENSE=/license.txt; "
                'source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null; exec "$@"'
            ),
            "bash",
            "mri_synthsr",
            "--i",
            f"/work/{source.relative_to(work)}",
            "--o",
            f"/work/{output.resolve().relative_to(work)}",
            "--threads",
            str(max(1, threads)),
            "--cpu",
        ]
        run_child(command, direct=True, env=dict(env), discard_stdout=True)

    return Step.python(
        name="Synthesize T1w Reference from T2w",
        inputs=(t2w, image, license_file),
        outputs=(output,),
        action=action,
        force=force,
        parameters=t1w_synthesis_contract(),
    )
