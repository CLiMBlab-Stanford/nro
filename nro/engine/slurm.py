"""Small Slurm launches that do not belong to the nro worker pool."""

from __future__ import annotations

import os
import shutil
import subprocess


def run_x11(command: list[str], *, partition: str, account: str | None) -> None:
    """Run a GUI command with X11 forwarding on a Slurm compute node.

    Reuse the caller's allocation when one is active; Slurm cannot add X11
    forwarding to a nested job step. Otherwise, request a dedicated allocation.
    The caller remains attached until the application exits.
    """
    if not os.environ.get("DISPLAY"):
        raise ValueError(
            "X11 forwarding is unavailable because DISPLAY is not set; reconnect with SSH X forwarding"
        )
    if os.environ.get("SLURM_JOB_ID"):
        argv = []
    else:
        launcher = shutil.which("srun")
        if launcher is None:
            raise ValueError("Cannot open the scene through Slurm because srun is unavailable")
        argv = [
            launcher,
            "--x11",
            f"--partition={partition}",
            "--job-name=nro-scene",
            "--ntasks=1",
            "--cpus-per-task=1",
        ]
        if account:
            argv.append(f"--account={account}")
    try:
        subprocess.run([*argv, *command], check=True)
    except KeyboardInterrupt:
        return
    except OSError:
        raise ValueError("Slurm could not start the scene viewer") from None
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Slurm scene viewer exited with status {error.returncode}") from None
