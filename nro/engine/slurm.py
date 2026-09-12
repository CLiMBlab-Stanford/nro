"""Small Slurm launches that do not belong to the nro worker pool."""

from __future__ import annotations

import os
import shutil
import subprocess


def run_x11(command: list[str], *, partition: str, account: str | None) -> None:
    """Run a GUI command in a Slurm allocation with X11 forwarding.

    The caller remains attached until the application exits. This keeps the
    allocation lifetime visible and lets an interrupt cancel queued or active
    work.
    """
    if not os.environ.get("DISPLAY"):
        raise ValueError(
            "X11 forwarding is unavailable because DISPLAY is not set; reconnect with SSH X forwarding"
        )
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
