"""Small Slurm launches that do not belong to the nro worker pool."""

from __future__ import annotations

from pathlib import Path

from nro.engine.viewer_broker import open_viewer


def run_x11(command: list[str], *, partition: str, account: str | None, control: Path) -> None:
    """Open a Workbench scene through the user's persistent viewer allocation.

    The command must use Workbench's direct scene-loading form. Existing Slurm
    allocations launch the viewer directly because X11 cannot be added to a
    nested job step.
    """
    if len(command) != 4 or command[1] != "-scene-load-hd" or command[3] != "1":
        raise ValueError("Persistent viewing requires one direct Workbench scene command")
    open_viewer(
        Path(command[2]),
        viewer=Path(command[0]),
        partition=partition,
        account=account,
        control=control,
    )
