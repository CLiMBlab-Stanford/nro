"""Run an ad hoc quality control for an nro derivative."""

from __future__ import annotations

from nro.qc.engine import main as run_quality_control


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.qc") -> None:
    """Dispatch a quality-control type to its engine implementation.

    argv excludes the executable name; None reads the process arguments.
    prog controls help/error labels. Invalid arguments raise SystemExit.
    """
    run_quality_control(argv, prog=prog)


if __name__ == "__main__":
    main()
