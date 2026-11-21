"""Run this installation's ./install again, for example to upgrade it.

``nro install`` runs the ``./install`` script of the checkout that the current
``nro`` command runs from, with the given options. ``nro install --upgrade``
first moves the checkout to the newest release on ``main``; add ``--maintain``
for a shared installation. A first installation runs ``./install`` from the
repository root instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def checkout() -> Path:
    """Return the installation's checkout, which holds ``./install``.

    An installed nro runs from a verified application snapshot, so the
    checkout comes from the installation record; a checkout run directly,
    without a record, is the package's own repository.
    """
    from nro.site.configuration import installation_record

    recorded = installation_record().get("checkout")
    root = Path(recorded) if recorded else Path(__file__).resolve().parents[2]
    if not (root / "install").is_file() or not (root / "pyproject.toml").is_file():
        raise SystemExit(
            f"nro install needs the nro checkout's ./install, but {root} is not a checkout; "
            "run ./install from the repository root"
        )
    return root


def main(argv: list[str] | None = None, *, prog: str = "nro install") -> None:
    """Replace this process with the checkout's ``./install`` and the given options."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        print(f"usage: {prog} [--upgrade] [./install options ...]\n\n{__doc__}")
        print("Run `./install --help` in the checkout for every option.")
        return
    script = checkout() / "install"
    print(f"Running {script} {' '.join(arguments)}".rstrip(), flush=True)
    os.chdir(script.parent)
    os.execv(str(script), [str(script), *arguments])
