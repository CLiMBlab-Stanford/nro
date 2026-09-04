"""Dispatch ad hoc quality controls implemented by the qc package."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from nro.qc import registration


QUALITY_CONTROLS: dict[str, Callable[..., None]] = {
    "registration": registration.main,
}


def build_parser(*, prog: str = "python -m nro.qc") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "qctype", choices=tuple(QUALITY_CONTROLS), help="Quality-control type"
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "python -m nro.qc") -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] in {"-h", "--help"}:
        build_parser(prog=prog).parse_args(values)
        return

    qctype = values.pop(0)
    implementation = QUALITY_CONTROLS.get(qctype)
    if implementation is None:
        build_parser(prog=prog).error(
            f"invalid qctype: {qctype!r} (choose from {', '.join(QUALITY_CONTROLS)})"
        )
    implementation(values, prog=f"{prog} {qctype}")
