"""Compatibility alias for ``nro site get``."""

from __future__ import annotations

import sys


def main(argv=None, *, prog="nro get") -> None:
    """Read durable site settings through the unified interface."""
    from nro.bin.site import main as site_main

    values = list(sys.argv[1:] if argv is None else argv)
    site_main(["get", *values], prog=prog)


if __name__ == "__main__":
    main()
