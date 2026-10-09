"""Compatibility alias for ``nro site set``."""

from __future__ import annotations

import sys


def main(argv=None, *, prog="nro set") -> None:
    """Update durable site settings through the unified interface."""
    values = list(sys.argv[1:] if argv is None else argv)
    action = "ls" if values[:1] == ["ls"] else "set"
    site_values = values[1:] if action == "ls" else values
    from nro.bin.site import main as site_main

    site_main([action, *site_values], prog=prog)


if __name__ == "__main__":
    main()
