"""Compatibility alias for ``nro site``."""

from __future__ import annotations

import sys

from nro.site.setup import edit_settings


def main(argv=None, *, prog="nro paths") -> None:
    """Translate the former paths interface to site configuration operations."""
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        try:
            edit_settings()
        except (KeyboardInterrupt, EOFError):
            print("\nSite configuration cancelled.", file=sys.stderr)
            raise SystemExit(130) from None
        return
    elif values[0] == "show":
        translated = ["ls", *values[1:]]
    elif values[0] == "set":
        translated = ["set", *values[1:]]
    else:
        translated = values
    from nro.bin.site import main as site_main

    site_main(translated, prog=prog)


if __name__ == "__main__":
    main()
