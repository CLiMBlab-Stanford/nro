"""Prepare a branch definitions layer from its installed Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.engine.bootstrap import prepare_branch_definitions


def main(argv: list[str] | None = None) -> None:
    """Migrate and validate one branch's definitions inheritance chain."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--registry-id", required=True)
    args = parser.parse_args(argv)
    prepare_branch_definitions(
        args.site,
        {
            "checkout": str(args.checkout.resolve()),
            "branch": args.branch,
            "registry_id": args.registry_id,
        },
    )


if __name__ == "__main__":
    main()
