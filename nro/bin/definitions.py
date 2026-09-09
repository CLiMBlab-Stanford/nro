"""Create or validate an external nro definitions store."""

import argparse
import json
from pathlib import Path

from nro.configuration.definitions import create_store, validate_store
from nro.configuration.site import definitions_root


def main(argv=None, *, prog="nro definitions"):
    """Manage a store without modifying site selection, derivatives, the registry, or Git."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("action", choices=("create", "validate"))
    parser.add_argument("path", nargs="?", type=Path, help="Default: selected definitions path")
    parser.add_argument(
        "--json", action="store_true", help="Report the path and validation counts as JSON"
    )
    args = parser.parse_args(argv)
    try:
        root = args.path.expanduser().absolute() if args.path is not None else definitions_root()
        if args.action == "create":
            create_store(root)
        counts = validate_store(root)
        if args.json:
            print(json.dumps({"path": str(root), **counts}, indent=2))
        else:
            print(f"{'Created' if args.action == 'create' else 'Validated'} {root}")
            print(", ".join(f"{key}: {value}" for key, value in counts.items()))
            if args.action == "create":
                from nro.configuration.site import installation_record

                command = (
                    "nro branch definitions --definitions PATH"
                    if installation_record().get("mode") == "branch"
                    else "nro paths set definitions=PATH"
                )
                print(f"Store selection is unchanged. Use {command} to select this store.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nDefinitions operation cancelled.\n")


if __name__ == "__main__":
    main()
