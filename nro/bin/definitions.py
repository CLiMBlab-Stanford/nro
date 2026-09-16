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
            from nro.configuration.site import installation_record, settings

            record = installation_record()
            if record.get("mode") == "branch":
                shared = Path(settings()[0]["definitions"])
                create_store(root, include_site=False, inherited_site=shared)
            else:
                create_store(root)
        from nro.configuration.site import installation_record, settings

        record = installation_record()
        shared = Path(settings()[0]["definitions"])
        private = record.get("mode") == "branch" and root.resolve() != shared.resolve()
        inherited_site = shared if private and not (root / "site/site.yml").is_file() else None
        counts = validate_store(
            root,
            require_site=not private,
            inherited_site=inherited_site,
        )
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
