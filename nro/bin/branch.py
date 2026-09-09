"""Register branches and resolve their centrally stored scientific registries."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.branches import checkout_identity
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import RegistryPaths


def build_parser(*, prog: str = "nro branch") -> argparse.ArgumentParser:
    """Construct branch administration options without discovering or creating state."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "action",
        choices=("list", "register", "attach", "show", "reparent", "retire", "definitions"),
    )
    parser.add_argument("name", nargs="?", help="Branch name for show, reparent, or retire")
    parser.add_argument(
        "--checkout", type=Path, default=Path.cwd(), help="Checkout to register or inspect"
    )
    parser.add_argument("--parent", help="Registered parent; new feature branches default to dev")
    parser.add_argument(
        "--bids-root", type=Path, default=BIDS_PATH, help="Select the shared registry context"
    )
    parser.add_argument("--json", action="store_true")
    definitions = parser.add_mutually_exclusive_group()
    definitions.add_argument(
        "--definitions", type=Path, help="Select an existing private definitions store"
    )
    definitions.add_argument(
        "--shared", action="store_true", help="Restore shared, read-only definitions"
    )
    return parser


def _description(store: BranchStore, name: str) -> dict:
    from nro.configuration.branch_definitions import read_selection
    from nro.configuration.site import settings
    from nro.orchestration.scheduler_implementation import implementation_path

    registry = store.registry(name)
    record = registry.record
    selected = read_selection(store.control, name, record.registry_id)
    return dict(
        branch=name,
        parent=record.parent,
        retired=record.retired,
        registry_id=record.registry_id,
        registry=str(registry.database),
        scheduler=str(ControlPaths(store.control).database),
        checkouts=[str(path) for path in record.checkouts],
        definitions=str(selected or Path(settings()[0]["definitions"]).resolve()),
        definitions_writable=name == "main" or selected is not None,
        execution_enabled=bool(
            record.checkouts and not record.retired and implementation_path(store.control).is_file()
        ),
    )


def main(argv: list[str] | None = None, *, prog: str = "nro branch") -> None:
    """Manage shared branch bindings without changing Git refs, outputs, or worker allocations.

    Registration creates scientific databases. An activated main scheduler is
    also required before a registered checkout can submit isolated work.
    """
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if args.definitions is not None or args.shared:
        if args.action != "definitions":
            parser.error("--definitions and --shared apply only to branch definitions")
    if args.action == "definitions" and (
        args.name or (args.definitions is None and not args.shared)
    ):
        parser.error(
            "definitions uses the checkout branch and requires --definitions PATH or --shared"
        )
    if args.name and args.action in {"list", "register", "attach"}:
        parser.error("register and attach derive the branch name from Git; list takes no name")
    if args.action in {"reparent", "retire"} and not args.name:
        parser.error(f"{args.action} requires a branch name")
    if args.action == "reparent" and not args.parent:
        parser.error("reparent requires --parent")
    if args.parent and args.action not in {"register", "reparent"}:
        parser.error("--parent applies only to register and reparent")
    paths = RegistryPaths.for_project("", bids_root=args.bids_root)
    store = BranchStore(paths.control)
    try:
        if args.action == "list":
            names = store.read().topology.records if store.path.exists() else ()
            result = [_description(store, name) for name in names]
        elif args.action == "register":
            root, name, _commit = checkout_identity(args.checkout)
            snapshot = store.initialize()
            existing = snapshot.topology.records.get(name)
            if existing is not None:
                if name not in {"main", "dev"} or existing.checkouts:
                    raise ValueError(f"Branch {name} is already registered; use nro branch attach")
                if args.parent is not None and args.parent != existing.parent:
                    raise ValueError("The main/dev parent relationship is fixed")
                store.authorize_checkout(name, root, revision=snapshot.revision)
            else:
                store.register(
                    name, args.parent or "dev", checkout=root, revision=snapshot.revision
                )
            result = _description(store, name)
        elif args.action == "attach":
            root, name, _commit = checkout_identity(args.checkout)
            snapshot = store.read()
            if name not in snapshot.topology.records:
                raise ValueError(f"Unregistered branch {name}; use nro branch register")
            store.authorize_checkout(name, root, revision=snapshot.revision)
            result = _description(store, name)
        elif args.action == "definitions":
            from nro.configuration.branch_definitions import select_definitions
            from nro.configuration.site import settings

            shared = Path(settings()[0]["definitions"])
            selected = select_definitions(store, args.checkout, shared, args.definitions)
            name = store.registry_for_checkout(args.checkout).record.name
            result = {
                **_description(store, name),
                "definitions": str(selected),
                "definitions_writable": not args.shared,
            }
        elif args.action == "show":
            name = args.name or store.registry_for_checkout(args.checkout).record.name
            result = _description(store, name)
        else:
            snapshot = store.read()
            from nro.orchestration.scheduler_implementation import implementation_path

            if implementation_path(store.control).is_file():
                from nro.orchestration.scheduler_client import maintenance

                maintenance(
                    store.control,
                    paths.bids_root,
                    checkout=args.checkout,
                    operation="branch_update",
                    branch=args.name,
                    action=args.action,
                    revision=snapshot.revision,
                    parent=args.parent,
                )
            elif args.action == "reparent":
                store.reparent(
                    args.name,
                    args.parent,
                    revision=snapshot.revision,
                    checkout=args.checkout,
                )
            else:
                store.retire(args.name, revision=snapshot.revision, checkout=args.checkout)
            result = _description(store, args.name)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif isinstance(result, list):
            print("Branch\tParent\tRetired\tScientific registry")
            for item in result:
                print(
                    f"{item['branch']}\t{item['parent'] or '-'}\t{item['retired']}\t{item['registry']}"
                )
        else:
            print(f"Branch: {result['branch']} (parent: {result['parent'] or '-'})")
            print(f"Scientific registry: {result['registry']}")
            print(f"Shared scheduler registry: {result['scheduler']}")
            if "definitions" in result:
                access = "writable" if result["definitions_writable"] else "read-only"
                print(f"Definitions: {result['definitions']} ({access})")
            print(
                "Branch execution: "
                + (
                    "enabled"
                    if result["execution_enabled"]
                    else "requires an activated main scheduler"
                )
            )
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nBranch operation interrupted; registration can be resumed.\n")


if __name__ == "__main__":
    main()
