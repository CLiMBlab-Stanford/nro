"""Create, migrate, validate, and safely update a definitions store."""

import argparse
import json
from pathlib import Path

from nro.configuration import site
from nro.configuration.definition_migrations import migrate_store, update_store
from nro.configuration.definitions import create_store, validate_store
from nro.engine.definition_editor import read_definition, review_definition


def _context(root: Path) -> tuple[bool, Path | None, tuple[Path, ...]]:
    """Return site and inheritance arguments for validating one selected store."""
    record = site.installation_record()
    shared = Path(site.settings()[0]["definitions"]).resolve()
    private = record.get("mode") == "branch" and root.resolve() != shared
    active = site.definitions_roots()
    inherited = (
        active[1:]
        if private and active and root.resolve() == active[0]
        else ((shared,) if private else ())
    )
    inherited_site = shared if private and not (root / "site/site.yml").is_file() else None
    return not private, inherited_site, tuple(inherited)


def _validate(root: Path) -> dict[str, int]:
    require_site, inherited_site, inherited_roots = _context(root)
    return validate_store(
        root,
        require_site=require_site,
        inherited_site=inherited_site,
        inherited_roots=inherited_roots,
    )


def _validator(root: Path):
    require_site, inherited_site, inherited_roots = _context(root)

    def validate(candidate: Path) -> None:
        validate_store(
            candidate,
            require_site=require_site,
            inherited_site=inherited_site,
            inherited_roots=inherited_roots,
        )

    return validate


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise argparse.ArgumentTypeError("definition paths must be relative to the store")
    if path == Path("site/site.yml"):
        raise argparse.ArgumentTypeError("use `nro paths` to update protected site settings")
    return path


def _root(value: Path | None) -> Path:
    return value.expanduser().absolute() if value is not None else site.definitions_root()


def _report(root: Path, counts: dict[str, int], *, json_output: bool, verb: str) -> None:
    if json_output:
        print(json.dumps({"path": str(root), **counts}, indent=2))
        return
    print(f"{verb} {root}")
    print(", ".join(f"{key}: {value}" for key, value in counts.items()))


def _create(args: argparse.Namespace) -> tuple[Path, str]:
    root = _root(args.path)
    record = site.installation_record()
    if record.get("mode") == "branch":
        shared = Path(site.settings()[0]["definitions"])
        create_store(root, include_site=False, inherited_site=shared)
    else:
        create_store(root)
    return root, "Created"


def _migrate(args: argparse.Namespace) -> tuple[Path, str]:
    root = _root(args.path)
    with site.definition_write(root):
        changed = migrate_store(root, validate=_validator(root))
    return root, "Migrated" if changed else "Validated"


def _apply(args: argparse.Namespace) -> tuple[Path, str]:
    root = site.definitions_root()
    updates: dict[Path, bytes | None] = {}
    for assignment in args.file:
        if "=" not in assignment:
            raise ValueError("--file requires RELATIVE_PATH=LOCAL_FILE")
        relative_text, source_text = assignment.split("=", 1)
        relative = _relative(relative_text)
        source = Path(source_text).expanduser()
        if relative in updates:
            raise ValueError(f"Definition was specified more than once: {relative}")
        updates[relative] = source.read_bytes()
    for relative in args.delete:
        if relative in updates:
            raise ValueError(f"Definition was specified more than once: {relative}")
        updates[relative] = None
    if not updates:
        raise ValueError("Specify at least one --file or --delete operation")
    first = next(iter(updates))
    with site.definition_write(root / first):
        for relative in updates:
            site.require_definition_write(root / relative)
        update_store(root, updates, validate=_validator(root), adopt_drift=True)
    return root, "Updated"


def _edit(args: argparse.Namespace) -> tuple[Path, str]:
    root = site.definitions_root()
    path = root / args.relative
    site.require_definition_write(path)
    expected = read_definition(path)
    if expected is None:
        raise ValueError(f"Definition does not exist: {path}; use `nro definitions apply`")
    changed = review_definition(
        path,
        expected.decode("utf-8"),
        expected=expected,
        validate=lambda _text: None,
        source=args.source,
        yes=args.yes,
        store_root=root,
        validate_store=_validator(root),
    )
    return root, "Updated" if changed else "Validated"


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "migrate", "validate"):
        command = commands.add_parser(action)
        command.add_argument("path", nargs="?", type=Path, help="Default: selected store")
        command.add_argument("--json", action="store_true")
    apply = commands.add_parser("apply", help="Publish one validated batch of local files")
    apply.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="RELATIVE=FILE",
        help="Set a store-relative definition from a local file; repeat as needed",
    )
    apply.add_argument(
        "--delete",
        action="append",
        default=[],
        type=_relative,
        metavar="RELATIVE",
        help="Delete a definition in the same transaction; repeat as needed",
    )
    apply.add_argument("--json", action="store_true")
    edit = commands.add_parser("edit", help="Edit an existing definition with full-store checks")
    edit.add_argument("relative", type=_relative, metavar="RELATIVE")
    edit.add_argument("--file", dest="source", type=Path)
    edit.add_argument("-y", "--yes", action="store_true")
    edit.add_argument("--json", action="store_true")
    return parser


def main(argv=None, *, prog="nro definitions"):
    """Manage definitions without changing derivatives, work, or registry state."""
    parser = _parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.action == "create":
            root, verb = _create(args)
        elif args.action == "migrate":
            root, verb = _migrate(args)
        elif args.action == "apply":
            root, verb = _apply(args)
        elif args.action == "edit":
            root, verb = _edit(args)
        else:
            root, verb = _root(args.path), "Validated"
        _report(root, _validate(root), json_output=args.json, verb=verb)
        if args.action == "create" and not args.json:
            command = (
                "nro branch definitions --definitions PATH"
                if site.installation_record().get("mode") == "branch"
                else "nro paths set definitions=PATH"
            )
            print(f"Store selection is unchanged. Use {command} to select this store.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nDefinitions operation cancelled.\n")


if __name__ == "__main__":
    main()
