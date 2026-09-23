"""Edit, inspect, validate, and maintain definition stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.configuration import site
from nro.configuration.authoring import main as author
from nro.configuration.definition_migrations import MANAGED_CATEGORIES, migrate_store, update_store
from nro.configuration.definitions import create_store, validate_store
from nro.configuration.store import CONFIGURATION_CLASSES, ConfigStore
from nro.engine.definition_editor import delete_definition, read_definition, review_definition

TYPED_KINDS = frozenset({"config", "model", "workflow", "markup"})


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


def _init(args: argparse.Namespace) -> tuple[Path, str]:
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
    for relative in args.rm:
        if relative in updates:
            raise ValueError(f"Definition was specified more than once: {relative}")
        updates[relative] = None
    if not updates:
        raise ValueError("Specify at least one --file or --rm operation")
    first = next(iter(updates))
    with site.definition_write(root / first):
        for relative in updates:
            site.require_definition_write(root / relative)
        update_store(root, updates, validate=_validator(root), adopt_drift=True)
    return root, "Updated"


def _edit_file(args: argparse.Namespace) -> tuple[Path, str]:
    root = site.definitions_root()
    path = root / args.relative
    site.require_definition_write(path)
    expected = read_definition(path)
    changed = review_definition(
        path,
        expected.decode("utf-8") if expected is not None else "",
        expected=expected,
        validate=lambda _text: None,
        source=args.source,
        store_root=root,
        validate_store=_validator(root),
    )
    return root, "Updated" if changed else "Unchanged"


def _rm_file(args: argparse.Namespace) -> tuple[Path, str]:
    root = site.definitions_root()
    path = root / args.relative
    site.require_definition_write(path)
    expected = read_definition(path)
    if expected is None:
        raise ValueError(f"Definition does not exist: {path}")
    print(f"Remove definition file: {path}")
    print("Derivatives, logs, registry records, and workers are unchanged.")
    if not args.yes:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            raise ValueError("Removal requires interactive confirmation or --yes")
        if input("Remove this definition? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Cancelled; the stored definition is unchanged.")
            return root, "Unchanged"
    backup = delete_definition(
        path,
        expected=expected,
        store_root=root,
        validate_store=_validator(root),
    )
    print(f"Removed {path}\nRecovery copy: {backup} (temporary; copy elsewhere to retain it)")
    return root, "Updated"


def _source_label(store: ConfigStore, path: Path) -> str:
    for index, root in enumerate(store.roots):
        if path.is_relative_to(root):
            return "selected" if index == 0 else f"inherited:{index}"
    return "packaged"


def _definition_rows(kind: str, selector: str | None) -> list[dict[str, str]]:
    store = ConfigStore()
    rows: list[dict[str, str]] = []
    if kind == "config":
        classes = (selector,) if selector else CONFIGURATION_CLASSES
        unknown = set(classes) - set(CONFIGURATION_CLASSES)
        if unknown:
            raise ValueError(
                "Choose a configuration class from " + ", ".join(CONFIGURATION_CLASSES)
            )
        for configuration_class in classes:
            suffix = f"_{configuration_class}.yml"
            identifiers = {"main"}
            for root in store.roots:
                identifiers.update(
                    path.name.removesuffix(suffix)
                    for path in (root / "configs" / configuration_class).glob(f"*{suffix}")
                    if path.is_file()
                )
            for identifier in sorted(identifiers, key=lambda value: (value != "main", value)):
                path = store.configuration_path(configuration_class, identifier)
                rows.append(
                    {
                        "class": configuration_class,
                        "id": identifier,
                        "source": _source_label(store, path),
                    }
                )
    elif kind == "model":
        seen: set[tuple[str, str]] = set()
        for root in store.roots:
            for path in sorted((root / "models").glob("*/*.yml")):
                key = (path.parent.name, path.stem)
                if key in seen or (selector and key[0] != selector):
                    continue
                seen.add(key)
                rows.append(
                    {"task": key[0], "variant": key[1], "source": _source_label(store, path)}
                )
    elif kind == "workflow":
        if selector:
            raise ValueError("Workflow listings do not take a selector")
        for identifier in store.workflow_ids():
            path = store.workflow_path(identifier)[1]
            rows.append({"id": identifier, "source": _source_label(store, path)})
    elif kind == "markup":
        if selector:
            raise ValueError("Markup listings do not take a selector")
        seen: set[str] = set()
        suffix = "_markup.yml"
        for root in store.roots:
            for path in sorted((root / "markup").glob(f"*{suffix}")):
                identifier = path.name.removesuffix(suffix)
                if identifier in seen:
                    continue
                seen.add(identifier)
                rows.append({"id": identifier, "source": _source_label(store, path)})
    else:
        if selector:
            raise ValueError("File listings do not take a selector")
        seen: set[Path] = set()
        for root in store.roots:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or any(
                    part.startswith(".") or part == "__pycache__"
                    for part in path.relative_to(root).parts
                ):
                    continue
                relative = path.relative_to(root)
                if relative.parts[0] not in MANAGED_CATEGORIES:
                    continue
                if relative in seen:
                    continue
                seen.add(relative)
                rows.append({"path": str(relative), "source": _source_label(store, path)})
    return rows


def _list(args: argparse.Namespace) -> None:
    rows = _definition_rows(args.kind, args.selector)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    columns = {
        "config": ("class", "id", "source"),
        "model": ("task", "variant", "source"),
        "workflow": ("id", "source"),
        "markup": ("id", "source"),
        "file": ("path", "source"),
    }[args.kind]
    widths = {
        column: max(len(column.upper()), *(len(row[column]) for row in rows)) for column in columns
    }
    print("  ".join(column.upper().ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("edit", "rm"):
        command = commands.add_parser(
            action,
            description=f"{action.capitalize()} a typed definition or store-relative file.",
            epilog=(
                f"Run `{prog} {action} config --help`, `{prog} {action} model --help`, "
                f"or the corresponding workflow, markup, or file command for details."
            ),
        )
        command.add_argument(
            "arguments",
            nargs=argparse.REMAINDER,
            metavar="{config,model,workflow,markup,file} ...",
        )
    listing = commands.add_parser("ls", help="List active definitions and their source layer")
    listing.add_argument("kind", choices=("config", "model", "workflow", "markup", "file"))
    listing.add_argument("selector", nargs="?", help="Configuration class or model task")
    listing.add_argument("--json", action="store_true")
    for action in ("init", "migrate", "validate"):
        command = commands.add_parser(action)
        command.add_argument("path", nargs="?", type=Path, help="Default: selected store")
        command.add_argument("--json", action="store_true")
    apply = commands.add_parser("apply", help="Publish one validated batch of local files")
    apply.add_argument(
        "--file", action="append", default=[], metavar="RELATIVE=FILE", help="Set one file"
    )
    apply.add_argument(
        "--rm",
        action="append",
        default=[],
        type=_relative,
        metavar="RELATIVE",
        help="Remove one file",
    )
    apply.add_argument("--json", action="store_true")
    return parser


def _file_parser(action: str, prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("relative", type=_relative, metavar="RELATIVE")
    if action == "edit":
        parser.add_argument("--file", dest="source", type=Path)
    else:
        parser.add_argument("-y", "--yes", action="store_true")
    return parser


def main(argv=None, *, prog="nro def"):
    """Manage definitions without changing derivatives, work, or registry state."""
    parser = _parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.action in {"edit", "rm"}:
            if not args.arguments:
                parser.error(f"{args.action} requires a definition type")
            kind, *arguments = args.arguments
            if kind in TYPED_KINDS:
                author(args.action, [kind, *arguments], prog=f"{prog} {args.action}")
                return
            if kind != "file":
                parser.error(
                    f"unknown definition type {kind!r}; choose from config, model, workflow, markup, file"
                )
            file_args = _file_parser(args.action, f"{prog} {args.action} file").parse_args(
                arguments
            )
            root, verb = _edit_file(file_args) if args.action == "edit" else _rm_file(file_args)
            _report(root, _validate(root), json_output=False, verb=verb)
            return
        if args.action == "ls":
            _list(args)
            return
        if args.action == "init":
            root, verb = _init(args)
        elif args.action == "migrate":
            root, verb = _migrate(args)
        elif args.action == "apply":
            root, verb = _apply(args)
        else:
            root, verb = _root(args.path), "Validated"
        _report(root, _validate(root), json_output=args.json, verb=verb)
        if args.action == "init" and not args.json:
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
