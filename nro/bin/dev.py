"""Run tests selected from the current change's development spheres."""

from __future__ import annotations

import argparse
from pathlib import Path

from nro.engine.development import changed_paths, load_scopes, run_selection, select_tests
from nro.engine.io import atomic_write_text


def build_parser(*, prog: str = "nro dev") -> argparse.ArgumentParser:
    """Build the development-validation command parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "operation", choices=("test", "schema"), help="Run tests or inspect registry schemas"
    )
    parser.add_argument("schema_operation", nargs="?", choices=("show", "build", "diff", "check"))
    parser.add_argument("--sphere", action="append", default=[], help="Select a named sphere")
    parser.add_argument("--full", action="store_true", help="Run the complete test suite")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the selection without testing"
    )
    parser.add_argument("--family", choices=("scheduler", "scientific"), default="scheduler")
    parser.add_argument("--version", type=int, help="Select a generated schema version")
    parser.add_argument("--from-version", type=int, help="Select the first schema in a diff")
    parser.add_argument("--to-version", type=int, help="Select the second schema in a diff")
    parser.add_argument("--output", type=Path, help="Write a generated schema to this path")
    parser.add_argument(
        "--base-ref", help="Check that a Git comparison target has the same baseline"
    )
    return parser


def main(argv: list[str] | None = None, *, prog: str = "nro dev") -> None:
    """Select validation from Git changes or explicit spheres and run it."""
    args = build_parser(prog=prog).parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.operation == "schema":
        from nro.orchestration.migrations.tools import (
            diff,
            render,
            require_unchanged_baselines,
            validate_families,
        )

        operation = args.schema_operation or "show"
        if operation == "check":
            validate_families()
            if args.base_ref:
                require_unchanged_baselines(root, args.base_ref)
            print("Registry migration chains are valid")
            return
        if operation == "diff":
            if args.from_version is None or args.to_version is None:
                raise SystemExit("schema diff requires --from-version and --to-version")
            value = diff(args.family, args.from_version, args.to_version)
        else:
            value = render(args.family, version=args.version)
        if operation == "build":
            if args.output is None:
                raise SystemExit("schema build requires --output")
            atomic_write_text(args.output, value)
            print(args.output)
        else:
            print(value, end="" if value.endswith("\n") else "\n")
        return
    scopes = load_scopes(root / "development/test-scopes.toml")
    changes = () if args.sphere else changed_paths(root)
    selection = select_tests(scopes, changes, requested=args.sphere)
    print("Development spheres: " + (", ".join(selection.spheres) or "none"))
    if selection.unmapped:
        print("Unmapped paths (full validation required): " + ", ".join(selection.unmapped))
    effective_full = args.full or selection.full or not selection.tests
    print("Validation: " + ("full suite" if effective_full else ", ".join(selection.tests)))
    if not args.dry_run:
        raise SystemExit(run_selection(root, selection, full=effective_full))


if __name__ == "__main__":
    main()
