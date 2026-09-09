"""Convert the flat private-control store to the shared/branch directory layout."""

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from nro.configuration.paths import REGISTRY_PATH
from nro.configuration.site import require_execution_support
from nro.orchestration.control_cutover import execute, preview, rollback
from nro.orchestration.control_paths import ControlPaths


def main(argv=None, *, prog="nro cutover") -> None:
    """Preview and confirm one quiescent layout conversion, or recover an interruption."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=REGISTRY_PATH, help="Site-wide private-control root"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run", action="store_true", help="Read and hash private files without writing"
    )
    action.add_argument(
        "--resume", action="store_true", help="Finish an interrupted, fully staged publication"
    )
    action.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the original store after an unfinished cutover",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Skip confirmation, never quiescence checks"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        require_execution_support()
        root = args.registry.expanduser().resolve()
        if (args.resume or args.rollback) and not ControlPaths(root).cutover_journal.is_file():
            raise ValueError("There is no interrupted cutover to recover")
        plan = None if args.resume or args.rollback else preview(root)
        if plan is not None:
            if args.json and args.dry_run:
                print(json.dumps(asdict(plan), default=str, indent=2))
            else:
                stream = sys.stderr if args.json else sys.stdout
                print(
                    f"Control root: {root}\nCopy: {plan.files} files, {plan.bytes / 2**30:.2f} GiB",
                    file=stream,
                )
                for source, target in plan.mappings:
                    print(f"  {source} -> {target}", file=stream)
        if args.dry_run:
            return
        if not args.force:
            print(
                "Coordinate a maintenance window: all clients must stop using this store. No workers will be stopped automatically."
            )
            if input(
                "Proceed with private-state cutover recovery? [y/N]: "
                if args.resume or args.rollback
                else "Copy and publish this layout, retaining the original store? [y/N]: "
            ).strip().lower() not in {"y", "yes"}:
                parser.exit(0, "Cutover cancelled; no state changed.\n")
        if args.rollback:
            retained = rollback(root)
            result = {"rolled_back": True, "retained_staging": str(retained) if retained else None}
        else:
            backup = execute(root, expected_fingerprint=plan.source_fingerprint if plan else None)
            result = {"complete": True, "backup": str(backup)}
        print(
            json.dumps(result)
            if args.json
            else "\n".join(f"{key}: {value}" for key, value in result.items())
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        sqlite3.Error,
        yaml.YAMLError,
    ) as error:
        parser.exit(1, f"Cutover not completed: {error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(
            130, "\nCutover interrupted. If a journal exists, use --resume or --rollback.\n"
        )


if __name__ == "__main__":
    main()
