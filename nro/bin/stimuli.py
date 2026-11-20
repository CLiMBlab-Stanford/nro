"""Publish stimulus annotations to the definitions catalog and copy them into projects."""

import argparse
from pathlib import Path

from nro.site import configuration as site


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    """Publish staged annotations, or synchronize one project's ``stimuli/`` directory."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser(
        "publish",
        help="Publish a staged directory of stimulus annotations to the definitions catalog",
        description=(
            "Validate a staged directory in the layout of DEFINITIONS/stimuli, assign each "
            "annotation's NROVersion, add new stimuli.tsv and annotations.tsv rows, and "
            "publish the batch to the selected definitions store in one transaction."
        ),
    )
    publish.add_argument("directory", type=Path, help="Staged catalog directory")
    publish.add_argument(
        "--dry-run", action="store_true", help="Report what would change without publishing"
    )
    sync = commands.add_parser(
        "sync",
        help="Copy the annotations of every stimulus that the project's events name",
    )
    sync.add_argument(
        "projects",
        nargs="+",
        metavar="PROJECT",
        help="BIDS projects under the configured BIDS root",
    )
    sync.add_argument(
        "--replace",
        action="store_true",
        help="Also replace stale copies that nro wrote from an older catalog version",
    )
    sync.add_argument(
        "--dry-run", action="store_true", help="Report differences without changing files"
    )
    args = parser.parse_args(argv)
    if args.command == "publish":
        _publish(args)
        return

    from nro.definitions.stimuli import StimulusCatalog
    from nro.definitions.stimulus_sync import describe, plan_sync

    roots = {}
    for project in args.projects:
        root = site.bids_root() / project
        if "/" in project or not root.is_dir() or not any(root.glob("sub-*")):
            parser.error(f"Unknown BIDS project: {project}")
        roots[project] = root
    if not args.dry_run and site.installation_record().get("mode") == "branch":
        parser.error(
            "A development installation cannot change shared source BIDS; use --dry-run, "
            "or run sync from the central installation"
        )
    try:
        catalog = StimulusCatalog()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    incomplete, failed = [], []
    for project, root in roots.items():
        if len(roots) > 1:
            print(f"== {project}")
        try:
            plan = plan_sync(root, catalog)
            copied = (
                []
                if args.dry_run or not (plan.writes or (args.replace and plan.stale))
                else _apply_guarded(plan, project, replace=args.replace)
            )
        except (OSError, ValueError, SystemExit) as error:
            # One project's failure does not stop the others.
            failed.append(f"{project}: {error}")
            continue
        for line in describe(plan, copied, replace=args.replace):
            print(line)
        if not plan.references:
            print(f"No events in {project} name a stimulus in stim_id.")
        if plan.unknown or plan.blocked:
            incomplete.append(project)
    if failed:
        raise SystemExit("Sync failed:\n" + "\n".join(failed))
    if incomplete:
        raise SystemExit(
            f"Some stimuli in {', '.join(incomplete)} could not be synchronized. Add unknown "
            "stimuli to the catalog with `nro def apply`, and resolve blocked files by hand."
        )


def _publish(args: argparse.Namespace) -> None:
    from nro.definitions.stimulus_publish import describe, publish_stimuli

    try:
        plan = publish_stimuli(args.directory, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    for line in describe(plan, dry_run=args.dry_run):
        print(line)
    if plan.changed and not args.dry_run:
        print("Copy the new versions into projects with `nro stimuli sync PROJECT`.")


def _apply_guarded(plan, project: str, *, replace: bool):
    """Apply a sync while holding the registry lock, refusing during active attempts.

    Changing a fit input under a running attempt would leave its result tied
    to files it did not read. Holding the write lock keeps new attempts from
    starting until the copy finishes, as bidsify publication does.
    """
    from nro.definitions.stimulus_sync import active_attempts, apply_sync
    from nro.orchestration.registry import Registry

    registry = Registry.for_project(project, bids_root=site.bids_root())
    if not registry.existing_database_path().is_file():
        return apply_sync(plan, replace=replace)
    with registry.connection(write=True) as db:
        active = active_attempts(db, project)
        if active:
            raise SystemExit(
                f"Project {project} has active derivative attempts ({', '.join(active)}); "
                "stop them or wait for them to finish before changing stimulus annotations"
            )
        return apply_sync(plan, replace=replace)


if __name__ == "__main__":
    main()
