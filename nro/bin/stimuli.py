"""Copy stimulus annotations from the definitions catalog into BIDS projects."""

import argparse

from nro.site import configuration as site


def main(argv: list[str] | None = None, *, prog: str | None = None) -> None:
    """Compare or update one project's ``stimuli/`` directory against the catalog."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser(
        "sync",
        help="Copy the annotations of every stimulus that the project's events name",
    )
    sync.add_argument("project", help="BIDS project under the configured BIDS root")
    sync.add_argument(
        "--replace",
        action="store_true",
        help="Also replace stale copies that nro wrote from an older catalog version",
    )
    sync.add_argument(
        "--dry-run", action="store_true", help="Report differences without changing files"
    )
    args = parser.parse_args(argv)

    from nro.definitions.stimuli import StimulusCatalog
    from nro.definitions.stimulus_sync import describe, plan_sync

    project_root = site.bids_root() / args.project
    if "/" in args.project or not project_root.is_dir() or not any(project_root.glob("sub-*")):
        parser.error(f"Unknown BIDS project: {args.project}")
    if not args.dry_run and site.installation_record().get("mode") == "branch":
        parser.error(
            "A development installation cannot change shared source BIDS; use --dry-run, "
            "or run sync from the central installation"
        )
    try:
        plan = plan_sync(project_root, StimulusCatalog())
        copied = (
            []
            if args.dry_run or not (plan.writes or (args.replace and plan.stale))
            else _apply_guarded(plan, args.project, replace=args.replace)
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    for line in describe(plan, copied, replace=args.replace):
        print(line)
    if not plan.references:
        print(f"No events in {args.project} name a stimulus in stim_id.")
    if plan.unknown or plan.blocked:
        raise SystemExit(
            "Some stimuli could not be synchronized. Add unknown stimuli to the catalog with "
            "`nro def apply`, and resolve blocked files by hand."
        )


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
