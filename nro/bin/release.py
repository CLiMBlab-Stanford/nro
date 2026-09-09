"""Approve main releases or activate an approved shared scheduler installation."""

import argparse
import json
from pathlib import Path

from nro.configuration.paths import BIDS_PATH
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.registry import RegistryPaths
from nro.orchestration.releases import ReleaseStore


def main(argv=None, *, prog="nro release"):
    """Record approval or select future worker code without changing Git refs or derivatives."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "version", nargs="?", help="Committed main version to approve; omit to list approvals"
    )
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--pr", help="Reference identifying the approved and merged PR")
    parser.add_argument(
        "--attest-merged",
        action="store_true",
        help="Affirm that the referenced PR was approved and merged into main",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Approve the tagged initial 0.0.1 main commit without a PR",
    )
    parser.add_argument("--bids-root", type=Path, default=BIDS_PATH)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Designate this approved shared installation for future workers; requires a quiescent pool",
    )
    parser.add_argument(
        "--repair-scheduler",
        action="store_true",
        help="Stop the global pool and rebuild shared scheduling state, retaining branch databases and a backup",
    )
    args = parser.parse_args(argv)
    if args.repair_scheduler and (
        args.version or args.activate or args.pr or args.attest_merged or args.bootstrap
    ):
        parser.error("--repair-scheduler is a separate maintenance operation")
    if args.activate and (args.pr or args.attest_merged or args.bootstrap):
        parser.error("Approve first, then activate separately without attestation flags")
    if args.version and not args.activate:
        if args.bootstrap and (args.pr or args.attest_merged):
            parser.error("--bootstrap cannot be combined with PR attestation options")
        if not args.bootstrap and (not args.pr or not args.attest_merged):
            parser.error("Approval requires --pr and --attest-merged, or --bootstrap for 0.0.1")
    if not args.version and (args.pr or args.attest_merged or args.bootstrap):
        parser.error("Attestation requires a version")
    try:
        paths = RegistryPaths.for_project("", bids_root=args.bids_root)
        store = ReleaseStore(BranchStore(paths.control))
        if args.repair_scheduler:
            from nro.orchestration.registry import Registry
            from nro.orchestration.scheduler_repair import repair

            def confirm(activity):
                print(
                    "This stops all branches’ workers and removes active request/attempt state. "
                    "Public derivatives and branch scientific databases are retained; a backup is kept."
                )
                return input("Repair the shared scheduler? [y/N] ").strip().lower() in {"y", "yes"}

            result = repair(Registry(paths), checkout=args.checkout, confirm=confirm)
        elif args.activate:
            approved = store.require_approved(args.checkout)
            if args.version and approved["version"] != args.version:
                parser.error("Activation version does not match the approved checkout")
            from nro.orchestration.registry import Registry
            from nro.orchestration.scheduler_implementation import activate

            result = activate(Registry(paths), args.checkout)
        else:
            result = (
                store.approve(
                    args.checkout,
                    args.version,
                    pr=args.pr,
                    attest_merged=args.attest_merged,
                    bootstrap=args.bootstrap,
                )
                if args.version
                else store.history()
            )
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    except (KeyboardInterrupt, EOFError):
        parser.exit(130, "\nRelease operation interrupted; inspect approvals before retrying.\n")


if __name__ == "__main__":
    main()
