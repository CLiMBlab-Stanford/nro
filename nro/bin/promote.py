"""Accept equivalent development derivatives after merging their implementation."""

import argparse
import json
import sys
from pathlib import Path

from nro.configuration.site import CHECKOUT, settings
from nro.configuration.store import ConfigStore
from nro.engine.cli import add_core_selection_arguments, core_selection
from nro.orchestration.branch_requests import register_requests
from nro.orchestration.branch_store import BranchStore
from nro.orchestration.catalog import MODULES, terminal_modules
from nro.orchestration.planner import Planner
from nro.orchestration.scheduler_client import maintenance
from nro.orchestration.selection import discover_bids_inventory


def main(argv=None, *, prog="nro promote"):
    """Compile target contracts, review transfers, and publish without requesting computation."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    add_core_selection_arguments(
        parser, module_choices=MODULES, planner_defaults=True, default_modules=terminal_modules()
    )
    parser.add_argument(
        "--from", dest="source", required=True, help="Development branch supplying artifacts"
    )
    parser.add_argument("--pr", required=True, help="Accepted pull request reference")
    parser.add_argument(
        "--attest-merged",
        action="store_true",
        help="Confirm the reviewed implementation was merged into this checkout",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Allow replacement of incompatible target-owned outputs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-f", "--force", action="store_true", help="Skip the transfer confirmation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.attest_merged:
            raise ValueError("--attest-merged is required; promotion does not merge code")
        selection = core_selection(args)
        values = settings()[0]
        control, bids = Path(values["registry"]), Path(values["bids"])
        scientific = BranchStore(control).registry_for_checkout(CHECKOUT)
        workflows = {name: ConfigStore().resolve(name) for name in selection.workflows}
        registered = {
            name: scientific.register_workflow(workflow) for name, workflow in workflows.items()
        }
        revisions = {row.key: row.revision for row in scientific.instances()}
        plan = Planner(scientific, bids_root=bids).plan(
            projects=selection.projects or tuple(discover_bids_inventory(bids)),
            requested_participants=selection.participants,
            modules=selection.modules,
            workflows=workflows,
            registered_workflows=registered,
            selectors=selection.runs,
            spaces=selection.spaces,
            smoothing_levels=selection.smoothing,
            memory_gb=8,
            max_memory_gb=64,
            models=selection.models,
            model_sets=selection.model_sets,
        )
        if plan.unavailable:
            raise ValueError(
                "Some selected work is unavailable: "
                + "; ".join(item.reason for item in plan.unavailable)
            )
        requests = register_requests(
            scientific,
            plan,
            selectors={},
            concurrency=1,
            partition=None,
            expected_revisions=revisions,
            demand=False,
        )
        report = maintenance(
            control,
            bids,
            checkout=CHECKOUT,
            operation="promotion_preview",
            source=args.source,
            requests=requests,
            pr=args.pr,
            attest=args.attest_merged,
        )
        if args.dry_run:
            print(json.dumps(report, indent=2))
            return
        for item in report["items"]:
            print(f"{item['action']}: {item['root']}", file=sys.stderr if args.json else sys.stdout)
        if not args.force and input(
            "Publish these artifacts, retaining the source copies? [y/N] "
        ).lower() not in {"y", "yes"}:
            return
        result = maintenance(
            control,
            bids,
            checkout=CHECKOUT,
            operation="promotion_publish",
            report=report,
            replace=args.replace,
            attest=args.attest_merged,
        )
        print(
            json.dumps(result, indent=2)
            if args.json
            else f"Promoted {result['promoted']} artifact(s); retained {result['retained']} fresh target artifact(s)."
        )
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "Promotion cancelled. No source artifacts were removed.\n")
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
