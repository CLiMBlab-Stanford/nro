"""Write a referenced methods section for the module lineages a workflow selects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nro.methods.writers import FORMATS


def build_parser(*, prog: str = "nro.bin.methods") -> argparse.ArgumentParser:
    """Construct the methods parser without executing the command."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("workflow", help="Workflow ID whose lineages are described.")
    parser.add_argument(
        "-P",
        "--project",
        nargs="+",
        action="extend",
        default=None,
        metavar="PROJECT",
        help="Projects whose fresh derivatives supply observed counts and run tables. "
        "Several projects are pooled into one report.",
    )
    parser.add_argument(
        "-p",
        "--participant",
        nargs="+",
        action="extend",
        default=None,
        metavar="ID",
        help="Count only these BIDS participant IDs. Requires --project.",
    )
    parser.add_argument(
        "-r",
        "--run",
        nargs="+",
        action="extend",
        default=None,
        metavar="ENTITY=VALUE[,VALUE...]",
        help="Count only matching BIDS runs; values are comma-delimited alternatives per "
        "entity. Requires --project.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        action="extend",
        default=None,
        help="Count only these source BIDS tasks. Requires --project.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Describe the configuration only, without reading the registry or manifests.",
    )
    parser.add_argument("--format", required=True, choices=FORMATS, dest="output_format")
    parser.add_argument(
        "-m",
        "--module",
        action="append",
        dest="modules",
        help="Describe this module and its upstream lineages. Repeatable. "
        "The default describes every module that has methods templates.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("."), help="Output directory (default: .)."
    )
    parser.add_argument(
        "--templates",
        type=Path,
        action="append",
        default=[],
        help="Override directory laid out as MODULE/NN_section.md. Earlier directories win.",
    )
    return parser


def _selection(args: argparse.Namespace):
    """Parse the participant and run selectors with the shared selection rules."""
    from nro.engine.cli import core_selection
    from nro.methods.report import ManifestSelection

    shared = argparse.Namespace(
        participant=args.participant,
        project=args.project,
        module=None,
        workflow=None,
        run=args.run,
        space=None,
        smoothing=None,
        task=args.task,
        lineage=None,
    )
    selection = core_selection(shared, apply_planner_defaults=False)
    return ManifestSelection(participants=selection.participants, runs=selection.runs)


def main(argv: list[str] | None = None, *, prog: str = "nro.bin.methods") -> None:
    """Render the methods files into the output directory.

    argv excludes the executable name; None reads the process arguments.
    Invalid arguments and unreportable workflows raise SystemExit.
    """
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if bool(args.project) == bool(args.preview):
        parser.error("give exactly one of --project or --preview")
    if args.preview and (args.participant or args.run or args.task):
        parser.error("--participant, --run, and --task require --project")
    try:
        selection = _selection(args)
    except ValueError as error:
        parser.error(str(error))

    from nro.definitions.store import ConfigStore, WorkflowError
    from nro.methods.bibliography import BibliographyError
    from nro.methods.report import (
        MethodsReportError,
        build_report,
        no_manifests,
        registry_manifests,
        write_report,
    )
    from nro.methods.software import installed_package_versions, pinned_sources, site_images
    from nro.methods.templates import MethodsTemplateError
    from nro.versioning import application_version

    try:
        workflow = ConfigStore().resolve(args.workflow)
        source = no_manifests
        if args.project:
            from nro.orchestration.registry import Registry
            from nro.site.configuration import settings

            bids = Path(settings()[0]["bids"])
            registries = {}
            for project in dict.fromkeys(args.project):
                registry = Registry.for_project(project, bids_root=bids)
                if not registry.existing_database_path().is_file():
                    raise MethodsReportError(f"No registry exists for project {project!r}")
                registries[project] = registry
            source = registry_manifests(registries, selection)
            print("Reading fresh derivative manifests...", file=sys.stderr, flush=True)
        report = build_report(
            workflow,
            output_format=args.output_format,
            modules=args.modules,
            manifests=source,
            overrides=tuple(args.templates),
            nro_version=application_version(),
            images=site_images,
            pins=pinned_sources(),
            package_versions=installed_package_versions(),
            scope=None
            if args.preview
            else {
                "projects": list(dict.fromkeys(args.project)),
                "participants": list(selection.participants),
                "runs": {
                    key: None if value is None else list(value)
                    for key, value in selection.runs.items()
                },
            },
        )
    except (WorkflowError, MethodsReportError, MethodsTemplateError, BibliographyError) as error:
        raise SystemExit(f"nro methods: {error}") from error
    for path in write_report(report, args.output):
        print(path)
    for lineage in report.trace["lineages"]:
        excluded = lineage["excluded_work_items"]
        if excluded:
            details = ", ".join(f"{count} {state}" for state, count in excluded.items())
            print(
                f"{lineage['module']}/{lineage['directory']}: excluded {details} work items",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
