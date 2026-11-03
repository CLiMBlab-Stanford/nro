"""Create, inspect, check, and run study analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nro.analysis.layout import (
    LoadedAnalysis,
    StudyError,
    StudySource,
    load_analysis,
    parse_study_config,
    split_analysis_name,
    studies,
)
from nro.analysis.manifest import ManifestError


def _selection(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise SystemExit(f"Selectors take the form KEY=VALUE, not {value!r}")
        result[key] = item
    return result


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("analysis", metavar="STUDY/ANALYSIS")
    parser.add_argument(
        "selectors",
        nargs="*",
        metavar="KEY=VALUE",
        help="Restrict work items by an expanded parameter (value or label) or a run entity.",
    )
    parser.add_argument("--project", action="append", help="Member project; repeatable.")
    parser.add_argument("--participant", action="append", help="Participant; repeatable.")


def build_parser(*, prog: str = "nro study") -> argparse.ArgumentParser:
    """Construct the ``nro study`` parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True, metavar="ACTION")

    init = commands.add_parser("init", help="Create a study repository and results directory.")
    init.add_argument("study")
    init.add_argument("--project", action="append", default=[], help="Member project; repeatable.")

    add = commands.add_parser("add", help="Write an analysis skeleton without committing it.")
    add.add_argument("analysis", metavar="STUDY/ANALYSIS")
    add.add_argument("--scope", default="subject", choices=("subject", "run"))

    listing = commands.add_parser("list", help="List studies and analyses on main.")
    listing.add_argument("study", nargs="?")

    validate = commands.add_parser("validate", help="Check manifests without running code.")
    validate.add_argument("target", nargs="?", metavar="STUDY[/ANALYSIS]")
    validate.add_argument("--worktree", action="store_true", help="Check uncommitted files.")

    check = commands.add_parser("check", help="Plan without running and print work items.")
    _add_selection(check)
    check.add_argument("--worktree", action="store_true", help="Plan from uncommitted files.")
    check.add_argument("--study-ref", default="main", help="Branch, tag, or commit to plan.")
    check.add_argument("--json", action="store_true", help="Print work items as JSON.")

    trial = commands.add_parser(
        "try", help="Run one work item from the working tree into a scratch directory."
    )
    _add_selection(trial)
    trial.add_argument("--scratch", type=Path, help="Directory for outputs and the log.")

    run = commands.add_parser("run", help="Run stale work items and publish their outputs.")
    _add_selection(run)
    run.add_argument("--study-ref", default="main", help="Branch, tag, or commit to run.")
    run.add_argument("--dry-run", action="store_true", help="Report what would run.")
    run.add_argument("--force", action="store_true", help="Rerun fresh work items too.")
    return parser


def _filter(items, selectors: dict[str, str]):
    if not selectors:
        return items
    result = []
    for item in items:
        keep = True
        for key, value in selectors.items():
            if key in item.labels:
                keep = value in {item.labels[key], str(item.parameters[key])}
            elif key in item.entities:
                keep = item.entities[key] == value
            else:
                raise SystemExit(f"{key} is neither an expanded parameter nor a run entity")
            if not keep:
                break
        if keep:
            result.append(item)
    return result


def _plan(analysis: LoadedAnalysis, args: argparse.Namespace):
    from nro.analysis.planning import plan_analysis
    from nro.analysis.resolve import default_index
    from nro.site.configuration import bids_root

    items = plan_analysis(
        analysis,
        index=default_index(),
        bids_root=bids_root(),
        projects=args.project,
        participants=args.participant,
    )
    return _filter(items, _selection(args.selectors))


def _load(name: str, ref: str | None) -> LoadedAnalysis:
    study, analysis = split_analysis_name(name)
    return load_analysis(StudySource.at(study, ref), analysis)


def _describe(item) -> dict:
    return {
        "key": item.key,
        "status": item.status,
        "message": item.message,
        "parameters": item.parameters,
        "inputs": {
            name: (
                None
                if value.value is None
                else [str(path) for path in value.value.paths]
                if hasattr(value.value, "paths")
                else str(value.value.path)
            )
            for name, value in item.inputs.items()
        },
        "outputs": {slot: str(path) for slot, path in item.outputs.items()},
    }


def _print_items(items, *, verbose: bool) -> None:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
        print(f"{item.status:<15} {item.key}")
        if item.message:
            print(f"    {item.message}")
        if verbose:
            for name, value in _describe(item)["inputs"].items():
                shown = value if isinstance(value, list) else [value]
                print(f"    input {name}:")
                for entry in shown:
                    print(f"      {entry}")
            for slot, path in item.outputs.items():
                print(f"    output {slot}: {path}")
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(f"{len(items)} work items" + (f": {summary}" if summary else ""))


def _list(study: str | None) -> None:
    names = [study] if study else list(studies())
    if not names:
        print("No studies found.")
    for name in names:
        try:
            source = StudySource.at(name)
            config = parse_study_config(source.read_text("study.yml"), study=name)
        except (StudyError, ValueError) as error:
            print(f"{name}: invalid ({error})")
            continue
        print(f"{name}  projects: {', '.join(config.projects) or '(none)'}")
        for analysis in source.analyses():
            try:
                loaded = load_analysis(source, analysis)
                state = f"{loaded.manifest.scope}, valid"
            except (StudyError, ManifestError) as error:
                state = f"invalid: {error}"
            last = source.last_commit(f"analyses/{analysis}") or "uncommitted"
            print(f"  {analysis:<30} {last:<20} {state}")


def _validate(target: str | None, *, worktree: bool) -> int:
    ref = None if worktree else "main"
    failures = 0
    if target is None:
        targets = [(name, None) for name in studies()]
    elif "/" in target:
        targets = [split_analysis_name(target)]
    else:
        targets = [(target, None)]
    for study, analysis in targets:
        try:
            source = StudySource.at(study, ref)
            parse_study_config(source.read_text("study.yml"), study=study)
        except (StudyError, ValueError) as error:
            print(f"{study}: {error}")
            failures += 1
            continue
        for name in [analysis] if analysis else source.analyses():
            try:
                loaded = load_analysis(source, name)
            except (StudyError, ManifestError) as error:
                print(f"{study}/{name}: {error}")
                failures += 1
                continue
            for warning in loaded.manifest.warnings:
                print(f"{study}/{name}: warning: {warning}")
            print(f"{study}/{name}: valid")
    return failures


def main(argv: list[str] | None = None, *, prog: str = "nro study") -> None:
    """Dispatch one ``nro study`` action."""
    args = build_parser(prog=prog).parse_args(argv)
    try:
        if args.action == "init":
            from nro.analysis.scaffold import init_study

            path = init_study(args.study, projects=tuple(args.project))
            print(f"Created {path}")
            if not args.project:
                print("Add member projects to study.yml and commit.")
        elif args.action == "add":
            from nro.analysis.scaffold import add_analysis

            study, analysis = split_analysis_name(args.analysis)
            path = add_analysis(study, analysis, scope=args.scope)
            print(f"Created {path}. Edit it, run nro study check --worktree, then commit.")
        elif args.action == "list":
            _list(args.study)
        elif args.action == "validate":
            if _validate(args.target, worktree=args.worktree):
                raise SystemExit(1)
        elif args.action == "check":
            analysis = _load(args.analysis, None if args.worktree else args.study_ref)
            items = _plan(analysis, args)
            if args.json:
                print(json.dumps([_describe(item) for item in items], indent=2, default=str))
            else:
                print(f"{analysis.name} at {analysis.source.commit[:12]}")
                _print_items(items, verbose=True)
        elif args.action == "try":
            _try(args)
        elif args.action == "run":
            _run(args)
    except (StudyError, ManifestError) as error:
        raise SystemExit(str(error)) from None


def _try(args: argparse.Namespace) -> None:
    from nro.analysis.execution import execute
    from nro.engine.paths import study_work_root

    analysis = _load(args.analysis, None)
    items = [item for item in _plan(analysis, args) if item.status in {"ready", "fresh"}]
    if len(items) != 1:
        raise SystemExit(
            f"{len(items)} runnable work items match; give selectors that pick exactly one"
            + "".join(f"\n  {item.key}" for item in items[:20])
        )
    item = items[0]
    scratch = (
        args.scratch or study_work_root(analysis.study.name) / "try" / analysis.manifest.analysis
    )
    result = execute(item, publish=False, work_dir=scratch.resolve())
    print(f"{item.key}: {'succeeded' if result.succeeded else 'failed'}")
    if result.message:
        print(f"  {result.message}")
    print(f"  log: {result.log}")
    for path in item.outputs.values():
        print(f"  output: {scratch.resolve() / 'outputs' / path.name}")
    if not result.succeeded:
        raise SystemExit(1)


def _run(args: argparse.Namespace) -> None:
    from nro.analysis.execution import execute

    analysis = _load(args.analysis, args.study_ref)
    items = _plan(analysis, args)
    runnable = [
        item for item in items if item.status == "ready" or (args.force and item.status == "fresh")
    ]
    blocked = [item for item in items if item.status in {"error", "waiting"}]
    fresh = sum(1 for item in items if item.status == "fresh" and item not in runnable)
    print(f"{analysis.name} at {analysis.source.commit[:12]}")
    if args.dry_run:
        _print_items(items, verbose=False)
        return
    failures = 0
    for item in runnable:
        result = execute(item)
        state = "published" if result.succeeded else "failed"
        print(f"{state:<15} {item.key}")
        if not result.succeeded:
            failures += 1
            print(f"    {result.message}; log: {result.log}")
    for item in blocked:
        print(f"{item.status:<15} {item.key}\n    {item.message}")
    print(
        f"{len(runnable) - failures} published, {failures} failed, "
        f"{fresh} fresh, {len(blocked)} blocked, "
        f"{sum(1 for item in items if item.status == 'not_applicable')} not applicable"
    )
    if failures or blocked:
        sys.exit(1)


if __name__ == "__main__":
    main()
