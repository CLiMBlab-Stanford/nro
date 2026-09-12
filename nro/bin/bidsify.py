"""Review and schedule Flywheel-to-BIDS ingestion without planning derivatives."""

import argparse
import json
import os
import sys
from pathlib import Path

from nro.bidsify.config import bids_label, identifier, load_config
from nro.bidsify.discovery import inferred_session, session_choices
from nro.bidsify.flywheel import FlywheelSource
from nro.bidsify.identity import destination_label
from nro.bidsify.publication import approval_snapshot
from nro.bidsify.review import SkipSession, ask, choose_indices, wizard
from nro.bidsify.store import IngestionStore, ReviewBusyError
from nro.configuration.site import bids_root as configured_bids_root
from nro.configuration.site import settings
from nro.orchestration.registry import Registry
from nro.orchestration.submission import _submit_workers, _write_worker_script


def build_parser(*, prog="nro bidsify"):
    """Build selectors for a single destination project and resumable ingestion requests."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "-f",
        "--flywheel-server",
        dest="flywheel_server",
        help="Configured Flywheel server profile",
    )
    parser.add_argument("--server", dest="flywheel_server", help=argparse.SUPPRESS)
    parser.add_argument("-P", "--project", help="Destination BIDS project")
    parser.add_argument("-F", "--flywheel-project", help="Source GROUP/PROJECT for new sessions")
    parser.add_argument("-p", "--participant", nargs="+", default=[])
    parser.add_argument("--session", nargs="+", default=[], help="Remote session IDs to include")
    parser.add_argument("--request", help="Resume one request without listing Flywheel sessions")
    parser.add_argument("--config", type=Path, help="Complete ingestion profile YAML")
    parser.add_argument(
        "--rebidsify",
        action="store_true",
        help="Include sessions already in BIDS, regardless of producer; replacement still needs approval",
    )
    parser.add_argument(
        "--no-submit", action="store_true", help="Save work without supplying new workers"
    )
    parser.add_argument("--execution", help=argparse.SUPPRESS)
    return parser


def select_source(
    config: dict, project: str, *, server: str | None, flywheel_project: str | None
) -> tuple[str, str]:
    """Select one server/project pair from the destination's allowed sources."""
    sources = config["project_sources"].get(project)
    if sources is None:
        sources = [
            dict(server=name, project=path)
            for name, profile in config["servers"].items()
            for path in profile["projects"]
        ]
    choices = list(
        dict.fromkeys(
            (s["server"], s["project"])
            for s in sources
            if (server is None or s["server"] == server)
            and (flywheel_project is None or s["project"] == flywheel_project)
        )
    )
    if not choices:
        raise ValueError(
            f"No source configured for BIDS project {project} matches the requested server/Flywheel project"
        )
    if len(choices) == 1:
        return choices[0]
    print(f"Select a source for BIDS project {project}:")
    for number, (name, path) in enumerate(choices, 1):
        print(f"{number}: {name}/{path}")
    while True:
        answer = ask("Source number, SERVER/GROUP/PROJECT, or unique GROUP/PROJECT")
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        matched = [pair for pair in choices if answer in {pair[1], "/".join(pair)}]
        if len(matched) == 1:
            return matched[0]
        print("Select one displayed source.")


def advance(store: IngestionStore, record: dict) -> dict:
    """Lease only the session being opened; skip occupied sessions without locking the selection."""
    record = store.admit_pending(record["id"])
    if record["state"] in {"running", "published", "cancelled"} or (
        record["state"] == "queued" and record["executor_uid"] == os.getuid()
    ):
        print(f"{record['id']}: {record['state']}")
        return record
    try:
        with store.review_session(record["id"]) as token:
            return _advance(store, store.get(record["id"]), review_token=token)
    except ReviewBusyError as error:
        print(f"{record['id']}: {error}")
        return store.get(record["id"])
    except SkipSession:
        print(f"{record['id']}: left for later")
        return store.get(record["id"])


def _advance(store: IngestionStore, record: dict, *, review_token: str) -> dict:
    print(f"\n{record['id']} {destination_label(record)}: {record['state']}")
    print("Type skip to leave this session for later, or q to exit.")
    if record["state"] in {"running", "published"}:
        return record
    if record["state"] == "queued":
        if (
            record["executor_uid"] != os.getuid()
            and ask("Execute this request using your authenticated account? y/N", "n").lower()
            == "y"
        ):
            return store.update(
                record, expected_revision=record["revision"], review_token=review_token
            )
        print("Already queued; no duplicate request will be submitted.")
        return record
    if record["state"] in {"failed", "interrupted"}:
        print("Log: " + str(store.root / f"{record['id']}.log"))
        print("\n".join(record.get("issues", [])))
        action = ask("Retry, review decisions, cancel, or leave? r/e/c/l", "l")
        if action == "r":
            record.update(state="queued")
        elif action == "e" and record["stage"] == "convert":
            record.update(
                state="needs_input",
                stage="convert",
                approval=None,
            )
        elif action == "c":
            record.update(state="cancelled")
        else:
            return record
        record = store.update(
            record, expected_revision=record["revision"], review_token=review_token
        )
    if record["state"] == "needs_input":
        return wizard(store, record, review_token=review_token)
    if record["state"] == "awaiting_approval":
        snapshot = approval_snapshot(record, store.registry, branch_paths=store.branch_paths)
        print("Staged outputs (SHA256):")
        for name, digest in snapshot["outputs"].items():
            print(f"  {name}  {digest}")
        if snapshot["target_exists"]:
            print(
                "Replacement removes the current session after atomic exchange. Stop external readers first."
            )
            for name in snapshot["existing"]:
                print("  Existing: " + name)
        print("Validation passed. No unresolved acquisition or event assignments remain.")
        answer = ask(
            "Approve publication of these exact files, edit decisions, or leave? y/e/N", "n"
        ).lower()
        if answer == "e":
            record.update(state="needs_input", stage="convert", approval=None)
            record = store.update(
                record, expected_revision=record["revision"], review_token=review_token
            )
            return wizard(store, record, review_token=review_token)
        if answer == "y":
            record.update(state="queued", stage="publish", approval=snapshot)
            return store.update(
                record, expected_revision=record["revision"], review_token=review_token
            )
    return record


def main(argv=None, *, prog="nro bidsify"):
    """Select new or unfinished sessions, collect decisions, and supply central workers."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        from nro.bidsify.execution import launch_review, validate_execution
        from nro.configuration.site import installation_record
        from nro.orchestration.scheduler_implementation import implementation_path

        site_values, _ = settings()
        flywheel_server = args.flywheel_server or site_values.get("flywheel_server") or None
        flywheel_project = args.flywheel_project or site_values.get("flywheel_project") or None
        if not args.execution and (
            installation_record().get("mode") == "branch"
            or implementation_path(Path(settings()[0]["registry"])).exists()
        ):
            launch_review(list(sys.argv[1:] if argv is None else argv))
            return
        pin = json.loads(args.execution) if args.execution else None
        branch_paths = validate_execution(pin) if pin else None
        bids_root = configured_bids_root()
        registry = Registry.for_project("", bids_root=bids_root)
        config = load_config(args.config)
        if not args.request and not config["servers"]:
            raise ValueError(
                "Configure Flywheel servers in the definitions store: bidsify/main.yml"
            )
        registry.initialize()
        registry.recover_orphaned_attempts()
        store = IngestionStore(registry, branch_paths=branch_paths, execution=pin)
        records = []
        if args.request:
            records = [store.get(args.request)]
        else:
            project = identifier(args.project or ask("Destination BIDS project"))
            server, remote_project = select_source(
                config,
                project,
                server=flywheel_server,
                flywheel_project=flywheel_project,
            )
            all_records = store.rows()
            saved = [
                r
                for r in all_records
                if r["server"] == server
                and r["project"] == project
                and (not args.participant or r["participant"] in args.participant)
                and (not args.session or r["remote_session"] in args.session)
            ]
            unfinished = [r for r in saved if r["state"] not in {"published", "cancelled"}]
            for number, row in enumerate(unfinished, 1):
                print(f"{number}: {destination_label(row)} {row['state']}")
            if unfinished:
                answer = ask("Resume listed request numbers, all, or n for new sessions", "all")
                if answer != "n":
                    indices = choose_indices(answer, len(unfinished))
                    records = [unfinished[i] for i in indices]
            if not records:
                print(f"Source: {server}/{remote_project} -> BIDS project: {project}")
                source = FlywheelSource({**config["servers"][server], "projects": [remote_project]})
                remote = source.sessions()
                discovery = session_choices(
                    remote,
                    server=server,
                    bids_root=store.project_root(project).parent,
                    rules=config["session_rules"],
                    records=all_records,
                    rebidsify=args.rebidsify,
                    sessions=args.session,
                )
                choices = discovery.rows
                if discovery.existing_hidden:
                    print(
                        f"Omitted {discovery.existing_hidden} sessions already bidsified; use --rebidsify to include them."
                    )
                if discovery.active_hidden:
                    print(
                        f"Omitted {discovery.active_hidden} sessions with unfinished nro requests; use nro status to locate them."
                    )
                for number, row in enumerate(choices, 1):
                    label = (
                        "ambiguous BIDS mapping; review required"
                        if row["mapping_ambiguous"]
                        else "already bidsified"
                        if row["already_bidsified"]
                        else "no known BIDS match"
                    )
                    print(
                        f"{number}: {row['id']} {row['label']} ({row['remote_project']}; {label})"
                    )
                    for match in row["existing_bids"]:
                        print(f"   Existing: {match.path}")
                if not choices:
                    print("No matching sessions available.")
                    return
                answer = ask("Select session numbers, or all")
                indices = choose_indices(answer, len(choices))
                for index in indices:
                    selected = choices[index]
                    print("Remote session: " + selected["id"])
                    if selected["mapping_ambiguous"]:
                        print(
                            "Existing destination is ambiguous; resolve the session rules before re-bidsification."
                        )
                        continue
                    existing = selected["existing_bids"][0] if selected["existing_bids"] else None
                    if existing and existing.project != project:
                        raise ValueError(
                            f"This session already belongs to BIDS project {existing.project}; "
                            "re-bidsification cannot reassign it"
                        )
                    prior = next(
                        (r for r in reversed(saved) if r["remote_session"] == selected["id"]), None
                    )
                    answer = ask(
                        "BIDS participant label; Enter to leave pending",
                        existing.participant
                        if existing
                        else (prior["participant"] or "")
                        if prior
                        else (args.participant[0] if len(args.participant) == 1 else ""),
                    )
                    participant = bids_label(answer) if answer else None
                    session = existing.session if existing else prior["session"] if prior else None
                    session = session or inferred_session(
                        selected, server=server, rules=config["session_rules"]
                    )
                    if session:
                        print(f"BIDS session: ses-{session}")
                    else:
                        answer = ask(
                            "BIDS session label could not be inferred; enter a label or leave pending"
                        )
                        session = bids_label(answer) if answer else None
                    if existing and (participant, session) != (
                        existing.participant,
                        existing.session,
                    ):
                        raise ValueError(
                            "Re-bidsification must retain the existing participant/session destination"
                        )
                    destination = destination_label(
                        dict(project=project, participant=participant, session=session)
                    )
                    if ask(f"Queue inspection for {destination}? y/N", "n").lower() != "y":
                        continue
                    records.append(
                        store.create(
                            server=server,
                            remote_session=selected["id"],
                            project=project,
                            participant=participant,
                            session=session,
                            config=config,
                            replace=args.rebidsify,
                        )
                    )
        for record in records:
            advance(store, record)
        queued = [store.get(r["id"]) for r in records if store.get(r["id"])["state"] == "queued"]
        if queued and not args.no_submit:
            memory = max(r["config"]["memory_gb"] for r in queued)
            script = _write_worker_script(
                registry,
                bids_root=bids_root,
                partition=site_values["partition"],
                account=site_values["account"],
                hours=max(r["config"]["hours"] for r in queued),
                memory_gb=memory,
                cpus=max(r["config"]["cpus"] for r in queued),
            )
            jobs = _submit_workers(registry, None, script, memory)
            print("Submitted workers: " + (", ".join(jobs) or "existing pool has capacity"))
        for record in records:
            print(f"Resume: nro bidsify --request {record['id']}")
    except (EOFError, KeyboardInterrupt, SkipSession):
        print("\nSaved decisions retained. Running jobs continue; reopen nro bidsify to resume.")
    except (ValueError, OSError, IndexError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
