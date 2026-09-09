"""Match remote session identities to existing BIDS directories without auditing data."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

from nro.bidsify.config import bids_label


def validate_session_rules(rules: object, servers: dict) -> None:
    """Validate exact-match rules and BIDS label templates without reading source data."""
    if not isinstance(rules, list):
        raise ValueError("session_rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {
            "server",
            "remote_project",
            "match",
            "participant",
            "session",
        }:
            raise ValueError(
                "Session rules require server, remote_project, match, participant, and session"
            )
        if not isinstance(rule["server"], str) or rule["server"] not in servers:
            raise ValueError("Session rule names an unknown server")
        if rule["remote_project"] not in servers[rule["server"]]["projects"]:
            raise ValueError("Session rule names an unconfigured remote project")
        match = rule["match"]
        if not isinstance(match, dict) or not match or set(match) - {"id", "label", "subject_code"}:
            raise ValueError("Session rule match needs id, label, and/or subject_code patterns")
        groups = set()
        for pattern in match.values():
            if not isinstance(pattern, str):
                raise ValueError("Session rule patterns must be strings")
            try:
                groups.update(re.compile(pattern).groupindex)
            except re.error as error:
                raise ValueError(f"Invalid session rule pattern: {error}") from error
        if rule["participant"] is None and rule["session"] is None:
            raise ValueError("Session rules must specify a participant or session label")
        for field in ("participant", "session"):
            template = rule[field]
            if template is None:
                continue
            if not isinstance(template, str):
                raise ValueError("BIDS label templates must be strings or null")
            for _, name, spec, conversion in Formatter().parse(template):
                if name is not None and (name not in groups or spec or conversion):
                    raise ValueError(
                        "BIDS templates may reference only named regex groups without formatting"
                    )
            bids_label(template.format_map({name: "x" for name in groups}))


@dataclass(frozen=True)
class ExistingSession:
    """An existing raw BIDS session directory, with no validity attestation."""

    project: str
    participant: str
    session: str
    path: Path


def existing_sessions(bids_root: Path) -> tuple[ExistingSession, ...]:
    """List raw subject/session directories across projects, excluding derivatives and sourcedata.

    Subject-only intake layouts do not count as completed session destinations.
    Do not inspect files or require nro receipts, images, metadata, or validation.
    """
    result = []
    if not bids_root.exists():
        return ()
    for project in sorted(bids_root.iterdir()):
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project.name)
            or project.name in {"derivatives", "sourcedata"}
            or not project.is_dir()
        ):
            continue
        for subject in sorted(project.iterdir()):
            participant = re.fullmatch(r"sub-([A-Za-z0-9]+)", subject.name)
            if participant is None or not subject.is_dir():
                continue
            sessions = []
            for path in subject.iterdir():
                match = re.fullmatch(r"ses-([A-Za-z0-9]+)", path.name)
                if match and path.is_dir():
                    sessions.append(ExistingSession(project.name, participant[1], match[1], path))
            result.extend(sessions)
    return tuple(sorted(result, key=lambda row: str(row.path)))


def _rule_labels(remote: dict, rule: dict) -> tuple[str | None, str | None] | None:
    groups = {}
    for field, pattern in rule["match"].items():
        value = remote.get(field)
        match = re.fullmatch(pattern, value) if isinstance(value, str) else None
        if match is None:
            return None
        for name, value in match.groupdict().items():
            if value is None or (name in groups and groups[name] != value):
                return None
            groups[name] = value
    try:
        return tuple(
            bids_label(rule[field].format_map(groups)) if rule[field] is not None else None
            for field in ("participant", "session")
        )
    except ValueError:
        return None


def inferred_session(remote: dict, *, server: str, rules: list[dict]) -> str | None:
    """Use the sole session label supplied by matching server rules, or require review.

    Sessionless discovery rules do not identify a stable participant. Never
    copy their participant labels or an unconfigured remote display label.
    """
    candidates = set()
    for rule in rules:
        if rule["server"] == server and rule["remote_project"] == remote["remote_project"]:
            labels = _rule_labels(remote, rule)
            if labels is not None and labels[1] is not None:
                candidates.add(labels[1])
    return next(iter(candidates)) if len(candidates) == 1 else None


@dataclass(frozen=True)
class SessionChoices:
    """Transient selection rows and counts of existing or active remote sessions omitted."""

    rows: tuple[dict, ...]
    existing_hidden: int
    active_hidden: int


def session_choices(
    remote: list[dict],
    *,
    server: str,
    bids_root: Path,
    rules: list[dict],
    records: list[dict],
    rebidsify: bool = False,
    sessions=(),
) -> SessionChoices:
    """Suppress known existing data while retaining ambiguous identity matches for review.

    Match across the shared BIDS root, not only the proposed destination project.
    Published nro records are authoritative identity mappings. Active nro requests
    stay in their recovery workflow even when their destination already exists.
    Do not create publication records for external data or inspect their contents.
    """
    inventory = existing_sessions(bids_root)
    index = defaultdict(set)
    for entry in inventory:
        index[(entry.participant, entry.session)].add(entry)
        index[(None, entry.session)].add(entry)
    saved = [r for r in records if r["server"] == server]
    active = {r["remote_session"] for r in saved if r["state"] not in {"published", "cancelled"}}
    published = {r["remote_session"] for r in saved if r["state"] == "published"}
    matched, users = {}, defaultdict(set)
    for row in remote:
        matches = set()
        for rule in rules:
            if rule["server"] != server or rule["remote_project"] != row["remote_project"]:
                continue
            labels = _rule_labels(row, rule)
            if labels is not None:
                matches.update(index[labels])
        matched[row["id"]] = tuple(sorted(matches, key=lambda entry: str(entry.path)))
        for entry in matches:
            users[entry].add(row["id"])
    choices, existing_hidden, active_hidden, seen = [], 0, 0, set()
    for row in remote:
        remote_id = row["id"]
        if remote_id in seen or (sessions and remote_id not in sessions):
            continue
        seen.add(remote_id)
        if remote_id in active:
            active_hidden += 1
            continue
        matches = matched[remote_id]
        ambiguous = len(matches) > 1 or any(len(users[entry]) > 1 for entry in matches)
        existing = remote_id in published or (len(matches) == 1 and not ambiguous)
        if existing and not rebidsify:
            existing_hidden += 1
            continue
        choices.append(
            {
                **row,
                "existing_bids": matches,
                "mapping_ambiguous": ambiguous,
                "already_bidsified": existing,
            }
        )
    return SessionChoices(tuple(choices), existing_hidden, active_hidden)
