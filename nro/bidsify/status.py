"""Read-only ingestion status independent of cloud inventory and derivative discovery."""

from .store import IngestionStore


def selected_records(registry, selection) -> list[dict]:
    """Filter unfinished requests using applicable project, participant, and session selectors."""
    return filter_records(IngestionStore(registry).rows(), selection)


def filter_records(records, selection) -> list[dict]:
    """Select unfinished records already scoped to an authorized branch."""
    if (
        selection.modules
        or selection.workflows
        or selection.models
        or selection.model_sets
        or selection.spaces
        or selection.smoothing
    ):
        return []
    sessions = selection.runs.get("ses", ())
    if set(selection.runs) - {"ses"}:
        return []
    return [
        {
            k: r[k]
            for k in (
                "id",
                "server",
                "project",
                "participant",
                "session",
                "state",
                "stage",
                "issues",
            )
        }
        for r in records
        if r["state"] not in {"published", "cancelled"}
        and (not selection.projects or r["project"] in selection.projects)
        and (not selection.participants or r["participant"] in selection.participants)
        and (not sessions or r["session"] in sessions)
    ]


def render(records: list[dict], *, color: bool = False) -> str:
    """Render unfinished requests with a resumable command and no remote labels."""
    if not records:
        return ""
    lines = [
        "\nBidsification",
        f"{'PROJECT':14} {'PARTICIPANT':14} {'SESSION':20} {'STATE':20} NEXT ACTION",
    ]
    for r in records:
        command = "—" if r["state"] in {"queued", "running"} else f"nro bidsify --request {r['id']}"
        state = f"{r['state'].replace('_', ' '):20}"
        if color:
            tone = (
                "91"
                if r["state"] in {"failed", "interrupted"}
                else "93"
                if command != "—"
                else "96"
            )
            state = f"\x1b[{tone}m{state}\x1b[0m"
        participant, session = r["participant"] or "(pending)", r["session"] or "(pending)"
        lines.append(f"{r['project']:14} {participant:14} {session:20} {state} {command}")
    return "\n".join(lines) + "\n"
