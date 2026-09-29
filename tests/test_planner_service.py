"""The planner broker serializes pinned planning outside the scheduler process."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from nro.orchestration import planner_bus, planner_client, planner_service, scheduler_service
from nro.orchestration.planner_bus import claim_launch, read_active, read_launch, update_launch_job


def test_planner_service_executes_requests_and_shuts_down(tmp_path, monkeypatch):
    control = tmp_path / "control"
    monkeypatch.setattr(planner_service.socket, "getfqdn", lambda: "127.0.0.1")
    claim = claim_launch(control)
    assert claim is not None
    seen = []
    monkeypatch.setattr(
        planner_service,
        "_run",
        lambda request, _stop=None: seen.append(request) or {"requests": ["request-1"]},
    )
    thread = threading.Thread(
        target=planner_service.serve,
        kwargs={"control": control, "launch_token": claim.token, "idle_grace": 30},
    )
    thread.start()
    for _ in range(100):
        if read_active(control) is not None:
            break
        time.sleep(0.01)

    request = {
        "source": {"root": "/source", "digest": "digest"},
        "site": "/site.toml",
        "python": "/python",
        "checkout": "/checkout",
        "argv": ["-P", "demo"],
    }
    assert planner_client.execute(control, request) == {"requests": ["request-1"]}
    assert seen == [request]
    assert planner_client.shutdown(control) == {"stopping": True}
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert read_active(control) is None


def test_planner_shutdown_interrupts_an_active_plan(tmp_path, monkeypatch):
    control = tmp_path / "control"
    monkeypatch.setattr(planner_service.socket, "getfqdn", lambda: "127.0.0.1")
    claim = claim_launch(control)
    assert claim is not None
    entered = threading.Event()

    def plan(_request, stop=None):
        entered.set()
        assert stop is not None
        stop.wait(5)
        raise RuntimeError("planning stopped")

    monkeypatch.setattr(planner_service, "_run", plan)
    service = threading.Thread(
        target=planner_service.serve,
        kwargs={"control": control, "launch_token": claim.token, "idle_grace": 30},
    )
    service.start()
    for _ in range(100):
        if read_active(control) is not None:
            break
        time.sleep(0.01)

    errors = []
    request = {
        "source": {"root": "/source", "digest": "digest"},
        "site": "/site.toml",
        "python": "/python",
        "checkout": "/checkout",
        "argv": ["-P", "demo"],
    }
    client = threading.Thread(
        target=lambda: _record_error(errors, planner_client.execute, control, request)
    )
    client.start()
    assert entered.wait(2)
    assert planner_client.shutdown(control) == {"stopping": True}
    client.join(timeout=5)
    service.join(timeout=5)

    assert not client.is_alive()
    assert not service.is_alive()
    assert errors and "planning stopped" in str(errors[0])


def test_planner_shutdown_cancels_a_pending_allocation(tmp_path, monkeypatch):
    control = tmp_path / "control"
    claim = claim_launch(control)
    assert claim is not None
    update_launch_job(claim, "12345")
    calls = []
    monkeypatch.setattr(
        planner_client.subprocess,
        "run",
        lambda command, **options: calls.append((command, options)),
    )

    assert planner_client.shutdown(control) == {"stopping": True}

    assert read_launch(control) is None
    assert calls[0][0] == ["scancel", "12345"]


def _record_error(errors, function, *args):
    try:
        function(*args)
    except BaseException as error:
        errors.append(error)


def test_planner_child_uses_pinned_checkout_and_structured_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    site = tmp_path / "site.toml"
    python = tmp_path / "python"
    checkout = tmp_path / "checkout"
    for path in (source, checkout):
        path.mkdir()
    for path in (site, python):
        path.write_text("")
    snapshot = SimpleNamespace(
        root=source,
        digest="digest",
        verify_manifest=lambda: None,
        command=lambda command, *, site: tuple(command),
    )
    monkeypatch.setattr(
        planner_service,
        "SourceSnapshot",
        lambda root, digest: snapshot,
    )
    calls = []

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            return json.dumps({"requests": []}), ""

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(planner_service.subprocess, "Popen", run)

    result = planner_service._run(
        {
            "source": {"root": str(source), "digest": "digest"},
            "site": str(site),
            "python": str(python),
            "checkout": str(checkout),
            "argv": ["-P", "demo", "--json"],
        }
    )

    assert result == {"requests": []}
    command, options = calls[0]
    assert command[-1] == "--json"
    assert command.count("--json") == 1
    assert options["env"]["NRO_REMOTE_PLANNER"] == "1"
    assert options["env"]["NRO_PLANNER_SOURCE_ROOT"] == str(source)
    assert options["env"]["NRO_PLANNER_SITE"] == str(site)
    assert options["env"]["NRO_CHECKOUT"] == str(checkout)


def test_scheduler_routes_plan_run_to_single_planner_lane():
    planner = object()
    command = object()
    executors = {
        "planner": planner,
        "command": command,
        "worker": object(),
        "poll": object(),
        "maintenance": object(),
    }

    assert (
        scheduler_service._executor_for({"payload": {"operation": "plan_run"}}, executors)
        is planner
    )
    assert (
        scheduler_service._executor_for({"payload": {"operation": "admit_many"}}, executors)
        is command
    )


def test_scheduler_dispatches_planning_through_the_broker(tmp_path, monkeypatch):
    endpoint = object()
    calls = []
    phases = []
    monkeypatch.setattr(
        "nro.orchestration.scheduler_bus.publish_progress",
        lambda _control, _message_id, *, phase, completed, total: phases.append(
            (phase, completed, total)
        ),
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_client.command",
        lambda control, bids_root: calls.append(("command", control, bids_root)) or endpoint,
    )
    monkeypatch.setattr(
        planner_client,
        "ensure",
        lambda value: calls.append(("ensure", value)),
    )

    def execute(control, request, *, progress):
        calls.append(("execute", control, request))
        progress("Planning requested work")
        return {
            "protocol": 1,
            "entries": [{"project": "demo", "payload": {}}],
            "options": {"no_submit": True},
            "projects": ["demo"],
            "participants": {"demo": ["01"]},
            "work_items": 1,
            "unavailable": [],
            "resumed": 0,
        }

    monkeypatch.setattr(planner_client, "execute", execute)

    def admit(_registry, entries, **options):
        progress = options.pop("progress")
        progress("Inspecting existing artifacts")
        progress("Registering requested work")
        calls.append(("admit", entries, options))
        return ["request-1"]

    monkeypatch.setattr(scheduler_service, "admit_many", admit)
    monkeypatch.setattr(
        scheduler_service,
        "supply",
        lambda registry, request_ids, options, **keywords: (
            calls.append(("supply", request_ids, options, keywords)) or {"submitted_workers": []}
        ),
    )
    registry = SimpleNamespace(
        paths=SimpleNamespace(control=tmp_path / "control", bids_root=tmp_path / "BIDS")
    )

    result = scheduler_service.dispatch(
        registry,
        {
            "operation": "plan_run",
            "checkout": str(tmp_path / "checkout"),
            "request": {"argv": ["-P", "demo"]},
        },
        values={},
        message_id="message",
    )

    assert result == {
        "submitted_workers": [],
        "requests": ["request-1"],
        "projects": ["demo"],
        "participants": {"demo": ["01"]},
        "work_items": 1,
        "unavailable": [],
        "resumed": 0,
    }
    assert calls == [
        ("command", registry.paths.control, registry.paths.bids_root),
        ("ensure", endpoint),
        ("execute", registry.paths.control, {"argv": ["-P", "demo"]}),
        (
            "admit",
            [{"project": "demo", "payload": {}}],
            {
                "checkout": tmp_path / "checkout",
                "site_values": {},
                "message_id": "message",
            },
        ),
        (
            "supply",
            ["request-1"],
            {"no_submit": True},
            {"checkout": tmp_path / "checkout"},
        ),
    ]
    assert [phase for phase, _completed, _total in phases] == [
        "Planning requested work",
        "Inspecting existing artifacts",
        "Registering requested work",
        "Reconciling worker pool",
    ]


def test_planner_script_requests_small_persistent_service(tmp_path):
    source = SimpleNamespace(
        command=lambda command, *, site: tuple(command),
    )
    script = planner_bus.write_script(
        tmp_path / "control",
        token="token",
        source=source,
        site=tmp_path / "site.toml",
        python=tmp_path / "python",
        partition="sphinx",
        account="nlp",
    )

    text = script.read_text()
    assert f"#SBATCH --cpus-per-task={planner_bus.PLANNER_CPUS}\n" in text
    assert f"#SBATCH --mem={planner_bus.PLANNER_MEMORY_GB}G\n" in text
    assert "#SBATCH --time=24:00:00\n" in text
    assert "#SBATCH --account=nlp\n" in text
