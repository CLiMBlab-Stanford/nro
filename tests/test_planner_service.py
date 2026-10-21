"""The scheduler runs pinned planning in one isolated FIFO subprocess lane."""

from __future__ import annotations

import json
import subprocess
import threading
from types import SimpleNamespace

from nro.orchestration import planning_subprocess, scheduler_service


def test_planner_child_uses_pinned_checkout_and_thread_limits(tmp_path, monkeypatch):
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
    monkeypatch.setattr(planning_subprocess, "SourceSnapshot", lambda root, digest: snapshot)
    calls = []

    class Process:
        returncode = 0

        def communicate(self, timeout=None):
            return json.dumps({"requests": []}), ""

    monkeypatch.setattr(
        planning_subprocess.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or Process(),
    )

    result = planning_subprocess.execute(
        {
            "source": {"root": str(source), "digest": "digest"},
            "site": str(site),
            "python": str(python),
            "checkout": str(checkout),
            "argv": ["-P", "demo", "--json"],
        },
        threads=8,
    )

    assert result == {"requests": []}
    command, options = calls[0]
    assert command[-1] == "--json"
    assert command.count("--json") == 1
    assert options["env"]["NRO_REMOTE_PLANNER"] == "1"
    assert options["env"]["NRO_PLANNER_SOURCE_ROOT"] == str(source)
    assert options["env"]["NRO_PLANNER_SITE"] == str(site)
    assert options["env"]["NRO_CHECKOUT"] == str(checkout)
    assert options["env"]["OMP_NUM_THREADS"] == "2"
    assert options["env"]["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] == "2"


def test_scheduler_shutdown_terminates_active_planner_child(tmp_path, monkeypatch):
    source = tmp_path / "source"
    site = tmp_path / "site.toml"
    python = tmp_path / "python"
    checkout = tmp_path / "checkout"
    source.mkdir()
    checkout.mkdir()
    site.write_text("")
    python.write_text("")
    snapshot = SimpleNamespace(
        root=source,
        digest="digest",
        verify_manifest=lambda: None,
        command=lambda command, *, site: tuple(command),
    )
    monkeypatch.setattr(planning_subprocess, "SourceSnapshot", lambda root, digest: snapshot)
    stopped = threading.Event()

    class Process:
        returncode = None

        def communicate(self, timeout=None):
            if timeout == 1.0:
                stopped.set()
                raise subprocess.TimeoutExpired("planner", timeout)
            return "", ""

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        planning_subprocess.subprocess, "Popen", lambda *_args, **_kwargs: Process()
    )
    stop = threading.Event()
    stop.set()

    try:
        planning_subprocess.execute(
            {
                "source": {"root": str(source), "digest": "digest"},
                "site": str(site),
                "python": str(python),
                "checkout": str(checkout),
                "argv": ["-P", "demo"],
            },
            stop=stop,
        )
    except RuntimeError as error:
        assert str(error) == "Planning stopped with the scheduler"
    else:
        raise AssertionError("cancelled planning unexpectedly succeeded")
    assert stopped.is_set()


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


def test_scheduler_dispatches_planning_in_its_allocation(tmp_path, monkeypatch):
    source = SimpleNamespace(root=tmp_path / "source", digest="digest")
    endpoint = SimpleNamespace(source=source)
    calls = []
    phases = []
    site_roots = []
    implementation = tmp_path / "implementation.json"
    checkout = tmp_path / "checkout"
    implementation.write_text(json.dumps({"checkout": str(checkout)}))
    monkeypatch.setattr(
        "nro.orchestration.scheduler_bus.publish_progress",
        lambda _control, _message_id, *, phase, completed, total: phases.append(phase),
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_client.command",
        lambda control, bids_root: calls.append(("command", control, bids_root)) or endpoint,
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_implementation.implementation_path",
        lambda _control: implementation,
    )
    monkeypatch.setattr(
        "nro.orchestration.execution_pins.capture_site",
        lambda root, _values: site_roots.append(root) or tmp_path / "site.toml",
    )

    def execute(request, *, stop, threads):
        calls.append(("execute", request, stop, threads))
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

    monkeypatch.setattr(planning_subprocess, "execute", execute)

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
            "checkout": str(checkout),
            "request": {
                "execution": {
                    "kind": "installed",
                    "digest": "digest",
                    "site_values": {},
                },
                "checkout": str(checkout),
                "python": "/python",
                "argv": ["-P", "demo"],
            },
        },
        values={"scheduler_cpus": 4},
        message_id="message",
    )

    assert result["requests"] == ["request-1"]
    assert [call[0] for call in calls] == ["command", "execute", "admit", "supply"]
    assert calls[1][1]["source"] == {"root": str(source.root), "digest": "digest"}
    assert calls[1][2] is scheduler_service._PLANNING_STOP
    assert calls[1][3] == 2
    assert phases == [
        "Planning requested work",
        "Inspecting existing artifacts",
        "Registering requested work",
        "Reconciling worker pool",
    ]
    assert site_roots == [tmp_path / "control/shared/cache/execution-sites"]
