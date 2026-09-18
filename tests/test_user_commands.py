from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.bin.run import (
    _resumable_rows,
    _resume_workflows,
    _terminal_resume_rows,
    _write_worker_script,
    build_parser,
)
from nro.bin.run import main as run_main
from nro.bin.set import main as set_main
from nro.bin.status import _render_report
from nro.bin.status import main as status_main
from nro.bin.stop import build_parser as stop_parser
from nro.bin.stop import main as stop_main
from nro.configuration.store import ConfigStore
from nro.engine.cli import CoreSelection, core_selection, page_text
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.registry import SCHEMA_VERSION, Registry


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_run_defaults() -> None:
    args = build_parser().parse_args([])
    selection = core_selection(args)
    assert selection.participants == ()
    assert selection.projects == ()
    assert selection.modules == ("dynconn", "networks", "firstlevels")
    assert selection.workflows == ("main",)
    assert selection.spaces == ("fsnative",)
    assert selection.smoothing == (2,)
    assert args.concurrency == 50
    assert args.cpus == 2
    assert args.partition == "sphinx"


def test_run_cpu_override() -> None:
    assert build_parser().parse_args(["--cpus", "8"]).cpus == 8


def test_scheduler_exchange_reports_timeout(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: None)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="did not respond within 0.01 seconds"):
        scheduler_client.exchange(endpoint, {"operation": "status"}, timeout=0.01)


def test_scheduler_exchange_does_not_time_out_while_slurm_controller_is_pending(
    monkeypatch, tmp_path
) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    responses = iter((None, None, {"result": {"ok": True}}))
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: next(responses))
    monkeypatch.setattr(
        scheduler_bus,
        "read_launch",
        lambda *_args: {"token": "launch", "job_id": "12345"},
    )
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "message_path", lambda *_args: tmp_path / "absent")
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_client.time, "sleep", lambda _seconds: None)

    assert scheduler_client.exchange(endpoint, {"operation": "status"}, timeout=0) == {"ok": True}


def test_scheduler_exchange_times_out_if_local_controller_does_not_activate(
    monkeypatch, tmp_path
) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: None)
    monkeypatch.setattr(
        scheduler_bus,
        "read_launch",
        lambda *_args: {"token": "launch", "job_id": "local-123"},
    )
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: None)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="did not respond within 0 seconds"):
        scheduler_client.exchange(endpoint, {"operation": "status"}, timeout=0)


def test_scheduler_exchange_reports_malformed_response(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: {"unexpected": True})
    monkeypatch.setattr(scheduler_bus, "message_path", lambda *_args: tmp_path / "absent")
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="malformed response"):
        scheduler_client.exchange(endpoint, {"operation": "status"})


def test_scheduler_exchange_reports_startup_failure(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_launch", lambda *_args: {"token": "launch"})
    monkeypatch.setattr(
        scheduler_bus,
        "read_startup_error",
        lambda *_args: {"error": "database is locked"},
    )
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="could not start: database is locked"):
        scheduler_client.exchange(endpoint, {"operation": "status"})


def test_scheduler_exchange_preserves_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(
        scheduler_client,
        "_ensure_coordinator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        scheduler_client.exchange(endpoint, {"operation": "purge"})


def test_installation_maintenance_uses_one_shot_coordinator(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    scheduler_bus.prepare(tmp_path)
    claim = SimpleNamespace(token="launch", owner_path=tmp_path / "owner.json")
    source = SimpleNamespace(command=lambda command, *, site: list(command))
    endpoint = scheduler_client.SchedulerEndpoint(
        tmp_path,
        tmp_path / "BIDS",
        source,
        tmp_path / "site.toml",
        Path("/usr/bin/python3"),
        maintenance=True,
    )
    calls = {}
    monkeypatch.setattr(scheduler_bus, "claim_launch", lambda _control: claim)
    monkeypatch.setattr(
        scheduler_bus, "update_launch_job", lambda _claim, job: calls.setdefault("job", job)
    )
    process = SimpleNamespace(
        pid=123,
        returncode=0,
        communicate=lambda timeout=None: ("", ""),
    )
    monkeypatch.setattr(
        scheduler_client.subprocess,
        "Popen",
        lambda command, **kwargs: calls.update(command=command, kwargs=kwargs) or process,
    )
    monkeypatch.setattr(
        scheduler_bus,
        "write_controller_script",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("maintenance submitted a Slurm controller")
        ),
    )

    assert scheduler_client._run_once(endpoint)
    assert calls["job"] == "local-123"
    assert "--once" in calls["command"]
    assert calls["kwargs"]["env"]["NRO_PROCESS_ROLE"] == "scheduler"
    assert calls["kwargs"]["env"]["NRO_SCHEDULER_MAINTENANCE"] == "1"


def test_maintenance_command_marks_its_scheduler_endpoint(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_client, scheduler_implementation

    control = tmp_path / "control"
    checkout = tmp_path / "checkout"
    scheduler_implementation.implementation_path(control).parent.mkdir(parents=True)
    scheduler_implementation.implementation_path(control).write_text("{}")
    captured = (object(), tmp_path / "site.toml", Path("/usr/bin/python3"))
    monkeypatch.setattr(
        scheduler_implementation,
        "capture_maintenance_implementation",
        lambda *_args: captured,
    )

    endpoint = scheduler_client.command(
        control,
        tmp_path / "BIDS",
        maintenance_checkout=checkout,
    )

    assert endpoint.maintenance


def test_regular_one_shot_coordinator_clears_maintenance_identity(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client

    scheduler_bus.prepare(tmp_path)
    claim = SimpleNamespace(token="launch", owner_path=tmp_path / "owner.json")
    source = SimpleNamespace(command=lambda command, *, site: list(command))
    endpoint = scheduler_client.SchedulerEndpoint(
        tmp_path,
        tmp_path / "BIDS",
        source,
        tmp_path / "site.toml",
        Path("/usr/bin/python3"),
    )
    calls = {}
    monkeypatch.setenv("NRO_SCHEDULER_MAINTENANCE", "1")
    monkeypatch.setattr(scheduler_bus, "claim_launch", lambda _control: claim)
    monkeypatch.setattr(scheduler_bus, "update_launch_job", lambda *_args: None)
    process = SimpleNamespace(
        pid=123,
        returncode=0,
        communicate=lambda timeout=None: ("", ""),
    )
    monkeypatch.setattr(
        scheduler_client.subprocess,
        "Popen",
        lambda command, **kwargs: calls.update(command=command, kwargs=kwargs) or process,
    )

    assert scheduler_client._run_once(endpoint)
    assert "NRO_SCHEDULER_MAINTENANCE" not in calls["kwargs"]["env"]


def test_direct_exchange_uses_live_endpoint_without_launch(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client, scheduler_rpc

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    record = {"id": "abc", "payload": {"operation": "status"}}
    active = {
        "protocol": 1,
        "token": "token",
        "generation": 2,
        "host": "node.example",
        "port": 41000,
    }
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: record)
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "read_launch", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: active)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_rpc, "request", lambda got, sent, **_kwargs: {"result": sent})

    assert scheduler_client.exchange(endpoint, {"operation": "status"}) == record


def test_direct_exchange_recovers_committed_response_after_connection_loss(
    monkeypatch, tmp_path
) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client, scheduler_rpc

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    active = {
        "protocol": 1,
        "token": "token",
        "generation": 2,
        "host": "node.example",
        "port": 41000,
    }
    responses = iter((None, {"result": {"committed": True}}))
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: next(responses))
    monkeypatch.setattr(scheduler_bus, "read_launch", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: active)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler_rpc,
        "request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConnectionError("lost")),
    )
    monkeypatch.setattr(scheduler_client.time, "sleep", lambda _seconds: None)

    assert scheduler_client.exchange(endpoint, {"operation": "status"}) == {"committed": True}


def test_durable_purge_waits_for_live_service_to_process_journal(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_bus, scheduler_client, scheduler_rpc

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    active = {
        "protocol": 1,
        "token": "token",
        "generation": 2,
        "host": "node.example",
        "port": 41000,
    }
    responses = iter((None, {"result": {"removed": 12}}))
    monkeypatch.setattr(scheduler_bus, "publish_message", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr(scheduler_bus, "consume_message", lambda *_args: {"id": "abc"})
    monkeypatch.setattr(scheduler_bus, "read_response", lambda *_args: next(responses))
    monkeypatch.setattr(scheduler_bus, "message_path", lambda *_args: tmp_path / "absent")
    monkeypatch.setattr(scheduler_bus, "read_launch", lambda *_args: None)
    monkeypatch.setattr(scheduler_bus, "read_active", lambda *_args: active)
    monkeypatch.setattr(scheduler_client, "_ensure_coordinator", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler_client.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        scheduler_rpc,
        "request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a durable purge bypassed the scheduler journal")
        ),
    )

    result = scheduler_client.exchange(endpoint, {"operation": "purge"})

    assert result == {"removed": 12}


def test_supply_skips_service_when_preflight_finds_no_capacity(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    calls = []
    monkeypatch.setattr(scheduler_client, "_endpoint", lambda *_args: endpoint)
    monkeypatch.setattr(
        scheduler_client,
        "exchange",
        lambda _endpoint, message, **options: (
            calls.append((message["operation"], options)) or {"needed": False}
        ),
    )

    result = scheduler_client.supply(
        tmp_path,
        tmp_path,
        checkout=tmp_path,
        request_ids=["request"],
        options={"no_submit": False, "local": False},
    )

    assert result == {"submitted_workers": []}
    assert calls == [("supply_needed", {})]


def test_no_submit_supply_never_requires_service(monkeypatch, tmp_path) -> None:
    from nro.orchestration import scheduler_client

    endpoint = scheduler_client.SchedulerEndpoint(tmp_path, tmp_path, object(), tmp_path, tmp_path)
    calls = []
    monkeypatch.setattr(scheduler_client, "_endpoint", lambda *_args: endpoint)
    monkeypatch.setattr(
        scheduler_client,
        "exchange",
        lambda _endpoint, message, **options: (
            calls.append((message["operation"], options)) or {"submitted_workers": []}
        ),
    )

    result = scheduler_client.supply(
        tmp_path,
        tmp_path,
        checkout=tmp_path,
        request_ids=["request"],
        options={"no_submit": True, "local": False},
    )

    assert result == {"submitted_workers": []}
    assert calls == [("supply", {})]


def test_resume_suppresses_new_request_defaults() -> None:
    args = build_parser().parse_args(["--resume"])
    selection = core_selection(args, apply_planner_defaults=False)

    assert selection.modules == ()
    assert selection.workflows == ()
    assert selection.spaces == ()
    assert selection.smoothing == ()


def test_resume_uses_active_or_latest_workflow_unless_user_selects_one() -> None:
    row = {"workflow_ids": "main,oslom", "resume_workflow_ids": "oslom"}

    assert _resume_workflows(row) == ("oslom",)
    assert _resume_workflows(row, requested={"main"}) == ("main",)


def test_resume_uses_demand_only_for_incomplete_artifact_states() -> None:
    selection = CoreSelection((), (), (), (), {}, (), ())
    rows = [
        {
            "id": index,
            "status": status,
            "demanded": demanded,
            "project": "demo",
            "participant": "01",
            "module": "clean",
            "workflow_ids": "main",
            "entities_json": "{}",
        }
        for index, (status, demanded) in enumerate(
            (
                ("Queued", False),
                ("Waiting", False),
                ("Stopped", False),
                ("Error", False),
                ("Missing", False),
                ("Stale", False),
                ("Blocked", False),
                ("Missing", True),
                ("Stale", True),
                ("Corrupt", True),
                ("Blocked", True),
                ("Success", True),
                ("Running", True),
            )
        )
    ]

    selected = _resumable_rows(rows, selection)

    assert [(row["status"], bool(row["demanded"])) for row in selected] == [
        ("Queued", False),
        ("Waiting", False),
        ("Stopped", False),
        ("Error", False),
        ("Missing", True),
        ("Stale", True),
        ("Corrupt", True),
        ("Blocked", True),
    ]


def test_resume_reconstructs_dependencies_from_requested_terminals() -> None:
    selection = CoreSelection((), (), (), (), {}, (), ())
    common = {
        "status": "Queued",
        "demanded": True,
        "project": "demo",
        "participant": "01",
        "workflow_ids": "main",
        "entities_json": "{}",
    }

    selected = _resumable_rows(
        [
            {**common, "id": 1, "module": "anat", "resume_workflow_ids": ""},
            {**common, "id": 2, "module": "networks", "resume_workflow_ids": "main"},
        ],
        selection,
    )

    assert [row["module"] for row in selected] == ["networks"]


def test_resume_collapses_old_scheduler_dependency_rows_to_endpoints() -> None:
    rows = [
        {"id": 1, "module": "anat"},
        {"id": 2, "module": "func"},
        {"id": 3, "module": "clean"},
        {"id": 4, "module": "networks"},
    ]

    selected = _terminal_resume_rows(rows, [(2, 1), (3, 2), (4, 3)])

    assert [row["module"] for row in selected] == ["networks"]


def test_run_can_disable_ancestor_reuse() -> None:
    assert build_parser().parse_args(["--no-inherit"]).no_inherit


def test_run_default_targets_follow_catalog(monkeypatch) -> None:
    from dataclasses import replace

    from nro.orchestration import catalog

    consumer = replace(
        catalog.module_descriptor("firstlevels"),
        name="consumer",
        upstream_modules=("firstlevels",),
    )
    monkeypatch.setitem(catalog.MODULE_CATALOG, consumer.name, consumer)
    selection = core_selection(build_parser().parse_args([]))
    assert selection.modules == ("dynconn", "networks", "consumer")
    explicit = core_selection(build_parser().parse_args(["-m", "func"]))
    assert explicit.modules == ("func",)


def test_shared_selection_options_accept_multiple_values() -> None:
    args = build_parser().parse_args(
        [
            "-p",
            "sub-01",
            "02",
            "-P",
            "alpha",
            "beta",
            "-m",
            "clean",
            "networks",
            "-w",
            "main",
            "experiment",
            "-i",
            "networks/main-2",
            "-r",
            "task=language,spatial",
            "dir=LR",
            "-s",
            "fsnative",
            "ACPC",
            "-S",
            "0",
            "2",
        ]
    )

    selection = core_selection(args)
    assert selection.participants == ("01", "02")
    assert selection.projects == ("alpha", "beta")
    assert selection.modules == ("clean", "networks")
    assert selection.workflows == ("main", "experiment")
    assert selection.lineages == ("networks/main-2",)
    assert selection.runs == {"task": ("language", "spatial"), "dir": ("LR",)}
    assert selection.spaces == ("fsnative", "ACPC")
    assert selection.smoothing == (0, 2)


def test_module_specific_selectors_can_accompany_mixed_module_requests() -> None:
    args = build_parser().parse_args(
        ["-m", "anat", "networks", "-r", "task=rest", "-s", "ACPC", "-S", "0"]
    )
    selection = core_selection(args)

    assert selection.modules == ("anat", "networks")
    assert selection.runs == {"task": ("rest",)}
    assert selection.spaces == ("ACPC",)
    assert selection.smoothing == (0,)


def test_stop_workers_short_flag_and_long_workflow_option() -> None:
    worker_args = stop_parser().parse_args(["-W"])
    assert worker_args.workers is True
    assert worker_args.workflow is None
    workflow_args = stop_parser().parse_args(["-w", "experiment"])
    assert workflow_args.workers is False
    assert workflow_args.workflow == ["experiment"]


def test_status_report_pages_only_interactive_output(monkeypatch, capsys) -> None:
    report = _render_report([])
    assert report == (
        f"{'PROJECT':14} {'PARTICIPANT':14} {'MODULE':20} {'LINEAGE':20} "
        f"{'STATUS':12} {'MEM':8} ENTITIES\n"
    )
    page_text(report, use_pager=False)
    assert capsys.readouterr().out == report

    calls = []

    class Interactive:
        def isatty(self):
            return True

        def write(self, _value):
            raise AssertionError("interactive status should use the pager")

    monkeypatch.setattr("nro.engine.cli.sys.stdout", Interactive())
    monkeypatch.setattr("nro.engine.cli.shutil.which", lambda _: "/usr/bin/less")
    monkeypatch.setattr(
        "nro.engine.cli.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    page_text(report, use_pager=True)
    assert calls == [(["/usr/bin/less", "-R"], {"input": report, "text": True, "check": False})]


@pytest.mark.parametrize(
    ("help_text", "expected_command"),
    [
        (
            "  --header N    keep N header lines",
            ["/usr/bin/less", "-R", "--header", "1"],
        ),
        ("classic less help", ["/usr/bin/less", "-R"]),
    ],
)
def test_status_pager_uses_sticky_header_only_when_less_supports_it(
    monkeypatch,
    help_text: str,
    expected_command: list[str],
) -> None:
    calls = []

    class Interactive:
        def isatty(self):
            return True

        def write(self, _value):
            raise AssertionError("interactive status should use the pager")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command == ["/usr/bin/less", "--help"]:
            return subprocess.CompletedProcess(command, 0, help_text, "")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("nro.engine.cli.sys.stdout", Interactive())
    monkeypatch.setattr("nro.engine.cli.shutil.which", lambda _: "/usr/bin/less")
    monkeypatch.setattr("nro.engine.cli.subprocess.run", run)

    page_text("PROJECT\n", use_pager=True, header_lines=1)

    assert calls == [
        (
            ["/usr/bin/less", "--help"],
            {"text": True, "capture_output": True, "check": False},
        ),
        (
            expected_command,
            {"input": "PROJECT\n", "text": True, "check": False},
        ),
    ]


def test_status_report_colors_statuses_without_changing_column_width() -> None:
    report = _render_report(
        [
            {
                "project": "demo",
                "participant": "01",
                "module": "anat",
                "lineage": "main",
                "route": [{"module": "anat", "config": "main"}],
                "workflows": ["main"],
                "status": "Success",
                "memory_gb": 32,
                "entities": {},
            },
            {
                "project": "demo",
                "participant": "02",
                "module": "func",
                "lineage": "main",
                "status": "Error",
                "memory_gb": 32,
                "entities": {"task": "rest"},
            },
        ],
        color=True,
        show_lineages=True,
    )

    assert "\x1b[1m\x1b[96mPROJECT" in report
    assert "\x1b[92mSuccess     \x1b[0m" in report
    assert "\x1b[91m\x1b[1mError       \x1b[0m" in report
    assert "\x1b[2mtask=rest\x1b[0m" in report
    assert "Route: anat=main" in report
    assert "Workflows: main" in report


def test_status_work_item_details_are_opt_in() -> None:
    rows = [
        {
            "project": "demo",
            "participant": "01",
            "module": "anat",
            "lineage": "main",
            "route": [{"module": "anat", "config": "main"}],
            "workflows": ["main"],
            "status": "Success",
            "memory_gb": 32,
            "entities": {},
        }
    ]

    assert "\nModule lineages\n" not in _render_report(rows)
    assert "\nModule lineages\n" in _render_report(rows, show_lineages=True)


def test_status_marks_downstream_failure_as_blocked_and_summarizes_root() -> None:
    report = _render_report(
        [
            {
                "project": "demo",
                "participant": "01",
                "module": "clean",
                "lineage": "main",
                "status": "Blocked",
                "memory_gb": 32,
                "entities": {},
                "blocked_by": ("sub-01 func",),
            }
        ],
        errors=[
            {
                "project": "demo",
                "participant": "01",
                "module": "func",
                "entities": "run=01",
                "step": "Registration",
                "message": "bad transform",
                "log": "/tmp/work-item.log",
                "blocked_work_items": ["demo sub-01 clean"],
            }
        ],
        blocked_work_items=[
            {
                "project": "demo",
                "participant": "01",
                "module": "clean",
                "entities": "",
                "upstream_errors": ["demo sub-01 func (run=01)"],
            }
        ],
    )
    assert "Errors" in report
    assert "Blocked work items" in report
    assert "Failed step: Registration" in report


def test_status_without_a_registry_prints_an_empty_table(tmp_path: Path, capsys) -> None:
    status_main(["--no-pager"])

    output = capsys.readouterr().out
    assert output.startswith("PROJECT")
    assert len(output.splitlines()) == 1


def test_branch_status_discovers_projects_from_its_single_scheduler_response(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import nro.bin.status as status_command
    import nro.configuration.site as site
    import nro.orchestration.scheduler_client as scheduler_client

    bids = (tmp_path / "bids").resolve()
    control = (tmp_path / "control").resolve()
    checkout = (tmp_path / "checkout").resolve()
    calls = []
    row = {
        "id": 1,
        "project": "demo",
        "participant": "01",
        "module": "anat",
        "entities_json": "{}",
        "workflow_ids": "main",
        "root_failure_ids": (),
        "status": "Error",
        "error_message": "direct failure",
        "log_path": None,
        "artifact_reason": "Current",
        "current_generation": 1,
        "memory_gb": 2,
    }
    second_row = {
        **row,
        "id": 2,
        "participant": "02",
        "error_message": "second direct failure",
    }

    monkeypatch.setattr(site, "CHECKOUT", checkout)
    monkeypatch.setattr(
        site, "settings", lambda: ({"registry": str(control), "bids": str(bids)}, None)
    )
    monkeypatch.setattr(site, "installation_record", lambda: {"mode": "branch"})
    monkeypatch.setattr(
        status_command,
        "selected_projects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("branch status must not make a discovery request")
        ),
    )

    def scheduler_status(*_args, **kwargs):
        calls.append(kwargs["mode"])
        return {
            "rows": [row, second_row],
            "visible_ids": [1, 2],
            "dependencies": [],
            "ingestion": [],
        }

    monkeypatch.setattr(scheduler_client, "status", scheduler_status)

    status_main(["--json"])

    report = json.loads(capsys.readouterr().out)
    assert calls == ["cached"]
    assert [item["project"] for item in report["work_items"]] == ["demo", "demo"]
    assert [item["message"] for item in report["errors"]] == [
        "direct failure",
        "second direct failure",
    ]


def test_run_repair_rebuilds_registry_and_discovers_source_tree(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    obsolete = registry.paths.database.parent / "obsolete-control-state"
    obsolete.write_text("old")
    with sqlite3.connect(registry.paths.database) as connection:
        connection.execute("PRAGMA user_version=10")

    run_main(["--repair", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["repaired"] is True
    assert result["projects"] == ["demo"]
    assert result["participants"] == 1
    assert result["work_items"] == 0
    assert result["requests"] == []
    assert result["submitted_workers"] == []
    assert not obsolete.exists()
    assert registry.work_item_rows() == []
    assert registry.request_rows() == []
    with registry.connection() as connection:
        assert tuple(
            connection.execute("SELECT project, participant FROM bids_participants").fetchone()
        ) == ("demo", "01")
    with sqlite3.connect(registry.paths.database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    status_main(["--no-pager"])
    status_output = capsys.readouterr().out
    assert status_output.startswith("PROJECT")
    assert len(status_output.splitlines()) == 1


def test_main_repair_uses_lab_wide_scheduler_during_release_transition(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import os

    from nro.configuration import site

    binding = tmp_path / "implementation.json"
    binding.write_text("{}")
    monkeypatch.setattr(
        site,
        "installation_record",
        lambda: {
            "mode": "shared",
            "ready": True,
            "site": os.environ["NRO_SITE_CONFIG"],
        },
    )
    monkeypatch.setattr(
        "nro.orchestration.scheduler_implementation.implementation_path",
        lambda _control: binding,
    )
    calls = []
    monkeypatch.setattr(
        "nro.orchestration.scheduler_repair.repair",
        lambda registry, **options: (
            calls.append((registry, options))
            or {
                "repaired": True,
                "registry": str(registry.paths.database),
                "backup": str(tmp_path / "backup"),
                "work_items": 0,
                "unavailable": [],
                "schema": SCHEMA_VERSION,
            }
        ),
    )

    run_main(["--repair", "--json"])

    assert len(calls) == 1
    assert calls[0][1]["checkout"] == site.CHECKOUT
    assert calls[0][1]["allow_release_transition"] is True
    assert json.loads(capsys.readouterr().out)["repaired"] is True


def test_run_repair_registers_existing_artifacts_without_demand(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    bids = tmp_path / "bids"
    monkeypatch.setattr("nro.engine.paths.BIDS_PATH", bids)
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(ConfigStore().resolve("main"))
    artifact = (
        bids
        / "demo"
        / "derivatives"
        / "nro"
        / "anat"
        / registered.directories["anat"]
        / "sub-01"
        / "anat"
        / "sub-01_desc-preprocessAnat_manifest.json"
    )
    _write(
        artifact,
        json.dumps(
            {
                "complete": True,
                "gradient_unwarping": {},
                "output_metadata_contract": module_descriptor("anat").processing_contract()[
                    "output_metadata"
                ],
            }
        ),
    )

    run_main(["--repair", "--json"])
    result = json.loads(capsys.readouterr().out)
    registry = Registry.for_project("demo", bids_root=bids)
    rows = registry.work_item_rows()

    assert result["artifacts"] == 1
    assert result["work_items"] == 1
    assert [(row["module"], row["participant"]) for row in rows] == [("anat", "01")]
    assert rows[0]["artifact_state"] == "fresh"
    assert rows[0]["demanded"] == 0
    assert registry.request_rows() == []


def test_repair_registers_only_existing_artifacts_and_their_dependencies(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(ConfigStore().resolve("main"))
    _write(
        bids
        / "demo"
        / "derivatives"
        / "nro"
        / "clean"
        / registered.directories["clean"]
        / "sub-01"
        / ("sub-01_task-rest_run-1_space-fsnative_smoothing-2mm_desc-clean_manifest.json"),
        json.dumps({"complete": True}),
    )

    run_main(["--repair", "--json"])
    result = json.loads(capsys.readouterr().out)
    registry = Registry.for_project("demo", bids_root=bids)

    assert result["artifacts"] == 1
    assert result["work_items"] == 3
    assert {row["module"] for row in registry.work_item_rows()} == {
        "anat",
        "func",
        "clean",
    }
    assert registry.request_rows() == []


def test_run_repair_requires_confirmation_before_stopping_active_workers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    registry.register_worker("active-worker", resource_class="large")
    called = False

    def unexpected_shutdown(_registry):
        nonlocal called
        called = True

    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    monkeypatch.setattr("nro.bin.run.stop_worker_pool_for_repair", unexpected_shutdown)

    with pytest.raises(SystemExit, match="repair cancelled"):
        run_main(
            [
                "--repair",
            ]
        )

    assert called is False
    assert registry.paths.database.is_file()
    assert registry.worker_pool_activity()["workers"][0]["id"] == "active-worker"


def test_run_repair_confirms_and_stops_active_workers_before_rebuild(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    registry.register_worker("active-worker", resource_class="large")
    calls: list[Registry] = []

    def shutdown(shared_registry: Registry) -> dict:
        calls.append(shared_registry)
        return {"cancellation_failures": []}

    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    monkeypatch.setattr("nro.bin.run.stop_worker_pool_for_repair", shutdown)

    run_main(["--repair", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert len(calls) == 1
    assert result["repaired"] is True
    assert registry.worker_pool_activity()["workers"] == []


def test_run_repair_discovers_participant_without_planning_anatomy(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-02"
    _write(subject / "func" / "sub-02_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-02_task-rest_run-1_bold.json", "{}")

    run_main(["--repair", "--json"])
    result = json.loads(capsys.readouterr().out)
    registry = Registry.for_project("demo", bids_root=bids)

    assert result["repaired"] is True
    assert result["projects"] == ["demo"]
    assert result["participants"] == 1
    assert result["work_items"] == 0
    assert registry.paths.database.is_file()
    assert registry.work_item_rows() == []


def test_run_repair_rejects_derivative_selection(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="all projects.*--project"):
        run_main(
            [
                "--repair",
                "-P",
                "demo",
            ]
        )


def test_run_reports_when_only_selected_participant_is_unavailable(
    tmp_path: Path,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-02"
    _write(subject / "func" / "sub-02_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-02_task-rest_run-1_bold.json", "{}")

    with pytest.raises(SystemExit, match="No requested work is available") as error:
        run_main(
            [
                "-p",
                "02",
                "-P",
                "demo",
                "--no-submit",
            ]
        )

    assert "No T1w or T2w images found" in str(error.value)


def test_run_continues_past_unavailable_participant(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    available = bids / "demo" / "sub-01"
    _write(available / "anat" / "sub-01_T1w.nii.gz")
    _write(available / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(available / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    unavailable = bids / "demo" / "sub-02"
    _write(unavailable / "func" / "sub-02_task-rest_run-1_bold.nii.gz")
    _write(unavailable / "func" / "sub-02_task-rest_run-1_bold.json", "{}")

    run_main(
        [
            "-P",
            "demo",
            "--no-submit",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert result["participants"] == {"demo": ["01"]}
    assert result["work_items"] == 6
    assert {
        "project": "demo",
        "participant": "01",
        "module": "firstlevels",
        "reason": f"No raw BOLD runs matched under {available}",
    } in result["unavailable"]
    assert [item for item in result["unavailable"] if item["module"] == "networks"] == [
        {
            "project": "demo",
            "participant": "02",
            "module": "networks",
            "reason": f"No T1w or T2w images found under {unavailable}",
        }
    ]


def test_run_resume_recreates_only_matching_registered_demand(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    for participant in ("01", "02"):
        subject = bids / "demo" / f"sub-{participant}"
        _write(subject / "anat" / f"sub-{participant}_T1w.nii.gz")
        stem = f"sub-{participant}_task-rest_run-1_bold"
        _write(subject / "func" / f"{stem}.nii.gz")
        _write(subject / "func" / f"{stem}.json", "{}")

    run_main(["-P", "demo", "-m", "func", "--no-submit", "--json"])
    capsys.readouterr()
    run_main(["--resume", "-P", "demo", "-p", "01", "--no-submit", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["resumed"] == 2
    assert result["participants"] == {"demo": ["01"]}
    assert result["work_items"] == 2
    registry = Registry.for_project("demo", bids_root=bids)
    requests = registry.request_rows()
    assert len(requests) == 2
    assert json.loads(requests[-1]["selectors_json"])["participants"] == ["01"]


def test_run_resume_does_not_plan_cartesian_product_of_registered_work(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    first = bids / "demo" / "sub-01"
    _write(first / "anat" / "sub-01_T1w.nii.gz")
    _write(first / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(first / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    second = bids / "demo" / "sub-02"
    _write(second / "anat" / "sub-02_T1w.nii.gz")

    run_main(["-P", "demo", "-p", "01", "-m", "networks", "--no-submit", "--json"])
    capsys.readouterr()
    run_main(["-P", "demo", "-p", "02", "-m", "anat", "--no-submit", "--json"])
    capsys.readouterr()

    run_main(["--resume", "-P", "demo", "--no-submit", "--json"])
    result = json.loads(capsys.readouterr().out)

    assert result["unavailable"] == []
    assert set(result["participants"]["demo"]) == {"01", "02"}


def test_clean_request_expands_all_matching_runs(tmp_path: Path, capsys) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-20"
    _write(subject / "anat" / "sub-20_T1w.nii.gz")
    for run in ("1", "2"):
        stem = f"sub-20_task-rest_run-{run}_bold"
        _write(subject / "func" / f"{stem}.nii.gz")
        _write(subject / "func" / f"{stem}.json", "{}")

    run_main(["-p", "20", "-P", "demo", "-m", "clean", "--no-submit", "--json"])
    request = json.loads(capsys.readouterr().out)
    assert request["modules"] == ["clean"]
    assert request["work_items"] == 5  # one anat plus func and clean for both runs
    assert request["concurrency"] == 50


def test_bare_request_keeps_networks_when_no_task_models_match(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    project = bids / "climblab_multisession"
    for participant in ("01", "02"):
        subject = project / f"sub-{participant}"
        _write(subject / "anat" / f"sub-{participant}_T1w.nii.gz")
        stem = f"sub-{participant}_task-rest_run-1_bold"
        _write(subject / "func" / f"{stem}.nii.gz")
        _write(subject / "func" / f"{stem}.json", "{}")

    run_main(["--no-submit", "--json"])
    request = json.loads(capsys.readouterr().out)
    assert request["projects"] == ["climblab_multisession"]
    assert request["participants"] == {"climblab_multisession": ["01", "02"]}
    assert request["modules"] == ["dynconn", "networks", "firstlevels"]
    assert request["work_items"] == 12
    assert request["concurrency"] == 50


@pytest.mark.parametrize("modules", [(), ("networks",), ("firstlevels",)])
def test_run_requests_each_selected_branch_across_projects(
    tmp_path: Path,
    capsys,
    monkeypatch,
    modules,
) -> None:
    from nro.modules.firstlevels import task_models

    models = tmp_path / "config" / "models" / "langlocSN"
    definition = "conditions: trial_type\ncontrasts:\n  S: {S: 1}\n"
    _write(models / "main.yml", "model_set: main\n" + definition)
    _write(models / "development.yml", "model_set: development\n" + definition)
    monkeypatch.setattr(task_models, "definitions_root", lambda: tmp_path / "config")
    bids = tmp_path / "bids"
    for project in ("alpha", "beta"):
        for participant, task in (("01", "langlocSN"), ("02", "rest")):
            subject = bids / project / f"sub-{participant}"
            _write(subject / "anat" / f"sub-{participant}_T1w.nii.gz")
            stem = f"sub-{participant}_task-{task}_run-1"
            _write(subject / "func" / f"{stem}_bold.nii.gz")
            _write(subject / "func" / f"{stem}_bold.json", "{}")
            _write(
                subject / "func" / f"{stem}_events.tsv", "onset\tduration\ttrial_type\n0\t1\tS\n"
            )

    argv = ["--no-submit", "--json"]
    if modules:
        argv.extend(["-m", *modules])
    run_main(argv)
    result = json.loads(capsys.readouterr().out)
    selected = modules or ("dynconn", "networks", "firstlevels")
    assert result["modules"] == list(selected)
    registry = Registry.for_project("alpha", bids_root=bids)
    rows = registry.work_item_rows()
    assert {row["project"] for row in rows} == {"alpha", "beta"}
    assert all(row["demanded"] for row in rows)
    for project in ("alpha", "beta"):
        project_rows = [row for row in rows if row["project"] == project]
        # Shared anatomy and functional work_items must not be registered twice.
        subjects = 2 if "networks" in selected else 1
        for module in ("anat", "func"):
            assert sum(row["module"] == module for row in project_rows) == subjects
        fits = [row for row in project_rows if row["module"] == "firstlevels"]
        assert len(fits) == int("firstlevels" in selected)
        if fits:
            assert json.loads(fits[0]["entities_json"])["model"] == "main"
        assert sum(row["module"] == "networks" for row in project_rows) == (
            2 if "networks" in selected else 0
        )
    assert len(registry.request_rows()) == len(selected) * 2


def test_participant_selection_spans_every_matching_project(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    for project in ("alpha", "beta"):
        subject = bids / project / "sub-01"
        _write(subject / "anat" / "sub-01_T1w.nii.gz")
        _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
        _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    unmatched = bids / "beta" / "sub-02"
    _write(unmatched / "anat" / "sub-02_T1w.nii.gz")

    run_main(
        [
            "-p",
            "01",
            "-m",
            "func",
            "--no-submit",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    registry = Registry.for_project("alpha", bids_root=bids)

    assert result["projects"] == ["alpha", "beta"]
    assert result["participants"] == {"alpha": ["01"], "beta": ["01"]}
    assert len(result["requests"]) == 2
    assert {row["project"] for row in registry.request_rows()} == {"alpha", "beta"}
    assert {row["project"] for row in registry.work_item_rows()} == {"alpha", "beta"}


def test_run_status_stop_roundtrip_without_submission(tmp_path: Path, capsys) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")

    run_main(["-p", "01", "-P", "demo", "-m", "func", "--no-submit", "--json"])
    request = json.loads(capsys.readouterr().out)
    assert request["work_items"] == 2
    assert request["submitted_workers"] == []

    status_main(["-p", "01", "-P", "demo", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"work_items", "errors", "blocked_work_items", "bidsification"}
    assert {row["status"] for row in report["work_items"]} == {"Queued", "Waiting"}
    assert report["errors"] == []
    assert report["blocked_work_items"] == []

    status_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-w",
            "main",
            "-r",
            "task=rest",
            "run=1",
            "--json",
        ]
    )
    selected = json.loads(capsys.readouterr().out)
    assert len(selected["work_items"]) == 1
    assert selected["work_items"][0]["module"] == "func"
    assert selected["work_items"][0]["workflows"] == ["main"]

    stop_main(
        [
            "-p",
            "01",
            "-m",
            "anat",
            "-P",
            "demo",
        ]
    )
    assert "Cancelled" in capsys.readouterr().out
    for mode in ([], ["--update"], []):
        status_main(["-p", "01", "-P", "demo", "--json", *mode])
        rows = json.loads(capsys.readouterr().out)["work_items"]
        assert {row["status"] for row in rows} == {"Missing", "Stale"}
        assert all(row["reason"] for row in rows)


def test_set_updates_active_concurrency_without_creating_new_demand(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    run_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "anat",
            "--concurrency",
            "2",
            "--no-submit",
            "--json",
        ]
    )
    capsys.readouterr()
    registry = Registry.for_project("demo", bids_root=bids)
    original_updated_at = registry.request_rows()[0]["updated_at"]

    set_main(
        [
            "unsupported=value",
            "concurrency=7",
            "another=setting",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    warnings = captured.err
    request = registry.request_rows()[0]

    assert result == {
        "settings": {"concurrency": 7},
        "updated_requests": 1,
    }
    assert request["concurrency"] == 7
    assert request["updated_at"] == original_updated_at
    assert len(registry.request_rows()) == 1
    assert "unsupported registry setting: unsupported" in warnings
    assert "unsupported registry setting: another" in warnings


def test_set_ignores_invocation_with_only_unsupported_settings(capsys) -> None:
    set_main(["future-setting=value", "--json"])
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {"settings": {}, "updated_requests": 0}
    assert "unsupported registry setting: future-setting" in captured.err


def test_status_reports_blocked_work_items_and_their_root_errors(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    run_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "func",
            "--no-submit",
            "--json",
        ]
    )
    capsys.readouterr()

    registry = Registry.for_project("demo", bids_root=bids)
    registry.register_worker("failed-worker", resource_class="large")
    claimed = registry.claim_ready_work_item("failed-worker", ("large",))
    assert claimed is not None
    assert claimed.module == "anat"
    registry.finish_attempt(
        claimed.attempt_id,
        state="error",
        error_type="RuntimeError",
        error_message="anatomical failure",
    )

    status_main(["-p", "01", "-P", "demo", "--json"])
    report = json.loads(capsys.readouterr().out)
    statuses = {row["module"]: row["status"] for row in report["work_items"]}

    assert statuses == {"anat": "Error", "func": "Blocked"}
    assert len(report["errors"]) == 1
    directories = {row["module"]: row["directory_label"] for row in registry.work_item_rows()}
    assert report["errors"][0]["blocked_work_items"] == [
        f"demo sub-01 func/{directories['func']} (run=1 task=rest)"
    ]
    assert len(report["blocked_work_items"]) == 1
    assert report["blocked_work_items"][0]["upstream_errors"] == [
        f"demo sub-01 anat/{directories['anat']}"
    ]


def test_status_is_strictly_read_only(tmp_path: Path, capsys) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.nii.gz")
    _write(subject / "func" / "sub-01_task-rest_run-1_bold.json", "{}")
    run_main(["-p", "01", "-P", "demo", "-m", "func", "--no-submit", "--json"])
    capsys.readouterr()
    registry = Registry.for_project("demo", bids_root=bids)
    database_mtime = registry.paths.database.stat().st_mtime_ns
    control_mtime = registry.paths.control.stat().st_mtime_ns

    status_main(["-p", "01", "-P", "demo", "--json"])
    capsys.readouterr()

    assert registry.paths.database.stat().st_mtime_ns == database_mtime
    assert registry.paths.control.stat().st_mtime_ns == control_mtime


def test_status_default_is_cached_and_update_persists_assessment(
    tmp_path: Path,
    capsys,
) -> None:
    bids = tmp_path / "bids"
    subject = bids / "demo" / "sub-01"
    _write(subject / "anat" / "sub-01_T1w.nii.gz")
    run_main(
        [
            "-p",
            "01",
            "-P",
            "demo",
            "-m",
            "anat",
            "--no-submit",
            "--json",
        ]
    )
    capsys.readouterr()
    registry = Registry.for_project("demo", bids_root=bids)
    row = registry.work_item_rows()[0]
    with registry.connection(write=True) as db:
        db.execute(
            "UPDATE work_items SET artifact_state='fresh', artifact_reason='Cached success' "
            "WHERE id=?",
            (row["id"],),
        )

    status_main(["-P", "demo", "--json"])
    cached = json.loads(capsys.readouterr().out)["work_items"]
    assert cached[0]["status"] == "Success"
    assert registry.work_item_rows()[0]["artifact_state"] == "fresh"

    status_main(["-P", "demo", "--update", "--json"])
    capsys.readouterr()
    assert registry.work_item_rows()[0]["artifact_state"] == "missing"


def test_worker_script_records_memory_tier(tmp_path: Path) -> None:
    bids = tmp_path / "bids"
    registry = Registry.for_project("demo", bids_root=bids)
    registry.initialize()
    script = _write_worker_script(
        registry,
        bids_root=bids,
        partition="sphinx",
        account="nlp",
        hours=24,
        memory_gb=64,
        cpus=8,
    )
    text = script.read_text()
    assert script.name.startswith("worker-large-64gb-")
    assert script.name.endswith(".sbatch")
    assert "#SBATCH --mem=64G" in text
    assert "--memory-gb 64" in text
    assert "--idle-timeout 30" in text
    assert "--walltime-seconds 86400" in text
    assert "--drain-seconds 900" in text
    assert "source_launcher.py" in text
    assert "--profile" in text
    assert f"cd {Path(__file__).resolve().parents[1]}" not in text

    other = _write_worker_script(
        registry,
        bids_root=bids,
        partition="john",
        account="nlp",
        hours=24,
        memory_gb=64,
        cpus=8,
    )
    assert other != script
    assert "#SBATCH --partition=sphinx" in script.read_text()
    assert "#SBATCH --partition=john" in other.read_text()
