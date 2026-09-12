"""Changing a checkout's role preserves its environment and shared scientific state."""

import json
import sqlite3
from pathlib import Path

import pytest

from nro.configuration import site
from nro.engine import bootstrap, user_launcher
from nro.engine import installation_transition as transition
from nro.orchestration.control_paths import ControlPaths
from nro.orchestration.registry import SCHEMA_SQL


@pytest.fixture
def installations(tmp_path, monkeypatch):
    control = tmp_path / "control"
    records = []
    for name in ("old", "main"):
        root = tmp_path / name
        python = root / ".nro-env/bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"original environment")
        config = root / "site.toml"
        config.write_text(f'registry = "{control}"\n')
        record = dict(
            checkout=str(root),
            environment=str(python.parents[1]),
            site=str(config),
            ready=True,
            mode="shared",
        )
        bootstrap.write_record(root / bootstrap.RECORD, record)
        records.append(record)
    old, main = records
    root = Path(old["checkout"])
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(transition, "checkout_identity", lambda _: (root, "dev", "a" * 40))
    calls = []

    def register(command, **kwargs):
        assert "nro.bin.branch" in command
        assert kwargs["env"]["NRO_SITE_CONFIG"] == main["site"]
        current = json.loads((root / bootstrap.RECORD).read_text())
        assert current["mode"] == "branch" and not current["ready"]
        calls.append(command)
        catalog = ControlPaths(control).catalog
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(
            json.dumps({"dev": dict(registry_id="b" * 32, checkouts=[str(root)], retired=False)})
        )

    monkeypatch.setattr(transition.subprocess, "run", register)
    return old, main, control, calls


def test_conversion_preserves_environment_and_site_and_user_default(installations, tmp_path):
    old, main, control, calls = installations
    root = Path(old["checkout"])
    bin_dir = tmp_path / "bin"
    bootstrap.connect_user(main, bin_dir=bin_dir, set_default=True)
    original_site = Path(old["site"]).read_bytes()
    bootstrap.main(["--convert-to-branch", "--bin-dir", str(bin_dir)])
    current = json.loads((root / bootstrap.RECORD).read_text())
    assert current["ready"] and current["mode"] == "branch" and current["branch"] == "dev"
    assert current["site"] == main["site"] and current["environment"] == old["environment"]
    assert (Path(current["environment"]) / "bin/python").read_bytes() == b"original environment"
    assert Path(old["site"]).read_bytes() == original_site
    assert json.loads((root / ".nro-installation-transition.json").read_text())["original"] == old
    assert (
        user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)["default"] == main["checkout"]
    )
    assert not ControlPaths(control).database.exists()
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "no_default",
        "self_default",
        "changed_site",
        "main_branch",
        "active_demand",
        "active_worker",
        "active_allocation",
        "active_ingestion",
        "active_review",
    ],
)
def test_failed_preflight_preserves_original_record(installations, monkeypatch, failure):
    old, main, control, calls = installations
    root = Path(old["checkout"])
    replacement = main
    if failure == "no_default":
        replacement = None
    elif failure == "self_default":
        replacement = old
    elif failure == "changed_site":
        Path(main["site"]).write_text(f'registry = "{control}"\nwork = "/different/work"\n')
    elif failure == "main_branch":
        monkeypatch.setattr(transition, "checkout_identity", lambda _: (root, "main", "a" * 40))
    elif failure in {"active_ingestion", "active_review"}:
        path = ControlPaths(control).ingestion / (
            "request.json" if failure == "active_ingestion" else "reviews/request.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {"state": "running"} if failure == "active_ingestion" else {"expires": 99999999999}
            )
        )
    else:
        database = ControlPaths(control).database
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as db:
            db.executescript(SCHEMA_SQL)
            if failure == "active_demand":
                db.execute(
                    "INSERT INTO workflow_revisions VALUES (1,'main',1,'definition','workflow.yml','{}','now')"
                )
                db.execute(
                    "INSERT INTO requests VALUES ('r','user','demo',1,'anat','{}',1,NULL,'active','now','now')"
                )
            elif failure == "active_worker":
                db.execute(
                    "INSERT INTO workers(id,user_name,hostname,pid,resource_class,state,started_at,updated_at) VALUES ('w','user','host',1,'large','running','now','now')"
                )
            else:
                db.execute(
                    "INSERT INTO scheduler_submissions(intent_token,resource_class,state,created_at) VALUES ('token','large','submitted','now')"
                )
    before = (root / bootstrap.RECORD).read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        transition.convert_shared(root, replacement)
    assert (root / bootstrap.RECORD).read_bytes() == before
    assert not (root / ".nro-installation-transition.json").exists()
    assert calls == []


def test_interrupted_registration_stays_blocked_and_can_resume(installations, monkeypatch):
    old, main, control, calls = installations
    root = Path(old["checkout"])
    register = transition.subprocess.run

    def fail(*args, **kwargs):
        register(*args, **kwargs)
        raise OSError("interrupted registration")

    with monkeypatch.context() as patch:
        patch.setattr(transition.subprocess, "run", fail)
        with pytest.raises(OSError, match="interrupted"):
            transition.convert_shared(root, main)
    record = json.loads((root / bootstrap.RECORD).read_text())
    assert record["mode"] == "branch" and not record["ready"]
    with monkeypatch.context() as patch:
        patch.setattr(site, "installation_record", lambda: record)
        with pytest.raises(ValueError, match="not permitted"):
            site.require_execution_support()
    record = transition.convert_shared(root, main)
    assert record["ready"]
    assert calls[-1][calls[-1].index("nro.bin.branch") + 1] == "attach"


def test_incomplete_shared_installation_allows_only_explicit_maintenance(monkeypatch):
    record = {"mode": "shared", "ready": False}
    monkeypatch.setattr(site, "installation_record", lambda: record)

    with pytest.raises(ValueError, match="undergoing setup or maintenance"):
        site.require_execution_support()
    site.require_execution_support(installation_maintenance=True)


def test_journal_before_record_replacement_blocks_execution_and_resumes(installations, monkeypatch):
    old, main, _, _ = installations
    root = Path(old["checkout"])
    write = bootstrap.write_record

    def fail(path, record):
        if path == root / bootstrap.RECORD:
            raise OSError("record publication failed")
        write(path, record)

    with monkeypatch.context() as patch:
        patch.setattr(bootstrap, "write_record", fail)
        with pytest.raises(OSError):
            transition.convert_shared(root, main)
    with monkeypatch.context() as patch:
        patch.setattr(site, "CHECKOUT", root)
        patch.setattr(site, "installation_record", lambda: old)
        for operation in (
            site.site_file,
            site.require_execution_support,
            site.require_definition_write,
        ):
            with pytest.raises(ValueError, match="conversion is incomplete"):
                operation()
    assert transition.convert_shared(root, main)["ready"]
