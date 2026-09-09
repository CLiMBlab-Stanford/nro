from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.configuration import site
from nro.configuration.store import ConfigStore
from nro.engine import bootstrap, dependencies, shared_installation, site_setup
from nro.engine.site_setup import edit_settings, save_settings


@pytest.fixture
def isolated_site(tmp_path, monkeypatch):
    path = tmp_path / "site.toml"
    path.write_text("")
    monkeypatch.setenv("NRO_SITE_CONFIG", str(path))
    for key in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(site, "installation_record", lambda: {})
    return path


def test_lab_defaults_preserve_preprocessing_identity(isolated_site):
    config = ConfigStore().load_configuration("preprocessing", "main")
    assert config.fingerprint == "ba2380d9f1488ed597aba5311ec155b37cd5621f7128994b3417e17a2a56f075"


def test_resource_changes_propagate_to_all_configurations(isolated_site, tmp_path):
    save_settings(
        isolated_site,
        {
            "images": str(tmp_path / "images"),
            "runtime": "apptainer",
            "binds": [],
            "templates": str(tmp_path / "templates"),
            "license": str(tmp_path / "license"),
        },
    )
    store = ConfigStore()
    preprocessing = store.load_configuration("preprocessing", "main").values
    clean = store.load_configuration("clean", "main").values
    assert preprocessing["container"]["image"] == clean["container"]
    assert clean["container"].startswith(str(tmp_path))
    assert preprocessing["container"]["engine"] == clean["container_engine"] == "apptainer"
    assert preprocessing["container"]["bind"] == clean["container_bind"] == []
    assert preprocessing["func"]["synbold_disco_license"] == str(tmp_path / "license")
    assert preprocessing["anat"]["mni_template"].startswith(str(tmp_path / "templates"))


def test_invalid_path_update_is_atomic(isolated_site):
    original = isolated_site.read_bytes()
    with pytest.raises(ValueError):
        edit_settings(["work=/new/work", "typo=/new/path"])
    assert isolated_site.read_bytes() == original
    with pytest.raises(ValueError):
        edit_settings(["work=relative/path"])


@pytest.mark.parametrize("shared", [False, True])
def test_interactive_generic_defaults_preserve_explicit_paths(
    isolated_site, tmp_path, monkeypatch, shared
):
    monkeypatch.setattr(site_setup, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        site_setup, "installation_record", lambda: {"mode": "shared" if shared else "personal"}
    )
    save_settings(isolated_site, {"work": "/configured/work"})
    monkeypatch.setenv("NRO_BIDS_PATH", "/configured/BIDS")
    _, sources = site.settings()
    proposed = site_setup.interactive_defaults(sources)
    assert proposed["images"] == str(tmp_path / "home/nro/images")
    assert proposed["registry"] == str(tmp_path / "home/nro/.nro")
    assert "work" not in proposed and "bids" not in proposed
    assert proposed["binds"] == []


def test_interactive_defaults_keep_reachable_lab(isolated_site, tmp_path, monkeypatch):
    monkeypatch.setattr(site_setup, "LAB", tmp_path)
    _, sources = site.settings()
    assert site_setup.interactive_defaults(sources) == {}
    monkeypatch.setattr(site_setup.os, "access", lambda *args: False)
    assert site_setup.interactive_defaults(sources)["binds"] == []


@pytest.mark.parametrize("accept_all", [False, True])
def test_interactive_paths_display_and_save_proposals(
    isolated_site, tmp_path, monkeypatch, capsys, accept_all
):
    monkeypatch.setattr(site_setup, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(site_setup, "installation_record", lambda: {})
    monkeypatch.setattr(site_setup.sys.stdin, "isatty", lambda: True)
    prompts = []

    def accept(prompt):
        prompts.append(prompt)
        if prompt.startswith("Accept all"):
            return "y" if accept_all else "n"
        if not accept_all and "work [" in prompt:
            return str(tmp_path / "scratch")
        return ""

    monkeypatch.setattr("builtins.input", accept)
    edit_settings()
    proposed_bids = str(tmp_path / "home/nro/bids")
    assert proposed_bids in capsys.readouterr().out
    if accept_all:
        assert len(prompts) == 2
    else:
        assert f"bids [{proposed_bids}]" in "\n".join(prompts)
        assert len(prompts) == len(site_setup.DESCRIPTIONS) + 2
        assert site.read_overrides(isolated_site)["work"] == str(tmp_path / "scratch")
    assert not any("root" in prompt for prompt in prompts)
    assert site.read_overrides(isolated_site)["bids"] == proposed_bids
    assert site.read_overrides(isolated_site)["binds"] == []
    assert not (tmp_path / "home").exists()


def test_shared_site_ignores_personal_environment(isolated_site, monkeypatch):
    save_settings(isolated_site, {"bids": "/shared/BIDS"})
    monkeypatch.setattr(
        site, "installation_record", lambda: {"mode": "shared", "site": str(isolated_site)}
    )
    monkeypatch.setattr("nro.engine.site_setup.installation_record", site.installation_record)
    monkeypatch.setenv("NRO_BIDS_PATH", "/personal/BIDS")
    assert site.settings()[0]["bids"] == "/shared/BIDS"
    with pytest.raises(ValueError, match="maintain"):
        edit_settings(["bids=/other"])
    monkeypatch.setenv("NRO_SITE_CONFIG", "/other/site.toml")
    with pytest.raises(ValueError, match="shared installation"):
        site.site_file()


def test_shared_onboarding_never_syncs_or_mutates_checkout(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    environment = root / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("interpreter")
    config = root / "site.toml"
    config.write_text("")
    record = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(environment),
        "site": str(config),
        "ready": True,
    }
    (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)

    def unexpected(*args, **kwargs):
        pytest.fail("User onboarding attempted shared maintenance")

    monkeypatch.setattr(bootstrap.subprocess, "run", unexpected)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    bin_dir = tmp_path / "user/bin"
    bootstrap.main(["--bin-dir", str(bin_dir)])
    bootstrap.main(["--bin-dir", str(bin_dir)])
    after = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert not (root / ".nro-install.lock").exists()
    assert "nro directory-aware launcher" in (bin_dir / "nro").read_text()
    assert json.loads((bin_dir / ".nro-launchers.json").read_text())["default"] == str(root)


def test_shared_maintenance_drains_and_publishes_checked_out_release(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    environment = root / ".nro-env"
    site = root / "site.toml"
    site.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(environment),
        "site": str(site),
        "ready": True,
        "with_oslom": True,
        "with_bidsify": False,
        "dev": False,
        "local": False,
    }
    (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr("nro.orchestration.releases.tagged_source", lambda checkout: ())
    events = []
    monkeypatch.setattr(
        shared_installation,
        "prepare_pool",
        lambda registry, **options: events.append(("drain", options["checkout"])),
    )
    monkeypatch.setattr(
        shared_installation,
        "publish",
        lambda checkout, registry: events.append(("publish", checkout)),
    )
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda command, **options: None)
    monkeypatch.setattr(bootstrap, "connect_user", lambda *args, **options: None)

    bootstrap.main(["--maintain", "--offline"])

    assert events == [("drain", root), ("publish", root)]
    assert json.loads((root / bootstrap.RECORD).read_text())["ready"] is True


@pytest.mark.parametrize("without_oslom", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_personal_setup_installs_oslom_by_default(tmp_path, monkeypatch, without_oslom, existing):
    root = tmp_path / "personal"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    config = root / "site.toml"
    config.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record = {
        "mode": "personal",
        "checkout": str(root),
        "environment": str(root / ".nro-env"),
        "site": str(config),
        "ready": True,
        "with_oslom": False,
    }
    if existing:
        (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(bootstrap, "connect_user", lambda *a, **kw: None)
    bootstrap.main(
        [
            "--offline",
            "--mode",
            "personal",
            "--site",
            str(config),
            *(["--without-oslom"] if without_oslom else []),
        ]
    )
    assert "sync" in calls[0][0] and "--frozen" in calls[0][0]
    assert ("oslom" in calls[0][0]) is not without_oslom
    assert calls[0][1]["cwd"] == root
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(root / ".nro-env")
    assert ("--without-oslom" in calls[1][0]) is without_oslom
    assert json.loads((root / bootstrap.RECORD).read_text())["ready"]


def test_maintenance_rejects_active_workers(tmp_path):
    control = tmp_path / "registry"
    (control / "shared/scheduler").mkdir(parents=True)
    with sqlite3.connect(control / "shared/scheduler/registry.sqlite3") as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT)")
        db.execute("CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT)")
        db.execute("INSERT INTO workers VALUES ('running', NULL)")
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})
    with pytest.raises(RuntimeError, match="Stop or drain"):
        bootstrap.check_workers(config)


@pytest.mark.parametrize(
    "scheduler", ["ended", "RUNNING", "PENDING", "COMPLETING", "failed", "timeout", "missing"]
)
def test_maintenance_checks_slurm_without_mutating_registry(tmp_path, monkeypatch, scheduler):
    control = tmp_path / "registry"
    (control / "shared/scheduler").mkdir(parents=True)
    database = control / "shared/scheduler/registry.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT)")
        db.execute("CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT)")
        db.execute("INSERT INTO workers VALUES ('running', '123')")
        db.executemany(
            "INSERT INTO scheduler_submissions VALUES (?, ?)",
            [
                ("submitted", "123"),
                ("cancel_requested", "124"),
            ],
        )
    original = database.read_bytes()
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})

    def query(command, **kwargs):
        assert command == ["squeue", "--noheader", "--jobs", "123,124", "--format", "%T"]
        assert kwargs["check"] and kwargs["timeout"] == 15
        if scheduler == "failed":
            raise subprocess.CalledProcessError(1, command)
        if scheduler == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if scheduler == "missing":
            raise FileNotFoundError("squeue")
        return subprocess.CompletedProcess(
            command, 0, stdout="" if scheduler == "ended" else scheduler + "\n"
        )

    monkeypatch.setattr(bootstrap.subprocess, "run", query)
    if scheduler == "ended":
        bootstrap.check_workers(config)
    else:
        with pytest.raises(RuntimeError):
            bootstrap.check_workers(config)
    assert database.read_bytes() == original


def test_launcher_does_not_replace_another_installation(tmp_path):
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("python")
    config = tmp_path / "site.toml"
    config.write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nro").write_text("another installation")
    record = {
        "checkout": str(tmp_path),
        "mode": "shared",
        "environment": str(python.parent.parent),
        "site": str(config),
        "ready": True,
    }
    (tmp_path / bootstrap.RECORD).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="another command"):
        bootstrap.connect_user(record, bin_dir=bin_dir)
    assert (bin_dir / "nro").read_text() == "another installation"


@pytest.mark.parametrize("header, checksum", [("99", None), ("4", "incorrect")])
def test_failed_download_never_replaces_target(tmp_path, monkeypatch, header, checksum):
    class Response(io.BytesIO):
        url = "https://example.org/file"
        headers = {"Content-Length": header}

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", lambda *a, **k: Response(b"data"))
    target = tmp_path / "image.sif"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        dependencies.download("https://example.org/file", target, checksum=checksum)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_archive_escape_is_rejected(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="escapes"):
        dependencies.extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()


def test_workbench_installs_wrapper_and_reuses_it(isolated_site, tmp_path, monkeypatch):
    command = tmp_path / "workbench/bin_linux64/wb_command"
    save_settings(isolated_site, {"workbench": str(command)})

    def archive_download(url, target, checksum=None):
        assert checksum == dependencies.WORKBENCH_SHA256["linux64"]
        with zipfile.ZipFile(target, "w") as zipped:
            zipped.writestr("workbench/bin_linux64/wb_command", "wrapper")
            zipped.writestr("workbench/exe_linux64/wb_command", "binary")

    calls = []
    monkeypatch.setattr(dependencies, "download", archive_download)
    monkeypatch.setattr(dependencies.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(dependencies.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu"})
    monkeypatch.setattr(dependencies, "run_probe", lambda command: calls.append(command))
    dependencies.install_workbench()
    assert command.read_text() == "wrapper"
    assert "/bin_linux64/" in calls[0][0]
    monkeypatch.setattr(
        dependencies, "download", lambda *a, **kw: pytest.fail("Unexpected download")
    )
    dependencies.install_workbench(offline=True)


def test_install_help_from_another_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "install"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--maintain" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("exists", [False, True])
def test_workers_capture_resolved_site_settings(tmp_path, monkeypatch, exists):
    from nro.bin.run import _write_worker_script

    config = tmp_path / "site settings.toml"
    if exists:
        config.write_text("")
    monkeypatch.setattr(site, "site_file", lambda: config)
    registry = SimpleNamespace(
        paths=SimpleNamespace(workers=tmp_path / "workers", control=tmp_path / "control")
    )
    script = _write_worker_script(
        registry,
        bids_root=tmp_path / "BIDS",
        partition="test",
        account=None,
        hours=1,
        memory_gb=4,
        cpus=1,
    ).read_text()
    assert "source_launcher.py" in script
    assert str(config) not in script
    (pinned_site,) = (registry.paths.control / "shared/cache/execution-sites").glob("*.toml")
    assert str(pinned_site) in script
    values = site.read_overrides(pinned_site)
    assert values["bids"] == str(tmp_path / "BIDS")
    assert values["registry"] == str(registry.paths.control)


@pytest.mark.parametrize(
    "url, checksum",
    [
        (dependencies.OSLOM_SOURCE, None),
        (dependencies.OSLOM_SOURCE, "wrong"),
        ("http://example.org/archive", dependencies.OSLOM_SHA256),
    ],
)
def test_http_download_requires_exact_oslom_pin(tmp_path, url, checksum):
    with pytest.raises(ValueError, match="HTTPS"):
        dependencies.download(url, tmp_path / "archive", checksum=checksum)


def test_official_oslom_http_download(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        url = dependencies.OSLOM_SOURCE
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", lambda *a, **kw: Response(b"data"))
    monkeypatch.setattr(dependencies, "sha256", lambda path: dependencies.OSLOM_SHA256)
    target = tmp_path / "archive"
    dependencies.download(dependencies.OSLOM_SOURCE, target, checksum=dependencies.OSLOM_SHA256)
    assert target.read_bytes() == b"data"


@pytest.mark.parametrize("failure", [None, "compile", "fit", "escape", "symlink"])
def test_oslom_build_publishes_only_after_validation(isolated_site, tmp_path, monkeypatch, failure):
    target = tmp_path / "installed/oslom_undir"
    save_settings(isolated_site, {"oslom": str(target)})
    monkeypatch.setattr(dependencies.shutil, "which", lambda name: "/usr/bin/g++")

    def archive_download(url, destination, checksum=None):
        assert url == dependencies.OSLOM_SOURCE
        assert checksum == dependencies.OSLOM_SHA256
        with tarfile.open(destination, "w:gz") as archive:
            member = tarfile.TarInfo("../escape" if failure == "escape" else "OSLOM2/example.dat")
            member.size = 4
            if failure == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                member.size = 0
            archive.addfile(member, io.BytesIO(b"data"))

    def probe(command, *, cwd=None, **kwargs):
        if "--version" in command:
            return "test compiler"
        if "-O3" in command:
            if failure == "compile":
                raise RuntimeError("compile failed")
            (cwd / "oslom_undir").write_bytes(b"binary")
        elif failure != "fit":
            output = cwd / "example.dat_oslo_files/tp"
            output.parent.mkdir()
            output.write_text("#module 0\n0 1 2\n")
        return ""

    monkeypatch.setattr(dependencies, "download", archive_download)
    monkeypatch.setattr(dependencies, "run_probe", probe)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            dependencies.install_oslom()
        assert not target.exists()
        assert not target.with_name(target.name + ".receipt.json").exists()
    else:
        dependencies.install_oslom()
        receipt = json.loads(target.with_name(target.name + ".receipt.json").read_text())
        assert receipt["source_sha256"] == dependencies.OSLOM_SHA256
        assert receipt["sha256"] == dependencies.sha256(target)
        assert target.stat().st_mode & 0o111
        monkeypatch.setattr(
            dependencies, "download", lambda *a, **kw: pytest.fail("Unexpected download")
        )
        dependencies.install_oslom(offline=True)
    assert not list(target.parent.glob(".oslom-*"))


def test_missing_oslom_offline_does_not_create_directories(isolated_site, tmp_path):
    target = tmp_path / "absent/oslom_undir"
    save_settings(isolated_site, {"oslom": str(target)})
    with pytest.raises(RuntimeError, match="offline"):
        dependencies.install_oslom(offline=True)
    assert not target.parent.exists()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("entry", ["bootstrap", "setup", "paths"])
def test_setup_interrupts_exit_without_tracebacks(monkeypatch, capsys, entry, interruption):
    from nro.bin import paths, setup

    module = {"bootstrap": bootstrap, "setup": setup, "paths": paths}[entry]

    def interrupt(*args, **kwargs):
        raise interruption()

    monkeypatch.setattr(module, "edit_settings" if entry == "paths" else "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 130
    output = capsys.readouterr().err
    assert "cancelled" in output
    assert "Traceback" not in output


def test_interrupted_path_prompt_preserves_settings(isolated_site, monkeypatch):
    original = isolated_site.read_bytes()
    monkeypatch.setattr(site_setup.sys.stdin, "isatty", lambda: True)

    def interrupt(prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        edit_settings()
    assert isolated_site.read_bytes() == original


def test_bootstrap_propagates_child_cancellation(monkeypatch):
    def interrupt(*args):
        raise subprocess.CalledProcessError(130, ["setup"])

    monkeypatch.setattr(bootstrap, "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        bootstrap.main([])
    assert error.value.code == 130


@pytest.mark.parametrize("entry", ["bootstrap", "setup"])
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("kind", ["signal", "exit"])
def test_only_outer_setup_reports_cancellation(monkeypatch, capsys, entry, child, kind):
    from nro.bin import setup

    module = bootstrap if entry == "bootstrap" else setup
    if child:
        monkeypatch.setenv("NRO_SETUP_CHILD", "1")
    else:
        monkeypatch.delenv("NRO_SETUP_CHILD", raising=False)

    def interrupt(*args, **kwargs):
        if kind == "signal":
            raise KeyboardInterrupt()
        if entry == "bootstrap":
            raise subprocess.CalledProcessError(130, ["setup"])
        raise SystemExit(130)

    monkeypatch.setattr(module, "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 130
    assert capsys.readouterr().err.count("Setup cancelled") == (0 if child else 1)
