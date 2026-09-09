"""Directory selection cannot silently redirect development commands to production."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.engine import bootstrap, user_launcher


def installation(root, *, branch=None):
    root.mkdir(parents=True)
    python = root / ".nro-env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("interpreter")
    site = root / "site.toml"
    site.write_text("")
    record = dict(
        checkout=str(root),
        environment=str(python.parent.parent),
        site=str(site),
        ready=True,
        mode="branch" if branch else "shared",
    )
    if branch:
        record.update(
            branch=branch, branch_catalog=str(root / "catalog.json"), registry_id="a" * 32
        )
        (root / "catalog.json").write_text(
            json.dumps({branch: dict(retired=False, registry_id="a" * 32, checkouts=[str(root)])})
        )
    (root / user_launcher.RECORD_NAME).write_text(json.dumps(record))
    return record


def test_checkout_selection_preserves_default(tmp_path, monkeypatch):
    main = installation(tmp_path / "main")
    dev = installation(tmp_path / "dev", branch="dev")
    monkeypatch.setattr(
        user_launcher.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="dev\n"),
    )
    bin_dir = tmp_path / "bin"
    bootstrap.connect_user(main, bin_dir=bin_dir)
    bootstrap.connect_user(dev, bin_dir=bin_dir)
    index = user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)
    assert index["default"] == main["checkout"]
    assert user_launcher.select_installation(index, tmp_path) == main
    assert user_launcher.select_installation(index, tmp_path / "dev/nro/modules/func") == dev
    assert user_launcher.select_installation(index, tmp_path / "main") == main


def test_explicit_default_changes_only_outside_checkout_selection(tmp_path, monkeypatch):
    old = installation(tmp_path / "old")
    main = installation(tmp_path / "main")
    dev = installation(tmp_path / "dev", branch="dev")
    monkeypatch.setattr(
        user_launcher.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="dev"),
    )
    bin_dir = tmp_path / "bin"
    for record in (old, main, dev):
        bootstrap.connect_user(record, bin_dir=bin_dir)
    assert (
        user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)["default"] == old["checkout"]
    )
    monkeypatch.setattr(bootstrap, "ROOT", Path(main["checkout"]))
    bootstrap.main(["--default", "--bin-dir", str(bin_dir)])
    index = user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)
    assert user_launcher.select_installation(index, tmp_path) == main
    assert user_launcher.select_installation(index, tmp_path / "dev/subdirectory") == dev


def fixed_launcher(record):
    return (
        "#!/bin/sh\n# nro installation launcher\nexport PYTHONDONTWRITEBYTECODE=1\n"
        f"export NRO_SITE_CONFIG={record['site']}\n"
        f'exec {record["environment"]}/bin/python -m nro.cli "$@"\n'
    )


@pytest.mark.parametrize("set_default", [False, True])
def test_fixed_launcher_replacement_is_explicit_and_retains_backup(tmp_path, set_default):
    old = installation(tmp_path / "old")
    main = installation(tmp_path / "main")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    original = fixed_launcher(old)
    (bin_dir / "nro").write_text(original)
    with pytest.raises(RuntimeError, match="--replace-launcher"):
        bootstrap.connect_user(main, bin_dir=bin_dir, set_default=set_default)
    assert (bin_dir / "nro").read_text() == original
    assert not (bin_dir / user_launcher.INDEX_NAME).exists()
    bootstrap.connect_user(main, bin_dir=bin_dir, set_default=set_default, replace_launcher=True)
    index = user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)
    assert index["default"] == (main if set_default else old)["checkout"]
    assert old["checkout"] in index["checkouts"]
    assert [p.read_text() for p in bin_dir.glob("nro.previous-*")] == [original]
    assert "nro directory-aware launcher" in (bin_dir / "nro").read_text()


def test_replacement_does_not_accept_modified_shell_commands(tmp_path):
    record = installation(tmp_path / "main")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    original = fixed_launcher(record) + "echo custom\n"
    (bin_dir / "nro").write_text(original)
    with pytest.raises(RuntimeError, match="another command"):
        bootstrap.connect_user(record, bin_dir=bin_dir, replace_launcher=True)
    assert (bin_dir / "nro").read_text() == original


def test_failed_launcher_publication_restores_previous_default(tmp_path, monkeypatch):
    old, new = installation(tmp_path / "old"), installation(tmp_path / "new")
    bin_dir = tmp_path / "bin"
    bootstrap.connect_user(old, bin_dir=bin_dir)
    before = (bin_dir / user_launcher.INDEX_NAME).read_bytes()
    replace = Path.replace

    def fail(source, destination):
        if Path(destination) == bin_dir / "nro":
            raise OSError("publication failed")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        bootstrap.connect_user(new, bin_dir=bin_dir, set_default=True)
    assert (bin_dir / user_launcher.INDEX_NAME).read_bytes() == before


def test_incomplete_installation_cannot_replace_default(tmp_path):
    old, new = installation(tmp_path / "old"), installation(tmp_path / "new")
    bin_dir = tmp_path / "bin"
    bootstrap.connect_user(old, bin_dir=bin_dir)
    new["ready"] = False
    with pytest.raises(RuntimeError, match="incomplete"):
        bootstrap.connect_user(new, bin_dir=bin_dir, set_default=True)
    assert (
        user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)["default"] == old["checkout"]
    )


def test_branch_only_installation_has_no_outside_default(tmp_path, monkeypatch):
    record = installation(tmp_path / "dev", branch="dev")
    bootstrap.connect_user(record, bin_dir=tmp_path / "bin")
    index = user_launcher.read_index(tmp_path / "bin" / user_launcher.INDEX_NAME)
    assert index["default"] is None
    with pytest.raises(ValueError, match="No default"):
        user_launcher.select_installation(index, tmp_path)


@pytest.mark.parametrize(
    "problem", ["not_ready", "wrong_branch", "detached", "retired", "identity", "checkout"]
)
def test_invalid_branch_never_falls_back(tmp_path, monkeypatch, problem):
    main = installation(tmp_path / "main")
    dev = installation(tmp_path / "dev", branch="dev")
    bootstrap.connect_user(main, bin_dir=tmp_path / "bin")
    bootstrap.connect_user(dev, bin_dir=tmp_path / "bin")
    index = user_launcher.read_index(tmp_path / "bin" / user_launcher.INDEX_NAME)
    if problem == "not_ready":
        dev["ready"] = False
        (tmp_path / "dev" / user_launcher.RECORD_NAME).write_text(json.dumps(dev))
    elif problem in {"retired", "identity", "checkout"}:
        catalog = json.loads(Path(dev["branch_catalog"]).read_text())
        catalog["dev"][
            {"retired": "retired", "identity": "registry_id", "checkout": "checkouts"}[problem]
        ] = True if problem == "retired" else "b" * 32 if problem == "identity" else []
        Path(dev["branch_catalog"]).write_text(json.dumps(catalog))
    monkeypatch.setattr(
        user_launcher.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1 if problem == "detached" else 0,
            stdout="other" if problem == "wrong_branch" else "dev",
        ),
    )
    with pytest.raises(ValueError):
        user_launcher.select_installation(index, tmp_path / "dev")


def test_unregistered_nested_checkout_does_not_use_ancestor(tmp_path):
    main = installation(tmp_path / "main")
    nested = tmp_path / "main/nested"
    (nested / "nro").mkdir(parents=True)
    (nested / "nro/cli.py").write_text("")
    (nested / "install").write_text("")
    bootstrap.connect_user(main, bin_dir=tmp_path / "bin")
    index = user_launcher.read_index(tmp_path / "bin" / user_launcher.INDEX_NAME)
    with pytest.raises(ValueError, match="not connected"):
        user_launcher.select_installation(index, nested)


def test_launcher_uses_isolated_interpreter_and_selected_site(tmp_path, monkeypatch):
    main = installation(tmp_path / "main")
    bootstrap.connect_user(main, bin_dir=tmp_path / "bin")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/wrong/code")
    monkeypatch.setenv("PYTHONHOME", "/wrong/environment")
    monkeypatch.setenv("NRO_SITE_CONFIG", "/wrong/site")
    calls = []
    monkeypatch.setattr(user_launcher.os, "execve", lambda *args: calls.append(args))
    user_launcher.main(["run", "-p", "t20"], index_path=tmp_path / "bin" / user_launcher.INDEX_NAME)
    python, argv, env = calls[0]
    assert python == str(Path(main["environment"]) / "bin/python")
    assert argv[1:] == ["-I", "-B", "-m", "nro.cli", "run", "-p", "t20"]
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["NRO_SITE_CONFIG"] == main["site"]


def test_branch_install_does_not_maintain_site_or_change_default(tmp_path, monkeypatch):
    from nro.orchestration import branches
    from nro.orchestration.branch_store import BranchStore

    main = installation(tmp_path / "main")
    site = Path(main["site"])
    control = tmp_path / "control"
    site.write_text(f'registry = "{control}"\n')
    original_site = site.read_bytes()
    bin_dir = tmp_path / "bin"
    bootstrap.connect_user(main, bin_dir=bin_dir)
    root = tmp_path / "feature"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(branches, "checkout_identity", lambda _: (root, "feature", "a" * 40))
    monkeypatch.setattr(
        bootstrap, "check_workers", lambda *a: pytest.fail("Branch setup tried shared maintenance")
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "sync" in command:
            python = root / ".nro-env/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("python")
        if "nro.bin.branch" in command:
            store = BranchStore(control)
            snapshot = store.initialize()
            store.register("feature", "dev", checkout=root, revision=snapshot.revision)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    # The Git validator shares subprocess with the bootstrap; branch IDs have
    # their own focused tests, so avoid invoking that validator in this fixture.
    monkeypatch.setattr(branches, "branch_id", lambda name: name)
    monkeypatch.setattr("nro.orchestration.control_paths.branch_id", lambda name: name)
    bootstrap.main(["--offline", "--bin-dir", str(bin_dir)])
    record = json.loads((root / bootstrap.RECORD).read_text())
    assert record["mode"] == "branch" and record["branch"] == "feature" and record["ready"]
    assert record["environment"] == str(root / ".nro-env")
    assert site.read_bytes() == original_site
    assert "--maintain" not in calls[1][0]
    assert "nro.bin.setup" in calls[1][0] and "--resources-only" in calls[1][0]
    index = user_launcher.read_index(bin_dir / user_launcher.INDEX_NAME)
    assert index["default"] == main["checkout"]
    assert not (control / "shared/scheduler/registry.sqlite3").exists()
