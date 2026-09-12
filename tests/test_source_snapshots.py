"""Source capture and isolated imports use temporary trees, not live workers."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nro.orchestration.source_snapshots import SourceStore


def source(tmp_path):
    root = tmp_path / "checkout"
    package = root / "nro"
    (package / "orchestration").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "probe.py").write_text('print("original")\n')
    (package / "resource.txt").write_text("resource")
    launcher = Path(__file__).resolve().parents[1] / "nro/orchestration/source_launcher.py"
    shutil.copyfile(launcher, package / "orchestration/source_launcher.py")
    (root / "pyproject.toml").write_text('[project]\nname = "nro"\n')
    return root


def test_capture_preserves_source_and_excludes_local_state(tmp_path):
    root = source(tmp_path)
    (root / ".env").write_text("not source")
    (root / "nro/__pycache__").mkdir()
    (root / "nro/__pycache__/ignore.pyc").write_bytes(b"cache")
    store = SourceStore(tmp_path / "snapshots")
    first = store.capture(root)
    first.verify()
    assert not (first.root / ".env").exists()
    assert not (first.root / "nro/__pycache__").exists()
    assert (first.root / "nro/resource.txt").read_text() == "resource"
    assert not (first.root / "nro/probe.py").stat().st_mode & 0o222
    (root / "nro/probe.py").touch()
    assert store.capture(root) == first
    (root / "nro/probe.py").write_text('print("edited")\n')
    second = store.capture(root)
    assert first.digest != second.digest
    assert (first.root / "nro/probe.py").read_text() == 'print("original")\n'
    assert not list(store.root.glob(".capture-*"))


def test_snapshot_launcher_imports_captured_not_live_code(tmp_path):
    root = source(tmp_path)
    snapshot = SourceStore(tmp_path / "snapshots").capture(root)
    command = snapshot.command((sys.executable, "-m", "nro.probe"))
    (root / "nro/probe.py").write_text('raise RuntimeError("live code")\n')
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "original"
    assert not list(snapshot.root.rglob("__pycache__"))


def test_manifest_only_launcher_avoids_reading_unrelated_package_data(tmp_path):
    root = source(tmp_path)
    snapshot = SourceStore(tmp_path / "snapshots").capture(root)
    command = snapshot.command((sys.executable, "-m", "nro.probe"), manifest_only=True)
    assert command[2] == "--manifest-only"
    (snapshot.root / "nro/resource.txt").chmod(0o644)
    (snapshot.root / "nro/resource.txt").write_text("changed")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "original"


def test_expected_capture_reuses_published_snapshot_without_copying(tmp_path, monkeypatch):
    root = source(tmp_path)
    store = SourceStore(tmp_path / "snapshots")
    snapshot = store.capture(root)

    def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("existing snapshots must not be copied again")

    monkeypatch.setattr(shutil, "copyfile", unexpected_copy)
    assert store.capture(root, expected_digest=snapshot.digest) == snapshot
    (root / "nro/probe.py").write_text('print("edited")\n')
    with pytest.raises(ValueError, match="changed during capture"):
        store.capture(root, expected_digest=snapshot.digest)


def test_manifest_only_capability_is_detected_from_verified_launcher(tmp_path):
    root = source(tmp_path)
    store = SourceStore(tmp_path / "snapshots")
    assert store.capture(root).supports_manifest_only()
    launcher = root / "nro/orchestration/source_launcher.py"
    launcher.write_text(launcher.read_text().replace("MANIFEST_ONLY_PROTOCOL = 1", ""))
    assert not store.capture(root).supports_manifest_only()


@pytest.mark.parametrize("mutation", ["changed", "extra", "symlink", "missing", "bytecode"])
def test_verification_rejects_source_changes_at_launch(tmp_path, mutation):
    root = source(tmp_path)
    store = SourceStore(tmp_path / "snapshots")
    snapshot = store.capture(root)
    command = snapshot.command((sys.executable, "-m", "nro.probe"))
    target = snapshot.root / "nro/probe.py"
    if mutation == "changed":
        target.chmod(0o644)
        target.write_text('print("changed")\n')
    elif mutation == "extra":
        (snapshot.root / "nro/extra.py").write_text("")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(root / "nro/probe.py")
    elif mutation == "bytecode":
        (snapshot.root / "nro/__pycache__").mkdir()
    else:
        target.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        snapshot.verify()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    with pytest.raises((ValueError, FileNotFoundError)):
        store.capture(root)


def test_capture_rejects_symlinks_and_detects_concurrent_edits(tmp_path, monkeypatch):
    root = source(tmp_path)
    store = SourceStore(tmp_path / "snapshots")
    link = root / "nro/link"
    link.symlink_to(root / "nro/resource.txt")
    with pytest.raises(ValueError, match="regular files"):
        store.capture(root)
    link.unlink()
    copy = shutil.copyfile

    def changing_copy(src, dest, **kwargs):
        copy(src, dest, **kwargs)
        if src.name == "probe.py":
            src.write_text('print("edited during capture")\n')

    monkeypatch.setattr(shutil, "copyfile", changing_copy)
    with pytest.raises(ValueError, match="changed during capture"):
        store.capture(root)
    assert not list(store.root.iterdir())


def test_snapshot_rejects_unrelated_commands(tmp_path):
    snapshot = SourceStore(tmp_path / "snapshots").capture(source(tmp_path))
    for command in [(sys.executable, "-c", "print(1)"), ("sh", "-c", "anything")]:
        with pytest.raises(ValueError, match="Python -m nro"):
            snapshot.command(command)


def test_captured_site_overrides_environment_and_rejects_later_edits(tmp_path):
    from nro.configuration.site import settings
    from nro.orchestration.execution_pins import capture_site

    root = source(tmp_path)
    config = root / "nro/configuration"
    config.mkdir()
    (config / "__init__.py").write_text("")
    original = Path(__file__).resolve().parents[1] / "nro/configuration/site.py"
    shutil.copyfile(original, config / "site.py")
    (root / "nro/probe.py").write_text(
        'from nro.configuration.site import settings\nprint(settings()[0]["bids"])\n'
    )
    snapshot = SourceStore(tmp_path / "snapshots").capture(root)
    values = settings()[0]
    values["bids"] = str(tmp_path / "selected-BIDS")
    site = capture_site(tmp_path / "sites", values)
    assert capture_site(tmp_path / "sites", values) == site
    command = snapshot.command((sys.executable, "-m", "nro.probe"), site=site)
    result = subprocess.run(
        command,
        env={"NRO_SITE_CONFIG": str(tmp_path / "wrong-site.toml")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == values["bids"]
    site.chmod(0o644)
    site.write_text('bids = "/changed"\n')
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert "settings changed" in result.stderr
    with pytest.raises(ValueError, match="settings changed"):
        capture_site(tmp_path / "sites", values)
