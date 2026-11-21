"""./install --upgrade moves a checkout to the newest release; nro install reruns ./install."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(path, *arguments):
    return subprocess.run(
        ["git", "-C", str(path), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(path, message):
    _git(path, "add", "-A")
    _git(path, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", message)


def _tag(path, name):
    _git(path, "-c", "user.name=t", "-c", "user.email=t@example.com", "tag", "-a", name, "-m", name)


@pytest.fixture
def checkout(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    shutil.copy2(ROOT / "install", origin / "install")
    (origin / "version").write_text("1\n")
    _commit(origin, "first release")
    _tag(origin, "v0.1.0")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True)
    (origin / "version").write_text("2\n")
    _commit(origin, "second release")
    _tag(origin, "v0.2.0")
    (origin / "version").write_text("3\n")
    _commit(origin, "unreleased")
    return origin, clone


def _install(clone, *arguments):
    return subprocess.run(
        [str(clone / "install"), *arguments], capture_output=True, text=True, cwd=clone
    )


def test_upgrade_fast_forwards_main_to_the_newest_release(checkout):
    origin, clone = checkout
    result = _install(clone, "--upgrade")
    assert "Updated main to v0.2.0" in result.stdout
    assert _git(clone, "rev-parse", "HEAD") == _git(origin, "rev-parse", "v0.2.0^{commit}")
    # The new release's ./install runs without --upgrade; here it stops for lack of pins.
    assert "missing bootstrap pins" in result.stderr
    again = _install(clone, "--upgrade")
    assert "already includes v0.2.0" in again.stdout


def test_upgrade_checks_out_a_detached_release(checkout):
    origin, clone = checkout
    _git(clone, "checkout", "-q", "--detach", "v0.1.0")
    result = _install(clone, "--upgrade")
    assert "Checked out v0.2.0" in result.stdout
    assert _git(clone, "rev-parse", "HEAD") == _git(origin, "rev-parse", "v0.2.0^{commit}")


def test_upgrade_refuses_changes_and_development_branches(checkout):
    _, clone = checkout
    (clone / "version").write_text("edited\n")
    assert "uncommitted changes" in _install(clone, "--upgrade").stderr
    _git(clone, "checkout", "-q", "--", "version")
    _git(clone, "checkout", "-q", "-b", "dev")
    result = _install(clone, "--upgrade")
    assert result.returncode != 0
    assert "the checkout is on dev" in result.stderr


def test_nro_install_runs_the_checkouts_install(monkeypatch, capsys):
    from nro.bin import install

    calls = []
    monkeypatch.setattr(install.os, "execv", lambda path, argv: calls.append((path, argv)))
    monkeypatch.setattr(install.os, "chdir", lambda path: None)
    monkeypatch.setattr(
        "nro.site.configuration.installation_record", lambda: {"checkout": str(ROOT)}
    )
    install.main(["--upgrade", "--maintain"])
    script = str(ROOT / "install")
    assert calls == [(script, [script, "--upgrade", "--maintain"])]
    assert "Running" in capsys.readouterr().out


def test_nro_install_uses_the_recorded_checkout_from_a_snapshot(tmp_path, monkeypatch):
    from nro.bin import install

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "install").write_text("#!/bin/sh\n")
    (checkout / "pyproject.toml").write_text("")
    monkeypatch.setattr(
        "nro.site.configuration.installation_record", lambda: {"checkout": str(checkout)}
    )
    assert install.checkout() == checkout
    monkeypatch.setattr("nro.site.configuration.installation_record", lambda: {})
    assert install.checkout() == ROOT
