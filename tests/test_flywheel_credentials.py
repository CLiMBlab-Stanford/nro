"""Private Flywheel credential storage and command-line behavior."""

from __future__ import annotations

import os
import stat

import pytest

from nro.bidsify.credentials import (
    PRIVATE_DIRECTORY,
    credential_path,
    has_key,
    read_key,
    remove_key,
    store_key,
)


def test_keys_are_private_per_user_but_group_maintainable(tmp_path):
    root = tmp_path / "definitions"
    root.mkdir()

    path = store_key(root, "cni", "cni.example.org:secret-value", host="cni.example.org")

    assert path == credential_path(root, "cni")
    assert path.relative_to(root).parts[:2] == (PRIVATE_DIRECTORY, f"user-{os.geteuid()}")
    assert read_key(root, "cni", host="cni.example.org") == "secret-value"
    assert has_key(root, "cni", host="cni.example.org")
    assert stat.S_IMODE(path.stat().st_mode) == 0o620
    assert stat.S_IMODE(path.parents[2].stat().st_mode) & 0o770 == 0o770
    assert stat.S_IMODE(path.parents[1].stat().st_mode) & 0o770 == 0o770
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o770 == 0o770

    assert remove_key(root, "cni")
    assert not remove_key(root, "cni")
    assert not has_key(root, "cni", host="cni.example.org")


def test_key_host_must_match_selected_server(tmp_path):
    root = tmp_path / "definitions"
    root.mkdir()
    with pytest.raises(ValueError, match="host does not match"):
        store_key(root, "cni", "other.example.org:secret", host="cni.example.org")


def test_unsafe_key_permissions_are_rejected(tmp_path):
    root = tmp_path / "definitions"
    root.mkdir()
    path = store_key(root, "cni", "secret", host="cni.example.org")
    path.chmod(0o660)
    with pytest.raises(ValueError, match="unsafe ownership or permissions"):
        read_key(root, "cni", host="cni.example.org")


def test_fw_cli_prompts_without_echo_and_never_prints_key(tmp_path, monkeypatch, capsys):
    from nro.bin import fw

    root = tmp_path / "definitions"
    root.mkdir()
    bidsify = {"servers": {"cni": {"host": "cni.example.org", "projects": []}}}
    monkeypatch.setattr(fw, "_site", lambda: (root, bidsify))
    monkeypatch.setattr(fw.getpass, "getpass", lambda prompt: "hidden-secret")

    fw.main(["addkey", "cni"])

    output = capsys.readouterr().out
    assert output == "Stored Flywheel key for cni.\n"
    assert "hidden-secret" not in output
    assert read_key(root, "cni", host="cni.example.org") == "hidden-secret"
    fw.main(["list"])
    assert capsys.readouterr().out == "cni\tconfigured\n"


def test_bidsify_authentication_offers_private_setup(tmp_path, monkeypatch, capsys):
    from nro.bin import bidsify

    root = tmp_path / "definitions"
    root.mkdir()
    config = {"servers": {"cni": {"host": "cni.example.org", "projects": []}}}
    answers = iter(["y"])
    monkeypatch.setattr(bidsify, "ask", lambda *args: next(answers))
    monkeypatch.setattr(bidsify.getpass, "getpass", lambda prompt: "new-secret")

    bidsify.authenticate(root, config, "cni")

    output = capsys.readouterr().out
    assert "requires an API key for your Unix account" in output
    assert "not written to tracked definitions" in output
    assert "new-secret" not in output
    assert read_key(root, "cni", host="cni.example.org") == "new-secret"


def test_bidsify_ls_lists_sources_and_sessions_without_scheduler(tmp_path, monkeypatch, capsys):
    from nro.bin import bidsify

    config = {"servers": {"cni": {"host": "cni.example.org", "projects": ["lab/one", "lab/two"]}}}
    rows = [
        {
            "id": "session-id",
            "label": "session-label",
            "subject_code": "subject-code",
            "remote_project": "lab/one",
        }
    ]

    class Source:
        def __init__(self, profile, **kwargs):
            assert profile["projects"] == ["lab/one", "lab/two"]

        def sessions(self):
            return rows

    monkeypatch.setattr(bidsify, "authenticate", lambda *args: None)
    monkeypatch.setattr(bidsify, "FlywheelSource", Source)

    bidsify.list_sources(tmp_path, config)

    assert capsys.readouterr().out.splitlines() == [
        "cni\tcni.example.org",
        "  lab/one",
        "    session-id\tsession-label",
        "  lab/two",
        "    (no sessions)",
    ]


def test_bidsify_parser_accepts_listing_action():
    from nro.bin.bidsify import build_parser

    args = build_parser().parse_args(["ls", "-f", "cni", "-F", "lab/study"])
    assert (args.action, args.flywheel_server, args.flywheel_project) == (
        "ls",
        "cni",
        "lab/study",
    )


def test_bidsify_ls_main_path_does_not_open_or_start_orchestration(tmp_path, monkeypatch):
    from nro.bin import bidsify

    config = {"servers": {}}
    monkeypatch.setattr(
        bidsify,
        "settings",
        lambda: ({"definitions": str(tmp_path), "registry": str(tmp_path / "registry")}, {}),
    )
    monkeypatch.setattr(bidsify, "load_config", lambda path: config)
    observed = []
    monkeypatch.setattr(
        bidsify,
        "list_sources",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        bidsify.Registry,
        "for_project",
        lambda *args, **kwargs: pytest.fail("listing opened the registry"),
    )

    bidsify.main(["ls"])

    assert observed == [((tmp_path.resolve(), config), {"server": None, "flywheel_project": None})]
