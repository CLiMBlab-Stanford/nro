from __future__ import annotations

from pathlib import Path

import pytest

from nro.bin.migrate import main


def _report(*, changed=(), contracts=0, errors=(), templates=(), source_links=()):
    return {
        "scanned": 12,
        "changed": list(changed),
        "contracts": contracts,
        "templates": list(templates),
        "source_links": list(source_links),
        "errors": list(errors),
        "preparation": "prepared-migration",
    }


def _configure(monkeypatch, reports):
    calls = []
    report_index = 0

    def maintenance(*args, **kwargs):
        nonlocal report_index
        calls.append(kwargs)
        if kwargs["operation"] == "preparation_cancel":
            return {"cancelled": True}
        report = reports[report_index]
        report_index += 1
        return report

    monkeypatch.setattr(
        "nro.bin.migrate.settings",
        lambda: ({"bids": "/bids", "registry": "/registry"}, None),
    )
    monkeypatch.setattr("nro.orchestration.scheduler_client.maintenance", maintenance)
    monkeypatch.setattr("nro.bin.migrate.package_version", lambda: "1.2.3")
    return calls


def test_migrate_previews_in_pager_then_confirms_execution(monkeypatch, capsys) -> None:
    path = Path("/bids/demo/derivatives/nro/manifest.json")
    calls = _configure(
        monkeypatch,
        [_report(changed=(path,), contracts=2), _report(changed=(path,), contracts=2)],
    )
    pages = []
    monkeypatch.setattr("nro.bin.migrate.page_text", pages.append)
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    main(["-P", "demo"])

    assert [call["execute"] for call in calls] == [False, True]
    assert calls[1]["preparation"] == "prepared-migration"
    assert "Planned project migration" in pages[0]
    assert str(path) in pages[0]
    assert (
        "Migrated 1 metadata file(s), 0 raw BIDS link(s), 0 template link(s), "
        "and 2 work-item contract(s)." in capsys.readouterr().out
    )


def test_migrate_dry_run_only_previews(monkeypatch) -> None:
    calls = _configure(monkeypatch, [_report(changed=(Path("/one"),))])
    monkeypatch.setattr("nro.bin.migrate.page_text", lambda _text: None)
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt"))
    )

    main(["-P", "demo", "--dry-run"])

    assert [call["operation"] for call in calls] == [
        "dataset_migration",
        "preparation_cancel",
    ]
    assert calls[1]["preparation"] == "prepared-migration"


def test_migrate_force_prints_preview_without_pager_or_prompt(monkeypatch, capsys) -> None:
    path = Path("/bids/demo/manifest.json")
    calls = _configure(monkeypatch, [_report(changed=(path,)), _report(changed=(path,))])
    monkeypatch.setattr(
        "nro.bin.migrate.page_text",
        lambda _text: (_ for _ in ()).throw(AssertionError("unexpected pager")),
    )
    monkeypatch.setattr(
        "builtins.input", lambda _prompt: (_ for _ in ()).throw(AssertionError("unexpected prompt"))
    )

    main(["-P", "demo", "-f"])

    assert [call["execute"] for call in calls] == [False, True]
    output = capsys.readouterr().out
    assert "Planned project migration" in output
    assert str(path) in output


def test_migrate_cancel_leaves_preview_unapplied(monkeypatch, capsys) -> None:
    calls = _configure(monkeypatch, [_report(changed=(Path("/one"),))])
    monkeypatch.setattr("nro.bin.migrate.page_text", lambda _text: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    main(["-P", "demo"])

    assert [call["operation"] for call in calls] == [
        "dataset_migration",
        "preparation_cancel",
    ]
    assert "Project migration cancelled." in capsys.readouterr().out


def test_migrate_rejects_removed_dataset_subcommand() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["dataset"])


def test_migrate_rejects_removed_project_subcommand() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["project", "demo"])


def test_migrate_rejects_nonproject_selectors() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["-p", "01"])
