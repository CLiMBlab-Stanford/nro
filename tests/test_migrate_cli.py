from __future__ import annotations

from pathlib import Path

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

    def maintenance(*args, **kwargs):
        calls.append(kwargs)
        return reports[len(calls) - 1]

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

    main(["dataset", "-P", "demo"])

    assert [call["execute"] for call in calls] == [False, True]
    assert calls[1]["preparation"] == "prepared-migration"
    assert "Planned dataset migration" in pages[0]
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

    main(["dataset", "-P", "demo", "--dry-run"])

    assert [call["execute"] for call in calls] == [False]


def test_migrate_cancel_leaves_preview_unapplied(monkeypatch, capsys) -> None:
    calls = _configure(monkeypatch, [_report(changed=(Path("/one"),))])
    monkeypatch.setattr("nro.bin.migrate.page_text", lambda _text: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    main(["dataset", "-P", "demo"])

    assert [call["execute"] for call in calls] == [False]
    assert "Dataset migration cancelled." in capsys.readouterr().out
