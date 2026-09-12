"""Recognize externally bidsified sessions by identity and directory existence only."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from nro.bidsify.config import load_config
from nro.bidsify.discovery import (
    existing_sessions,
    inferred_session,
    session_choices,
    validate_session_rules,
)
from nro.bidsify.flywheel import FlywheelSource


def rules():
    return [
        dict(
            server="cni",
            remote_project="test/demo",
            match={"label": r"(?P<exam>[0-9]+)"},
            participant="ex{exam}",
            session=None,
        ),
        dict(
            server="cni",
            remote_project="test/demo",
            match={"label": r"(?P<exam>[0-9]+)"},
            participant=None,
            session="ex{exam}",
        ),
    ]


def remote(identifier="remote1", label="123", project="test/demo", **values):
    return dict(id=identifier, label=label, remote_project=project, **values)


def test_infer_session_only_from_agreeing_server_rules():
    assert inferred_session(remote(), server="cni", rules=rules()) == "ex123"
    assert inferred_session(remote(), server="lucas", rules=rules()) is None
    assert inferred_session(remote(project="other/project"), server="cni", rules=rules()) is None
    assert (
        inferred_session(remote(label="private display label"), server="cni", rules=rules()) is None
    )
    assert inferred_session(remote(), server="cni", rules=rules()[:1]) is None
    conflict = {**rules()[1], "session": "other{exam}"}
    assert inferred_session(remote(), server="cni", rules=[*rules(), conflict]) is None
    assert inferred_session(remote(), server="cni", rules=rules() * 2) == "ex123"


@pytest.mark.parametrize("configured", [True, False])
def test_cli_can_register_without_participant_and_infer_session(
    tmp_path, monkeypatch, capsys, configured
):
    from nro.bidsify.store import IngestionStore
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    registry = Registry.for_project("", bids_root=root)
    profile = load_config()
    profile["staging"] = str(tmp_path / "staging")
    profile["session_rules"] = rules() if configured else []
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    monkeypatch.setattr(cli, "load_config", lambda _: profile)
    monkeypatch.setattr(
        cli, "FlywheelSource", lambda _: SimpleNamespace(sessions=lambda: [remote()])
    )
    prompts = []
    answers = iter(["all", "", "y"] if configured else ["all", "", "", "y"])

    def answer(prompt, *args):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(cli, "ask", answer)
    cli.main(["--server", "cni", "-P", "demo", "--no-submit"])
    (row,) = IngestionStore(registry).rows()
    assert row["participant"] is None
    assert row["session"] == ("ex123" if configured else None)
    assert row["state"] == "queued" and row["stage"] == "inspect"
    assert any("BIDS session label" in prompt for prompt in prompts) is not configured
    assert "(participant pending)" in prompts[-1]
    assert not (root / "demo").exists()


def discover(root, rows=None, **kwargs):
    return session_choices(
        rows if rows is not None else [remote()],
        bids_root=root,
        server="cni",
        rules=kwargs.pop("rules", rules()),
        records=kwargs.pop("records", []),
        **kwargs,
    )


@pytest.mark.parametrize("relative", ["demo/sub-t20/ses-ex123", "other/sub-t20/ses-ex123"])
def test_existing_directory_is_sufficient_without_receipts_or_images(tmp_path, relative):
    target = tmp_path / relative
    target.mkdir(parents=True)
    before = set(tmp_path.rglob("*"))
    result = discover(tmp_path)
    assert not result.rows
    assert result.existing_hidden == 1
    assert set(tmp_path.rglob("*")) == before
    repeated = discover(tmp_path, rebidsify=True)
    assert repeated.rows[0]["already_bidsified"]
    assert repeated.rows[0]["existing_bids"][0].path == target


@pytest.mark.parametrize(
    "relative",
    [
        "demo/derivatives/anat/sub-ex123",
        "demo/sourcedata/sub-ex123",
        "derivatives/sub-ex123",
        ".hidden/sub-ex123",
        "_ignored_bids_backups/sub-ex123",
        "demo/sub-ex123/ses-unrelated",
        "demo/sub-ex123",
    ],
)
def test_nonmatching_or_nonraw_directories_do_not_hide_sessions(tmp_path, relative):
    (tmp_path / relative).mkdir(parents=True)
    result = discover(tmp_path)
    assert len(result.rows) == 1
    assert result.existing_hidden == 0


def test_ambiguous_existing_locations_stay_visible(tmp_path):
    for project in ("alpha", "beta"):
        (tmp_path / project / "sub-t20/ses-ex123").mkdir(parents=True)
    result = discover(tmp_path)
    assert result.rows[0]["mapping_ambiguous"]
    assert len(result.rows[0]["existing_bids"]) == 2
    assert not result.rows[0]["already_bidsified"]


def test_multiple_remote_sessions_for_one_directory_stay_visible(tmp_path):
    (tmp_path / "demo/sub-t20/ses-ex123").mkdir(parents=True)
    result = discover(tmp_path, [remote("a"), remote("b")], sessions=["a"])
    assert len(result.rows) == 1
    assert result.rows[0]["mapping_ambiguous"]


@pytest.mark.parametrize(
    "state", ["queued", "running", "needs_input", "failed", "interrupted", "awaiting_approval"]
)
def test_unfinished_requests_keep_recovery_priority(tmp_path, state):
    (tmp_path / "demo/sub-ex123").mkdir(parents=True)
    record = dict(server="cni", remote_session="remote1", state=state, project="another")
    result = discover(tmp_path, records=[record], rebidsify=True)
    assert not result.rows
    assert result.active_hidden == 1
    assert result.existing_hidden == 0
    assert record["state"] == state


def test_published_receipts_remain_authoritative_and_server_scoped(tmp_path):
    record = dict(server="lucas", remote_session="remote1", state="published")
    assert len(discover(tmp_path, records=[record]).rows) == 1
    record["server"] = "cni"
    assert discover(tmp_path, records=[record]).existing_hidden == 1
    assert len(discover(tmp_path, records=[record], rebidsify=True).rows) == 1


def test_rules_do_not_match_substrings_or_other_remote_projects(tmp_path):
    (tmp_path / "demo/sub-t20/ses-ex123").mkdir(parents=True)
    rows = [remote("a", "0123"), remote("b", "123-extra"), remote("c", project="other/project")]
    assert len(discover(tmp_path, rows).rows) == 3


def test_session_and_subject_fields_must_agree(tmp_path):
    (tmp_path / "demo/sub-t20/ses-ex123").mkdir(parents=True)
    rule = rules()[1]
    rule["match"]["subject_code"] = r"ex(?P<exam>[0-9]+)"
    assert discover(tmp_path, [remote(subject_code="ex123")], rules=[rule]).existing_hidden == 1
    assert len(discover(tmp_path, [remote(subject_code="ex456")], rules=[rule]).rows) == 1
    assert len(discover(tmp_path, rules=[rule]).rows) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("server", "unknown"),
        ("remote_project", "wrong/project"),
        ("match", {}),
        ("match", {"label": "["}),
        ("match", {"label": 123}),
        ("match", {"unknown": ".*"}),
        ("participant", "{missing}"),
        ("participant", "{exam.__class__}"),
        ("participant", "../{exam}"),
        ("participant", "{exam!r}"),
    ],
)
def test_invalid_rules_are_rejected(field, value):
    rule = deepcopy(rules()[0])
    rule[field] = value
    with pytest.raises(ValueError):
        validate_session_rules([rule], {"cni": {"projects": ["test/demo"]}})


def test_inventory_does_not_descend_into_data_files(tmp_path):
    path = tmp_path / "demo/sub-t20/ses-ex123/func"
    path.mkdir(parents=True)
    (path / "not-a-valid-image.nii.gz").write_bytes(b"not an image")
    assert len(existing_sessions(tmp_path)) == 1
    assert discover(tmp_path).existing_hidden == 1


def test_session_listing_uses_embedded_subject_without_extra_calls():
    session = SimpleNamespace(id="r", label="123", subject=SimpleNamespace(code="ex123"))
    project = SimpleNamespace(sessions=SimpleNamespace(iter=lambda: iter([session])))
    client = SimpleNamespace(lookup=lambda _: project)
    rows = FlywheelSource({"projects": ["test/demo"]}, client=client).sessions()
    assert rows == [remote("r", subject_code="ex123")]


def test_source_selection_requires_one_configured_project(monkeypatch):
    from nro.bin.bidsify import select_source

    config = load_config()
    config["servers"]["cni"]["projects"] = ["lab/one", "lab/two"]
    answers = iter(["all", "0", "2"])
    monkeypatch.setattr("nro.bin.bidsify.ask", lambda *_: next(answers))
    assert select_source(config, "demo", server="cni", flywheel_project=None) == ("cni", "lab/two")
    monkeypatch.setattr("nro.bin.bidsify.ask", lambda *_: pytest.fail("Explicit source prompted"))
    assert select_source(config, "demo", server="cni", flywheel_project="lab/one") == (
        "cni",
        "lab/one",
    )
    with pytest.raises(ValueError, match="No source configured"):
        select_source(config, "demo", server="cni", flywheel_project="other/lab")


def test_bids_project_mapping_selects_server_and_source_without_prompts(monkeypatch):
    from nro.bin.bidsify import select_source

    config = load_config()
    config["project_sources"] = {"demo": [{"server": "cni", "project": "test/demo"}]}
    monkeypatch.setattr("nro.bin.bidsify.ask", lambda *_: pytest.fail("Mapped source prompted"))
    assert select_source(config, "demo", server=None, flywheel_project=None) == ("cni", "test/demo")
    with pytest.raises(ValueError, match="No source configured"):
        select_source(config, "demo", server="lucas", flywheel_project=None)
    with pytest.raises(ValueError, match="No source configured"):
        select_source(config, "demo", server="cni", flywheel_project="other/lab")


def test_multisite_mapping_filters_and_prompts_by_server(monkeypatch):
    from nro.bin.bidsify import select_source

    config = load_config()
    config["project_sources"] = {
        "demo": [
            {"server": "cni", "project": "test/demo"},
            {"server": "lucas", "project": "test/demo"},
        ]
    }
    monkeypatch.setattr("nro.bin.bidsify.ask", lambda *_: pytest.fail("Site selector prompted"))
    assert select_source(config, "demo", server="lucas", flywheel_project=None) == (
        "lucas",
        "test/demo",
    )
    answers = iter(["test/demo", "lucas/test/demo"])
    monkeypatch.setattr("nro.bin.bidsify.ask", lambda *_: next(answers))
    assert select_source(config, "demo", server=None, flywheel_project=None) == (
        "lucas",
        "test/demo",
    )


def test_multisite_configuration_loads(tmp_path):
    import yaml

    from nro.configuration.site import definitions_root

    config = yaml.safe_load((definitions_root() / "bidsify/main.yml").read_text())
    config["project_sources"] = {
        "demo": [
            {"server": "cni", "project": "test/demo"},
            {"server": "lucas", "project": "test/demo"},
        ]
    }
    path = tmp_path / "profile.yml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["project_sources"] == config["project_sources"]


def test_legacy_intake_does_not_hide_or_conflict_with_multisession(tmp_path):
    (tmp_path / "climblab/sub-ex123/func").mkdir(parents=True)
    assert discover(tmp_path).existing_hidden == 0
    target = tmp_path / "climblab_multisession/sub-t20/ses-ex123"
    target.mkdir(parents=True)
    assert discover(tmp_path).existing_hidden == 1
    (row,) = discover(tmp_path, rebidsify=True).rows
    assert not row["mapping_ambiguous"]
    assert [match.path for match in row["existing_bids"]] == [target]


def test_rebidsify_cannot_redirect_external_session(tmp_path, monkeypatch):
    from nro.bidsify.store import IngestionStore
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    (root / "original/sub-t20/ses-ex123").mkdir(parents=True)
    registry = Registry.for_project("", bids_root=root)
    config = load_config()
    config["session_rules"] = rules()
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli, "FlywheelSource", lambda _: SimpleNamespace(sessions=lambda: [remote()])
    )
    monkeypatch.setattr(cli, "ask", lambda *_: "all")
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--server",
                "cni",
                "-P",
                "different",
                "--rebidsify",
                "--no-submit",
            ]
        )
    assert error.value.code == 1
    assert IngestionStore(registry).rows() == []


@pytest.mark.parametrize("participant", ["t20", "different"])
def test_external_rebidsify_retains_subject_and_session(tmp_path, monkeypatch, participant):
    from nro.bidsify.store import IngestionStore
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    target = root / "demo/sub-t20/ses-ex123"
    target.mkdir(parents=True)
    (target / "untouched").write_text("existing data")
    registry = Registry.for_project("", bids_root=root)
    config = load_config()
    config["staging"] = str(tmp_path / "staging")
    config["session_rules"] = rules()
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli, "FlywheelSource", lambda _: SimpleNamespace(sessions=lambda: [remote()])
    )
    answers = iter(["all", participant, "y"])
    monkeypatch.setattr(cli, "ask", lambda *_: next(answers))
    args = ["--server", "cni", "-P", "demo", "--rebidsify", "--no-submit"]
    if participant == "t20":
        cli.main(args)
        (row,) = IngestionStore(registry).rows()
        assert (row["project"], row["participant"], row["session"]) == ("demo", "t20", "ex123")
        assert row["replace"] and row["state"] == "queued"
    else:
        with pytest.raises(SystemExit):
            cli.main(args)
        assert not IngestionStore(registry).rows()
    assert (target / "untouched").read_text() == "existing data"


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        {"demo": {}},
        {"demo": []},
        {"demo": [{"server": "unknown", "project": "test/demo"}]},
        {"demo": [{"server": "cni", "project": "other/lab"}]},
        {"../bad": [{"server": "cni", "project": "test/demo"}]},
        {"demo": [{"server": "cni", "project": "test/demo"}] * 2},
    ],
)
def test_invalid_project_sources_are_rejected(tmp_path, mapping):
    import yaml

    from nro.configuration.site import definitions_root

    config = yaml.safe_load((definitions_root() / "bidsify/main.yml").read_text())
    config["project_sources"] = mapping
    path = tmp_path / "profile.yml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("mapped", [True, False])
def test_cli_only_contacts_selected_flywheel_project(tmp_path, monkeypatch, capsys, mapped):
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    registry = Registry.for_project("", bids_root=root)
    config = load_config()
    config["servers"]["cni"]["projects"] = ["test/demo", "another/lab"]
    if mapped:
        config["project_sources"] = {"destination": [{"server": "cni", "project": "test/demo"}]}
    calls = []

    def lookup(project):
        calls.append(project)
        assert project == "test/demo"
        session = SimpleNamespace(id="selected-session", label="123", subject=None)
        return SimpleNamespace(sessions=SimpleNamespace(iter=lambda: iter([session])))

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    site_values, site_sources = cli.settings()
    monkeypatch.setattr(
        cli,
        "settings",
        lambda: (
            {
                **site_values,
                "flywheel_server": "cni",
                "flywheel_project": "test/demo",
            },
            site_sources,
        ),
    )
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    monkeypatch.setattr(
        cli,
        "FlywheelSource",
        lambda profile: FlywheelSource(profile, client=SimpleNamespace(lookup=lookup)),
    )

    def answer(prompt, *args):
        assert prompt == "Select session numbers, or all"
        raise EOFError

    monkeypatch.setattr(cli, "ask", answer)
    argv = ["-P", "destination", "--no-submit"]
    cli.main(argv)
    assert calls == ["test/demo"]
    output = capsys.readouterr().out
    assert "selected-session" in output and "another/lab" not in output
    assert "BIDS project: destination" in output


def test_bidsify_parser_accepts_short_flywheel_selectors() -> None:
    from nro.bin.bidsify import build_parser

    args = build_parser().parse_args(["-f", "cni", "-F", "group/project"])

    assert args.flywheel_server == "cni"
    assert args.flywheel_project == "group/project"


def test_cli_omits_existing_before_prompting_and_writes_no_records(tmp_path, monkeypatch, capsys):
    from nro.bidsify.store import IngestionStore
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    (root / "demo/sub-t20/ses-ex123").mkdir(parents=True)
    registry = Registry.for_project("", bids_root=root)
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    profile = load_config()
    profile["session_rules"] = rules()
    monkeypatch.setattr(cli, "load_config", lambda _: profile)
    monkeypatch.setattr(
        cli, "FlywheelSource", lambda _: SimpleNamespace(sessions=lambda: [remote()])
    )
    monkeypatch.setattr(
        cli, "ask", lambda *a: pytest.fail("An existing external session was offered")
    )
    cli.main(["--server", "cni", "-P", "demo", "--no-submit"])
    assert "Omitted 1 sessions already bidsified" in capsys.readouterr().out
    assert IngestionStore(registry).rows() == []


def test_cli_rebidsify_lists_existing_before_mapping(tmp_path, monkeypatch, capsys):
    from nro.bin import bidsify as cli
    from nro.orchestration.registry import Registry

    root = tmp_path / "bids"
    target = root / "other/sub-t20/ses-ex123"
    target.mkdir(parents=True)
    registry = Registry.for_project("", bids_root=root)
    monkeypatch.setattr(cli.Registry, "for_project", lambda *a, **k: registry)
    profile = load_config()
    profile["session_rules"] = rules()
    monkeypatch.setattr(cli, "load_config", lambda _: profile)
    monkeypatch.setattr(
        cli, "FlywheelSource", lambda _: SimpleNamespace(sessions=lambda: [remote()])
    )

    def cancel(prompt, *args):
        assert prompt == "Select session numbers, or all"
        raise EOFError

    monkeypatch.setattr(cli, "ask", cancel)
    cli.main(["--server", "cni", "-P", "demo", "--rebidsify"])
    output = capsys.readouterr().out
    assert "already bidsified" in output and str(target) in output
