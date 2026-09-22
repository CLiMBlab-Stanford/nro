from __future__ import annotations

import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

import nro.cli as cli
from nro.bin.run import build_parser as run_parser
from nro.configuration.site import ENVIRONMENT_KEYS

ROOT = Path(__file__).parents[1]


def test_installed_commands_are_exactly_the_bin_executables() -> None:
    expected = tuple(
        sorted(
            path.stem
            for path in (ROOT / "nro" / "bin").glob("*.py")
            if path.stem != "__init__" and not path.stem.startswith("_")
        )
    )

    assert cli.available_commands() == expected
    assert "qc" in expected
    assert set(cli.COMMAND_HELP) == set(expected)
    assert cli.CENTRAL_ONLY_COMMANDS <= set(expected)


def test_dispatcher_forwards_arguments_and_installed_program_name(monkeypatch) -> None:
    calls = []

    def command_main(argv, *, prog):
        calls.append((argv, prog))

    monkeypatch.setattr(cli, "_command_main", lambda command: command_main)

    cli.main(["run", "-p", "t12"])

    assert calls == [(["-p", "t12"], "nro run")]


def test_version_reports_installed_distribution_without_site_loading(monkeypatch, capsys) -> None:
    import nro.configuration.site as site

    monkeypatch.setattr(
        site,
        "installation_record",
        lambda: pytest.fail("--version must not load installation state"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"nro {version('nro')}"


def test_version_reports_verified_application_source(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.3.4"\n')
    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(tmp_path))

    assert cli.package_version() == "2.3.4"


def test_help_lists_task_guides_and_installed_commands(capsys) -> None:
    from nro.bin import help as help_command

    help_command.main([])

    output = capsys.readouterr().out
    assert "Task guides:" in output
    assert "definitions" in output
    assert "Installed commands:" in output
    assert "run" in output


def test_help_shows_task_guidance(capsys) -> None:
    from nro.bin import help as help_command

    help_command.main(["definitions"])

    output = capsys.readouterr().out
    assert "nro create config MODULE/ID" in output
    assert "nro definitions create [PATH]" in output


def test_help_delegates_command_reference(monkeypatch) -> None:
    import nro.cli as cli_module
    from nro.bin import help as help_command

    calls = []

    def command_main(argv, *, prog):
        calls.append((argv, prog))
        raise SystemExit(0)

    monkeypatch.setattr(cli_module, "available_commands", lambda: ("run",))
    monkeypatch.setattr(cli_module, "_command_main", lambda command: command_main)

    help_command.main(["--command", "run"])

    assert calls == [(["--help"], "nro run")]


def test_development_installation_restricts_only_central_maintenance(monkeypatch, capsys) -> None:
    import nro.configuration.site as site

    calls = []

    def command_main(argv, *, prog):
        calls.append((argv, prog))

    monkeypatch.setattr(site, "installation_record", lambda: {"mode": "branch"})
    monkeypatch.setattr(cli, "_command_main", lambda command: command_main)

    cli.main(["status"])
    assert calls == [([], "nro status")]

    with pytest.raises(SystemExit) as error:
        cli.main(["release"])

    assert error.value.code == 2
    assert "release requires the central/main installation" in capsys.readouterr().err


def test_dispatcher_reports_scheduler_failure_without_internal_traceback(monkeypatch) -> None:
    from nro.orchestration.scheduler_client import SchedulerError

    def command_main(_argv, *, prog):
        raise SchedulerError(f"{prog}: scheduler unavailable")

    monkeypatch.setattr(cli, "_command_main", lambda command: command_main)

    with pytest.raises(SystemExit, match="nro status: scheduler unavailable"):
        cli.main(["status"])


def test_installed_command_help_uses_subcommand_syntax(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--help"])

    assert error.value.code == 0
    assert capsys.readouterr().out.startswith("usage: nro run ")


def test_installed_qc_help_preserves_nested_qctype_syntax(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["qc", "registration", "--help"])

    assert error.value.code == 0
    assert capsys.readouterr().out.startswith("usage: nro qc registration ")


def test_python_module_parser_name_remains_available() -> None:
    assert run_parser().prog == "nro.bin.run"


def test_pyproject_defines_only_the_single_nro_console_script() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert metadata["project"]["scripts"] == {"nro": "nro.cli:main"}


def test_public_commands_cannot_override_the_site_bids_root() -> None:
    sources = list((ROOT / "nro" / "bin").glob("*.py"))
    sources.append(ROOT / "nro" / "configuration" / "authoring.py")

    assert all("--bids-root" not in path.read_text() for path in sources)
    assert "NRO_BIDS_PATH" not in ENVIRONMENT_KEYS
