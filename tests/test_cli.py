from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import nro.cli as cli
from nro.bin.run import build_parser as run_parser

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


def test_dispatcher_forwards_arguments_and_installed_program_name(monkeypatch) -> None:
    calls = []

    def command_main(argv, *, prog):
        calls.append((argv, prog))

    monkeypatch.setattr(cli, "_command_main", lambda command: command_main)

    cli.main(["run", "-p", "t12"])

    assert calls == [(["-p", "t12"], "nro run")]


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

    assert metadata["project"]["version"] == "0.1.0"
    assert metadata["project"]["scripts"] == {"nro": "nro.cli:main"}
