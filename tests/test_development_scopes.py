import subprocess
from pathlib import Path

import pytest

from nro.bin.dev import build_parser
from nro.engine.development import changed_paths, load_scopes, select_tests

ROOT = Path(__file__).resolve().parents[1]


def test_developer_command_requires_the_test_operation() -> None:
    args = build_parser().parse_args(["test", "--sphere", "anat", "--dry-run"])
    assert args.operation == "test"
    assert args.sphere == ["anat"]


def test_developer_command_exposes_generated_registry_schemas() -> None:
    args = build_parser().parse_args(["schema", "show", "--family", "scientific"])
    assert args.operation == "schema"
    assert args.schema_operation == "show"
    assert args.family == "scientific"


def test_module_change_selects_its_sphere_and_downstream_boundaries() -> None:
    scopes = load_scopes(ROOT / "development/test-scopes.toml")
    selection = select_tests(scopes, ("nro/modules/microparcellation/module.py",))

    assert {"microparcellation", "networks"} <= set(selection.spheres)
    assert "tests/test_microparcellation_targets.py" in selection.tests
    assert "tests/test_networks_module.py" in selection.tests
    assert not selection.full


def test_unmapped_change_fails_closed_to_full_validation() -> None:
    scopes = load_scopes(ROOT / "development/test-scopes.toml")
    selection = select_tests(scopes, ("unexpected.file",))

    assert selection.unmapped == ("unexpected.file",)
    assert selection.full


def test_changed_paths_keeps_both_sides_of_cross_sphere_rename(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    source = repository / "nro" / "engine" / "old.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    destination = repository / "nro" / "modules" / "anat" / "new.py"
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)

    paths = changed_paths(repository)

    assert "nro/engine/old.py" in paths
    assert "nro/modules/anat/new.py" in paths


def test_scope_loader_rejects_missing_test_targets(tmp_path) -> None:
    development = tmp_path / "development"
    development.mkdir()
    path = development / "test-scopes.toml"
    path.write_text(
        """\
version = 1
[scope.example]
paths = ["nro/example.py"]
tests = ["tests/missing.py"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing test files"):
        load_scopes(path)


def test_every_tracked_python_and_policy_file_has_a_scope() -> None:
    scopes = load_scopes(ROOT / "development/test-scopes.toml")
    tracked = [
        str(path.relative_to(ROOT))
        for base in (ROOT / "nro", ROOT / "tests")
        for path in base.rglob("*.py")
    ]
    tracked.extend(("install", "pyproject.toml", "AGENTS.md", "README.md"))

    selection = select_tests(scopes, tracked)
    assert selection.unmapped == ()
