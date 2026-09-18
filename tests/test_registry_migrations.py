"""Registry schemas are derived from immutable baselines and checked migrations."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

from nro.orchestration.migrations import (
    AddColumn,
    Column,
    MapValues,
    Migration,
    RegistrySchema,
    migrate_database,
)
from nro.orchestration.migrations.core import load_chain
from nro.orchestration.migrations.tools import render, validate_families


def _example_schema(*operations) -> RegistrySchema:
    return RegistrySchema(
        name="test",
        application_id=731,
        baseline_version=1,
        baseline_sql="CREATE TABLE items (id INTEGER PRIMARY KEY, state TEXT NOT NULL);",
        migrations=(Migration(2, tuple(operations), "Exercise a checked transition"),),
    )


def _database(path: Path, *, state: str = "old") -> None:
    with sqlite3.connect(path) as database:
        database.execute("PRAGMA application_id=731")
        database.execute("PRAGMA user_version=1")
        database.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
        database.execute("INSERT INTO items VALUES (1, ?)", (state,))


def test_repository_registry_chains_match_their_immutable_baselines() -> None:
    validate_families()
    assert "CREATE TABLE work_items" in render("scheduler")
    assert "CREATE TABLE work_items" in render("scientific")


def test_migration_preserves_rows_and_matches_a_fresh_generated_schema(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    _database(path)
    schema = _example_schema(
        AddColumn("items", Column("generation", "INTEGER", nullable=False, default=0)),
        MapValues("items", "state", {"old": "ready"}),
    )

    backup = migrate_database(path, schema)

    assert backup is not None and backup.is_file()
    with sqlite3.connect(path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 2
        assert database.execute("SELECT state,generation FROM items").fetchone() == ("ready", 0)
        fresh = schema.build()
        try:
            assert schema.signature(database) == schema.signature(fresh)
        finally:
            fresh.close()
    with sqlite3.connect(backup) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 1


def test_migration_snapshot_includes_committed_wal_content(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    writer = sqlite3.connect(path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA application_id=731")
        writer.execute("PRAGMA user_version=1")
        writer.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
        writer.execute("INSERT INTO items VALUES (1, 'old')")
        writer.commit()

        backup = migrate_database(
            path,
            _example_schema(MapValues("items", "state", {"old": "ready"})),
        )
    finally:
        writer.close()

    assert backup is not None
    with sqlite3.connect(path) as database:
        assert database.execute("SELECT state FROM items").fetchone()[0] == "ready"
    with sqlite3.connect(backup) as database:
        assert database.execute("SELECT state FROM items").fetchone()[0] == "old"


def test_failed_migration_does_not_replace_the_registry(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite3"
    _database(path, state="unexpected")
    before = path.read_bytes()
    schema = _example_schema(MapValues("items", "state", {"old": "ready"}))

    with pytest.raises(ValueError, match="does not map every"):
        migrate_database(path, schema)

    assert path.read_bytes() == before
    assert not tuple(tmp_path.glob("registry-before-schema-*.sqlite3"))


def test_migration_chain_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="expected destination 2"):
        RegistrySchema(
            name="test",
            application_id=731,
            baseline_version=1,
            baseline_sql="CREATE TABLE items (id INTEGER);",
            migrations=(Migration(3, (), "Skipped a version"),),
        )


def test_required_added_column_needs_a_default() -> None:
    with pytest.raises(ValueError, match="needs a default"):
        Column("required", "TEXT", nullable=False).sql()


def test_schema_literals_reject_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        Column("value", "REAL", default=float("nan")).sql()


def test_value_mapping_is_immutable_after_declaration() -> None:
    values = {"old": "ready"}
    operation = MapValues("items", "state", values)
    values["old"] = "changed"

    assert operation.mapping["old"] == "ready"
    with pytest.raises(TypeError):
        operation.mapping["old"] = "changed"  # type: ignore[index]


def test_migration_package_rejects_unversioned_python_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "example_migrations"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop("example_migrations", None)

    with pytest.raises(ValueError, match="must be named vNNNN.py"):
        load_chain("example_migrations", baseline_version=1)
