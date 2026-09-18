"""Derive and apply registry schemas through a restricted migration language."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KINDS = frozenset({"INTEGER", "REAL", "TEXT", "BLOB"})
_MISSING = object()


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid SQLite identifier: {value!r}")
    return f'"{value}"'


def _literal(value: object) -> str:
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "1" if value else "0"
    if type(value) is int:
        return repr(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("SQLite schema literals must be finite")
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, bytes):
        return f"X'{value.hex()}'"
    raise TypeError(f"Unsupported SQLite default value: {value!r}")


@dataclass(frozen=True)
class Column:
    """Describe a column that can be added without rebuilding its table."""

    name: str
    kind: str
    nullable: bool = True
    default: object = _MISSING

    def sql(self) -> str:
        """Render a checked SQLite column declaration."""
        name = _identifier(self.name)
        kind = self.kind.upper()
        if kind not in _KINDS:
            raise ValueError(f"Unsupported SQLite column kind: {self.kind!r}")
        pieces = [name, kind]
        if not self.nullable:
            pieces.append("NOT NULL")
        if self.default is not _MISSING:
            pieces.extend(("DEFAULT", _literal(self.default)))
        elif not self.nullable:
            raise ValueError("A required added column needs a default for existing rows")
        return " ".join(pieces)


class Operation(Protocol):
    """A checked schema or data transformation supported by the migration engine."""

    def apply(self, database: sqlite3.Connection) -> None:
        """Apply the transformation to one transaction."""


@dataclass(frozen=True)
class AddColumn:
    """Add a typed column whose declaration is valid for populated tables."""

    table: str
    column: Column

    def apply(self, database: sqlite3.Connection) -> None:
        """Add the column to its declared table."""
        database.execute(f"ALTER TABLE {_identifier(self.table)} ADD COLUMN {self.column.sql()}")


@dataclass(frozen=True)
class RenameColumn:
    """Rename a column without changing its values or declared type."""

    table: str
    old: str
    new: str

    def apply(self, database: sqlite3.Connection) -> None:
        """Rename the column in its declared table."""
        database.execute(
            f"ALTER TABLE {_identifier(self.table)} RENAME COLUMN "
            f"{_identifier(self.old)} TO {_identifier(self.new)}"
        )


@dataclass(frozen=True)
class RenameTable:
    """Rename a table while preserving its rows."""

    old: str
    new: str

    def apply(self, database: sqlite3.Connection) -> None:
        """Rename the declared table."""
        database.execute(f"ALTER TABLE {_identifier(self.old)} RENAME TO {_identifier(self.new)}")


@dataclass(frozen=True)
class CreateIndex:
    """Create a checked ordinary or unique index over named columns."""

    name: str
    table: str
    columns: tuple[str, ...]
    unique: bool = False

    def apply(self, database: sqlite3.Connection) -> None:
        """Create the declared index."""
        if not self.columns:
            raise ValueError("An index needs at least one column")
        unique = "UNIQUE " if self.unique else ""
        columns = ",".join(_identifier(column) for column in self.columns)
        database.execute(
            f"CREATE {unique}INDEX {_identifier(self.name)} ON {_identifier(self.table)}({columns})"
        )


@dataclass(frozen=True)
class DropIndex:
    """Remove a named index without changing table data."""

    name: str

    def apply(self, database: sqlite3.Connection) -> None:
        """Remove the declared index."""
        database.execute(f"DROP INDEX {_identifier(self.name)}")


@dataclass(frozen=True)
class DropColumn:
    """Remove a column only when its contents are explicitly reconstructible."""

    table: str
    column: str
    reconstructible: bool = False

    def apply(self, database: sqlite3.Connection) -> None:
        """Remove the column after checking its data-loss declaration."""
        if not self.reconstructible:
            raise ValueError("Dropping stored data requires reconstructible=True")
        database.execute(
            f"ALTER TABLE {_identifier(self.table)} DROP COLUMN {_identifier(self.column)}"
        )


@dataclass(frozen=True)
class MapValues:
    """Exhaustively replace the existing values of one column."""

    table: str
    column: str
    mapping: Mapping[object, object]
    allow_null: bool = False

    def __post_init__(self) -> None:
        """Freeze and validate the declared mapping at construction time."""
        copied = dict(self.mapping)
        for old, new in copied.items():
            _literal(old)
            _literal(new)
        object.__setattr__(self, "mapping", MappingProxyType(copied))

    def apply(self, database: sqlite3.Connection) -> None:
        """Map every present non-exempt value or reject the migration."""
        table = _identifier(self.table)
        column = _identifier(self.column)
        present = {row[0] for row in database.execute(f"SELECT DISTINCT {column} FROM {table}")}
        uncovered = present - set(self.mapping)
        if self.allow_null:
            uncovered.discard(None)
        if uncovered:
            raise ValueError(
                f"Migration does not map every {self.table}.{self.column} value: "
                + ", ".join(sorted(map(repr, uncovered)))
            )
        for old, new in self.mapping.items():
            database.execute(
                f"UPDATE {table} SET {column}=? WHERE {column} IS ?",
                (new, old),
            )


@dataclass(frozen=True)
class Migration:
    """Transform one registry schema version into its immediate successor."""

    destination: int
    operations: tuple[Operation, ...]
    summary: str

    def apply(self, database: sqlite3.Connection) -> None:
        """Apply each operation and advance the database schema marker."""
        for operation in self.operations:
            operation.apply(database)
        database.execute(f"PRAGMA user_version={self.destination}")


def load_chain(package: str, *, baseline_version: int) -> tuple[Migration, ...]:
    """Load one immutable migration module per destination version."""
    migrations = []
    for resource in sorted(files(package).iterdir(), key=lambda item: item.name):
        if resource.name == "__init__.py" or not resource.name.endswith(".py"):
            continue
        if not re.fullmatch(r"v[0-9]{4}\.py", resource.name):
            raise ValueError(
                f"Unexpected Python file in {package}: {resource.name}; "
                "migration files must be named vNNNN.py"
            )
        destination = int(resource.name[1:5])
        module = import_module(f"{package}.{resource.name[:-3]}")
        migration = getattr(module, "MIGRATION", None)
        if not isinstance(migration, Migration):
            raise ValueError(f"{package}.{resource.name[:-3]} must define one MIGRATION")
        if migration.destination != destination:
            raise ValueError(
                f"Migration filename {resource.name} does not match destination "
                f"{migration.destination}"
            )
        migrations.append(migration)
    expected = list(range(baseline_version + 1, baseline_version + 1 + len(migrations)))
    destinations = [migration.destination for migration in migrations]
    if destinations != expected:
        raise ValueError(
            f"{package} migration files must be contiguous after {baseline_version}: {destinations}"
        )
    return tuple(migrations)


def _schema_rows(database: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (row[0], row[1], row[2], " ".join(row[3].split()))
        for row in database.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_schema "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name"
        )
    )


def _schema_sql(database: sqlite3.Connection) -> str:
    rows = database.execute(
        "SELECT type,name,sql FROM sqlite_schema "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 1 ELSE 2 END,name"
    )
    return "\n\n".join(f"{row[2].rstrip(';')};" for row in rows) + "\n"


def schema_fingerprint(sql: str) -> str:
    """Hash the normalized SQLite structure created by a schema script."""
    database = sqlite3.connect(":memory:")
    try:
        database.executescript(sql)
        payload = json.dumps(_schema_rows(database), separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
    finally:
        database.close()


@dataclass(frozen=True)
class RegistrySchema:
    """Define one registry family from an immutable baseline and ordered migrations."""

    name: str
    application_id: int
    baseline_version: int
    baseline_sql: str
    migrations: tuple[Migration, ...] = ()

    def __post_init__(self) -> None:
        expected = self.baseline_version + 1
        for migration in self.migrations:
            if migration.destination != expected:
                raise ValueError(
                    f"{self.name} migration chain expected destination {expected}, "
                    f"found {migration.destination}"
                )
            if not migration.summary.strip():
                raise ValueError(f"{self.name} migration {expected} needs a summary")
            expected += 1

    @property
    def version(self) -> int:
        """Return the current schema version derived from the migration chain."""
        return self.baseline_version + len(self.migrations)

    def supports(self, stored_version: int) -> bool:
        """Report whether the chain can migrate a stored version in place."""
        return self.baseline_version <= stored_version <= self.version

    def migrations_after(self, version: int) -> tuple[Migration, ...]:
        """Return the contiguous suffix needed to reach the current version."""
        if not self.supports(version):
            raise ValueError(
                f"{self.name} schema {version} is outside the supported migration range "
                f"{self.baseline_version}..{self.version}"
            )
        offset = version - self.baseline_version
        return self.migrations[offset:]

    def build(self, *, version: int | None = None) -> sqlite3.Connection:
        """Construct an in-memory schema at a supported version for inspection or tests."""
        target = self.version if version is None else version
        if not self.supports(target):
            raise ValueError(f"Unsupported {self.name} schema target: {target}")
        database = sqlite3.connect(":memory:")
        try:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute(f"PRAGMA application_id={self.application_id}")
            database.executescript(self.baseline_sql)
            database.execute(f"PRAGMA user_version={self.baseline_version}")
            for migration in self.migrations_after(self.baseline_version):
                if migration.destination > target:
                    break
                migration.apply(database)
            application_id = int(database.execute("PRAGMA application_id").fetchone()[0])
            actual_version = int(database.execute("PRAGMA user_version").fetchone()[0])
            if application_id != self.application_id or actual_version != target:
                raise ValueError(f"Could not derive {self.name} schema {target}")
            if database.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError(f"Generated {self.name} schema has invalid foreign keys")
            return database
        except BaseException:
            database.close()
            raise

    def sql(self, *, version: int | None = None) -> str:
        """Render the generated schema without writing a source file."""
        database = self.build(version=version)
        try:
            return _schema_sql(database)
        finally:
            database.close()

    def signature(self, database: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
        """Return a normalized structural signature for exact comparisons."""
        return _schema_rows(database)

    def validate(
        self, database: sqlite3.Connection, *, expected_version: int | None = None
    ) -> None:
        """Check identity, version, structure, foreign keys, and SQLite integrity."""
        target = self.version if expected_version is None else expected_version
        application_id = int(database.execute("PRAGMA application_id").fetchone()[0])
        version = int(database.execute("PRAGMA user_version").fetchone()[0])
        if application_id != self.application_id:
            raise ValueError(f"Database is not an nro {self.name} registry")
        if version != target:
            raise ValueError(f"Expected {self.name} schema {target}; found {version}")
        expected = self.build(version=target)
        try:
            if self.signature(database) != self.signature(expected):
                raise ValueError(f"{self.name} registry structure does not match schema {target}")
        finally:
            expected.close()
        violations = database.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise ValueError(f"{self.name} registry has foreign-key violations: {violations[:3]}")
        result = database.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ValueError(f"{self.name} registry failed integrity check: {result}")


def migrate_database(path: Path, schema: RegistrySchema) -> Path | None:
    """Migrate a registry copy and atomically replace the original after validation.

    The caller must hold the registry's cross-host maintenance lock. The returned
    path is a durable copy of the pre-migration database.
    """
    path = Path(path)
    source_uri = path.resolve().as_uri() + "?mode=rw"
    with sqlite3.connect(source_uri, uri=True) as source:
        application_id = int(source.execute("PRAGMA application_id").fetchone()[0])
        stored = int(source.execute("PRAGMA user_version").fetchone()[0])
    if application_id != schema.application_id:
        raise ValueError(f"Database is not an nro {schema.name} registry: {path}")
    if stored == schema.version:
        return None
    migrations = schema.migrations_after(stored)
    temporary = path.with_name(f".{path.name}.migrating-{uuid.uuid4().hex}")
    backup = path.with_name(f"{path.stem}-before-schema-{schema.version}{path.suffix}")
    if backup.exists():
        backup = path.with_name(
            f"{path.stem}-before-schema-{schema.version}-{uuid.uuid4().hex[:8]}{path.suffix}"
        )
    activated = False
    backup_created = False
    try:
        # SQLite's backup interface includes committed data that may still live
        # in a write-ahead log. Copying only the main file is not a complete
        # database snapshot in WAL mode.
        with sqlite3.connect(source_uri, uri=True) as source:
            if str(source.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal":
                checkpoint = source.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is None or int(checkpoint[0]) != 0:
                    raise RuntimeError(
                        f"Could not checkpoint {schema.name} registry before migration: {path}"
                    )
            with sqlite3.connect(backup) as snapshot:
                source.backup(snapshot)
        backup.chmod(path.stat().st_mode & 0o777)
        backup_created = True
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
        shutil.copy2(backup, temporary)
        database = sqlite3.connect(temporary)
        try:
            database.execute("PRAGMA foreign_keys=ON")
            database.execute("BEGIN IMMEDIATE")
            for migration in migrations:
                migration.apply(database)
            tables = {
                row[0]
                for row in database.execute("SELECT name FROM sqlite_schema WHERE type='table'")
            }
            if "metadata" in tables:
                database.execute(
                    "UPDATE metadata SET value=? WHERE key='schema_version'",
                    (str(schema.version),),
                )
            schema.validate(database)
            database.commit()
        except BaseException:
            database.rollback()
            raise
        finally:
            database.close()
        temporary.chmod(path.stat().st_mode & 0o777)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        activated = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return backup
    except BaseException:
        if backup_created and not activated:
            backup.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
