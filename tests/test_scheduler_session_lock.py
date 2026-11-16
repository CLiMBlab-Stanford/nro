"""The scheduler session holds SQLite's file lock for its whole lifetime."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nro.orchestration.registry import Registry


def test_scheduler_session_excludes_other_sqlite_connections(tmp_path: Path) -> None:
    registry = Registry.for_project("demo", bids_root=tmp_path / "bids")
    registry.initialize()
    with registry.scheduler_session():
        with registry.connection(write=True) as database:
            database.execute("INSERT INTO metadata(key,value) VALUES ('session-lock','1')")
        other = sqlite3.connect(registry.paths.database, timeout=0.1)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("SELECT value FROM metadata WHERE key='session-lock'").fetchone()
        finally:
            other.close()
    other = sqlite3.connect(registry.paths.database, timeout=0.1)
    try:
        row = other.execute("SELECT value FROM metadata WHERE key='session-lock'").fetchone()
        assert row == ("1",)
        assert other.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        other.close()
