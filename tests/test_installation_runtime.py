from __future__ import annotations

import sqlite3

import pytest

from nro.site.runtime_requirements import MINIMUM_SQLITE, require_current_runtime


def test_supported_sqlite_runtime_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", MINIMUM_SQLITE)

    require_current_runtime()


def test_old_sqlite_runtime_reports_the_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.34.1")

    with pytest.raises(RuntimeError, match=r"requires SQLite 3\.35\.0.*provides 3\.34\.1"):
        require_current_runtime()
