"""Keep unit tests independent of the checkout's live installation settings."""

import os
import shutil
import sqlite3
from pathlib import Path

import pytest
import yaml

from nro.orchestration import scheduler_implementation
from nro.site import configuration as site
from nro.site import setup as site_setup

PROCESS_VARIABLE_PREFIXES = ("NRO_", "SLURM_")
PROCESS_VARIABLES = {"CUDA_VISIBLE_DEVICES"}


@pytest.fixture(scope="session")
def definitions_fixture(tmp_path_factory):
    with pytest.MonkeyPatch.context() as session_patch:
        # Session fixtures run before the per-test isolation fixtures, so the
        # checkout's own installation record must be hidden here as well.
        session_patch.setattr(site, "CHECKOUT", tmp_path_factory.mktemp("checkout"))
        for variable in tuple(os.environ):
            if variable.startswith(PROCESS_VARIABLE_PREFIXES) or variable in PROCESS_VARIABLES:
                session_patch.delenv(variable)
        return _definitions_fixture(tmp_path_factory)


def _definitions_fixture(tmp_path_factory):
    from nro.definitions.migrations import refresh_manifest
    from nro.definitions.repository import create_store
    from nro.site.configuration import read_site_definition, write_site_definition

    root = create_store(tmp_path_factory.mktemp("definitions") / "store")
    shutil.copytree(Path(__file__).parent / "fixtures/models", root / "models", dirs_exist_ok=True)
    task = root / "events/example"
    task.mkdir()
    (task / "main.tsv").write_text("onset\tduration\ttrial_type\n0\t1\tA\n")
    (task / "index.yml").write_text(
        yaml.safe_dump(
            {
                "tasks": ["example"],
                "files": {
                    variant: {"path": "example/main.tsv", "source_names": []}
                    for variant in ("main", "same")
                },
            }
        )
    )
    # Test fixtures populate synthetic definitions directly. Record that
    # controlled setup before exercising the public authoring paths.
    refresh_manifest(root)
    settings, bidsify = read_site_definition(root)
    bidsify["servers"] = {
        "cni": {
            "host": "cni.example.org",
            "projects": ["test/demo"],
        },
        "lucas": {
            "host": "lucas.example.org",
            "projects": ["test/demo"],
        },
    }
    write_site_definition(root, settings, bidsify=bidsify)
    from nro.bidsify.credentials import store_key

    store_key(root, "cni", "test-cni-key", host="cni.example.org")
    store_key(root, "lucas", "test-lucas-key", host="lucas.example.org")
    return root


@pytest.fixture(autouse=True)
def historical_sqlite_variable_limit(monkeypatch):
    """Apply the 999-parameter limit of older SQLite builds to every connection.

    Current builds allow 32766 bound parameters, so a query that binds one
    parameter per work item passes small tests and fails only on large sites.
    """
    original = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Start each test without inherited nro, Slurm, or Git state, and restore it afterward.

    Production entry points such as the scheduler service write process
    variables directly, so the environment is snapshotted and restored instead
    of relying only on monkeypatch.
    """
    saved = dict(os.environ)
    for variable in tuple(os.environ):
        if variable.startswith(PROCESS_VARIABLE_PREFIXES) or variable in PROCESS_VARIABLES:
            monkeypatch.delenv(variable)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    yield
    monkeypatch.undo()
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def isolated_installation(
    tmp_path, tmp_path_factory, monkeypatch, definitions_fixture, isolated_environment
):
    root = tmp_path_factory.mktemp("installation-settings")
    config = root / "site.toml"
    config.write_text(
        f'bids = "{tmp_path / "bids"}"\n'
        f'registry = "{tmp_path / "registry"}"\n'
        f'work = "{tmp_path / "work"}"\n'
    )
    monkeypatch.setenv("NRO_SITE_CONFIG", str(config))
    for variable in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(variable, raising=False)
    lab = str(site.LAB)
    monkeypatch.setattr(site, "LAB", root)
    for key, value in tuple(site.DEFAULTS.items()):
        if isinstance(value, str) and value.startswith(lab + "/"):
            monkeypatch.setitem(site.DEFAULTS, key, str(root / value.removeprefix(lab + "/")))
    monkeypatch.setitem(site.DEFAULTS, "binds", [])
    monkeypatch.setattr(site, "CHECKOUT", root)
    monkeypatch.setattr(site, "installation_record", lambda: {})
    monkeypatch.setattr(site_setup, "installation_record", lambda: {})
    installed_record = scheduler_implementation.installation_record
    monkeypatch.setattr(
        scheduler_implementation,
        "installation_record",
        lambda selected=None: {} if selected is None else installed_record(selected),
    )
    monkeypatch.setitem(site.DEFAULTS, "definitions", str(definitions_fixture))
    from nro.definitions.store import _scientific_defaults

    _scientific_defaults.cache_clear()
    yield
    _scientific_defaults.cache_clear()
