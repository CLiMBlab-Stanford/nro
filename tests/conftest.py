"""Keep unit tests independent of the checkout's live installation settings."""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from nro.configuration import site
from nro.engine import site_setup
from nro.orchestration import scheduler_implementation


@pytest.fixture(scope="session")
def definitions_fixture(tmp_path_factory):
    from nro.configuration.definition_migrations import refresh_manifest
    from nro.configuration.definitions import create_store
    from nro.configuration.site import read_site_definition, write_site_definition

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
def isolated_installation(tmp_path, tmp_path_factory, monkeypatch, definitions_fixture):
    os.environ.pop("NRO_SOURCE_MARKUP", None)
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
    monkeypatch.setattr(site, "LAB", root)
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
    yield
    os.environ.pop("NRO_SOURCE_MARKUP", None)
