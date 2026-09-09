"""Keep unit tests independent of the checkout's live installation settings."""

import shutil
from pathlib import Path

import pytest
import yaml

from nro.configuration import site
from nro.engine import site_setup


@pytest.fixture(scope="session")
def definitions_fixture(tmp_path_factory):
    from nro.configuration.definitions import create_store

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
    profile = root / "bidsify/main.yml"
    value = yaml.safe_load(profile.read_text())
    value["servers"] = {
        "cni": {
            "host": "cni.example.org",
            "credential_env": "TEST_CNI_KEY",
            "projects": ["test/demo"],
        },
        "lucas": {
            "host": "lucas.example.org",
            "credential_env": "TEST_LUCAS_KEY",
            "projects": ["test/demo"],
        },
    }
    profile.write_text(yaml.safe_dump(value))
    return root


@pytest.fixture(autouse=True)
def isolated_installation(tmp_path_factory, monkeypatch, definitions_fixture):
    root = tmp_path_factory.mktemp("installation-settings")
    config = root / "site.toml"
    config.write_text(
        f'bids = "{root / "bids"}"\nregistry = "{root / "registry"}"\nwork = "{root / "work"}"\n'
    )
    monkeypatch.setenv("NRO_SITE_CONFIG", str(config))
    for variable in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(site, "installation_record", lambda: {})
    monkeypatch.setattr(site_setup, "installation_record", lambda: {})
    monkeypatch.setitem(site.DEFAULTS, "definitions", str(definitions_fixture))
