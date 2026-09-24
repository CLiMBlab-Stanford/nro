"""Definitions-store migration, integrity, and ownership tests."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nro.configuration.definition_migrations import (
    LEGACY_MANAGED_NOTICES,
    MANAGED_NOTICE,
    MANIFEST,
    _begin_recovery,
    _manifest_text,
    migrate_store,
    store_lock,
    update_store,
    validate_store_integrity,
)
from nro.configuration.definitions import create_store, validate_store


def _files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".nro-definitions.lock"
    }


def test_new_store_is_versioned_and_warns_against_direct_edits(tmp_path):
    root = create_store(tmp_path / "definitions")
    assert (root / MANIFEST).read_text().startswith(MANAGED_NOTICE)
    for path in root.rglob("*"):
        if path.suffix in {".yml", ".yaml", ".py"}:
            text = path.read_text()
            if path.suffix == ".py" and text.startswith("#!"):
                text = text.splitlines(keepends=True)[0] + text.splitlines(keepends=True)[1]
            assert MANAGED_NOTICE in text[: len(MANAGED_NOTICE) + 128]
    validate_store_integrity(root)


def test_new_store_is_group_maintainable_without_adding_read_access(tmp_path):
    root = create_store(tmp_path / "definitions")
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o020
        if path.is_dir():
            assert mode & 0o010


def test_schema_two_moves_flywheel_keys_out_of_tracked_site_metadata(tmp_path, monkeypatch):
    from nro.bidsify.credentials import read_key, store_key

    root = create_store(tmp_path / "definitions")
    store_key(root, "cni", "private-key", host="cni.example.org")
    site = root / "site/site.yml"
    value = yaml.safe_load(site.read_text())
    value["version"] = 1
    value["bidsify"]["servers"] = {
        "cni": {
            "host": "cni.example.org",
            "credential_env": "CNI_API_KEY",
            "projects": ["lab/study"],
        }
    }
    site.write_text(MANAGED_NOTICE + yaml.safe_dump(value, sort_keys=False))
    (root / MANIFEST).write_text(_manifest_text(root, 1))
    locator = tmp_path / "site.toml"
    locator.write_text(f'definitions = "{root}"\n')
    monkeypatch.setenv("NRO_SITE_CONFIG", str(locator))

    assert migrate_store(
        root, validate=lambda candidate: validate_store(candidate, require_site=True)
    )

    migrated = yaml.safe_load(site.read_text())
    assert migrated["version"] == 2
    assert migrated["bidsify"]["servers"]["cni"] == {
        "host": "cni.example.org",
        "projects": ["lab/study"],
    }
    assert read_key(root, "cni", host="cni.example.org") == "private-key"
    assert ".definition-secrets" not in (root / MANIFEST).read_text()


def test_application_layer_import_precedes_schema_one_site_migration(tmp_path):
    """Match the shared installer's import and migration order for an old store."""
    root = create_store(tmp_path / "definitions")
    site = root / "site/site.yml"
    value = yaml.safe_load(site.read_text())
    value["version"] = 1
    site.write_text(MANAGED_NOTICE + yaml.safe_dump(value, sort_keys=False))
    (root / MANIFEST).write_text(_manifest_text(root, 1))
    locator = tmp_path / "site.toml"
    locator.write_text(f'definitions = "{root}"\n')
    checkout = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
from pathlib import Path

class RejectApplicationDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.', 1)[0] in {'numpy', 'pandas'}:
            raise ImportError(f'application dependency imported during bootstrap: {fullname}')
        return None

sys.meta_path.insert(0, RejectApplicationDependencies())
from nro.engine import installation_layers
from nro.engine.site_setup import migrate_site_configuration

site, checkout, applications = map(Path, sys.argv[1:])
migrate_site_configuration(site)
from nro.orchestration.source_snapshots import SourceStore
SourceStore(applications).capture(checkout)
"""
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(locator),
            str(checkout),
            str(tmp_path / "applications"),
        ],
        cwd=checkout,
        env={
            **os.environ,
            "NRO_EXECUTION_SOURCE_ROOT": str(checkout),
            "NRO_SITE_CONFIG": str(locator),
            "PYTHONPATH": str(checkout),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=True,
    )

    assert yaml.safe_load(site.read_text())["version"] == 2
    validate_store(root, require_site=True)


def test_schema_three_updates_managed_definition_cli_guidance(tmp_path):
    root = create_store(tmp_path / "definitions")
    workflow = root / "workflows/main_workflow.yml"
    workflow.write_text(workflow.read_text().replace(MANAGED_NOTICE, LEGACY_MANAGED_NOTICES[0], 1))
    (root / MANIFEST).write_text(_manifest_text(root, 2))

    assert migrate_store(
        root, validate=lambda candidate: validate_store(candidate, require_site=True)
    )

    assert workflow.read_text().startswith(MANAGED_NOTICE)
    assert LEGACY_MANAGED_NOTICES[0] not in workflow.read_text()


def test_direct_changes_are_rejected_but_explicit_apply_can_adopt_them(tmp_path):
    root = create_store(tmp_path / "definitions")
    relative = Path("configs/clean/alternative_clean.yml")
    target = root / relative
    target.write_text("standardize: false\n")
    with pytest.raises(ValueError, match="outside nro"):
        validate_store_integrity(root)
    update_store(
        root,
        {relative: target.read_bytes()},
        validate=lambda candidate: validate_store(candidate, require_site=True),
        adopt_drift=True,
    )
    validate_store_integrity(root)
    assert target.read_text().startswith(MANAGED_NOTICE)


def test_legacy_migration_rewrites_files_in_place_and_is_idempotent(tmp_path):
    root = create_store(tmp_path / "definitions")
    (root / MANIFEST).unlink()
    workflow = root / "workflows/main_workflow.yml"
    workflow.write_text(workflow.read_text().removeprefix(MANAGED_NOTICE))
    assert migrate_store(
        root, validate=lambda candidate: validate_store(candidate, require_site=True)
    )
    assert workflow.read_text().startswith(MANAGED_NOTICE)
    before = _files(root)
    assert not migrate_store(
        root, validate=lambda candidate: validate_store(candidate, require_site=True)
    )
    assert _files(root) == before


def test_legacy_migration_ignores_unmanaged_editor_directories(tmp_path):
    root = create_store(tmp_path / "definitions")
    (root / MANIFEST).unlink()
    checkpoint = root / "configs/clean/.ipynb_checkpoints"
    checkpoint.mkdir()
    scratch = checkpoint / "draft.yml"
    scratch.write_text("not: a definition\n")
    checkpoint.chmod(0)
    try:
        assert migrate_store(
            root, validate=lambda candidate: validate_store(candidate, require_site=True)
        )
    finally:
        checkpoint.chmod(0o755)
    assert scratch.read_text() == "not: a definition\n"
    validate_store_integrity(root)


def test_failed_transaction_preserves_store_bytes(tmp_path):
    root = create_store(tmp_path / "definitions")
    before = _files(root)
    with pytest.raises(ValueError, match="typo"):
        update_store(
            root,
            {Path("configs/clean/bad_clean.yml"): b"typo: true\n"},
            validate=lambda candidate: validate_store(candidate, require_site=True),
        )
    assert _files(root) == before


def test_next_writer_recovers_an_interrupted_publication(tmp_path):
    root = create_store(tmp_path / "definitions")
    relative = Path("workflows/main_workflow.yml")
    before = _files(root)
    _begin_recovery(root, {relative, Path(MANIFEST)})
    (root / relative).write_text("corrupt: partial publication\n")
    with store_lock(root):
        pass
    assert _files(root) == before
    validate_store_integrity(root)


def test_branch_install_migrates_only_its_private_layer(tmp_path, monkeypatch):
    from nro.configuration import branch_definitions
    from nro.engine import bootstrap

    shared = create_store(tmp_path / "shared")
    parent = create_store(tmp_path / "parent", include_site=False, inherited_site=shared)
    private = create_store(tmp_path / "private", include_site=False, inherited_site=shared)
    (private / MANIFEST).unlink()
    before_shared, before_parent = _files(shared), _files(parent)
    monkeypatch.setattr(
        bootstrap,
        "settings",
        lambda path=None: ({"registry": str(tmp_path / "control"), "definitions": str(shared)}, {}),
    )
    monkeypatch.setattr(branch_definitions, "read_selection", lambda *args: private)
    monkeypatch.setattr(
        branch_definitions,
        "inherited_definitions",
        lambda *args: (private, parent, shared),
    )
    bootstrap.prepare_branch_definitions(
        tmp_path / "site.toml", {"branch": "dev", "registry_id": "branch-id"}
    )
    validate_store_integrity(private)
    assert _files(shared) == before_shared
    assert _files(parent) == before_parent


def test_branch_definition_setup_passes_complete_branch_identity(tmp_path, monkeypatch):
    from nro.engine import branch_definition_setup

    captured = {}
    monkeypatch.setattr(
        branch_definition_setup,
        "prepare_branch_definitions",
        lambda site, record: captured.update(site=site, record=record),
    )
    checkout = tmp_path / "checkout"

    branch_definition_setup.main(
        [
            "--site",
            str(tmp_path / "site.toml"),
            "--checkout",
            str(checkout),
            "--branch",
            "dev",
            "--registry-id",
            "branch-id",
        ]
    )

    assert captured == {
        "site": tmp_path / "site.toml",
        "record": {
            "checkout": str(checkout.resolve()),
            "branch": "dev",
            "registry_id": "branch-id",
        },
    }
