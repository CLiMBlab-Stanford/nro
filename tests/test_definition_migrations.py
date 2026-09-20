"""Definitions-store migration, integrity, and ownership tests."""

from pathlib import Path

import pytest

from nro.configuration.definition_migrations import (
    MANAGED_NOTICE,
    MANIFEST,
    _begin_recovery,
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
