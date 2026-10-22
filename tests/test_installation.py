from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nro.definitions.store import ConfigStore
from nro.engine.templates import FSAVERAGE_VERTEX_COUNTS
from nro.site import (
    bootstrap,
    bootstrap_dependencies,
    dependencies,
    installation_layers,
    shared_installation,
)
from nro.site import configuration as site
from nro.site import (
    setup as site_setup,
)
from nro.site.setup import edit_settings, save_settings


def test_locked_native_dependencies_have_python312_glibc217_wheels() -> None:
    lock = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    native = {
        "contourpy",
        "greenlet",
        "h5py",
        "igraph",
        "leidenalg",
        "matplotlib",
        "numpy",
        "pandas",
        "pillow",
        "pyzmq",
        "scikit-learn",
        "scipy",
        "zstandard",
    }
    for name in native:
        wheels = {Path(wheel["url"]).name for wheel in packages[name]["wheels"]}
        assert any(
            ("-cp312-" in wheel or "-abi3-" in wheel)
            and ("manylinux2014_x86_64" in wheel or "manylinux_2_17_x86_64" in wheel)
            for wheel in wheels
        ), name


def test_managed_bootstrap_symlinks_are_ignored(tmp_path) -> None:
    root = Path(__file__).parents[1]
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / ".nro-bootstraps/target").mkdir(parents=True)
    (tmp_path / ".nro-bootstrap").symlink_to(".nro-bootstraps/target")
    (tmp_path / ".nro-bootstrap.previous-test").symlink_to(".nro-bootstraps/target")

    for path in (".nro-bootstrap", ".nro-bootstrap.previous-test"):
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", path],
            cwd=tmp_path,
            check=True,
        )


def test_container_versions_match_managed_resource_names() -> None:
    for name, version in site.CONTAINER_VERSIONS.items():
        assert version in site.DERIVED[name][1]


def test_missing_freesurfer_license_is_optional_during_setup(tmp_path, monkeypatch) -> None:
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    templates = tmp_path / "templates"
    for relative in (
        "tpl-fsaverage/tpl-fsaverage_hemi-L_den-41k_midthickness.surf.gii",
        "tpl-fsaverage/tpl-fsaverage_hemi-R_den-41k_midthickness.surf.gii",
    ):
        path = templates / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("template")
    mni = templates / "tpl-MNI152NLin2009cAsym/template.nii.gz"
    mni.parent.mkdir(parents=True)
    mni.write_text("template")
    bids = tmp_path / "missing-BIDS"
    values = {
        "definitions": str(definitions),
        "runtime": "/bin/true",
        "license": str(tmp_path / "missing-license.txt"),
        "mni_template": str(mni),
        "templates": str(templates),
        "bids": str(bids),
        "work": str(tmp_path / "WORK"),
        "registry": str(tmp_path / "registry"),
        "oslom": "/bin/true",
    }
    monkeypatch.setattr(dependencies, "settings", lambda: (values, tmp_path / "site.toml"))
    monkeypatch.setattr(dependencies, "required_images", lambda: {})

    results = dependencies.check_installation(
        quick=True,
        with_oslom=False,
        slurm=False,
        with_viewer=False,
    )

    license_check = next(row for row in results if row["name"] == "FreeSurfer license")
    assert license_check["ok"] is False
    assert license_check["required"] is False
    assert "Register at" in license_check["detail"]
    bids_check = next(row for row in results if row["name"] == "bids")
    assert bids_check["ok"] is False
    assert bids_check["required"] is False
    assert "BIDS root is not readable" in bids_check["detail"]


@pytest.mark.parametrize("response", ["", "y", "Y", "yes", "YES"])
def test_setup_resource_terms_default_to_accept(monkeypatch, response):
    from nro.bin import setup

    monkeypatch.setattr("builtins.input", lambda prompt: response)
    assert setup._accept_resource_terms()


@pytest.mark.parametrize("response", ["n", "no", "anything else"])
def test_setup_resource_terms_can_be_declined(monkeypatch, response):
    from nro.bin import setup

    monkeypatch.setattr("builtins.input", lambda prompt: response)
    assert not setup._accept_resource_terms()


@pytest.fixture
def personal_publication(monkeypatch):
    from nro.orchestration import registry, releases

    monkeypatch.setattr(site_setup, "migrate_site_configuration", lambda path: None)
    monkeypatch.setattr(releases, "tagged_source", lambda root: None)
    monkeypatch.setattr(registry.Registry, "for_project", lambda *args, **kwargs: object())
    monkeypatch.setattr(shared_installation, "prepare_pool", lambda *args, **kwargs: None)

    def publish(root, registry, *, installation):
        assert installation["mode"] == "personal"
        assert installation["ready"]
        bootstrap.write_record(root / bootstrap.RECORD, installation)

    monkeypatch.setattr(shared_installation, "publish", publish)


@pytest.fixture
def isolated_site(tmp_path, monkeypatch):
    path = tmp_path / "site.toml"
    path.write_text("")
    monkeypatch.setenv("NRO_SITE_CONFIG", str(path))
    for key in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(site, "installation_record", lambda: {})
    return path


def test_bootstrap_pins_are_shell_compatible_and_complete(tmp_path):
    pins = tmp_path / "bootstrap.pins"
    pins.write_text(
        "UV_VERSION=1.2.3\n"
        "PYTHON_VERSION=3.12.11\n"
        "PYYAML_VERSION=6.0.3\n"
        f"PYYAML_SHA256={'a' * 64}\n"
    )
    assert bootstrap_dependencies.read_pins(pins)["PYTHON_VERSION"] == "3.12.11"
    pins.write_text("UV_VERSION=1.2.3\n")
    with pytest.raises(ValueError, match="Missing bootstrap pin"):
        bootstrap_dependencies.read_pins(pins)
    pins.write_text(
        "UV_VERSION=1.2.3\nPYTHON_VERSION=3.12.11\nPYYAML_VERSION=6.0.3\nPYYAML_SHA256=invalid\n"
    )
    with pytest.raises(ValueError, match="Invalid SHA-256"):
        bootstrap_dependencies.read_pins(pins)


def test_bootstrap_ready_rejects_host_python(tmp_path, monkeypatch):
    python = tmp_path / "bootstrap/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setattr(
        bootstrap_dependencies.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"version": [3, 12, 11], "base": "/usr/bin/python3", "yaml": "6.0.3"}
            ),
        ),
    )
    assert not bootstrap_dependencies._ready(python, root=tmp_path)


def test_offline_bootstrap_uses_uv_without_pip(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    uv = root / "uv"
    uv.write_text("")
    states = iter((False, True))
    monkeypatch.setattr(bootstrap_dependencies, "_ready", lambda *args, **kwargs: next(states))
    commands = []
    monkeypatch.setattr(
        bootstrap_dependencies.subprocess,
        "run",
        lambda command, **options: commands.append(command),
    )
    target = bootstrap_dependencies.prepare_target(root, uv, offline=True)
    assert target.parent == root / ".nro-bootstraps"
    assert commands[0][1:5] == ["venv", str(target), "--python", "3.12.11"]
    assert "--managed-python" in commands[0] and "--offline" in commands[0]
    assert commands[1][1] == "pip"
    assert "--require-hashes" in commands[1] and "--offline" in commands[1]
    assert all("pip" not in command[:1] for command in commands)


def test_install_entrypoint_is_valid_posix_shell():
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["sh", "-n", str(root / "install")], check=True)


def _copy_shell_installer(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "checkout"
    (root / "nro/site").mkdir(parents=True)
    (root / "install").write_bytes((source / "install").read_bytes())
    (root / "nro/site/bootstrap.pins").write_bytes(
        (source / "nro/site/bootstrap.pins").read_bytes()
    )
    return root


def _linux_x86_path(tmp_path: Path) -> str:
    commands = tmp_path / "linux-x86-commands"
    commands.mkdir()
    for name, output in (("uname", "x86_64"), ("ldd", "glibc 2.28")):
        command = commands / name
        command.write_text(f"#!/bin/sh\necho '{output}'\n")
        command.chmod(0o755)
    return f"{commands}:{os.environ['PATH']}"


def test_shell_installer_rejects_unsupported_platform(tmp_path):
    root = _copy_shell_installer(tmp_path)
    installer = tmp_path / "install-link"
    installer.symlink_to(root / "install")
    commands = tmp_path / "commands"
    commands.mkdir()
    uname = commands / "uname"
    uname.write_text("#!/bin/sh\necho sparc64\n")
    uname.chmod(0o755)
    result = subprocess.run(
        ["sh", str(installer), "--offline"],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "no pinned uv archive supports Linux architecture sparc64" in result.stderr


def test_shell_installer_rejects_bad_cached_archive(tmp_path):
    root = _copy_shell_installer(tmp_path)
    pins = bootstrap_dependencies.read_pins(root / "nro/site/bootstrap.pins")
    archive = root / ".nro-cache" / pins["UV_ARCHIVE_X86_64_GNU"]
    archive.parent.mkdir()
    archive.write_bytes(b"not uv")
    result = subprocess.run(
        ["sh", str(root / "install")],
        env={**os.environ, "PATH": _linux_x86_path(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "cached uv archive failed checksum verification" in result.stderr


def test_offline_shell_installer_requires_cached_uv(tmp_path):
    root = _copy_shell_installer(tmp_path)
    result = subprocess.run(
        ["sh", str(root / "install"), "--offline"],
        env={**os.environ, "PATH": _linux_x86_path(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "offline setup needs cached uv archive" in result.stderr


def test_offline_shell_installer_requires_cached_python(tmp_path):
    root = _copy_shell_installer(tmp_path)
    pins = bootstrap_dependencies.read_pins(root / "nro/site/bootstrap.pins")
    uv = root / ".nro-tools" / f"uv-{pins['UV_VERSION']}" / "uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(
        "#!/bin/sh\n"
        f"if [ \"$1\" = --version ]; then echo 'uv {pins['UV_VERSION']}'; exit 0; fi\n"
        "exit 1\n"
    )
    uv.chmod(0o755)
    result = subprocess.run(
        ["sh", str(root / "install"), "--offline"],
        env={**os.environ, "PATH": _linux_x86_path(tmp_path)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "offline setup needs cached managed CPython 3.12.11" in result.stderr


def test_interrupted_bootstrap_cutover_rolls_forward(tmp_path, monkeypatch):
    target = tmp_path / ".nro-bootstraps/pinned"
    (target / "bin").mkdir(parents=True)
    (target / "bin/python").write_text("")
    backup = tmp_path / ".nro-bootstrap.previous-test"
    backup.mkdir()
    journal = {
        "target": str(target.relative_to(tmp_path)),
        "backup": str(backup.relative_to(tmp_path)),
        "phase": "renamed",
    }
    (tmp_path / bootstrap_dependencies.TRANSITION).write_text(json.dumps(journal))
    monkeypatch.setattr(bootstrap_dependencies, "_ready", lambda *args, **kwargs: True)

    bootstrap_dependencies.activate_target(tmp_path, target)

    active = tmp_path / ".nro-bootstrap"
    assert active.is_symlink()
    assert active.resolve() == target
    assert not backup.exists()
    assert not (tmp_path / bootstrap_dependencies.TRANSITION).exists()


def test_failed_bootstrap_cutover_restores_legacy_directory(tmp_path, monkeypatch):
    target = tmp_path / ".nro-bootstraps/pinned"
    (target / "bin").mkdir(parents=True)
    (target / "bin/python").write_text("")
    active = tmp_path / ".nro-bootstrap"
    active.mkdir()
    (active / "legacy").write_text("retained")
    monkeypatch.setattr(bootstrap_dependencies, "_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        bootstrap_dependencies,
        "_publish_symlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap_dependencies.activate_target(tmp_path, target)

    assert active.is_dir() and not active.is_symlink()
    assert (active / "legacy").read_text() == "retained"
    assert not (tmp_path / bootstrap_dependencies.TRANSITION).exists()


def test_interrupted_cutover_rolls_back_when_target_is_invalid(tmp_path, monkeypatch):
    target = tmp_path / ".nro-bootstraps/pinned"
    target.mkdir(parents=True)
    backup = tmp_path / ".nro-bootstrap.previous-test"
    backup.mkdir()
    (backup / "legacy").write_text("retained")
    journal = {
        "target": str(target.relative_to(tmp_path)),
        "backup": str(backup.relative_to(tmp_path)),
        "phase": "renamed",
    }
    (tmp_path / bootstrap_dependencies.TRANSITION).write_text(json.dumps(journal))
    monkeypatch.setattr(bootstrap_dependencies, "_ready", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError, match="Rolled back"):
        bootstrap_dependencies.activate_target(tmp_path, target)

    active = tmp_path / ".nro-bootstrap"
    assert active.is_dir() and not active.is_symlink()
    assert (active / "legacy").read_text() == "retained"


def test_lab_defaults_select_medium_fsaverage_anatomy(isolated_site):
    anatomy = ConfigStore().load_configuration("anat", "main")
    functional = ConfigStore().load_configuration("func", "main")
    assert anatomy.values["fsaverage_template"] == "fsaverage6"
    assert "output_spaces" not in functional.values
    assert "fsaverage_template" not in functional.values


def test_fsaverage6_midthickness_is_derived_from_pinned_surfaces(tmp_path):
    import nibabel as nib
    import numpy as np

    directory = tmp_path / "tpl-fsaverage"
    directory.mkdir()
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    for hemisphere in ("L", "R"):
        for surface, offset in (("white", 0.0), ("pial", 2.0)):
            image = nib.GiftiImage(
                darrays=[
                    nib.gifti.GiftiDataArray(
                        np.full((3, 3), offset, dtype=np.float32),
                        intent="NIFTI_INTENT_POINTSET",
                    ),
                    nib.gifti.GiftiDataArray(triangles, intent="NIFTI_INTENT_TRIANGLE"),
                ]
            )
            nib.save(
                image,
                directory / f"tpl-fsaverage_hemi-{hemisphere}_den-41k_{surface}.surf.gii",
            )

    dependencies.ensure_fsaverage6_midthickness(tmp_path)
    dependencies.ensure_fsaverage6_midthickness(tmp_path)

    for hemisphere in ("L", "R"):
        path = directory / f"tpl-fsaverage_hemi-{hemisphere}_den-41k_midthickness.surf.gii"
        image = nib.load(str(path))
        pointset = image.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
        assert pointset.meta["AnatomicalStructureSecondary"] == "MidThickness"
        np.testing.assert_array_equal(
            pointset.data,
            np.ones((3, 3), dtype=np.float32),
        )


def test_resource_changes_propagate_to_all_configurations(isolated_site, tmp_path):
    save_settings(
        isolated_site,
        {
            "images": str(tmp_path / "images"),
            "runtime": "apptainer",
            "binds": [],
            "templates": str(tmp_path / "templates"),
            "license": str(tmp_path / "license"),
        },
    )
    store = ConfigStore()
    anatomy = store.load_configuration("anat", "main").values
    functional = store.load_configuration("func", "main").values
    clean = store.load_configuration("clean", "main").values
    assert anatomy["container"]["image"] == functional["container"]["image"]
    assert functional["container"]["image"] == clean["container"]["image"]
    assert clean["container"]["image"].startswith(str(tmp_path))
    assert functional["container"]["engine"] == clean["container"]["engine"] == "apptainer"
    assert functional["container"]["bind"] == clean["container"]["bind"] == []
    assert functional["synbold_disco_license"] == str(tmp_path / "license")
    assert anatomy["mni_template"].startswith(str(tmp_path / "templates"))


def test_install_runtime_acquires_and_reuses_verified_managed_tree(
    isolated_site, tmp_path, monkeypatch
):
    import hashlib
    import tarfile

    source = tmp_path / "source"
    executable = source / "bin/apptainer"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\necho apptainer version 1.4.5-1\n", encoding="utf-8")
    executable.chmod(0o755)
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source / "bin", arcname="bin")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    specification = {
        **dependencies.MANAGED_RUNTIME,
        "artifact": archive.name,
        "sha256": digest,
        "source": "https://example.test/runtime.tar.gz",
    }
    monkeypatch.setattr(dependencies, "MANAGED_RUNTIME", specification)
    monkeypatch.setattr(
        dependencies,
        "run_probe",
        lambda command, **_kwargs: "apptainer version 1.4.5-1",
    )
    images = tmp_path / "resources/images"
    save_settings(isolated_site, {"images": str(images), "runtime": "singularity"})
    artifact = images.parent / "runtimes/artifacts" / archive.name
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(archive.read_bytes())

    dependencies.install_runtime(offline=True)
    runtime = Path(site.settings()[0]["runtime"])
    manifest = json.loads((runtime.parent.parent / dependencies.RUNTIME_MANIFEST).read_text())

    assert runtime.is_file()
    assert manifest["specification"] == specification
    assert manifest["inventory"]["bin/apptainer"] == dependencies.sha256(runtime)
    dependencies.install_runtime(offline=True)
    runtime.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed verification"):
        dependencies.install_runtime(offline=True)


def test_managed_runtime_supports_oldest_linux_abi(monkeypatch):
    monkeypatch.setattr(dependencies.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependencies.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(dependencies.platform, "libc_ver", lambda: ("glibc", "2.17"))

    dependencies._require_runtime_platform()

    monkeypatch.setattr(dependencies.platform, "libc_ver", lambda: ("glibc", "2.16"))
    with pytest.raises(RuntimeError, match="requires glibc 2.17 or newer"):
        dependencies._require_runtime_platform()


def test_managed_runtime_environment_lock_matches_specification():
    path = (
        Path(__file__).parents[1]
        / "nro/site/resources/apptainer-1.4.5-conda-glibc217-linux-64.lock"
    )

    assert dependencies.sha256(path) == dependencies.MANAGED_RUNTIME["environment_lock_sha256"]


def test_install_runtime_replaces_prior_managed_specification(isolated_site, tmp_path, monkeypatch):
    old_root = tmp_path / "runtimes/apptainer-old"
    old_executable = old_root / "bin/apptainer"
    old_executable.parent.mkdir(parents=True)
    old_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    old_executable.chmod(0o755)
    (old_root / dependencies.RUNTIME_MANIFEST).write_text(
        json.dumps({"format": 1, "specification": {"version": "old"}}),
        encoding="utf-8",
    )
    current_root = tmp_path / "runtimes/apptainer-current"
    current_executable = current_root / "bin/apptainer"
    current_executable.parent.mkdir(parents=True)
    current_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    current_executable.chmod(0o755)
    save_settings(isolated_site, {"runtime": str(old_executable)})
    monkeypatch.setattr(dependencies, "_runtime_root", lambda _values: current_root)
    monkeypatch.setattr(dependencies, "_require_runtime_platform", lambda: None)
    monkeypatch.setattr(
        dependencies,
        "_validate_runtime_tree",
        lambda root, **_kwargs: {
            "format": 1,
            "specification": dependencies.MANAGED_RUNTIME,
            "executable": "bin/apptainer",
            "executable_sha256": dependencies.sha256(root / "bin/apptainer"),
            "probe": "apptainer version 1.4.5",
        },
    )

    dependencies.install_runtime(offline=True)

    assert site.settings()[0]["runtime"] == str(current_executable)


def test_install_runtime_offline_reports_missing_managed_artifact(
    isolated_site, tmp_path, monkeypatch
):
    images = tmp_path / "resources/images"
    save_settings(isolated_site, {"images": str(images), "runtime": "singularity"})
    monkeypatch.setattr(
        dependencies,
        "run_probe",
        lambda command, **_kwargs: "apptainer version 1.4.5-1",
    )

    with pytest.raises(RuntimeError, match="Offline setup cannot obtain managed Apptainer"):
        dependencies.install_runtime(offline=True)


def test_install_runtime_preserves_explicit_external_executable(
    isolated_site, tmp_path, monkeypatch
):
    executable = tmp_path / "external/apptainer"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    save_settings(isolated_site, {"runtime": str(executable)})
    monkeypatch.setattr(
        dependencies,
        "_runtime_root",
        lambda _values: pytest.fail("explicit external runtime must not install a managed tree"),
    )
    monkeypatch.setattr(
        dependencies,
        "run_probe",
        lambda command, **_kwargs: "external runtime",
    )

    dependencies.install_runtime(offline=True)

    assert site.settings()[0]["runtime"] == str(executable)


def test_runtime_diagnostic_distinguishes_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setattr(dependencies.platform, "node", lambda: "compute01")
    path = tmp_path / "missing/apptainer"

    assert dependencies.runtime_diagnostic(str(path)) == (
        f"Container runtime {path} is missing on compute01"
    )
    assert dependencies.runtime_diagnostic("apptainer") == (
        "Container runtime 'apptainer' is not on PATH on compute01"
    )


def managed_runtime_tree(root: Path) -> Path:
    executable = root / "bin/apptainer"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    (root / dependencies.RUNTIME_MANIFEST).write_text("{}", encoding="utf-8")
    return executable


def test_managed_runtime_helpers_follow_existing_path(tmp_path, monkeypatch):
    from nro.site import managed_runtime

    executable = managed_runtime_tree(tmp_path / "runtime")
    monkeypatch.setattr(managed_runtime.os, "access", lambda path, mode: path == "/dev/fuse")
    monkeypatch.setenv("PATH", "/host/bin:/usr/bin")

    managed_runtime.expose_managed_runtime_helpers(executable)
    managed_runtime.expose_managed_runtime_helpers(executable)

    assert os.environ["PATH"] == f"/host/bin:/usr/bin:{executable.parent}"


def test_managed_runtime_helpers_leave_path_without_fuse_or_manifest(tmp_path, monkeypatch):
    from nro.site import managed_runtime

    managed = managed_runtime_tree(tmp_path / "runtime")
    external = tmp_path / "external/apptainer"
    external.parent.mkdir()
    external.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin")

    monkeypatch.setattr(managed_runtime.os, "access", lambda _path, _mode: True)
    managed_runtime.expose_managed_runtime_helpers(external)
    managed_runtime.expose_managed_runtime_helpers("apptainer")
    assert os.environ["PATH"] == "/usr/bin"

    monkeypatch.setattr(managed_runtime.os, "access", lambda _path, _mode: False)
    managed_runtime.expose_managed_runtime_helpers(managed)
    assert os.environ["PATH"] == "/usr/bin"


def test_template_catalog_covers_host_template_inputs():
    catalog = {Path(path) for path in dependencies.template_catalog()}
    configured = Path(site.DERIVED["mni_template"][1])
    required = {
        configured,
        Path(str(configured).replace("_T1w.nii.gz", "_desc-brain_T1w.nii.gz")),
        Path(str(configured).replace("_T1w.nii.gz", "_desc-brain_mask.nii.gz")),
        Path(str(configured).replace("_T1w.nii.gz", "_label-GM_probseg.nii.gz")),
    }
    assert required <= catalog

    installed = set(catalog)
    for hemisphere in ("L", "R"):
        installed.add(
            Path(f"tpl-fsaverage/tpl-fsaverage_hemi-{hemisphere}_den-41k_midthickness.surf.gii")
        )
    densities = {163842: "164k", 40962: "41k"}
    for vertices in FSAVERAGE_VERTEX_COUNTS.values():
        density = densities[vertices]
        required = {
            Path(f"tpl-fsaverage/tpl-fsaverage_hemi-{hemisphere}_den-{density}_{surface}.surf.gii")
            for hemisphere in ("L", "R")
            for surface in ("white", "pial", "midthickness", "sphere")
        }
        assert required <= installed


def test_invalid_path_update_is_atomic(isolated_site):
    original = isolated_site.read_bytes()
    with pytest.raises(ValueError):
        edit_settings(["work=/new/work", "typo=/new/path"])
    assert isolated_site.read_bytes() == original
    with pytest.raises(ValueError):
        edit_settings(["work=relative/path"])


def test_flywheel_defaults_are_optional_validated_site_settings(isolated_site):
    edit_settings(["flywheel_server=cni", "flywheel_project=group/project"])
    values, _sources = site.settings()
    assert values["flywheel_server"] == "cni"
    assert values["flywheel_project"] == "group/project"

    with pytest.raises(ValueError, match="GROUP/PROJECT"):
        edit_settings(["flywheel_project=project-only"])


def test_site_locator_updates_versioned_protected_definition(isolated_site, tmp_path):
    from nro.definitions.repository import create_store
    from nro.site.configuration import read_site_definition

    definitions = create_store(tmp_path / "definitions")
    isolated_site.write_text(f'definitions = "{definitions}"\n')
    destination = save_settings(isolated_site, {"work": "/updated/work"})
    assert destination == definitions / "site/site.yml"
    assert site.read_overrides(isolated_site) == {"definitions": str(definitions)}
    protected, _ = read_site_definition(definitions)
    assert protected["work"] == "/updated/work"


def test_installation_migrates_legacy_site_and_bidsify_settings(isolated_site, tmp_path):
    from nro.definitions.repository import create_store
    from nro.site.configuration import read_site_definition

    definitions = create_store(tmp_path / "definitions")
    (definitions / ".nro-definitions.yml").unlink()
    (definitions / "site/site.yml").unlink()
    profile_path = definitions / "bidsify/main.yml"
    profile = yaml.safe_load(profile_path.read_text())
    profile.update(
        servers={
            "cni": {
                "host": "cni.example.org",
                "credential_env": "CNI_API_KEY",
                "projects": ["lab/study"],
            }
        },
        project_sources={"study": [{"server": "cni", "project": "lab/study"}]},
        session_rules=[],
        event_rules=[],
    )
    profile["scanplans"].update(
        location="https://drive.google.com/drive/folders/example",
        credential_env="GOOGLE_APPLICATION_CREDENTIALS",
    )
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False))
    isolated_site.write_text(
        f'definitions = "{definitions}"\n'
        'bids = "/legacy/BIDS"\n'
        'flywheel_server = "cni"\n'
        'flywheel_project = "lab/study"\n'
    )

    destination = site_setup.migrate_site_configuration(isolated_site)

    assert destination == definitions / "site/site.yml"
    assert site.read_overrides(isolated_site) == {"definitions": str(definitions)}
    protected, bidsify = read_site_definition(definitions)
    assert protected["bids"] == "/legacy/BIDS"
    assert bidsify["default_server"] == "cni"
    assert bidsify["project_sources"]["study"][0]["project"] == "lab/study"
    assert bidsify["servers"]["cni"] == {
        "host": "cni.example.org",
        "projects": ["lab/study"],
    }
    migrated_profile = yaml.safe_load(profile_path.read_text())
    assert set(migrated_profile["scanplans"]) == {"parser"}
    assert "servers" not in migrated_profile


def test_first_install_creates_complete_definitions_store(isolated_site, tmp_path):
    from nro.definitions.repository import validate_store

    definitions = tmp_path / "definitions"
    save_settings(
        isolated_site,
        {"definitions": str(definitions), "bids": str(tmp_path / "BIDS")},
    )

    destination = site_setup.migrate_site_configuration(isolated_site)

    assert destination == definitions / "site/site.yml"
    assert site.read_overrides(isolated_site) == {"definitions": str(definitions)}
    assert validate_store(definitions, require_site=True)["workflows"] == 4


def test_first_install_defers_semantic_definition_validation(isolated_site, tmp_path, monkeypatch):
    from nro.definitions import repository

    definitions = tmp_path / "definitions"
    save_settings(
        isolated_site,
        {"definitions": str(definitions), "bids": str(tmp_path / "BIDS")},
    )
    monkeypatch.setattr(
        repository,
        "validate_store",
        lambda *args, **kwargs: pytest.fail("bootstrap invoked semantic validation"),
    )

    destination = site_setup.migrate_site_configuration(isolated_site)

    assert destination == definitions / "site/site.yml"
    assert (definitions / "workflows/main_workflow.yml").is_file()
    assert (definitions / "bidsify/main.yml").is_file()


def test_early_site_migration_imports_without_numpy() -> None:
    script = """
import builtins

original_import = builtins.__import__

def reject_numpy(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("numpy is unavailable during bootstrap")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_numpy
import nro.site.setup
import nro.definitions.repository
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_first_install_recovers_site_only_definitions_store(isolated_site, tmp_path):
    from nro.definitions.repository import validate_store
    from nro.site.configuration import write_site_definition

    definitions = tmp_path / "definitions"
    save_settings(
        isolated_site,
        {"definitions": str(definitions), "bids": str(tmp_path / "BIDS")},
    )
    values, _ = site.settings(path=isolated_site)
    protected, bidsify = site_setup._split_site_values(values)
    write_site_definition(definitions, protected, bidsify=bidsify)

    destination = site_setup.migrate_site_configuration(isolated_site)

    assert destination == definitions / "site/site.yml"
    assert validate_store(definitions, require_site=True)["bidsify"] == 1


def test_first_install_does_not_overwrite_unrecognized_partial_store(isolated_site, tmp_path):
    from nro.site.configuration import write_site_definition

    definitions = tmp_path / "definitions"
    save_settings(isolated_site, {"definitions": str(definitions)})
    values, _ = site.settings(path=isolated_site)
    protected, bidsify = site_setup._split_site_values(values)
    write_site_definition(definitions, protected, bidsify=bidsify)
    unexpected = definitions / "notes.txt"
    unexpected.write_text("keep me\n")

    with pytest.raises(ValueError, match="not part of the starter store"):
        site_setup.migrate_site_configuration(isolated_site)

    assert unexpected.read_text() == "keep me\n"


@pytest.mark.parametrize("shared", [False, True])
def test_interactive_generic_defaults_preserve_explicit_paths(
    isolated_site, tmp_path, monkeypatch, shared
):
    monkeypatch.setattr(site_setup, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(site, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        site_setup, "installation_record", lambda: {"mode": "shared" if shared else "personal"}
    )
    save_settings(isolated_site, {"work": "/configured/work"})
    _, sources = site.settings()
    values, _ = site.settings()
    assert values["images"] == str(tmp_path / "home/nro/images")
    assert values["registry"] == str(tmp_path / "home/nro/.nro")
    assert values["work"] == "/configured/work"
    assert values["bids"] == str(tmp_path / "home/nro/bids")
    assert values["binds"] == []


def test_interactive_defaults_keep_reachable_lab(isolated_site, tmp_path, monkeypatch):
    monkeypatch.setattr(site_setup, "LAB", tmp_path)
    monkeypatch.setattr(site, "LAB", tmp_path)
    monkeypatch.setitem(site.DEFAULTS, "definitions", str(tmp_path / "absent-definitions"))
    _, sources = site.settings()
    assert site_setup.interactive_defaults(sources) == {}
    monkeypatch.setattr(site_setup.os, "access", lambda *args: False)
    assert site_setup.interactive_defaults(sources)["binds"] == []


def test_explicit_default_parent_rebases_site_paths(isolated_site, tmp_path, monkeypatch):
    parent = tmp_path / "site-root"
    values = site.default_parent_paths(parent)

    assert values["definitions"] == str(parent / "definitions")
    assert values["bids"] == str(parent / "bids")
    assert values["registry"] == str(parent / ".nro")
    assert values["qunex"] == str(parent / "images/qunex_suite-1.5.1.sif")
    assert values["mni_template"] == str(
        parent / "templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz"
    )
    assert "partition" not in values
    assert "binds" not in values

    monkeypatch.setattr(site_setup, "LAB", tmp_path)
    proposals = site_setup.interactive_defaults({}, default_parent_dir=parent)
    assert proposals == values


@pytest.mark.parametrize("accept_all", [False, True])
def test_interactive_paths_display_and_save_proposals(
    isolated_site, tmp_path, monkeypatch, capsys, accept_all
):
    monkeypatch.setattr(site_setup, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(site, "LAB", tmp_path / "absent-lab")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(site_setup, "installation_record", lambda: {})
    monkeypatch.setattr(site_setup.sys.stdin, "isatty", lambda: True)
    prompts = []

    def accept(prompt):
        prompts.append(prompt)
        if prompt.startswith("Accept all"):
            return "y" if accept_all else "n"
        if not accept_all and "work [" in prompt:
            return str(tmp_path / "scratch")
        return ""

    monkeypatch.setattr("builtins.input", accept)
    edit_settings()
    proposed_bids = str(tmp_path / "home/nro/bids")
    assert proposed_bids in capsys.readouterr().out
    if accept_all:
        assert len(prompts) == 2
    else:
        assert f"bids [{proposed_bids}]" in "\n".join(prompts)
        assert len(prompts) == len(site_setup.DESCRIPTIONS) + 2
        assert site.read_overrides(isolated_site)["work"] == str(tmp_path / "scratch")
    assert not any("root" in prompt for prompt in prompts)
    assert site.read_overrides(isolated_site)["bids"] == proposed_bids
    assert site.read_overrides(isolated_site)["binds"] == []
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("mode", ["personal", "shared"])
def test_production_site_ignores_personal_environment(isolated_site, monkeypatch, mode):
    save_settings(isolated_site, {"bids": "/shared/BIDS"})
    monkeypatch.setattr(
        site, "installation_record", lambda: {"mode": mode, "site": str(isolated_site)}
    )
    monkeypatch.setattr("nro.site.setup.installation_record", site.installation_record)
    assert site.settings()[0]["bids"] == "/shared/BIDS"
    if mode == "shared":
        with pytest.raises(ValueError, match="maintain"):
            edit_settings(["bids=/other"])
    monkeypatch.setenv("NRO_SITE_CONFIG", "/other/site.toml")
    with pytest.raises(ValueError, match=f"{mode} installation"):
        site.site_file()


@pytest.mark.parametrize("mode", ["personal", "shared", "branch"])
def test_verified_execution_snapshot_uses_pinned_site(isolated_site, tmp_path, monkeypatch, mode):
    configured = tmp_path / "configured.toml"
    configured.write_text("")
    snapshot = tmp_path / "execution-site.toml"
    snapshot.write_text("")
    monkeypatch.setattr(
        site,
        "installation_record",
        lambda: {"mode": mode, "site": str(configured)},
    )
    monkeypatch.setenv("NRO_SITE_CONFIG", str(snapshot))
    monkeypatch.setenv("NRO_EXECUTION_SOURCE_ROOT", str(tmp_path / "source"))

    assert site.site_file() == snapshot


def test_shared_onboarding_never_syncs_or_mutates_checkout(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    bootstrap_python = root / ".nro-bootstrap/bin/python"
    bootstrap_python.parent.mkdir(parents=True)
    bootstrap_python.write_text("interpreter")
    environment = root / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("interpreter")
    config = root / "site.toml"
    config.write_text("")
    record = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(environment),
        "site": str(config),
        "ready": True,
    }
    (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)

    def unexpected(*args, **kwargs):
        pytest.fail("User onboarding attempted shared maintenance")

    monkeypatch.setattr(bootstrap.subprocess, "run", unexpected)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    bin_dir = tmp_path / "user/bin"
    bootstrap.main(["--bin-dir", str(bin_dir)])
    bootstrap.main(["--bin-dir", str(bin_dir)])
    after = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert not (root / ".nro-install.lock").exists()
    assert "nro directory-aware launcher" in (bin_dir / "nro").read_text()
    assert json.loads((bin_dir / ".nro-launchers.json").read_text())["default"] == str(root)


def test_shared_maintenance_drains_and_publishes_checked_out_release(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    environment = root / ".nro-env"
    site = root / "site.toml"
    site.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(environment),
        "site": str(site),
        "ready": True,
        "with_oslom": True,
        "with_bidsify": False,
        "dev": False,
        "local": False,
    }
    (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)
    events = []
    monkeypatch.setattr(
        site_setup,
        "migrate_site_configuration",
        lambda path: events.append(("migrate", path)) or path,
    )
    monkeypatch.setattr("nro.orchestration.releases.tagged_source", lambda checkout: ())
    monkeypatch.setattr(
        shared_installation,
        "prepare_pool",
        lambda registry, **options: events.append(("drain", options["checkout"])),
    )
    monkeypatch.setattr(
        bootstrap,
        "activate_managed_bootstrap",
        lambda: events.append(("bootstrap", root)),
    )

    def publish(checkout, registry, *, installation):
        events.append(("publish", checkout))
        bootstrap.write_record(checkout / bootstrap.RECORD, installation)

    monkeypatch.setattr(shared_installation, "publish", publish)
    dependencies = root / ".nro-environments" / ("dependencies-" + "d" * 64)
    (dependencies / "bin").mkdir(parents=True)
    (dependencies / "bin/python").write_text("python")
    application_root = root / ".nro-environments/applications" / ("a" * 64)
    (application_root / "nro/orchestration").mkdir(parents=True)
    (application_root / "nro/orchestration/source_launcher.py").write_text("")
    application = SimpleNamespace(
        root=application_root,
        digest="a" * 64,
        command=lambda command, **options: (
            command[0],
            str(application_root / "nro/orchestration/source_launcher.py"),
            "a" * 64,
            str(site),
            "site-digest",
            *command[2:],
        ),
    )
    monkeypatch.setattr(
        installation_layers,
        "prepare_dependencies",
        lambda *args, **kwargs: events.append(("dependencies", root)) or (dependencies, "d" * 64),
    )
    monkeypatch.setattr(
        installation_layers,
        "capture_application",
        lambda selected: events.append(("application", selected)) or application,
    )
    commands = []
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **options: commands.append(command),
    )
    monkeypatch.setattr(bootstrap, "connect_user", lambda *args, **options: None)

    bootstrap.main(["--maintain", "--offline"])

    assert events == [
        ("migrate", site),
        ("dependencies", root),
        ("application", root),
        ("drain", root),
        ("bootstrap", root),
        ("publish", root),
    ]
    assert "nro.bin.setup" in commands[0]
    assert "--prepared-maintenance" in commands[0]
    saved = json.loads((root / bootstrap.RECORD).read_text())
    assert saved["ready"] is True
    assert Path(saved["environment"]) == dependencies
    assert saved["dependency_key"] == "d" * 64
    assert Path(saved["application"]) == application_root
    assert saved["application_digest"] == "a" * 64


def test_failed_shared_candidate_keeps_the_active_installation(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    active = root / ".nro-env"
    site = root / "site.toml"
    site.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record = {
        "mode": "shared",
        "checkout": str(root),
        "environment": str(active),
        "site": str(site),
        "ready": True,
        "with_oslom": True,
        "with_bidsify": False,
        "with_marss": True,
        "dev": False,
        "local": False,
    }
    record_path = root / bootstrap.RECORD
    record_path.write_text(json.dumps(record))
    original = record_path.read_bytes()
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(site_setup, "migrate_site_configuration", lambda path: path)
    monkeypatch.setattr("nro.orchestration.releases.tagged_source", lambda checkout: ())
    monkeypatch.setattr(shared_installation, "prepare_pool", lambda *args, **options: None)
    monkeypatch.setattr(
        shared_installation,
        "publish",
        lambda *args, **options: pytest.fail("Invalid candidate was published"),
    )
    dependencies = root / ".nro-environments" / ("dependencies-" + "d" * 64)
    (dependencies / "bin").mkdir(parents=True)
    (dependencies / "bin/python").write_text("python")
    application_root = root / ".nro-environments/applications" / ("a" * 64)
    (application_root / "nro/orchestration").mkdir(parents=True)
    (application_root / "nro/orchestration/source_launcher.py").write_text("")
    application = SimpleNamespace(
        root=application_root,
        digest="a" * 64,
        command=lambda command, **options: (
            command[0],
            str(application_root / "nro/orchestration/source_launcher.py"),
            "a" * 64,
            str(site),
            "site-digest",
            *command[2:],
        ),
    )
    monkeypatch.setattr(
        installation_layers,
        "prepare_dependencies",
        lambda *args, **kwargs: (dependencies, "d" * 64),
    )
    monkeypatch.setattr(installation_layers, "capture_application", lambda root: application)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)

    with pytest.raises(SystemExit) as stopped:
        bootstrap.main(["--maintain", "--offline"])

    assert stopped.value.code == 1
    assert record_path.read_bytes() == original
    assert dependencies != active
    assert "nro.bin.setup" in calls[0][0]
    assert calls[0][1]["env"]["NRO_CHECKOUT"] == str(root)


def test_shared_dependencies_are_reused_until_their_inputs_change(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n'
        '[[package]]\nname = "nro"\nversion = "1.0.0"\nsource = { editable = "." }\n'
    )
    uv = root / "uv"
    uv.write_text("")
    monkeypatch.setattr(bootstrap, "ROOT", root)
    calls = []

    def sync(command, **options):
        calls.append(command)
        if command[1:3] == ["lock", "--check"]:
            return
        environment = Path(options["env"]["UV_PROJECT_ENVIRONMENT"])
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python").write_text("python")

    monkeypatch.setattr(installation_layers.subprocess, "run", sync)
    monkeypatch.setattr(
        installation_layers,
        "_installed_packages",
        lambda _environment: [["example", "1.0"]],
    )
    record = {
        "with_oslom": True,
        "with_bidsify": False,
        "with_lesion": False,
        "with_marss": True,
        "dev": False,
    }
    env = {"UV_CACHE_DIR": str(root / "cache")}

    first, first_key = installation_layers.prepare_dependencies(
        root, uv, record, uv_version=bootstrap.UV_VERSION, env=env, offline=True
    )
    second, second_key = installation_layers.prepare_dependencies(
        root, uv, record, uv_version=bootstrap.UV_VERSION, env=env, offline=True
    )

    assert first == second
    assert first_key == second_key
    sync_calls = [command for command in calls if "sync" in command]
    assert len(sync_calls) == 1
    assert "--managed-python" in sync_calls[0]
    assert "--no-install-project" in sync_calls[0]
    assert "--offline" in sync_calls[0]
    assert bootstrap_dependencies.PYTHON_VERSION in sync_calls[0]
    marker = json.loads((first / ".nro-dependencies.json").read_text())
    assert marker["spec"]["python"] == "managed-cpython-3.12.11"

    (root / "uv.lock").write_text(
        (root / "uv.lock").read_text().replace('version = "1.0.0"', 'version = "1.0.1"')
    )
    version_only, version_only_key = installation_layers.prepare_dependencies(
        root, uv, record, uv_version=bootstrap.UV_VERSION, env=env, offline=True
    )
    assert version_only == first and version_only_key == first_key

    changed, changed_key = installation_layers.prepare_dependencies(
        root,
        uv,
        {**record, "with_bidsify": True},
        uv_version=bootstrap.UV_VERSION,
        env=env,
        offline=True,
    )
    assert changed != first and changed_key != first_key
    assert len([command for command in calls if "sync" in command]) == 2


def test_shared_dependencies_adopt_the_verified_active_environment(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n'
        '[[package]]\nname = "nro"\nversion = "1.0.0"\nsource = { editable = "." }\n'
    )
    uv = root / "uv"
    uv.write_text("")
    active = root / ".nro-environments/candidate-existing"
    (active / "bin").mkdir(parents=True)
    (active / "bin/python").write_text("python")
    (active / ".nro-checkout").write_text(f"{root}\n")
    calls = []
    monkeypatch.setattr(
        installation_layers.subprocess,
        "run",
        lambda command, **_options: calls.append(command),
    )
    monkeypatch.setattr(
        installation_layers,
        "_installed_packages",
        lambda _environment: [["example", "1.0"], ["nro", "0.9.0"]],
    )
    options = {
        "with_oslom": True,
        "with_bidsify": False,
        "with_lesion": False,
        "with_marss": True,
        "dev": False,
    }

    first, key = installation_layers.prepare_dependencies(
        root,
        uv,
        options,
        uv_version=bootstrap.UV_VERSION,
        env={},
        offline=False,
        previous_environment=active,
    )
    second, second_key = installation_layers.prepare_dependencies(
        root,
        uv,
        options,
        uv_version=bootstrap.UV_VERSION,
        env={},
        offline=False,
        previous_environment=active,
    )

    assert first == second == active
    assert key == second_key
    assert calls[0][1:3] == ["sync", "--frozen"]
    assert "--check" in calls[0] and "--inexact" in calls[0]
    assert calls[1][1:3] == ["lock", "--check"]


@pytest.mark.parametrize("without_oslom", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("without_marss", [False, True])
@pytest.mark.parametrize("with_lesion", [False, True])
@pytest.mark.parametrize("with_cicada", [False, True])
@pytest.mark.parametrize("without_viewer", [False, True])
def test_personal_setup_installs_selected_extras(
    tmp_path,
    monkeypatch,
    personal_publication,
    without_oslom,
    existing,
    without_marss,
    with_lesion,
    with_cicada,
    without_viewer,
):
    root = tmp_path / "personal"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    config = root / "site.toml"
    config.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record = {
        "mode": "personal",
        "checkout": str(root),
        "environment": str(root / ".nro-env"),
        "site": str(config),
        "ready": True,
        "with_oslom": False,
    }
    if existing:
        (root / bootstrap.RECORD).write_text(json.dumps(record))
    monkeypatch.setattr(bootstrap, "ROOT", root)
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    monkeypatch.setattr(bootstrap, "connect_user", lambda *a, **kw: None)
    environment = root / ".nro-environments" / ("dependencies-" + "d" * 64)
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("python")
    application_root = root / ".nro-environments/applications" / ("a" * 64)
    (application_root / "nro/orchestration").mkdir(parents=True)
    (application_root / "nro/orchestration/source_launcher.py").write_text("")
    application = SimpleNamespace(
        root=application_root,
        digest="a" * 64,
        command=lambda command, **options: (
            command[0],
            str(application_root / "nro/orchestration/source_launcher.py"),
            "a" * 64,
            str(config),
            "site-digest",
            *command[2:],
        ),
    )
    dependency_options = {}

    def dependencies(_root, _uv, options, **_kwargs):
        dependency_options.update(options)
        return environment, "d" * 64

    monkeypatch.setattr(installation_layers, "prepare_dependencies", dependencies)
    monkeypatch.setattr(installation_layers, "capture_application", lambda _root: application)
    monkeypatch.setattr(installation_layers, "prune_environments", lambda *args, **kwargs: None)
    bootstrap.main(
        [
            "--offline",
            "--mode",
            "personal",
            "--site",
            str(config),
            *(["--without-oslom"] if without_oslom else []),
            *(["--without-marss"] if without_marss else []),
            *(["--with-lesion"] if with_lesion else []),
            *(["--with-cicada"] if with_cicada else []),
            *(["--without-viewer"] if without_viewer else []),
        ]
    )
    assert dependency_options["with_oslom"] is not without_oslom
    assert dependency_options["with_marss"] is not without_marss
    assert dependency_options["with_lesion"] is with_lesion
    assert dependency_options["with_cicada"] is with_cicada
    assert calls[0][1]["cwd"] == root
    assert ("--without-oslom" in calls[0][0]) is without_oslom
    assert ("--without-viewer" in calls[0][0]) is without_viewer
    installed = json.loads((root / bootstrap.RECORD).read_text())
    assert installed["ready"]
    assert Path(installed["environment"]) == environment
    assert Path(installed["application"]) == application_root


def test_definition_adoption_runs_in_staged_application(
    tmp_path, monkeypatch, personal_publication
):
    root = tmp_path / "personal"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    config = root / "site.toml"
    config.write_text(f'definitions = "{tmp_path / "definitions"}"\n')
    (root / bootstrap.RECORD).write_text(
        json.dumps(
            {
                "mode": "personal",
                "checkout": str(root),
                "environment": str(root / ".nro-env"),
                "site": str(config),
                "ready": True,
                "with_oslom": False,
            }
        )
    )
    environment = root / ".nro-environments" / ("dependencies-" + "d" * 64)
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("python")
    application_root = root / ".nro-environments/applications" / ("a" * 64)
    (application_root / "nro/orchestration").mkdir(parents=True)
    (application_root / "nro/orchestration/source_launcher.py").write_text("")
    application = SimpleNamespace(
        root=application_root,
        digest="a" * 64,
        command=lambda command, **options: (
            command[0],
            str(application_root / "nro/orchestration/source_launcher.py"),
            "a" * 64,
            str(config),
            "site-digest",
            *command[2:],
        ),
    )
    calls = []
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(bootstrap, "check_workers", lambda _site: None)
    monkeypatch.setattr(bootstrap, "connect_user", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        installation_layers,
        "prepare_dependencies",
        lambda *args, **kwargs: (environment, "d" * 64),
    )
    monkeypatch.setattr(installation_layers, "capture_application", lambda _root: application)
    monkeypatch.setattr(installation_layers, "prune_environments", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **options: calls.append((command, options)),
    )

    bootstrap.main(["--maintain", "--offline", "--adopt-definitions"])

    assert len(calls) == 2
    assert calls[0][0][-2:] == ["nro.bin.def", "apply"]
    assert calls[0][0][0] == str(environment / "bin/python")
    assert calls[0][1]["env"]["NRO_SETUP_CHILD"] == "1"
    assert "nro.bin.setup" in calls[1][0]


@pytest.mark.parametrize("existing", [False, True])
def test_failed_personal_candidate_never_replaces_active_installation(
    tmp_path, monkeypatch, existing, personal_publication
):
    root = tmp_path / "personal"
    (root / ".nro-bootstrap/bin").mkdir(parents=True)
    (root / ".nro-bootstrap/bin/uv").write_text("uv")
    config = root / "site.toml"
    config.write_text(f'registry = "{tmp_path / "registry"}"\n')
    record_path = root / bootstrap.RECORD
    original = None
    if existing:
        record = {
            "mode": "personal",
            "checkout": str(root),
            "environment": str(root / ".nro-env"),
            "site": str(config),
            "ready": True,
            "with_oslom": True,
        }
        record_path.write_text(json.dumps(record))
        original = record_path.read_bytes()
    environment = root / ".nro-environments" / ("dependencies-" + "d" * 64)
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").write_text("python")
    application_root = root / ".nro-environments/applications" / ("a" * 64)
    (application_root / "nro/orchestration").mkdir(parents=True)
    (application_root / "nro/orchestration/source_launcher.py").write_text("")
    application = SimpleNamespace(
        root=application_root,
        digest="a" * 64,
        command=lambda command, **options: tuple(command),
    )
    monkeypatch.setattr(bootstrap, "ROOT", root)
    monkeypatch.setattr(bootstrap, "check_workers", lambda _site: None)
    monkeypatch.setattr(
        installation_layers,
        "prepare_dependencies",
        lambda *args, **kwargs: (environment, "d" * 64),
    )
    monkeypatch.setattr(installation_layers, "capture_application", lambda _root: application)

    def fail(command, **options):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    arguments = (
        ["--offline"] if existing else ["--offline", "--mode", "personal", "--site", str(config)]
    )
    with pytest.raises(SystemExit) as stopped:
        bootstrap.main(arguments)

    assert stopped.value.code == 1
    if original is None:
        assert not record_path.exists()
    else:
        assert record_path.read_bytes() == original


def test_maintenance_rejects_active_workers(tmp_path):
    control = tmp_path / "registry"
    (control / "shared/scheduler").mkdir(parents=True)
    with sqlite3.connect(control / "shared/scheduler/registry.sqlite3") as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT)")
        db.execute("CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT)")
        db.execute("INSERT INTO workers VALUES ('running', NULL)")
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})
    with pytest.raises(RuntimeError, match="Stop or drain"):
        bootstrap.check_workers(config)


def test_maintenance_ignores_unbound_records_from_dead_scheduler(tmp_path, monkeypatch):
    from nro.orchestration import scheduler_bus

    control = tmp_path / "registry"
    scheduler = control / "shared/scheduler"
    scheduler.mkdir(parents=True)
    (scheduler / "registry.sqlite3").touch()
    scheduler_bus.publish_snapshot(
        control,
        {
            "protocol": scheduler_bus.PROTOCOL,
            "generation": 1,
            "published_at": "test",
            "service_active": True,
            "workers": [],
            "submissions": [
                {
                    "id": 1,
                    "state": "prepared",
                    "slurm_job_id": None,
                    "memory_gb": 32,
                }
            ],
            "branches": {},
        },
    )
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("An unbound record was queried through Slurm"),
    )

    bootstrap.check_workers(config)


def test_maintenance_ignores_expired_unbound_database_records(tmp_path, monkeypatch):
    control = tmp_path / "registry"
    scheduler = control / "shared/scheduler"
    scheduler.mkdir(parents=True)
    with sqlite3.connect(scheduler / "registry.sqlite3") as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT, lease_expires_at REAL)")
        db.execute(
            "CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT, created_at TEXT)"
        )
        db.execute("INSERT INTO workers VALUES ('running', NULL, 0)")
        db.execute(
            "INSERT INTO scheduler_submissions VALUES "
            "('prepared', NULL, '2000-01-01T00:00:00+00:00')"
        )
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "An expired unbound record was queried through Slurm"
        ),
    )

    bootstrap.check_workers(config)


def test_maintenance_rejects_live_unbound_worker_lease(tmp_path):
    control = tmp_path / "registry"
    scheduler = control / "shared/scheduler"
    scheduler.mkdir(parents=True)
    with sqlite3.connect(scheduler / "registry.sqlite3") as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT, lease_expires_at REAL)")
        db.execute(
            "CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT, created_at TEXT)"
        )
        db.execute("INSERT INTO workers VALUES ('running', NULL, ?)", (time.time() + 60,))
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})

    with pytest.raises(RuntimeError, match="Stop or drain"):
        bootstrap.check_workers(config)


def test_prepared_shared_setup_verifies_its_installation_barrier(tmp_path):
    control = tmp_path / "registry"
    scheduler = control / "shared/scheduler"
    scheduler.mkdir(parents=True)
    checkout = tmp_path / "main"
    checkout.mkdir()
    with sqlite3.connect(scheduler / "registry.sqlite3") as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            (
                ("maintenance_mode", "installation"),
                ("installation_checkout", str(checkout.resolve())),
            ),
        )
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})

    bootstrap.check_installation_barrier(config, checkout)

    with sqlite3.connect(scheduler / "registry.sqlite3") as db:
        db.execute(
            "UPDATE metadata SET value=? WHERE key='installation_checkout'",
            (str(tmp_path / "other"),),
        )
    with pytest.raises(RuntimeError, match="not owned"):
        bootstrap.check_installation_barrier(config, checkout)


@pytest.mark.parametrize(
    "scheduler",
    [
        "ended",
        "invalid",
        "mixed",
        "RUNNING",
        "PENDING",
        "COMPLETING",
        "failed",
        "timeout",
        "missing",
    ],
)
def test_maintenance_checks_slurm_without_mutating_registry(tmp_path, monkeypatch, scheduler):
    control = tmp_path / "registry"
    (control / "shared/scheduler").mkdir(parents=True)
    database = control / "shared/scheduler/registry.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE workers (state TEXT, slurm_job_id TEXT)")
        db.execute("CREATE TABLE scheduler_submissions (state TEXT, slurm_job_id TEXT)")
        db.execute("INSERT INTO workers VALUES ('running', '123')")
        db.executemany(
            "INSERT INTO scheduler_submissions VALUES (?, ?)",
            [
                ("submitted", "123"),
                ("cancel_requested", "124"),
            ],
        )
    original = database.read_bytes()
    config = tmp_path / "site.toml"
    save_settings(config, {"registry": str(control)})

    def query(command, **kwargs):
        assert command[:3] == ["squeue", "--noheader", "--jobs"]
        assert command[4:] == ["--format", "%T"]
        assert not kwargs["check"] and kwargs["timeout"] == 15
        if scheduler == "failed":
            raise subprocess.CalledProcessError(1, command)
        if scheduler == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if scheduler == "missing":
            raise FileNotFoundError("squeue")
        if scheduler == "invalid":
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr="slurm_load_jobs error: Invalid job id specified\n"
            )
        if scheduler == "mixed":
            if "," in command[3] or command[3] == "123":
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="slurm_load_jobs error: Invalid job id specified\n",
                )
            return subprocess.CompletedProcess(command, 0, stdout="RUNNING\n")
        return subprocess.CompletedProcess(
            command, 0, stdout="" if scheduler == "ended" else scheduler + "\n"
        )

    monkeypatch.setattr(bootstrap.subprocess, "run", query)
    if scheduler in {"ended", "invalid"}:
        bootstrap.check_workers(config)
    else:
        with pytest.raises(RuntimeError):
            bootstrap.check_workers(config)
    assert database.read_bytes() == original


def test_launcher_does_not_replace_another_installation(tmp_path):
    bootstrap_python = tmp_path / ".nro-bootstrap/bin/python"
    bootstrap_python.parent.mkdir(parents=True)
    bootstrap_python.write_text("python")
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("python")
    config = tmp_path / "site.toml"
    config.write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nro").write_text("another installation")
    record = {
        "checkout": str(tmp_path),
        "mode": "shared",
        "environment": str(python.parent.parent),
        "site": str(config),
        "ready": True,
    }
    (tmp_path / bootstrap.RECORD).write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="another command"):
        bootstrap.connect_user(record, bin_dir=bin_dir)
    assert (bin_dir / "nro").read_text() == "another installation"


@pytest.mark.parametrize("header, checksum", [("99", None), ("4", "incorrect")])
def test_failed_download_never_replaces_target(tmp_path, monkeypatch, header, checksum):
    class Response(io.BytesIO):
        url = "https://example.org/file"
        headers = {"Content-Length": header}

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", lambda *a, **k: Response(b"data"))
    target = tmp_path / "image.sif"
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        dependencies.download("https://example.org/file", target, checksum=checksum)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


def test_download_combines_system_and_managed_certificate_authorities(tmp_path, monkeypatch):
    class Context:
        cafile = None

        def load_verify_locations(self, *, cafile):
            self.cafile = cafile

    class Response(io.BytesIO):
        url = "https://example.org/file"
        headers = {"Content-Length": "4"}

    context = Context()
    monkeypatch.setattr(dependencies.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(dependencies.certifi, "where", lambda: "/managed/cacert.pem")

    def download(url, **kwargs):
        assert url == "https://example.org/file"
        assert kwargs["context"] is context
        return Response(b"data")

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", download)

    target = tmp_path / "file"
    dependencies.download("https://example.org/file", target)

    assert context.cafile == "/managed/cacert.pem"
    assert target.read_bytes() == b"data"


def test_container_pull_is_verbose_and_uses_node_local_build_directory(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "image.sif"
    build_directories: list[Path] = []
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )
    monkeypatch.setattr(dependencies.shutil, "which", lambda _command: "/usr/bin/singularity")

    def run(command, **kwargs):
        if "pull" in command:
            assert command[1:3] == ["--verbose", "pull"]
            environment = kwargs["env"]
            assert environment["APPTAINER_TMPDIR"] == environment["SINGULARITY_TMPDIR"]
            build_directory = Path(environment["APPTAINER_TMPDIR"])
            assert build_directory.parent == Path("/tmp")
            build_directories.append(build_directory)
            Path(command[3]).write_bytes(b"container")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    dependencies._install_image("test_image", "docker://example/image@sha256:abc", offline=False)

    assert target.read_bytes() == b"container"
    assert build_directories and not build_directories[0].exists()


def test_pinned_image_rejects_an_outdated_receipt_offline(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.sif"
    target.write_bytes(b"container")
    receipt = target.with_name(target.name + ".receipt.json")
    receipt.write_text(
        json.dumps({"sha256": dependencies.hashlib.sha256(b"container").hexdigest()})
    )
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )

    with pytest.raises(RuntimeError, match="cannot replace outdated"):
        dependencies._install_image(
            "test_image",
            "https://example.org/image.sif",
            offline=True,
            checksum=dependencies.hashlib.sha256(b"new container").hexdigest(),
        )


def test_pinned_image_update_atomically_replaces_an_old_release(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.sif"
    old = b"old container"
    new = b"new container"
    target.write_bytes(old)
    receipt = target.with_name(target.name + ".receipt.json")
    receipt.write_text(json.dumps({"sha256": dependencies.hashlib.sha256(old).hexdigest()}))
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )
    monkeypatch.setattr(dependencies.shutil, "which", lambda _command: "/usr/bin/singularity")
    monkeypatch.setattr(
        dependencies,
        "download",
        lambda _source, destination, *, checksum: destination.write_bytes(new),
    )
    monkeypatch.setattr(dependencies, "run_probe", lambda _command: None)

    dependencies._install_image(
        "test_image",
        "https://example.org/image.sif",
        offline=False,
        checksum=dependencies.hashlib.sha256(new).hexdigest(),
    )

    assert target.read_bytes() == new
    assert json.loads(receipt.read_text())["sha256"] == dependencies.hashlib.sha256(new).hexdigest()


def test_oci_image_update_replaces_an_old_pinned_source(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.sif"
    old = b"old container"
    new = b"new container"
    target.write_bytes(old)
    receipt = target.with_name(target.name + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "source": "docker://example/image@sha256:old",
                "sha256": dependencies.hashlib.sha256(old).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )
    monkeypatch.setattr(dependencies.shutil, "which", lambda _command: "/usr/bin/singularity")
    monkeypatch.setattr(dependencies, "run_probe", lambda _command: None)

    def run(command, **_options):
        if "pull" in command:
            Path(command[3]).write_bytes(new)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    source = "docker://example/image@sha256:new"

    dependencies._install_image("test_image", source, offline=False)

    document = json.loads(receipt.read_text())
    assert target.read_bytes() == new
    assert document["source"] == source
    assert document["sha256"] == dependencies.hashlib.sha256(new).hexdigest()


def test_oci_image_without_receipt_is_adopted_after_inspection(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.sif"
    payload = b"existing container"
    target.write_bytes(payload)
    calls = []
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )
    monkeypatch.setattr(dependencies, "run_probe", lambda command: calls.append(command))
    source = "docker://example/image@sha256:current"

    dependencies._install_image("test_image", source, offline=False)

    receipt = json.loads(target.with_name(target.name + ".receipt.json").read_text())
    assert target.read_bytes() == payload
    assert calls == [["singularity", "inspect", str(target)]]
    assert receipt["source"] == source
    assert receipt["sha256"] == dependencies.hashlib.sha256(payload).hexdigest()


def test_oci_image_rejects_an_old_pinned_source_offline(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.sif"
    payload = b"container"
    target.write_bytes(payload)
    target.with_name(target.name + ".receipt.json").write_text(
        json.dumps(
            {
                "source": "docker://example/image@sha256:old",
                "sha256": dependencies.hashlib.sha256(payload).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", "test_image": str(target)}, {}),
    )

    with pytest.raises(RuntimeError, match="cannot replace outdated"):
        dependencies._install_image("test_image", "docker://example/image@sha256:new", offline=True)


@pytest.mark.parametrize(
    ("key", "verifier"),
    [
        ("freesurfer", "verify_freesurfer_image"),
        ("fastsurfer", "verify_fastsurfer_image"),
    ],
)
def test_oci_image_update_replaces_an_invalid_pinned_identity(
    tmp_path, monkeypatch, key, verifier
) -> None:
    target = tmp_path / "image.sif"
    old = b"old container"
    new = b"new container"
    source = "docker://example/image@sha256:current"
    target.write_bytes(old)
    receipt = target.with_name(target.name + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "source": source,
                "sha256": dependencies.hashlib.sha256(old).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"runtime": "singularity", key: str(target)}, {}),
    )
    monkeypatch.setattr(dependencies.shutil, "which", lambda _command: "/usr/bin/singularity")
    monkeypatch.setattr(
        dependencies,
        verifier,
        lambda _runtime, _path: (_ for _ in ()).throw(RuntimeError("unexpected digest")),
    )
    monkeypatch.setattr(dependencies, "run_probe", lambda _command: None)

    def run(command, **_options):
        if "pull" in command:
            Path(command[3]).write_bytes(new)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(dependencies.subprocess, "run", run)

    dependencies._install_image(key, source, offline=False)

    assert target.read_bytes() == new
    assert json.loads(receipt.read_text())["source"] == source


def test_fastsurfer_image_is_standard() -> None:
    assert "freesurfer/freesurfer@sha256:" in dependencies.required_images()["freesurfer"]
    assert dependencies.required_images()["fastsurfer"].endswith(dependencies.FASTSURFER_OCI_DIGEST)


def test_container_identity_checks_use_metadata_without_starting_images(
    tmp_path, monkeypatch
) -> None:
    image = tmp_path / "image.sif"
    image.write_bytes(b"image")
    calls = []

    def inspect(command, **_options):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "data": {
                        "attributes": {
                            "labels": {
                                "org.opencontainers.image.revision": "cdfccea",
                            }
                        }
                    }
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(dependencies.subprocess, "run", inspect)

    receipt = image.with_name(image.name + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "source": dependencies.IMAGES["fastsurfer"],
                "sha256": dependencies.sha256(image),
                "file_identity": dependencies._file_identity(image),
            }
        )
    )
    dependencies.verify_fastsurfer_image("singularity", image)
    receipt.write_text(
        json.dumps(
            {
                "source": dependencies.IMAGES["freesurfer"],
                "sha256": dependencies.sha256(image),
                "file_identity": dependencies._file_identity(image),
            }
        )
    )
    dependencies.verify_freesurfer_image("singularity", image)

    assert len(calls) == 1
    assert calls[0][1:3] == ["inspect", "--json"]


def test_image_receipt_caches_and_rechecks_filesystem_identity(tmp_path, monkeypatch) -> None:
    image = tmp_path / "image.sif"
    image.write_bytes(b"image")
    receipt = tmp_path / "image.sif.receipt.json"
    receipt.write_text(json.dumps({"sha256": dependencies.sha256(image)}))
    os.utime(receipt, ns=(1, 1))
    dependencies._SHA256_CACHE.clear()
    calls = []
    checksum = dependencies.sha256

    def counted(path):
        calls.append(path)
        return checksum(path)

    monkeypatch.setattr(dependencies, "sha256", counted)
    dependencies.verify_image_receipt(image, receipt)
    dependencies.verify_image_receipt(image, receipt)
    assert calls == [image]

    image.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="differs from its acquisition receipt"):
        dependencies.verify_image_receipt(image, receipt)


def test_legacy_image_receipt_adopts_an_unchanged_acquisition(tmp_path, monkeypatch) -> None:
    image = tmp_path / "image.sif"
    image.write_bytes(b"image")
    receipt = tmp_path / "image.sif.receipt.json"
    receipt.write_text(json.dumps({"sha256": "already-verified-at-acquisition"}))
    monkeypatch.setattr(
        dependencies,
        "sha256",
        lambda _path: pytest.fail("unchanged legacy acquisition was rehashed"),
    )

    dependencies.verify_image_receipt(image, receipt)

    assert json.loads(receipt.read_text())["file_identity"] == dependencies._file_identity(image)


def test_synthstroke_model_is_installed_for_offline_workers(tmp_path, monkeypatch) -> None:
    payloads = {"config.json": b"config", "model.safetensors": b"weights"}
    digests = {
        name: dependencies.hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    data = tmp_path / "synthstroke"
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"synthstroke_data": str(data)}, {}),
    )
    monkeypatch.setattr(dependencies, "MASKER_RESOURCES", digests)
    monkeypatch.setattr(
        dependencies,
        "SYNTHSTROKE_URLS",
        {name: f"https://example.org/{name}" for name in payloads},
    )
    calls = []

    def acquire(url, target, *, checksum=None, md5=None):
        calls.append((url, target, checksum, md5))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[target.name])

    monkeypatch.setattr(dependencies, "download", acquire)
    dependencies.install_synthstroke_model()
    dependencies.install_synthstroke_model(offline=True)

    assert calls == [
        (f"https://example.org/{name}", data / name, digests[name], None) for name in payloads
    ]


def test_branch_setup_acquires_only_explicit_lesion_resources(monkeypatch) -> None:
    from nro.bin import setup

    calls: list[bool] = []
    checks: list[dict] = []
    monkeypatch.setattr(setup, "installation_record", lambda: {"mode": "branch"})
    monkeypatch.setattr(
        setup,
        "install_lesion_resources",
        lambda *, offline=False: calls.append(offline),
    )
    monkeypatch.setattr(
        setup,
        "check_installation",
        lambda **kwargs: (
            checks.append(kwargs)
            or [{"ok": True, "required": True, "name": "test", "detail": "ok"}]
        ),
    )

    setup._main(["--resources-only", "--with-lesion", "--without-viewer", "--offline"])

    assert calls == [True]
    assert checks[0]["with_viewer"] is False


def test_workbench_installs_container_launchers(isolated_site, tmp_path, monkeypatch):
    command = tmp_path / "workbench/bin_linux64/wb_command"
    runtime = tmp_path / "runtime/bin/apptainer"
    save_settings(
        isolated_site,
        {
            "workbench": str(command),
            "runtime": str(runtime),
            "binds": [str(tmp_path) + ":" + str(tmp_path)],
        },
    )
    acquisitions = []
    probes = []
    monkeypatch.setattr(
        dependencies,
        "_install_image",
        lambda key, source, **options: acquisitions.append((key, source, options)),
    )
    monkeypatch.setattr(dependencies, "run_probe", lambda command: probes.append(command))

    dependencies.install_workbench()
    viewer = command.with_name("wb_view")
    assert acquisitions == [
        (
            "workbench_image",
            dependencies.WORKBENCH_IMAGE_URL,
            {"offline": False, "checksum": dependencies.WORKBENCH_IMAGE_SHA256},
        )
    ]
    assert probes == [[str(command), "-version"]]
    assert command.stat().st_mode & 0o111
    assert viewer.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(command)], check=True)
    subprocess.run(["bash", "-n", str(viewer)], check=True)
    assert "wb_command" in command.read_text()
    assert "wb_view" in viewer.read_text()
    launcher = viewer.read_text()
    assert "XAUTHORITY=/run/nro/xauthority" in launcher
    assert "$HOME/.Xauthority" in launcher
    assert "LIBGL_ALWAYS_SOFTWARE=1" in launcher
    assert "LD_LIBRARY_PATH=" not in launcher
    assert "--no-mount bind-paths" in launcher
    assert "unset APPTAINER_BIND APPTAINER_BINDPATH" in launcher
    assert "SINGULARITY_BIND SINGULARITY_BINDPATH" in launcher
    assert "QT_X11_NO_MITSHM=1" in launcher
    assert f'export PATH={runtime.parent}:"$PATH"' in launcher
    assert str(tmp_path) in command.read_text()
    receipt = json.loads((command.parent.parent / "nro-workbench-container.json").read_text())
    assert receipt["sha256"] == dependencies.WORKBENCH_IMAGE_SHA256


def test_workbench_reuses_an_explicit_native_override(isolated_site, tmp_path, monkeypatch):
    command = tmp_path / "custom/wb_command"
    command.parent.mkdir()
    command.write_text("#!/bin/sh\nexit 0\n")
    command.chmod(0o755)
    save_settings(isolated_site, {"workbench": str(command)})
    probes = []
    monkeypatch.setattr(dependencies, "run_probe", lambda value: probes.append(value))
    monkeypatch.setattr(
        dependencies,
        "_install_image",
        lambda *_args, **_kwargs: pytest.fail("native override must not acquire an image"),
    )

    dependencies.install_workbench(offline=True)

    assert probes == [[str(command), "-version"]]


def test_install_help_from_another_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "nro/site/bootstrap.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--maintain" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("exists", [False, True])
def test_workers_capture_resolved_site_settings(tmp_path, monkeypatch, exists):
    from nro.bin.run import _write_worker_script

    config = tmp_path / "site settings.toml"
    if exists:
        config.write_text("")
    monkeypatch.setattr(site, "site_file", lambda: config)
    registry = SimpleNamespace(
        paths=SimpleNamespace(workers=tmp_path / "workers", control=tmp_path / "control")
    )
    script = _write_worker_script(
        registry,
        bids_root=tmp_path / "BIDS",
        partition="test",
        account=None,
        hours=1,
        memory_gb=4,
        cpus=1,
    ).read_text()
    assert "source_launcher.py" in script
    assert str(config) not in script
    (pinned_site,) = (registry.paths.control / "shared/cache/execution-sites").glob("*.toml")
    assert str(pinned_site) in script
    values = site.read_overrides(pinned_site)
    assert values["bids"] == str(tmp_path / "BIDS")
    assert values["registry"] == str(registry.paths.control)


@pytest.mark.parametrize(
    "url, checksum",
    [
        (dependencies.OSLOM_SOURCE, None),
        (dependencies.OSLOM_SOURCE, "wrong"),
        ("http://example.org/archive", dependencies.OSLOM_SHA256),
    ],
)
def test_http_download_requires_exact_oslom_pin(tmp_path, url, checksum):
    with pytest.raises(ValueError, match="HTTPS"):
        dependencies.download(url, tmp_path / "archive", checksum=checksum)


def test_official_oslom_http_download(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        url = dependencies.OSLOM_SOURCE
        headers = {"Content-Length": "4"}

    monkeypatch.setattr(dependencies.urllib.request, "urlopen", lambda *a, **kw: Response(b"data"))
    monkeypatch.setattr(dependencies, "sha256", lambda path: dependencies.OSLOM_SHA256)
    target = tmp_path / "archive"
    dependencies.download(dependencies.OSLOM_SOURCE, target, checksum=dependencies.OSLOM_SHA256)
    assert target.read_bytes() == b"data"


@pytest.mark.parametrize("failure", [None, "compile", "fit", "escape", "symlink"])
def test_oslom_build_publishes_only_after_validation(isolated_site, tmp_path, monkeypatch, failure):
    target = tmp_path / "installed/oslom_undir"
    save_settings(isolated_site, {"oslom": str(target)})
    monkeypatch.setattr(dependencies.shutil, "which", lambda name: "/usr/bin/g++")

    def archive_download(url, destination, checksum=None):
        assert url == dependencies.OSLOM_SOURCE
        assert checksum == dependencies.OSLOM_SHA256
        with tarfile.open(destination, "w:gz") as archive:
            member = tarfile.TarInfo("../escape" if failure == "escape" else "OSLOM2/example.dat")
            member.size = 4
            if failure == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = "/etc/passwd"
                member.size = 0
            archive.addfile(member, io.BytesIO(b"data"))

    def probe(command, *, cwd=None, **kwargs):
        if "--version" in command:
            return "test compiler"
        if "-O3" in command:
            if failure == "compile":
                raise RuntimeError("compile failed")
            (cwd / "oslom_undir").write_bytes(b"binary")
        elif failure != "fit":
            output = cwd / "example.dat_oslo_files/tp"
            output.parent.mkdir()
            output.write_text("#module 0\n0 1 2\n")
        return ""

    monkeypatch.setattr(dependencies, "download", archive_download)
    monkeypatch.setattr(dependencies, "run_probe", probe)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            dependencies.install_oslom()
        assert not target.exists()
        assert not target.with_name(target.name + ".receipt.json").exists()
    else:
        dependencies.install_oslom()
        receipt = json.loads(target.with_name(target.name + ".receipt.json").read_text())
        assert receipt["source_sha256"] == dependencies.OSLOM_SHA256
        assert receipt["sha256"] == dependencies.sha256(target)
        assert target.stat().st_mode & 0o111
        monkeypatch.setattr(
            dependencies, "download", lambda *a, **kw: pytest.fail("Unexpected download")
        )
        dependencies.install_oslom(offline=True)
    assert not list(target.parent.glob(".oslom-*"))


def test_missing_oslom_offline_does_not_create_directories(isolated_site, tmp_path):
    target = tmp_path / "absent/oslom_undir"
    save_settings(isolated_site, {"oslom": str(target)})
    with pytest.raises(RuntimeError, match="offline"):
        dependencies.install_oslom(offline=True)
    assert not target.parent.exists()


def test_container_probe_explains_nested_namespace_restriction(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "run_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Could not write info to setgroups: Permission denied")
        ),
    )

    with pytest.raises(RuntimeError, match="current process sandbox"):
        dependencies.run_container_probe(["singularity", "exec"])


def test_container_probe_preserves_other_runtime_failures(monkeypatch):
    monkeypatch.setattr(
        dependencies,
        "run_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("image is corrupt")),
    )

    with pytest.raises(RuntimeError, match="image is corrupt"):
        dependencies.run_container_probe(["singularity", "exec"])


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize("entry", ["bootstrap", "setup", "paths"])
def test_setup_interrupts_exit_without_tracebacks(monkeypatch, capsys, entry, interruption):
    from nro.bin import paths, setup

    module = {"bootstrap": bootstrap, "setup": setup, "paths": paths}[entry]

    def interrupt(*args, **kwargs):
        raise interruption()

    monkeypatch.setattr(module, "edit_settings" if entry == "paths" else "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 130
    output = capsys.readouterr().err
    assert "cancelled" in output
    assert "Traceback" not in output


def test_interrupted_path_prompt_preserves_settings(isolated_site, monkeypatch):
    original = isolated_site.read_bytes()
    monkeypatch.setattr(site_setup.sys.stdin, "isatty", lambda: True)

    def interrupt(prompt):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(KeyboardInterrupt):
        edit_settings()
    assert isolated_site.read_bytes() == original


def test_bootstrap_propagates_child_cancellation(monkeypatch):
    def interrupt(*args):
        raise subprocess.CalledProcessError(130, ["setup"])

    monkeypatch.setattr(bootstrap, "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        bootstrap.main([])
    assert error.value.code == 130


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("p", "personal"),
        ("personal", "personal"),
        ("s", "shared"),
        ("shared", "shared"),
        ("b", "branch"),
        ("branch", "branch"),
    ],
)
def test_installation_role_abbreviations(value, expected):
    assert bootstrap._installation_mode(value) == expected


def test_invalid_installation_role_is_rejected():
    with pytest.raises(argparse.ArgumentTypeError, match="personal \\(p\\)"):
        bootstrap._installation_mode("invalid")


@pytest.mark.parametrize("entry", ["bootstrap", "setup"])
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("kind", ["signal", "exit"])
def test_only_outer_setup_reports_cancellation(monkeypatch, capsys, entry, child, kind):
    from nro.bin import setup

    module = bootstrap if entry == "bootstrap" else setup
    if child:
        monkeypatch.setenv("NRO_SETUP_CHILD", "1")
    else:
        monkeypatch.delenv("NRO_SETUP_CHILD", raising=False)

    def interrupt(*args, **kwargs):
        if kind == "signal":
            raise KeyboardInterrupt()
        if entry == "bootstrap":
            raise subprocess.CalledProcessError(130, ["setup"])
        raise SystemExit(130)

    monkeypatch.setattr(module, "_main", interrupt)
    with pytest.raises(SystemExit) as error:
        module.main([])
    assert error.value.code == 130
    assert capsys.readouterr().err.count("Installation cancelled") == (0 if child else 1)
