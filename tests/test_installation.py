from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nro.configuration import site
from nro.configuration.store import ConfigStore
from nro.engine import bootstrap, dependencies, shared_installation, site_setup
from nro.engine.site_setup import edit_settings, save_settings


@pytest.fixture
def isolated_site(tmp_path, monkeypatch):
    path = tmp_path / "site.toml"
    path.write_text("")
    monkeypatch.setenv("NRO_SITE_CONFIG", str(path))
    for key in site.ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(site, "installation_record", lambda: {})
    return path


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


def test_pycicada_executable_resolves_from_site_settings(isolated_site, tmp_path):
    executable = tmp_path / "pycicada/bin/cicada-python"
    save_settings(isolated_site, {"pycicada": str(executable)})

    values, _sources = site.settings()

    assert values["pycicada"] == str(executable)
    assert site.resolve_resources("site:pycicada") == str(executable)


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
    from nro.configuration.definitions import create_store
    from nro.configuration.site import read_site_definition

    definitions = create_store(tmp_path / "definitions")
    isolated_site.write_text(f'definitions = "{definitions}"\n')
    destination = save_settings(isolated_site, {"work": "/updated/work"})
    assert destination == definitions / "site/site.yml"
    assert site.read_overrides(isolated_site) == {"definitions": str(definitions)}
    protected, _ = read_site_definition(definitions)
    assert protected["work"] == "/updated/work"


def test_installation_migrates_legacy_site_and_bidsify_settings(isolated_site, tmp_path):
    from nro.configuration.definitions import create_store
    from nro.configuration.site import read_site_definition

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
    migrated_profile = yaml.safe_load(profile_path.read_text())
    assert set(migrated_profile["scanplans"]) == {"parser"}
    assert "servers" not in migrated_profile


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


def test_shared_site_ignores_personal_environment(isolated_site, monkeypatch):
    save_settings(isolated_site, {"bids": "/shared/BIDS"})
    monkeypatch.setattr(
        site, "installation_record", lambda: {"mode": "shared", "site": str(isolated_site)}
    )
    monkeypatch.setattr("nro.engine.site_setup.installation_record", site.installation_record)
    assert site.settings()[0]["bids"] == "/shared/BIDS"
    with pytest.raises(ValueError, match="maintain"):
        edit_settings(["bids=/other"])
    monkeypatch.setenv("NRO_SITE_CONFIG", "/other/site.toml")
    with pytest.raises(ValueError, match="shared installation"):
        site.site_file()


def test_shared_onboarding_never_syncs_or_mutates_checkout(tmp_path, monkeypatch):
    root = tmp_path / "shared"
    root.mkdir()
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
    monkeypatch.setattr("nro.orchestration.releases.tagged_source", lambda checkout: ())
    events = []
    monkeypatch.setattr(
        shared_installation,
        "prepare_pool",
        lambda registry, **options: events.append(("drain", options["checkout"])),
    )

    def publish(checkout, registry, *, installation):
        events.append(("publish", checkout))
        bootstrap.write_record(checkout / bootstrap.RECORD, installation)

    monkeypatch.setattr(shared_installation, "publish", publish)
    commands = []
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **options: commands.append(command),
    )
    monkeypatch.setattr(bootstrap, "connect_user", lambda *args, **options: None)

    bootstrap.main(["--maintain", "--offline"])

    assert events == [("drain", root), ("publish", root)]
    assert "--prepared-maintenance" in commands[1]
    saved = json.loads((root / bootstrap.RECORD).read_text())
    assert saved["ready"] is True
    assert Path(saved["environment"]).parent == root / ".nro-environments"
    assert Path(saved["environment"]).name.startswith("candidate-")


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
    monkeypatch.setattr("nro.orchestration.releases.tagged_source", lambda checkout: ())
    monkeypatch.setattr(shared_installation, "prepare_pool", lambda *args, **options: None)
    monkeypatch.setattr(
        shared_installation,
        "publish",
        lambda *args, **options: pytest.fail("Invalid candidate was published"),
    )
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(bootstrap.subprocess, "run", run)

    with pytest.raises(SystemExit) as stopped:
        bootstrap.main(["--maintain", "--offline"])

    assert stopped.value.code == 1
    assert record_path.read_bytes() == original
    candidate = Path(calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"])
    assert candidate != active
    assert candidate.parent == root / ".nro-environments"


@pytest.mark.parametrize("without_oslom", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("without_marss", [False, True])
@pytest.mark.parametrize("with_lesion", [False, True])
def test_personal_setup_installs_selected_extras(
    tmp_path, monkeypatch, without_oslom, existing, without_marss, with_lesion
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
        ]
    )
    assert "sync" in calls[0][0] and "--frozen" in calls[0][0]
    assert ("oslom" in calls[0][0]) is not without_oslom
    assert ("marss" in calls[0][0]) is not without_marss
    assert ("lesion" in calls[0][0]) is with_lesion
    assert calls[0][1]["cwd"] == root
    assert calls[0][1]["env"]["UV_PROJECT_ENVIRONMENT"] == str(root / ".nro-env")
    assert ("--without-oslom" in calls[1][0]) is without_oslom
    assert json.loads((root / bootstrap.RECORD).read_text())["ready"]


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
    "scheduler", ["ended", "RUNNING", "PENDING", "COMPLETING", "failed", "timeout", "missing"]
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
        assert command == ["squeue", "--noheader", "--jobs", "123,124", "--format", "%T"]
        assert kwargs["check"] and kwargs["timeout"] == 15
        if scheduler == "failed":
            raise subprocess.CalledProcessError(1, command)
        if scheduler == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if scheduler == "missing":
            raise FileNotFoundError("squeue")
        return subprocess.CompletedProcess(
            command, 0, stdout="" if scheduler == "ended" else scheduler + "\n"
        )

    monkeypatch.setattr(bootstrap.subprocess, "run", query)
    if scheduler == "ended":
        bootstrap.check_workers(config)
    else:
        with pytest.raises(RuntimeError):
            bootstrap.check_workers(config)
    assert database.read_bytes() == original


def test_launcher_does_not_replace_another_installation(tmp_path):
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


def test_lesion_resources_are_explicit_and_checkpointed(tmp_path, monkeypatch) -> None:
    assert dependencies.required_images()["fastsurfer"].endswith(dependencies.FASTSURFER_OCI_DIGEST)
    assert dependencies.required_images(with_lesion=True)["fastsurfer"].endswith(
        dependencies.FASTSURFER_OCI_DIGEST
    )

    payload = b"pinned checkpoint"
    digest = dependencies.hashlib.sha256(payload).hexdigest()
    data = tmp_path / "lit-data"
    monkeypatch.setattr(
        dependencies,
        "settings",
        lambda: ({"fastsurfer_data": str(data)}, {}),
    )
    monkeypatch.setattr(dependencies, "NEUROLIT_CHECKPOINTS", {"model.pt": digest})
    monkeypatch.setattr(
        dependencies,
        "NEUROLIT_URLS",
        {"model.pt": "https://example.org/model.pt"},
    )
    calls = []

    def acquire(url, target, *, checksum=None, md5=None):
        calls.append((url, target, checksum, md5))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    monkeypatch.setattr(dependencies, "download", acquire)
    dependencies.install_neurolit_checkpoints()
    dependencies.install_neurolit_checkpoints(offline=True)

    target = data / "LIT" / "weights" / "model.pt"
    assert target.read_bytes() == payload
    assert calls == [("https://example.org/model.pt", target, digest, None)]


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
    monkeypatch.setattr(setup, "installation_record", lambda: {"mode": "branch"})
    monkeypatch.setattr(
        setup,
        "install_lesion_resources",
        lambda *, offline=False: calls.append(offline),
    )
    monkeypatch.setattr(
        setup,
        "check_installation",
        lambda **_kwargs: [{"ok": True, "required": True, "name": "test", "detail": "ok"}],
    )

    setup._main(["--resources-only", "--with-lesion", "--offline"])

    assert calls == [True]


def test_archive_escape_is_rejected(tmp_path):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="escapes"):
        dependencies.extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()


def test_workbench_installs_wrapper_and_reuses_it(isolated_site, tmp_path, monkeypatch):
    command = tmp_path / "workbench/bin_linux64/wb_command"
    save_settings(isolated_site, {"workbench": str(command)})

    def archive_download(url, target, checksum=None):
        assert checksum == dependencies.WORKBENCH_SHA256["linux64"]
        with zipfile.ZipFile(target, "w") as zipped:
            zipped.writestr("workbench/bin_linux64/wb_command", "wrapper")
            zipped.writestr("workbench/exe_linux64/wb_command", "binary")

    calls = []
    monkeypatch.setattr(dependencies, "download", archive_download)
    monkeypatch.setattr(dependencies.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(dependencies.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu"})
    monkeypatch.setattr(dependencies, "run_probe", lambda command: calls.append(command))
    dependencies.install_workbench()
    assert command.read_text() == "wrapper"
    assert "/bin_linux64/" in calls[0][0]
    monkeypatch.setattr(
        dependencies, "download", lambda *a, **kw: pytest.fail("Unexpected download")
    )
    dependencies.install_workbench(offline=True)


def test_install_help_from_another_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "install"), "--help"],
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
    assert capsys.readouterr().err.count("Setup cancelled") == (0 if child else 1)
