"""Acquire versioned resources and check the configured execution environment."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import certifi

from nro.definitions.hardware import GRADIENT_UNWARP_IMAGE, gradient_unwarping_configured
from nro.engine.io import atomic_output_path, atomic_write_json, atomic_write_text
from nro.modules.anat.lesion_policy import (
    MASKER_MODEL,
    MASKER_RESOURCES,
    MASKER_REVISION,
    NEUROLIT_CHECKPOINTS,
)
from nro.modules.anat.policy import FASTSURFER_OCI_DIGEST
from nro.site.configuration import settings

FASTSURFER_IMAGE = "docker://deepmi/fastsurfer@" + FASTSURFER_OCI_DIGEST
IMAGES = {
    "qunex": "docker://qunex/qunex_suite@sha256:a06befbb64f93ab289bbef94d1d00bf957c7cdff920f35e107f90b186ff9f09d",
    "synthstrip": "docker://freesurfer/synthstrip@sha256:801924ea011be040346c0e68f9c25d175ab5b869afbd156560b01fb74059f1b1",
    "synbold": "docker://ytzero/synbold-disco@sha256:18814dd2f419dfe8375632cf9239a0fb31a1300a2bfe66d234e599450af66555",
    "gradient_unwarp": GRADIENT_UNWARP_IMAGE,
    "freesurfer": "docker://freesurfer/freesurfer@sha256:10b6468cbd9fcd2db3708f4651d59ad75d4da849a2c5d8bb6dba217f08b8c46b",
    "fastsurfer": FASTSURFER_IMAGE,
}
NEUROLIT_URLS = {
    name: f"https://zenodo.org/api/records/14510136/files/{name}/content"
    for name in NEUROLIT_CHECKPOINTS
}
SYNTHSTROKE_URLS = {
    name: f"https://huggingface.co/{MASKER_MODEL}/resolve/{MASKER_REVISION}/{name}?download=true"
    for name in MASKER_RESOURCES
}
_SHA256_CACHE: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}


def required_images() -> dict[str, str]:
    """Return images needed by the site's configured scientific features."""
    images = dict(IMAGES)
    if not gradient_unwarping_configured():
        images.pop("gradient_unwarp")
    return images


def install_neurolit_checkpoints(*, offline: bool = False) -> None:
    """Acquire and verify the pinned NeuroLIT inpainting checkpoints."""
    values, _ = settings()
    root = Path(values["fastsurfer_data"]) / "LIT" / "weights"
    for name, expected in NEUROLIT_CHECKPOINTS.items():
        target = root / name
        if target.is_file() and sha256(target) == expected:
            print(f"Reuse NeuroLIT checkpoint: {target}")
            continue
        if offline:
            raise RuntimeError(f"Offline setup cannot obtain NeuroLIT checkpoint: {target}")
        with resource_lock(target):
            if target.is_file() and sha256(target) == expected:
                continue
            print(f"Downloading NeuroLIT checkpoint {name}", flush=True)
            download(NEUROLIT_URLS[name], target, checksum=expected)


def install_synthstroke_model(*, offline: bool = False) -> None:
    """Acquire and verify the pinned SynthStroke model for offline workers."""
    values, _ = settings()
    root = Path(values["synthstroke_data"])
    for name, expected in MASKER_RESOURCES.items():
        target = root / name
        if target.is_file() and sha256(target) == expected:
            print(f"Reuse SynthStroke resource: {target}")
            continue
        if offline:
            raise RuntimeError(f"Offline setup cannot obtain SynthStroke resource: {target}")
        with resource_lock(target):
            if target.is_file() and sha256(target) == expected:
                continue
            print(f"Downloading SynthStroke resource {name}", flush=True)
            download(SYNTHSTROKE_URLS[name], target, checksum=expected)


LICENSE_HELP = "Register at https://surfer.nmr.mgh.harvard.edu/registration.html and set license=/path/to/license.txt"
QUNEX_TERMS = "https://qunex.yale.edu/access/"
WORKBENCH_IMAGE_URL = (
    "https://github.com/CLiMBlab-Stanford/nro-workbench/releases/download/"
    "v2.2.1-7/nro-workbench.sif"
)
WORKBENCH_IMAGE_SHA256 = "04df826d74fb71c6c873216707563b1dd1df280d495413f9cd70d1e7fe2421cd"
OSLOM_SOURCE = "http://www.oslom.org/code/OSLOM2.tar.gz"
OSLOM_SHA256 = "3d1ff087449dbb715d1f74f37a5e064d0b61bfba29d64411e4f454d407d96d5e"
MANAGED_RUNTIME = {
    "name": "apptainer",
    "version": "1.4.5-1",
    "distribution": "el8",
    "architecture": "x86_64",
    "artifact": "apptainer-1.4.5-1-el8-x86_64.tar.gz",
    "sha256": "cd6c28a45e32dc1dc365dc8c0b2eb94ae1ebde15be2b52d46f9a4032163bf939",
    "source": (
        "https://github.com/CLiMBlab-Stanford/nro/releases/download/"
        "resources-apptainer-1.4.5-1-el8-x86_64-v1/"
        "apptainer-1.4.5-1-el8-x86_64.tar.gz"
    ),
    "upstream_installer_sha256": (
        "33d416ca870fdfcfc6b5fd8791f02bf041d742a5b718d015a9a1cf61aa1b30dd"
    ),
    "upstream_rpm_sha256": ("d63fffcce94e472a838de6d0eef1e44b6ecb6265656768e087f3799adb36b356"),
}
RUNTIME_MANIFEST = ".nro-runtime.json"


def template_catalog() -> dict:
    """Read the installed mapping of template paths to checksums and S3 versions."""
    path = Path(__file__).resolve().parents[1] / "definitions/template_resources.json"
    return json.loads(path.read_text())


def verify_template(path: Path, md5: str) -> None:
    """Check a template against its MD5 identity; mismatch raises RuntimeError."""
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "md5").hexdigest()
    if actual != md5:
        raise RuntimeError(f"Template differs from the pinned resource: {path}")


def ensure_fsaverage6_midthickness(root: Path) -> None:
    """Create the 41k midthickness surfaces from pinned white and pial geometry."""
    import nibabel as nib
    import numpy as np

    directory = Path(root) / "tpl-fsaverage"
    for hemisphere in ("L", "R"):
        white_path = directory / f"tpl-fsaverage_hemi-{hemisphere}_den-41k_white.surf.gii"
        pial_path = directory / f"tpl-fsaverage_hemi-{hemisphere}_den-41k_pial.surf.gii"
        target = directory / f"tpl-fsaverage_hemi-{hemisphere}_den-41k_midthickness.surf.gii"
        white = nib.load(str(white_path))
        pial = nib.load(str(pial_path))
        white_points = white.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        pial_points = pial.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
        if len(white_points) != 1 or len(pial_points) != 1:
            raise RuntimeError("fsaverage6 white and pial surfaces require one coordinate array")
        coordinates = (
            np.asarray(white_points[0].data, dtype=np.float64)
            + np.asarray(pial_points[0].data, dtype=np.float64)
        ) / 2.0
        if target.is_file():
            existing = nib.load(str(target)).get_arrays_from_intent("NIFTI_INTENT_POINTSET")
            if (
                len(existing) != 1
                or not np.allclose(existing[0].data, coordinates, rtol=0, atol=1e-5)
                or existing[0].meta.get("AnatomicalStructureSecondary") != "MidThickness"
            ):
                raise RuntimeError(f"Derived fsaverage6 surface differs from its inputs: {target}")
            continue
        result = deepcopy(white)
        pointset = result.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
        pointset.data[:] = coordinates
        pointset.meta["AnatomicalStructureSecondary"] = "MidThickness"
        for data_array in result.darrays:
            if "Name" in data_array.meta:
                data_array.meta["Name"] = target.name
        with atomic_output_path(target) as staged:
            nib.save(result, str(staged))


def _runtime_root(values: dict) -> Path:
    identity = MANAGED_RUNTIME["sha256"][:16]
    return (
        Path(values["images"]).expanduser().resolve().parent
        / "runtimes"
        / f"apptainer-{MANAGED_RUNTIME['version']}-{MANAGED_RUNTIME['architecture']}-{identity}"
    )


def _runtime_inventory(root: Path) -> dict[str, str]:
    """Return checksums for regular files in a managed runtime tree."""
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != RUNTIME_MANIFEST
    }


def _validate_runtime_tree(root: Path, *, complete: bool, verify_inventory: bool = False) -> dict:
    """Validate a managed runtime and return its completion manifest."""
    executable = root / "bin/apptainer"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(
            f"Managed Apptainer executable is missing or not executable: {executable}"
        )
    manifest_path = root / RUNTIME_MANIFEST
    if not manifest_path.is_file():
        if complete:
            raise RuntimeError(f"Managed Apptainer installation is incomplete: {root}")
        manifest = {}
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid managed Apptainer manifest: {manifest_path}") from error
    expected = {
        "format": 1,
        "specification": MANAGED_RUNTIME,
        "executable": "bin/apptainer",
    }
    if complete and any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Managed Apptainer identity differs from this nro release: {root}")
    if complete:
        executable_digest = manifest.get("executable_sha256")
        if not isinstance(executable_digest, str) or sha256(executable) != executable_digest:
            raise RuntimeError(f"Managed Apptainer executable failed verification: {executable}")
        if verify_inventory and manifest.get("inventory") != _runtime_inventory(root):
            raise RuntimeError(f"Managed Apptainer file inventory failed verification: {root}")
    output = run_probe([str(executable), "--version"])
    if "1.4.5" not in output:
        raise RuntimeError(f"Managed Apptainer returned an unexpected version: {output}")
    return {**expected, "executable_sha256": sha256(executable), "probe": output}


def validate_managed_runtime(executable: str | Path) -> None:
    """Verify a managed runtime when the executable belongs to one."""
    path = Path(executable).expanduser()
    if not path.is_absolute():
        return
    manifest = path.parent.parent / RUNTIME_MANIFEST
    if manifest.exists():
        _validate_runtime_tree(manifest.parent, complete=True)


def runtime_diagnostic(value: str) -> str:
    """Describe why a configured container executable cannot be used."""
    host = platform.node() or "unknown host"
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_symlink() and not path.exists():
            detail = "a broken symbolic link"
        elif not path.exists():
            detail = "missing"
        elif not path.is_file():
            detail = "not a regular file"
        elif not os.access(path, os.X_OK):
            detail = "not executable"
        else:
            detail = "present but unavailable to the process"
        return f"Container runtime {path} is {detail} on {host}"
    return f"Container runtime {value!r} is not on PATH on {host}"


def _extract_runtime(archive: Path, destination: Path) -> None:
    """Extract a verified runtime archive without accepting unsafe members."""
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(destination, filter="data")


def _seal_runtime_tree(root: Path) -> None:
    """Make a completed runtime readable and executable but not mutable in place."""
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o755 if os.access(path, os.X_OK) else 0o644)
    root.chmod(0o755)


def install_runtime(*, offline=False) -> None:
    """Select an explicit external runtime or install managed Apptainer."""
    from nro.site.configuration import read_overrides, site_file
    from nro.site.setup import save_settings

    values, _ = settings()
    configured = values["runtime"]
    configured_path = Path(configured).expanduser()
    if configured_path.is_absolute():
        resolved = shutil.which(str(configured_path))
        if resolved is None:
            raise RuntimeError(runtime_diagnostic(configured))
        validate_managed_runtime(resolved)
        print(f"Reuse container runtime: {resolved}")
        return
    if configured not in {"apptainer", "singularity"}:
        resolved = shutil.which(configured)
        if resolved is None:
            raise RuntimeError(runtime_diagnostic(configured))
        found = str(Path(resolved).resolve())
        print(f"Selected external container runtime: {found}")
    else:
        if platform.system() != "Linux" or platform.machine() != MANAGED_RUNTIME["architecture"]:
            raise RuntimeError(
                "Managed Apptainer supports Linux x86_64; configure execution.runtime "
                "with an absolute external executable on this platform"
            )
        root = _runtime_root(values)
        artifact = root.parent / "artifacts" / MANAGED_RUNTIME["artifact"]
        with resource_lock(root):
            if root.exists():
                _validate_runtime_tree(root, complete=True, verify_inventory=True)
            else:
                if not artifact.is_file() or sha256(artifact) != MANAGED_RUNTIME["sha256"]:
                    if offline:
                        raise RuntimeError(
                            "Offline setup cannot obtain managed Apptainer artifact: "
                            + str(artifact)
                        )
                    print(f"Downloading managed Apptainer {MANAGED_RUNTIME['version']}", flush=True)
                    download(
                        MANAGED_RUNTIME["source"],
                        artifact,
                        checksum=MANAGED_RUNTIME["sha256"],
                    )
                with tempfile.TemporaryDirectory(
                    dir=root.parent, prefix=".apptainer-stage-"
                ) as temporary:
                    staged = Path(temporary) / "runtime"
                    _extract_runtime(artifact, staged)
                    manifest = _validate_runtime_tree(staged, complete=False)
                    manifest["inventory"] = _runtime_inventory(staged)
                    atomic_write_json(staged / RUNTIME_MANIFEST, manifest)
                    _validate_runtime_tree(staged, complete=True, verify_inventory=True)
                    _seal_runtime_tree(staged)
                    staged.rename(root)
                    _validate_runtime_tree(root, complete=True, verify_inventory=True)
        found = str(root / "bin/apptainer")
        print(f"Selected managed container runtime: {found}")
    overrides = read_overrides(site_file())
    overrides["runtime"] = found
    save_settings(site_file(), overrides)


@contextmanager
def resource_lock(path: Path):
    """Serialize acquisition of a resource with a sibling file lock.

    Create the parent and lock file; hold the advisory lock for the context.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".install.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def sha256(path: Path) -> str:
    """Compute a file checksum in bounded chunks."""
    path = path.resolve()
    before = path.stat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    cached = _SHA256_CACHE.get(path)
    if cached is not None and cached[0] == identity:
        return cached[1]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"File changed during checksum verification: {path}")
    value = digest.hexdigest()
    _SHA256_CACHE[path] = (identity, value)
    return value


def _file_identity(path: Path) -> dict[str, int]:
    """Return the filesystem identity used to reuse a prior checksum."""
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def verify_image_receipt(path: Path, receipt: Path) -> None:
    """Validate an acquired image, caching its verified filesystem identity."""
    document = json.loads(receipt.read_text(encoding="utf-8"))
    identity = _file_identity(path)
    if document.get("file_identity") == identity:
        return
    if "file_identity" not in document and receipt.stat().st_mtime_ns >= path.stat().st_ctime_ns:
        # Legacy receipts were written only after the acquired image's checksum
        # passed. An older file ctime therefore proves that the recorded bytes
        # have not been modified since that verification.
        document["file_identity"] = identity
        atomic_write_json(receipt, document)
        return
    print(f"Verify container checksum: {path}", flush=True)
    if sha256(path) != document["sha256"]:
        raise RuntimeError(f"Installed image differs from its acquisition receipt: {path}")
    document["file_identity"] = identity
    atomic_write_json(receipt, document)


def verify_sha256(path: Path, expected: str, label: str) -> None:
    """Require one file to exist and match its pinned SHA-256 digest."""
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Missing or invalid {label}: {path}")


def _image_labels(runtime: str, path: Path) -> dict[str, str]:
    """Read container identity without starting the image on the login host."""
    result = subprocess.run(
        [runtime, "inspect", "--json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    document = json.loads(result.stdout)
    labels = document["data"]["attributes"]["labels"]
    if not isinstance(labels, dict):
        raise RuntimeError(f"Container has invalid identity labels: {path}")
    return labels


def verify_fastsurfer_image(runtime: str, path: Path) -> None:
    """Require the exact pinned FastSurfer image metadata."""
    labels = _image_labels(runtime, path)
    if labels.get("org.opencontainers.image.base.digest") != FASTSURFER_OCI_DIGEST:
        raise RuntimeError("FastSurfer image has an unexpected OCI base digest")
    if labels.get("org.opencontainers.image.revision") != "cdfccea":
        raise RuntimeError("FastSurfer image has an unexpected source revision")


def verify_freesurfer_image(runtime: str, path: Path) -> None:
    """Require the exact digest-pinned conventional FreeSurfer image metadata."""
    labels = _image_labels(runtime, path)
    expected = IMAGES["freesurfer"].rsplit("@", 1)[1]
    if labels.get("org.opencontainers.image.base.digest") != expected:
        raise RuntimeError("FreeSurfer image has an unexpected OCI base digest")


def download(
    url: str, target: Path, *, checksum: str | None = None, md5: str | None = None
) -> None:
    """Verify and publish a download, allowing HTTP only for pinned official OSLOM."""
    official_oslom = url == OSLOM_SOURCE and checksum == OSLOM_SHA256
    if not url.startswith("https://") and not official_oslom:
        raise ValueError("Downloads require HTTPS")
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    with atomic_output_path(target) as staged:
        with (
            urllib.request.urlopen(url, timeout=60, context=context) as response,
            staged.open("wb") as stream,
        ):
            if not response.url.startswith("https://") and not (
                official_oslom and response.url == OSLOM_SOURCE
            ):
                raise ValueError("Refusing a download redirected to an insecure URL")
            progress = time.monotonic()
            for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                stream.write(chunk)
                if time.monotonic() - progress >= 5:
                    print(f"  {target.name}: {stream.tell() / 2**20:.0f} MiB", flush=True)
                    progress = time.monotonic()
            expected = response.headers.get("Content-Length")
        if expected is not None and staged.stat().st_size != int(expected):
            raise RuntimeError(f"Incomplete transfer from {url}")
        if checksum is not None and sha256(staged) != checksum:
            raise RuntimeError(f"Checksum mismatch for {url}")
        if md5 is not None:
            with staged.open("rb") as stream:
                actual = hashlib.file_digest(stream, "md5").hexdigest()
            if actual != md5:
                raise RuntimeError(f"Template checksum mismatch for {url}")


def run_probe(command: list[str], *, timeout=60, cwd: Path | None = None) -> str:
    """Run a bounded diagnostic command and return the last 1000 stdout characters.

    Raise RuntimeError on nonzero exit; propagate launch and timeout errors.
    """
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-1500:] or f"Exit {result.returncode}")
    return result.stdout.strip()[-1000:]


def run_container_probe(command: list[str], *, timeout=120) -> str:
    """Run a container probe and identify restrictions imposed by the caller."""
    try:
        return run_probe(command, timeout=timeout)
    except RuntimeError as error:
        message = str(error)
        if "Could not write info to setgroups: Permission denied" in message:
            raise RuntimeError(
                "Container execution is blocked by the current process sandbox. "
                "Run ./install from an ordinary host shell outside nested user namespaces "
                "or no-new-privileges isolation."
            ) from error
        raise


def check_installation(
    *,
    deep=False,
    with_oslom=True,
    slurm=True,
    quick=False,
    container_execution=True,
    with_lesion=False,
    with_cicada=False,
    with_viewer=True,
) -> list[dict]:
    """Return named dependency checks with ok, required, and detail fields.

    Quick mode checks whether configured resources and Python modules are
    available without loading scientific libraries or parsing every definition.
    Deep mode verifies resource identities. ``container_execution`` also starts
    each image in the current process environment. ``with_viewer`` controls the
    Workbench launcher check; scientific container checks are unchanged. Neither
    mode installs resources or processes subject data.
    """
    values, _ = settings()
    results = []

    def check(name, function, required=True):
        try:
            detail = function()
            results.append(
                {
                    "name": name,
                    "ok": True,
                    "required": required,
                    "detail": str(detail or "available"),
                }
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            ImportError,
            subprocess.SubprocessError,
        ) as error:
            results.append({"name": name, "ok": False, "required": required, "detail": str(error)})

    def file(key):
        path = Path(values[key])
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing or empty: {path}")
        return path

    def executable(value):
        resolved = shutil.which(value)
        if not resolved:
            raise RuntimeError(f"Executable unavailable: {value}")
        return resolved

    def available_module(name):
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"Python module unavailable: {name}")

    if quick:

        def definitions_check():
            root = Path(values["definitions"])
            if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
                raise RuntimeError(f"Definitions store is not readable: {root}")
            return root

        check("definitions store", definitions_check)
    else:
        from nro.definitions.repository import validate_store

        check("definitions store", lambda: validate_store(Path(values["definitions"])))
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "nibabel",
        "yaml",
        "sklearn",
        "nilearn",
        "nitransforms",
    ):

        def import_check(name=name):
            if quick:
                available_module(name)
            else:
                __import__(name)

        check(name, import_check)
    if with_cicada:
        check("pycicada", lambda: available_module("cicada_python"))
    check("container runtime", lambda: run_probe([executable(values["runtime"]), "--version"]))
    images = required_images()
    for key in images:
        check(key, lambda key=key: file(key))
    check("FreeSurfer license", lambda: file("license"))
    if with_viewer:
        check("Workbench", lambda: run_probe([executable(values["workbench"]), "-version"]))
    check("MNI template", lambda: file("mni_template"))
    if with_lesion:
        for name, expected in MASKER_RESOURCES.items():
            resource = Path(values["synthstroke_data"]) / name
            check(
                f"SynthStroke resource {name}",
                lambda resource=resource, expected=expected: verify_sha256(
                    resource, expected, "SynthStroke resource"
                ),
            )
        for name, expected in NEUROLIT_CHECKPOINTS.items():
            checkpoint = Path(values["fastsurfer_data"]) / "LIT" / "weights" / name

            check(
                f"NeuroLIT checkpoint {name}",
                lambda checkpoint=checkpoint, expected=expected: verify_sha256(
                    checkpoint, expected, "NeuroLIT checkpoint"
                ),
            )
    if deep:
        check(
            "FreeSurfer container identity",
            lambda: verify_freesurfer_image(
                executable(values["runtime"]), Path(values["freesurfer"])
            ),
        )
        check(
            "FastSurfer container identity",
            lambda: verify_fastsurfer_image(
                executable(values["runtime"]), Path(values["fastsurfer"])
            ),
        )
    for space, pattern in (
        ("MNI152NLin2009cAsym", "*_label-GM_probseg.nii*"),
        ("fsaverage", "*_hemi-L_den-41k_midthickness.surf.gii"),
        ("fsaverage", "*_hemi-R_den-41k_midthickness.surf.gii"),
    ):

        def template_check(space=space, pattern=pattern):
            found = list((Path(values["templates"]) / f"tpl-{space}").glob(pattern))
            if not found or any(not p.stat().st_size for p in found):
                raise RuntimeError(f"Missing template: tpl-{space}/{pattern}")

        check(f"template {space}/{pattern}", template_check)
    for key in ("bids", "work", "registry"):

        def directory_check(key=key):
            path = Path(values[key])
            if key == "bids":
                if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
                    raise RuntimeError(f"BIDS root is not readable: {path}")
                return path
            parent = path
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
                raise RuntimeError(f"Directory cannot be written or created: {path}")
            return path

        check(key, directory_check)
    for command in ("sbatch", "squeue", "sacct", "scancel"):
        check(command, lambda command=command: executable(command), required=slurm)
    check("OSLOM", lambda: executable(values["oslom"]), required=with_oslom)
    if with_oslom:
        for name in ("igraph", "leidenalg"):
            if quick:
                check(name, lambda name=name: available_module(name))
            else:
                check(
                    name,
                    lambda name=name: run_probe(
                        [__import__("sys").executable, "-c", f"import {name}"]
                    ),
                )
    if deep:
        for relative, (md5, _version) in template_catalog().items():
            check(
                f"template checksum {relative}",
                lambda relative=relative, md5=md5: verify_template(
                    Path(values["templates"]) / relative, md5
                ),
            )
        for key in images:
            path = Path(values[key])
            receipt = path.with_name(path.name + ".receipt.json")
            if receipt.is_file():
                check(
                    f"{key} checksum",
                    lambda path=path, receipt=receipt: verify_image_receipt(path, receipt),
                )
    if deep and container_execution:
        tools = " ".join(
            shlex.quote(x)
            for x in (
                "fslmaths",
                "flirt",
                "applywarp",
                "3dNwarpApply",
                "antsRegistration",
                "antsApplyTransforms",
                "recon-all",
                "mri_convert",
                "wb_command",
            )
        )
        script = (
            "test -s /nro-license && source /opt/qunex/env/qunex_environment.sh >/dev/null 2>&1 || exit 1; "
            "for tool in " + tools + ' msm octave; do command -v "$tool" || exit 1; done; '
            'test -x "$HCPPIPEDIR/MSMAll/MSMAllPipeline.sh"; '
            'test -x "$HCPPIPEDIR/DeDriftAndResample/DeDriftAndResamplePipeline.sh"; '
            'test -s "$HCPPIPEDIR/global/templates/MSMAll/'
            'Q1-Q6_RelatedParcellation210.MyelinMap_BC_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"; '
            "/opt/fsl/fsl/bin/python -c 'import pathlib, pyfix; "
            "root=pathlib.Path(pyfix.__file__).parent; "
            'needle=b"HCP_Style_Single_Multirun_Dedrift"; '
            'assert any(needle in path.read_bytes() for path in root.rglob("*") '
            "if path.is_file())'"
        )
        check(
            "QuNex execution",
            lambda: run_container_probe(
                [
                    values["runtime"],
                    "exec",
                    "--cleanenv",
                    "--bind",
                    values["license"] + ":/nro-license:ro",
                    values["qunex"],
                    "bash",
                    "-c",
                    script,
                ],
            ),
        )
        probe_images = ["synthstrip", "synbold", "freesurfer", "fastsurfer"]
        if "gradient_unwarp" in images:
            probe_images.append("gradient_unwarp")
        for key in probe_images:
            check(
                f"{key} execution",
                lambda key=key: run_container_probe(
                    [
                        values["runtime"],
                        "exec",
                        "--cleanenv",
                        values[key],
                        "/bin/true",
                    ],
                ),
            )
    return results


def _install_image(key: str, source: str, *, offline: bool, checksum: str | None = None) -> None:
    """Acquire one configured container image under its resource lock."""
    values, _ = settings()
    path = Path(values[key])

    def reuse_existing() -> bool:
        if not path.is_file() or not path.stat().st_size:
            return False
        if checksum is not None:
            receipt = path.with_name(path.name + ".receipt.json")
            if receipt.is_file():
                document = json.loads(receipt.read_text(encoding="utf-8"))
                if document.get("sha256") != checksum:
                    raise RuntimeError(f"Installed image has the wrong identity: {path}")
                verify_image_receipt(path, receipt)
            else:
                verify_sha256(path, checksum, key)
                atomic_write_json(
                    receipt,
                    {
                        "source": source,
                        "sha256": checksum,
                        "file_identity": _file_identity(path),
                        "verification": "Pinned SHA-256 verified before adoption",
                    },
                )
        return True

    if reuse_existing():
        print(f"Reuse {key}: {path}")
        return
    if offline:
        raise RuntimeError(f"Offline setup cannot obtain {key}: {path}")
    if not shutil.which(values["runtime"]):
        raise RuntimeError(
            "Install or load Singularity/Apptainer, then set execution.runtime with nro site."
        )
    with resource_lock(path):
        if reuse_existing():
            return
        print(f"Downloading {key} from {source}", flush=True)
        with atomic_output_path(path) as staged:
            if source.startswith("docker://"):
                # OCI extraction needs ordinary extended-attribute support. Some
                # shared filesystems can store the completed SIF but cannot host
                # Apptainer's temporary root filesystem.
                with tempfile.TemporaryDirectory(
                    prefix="nro-container-build-", dir="/tmp"
                ) as build_tmp:
                    environment = os.environ.copy()
                    environment["APPTAINER_TMPDIR"] = build_tmp
                    environment["SINGULARITY_TMPDIR"] = build_tmp
                    subprocess.run(
                        [values["runtime"], "pull", str(staged), source],
                        check=True,
                        env=environment,
                    )
            else:
                download(source, staged, checksum=checksum)
            run_probe([values["runtime"], "inspect", str(staged)])
            digest = sha256(staged)
            if checksum is not None and digest != checksum:
                raise RuntimeError(f"Container checksum mismatch for {source}")
        atomic_write_json(
            path.with_name(path.name + ".receipt.json"),
            {
                "source": source,
                "sha256": digest,
                "file_identity": _file_identity(path),
                "verification": "HTTPS or container transport, runtime inspect; SHA-256 recorded after acquisition",
            },
        )


def install_images(*, offline: bool = False) -> None:
    """Acquire missing configured container images under resource locks.

    Stage and inspect downloads before publishing; write acquisition receipts.
    Existing nonempty images are reused. Offline mode rejects missing images.
    """
    for key, source in required_images().items():
        _install_image(key, source, offline=offline)


def install_lesion_resources(*, offline: bool = False) -> None:
    """Install only the pinned resources selected by the lesion feature."""
    install_synthstroke_model(offline=offline)
    install_neurolit_checkpoints(offline=offline)


def _workbench_launcher(values: dict, program: str) -> str:
    """Build a host launcher for one command in the pinned Workbench image."""
    binds = list(values["binds"])
    for key in ("bids", "work", "development", "templates"):
        path = Path(values[key]).expanduser().resolve()
        if path.exists():
            binds.append(f"{path}:{path}")
    arguments = [values["runtime"], "exec", "--cleanenv"]
    for bind in dict.fromkeys(binds):
        arguments.extend(("--bind", bind))
    command = " ".join(shlex.quote(str(argument)) for argument in arguments)
    image = shlex.quote(str(values["workbench_image"]))
    executable = shlex.quote(program)
    return f"""#!/usr/bin/env bash
set -euo pipefail
export APPTAINERENV_DISPLAY="${{DISPLAY:-}}"
export SINGULARITYENV_DISPLAY="${{DISPLAY:-}}"
export APPTAINERENV_XAUTHORITY="${{XAUTHORITY:-}}"
export SINGULARITYENV_XAUTHORITY="${{XAUTHORITY:-}}"
arguments=({command})
if [[ -n "${{XAUTHORITY:-}}" && -e "$XAUTHORITY" ]]; then
    arguments+=(--bind "$XAUTHORITY:$XAUTHORITY:ro")
fi
if [[ -d /tmp/.X11-unix ]]; then
    arguments+=(--bind /tmp/.X11-unix:/tmp/.X11-unix)
fi
exec "${{arguments[@]}}" {image} {executable} "$@"
"""


def install_workbench(*, offline=False) -> None:
    """Install container-backed Workbench launchers or reuse a native override."""
    values, _ = settings()
    executable = Path(values["workbench"])
    root = executable.parent.parent
    receipt = root / "nro-workbench-container.json"
    legacy_receipt = root / "nro-download.json"
    if executable.is_file() and not receipt.is_file() and not legacy_receipt.is_file():
        run_probe([str(executable), "-version"])
        return
    _install_image(
        "workbench_image",
        WORKBENCH_IMAGE_URL,
        offline=offline,
        checksum=WORKBENCH_IMAGE_SHA256,
    )
    with resource_lock(root):
        executable.parent.mkdir(parents=True, exist_ok=True)
        for program, target in (
            ("wb_command", executable),
            ("wb_view", executable.with_name("wb_view")),
        ):
            atomic_write_text(
                target,
                _workbench_launcher(values, program),
                mode=0o755,
                durable=True,
            )
        atomic_write_json(
            receipt,
            {
                "image": values["workbench_image"],
                "source": WORKBENCH_IMAGE_URL,
                "sha256": WORKBENCH_IMAGE_SHA256,
            },
        )
    run_probe([str(executable), "-version"])


def install_templates(*, offline=False) -> None:
    """Acquire pinned template objects, verifying existing files before reuse."""
    values, _ = settings()
    root = Path(values["templates"])
    if offline:
        return
    with resource_lock(root):
        for relative, (md5, version) in template_catalog().items():
            target = root / relative
            if target.is_file() and target.stat().st_size:
                verify_template(target, md5)
                continue
            source = "https://templateflow.s3.amazonaws.com/" + relative + "?versionId=" + version
            print(f"Downloading template {relative}", flush=True)
            download(source, target, md5=md5)
            atomic_write_json(
                target.with_name(target.name + ".receipt.json"),
                {
                    "source": source,
                    "sha256": sha256(target),
                },
            )
        ensure_fsaverage6_midthickness(root)


def install_oslom(*, offline=False) -> None:
    """Build the official undirected solver and publish it after an example fit."""
    values, _ = settings()
    target = Path(values["oslom"])
    if target.is_file():
        return
    if offline:
        raise RuntimeError(f"OSLOM is missing for offline setup: {target}")
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError(
            "Building OSLOM requires g++; install the host C++ compiler and rerun setup."
        )
    with resource_lock(target):
        if target.is_file():
            return
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=".oslom-") as temporary:
            temporary = Path(temporary)
            archive = temporary / "OSLOM2.tar.gz"
            print("Downloading official OSLOM 2.5 source (HTTP, pinned SHA-256)", flush=True)
            download(OSLOM_SOURCE, archive, checksum=OSLOM_SHA256)
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    if not (temporary / member.name).resolve().is_relative_to(temporary.resolve()):
                        raise ValueError(f"Archive path escapes destination: {member.name}")
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"Unsupported archive entry: {member.name}")
                source.extractall(temporary, filter="data")
            root = temporary / "OSLOM2"
            executable = root / "oslom_undir"
            arguments = [
                "-o",
                "oslom_undir",
                "Sources_2_5/OSLOM_files/main_undirected.cpp",
                "-O3",
                "-Wall",
            ]
            print("Compiling OSLOM; this may take a few minutes", flush=True)
            run_probe([compiler, *arguments], cwd=root, timeout=600)
            run_probe(
                [
                    str(executable),
                    "-f",
                    "example.dat",
                    "-uw",
                    "-r",
                    "1",
                    "-hr",
                    "1",
                    "-seed",
                    "1",
                ],
                cwd=root,
                timeout=120,
            )
            partition = root / "example.dat_oslo_files/tp"
            if not partition.is_file() or not partition.stat().st_size:
                raise RuntimeError("OSLOM example fit did not produce a partition")
            receipt = {
                "source": OSLOM_SOURCE,
                "source_sha256": OSLOM_SHA256,
                "sha256": sha256(executable),
                "compiler": run_probe([compiler, "--version"]),
                "build_arguments": arguments,
                "verification": "Bundled example graph fit",
            }
            with atomic_output_path(target) as staged:
                shutil.copyfile(executable, staged)
                staged.chmod(0o755)
            atomic_write_json(target.with_name(target.name + ".receipt.json"), receipt)
