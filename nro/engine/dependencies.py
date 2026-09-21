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
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from nro.configuration.hardware import GRADIENT_UNWARP_IMAGE, gradient_unwarping_configured
from nro.configuration.site import settings
from nro.engine.io import atomic_output_path, atomic_write_json
from nro.modules.anat.lesion_policy import (
    FASTSURFER_OCI_DIGEST,
    MASKER_MODEL,
    MASKER_RESOURCES,
    MASKER_REVISION,
    NEUROLIT_CHECKPOINTS,
)
from nro.modules.anat.policy import FREESURFER_BUILD

IMAGES = {
    "qunex": "docker://qunex/qunex_suite@sha256:a06befbb64f93ab289bbef94d1d00bf957c7cdff920f35e107f90b186ff9f09d",
    "synthstrip": "docker://freesurfer/synthstrip@sha256:801924ea011be040346c0e68f9c25d175ab5b869afbd156560b01fb74059f1b1",
    "synbold": "docker://ytzero/synbold-disco@sha256:18814dd2f419dfe8375632cf9239a0fb31a1300a2bfe66d234e599450af66555",
    "gradient_unwarp": GRADIENT_UNWARP_IMAGE,
    "freesurfer": "docker://freesurfer/freesurfer@sha256:10b6468cbd9fcd2db3708f4651d59ad75d4da849a2c5d8bb6dba217f08b8c46b",
}
FASTSURFER_IMAGE = "docker://deepmi/fastsurfer@" + FASTSURFER_OCI_DIGEST
NEUROLIT_URLS = {
    name: f"https://zenodo.org/api/records/14510136/files/{name}/content"
    for name in NEUROLIT_CHECKPOINTS
}
SYNTHSTROKE_URLS = {
    name: f"https://huggingface.co/{MASKER_MODEL}/resolve/{MASKER_REVISION}/{name}?download=true"
    for name in MASKER_RESOURCES
}


def required_images(*, with_lesion: bool = False) -> dict[str, str]:
    """Return images needed by the site's configured scientific features."""
    images = dict(IMAGES)
    if with_lesion:
        images["fastsurfer"] = FASTSURFER_IMAGE
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
WORKBENCH_BASE = "https://www.humanconnectome.org/storage/app/media/workbench/"
OSLOM_SOURCE = "http://www.oslom.org/code/OSLOM2.tar.gz"
OSLOM_SHA256 = "3d1ff087449dbb715d1f74f37a5e064d0b61bfba29d64411e4f454d407d96d5e"
WORKBENCH_SHA256 = {
    "linux64": "4a4cf2b8ae79fe476adabf51ece716aaad0a73991d96fb2cdee45b4edb46f9c4",
    "rh_linux64": "6e46818fe7f2debe8ac4cf7236994fb0c7d3f1793c16ee7402949b7cd7404ae4",
}


def template_catalog() -> dict:
    """Read the installed mapping of template paths to checksums and S3 versions."""
    path = Path(__file__).resolve().parents[1] / "configuration/template_resources.json"
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


def install_runtime(*, offline=False) -> None:
    """Reuse a host runtime or install unprivileged Apptainer when supported."""
    from nro.configuration.site import CHECKOUT, read_overrides, site_file
    from nro.engine.site_setup import save_settings

    values, _ = settings()
    if shutil.which(values["runtime"]):
        return
    found = next(
        (shutil.which(name) for name in ("apptainer", "singularity") if shutil.which(name)), None
    )
    if found is None:
        if offline:
            raise RuntimeError("No container runtime available for offline setup")
        missing = [name for name in ("curl", "rpm2cpio", "cpio") if not shutil.which(name)]
        if missing:
            raise RuntimeError(
                "Automatic unprivileged Apptainer installation requires "
                + ", ".join(missing)
                + ". Ask the cluster administrator to supply these or load a Singularity/Apptainer module."
            )
        root = CHECKOUT / ".nro-runtime"
        with resource_lock(root):
            if not (root / "bin/apptainer").is_file():
                if root.exists():
                    raise RuntimeError(f"Incomplete runtime directory requires inspection: {root}")
                with tempfile.TemporaryDirectory(dir=CHECKOUT, prefix=".apptainer-") as temporary:
                    temporary = Path(temporary)
                    script = temporary / "install.sh"
                    download(
                        "https://raw.githubusercontent.com/apptainer/apptainer/v1.4.5/tools/install-unprivileged.sh",
                        script,
                        checksum="33d416ca870fdfcfc6b5fd8791f02bf041d742a5b718d015a9a1cf61aa1b30dd",
                    )
                    staged = temporary / "runtime"
                    subprocess.run(
                        ["bash", str(script), "-e", "-v", "1.4.5", str(staged)], check=True
                    )
                    run_probe([str(staged / "bin/apptainer"), "--version"])
                    staged.rename(root)
            found = str(root / "bin/apptainer")
    overrides = read_overrides(site_file())
    overrides["runtime"] = found
    save_settings(site_file(), overrides)
    print(f"Selected container runtime: {found}")


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str, label: str) -> None:
    """Require one file to exist and match its pinned SHA-256 digest."""
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Missing or invalid {label}: {path}")


def verify_fastsurfer_image(runtime: str, path: Path) -> None:
    """Require the pinned FastSurfer image and embedded FreeSurfer build."""
    result = subprocess.run(
        [runtime, "inspect", "--json", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    document = json.loads(result.stdout)
    labels = document["data"]["attributes"]["labels"]
    if labels.get("org.opencontainers.image.base.digest") != FASTSURFER_OCI_DIGEST:
        raise RuntimeError("FastSurfer image has an unexpected OCI base digest")
    if labels.get("org.opencontainers.image.revision") != "cdfccea":
        raise RuntimeError("FastSurfer image has an unexpected source revision")
    build = subprocess.run(
        [runtime, "exec", "--cleanenv", str(path), "cat", "/opt/freesurfer/build-stamp.txt"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()
    if build != FREESURFER_BUILD:
        raise RuntimeError(f"FastSurfer image contains an unexpected FreeSurfer build: {build}")


def verify_freesurfer_image(runtime: str, path: Path) -> None:
    """Require a complete conventional FreeSurfer installation at the pinned build."""
    script = (
        'test "$(cat /usr/local/freesurfer/build-stamp.txt)" = '
        + shlex.quote(FREESURFER_BUILD)
        + " && command -v recon-all >/dev/null"
        + " && command -v mri_nu_correct.mni >/dev/null"
    )
    subprocess.run(
        [runtime, "exec", "--cleanenv", str(path), "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def download(
    url: str, target: Path, *, checksum: str | None = None, md5: str | None = None
) -> None:
    """Verify and publish a download, allowing HTTP only for pinned official OSLOM."""
    official_oslom = url == OSLOM_SOURCE and checksum == OSLOM_SHA256
    if not url.startswith("https://") and not official_oslom:
        raise ValueError("Downloads require HTTPS")
    with atomic_output_path(target) as staged:
        with urllib.request.urlopen(url, timeout=60) as response, staged.open("wb") as stream:
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
) -> list[dict]:
    """Return named dependency checks with ok, required, and detail fields.

    Quick mode checks whether configured resources and Python modules are
    available without loading scientific libraries or parsing every definition.
    Deep mode verifies resource identities. ``container_execution`` also starts
    each image in the current process environment. Neither mode installs
    resources or processes subject data.
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
        from nro.configuration.definitions import validate_store

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
    check("container runtime", lambda: run_probe([executable(values["runtime"]), "--version"]))
    images = required_images(with_lesion=with_lesion)
    for key in images:
        check(key, lambda key=key: file(key))
    check("FreeSurfer license", lambda: file("license"))
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
    if deep and with_lesion:
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

                def receipt_check(path=path, receipt=receipt):
                    if sha256(path) != json.loads(receipt.read_text())["sha256"]:
                        raise RuntimeError(
                            f"Installed image differs from its acquisition receipt: {path}"
                        )

                check(f"{key} checksum", receipt_check)
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
            "for tool in " + tools + '; do command -v "$tool" || exit 1; done'
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
        probe_images = ["synthstrip", "synbold", "freesurfer"]
        if with_lesion:
            probe_images.append("fastsurfer")
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


def _install_image(key: str, source: str, *, offline: bool) -> None:
    """Acquire one configured container image under its resource lock."""
    values, _ = settings()
    path = Path(values[key])
    if path.is_file() and path.stat().st_size:
        print(f"Reuse {key}: {path}")
        return
    if offline:
        raise RuntimeError(f"Offline setup cannot obtain {key}: {path}")
    if not shutil.which(values["runtime"]):
        raise RuntimeError(
            "Install or load Singularity/Apptainer, then set runtime with nro paths."
        )
    with resource_lock(path):
        if path.is_file() and path.stat().st_size:
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
                download(source, staged)
            run_probe([values["runtime"], "inspect", str(staged)])
            digest = sha256(staged)
        atomic_write_json(
            path.with_name(path.name + ".receipt.json"),
            {
                "source": source,
                "sha256": digest,
                "verification": "HTTPS or container transport, runtime inspect; SHA-256 recorded after acquisition",
            },
        )


def install_images(*, offline=False, with_lesion=False) -> None:
    """Acquire missing configured container images under resource locks.

    Stage and inspect downloads before publishing; write acquisition receipts.
    Existing nonempty images are reused. Offline mode rejects missing images.
    """
    for key, source in required_images(with_lesion=with_lesion).items():
        _install_image(key, source, offline=offline)


def install_lesion_resources(*, offline: bool = False) -> None:
    """Install only the pinned resources selected by the lesion feature."""
    _install_image("fastsurfer", FASTSURFER_IMAGE, offline=offline)
    install_synthstroke_model(offline=offline)
    install_neurolit_checkpoints(offline=offline)


def extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP after rejecting traversal and symlink entries; preserve executable bits."""
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes destination: {item.filename}")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink is not supported: {item.filename}")
        zipped.extractall(destination)
        for item in zipped.infolist():
            path = destination / item.filename
            if path.is_file() and (item.external_attr >> 16) & 0o111:
                path.chmod(0o755)


def install_workbench(*, offline=False) -> None:
    """Reuse Workbench or install a checksum-pinned platform archive after an executable probe."""
    values, _ = settings()
    executable = Path(values["workbench"])
    if executable.is_file():
        return
    if offline:
        raise RuntimeError(f"Workbench is missing: {executable}")
    if platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "Automatic Workbench installation supports Linux x86_64; configure an existing executable."
        )
    if executable.parent.name not in {"bin_linux64", "bin_rh_linux64"}:
        raise RuntimeError(
            "For automatic Workbench installation select a path ending in workbench/bin_linux64/wb_command."
        )
    root = executable.parent.parent
    with resource_lock(root):
        if executable.is_file():
            return
        if root.exists():
            raise RuntimeError(
                f"Refusing to replace an existing incomplete Workbench directory: {root}"
            )
        os_release = platform.freedesktop_os_release()
        flavor = (
            "rh_linux64"
            if any(
                x in (os_release.get("ID", "") + " " + os_release.get("ID_LIKE", ""))
                for x in ("rhel", "centos", "fedora")
            )
            else "linux64"
        )
        source = WORKBENCH_BASE + f"workbench-{flavor}-v2.2.1.zip"
        with tempfile.TemporaryDirectory(dir=root.parent, prefix=".workbench-") as temporary:
            temporary = Path(temporary)
            archive = temporary / "download.zip"
            download(source, archive, checksum=WORKBENCH_SHA256[flavor])
            extract_zip(archive, temporary / "extracted")
            candidates = [
                path
                for path in (temporary / "extracted").rglob("wb_command")
                if path.parent.name.startswith("bin_")
            ]
            if len(candidates) != 1:
                raise RuntimeError("Unexpected Workbench archive layout")
            command = candidates[0]
            command.chmod(0o755)
            run_probe([str(command), "-version"])
            tree = command.parent.parent
            if command.parent.name != executable.parent.name:
                command.parent.rename(tree / executable.parent.name)
            atomic_write_json(
                tree / "nro-download.json", {"source": source, "sha256": sha256(archive)}
            )
            tree.rename(root)


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
