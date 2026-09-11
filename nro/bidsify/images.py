"""Bounded raw-image staging, conversion, and metadata sanitization."""

import json
import os
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.io import atomic_output_path, atomic_write_text

from .config import identifier
from .errors import BidsificationError
from .paths import secure_directory

METADATA_FIELDS = {
    "RepetitionTime",
    "EchoTime",
    "EchoNumber",
    "FlipAngle",
    "MagneticFieldStrength",
    "NonlinearGradientCorrection",
    "PhaseEncodingDirection",
    "TotalReadoutTime",
    "EffectiveEchoSpacing",
    "SliceTiming",
    "MultibandAccelerationFactor",
    "ParallelReductionFactorInPlane",
    "ReconMatrixPE",
    "AcquisitionMatrixPE",
    "BandwidthPerPixelPhaseEncode",
    "InPlanePhaseEncodingDirectionDICOM",
    "ImageOrientationPatientDICOM",
    "PartialFourier",
    "SliceThickness",
    "SpacingBetweenSlices",
    "DwellTime",
    "InversionTime",
    "NumberOfVolumesDiscardedByScanner",
    "NumberOfVolumesDiscardedByUser",
    "AcquisitionTime",
    "SeriesNumber",
    "ImageType",
    "Manufacturer",
    "ManufacturersModelName",
    "ReceiveCoilName",
    "MRAcquisitionType",
}


def command(argv: list[str], *, env: dict | None = None) -> None:
    """Run a converter without persisting potentially identifying tool output."""
    try:
        subprocess.run(
            argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env
        )
    except (OSError, subprocess.CalledProcessError):
        raise BidsificationError(
            f"{Path(argv[0]).name} failed; no output has been approved for publication"
        ) from None


def temporary_anatomy(record: dict) -> Path:
    """Resolve deterministic node-local anatomy staging without creating it."""
    return (
        Path("/tmp/nro/bidsify")
        / identifier(record["server"])
        / identifier(record["remote_session"])
        / identifier(record["id"])
    )


def extract(source: Path, destination: Path) -> None:
    """Extract regular archive members under opaque names, excluding links and traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for index, member in enumerate(archive.infolist()):
                if member.is_dir():
                    continue
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise BidsificationError("Archive links are not accepted")
                with (
                    archive.open(member) as src,
                    (destination / f"{index:08d}.dcm").open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            for index, member in enumerate(archive):
                if member.isdir():
                    continue
                if not member.isfile():
                    raise BidsificationError("Only regular archive members are accepted")
                with (
                    archive.extractfile(member) as src,
                    (destination / f"{index:08d}.dcm").open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
    else:
        shutil.copyfile(source, destination / "00000000.dcm")


def sanitize_image(source: Path, destination: Path) -> None:
    """Validate image values and write a NIfTI header without free-text extensions."""
    image = nib.load(source)
    data = np.asanyarray(image.dataobj)
    if data.ndim not in (3, 4) or not np.isfinite(data).all() or not np.any(data):
        raise BidsificationError(
            "Converted image is empty, nonfinite, or has unsupported dimensions"
        )
    header = image.header.copy()
    header.extensions.clear()
    for field in ("descrip", "aux_file", "intent_name", "db_name"):
        if field in header:
            header[field] = b""
    with atomic_output_path(destination) as temporary:
        nib.save(nib.Nifti1Image(data, image.affine, header), temporary)
        temporary.chmod(0o660)


def prepare_image(record: dict, item: dict, shared: Path, source) -> dict:
    """Convert one approved acquisition and retain only sanitized helper files.

    Anatomy is downloaded, converted, and stripped entirely in /tmp. Raw
    downloads and extracted DICOMs are removed on normal or exceptional exit.
    Existing complete helpers are validated before reuse.
    """
    import pydicom

    output = shared / identifier(item["id"])
    marker = output / ".prepared.json"
    source_identity = {
        k: item[k]
        for k in ("acquisition", "file_token", "bytes", "datatype", "suffix", "source_revision")
    }
    if marker.is_file():
        saved = json.loads(marker.read_text())
        from .publication import file_hash

        if saved["source"] == source_identity and all(
            (output / name).is_file() and file_hash(output / name) == digest
            for name, digest in saved["hashes"].items()
        ):
            image = nib.load(output / "image.nii.gz")
            if np.isfinite(np.asanyarray(image.dataobj)).all():
                return saved["metadata"]
    anatomy = item["datatype"] == "anat"
    root = (
        temporary_anatomy(record) / identifier(item["id"])
        if anatomy
        else shared / "raw" / identifier(item["id"])
    )
    secure_directory(root)
    try:
        if shutil.disk_usage(root).free < max(item["bytes"] * 5, 100_000_000):
            raise BidsificationError(
                "Insufficient staging space; no shared fallback for raw anatomy"
            )
        raw = root / "source"
        if raw.is_symlink():
            raise BidsificationError("Unsafe raw download path")
        # An interrupted download is not trusted by size alone. Completed
        # sanitized helpers have a separate hash-checked checkpoint above.
        source.download(item, raw)
        dicoms, converted = root / "dicoms", root / "converted"
        for directory in (dicoms, converted):
            if directory.is_symlink():
                raise BidsificationError("Unsafe conversion directory")
            if directory.exists():
                shutil.rmtree(directory)
        extract(raw, dicoms)
        converted.mkdir(exist_ok=True)
        # The operator confirms the anatomy classification before transfer.
        # Conversion accepts MR DICOMs only; it cannot correct that decision.
        for path in dicoms.iterdir():
            header = pydicom.dcmread(path, stop_before_pixels=True)
            if getattr(header, "Modality", "") != "MR":
                raise BidsificationError("Only MR DICOM acquisitions are supported")
        config = record["config"]
        converter = [value.replace("{staging}", str(root)) for value in config["dcm2niix"]]
        stripper = [value.replace("{staging}", str(root)) for value in config["synthstrip"]]
        command(
            [
                *converter,
                "-b",
                "y",
                "-ba",
                "y",
                "-z",
                "y",
                "-f",
                "converted",
                "-o",
                str(converted),
                str(dicoms),
            ],
            env={**os.environ, "TMPDIR": str(root)},
        )
        images = list(converted.glob("*.nii.gz"))
        if len(images) != 1:
            raise BidsificationError(
                "Acquisition produced multiple images; split the acquisition into explicit conversion selections"
            )
        original = images[0]
        metadata = json.loads(original.with_name(original.name[:-7] + ".json").read_text())
        sanitized = {k: v for k, v in metadata.items() if k in METADATA_FIELDS}
        if anatomy:
            stripped = root / "stripped.nii.gz"
            command(
                [*stripper, "-i", str(original), "-o", str(stripped)],
                env={**os.environ, "TMPDIR": str(root)},
            )
            checked = root / "checked.nii.gz"
            sanitize_image(stripped, checked)
            original.unlink()
            original = checked
        secure_directory(output)
        sanitize_image(original, output / "image.nii.gz")
        atomic_write_text(output / "image.json", json.dumps(sanitized, indent=2), mode=0o660)
        geometry = nib.load(output / "image.nii.gz")
        sanitized["_shape"] = list(geometry.shape)
        sanitized["_affine"] = geometry.affine.tolist()
        from .publication import file_hash

        atomic_write_text(
            marker,
            json.dumps(
                {
                    "source": source_identity,
                    "metadata": sanitized,
                    "hashes": {
                        name: file_hash(output / name) for name in ("image.nii.gz", "image.json")
                    },
                }
            ),
            mode=0o660,
            durable=True,
        )
        return sanitized
    finally:
        if root.exists():
            shutil.rmtree(root)
