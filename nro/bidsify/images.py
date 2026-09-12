"""Bounded raw-image staging, conversion, and metadata sanitization."""

import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from nro.engine.io import atomic_output_path, atomic_write_text

from .config import ALLOWED_TYPES, identifier
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


def temporary_raw(record: dict) -> Path:
    """Resolve deterministic node-local raw staging without creating it."""
    return (
        Path("/tmp/nro/bidsify")
        / identifier(record["server"])
        / identifier(record["remote_session"])
        / identifier(record["id"])
    )


def _bids_guess(metadata: dict) -> tuple[str, str] | None:
    """Normalize dcm2niix's metadata-derived BIDS type suggestion."""
    value = metadata.get("BidsGuess")
    if (
        not isinstance(value, list)
        or len(value) < 2
        or not all(isinstance(part, str) for part in value[:2])
    ):
        return None
    datatype, name = value[:2]
    suffix = name.rsplit("_", 1)[-1].split(".", 1)[0]
    return datatype, suffix


def classify(metadata: dict, rules: list[dict]) -> dict:
    """Classify converted data from `BidsGuess`, then apply configured refinements."""
    guess = _bids_guess(metadata)
    evidence = list(guess) if guess is not None else None
    if guess is None:
        return {
            "datatype": "ignore",
            "suffix": "ignore",
            "confirmed": False,
            "classification": {
                "bids_guess": evidence,
                "source": "unmatched",
                "reason": "dcm2niix did not provide a usable BidsGuess",
            },
        }
    if guess[0].lower() in {"derived", "discard"}:
        return {
            "datatype": "ignore",
            "suffix": "ignore",
            "confirmed": False,
            "classification": {
                "bids_guess": evidence,
                "source": "dcm2niix",
                "reason": f"dcm2niix classified the acquisition as {guess[0]}",
            },
        }
    if guess not in ALLOWED_TYPES:
        return {
            "datatype": "ignore",
            "suffix": "ignore",
            "confirmed": False,
            "classification": {
                "bids_guess": evidence,
                "source": "unsupported",
                "reason": "the BidsGuess type is outside nro's supported raw inputs",
            },
        }
    kind = guess
    text = "\n".join(
        str(metadata.get(field, ""))
        for field in ("SeriesDescription", "ProtocolName", "SequenceName")
    )
    for rule in rules:
        if re.search(rule["pattern"], text):
            kind = rule["datatype"], rule["suffix"]
            source = "protocol_rule"
            break
    else:
        source = "dcm2niix"
    return {
        "datatype": kind[0],
        "suffix": kind[1],
        "confirmed": True,
        "classification": {
            "bids_guess": evidence,
            "source": source,
            "reason": "accepted metadata-derived image type",
        },
    }


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
    """Classify and convert one acquisition, retaining only sanitized helpers.

    Raw data is downloaded and converted entirely in /tmp. Anatomy is also
    stripped there. Raw downloads and extracted DICOMs are removed on normal
    or exceptional exit.
    Existing complete helpers are validated before reuse.
    """
    import pydicom

    output = shared / identifier(item["id"])
    marker = output / ".prepared.json"
    source_identity = {
        k: item[k] for k in ("acquisition", "file_token", "bytes", "source_revision")
    }
    override = item.get("classification_override")
    checkpoint_identity = {"source": source_identity, "override": override}
    if marker.is_file():
        saved = json.loads(marker.read_text())
        from .publication import file_hash

        hashes = saved.get("hashes")
        if (
            saved.get("identity") == checkpoint_identity
            and isinstance(hashes, dict)
            and all(
                (output / name).is_file() and file_hash(output / name) == digest
                for name, digest in hashes.items()
            )
        ):
            image = output / "image.nii.gz"
            if not hashes or np.isfinite(np.asanyarray(nib.load(image).dataobj)).all():
                item.update(saved["classification"])
                return saved["metadata"]
    root = temporary_raw(record) / identifier(item["id"])
    secure_directory(root)
    try:
        if shutil.disk_usage(root).free < max(item["bytes"] * 5, 100_000_000):
            raise BidsificationError("Insufficient node-local space for raw image conversion")
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
        classification = (
            {
                "datatype": override[0],
                "suffix": override[1],
                "confirmed": True,
                "classification": {
                    "bids_guess": list(_bids_guess(metadata) or ()),
                    "source": "review",
                    "reason": "operator-supplied classification",
                },
            }
            if override
            else classify(metadata, record["config"]["protocols"])
        )
        item.update(classification)
        sanitized = {k: v for k, v in metadata.items() if k in METADATA_FIELDS}
        anatomy = item["datatype"] == "anat"
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
        hashes = {}
        if item["datatype"] != "ignore":
            sanitize_image(original, output / "image.nii.gz")
            atomic_write_text(output / "image.json", json.dumps(sanitized, indent=2), mode=0o660)
            geometry = nib.load(output / "image.nii.gz")
            sanitized["_shape"] = list(geometry.shape)
            sanitized["_affine"] = geometry.affine.tolist()
        else:
            for name in ("image.nii.gz", "image.json"):
                (output / name).unlink(missing_ok=True)
        from .publication import file_hash

        if item["datatype"] != "ignore":
            hashes = {name: file_hash(output / name) for name in ("image.nii.gz", "image.json")}

        atomic_write_text(
            marker,
            json.dumps(
                {
                    "identity": checkpoint_identity,
                    "metadata": sanitized,
                    "classification": classification,
                    "hashes": hashes,
                }
            ),
            mode=0o660,
            durable=True,
        )
        return sanitized
    finally:
        if root.exists():
            shutil.rmtree(root)
