#!/usr/bin/env python3
"""Shared helpers for the anatomical preprocessing module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Optional, Sequence

from nro.configuration.runtime import SETTINGS
from nro.engine.bids import parse_bids_entities
from nro.engine.images import sidecar_json_path
from nro.engine.io import read_json


def parse_acq_time(raw: str) -> float:
    item = str(raw).strip()
    hh, mm, rest = item.split(":", 2)
    if "." in rest:
        ss, frac = rest.split(".", 1)
        usec = int((frac + "000000")[:6])
    else:
        ss = rest
        usec = 0
    t = time(hour=int(hh), minute=int(mm), second=int(ss), microsecond=usec)
    return t.hour * 3600.0 + t.minute * 60.0 + t.second + (t.microsecond / 1e6)


def get_time_key(meta: dict[str, Any], path: Path) -> tuple[str, float]:
    if "AcquisitionDateTime" in meta:
        return ("acqdt", datetime.fromisoformat(str(meta["AcquisitionDateTime"])).timestamp())
    if "AcquisitionTime" in meta:
        return ("acqtime", parse_acq_time(str(meta["AcquisitionTime"])))
    if "SeriesNumber" in meta:
        return ("series", float(meta["SeriesNumber"]))
    if "AcquisitionNumber" in meta:
        return ("acqnum", float(meta["AcquisitionNumber"]))
    return ("mtime", path.stat().st_mtime)


def infer_session_id(path: Path, *, default_session: str | None = None) -> str:
    for parent in [path.parent, *path.parents]:
        if parent.name.startswith("ses-"):
            return parent.name
    fallback = default_session or str(SETTINGS.common.multi_session_label)
    return fallback


@dataclass(frozen=True)
class AnatImage:
    image: Path
    json: Optional[Path]
    modality: str
    session_id: str
    entities: dict[str, str]
    time_kind: str
    time_value: float


def load_anat_image(path: Path, *, default_session: str | None = None) -> AnatImage:
    img = Path(path).resolve()
    if not img.exists():
        raise FileNotFoundError(f"Missing anatomical image: {img}")
    js = sidecar_json_path(img)
    meta: dict[str, Any] = {}
    json_path: Optional[Path] = None
    if js.exists():
        meta = read_json(js)
        json_path = js
    ents = parse_bids_entities(img.name)
    suffix = ents.get("suffix")
    if suffix is None:
        stem = img.name
        if stem.endswith(".nii.gz"):
            stem = stem[: -len(".nii.gz")]
        elif stem.endswith(".nii"):
            stem = stem[: -len(".nii")]
        suffix = stem.split("_")[-1]
    if suffix not in {"T1w", "T2w"}:
        raise ValueError(f"Unsupported anatomical modality for {img}: {suffix!r}")
    time_kind, time_value = get_time_key(meta, img)
    return AnatImage(
        image=img,
        json=json_path,
        modality=str(suffix),
        session_id=infer_session_id(img, default_session=default_session),
        entities=ents,
        time_kind=time_kind,
        time_value=float(time_value),
    )


def sort_anat_images(images: Sequence[AnatImage]) -> list[AnatImage]:
    return sorted(images, key=lambda item: (item.time_kind, item.time_value, item.image.name))


def robust_template_cmd(inputs: Sequence[Path], out_template: Path, out_transform_prefix: Path) -> list[str]:
    cmd = [
        "mri_robust_template",
        "--template",
        str(out_template),
        "--satit",
        "--mapmov",
        str(out_transform_prefix),
    ]
    for path in inputs:
        cmd += ["--mov", str(path)]
    return cmd
