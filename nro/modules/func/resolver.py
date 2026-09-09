"""Shared BIDS functional-run resolution helpers."""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from nro.engine.bids import (
    acquisition_order_key,
    parse_bids_entities,
    resolve_bids_metadata,
)
from nro.engine.images import nifti_stem
from nro.engine.paths import project_data_root


def pe_axis_and_sign(ped: str) -> tuple[str, int]:
    """Split a BIDS phase-encoding direction into axis and sign."""
    ped = ped.strip()
    if ped.endswith("-"):
        axis = ped[:-1]
        sign = -1
    else:
        axis = ped
        sign = 1
    if axis not in ("i", "j", "k"):
        raise ValueError(f"Unsupported PhaseEncodingDirection: {ped!r}")
    return axis, sign


def infer_total_readout_time(meta: dict[str, Any]) -> Optional[float]:
    """Read or derive total readout time from BIDS metadata."""
    trt = meta.get("TotalReadoutTime", None)
    if trt is not None:
        try:
            return float(trt)
        except Exception:
            return None

    ees = meta.get("EffectiveEchoSpacing", None)
    if ees is None:
        return None
    try:
        ees_f = float(ees)
    except Exception:
        return None

    rmp = meta.get("ReconMatrixPE", None)
    if rmp is None:
        rmp = meta.get("AcquisitionMatrixPE", None)
    if rmp is None:
        return None
    try:
        rmp_i = int(rmp)
    except Exception:
        return None
    if rmp_i <= 1:
        return None
    return ees_f * float(rmp_i - 1)


@lru_cache(maxsize=None)
def read_nifti_shape_and_zooms(
    img: Path,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Read spatial shape and voxel sizes, with a header-only fallback."""
    try:
        import nibabel as nib
    except Exception:
        nib = None

    if nib is not None:
        im = nib.load(str(img))
        shp = im.shape
        if len(shp) < 3:
            raise RuntimeError(f"Not a valid NIfTI with 3D shape: {img}")
        z = im.header.get_zooms()
        if len(z) < 3:
            raise RuntimeError(f"Not a valid NIfTI with 3D zooms: {img}")
        return (int(shp[0]), int(shp[1]), int(shp[2])), (float(z[0]), float(z[1]), float(z[2]))

    opener = gzip.open if img.name.endswith(".gz") else open
    with opener(img, "rb") as f:
        hdr = f.read(348)
    if len(hdr) < 348:
        raise RuntimeError(f"Not a complete NIfTI header: {img}")

    sizeof_hdr_le = struct.unpack("<I", hdr[0:4])[0]
    sizeof_hdr_be = struct.unpack(">I", hdr[0:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        raise RuntimeError(
            f"Unrecognized NIfTI header size for {img}: {sizeof_hdr_le}/{sizeof_hdr_be}"
        )

    dim = struct.unpack(endian + "8h", hdr[40:56])
    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    ndim = int(dim[0])
    if ndim < 3:
        raise RuntimeError(f"Not a valid NIfTI with 3D shape: {img}")
    shape = (int(dim[1]), int(dim[2]), int(dim[3]))
    zooms = (float(pixdim[1]), float(pixdim[2]), float(pixdim[3]))
    return shape, zooms


@lru_cache(maxsize=None)
def nifti_shape3(img: Path) -> tuple[int, int, int]:
    """Return a cached NIfTI spatial shape."""
    shape, _ = read_nifti_shape_and_zooms(img)
    return shape


@lru_cache(maxsize=None)
def voxel_sizes_3d(img: Path) -> tuple[float, float, float]:
    """Return cached NIfTI spatial voxel sizes."""
    _, zooms = read_nifti_shape_and_zooms(img)
    return zooms


def same_voxel_sizes(
    a: tuple[float, float, float], b: tuple[float, float, float], *, tol: float = 1e-6
) -> bool:
    """Compare voxel-size triples within an absolute tolerance."""
    return (abs(a[0] - b[0]) <= tol) and (abs(a[1] - b[1]) <= tol) and (abs(a[2] - b[2]) <= tol)


def same_readout_time(a: Optional[float], b: Optional[float], *, tol: float = 1e-6) -> bool:
    """Compare two defined readout times within an absolute tolerance."""
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


@dataclass(frozen=True)
class ImageRec:
    """Image path, inherited metadata, geometry, and acquisition ordering for reference selection."""

    img: Path
    metadata_path: Path
    metadata_sources: tuple[Path, ...]
    ents: dict[str, str]
    ped: Optional[str]
    readout: Optional[float]
    intended_for: tuple[str, ...]
    tkind: str
    tval: float
    metadata: dict[str, Any]
    metadata_inheritance: Optional[dict[str, Any]] = None

    @property
    def key(self) -> tuple[str, float]:
        """Return the acquisition ordering key used to compare candidate references."""
        return (self.tkind, self.tval)

    def before_or_equal(self, other: "ImageRec") -> bool:
        """Return whether this acquisition precedes or matches the supplied acquisition."""
        if self.tkind == other.tkind:
            return self.tval <= other.tval
        return True


@dataclass(frozen=True)
class FmapPair:
    """Compatible opposite-phase spin-echo images selected as a fieldmap pair."""

    se1: ImageRec
    se2: ImageRec


@dataclass(frozen=True)
class ResolvedFuncRun:
    """Selected BOLD, SBRef, fieldmap pair, and metadata provenance for a run."""

    func_root: Path
    fmap_root: Path
    bold: ImageRec
    sbref: Optional[ImageRec]
    pair: Optional[FmapPair]
    registration_method: str
    selection_warning: Optional[str]
    requested_run_stem: str
    resolved_run_stem: str


def load_rec(img: Path) -> ImageRec:
    """Load an image record with effective inherited BIDS metadata."""
    resolved_metadata = resolve_bids_metadata(img)
    meta = dict(resolved_metadata.values)
    ents = parse_bids_entities(img.name)
    ped = meta.get("PhaseEncodingDirection", None)
    readout = infer_total_readout_time(meta)
    tkind, tval = acquisition_order_key(meta, img)
    intended_for = meta.get("IntendedFor", ())
    if isinstance(intended_for, str):
        intended_for = (intended_for,)
    elif isinstance(intended_for, list):
        intended_for = tuple(str(v) for v in intended_for)
    else:
        intended_for = ()
    return ImageRec(
        img=img,
        metadata_path=resolved_metadata.sources[-1],
        metadata_sources=resolved_metadata.sources,
        ents=ents,
        ped=str(ped) if ped is not None else None,
        readout=float(readout) if readout is not None else None,
        intended_for=intended_for,
        tkind=tkind,
        tval=float(tval),
        metadata=meta,
    )


def inherit_sbref_metadata(sbref_img: Path, bold: ImageRec) -> ImageRec:
    """Construct an SBRef record using metadata from its exact BOLD run.

    This is deliberately restricted to an exact match of all filename entities.
    The inherited record retains the BOLD sidecar as its source and carries the
    effective metadata in memory; raw BIDS data are never modified or supplemented.
    """
    sbref_entities = parse_bids_entities(sbref_img.name)
    if sbref_entities != bold.ents:
        raise ValueError(
            f"SBRef entities do not exactly match BOLD entities: {sbref_img.name} vs {bold.img.name}"
        )
    provenance: dict[str, Any] = {
        "Applied": True,
        "Reason": "The SBRef had no JSON sidecar and exactly matched all BIDS entities of the selected BOLD run.",
        "SourceImage": str(bold.img),
        "SourceMetadata": [str(path) for path in bold.metadata_sources],
        "TargetImage": str(sbref_img),
        "MatchedEntities": dict(sorted(bold.ents.items())),
    }
    return ImageRec(
        img=sbref_img,
        metadata_path=bold.metadata_path,
        metadata_sources=bold.metadata_sources,
        ents=sbref_entities,
        ped=bold.ped,
        readout=bold.readout,
        intended_for=(),
        tkind=bold.tkind,
        tval=bold.tval,
        metadata=dict(bold.metadata),
        metadata_inheritance=provenance,
    )


def fmap_targets_bold(fmap: ImageRec, bold: ImageRec) -> bool:
    """Return whether a field map's IntendedFor metadata names a BOLD run."""
    if not fmap.intended_for:
        return False
    bold_name = bold.img.name
    bold_rel = f"{bold.img.parent.name}/{bold_name}"
    ses_rel = f"{bold.img.parent.parent.name}/{bold_rel}"
    return any(
        target == bold_name
        or target.endswith("/" + bold_name)
        or target == bold_rel
        or target.endswith("/" + bold_rel)
        or target == ses_rel
        or target.endswith("/" + ses_rel)
        for target in fmap.intended_for
    )


def pick_prev_sbref(sbrefs: Sequence[ImageRec], bold: ImageRec) -> ImageRec:
    """Select the latest preceding SBRef with matching phase encoding."""
    if bold.ped is None:
        raise RuntimeError(
            "BOLD metadata missing PhaseEncodingDirection: "
            + ", ".join(str(path) for path in bold.metadata_sources)
        )
    cands = [s for s in sbrefs if s.ped == bold.ped and s.before_or_equal(bold)]
    if not cands:
        raise RuntimeError(
            f"No previous SBRef with PhaseEncodingDirection={bold.ped!r} found for {bold.img.name}"
        )
    same_kind = [c for c in cands if c.tkind == bold.tkind]
    use = same_kind if same_kind else cands
    return max(use, key=lambda r: (r.tval, r.img.name))


def opp_ped(ped: str) -> str:
    """Return the opposite BIDS phase-encoding direction."""
    axis, sign = pe_axis_and_sign(ped)
    return axis if sign < 0 else axis + "-"


def pick_nearest_opp_sbref(sbrefs: Sequence[ImageRec], sbref: ImageRec) -> ImageRec:
    """Select the nearest compatible opposite-encoding SBRef."""
    if sbref.ped is None:
        raise RuntimeError(
            "SBRef metadata missing PhaseEncodingDirection: "
            + ", ".join(str(path) for path in sbref.metadata_sources)
        )
    if sbref.readout is None:
        raise RuntimeError(
            "SBRef metadata missing TotalReadoutTime (or inferable): "
            + ", ".join(str(path) for path in sbref.metadata_sources)
        )

    opp = opp_ped(sbref.ped)
    cands_all = [s for s in sbrefs if s.ped == opp and s.readout is not None]
    if not cands_all:
        raise RuntimeError(
            f"No opposite-PE SBRef found for {sbref.img.name} (need PhaseEncodingDirection={opp!r})."
        )

    cands = [c for c in cands_all if c.tkind == sbref.tkind]
    if not cands:
        raise RuntimeError(
            f"No opposite-PE SBRef with comparable timestamp kind found for {sbref.img.name}. "
            f"SBRef time kind: {sbref.tkind}. Opposite-PE candidates kinds: {sorted({c.tkind for c in cands_all})}."
        )

    sbref_shape = nifti_shape3(sbref.img)
    sbref_zooms = voxel_sizes_3d(sbref.img)
    sbref_readout = float(sbref.readout)

    def ok_geom(s: ImageRec) -> bool:
        try:
            return nifti_shape3(s.img) == sbref_shape and same_voxel_sizes(
                voxel_sizes_3d(s.img), sbref_zooms
            )
        except Exception:
            return False

    def ok_readout(s: ImageRec, *, tol: float = 1e-6) -> bool:
        if s.readout is None:
            return False
        return abs(float(s.readout) - sbref_readout) <= tol

    cands_geom = [c for c in cands if ok_geom(c)]
    if not cands_geom:
        raise RuntimeError(
            "No opposite-PE SBRef candidates had matching geometry for topup. "
            f"Reference SBRef: {sbref.img.name} shape={sbref_shape} zooms={sbref_zooms}"
        )

    cands_ok = [c for c in cands_geom if ok_readout(c)]
    if not cands_ok:
        raise RuntimeError(
            "No opposite-PE SBRef candidates had matching TotalReadoutTime. "
            f"Reference SBRef: {sbref.img.name} readout={sbref_readout}"
        )

    return min(cands_ok, key=lambda r: (abs(r.tval - sbref.tval), r.tval, r.img.name))


def pick_prev_fmap_pair(fmaps: Sequence[ImageRec], bold: ImageRec) -> FmapPair:
    """Select a compatible preceding EPI field-map pair for a BOLD run."""
    if bold.ped is None:
        raise RuntimeError(
            "BOLD metadata missing PhaseEncodingDirection: "
            + ", ".join(str(path) for path in bold.metadata_sources)
        )

    bold_axis, bold_sign = pe_axis_and_sign(bold.ped)
    opp = bold_axis if bold_sign < 0 else (bold_axis + "-")

    def ok_fmap(f: ImageRec) -> bool:
        return f.ped in (bold.ped, opp) and f.readout is not None

    def pair_is_compatible(a: ImageRec, b: ImageRec) -> bool:
        try:
            if nifti_shape3(a.img) != nifti_shape3(b.img):
                return False
            if not same_voxel_sizes(voxel_sizes_3d(a.img), voxel_sizes_3d(b.img)):
                return False
        except Exception:
            return False
        return same_readout_time(a.readout, b.readout)

    intended = [f for f in fmaps if ok_fmap(f) and fmap_targets_bold(f, bold)]
    if intended:
        candidates = intended
        selection_label = "IntendedFor-linked"
    else:
        prev = [
            f for f in fmaps if f.ped is not None and f.tkind == bold.tkind and f.tval < bold.tval
        ]
        candidates = [f for f in prev if ok_fmap(f)]
        selection_label = "previous"
    if not candidates:
        raise RuntimeError(
            "No SE fieldmaps matched all constraints for "
            f"{bold.img.name}: axis/opposite PE directions {bold.ped!r} and {opp!r}"
        )

    same_dir = [f for f in candidates if f.ped == bold.ped]
    opp_dir = [f for f in candidates if f.ped == opp]
    if not same_dir or not opp_dir:
        raise RuntimeError(
            f"Missing an opposite-PE {selection_label} SE fieldmap for "
            f"{bold.img.name} after enforcing readout/geometry matching."
        )

    compatible_pairs = [
        (same, opposite)
        for same in same_dir
        for opposite in opp_dir
        if pair_is_compatible(same, opposite)
    ]
    if not compatible_pairs:
        raise RuntimeError(
            "No opposite-PE fieldmap pair had mutually compatible geometry/readout for topup. "
            f"Candidate counts after timing/IntendedFor filtering: {len(same_dir)} for {bold.ped!r}, {len(opp_dir)} for {opp!r}."
        )

    if selection_label == "previous":
        se_same, se_opp = max(
            compatible_pairs,
            key=lambda pair: (
                max(pair[0].tval, pair[1].tval),
                min(pair[0].tval, pair[1].tval),
                -abs(pair[0].tval - pair[1].tval),
                pair[0].img.name,
                pair[1].img.name,
            ),
        )
    else:
        se_same, se_opp = min(
            compatible_pairs,
            key=lambda pair: (
                max(abs(pair[0].tval - bold.tval), abs(pair[1].tval - bold.tval)),
                abs(pair[0].tval - pair[1].tval),
                pair[0].tval,
                pair[1].tval,
                pair[0].img.name,
                pair[1].img.name,
            ),
        )
    return FmapPair(se1=se_same, se2=se_opp)


def func_root_for_subject(project: str, sub_id: str, ses_id: Optional[str]) -> Path:
    """Return the source-BIDS subject or session directory."""
    sub_dir = project_data_root(project) / str(sub_id)
    return (sub_dir / str(ses_id)).resolve() if ses_id is not None else sub_dir.resolve()


def _run_prefix(sub_id: str, ses_id: Optional[str]) -> str:
    return f"{sub_id}_{ses_id}" if ses_id is not None else sub_id


def _list_bold_imgs(func_dir: Path, sub_id: str, ses_id: Optional[str]) -> list[Path]:
    return sorted(func_dir.glob(f"{_run_prefix(sub_id, ses_id)}_*_bold.nii*"))


def _match_entities(
    candidate: dict[str, str], target: dict[str, str], *, ignore: Sequence[str] = ()
) -> bool:
    ignored = set(ignore)
    for key, value in target.items():
        if key in ignored:
            continue
        if candidate.get(key) != value:
            return False
    return True


def resolve_bold_from_run_stem(
    func_dir: Path, *, sub_id: str, ses_id: Optional[str], run_stem: str
) -> tuple[Path, Optional[str]]:
    """Resolve a requested run stem to one source BOLD image."""
    stem = str(run_stem).strip()
    if not stem:
        raise RuntimeError("run_stem must be non-empty")
    if stem.endswith("_bold"):
        stem = stem[: -len("_bold")]

    exact = sorted(func_dir.glob(f"{stem}_bold.nii*"))
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        raise RuntimeError(f"Multiple BOLD files matched exact run stem {stem!r} under {func_dir}")

    target_ents = parse_bids_entities(stem)
    candidates = _list_bold_imgs(func_dir, sub_id, ses_id)
    if not candidates:
        raise RuntimeError(f"No BOLD files found under {func_dir}")

    fallback = []
    for path in candidates:
        cand_ents = parse_bids_entities(path.name)
        if _match_entities(cand_ents, target_ents, ignore=("task",)):
            fallback.append(path)
    if len(fallback) == 1:
        chosen = fallback[0]
        return chosen, (
            f"Resolved run stem {stem!r} to {chosen.name!r} after exact match failed; "
            "matched uniquely after ignoring task label."
        )
    if len(fallback) > 1:
        target_task = str(target_ents.get("task", "")).strip()
        if target_task:
            task_like = []
            for path in fallback:
                cand_task = str(parse_bids_entities(path.name).get("task", "")).strip()
                if cand_task and (
                    cand_task.startswith(target_task) or target_task.startswith(cand_task)
                ):
                    task_like.append(path)
            if len(task_like) == 1:
                chosen = task_like[0]
                return chosen, (
                    f"Resolved run stem {stem!r} to {chosen.name!r} after exact match failed; "
                    "matched uniquely after ignoring task label and preferring the closest task-name prefix match."
                )
        raise RuntimeError(
            f"Run stem {stem!r} did not match an exact BOLD file under {func_dir}, "
            f"and {len(fallback)} candidates matched after ignoring task label: {[p.name for p in fallback]}"
        )
    raise RuntimeError(f"No BOLD file matched run stem {stem!r} under {func_dir}")


def resolve_func_run_request(
    *,
    project: str,
    sub_id: str,
    ses_id: Optional[str],
    run_stem: str,
    sdc_from_sbref_pair: bool,
) -> ResolvedFuncRun:
    """Resolve a BOLD run and its applicable reference images."""
    root = func_root_for_subject(project, sub_id, ses_id)
    func_dir = root / "func"
    fmap_dir = root / "fmap"
    if not func_dir.exists():
        raise RuntimeError(f"Missing func directory: {func_dir}")

    bold_path, stem_warning = resolve_bold_from_run_stem(
        func_dir, sub_id=sub_id, ses_id=ses_id, run_stem=run_stem
    )
    bold = load_rec(bold_path)
    prefix = _run_prefix(sub_id, ses_id)

    sbref_imgs = sorted(func_dir.glob(f"{prefix}_*_sbref.nii*"))
    sbrefs: list[ImageRec] = []
    unusable_sbrefs: list[str] = []
    sidecarless_sbrefs: list[Path] = []
    for p in sbref_imgs:
        try:
            sbrefs.append(load_rec(p))
        except FileNotFoundError as error:
            unusable_sbrefs.append(str(error))
            sidecarless_sbrefs.append(p)

    exact_sidecarless = [p for p in sidecarless_sbrefs if parse_bids_entities(p.name) == bold.ents]
    inheritance_warning: Optional[str] = None
    if len(exact_sidecarless) == 1:
        sbrefs.append(inherit_sbref_metadata(exact_sidecarless[0], bold))
    elif len(exact_sidecarless) > 1:
        inheritance_warning = (
            "SBRef metadata inheritance was not applied because multiple sidecarless SBRefs "
            f"exactly matched the selected BOLD entities: {[p.name for p in exact_sidecarless]}"
        )

    sbref: Optional[ImageRec] = None
    pair: Optional[FmapPair] = None
    selection_warning: Optional[str] = stem_warning
    if inheritance_warning:
        selection_warning = (
            inheritance_warning
            if selection_warning is None
            else f"{selection_warning} {inheritance_warning}"
        )
    explicit_references = bold.metadata.get("NROReferencePolicy") == "explicit"
    if explicit_references:
        selected = bold.metadata.get("NROSBRef")
        if selected is not None:
            if not isinstance(selected, str) or Path(selected).name != selected:
                raise ValueError("Invalid explicit SBRef filename")
            matches = [s for s in sbrefs if s.img.name == selected]
            if len(matches) != 1:
                raise ValueError("The explicitly assigned SBRef is missing or invalid")
            sbref = matches[0]
    elif sbrefs:
        try:
            sbref = pick_prev_sbref(sbrefs, bold)
        except Exception as e:
            selection_warning = str(e) if selection_warning is None else f"{selection_warning} {e}"
    else:
        if selection_warning is None:
            if unusable_sbrefs:
                selection_warning = (
                    f"No usable SBRef under {func_dir}; {len(unusable_sbrefs)} candidate(s) "
                    f"lacked required sidecars. {unusable_sbrefs[0]}"
                )
            else:
                selection_warning = f"No SBRef available under {func_dir}"

    if explicit_references:
        sources = bold.metadata.get("B0FieldSource", [])
        if sources:
            if isinstance(sources, str):
                sources = [sources]
            fmaps = (
                [load_rec(p) for p in sorted(fmap_dir.glob(f"{prefix}_*_epi.nii*"))]
                if fmap_dir.exists()
                else []
            )
            linked = [f for f in fmaps if f.metadata.get("B0FieldIdentifier") in sources]
            if (
                len(sources) != 1
                or len(linked) != 2
                or not all(fmap_targets_bold(f, bold) for f in linked)
            ):
                raise ValueError("Explicit fieldmap association is missing or inconsistent")
            pair = pick_prev_fmap_pair(linked, bold)
    elif sdc_from_sbref_pair and sbref is not None:
        try:
            sbref_opp = pick_nearest_opp_sbref(sbrefs, sbref)
            pair = FmapPair(se1=sbref, se2=sbref_opp)
        except Exception as e:
            selection_warning = str(e) if selection_warning is None else f"{selection_warning} {e}"
    elif not sdc_from_sbref_pair:
        try:
            fmap_imgs = sorted(fmap_dir.glob(f"{prefix}_*_epi.nii*")) if fmap_dir.exists() else []
            fmaps = [load_rec(p) for p in fmap_imgs]
            pair = pick_prev_fmap_pair(fmaps, bold)
        except Exception as e:
            selection_warning = str(e) if selection_warning is None else f"{selection_warning} {e}"

    if pair is not None and (bold.readout is None or bold.readout <= 0):
        warning = (
            "Ignoring the reverse-PE pair because the effective BOLD metadata "
            "do not provide a positive total readout time."
        )
        selection_warning = (
            warning if selection_warning is None else f"{selection_warning} {warning}"
        )
        pair = None

    registration_method = "topup_bbregister" if pair is not None else "ants_syn"
    return ResolvedFuncRun(
        func_root=func_dir,
        fmap_root=fmap_dir,
        bold=bold,
        sbref=sbref,
        pair=pair,
        registration_method=registration_method,
        selection_warning=selection_warning,
        requested_run_stem=str(run_stem).strip(),
        resolved_run_stem=nifti_stem(bold.img).removesuffix("_bold"),
    )
