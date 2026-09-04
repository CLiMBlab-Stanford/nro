"""Heuristic functional labels for individualized network maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_RESOURCE_DIR = Path(__file__).with_name("resources")


@dataclass(frozen=True)
class ReferenceAtlas:
    identifier: str
    filename: str

    @property
    def path(self) -> Path:
        return _RESOURCE_DIR / self.filename


REFERENCE_ATLASES = (
    ReferenceAtlas("lana", "LanA_n806.nii"),
    ReferenceAtlas("aud", "DU15_AUD.nii.gz"),
    ReferenceAtlas("cgop", "DU15_CG_OP.nii.gz"),
    ReferenceAtlas("datna", "DU15_dATN_A.nii.gz"),
    ReferenceAtlas("datnb", "DU15_dATN_B.nii.gz"),
    ReferenceAtlas("dna", "DU15_DN_A.nii.gz"),
    ReferenceAtlas("dnb", "DU15_DN_B.nii.gz"),
    ReferenceAtlas("fpna", "DU15_FPN_A.nii.gz"),
    ReferenceAtlas("fpnb", "DU15_FPN_B.nii.gz"),
    ReferenceAtlas("lang", "DU15_LANG.nii.gz"),
    ReferenceAtlas("pmppr", "DU15_PM_PPr.nii.gz"),
    ReferenceAtlas("salpmn", "DU15_SAL_PMN.nii.gz"),
    ReferenceAtlas("smota", "DU15_SMOT_A.nii.gz"),
    ReferenceAtlas("smotb", "DU15_SMOT_B.nii.gz"),
    ReferenceAtlas("visc", "DU15_VIS_C.nii.gz"),
    ReferenceAtlas("visp", "DU15_VIS_P.nii.gz"),
)


def reference_paths() -> tuple[Path, ...]:
    paths = tuple(atlas.path for atlas in REFERENCE_ATLASES)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Missing network-labeling reference atlas resources: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _pearson(reference: np.ndarray, network: np.ndarray) -> float:
    valid = np.isfinite(reference) & np.isfinite(network)
    if int(valid.sum()) < 2:
        return -1.0
    x = np.asarray(reference[valid], dtype=np.float64)
    y = np.asarray(network[valid], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else -1.0


def rank_reference_candidates(
    network_maps: np.ndarray,
    reference_maps: dict[str, np.ndarray],
    candidates_per_reference: int,
) -> list[dict[str, int | float | str]]:
    """Rank individualized maps independently against every reference map."""
    networks = np.asarray(network_maps, dtype=np.float32)
    if networks.ndim != 2:
        raise ValueError("Network maps must have shape networks by brainordinates")
    count = min(int(candidates_per_reference), networks.shape[0])
    records: list[dict[str, int | float | str]] = []
    for atlas in REFERENCE_ATLASES:
        reference = np.asarray(reference_maps[atlas.identifier], dtype=np.float32).reshape(-1)
        if reference.shape[0] != networks.shape[1]:
            raise ValueError(
                f"Reference {atlas.identifier} has {reference.shape[0]} values; "
                f"expected {networks.shape[1]}"
            )
        scores = np.asarray([_pearson(reference, network) for network in networks])
        order = np.argsort(-scores, kind="stable")[:count]
        for rank, network_index in enumerate(order, start=1):
            records.append(
                {
                    "reference": atlas.identifier,
                    "candidate": f"{atlas.identifier}{rank:03d}",
                    "network": int(network_index) + 1,
                    "similarity_rank": rank,
                    "similarity_score": float(scores[network_index]),
                }
            )
    return records


def network_map_names(
    network_count: int,
    records: list[dict[str, int | float | str]],
) -> list[str]:
    candidates: dict[int, list[str]] = {}
    for record in records:
        candidates.setdefault(int(record["network"]), []).append(str(record["candidate"]))
    return [
        f"Network {index:03d}"
        + (f" | {', '.join(candidates[index])}" if index in candidates else "")
        for index in range(1, network_count + 1)
    ]


def project_references_to_cifti(
    reference_dlabel: Path,
    *,
    space: str,
    source_surfaces: tuple[Path, ...],
    anatomical_reference: Path | None,
    mni_to_t1_transform: Path | None,
) -> dict[str, np.ndarray]:
    """Project MNI reference volumes onto a CIFTI brain-model axis."""
    import nibabel as nib
    from nilearn import image, surface

    cifti = nib.load(str(reference_dlabel))
    brain_axis = cifti.header.get_axis(1)
    if not isinstance(brain_axis, nib.cifti2.BrainModelAxis):
        raise ValueError(f"Expected CIFTI brain-model axis: {reference_dlabel}")

    target_image = None
    transform = None
    if space in {"T1w", "fsnative"}:
        if anatomical_reference is None or mni_to_t1_transform is None:
            raise ValueError(f"space-{space} reference projection requires MNI-to-T1w provenance")
        target_image = nib.load(str(anatomical_reference))
        try:
            from nitransforms import manip, resampling
        except ImportError as error:  # pragma: no cover - production environment dependency
            raise RuntimeError(
                "nitransforms is required to project network-labeling atlases to native space"
            ) from error
        transform = manip.load(str(mni_to_t1_transform))

    projected: dict[str, np.ndarray] = {}
    for atlas in REFERENCE_ATLASES:
        atlas_image = nib.load(str(atlas.path))
        if target_image is not None:
            assert transform is not None
            atlas_image = resampling.apply(transform, atlas_image, target_image)

        if source_surfaces:
            if len(source_surfaces) != 2:
                raise ValueError("Reference projection requires ordered left/right surfaces")
            values: list[np.ndarray] = []
            surface_by_structure = {
                "CIFTI_STRUCTURE_CORTEX_LEFT": source_surfaces[0],
                "CIFTI_STRUCTURE_CORTEX_RIGHT": source_surfaces[1],
            }
            for structure, structure_slice, _structure_axis in brain_axis.iter_structures():
                try:
                    mesh = surface_by_structure[structure]
                except KeyError as error:
                    raise ValueError(
                        f"Unsupported surface brain structure for reference labeling: {structure}"
                    ) from error
                sampled = np.asarray(surface.vol_to_surf(atlas_image, mesh), dtype=np.float32)
                values.append(sampled[np.asarray(brain_axis.vertex[structure_slice], dtype=int)])
            projected[atlas.identifier] = np.concatenate(values)
        else:
            if brain_axis.affine is None or brain_axis.volume_shape is None:
                raise ValueError("Volumetric CIFTI brain model lacks its volume grid")
            template = nib.Nifti1Image(
                np.zeros(brain_axis.volume_shape, dtype=np.float32),
                brain_axis.affine,
            )
            resampled = image.resample_to_img(
                atlas_image,
                template,
                interpolation="continuous",
                force_resample=True,
                copy_header=True,
            )
            data = np.asarray(resampled.dataobj, dtype=np.float32)
            voxels = np.asarray(brain_axis.voxel, dtype=int)
            projected[atlas.identifier] = data[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return projected
