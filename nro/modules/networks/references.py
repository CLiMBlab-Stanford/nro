"""Packaged population references used for network labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_RESOURCE_DIR = Path(__file__).with_name("resources")


@dataclass(frozen=True)
class ReferenceAtlas:
    """A named bundled population map used for heuristic network labeling."""

    identifier: str
    filename: str

    @property
    def path(self) -> Path:
        """Return the installed resource path for this reference atlas."""
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
    """Return the complete packaged functional-reference atlas set."""
    paths = tuple(atlas.path for atlas in REFERENCE_ATLASES)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Missing network-labeling reference atlas resources: "
            + ", ".join(str(path) for path in missing)
        )
    return paths
