"""Substantive functional-processing policies shared with orchestration."""

from __future__ import annotations


FINAL_RESAMPLING_TOOL = "AFNI 3dNwarpApply"
FINAL_RESAMPLING_INTERPOLATION = "wsinc5"
FINAL_WARP_INTERPOLATION = "linear"


def final_resampling_contract() -> dict[str, object]:
    """Return the processing contract for final volumetric BOLD resampling."""
    return {
        "tool": FINAL_RESAMPLING_TOOL,
        "data_interpolation": FINAL_RESAMPLING_INTERPOLATION,
        "warp_interpolation": FINAL_WARP_INTERPOLATION,
        "combined_spatial_warp_and_per_volume_motion": True,
        "interpolation_count": 1,
    }


def final_resampling_metadata() -> dict[str, object]:
    """Return the same policy using BIDS-sidecar key conventions."""
    contract = final_resampling_contract()
    return {
        "Tool": contract["tool"],
        "DataInterpolation": contract["data_interpolation"],
        "WarpInterpolation": contract["warp_interpolation"],
        "CombinedSpatialWarpAndPerVolumeMotion": contract[
            "combined_spatial_warp_and_per_volume_motion"
        ],
        "InterpolationCount": contract["interpolation_count"],
    }
