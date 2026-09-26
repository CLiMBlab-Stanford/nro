"""Anatomical artifact-contract migrations."""

from nro.modules.anat.contract import (
    FASTSURFER_RECONSTRUCTION_VOXEL_SIZE_MM,
    bias_correction_contract,
    surface_reconstruction_contract,
)

from .core import (
    INDETERMINATE,
    AddField,
    ContractMigration,
    ContractMigrationChain,
    RemoveField,
)

CHAIN = ContractMigrationChain(
    module="anat",
    migrations=(
        ContractMigration(
            destination=2,
            summary="Record whether source markup selects lesion-aware anatomy",
            contract=(
                AddField(
                    "processing.source_markup.lesion",
                    default=False,
                    historical=False,
                ),
            ),
            configuration=(
                AddField(
                    "lesion",
                    default={
                        "masker_command": None,
                        "fastsurfer_image": None,
                        "use_gpu": True,
                    },
                    historical={
                        "masker_command": None,
                        "fastsurfer_image": None,
                        "use_gpu": True,
                    },
                ),
            ),
        ),
        ContractMigration(
            destination=3,
            summary="Adopt brain-guided N4 and masked FreeSurfer 7.4.1 reconstruction",
            contract=(
                AddField(
                    "processing.bias_correction",
                    default=bias_correction_contract(),
                    historical={
                        "method": "N4BiasFieldCorrection",
                        "mask_source": None,
                        "mask_application": None,
                        "bias_field_retained": False,
                    },
                ),
                AddField(
                    "processing.surface_reconstruction",
                    default=surface_reconstruction_contract(),
                    historical=INDETERMINATE,
                ),
            ),
        ),
        ContractMigration(
            destination=4,
            summary="Pin lesion-aware FastSurfer reconstruction to a 1 mm grid",
            contract=(
                AddField(
                    "processing.lesion_reconstruction.fastsurfer_voxel_size_mm",
                    default=FASTSURFER_RECONSTRUCTION_VOXEL_SIZE_MM,
                    historical=INDETERMINATE,
                ),
            ),
        ),
        ContractMigration(
            destination=5,
            summary="Select the surface-reconstruction engine explicitly",
            configuration=(
                AddField(
                    "surface_reconstruction_engine",
                    default="freesurfer",
                    historical="freesurfer",
                ),
            ),
        ),
        ContractMigration(
            destination=6,
            summary="Separate lesion inpainting from configurable surface reconstruction",
            contract=(
                AddField(
                    "processing.lesion_reconstruction.pipeline",
                    default="inpainting_surface_reconstruction_excision",
                    historical=INDETERMINATE,
                ),
                *(
                    RemoveField(
                        f"processing.lesion_reconstruction.{field}",
                        reconstructible=True,
                    )
                    for field in (
                        "surface_backend",
                        "fastsurfer_version",
                        "fastsurfer_source_revision",
                        "fastsurfer_oci_digest",
                        "fastsurfer_freesurfer_build",
                        "fastsurfer_voxel_size_mm",
                        "fastsurfer_mask_reconciliation",
                    )
                ),
            ),
        ),
        ContractMigration(
            destination=7,
            summary="Use the whole-brain mask for lesion-aware surface reconstruction",
            contract=(
                AddField(
                    "processing.lesion_reconstruction.reconstruction_brain_mask",
                    default="whole_brain_reference",
                    historical="neurolit_lesion_mask",
                ),
            ),
        ),
        ContractMigration(
            destination=8,
            summary="Assign hemisphere-specific anatomical structures to surface metrics",
            contract=(
                AddField(
                    "processing.output_metadata.surface_metric_structure",
                    default="hemisphere_specific",
                    historical="unspecified",
                ),
            ),
        ),
    ),
)
