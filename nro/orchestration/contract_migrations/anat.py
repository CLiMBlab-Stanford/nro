"""Anatomical artifact-contract migrations."""

from nro.modules.anat.contract import bias_correction_contract, surface_reconstruction_contract
from nro.modules.anat.lesion_policy import FASTSURFER_VOXEL_SIZE_MM

from .core import INDETERMINATE, AddField, ContractMigration, ContractMigrationChain

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
                    default=FASTSURFER_VOXEL_SIZE_MM,
                    historical=INDETERMINATE,
                ),
            ),
        ),
    ),
)
