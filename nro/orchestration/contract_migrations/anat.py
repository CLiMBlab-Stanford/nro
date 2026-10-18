"""Anatomical artifact-contract migrations."""

from nro.modules.anat.contract import (
    FASTSURFER_RECONSTRUCTION_VOXEL_SIZE_MM,
    bias_correction_contract,
    msmall_structural_input_contract,
    surface_reconstruction_contract,
    synthsr_scaffold_contract,
)

from .core import (
    INDETERMINATE,
    AddField,
    AddFieldWhen,
    ContractMigration,
    ContractMigrationChain,
    RemoveField,
)

_MSMALL_DEFAULT = {
    "enabled": True,
    "high_resolution_mesh": 164,
    "low_resolution_mesh": 32,
    "grayordinates_resolution_mm": 2.0,
    "functional_resolution_mm": 2.0,
    "surface_smoothing_fwhm_mm": 2.0,
    "input_registration": "MSMSulc",
    "output_registration": "MSMAll",
    "iteration_modes": "CA_CAT",
    "method": "WRN",
    "ica_dimension": 40,
    "high_pass_seconds": 0.0,
    "fix_threshold": 10.0,
    "fix_training_model": "HCP_Style_Single_Multirun_Dedrift",
    "matlab_run_mode": "octave",
}

_NEUROLIT_SCAFFOLD = {
    "backend": "NeuroLIT",
    "version": "0.6.1",
    "record": "https://doi.org/10.5281/zenodo.14510136",
    "checkpoint_sha256": {
        "model_axial.pt": "f665c8af703032f4604117c5284f1659a2034fc5916484874fc5633d0c2474d2",
        "model_coronal.pt": "b699ff0c8aa75072fd4c6573247372e7c643bc05035d490009d25aee9975f316",
        "model_sagittal.pt": "7c9dba74c19212f1201ac7fa05654dcf7adad987a91f8b7d08b7efb1852f7efa",
    },
}

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
        ContractMigration(
            destination=9,
            summary="Track inherited source metadata by declared scientific fields",
        ),
        ContractMigration(
            destination=10,
            summary="Add explicitly calibrated optional MSMAll anatomy",
            contract=(
                AddField(
                    "processing.source_markup.msmall",
                    default={"rest": []},
                    historical={"rest": []},
                ),
            ),
            configuration=(
                AddField(
                    "msmall",
                    default=_MSMALL_DEFAULT,
                    historical=_MSMALL_DEFAULT,
                ),
            ),
        ),
        ContractMigration(
            destination=11,
            summary="Use pose-normalized participant references for MSMAll structure",
            contract=(
                AddField(
                    "processing.msmall.structural_inputs",
                    default=msmall_structural_input_contract(),
                    historical={
                        "t1w": "selected_raw_source_images",
                        "t2w": "selected_raw_source_images",
                    },
                ),
            ),
        ),
        ContractMigration(
            destination=12,
            summary="Add deterministic fallback for FreeSurfer topology failures",
            contract=(
                AddFieldWhen(
                    "processing.surface_reconstruction.random_seed_policy",
                    discriminator="processing.surface_reconstruction.backend",
                    value="FreeSurfer",
                    default=surface_reconstruction_contract()["random_seed_policy"],
                    historical=surface_reconstruction_contract()["random_seed_policy"],
                ),
            ),
        ),
        ContractMigration(
            destination=13,
            summary="Synthesize a canonical T1w reference when only T2w anatomy is available",
            configuration=(
                AddField(
                    "t1w_fallback",
                    default="synthesize_from_t2w",
                    historical="synthesize_from_t2w",
                ),
            ),
        ),
        ContractMigration(
            destination=14,
            summary="Use whole-brain SynthSR for lesion reconstruction scaffolds",
            contract=(
                AddFieldWhen(
                    "processing.lesion_reconstruction.scaffold_synthesis",
                    discriminator="processing.source_markup.lesion",
                    value=True,
                    default=synthsr_scaffold_contract(),
                    historical=_NEUROLIT_SCAFFOLD,
                ),
                AddFieldWhen(
                    "processing.lesion_reconstruction.strategy",
                    discriminator="processing.source_markup.lesion",
                    value=True,
                    default="whole_brain_synthesis_surface_reconstruction_excision",
                    historical="mask_conditioned_inpainting_surface_reconstruction_excision",
                ),
                *(
                    RemoveField(
                        f"processing.lesion_reconstruction.{field}",
                        reconstructible=True,
                    )
                    for field in (
                        "neurolit_version",
                        "neurolit_record",
                        "neurolit_checkpoint_sha256",
                    )
                ),
            ),
            configuration=(
                RemoveField("lesion.fastsurfer_image", reconstructible=True),
                RemoveField("lesion.use_gpu", reconstructible=True),
            ),
        ),
        ContractMigration(
            destination=15,
            summary="Treat the host container runtime as execution infrastructure",
        ),
    ),
)
