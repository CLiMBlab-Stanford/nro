import logging
from itertools import count
from pathlib import Path

import pytest

from nro.configuration.runtime import configure
from nro.configuration.store import ConfigStore

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.engine.execution import resolve_runner_command
from nro.engine.neuroimaging import create_copy_nifti_step
from nro.modules.func.steps import (
    _canonical_fieldmap_order,
    _create_ants_registration_step,
    _create_bold_ref_to_topup_transform_step,
    _create_t1_epi_vox_target_step,
    _create_target_readout_warp_step,
    _create_target_shift_step,
    _create_topup_dfout_step,
    _normalized_topup_matrix,
    _pe_to_fsl_shift_direction,
    _resolve_fieldmapless_sdc_method,
    _resolve_sdc_reference_policy,
    _write_topup_datain,
    rigid_transform_metrics,
)
from nro.orchestration.runner import Runner


def test_func_exposes_shared_rigid_transform_metrics(tmp_path: Path) -> None:
    import numpy as np

    initial = tmp_path / "initial.mat"
    selected = tmp_path / "selected.mat"
    np.savetxt(initial, np.eye(4))
    np.savetxt(selected, np.eye(4))

    assert rigid_transform_metrics(
        matrix=selected,
        initial_matrix=initial,
    ) == {
        "RotationDegrees": 0.0,
        "CenterDisplacementMillimeters": 0.0,
    }


def test_t1_epi_voxel_target_is_constructed_from_source_bold_grid(
    tmp_path: Path,
) -> None:
    import nibabel as nib
    import numpy as np

    source_epi = tmp_path / "sub-01_task-rest_bold.nii.gz"
    affine = np.diag([2.1, 2.2, 2.3, 1.0])
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5, 2), dtype=np.float32), affine),
        source_epi,
    )
    t1_image = tmp_path / "sub-01_T1w.nii.gz"
    t1_image.touch()
    output = tmp_path / "work" / "sub-01_t1_grid_epi_vox.nii.gz"

    step = _create_t1_epi_vox_target_step(
        t1_image=t1_image,
        source_epi=source_epi,
        out_target=output,
        env={},
        force=False,
    )

    assert step.inputs == (t1_image, source_epi)
    assert step.command[:5] == (
        "mri_convert",
        "--voxsize",
        "2.100000",
        "2.200000",
        "2.300000",
    )


def test_topup_graph_uses_declared_geometry_for_future_inputs(tmp_path: Path) -> None:
    result = _create_topup_dfout_step(
        run_child=lambda *_args, **_kwargs: None,
        se_a=tmp_path / "future_reference.nii.gz",
        se_b=tmp_path / "future_synthetic_reference.nii.gz",
        ped_a="j",
        ped_b="j",
        readout_time=0.05,
        readout_time_b=0.0,
        topup_dir=tmp_path / "topup",
        topup_config="auto",
        env={},
        force=False,
        spatial_shape=(3, 4, 5),
        volumes_a=1,
        volumes_b=1,
    )

    assert result.a_nvols == 1
    assert result.b_nvols == 1
    assert result.step.inputs == (
        tmp_path / "future_reference.nii.gz",
        tmp_path / "future_synthetic_reference.nii.gz",
    )


class _RecordingRunner(Runner):
    def __init__(self) -> None:
        super().__init__(
            module_name="Functional helper test",
            container=None,
            binds=(),
            logger=logging.getLogger("test.func.recording-runner"),
            next_step=count(1).__next__,
        )
        self.commands: list[list[str]] = []

    def run(self, command, **kwargs) -> None:
        self.commands.append([str(value) for value in command])
        with self.python_step(
            step_name=kwargs.get("step_name") or "Recorded command",
            outputs=kwargs.get("outputs"),
            reason=kwargs.get("reason"),
        ):
            for value in kwargs.get("outputs") or ():
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("recorded\n")

    def log_skip(self, command, **kwargs) -> None:
        self.commands.append([str(value) for value in command])
        super().log_skip(command, **kwargs)


def test_container_command_resolution_preserves_configured_path() -> None:
    calls: list[list[str]] = []

    class _ResolverRunner:
        def run_child(self, command, **_kwargs) -> str:
            calls.append([str(value) for value in command])
            return "__NRO_CMD__=/opt/fsl/bin/applywarp"

    assert resolve_runner_command(_ResolverRunner(), {}, ["applywarp"]) == "/opt/fsl/bin/applywarp"
    assert calls[0][:2] == ["bash", "-c"]


def test_derivative_copy_failure_is_attributed_to_its_numbered_step(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = Runner(
        module_name="Functional Preprocessing Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.func.derivative-copy"),
        next_step=count(1).__next__,
    )

    with caplog.at_level(logging.INFO, logger="test.func.derivative-copy"):
        with pytest.raises(FileNotFoundError):
            runner.add_step(
                create_copy_nifti_step(
                    src=tmp_path / "missing.nii.gz",
                    dst=tmp_path / "derivative.nii.gz",
                    force=True,
                    step_name="Finalize Fieldmap Derivative",
                )
            )
            with runner.run_context():
                runner.execute()

    assert "Failed Step: 002 — Finalize Fieldmap Derivative" in caplog.text


def test_bold_to_topup_transform_always_composes_selected_reference_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg_ref_to_topup = tmp_path / "pose" / "selectedRef2topup_6dof.mat"
    epi_to_selected = tmp_path / "pose" / "epiRef2selectedRef_6dof.mat"
    step = _create_bold_ref_to_topup_transform_step(
        reg_ref_to_topup_mat=reg_ref_to_topup,
        epi_ref_to_reg_ref_mat=epi_to_selected,
        work_dir=tmp_path,
        env={},
        force=False,
    )

    assert step.outputs == (tmp_path / "pose" / "epiRef2topup_6dof.mat",)
    assert list(step.command) == [
        "convert_xfm",
        "-omat",
        str(step.outputs[0]),
        "-concat",
        str(reg_ref_to_topup),
        str(epi_to_selected),
    ]


@pytest.mark.parametrize("requested_sdc_method", ["syn", "synbold_disco"])
def test_real_fieldmaps_route_to_unrestricted_anatomical_syn(
    requested_sdc_method: str,
) -> None:
    use_synbold_reference, refinement_target = _resolve_sdc_reference_policy(
        requested_sdc_method=requested_sdc_method,
        fieldmap_pair_available=True,
        fieldmap_syn_refine=True,
    )

    assert use_synbold_reference is False
    assert refinement_target == "T1wAnatomicalSyN"


@pytest.mark.parametrize("requested_sdc_method", ["syn", "synbold_disco"])
def test_real_fieldmaps_default_to_base_registration_without_anatomical_syn(
    requested_sdc_method: str,
) -> None:
    use_synbold_reference, refinement_target = _resolve_sdc_reference_policy(
        requested_sdc_method=requested_sdc_method,
        fieldmap_pair_available=True,
        fieldmap_syn_refine=False,
    )

    assert use_synbold_reference is False
    assert refinement_target is None


def test_fieldmap_anatomical_syn_cli_default_is_disabled() -> None:
    config = ConfigStore().load_configuration("preprocessing", "main").values
    assert config["func"]["fieldmap_syn_refine"] is False


def test_synbold_reference_is_reserved_for_fieldmapless_fallback() -> None:
    assert _resolve_sdc_reference_policy(
        requested_sdc_method="synbold_disco",
        fieldmap_pair_available=False,
        fieldmap_syn_refine=True,
    ) == (True, None)
    assert _resolve_sdc_reference_policy(
        requested_sdc_method="syn",
        fieldmap_pair_available=False,
        fieldmap_syn_refine=True,
    ) == (False, None)


def test_synbold_falls_back_to_syn_without_required_metadata() -> None:
    method, reason = _resolve_fieldmapless_sdc_method(
        "synbold_disco",
        fieldmap_pair_available=False,
        bold_metadata={"RepetitionTime": 2.0},
    )

    assert method == "syn"
    assert reason is not None
    assert "PhaseEncodingDirection" in reason
    assert "TotalReadoutTime" in reason


def test_synbold_remains_selected_with_required_metadata() -> None:
    assert _resolve_fieldmapless_sdc_method(
        "synbold_disco",
        fieldmap_pair_available=False,
        bold_metadata={
            "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.05,
        },
    ) == ("synbold_disco", None)


@pytest.mark.parametrize(
    ("bids_direction", "fsl_direction"),
    [
        ("i", "x"),
        ("i-", "x-"),
        ("j", "y"),
        ("j-", "y-"),
        ("k", "z"),
        ("k-", "z-"),
    ],
)
def test_pe_to_fsl_shift_direction(bids_direction: str, fsl_direction: str) -> None:
    assert _pe_to_fsl_shift_direction(bids_direction) == fsl_direction


def test_canonical_fieldmap_order_is_independent_of_discovery_order() -> None:
    positive = (Path("sub-1_dir-AP_epi.nii.gz"), {"PhaseEncodingDirection": "j"})
    negative = (Path("sub-1_dir-PA_epi.nii.gz"), {"PhaseEncodingDirection": "j-"})

    assert _canonical_fieldmap_order(positive, negative) == (positive, negative)
    assert _canonical_fieldmap_order(negative, positive) == (positive, negative)


def test_normalized_topup_matrix_uses_fixed_one_based_path(tmp_path: Path) -> None:
    prefix = tmp_path / "normalized" / "MotionMatrix"
    expected = tmp_path / "normalized" / "MotionMatrix_0002.mat"
    expected.parent.mkdir()
    expected.write_text("matrix")
    assert _normalized_topup_matrix(prefix, index_1based=2) == (
        tmp_path / "normalized" / "MotionMatrix_0002.mat"
    )
    with pytest.raises(SystemExit, match=">= 1"):
        _normalized_topup_matrix(prefix, index_1based=0)


def test_topup_datain_preserves_group_specific_readout_times(tmp_path: Path) -> None:
    datain = tmp_path / "acqparams.txt"
    _write_topup_datain(
        out_txt=datain,
        ped_a="j",
        ped_b="j-",
        readout_time=0.05,
        readout_time_b=0.04,
        a_nvols=2,
        b_nvols=1,
    )

    assert datain.read_text(encoding="utf-8").splitlines() == [
        "0 1 0 0.05000000",
        "0 1 0 0.05000000",
        "0 -1 0 0.04000000",
    ]


def test_target_readout_warp_scales_hz_field_and_uses_signed_pe_axis(
    tmp_path: Path,
) -> None:
    runner = _RecordingRunner()
    field_hz = tmp_path / "field_hz.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    field_hz.touch()
    reference.touch()
    shift = tmp_path / "shift.nii.gz"
    warp = tmp_path / "warp.nii.gz"

    runner.add_step(
        _create_target_shift_step(
            field_hz=field_hz,
            reference=reference,
            readout_time=0.052,
            shift_vox=shift,
            env={},
            force=False,
        )
    )
    runner.add_step(
        _create_target_readout_warp_step(
            field_hz=field_hz,
            reference=reference,
            phase_encoding_direction="j-",
            shift_vox=shift,
            out_warp=warp,
            env={},
            force=False,
        )
    )
    with runner.run_context():
        runner.execute()

    assert runner.commands[0] == [
        "fslmaths",
        str(field_hz),
        "-mul",
        "0.052",
        str(shift),
    ]
    assert "--shiftmap=" + str(shift) in runner.commands[1]
    assert "--shiftdir=y-" in runner.commands[1]


def test_pe_residual_refinement_is_pe_only_and_run_independent(
    tmp_path: Path,
) -> None:
    moving = tmp_path / "sbref_postrigid.nii.gz"
    fixed = tmp_path / "corrected_se.nii.gz"
    commands: list[list[str]] = []
    result_holder: dict[str, object] = {}

    def run_child(command, **_kwargs) -> None:
        commands.append([str(value) for value in command])
        result = result_holder["result"]
        artifact_dir = result.warped.parent  # type: ignore[union-attr]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "raw_Warped.nii.gz").write_bytes(b"warped")
        (artifact_dir / "raw_1Warp.nii.gz").write_bytes(b"forward")
        (artifact_dir / "raw_1InverseWarp.nii.gz").write_bytes(b"inverse")

    result = _create_ants_registration_step(
        run_child=run_child,
        moving_img=moving,
        fixed_img=fixed,
        work_dir=tmp_path / "residual",
        out_prefix="PEResidual_",
        env={},
        force=True,
        include_linear=False,
        write_composite=False,
        restrict_deformation="0x1x0",
    )
    result_holder["result"] = result
    runner = _RecordingRunner()
    runner.add_step(result.step)
    with runner.run_context():
        runner.execute()

    command = commands[0]
    assert "--restrict-deformation" in command
    assert command[command.index("--restrict-deformation") + 1] == "0x1x0"
    assert "Rigid[0.1]" not in command
    assert "Affine[0.1]" not in command
    assert str(moving) in " ".join(command)
    assert str(fixed) in " ".join(command)
