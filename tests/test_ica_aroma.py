from __future__ import annotations

import os
import logging
from itertools import count
from pathlib import Path

import numpy as np

from nro.func import ica_aroma
from nro.configuration.runtime import configure

configure({"common": {"qunex_container": "/tmp/qunex.sif"}})

from nro.func import module as func_module
from nro.func.ica_aroma import denoising, make_dilated_anatomical_epi_mask
from nro.engine.neuroimaging import create_copy_nifti_step
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import artifact_decision


def _runner(name: str = "test.ica-aroma") -> Runner:
    return Runner(
        module_name="ICA-AROMA Test Module",
        container=None,
        binds=(),
        logger=logging.getLogger(name),
        next_step=count(1).__next__,
    )


def test_denoising_restricts_regression_to_brain_mask(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    epi = tmp_path / "bold.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    mix = tmp_path / "melodic_mix"
    out_dir = tmp_path / "aroma"
    out_dir.mkdir()

    denoising(
        run_cmd=lambda command: commands.append(list(command)),
        fsl_cmds={"fsl_regfilt": "fsl_regfilt"},
        in_file=epi,
        mask=mask,
        out_dir=out_dir,
        melmix=mix,
        denoise_type="both",
        denoise_indices=np.array([0, 2]),
    )

    assert len(commands) == 2
    for command in commands:
        assert f"--mask={mask}" in command
        assert f"--in={epi}" in command
        assert f"--design={mix}" in command
        assert "--filter=1,3" in command
    assert "-a" not in commands[0]
    assert "-a" in commands[1]


def test_single_component_melodic_features_remain_column_oriented(tmp_path: Path) -> None:
    n_timepoints = 40
    phase = np.linspace(0.0, 4.0 * np.pi, n_timepoints)
    mix = np.sin(phase)
    motion = np.column_stack(
        [
            mix,
            np.cos(phase),
            np.linspace(-1.0, 1.0, n_timepoints),
            np.sin(phase / 2.0),
            np.cos(phase / 2.0),
            np.linspace(1.0, -1.0, n_timepoints),
        ]
    )
    spectrum = np.linspace(1.0, 0.1, 20)

    mix_path = tmp_path / "melodic_mix"
    motion_path = tmp_path / "motion.par"
    spectrum_path = tmp_path / "melodic_FTmix"
    np.savetxt(mix_path, mix)
    np.savetxt(motion_path, motion)
    np.savetxt(spectrum_path, spectrum)

    time_feature = ica_aroma.feature_time_series(mix_path, motion_path)
    frequency_feature = ica_aroma.feature_frequency(spectrum_path, tr=1.0)

    assert time_feature.shape == (1,)
    assert frequency_feature.shape == (1,)
    assert np.isfinite(time_feature).all()
    assert np.isfinite(frequency_feature).all()


def test_workflow_estimates_from_smoothed_copy_but_denoises_unsmoothed_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unsmoothed = tmp_path / "bold_unsmoothed.nii.gz"
    smoothed = tmp_path / "bold_smooth6mm.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    regression_mask = tmp_path / "regression_mask.nii.gz"
    motion = tmp_path / "motion.par"
    mni_ref = tmp_path / "mni.nii.gz"
    for path in (unsmoothed, smoothed, mask, regression_mask, motion, mni_ref):
        path.write_bytes(b"test")

    observed: dict[str, Path] = {}

    def fake_melodic(**kwargs) -> None:
        observed["melodic_input"] = kwargs["in_file"]

    def fake_register(**kwargs) -> None:
        observed["registered_component_maps"] = kwargs["in_file"]

    def fake_spatial(**kwargs):
        return np.array([0.0]), np.array([0.0])

    def fake_time_series(melmix: Path, mc: Path):
        return np.array([0.0])

    def fake_frequency(mel_ftmix: Path, tr: float):
        return np.array([0.0])

    def fake_classification(*args, **kwargs):
        return np.array([0])

    def fake_denoising(**kwargs) -> None:
        observed["denoising_input"] = kwargs["in_file"]
        observed["denoising_design"] = kwargs["melmix"]
        observed["denoising_mask"] = kwargs["mask"]

    monkeypatch.setattr(ica_aroma, "_run_melodic_and_merge_thresholded_maps", fake_melodic)
    monkeypatch.setattr(ica_aroma, "_register_to_mni", fake_register)
    monkeypatch.setattr(ica_aroma, "feature_spatial", fake_spatial)
    monkeypatch.setattr(ica_aroma, "feature_time_series", fake_time_series)
    monkeypatch.setattr(ica_aroma, "feature_frequency", fake_frequency)
    monkeypatch.setattr(ica_aroma, "classification", fake_classification)
    monkeypatch.setattr(ica_aroma, "denoising", fake_denoising)

    out_dir = tmp_path / "aroma"
    ica_aroma.run_ica_aroma_workflow(
        run_cmd=lambda command: None,
        run_out=lambda command: "",
        fsl_cmds={},
        in_file=unsmoothed,
        melodic_in_file=smoothed,
        out_dir=out_dir,
        mc=motion,
        affmat=None,
        warp=None,
        mask=mask,
        regression_mask=regression_mask,
        tr=1.0,
        denoise_type="aggr",
        mni_ref=mni_ref,
    )

    assert observed["melodic_input"] == smoothed
    assert observed["denoising_input"] == unsmoothed
    assert observed["denoising_design"] == out_dir / "melodic.ica" / "melodic_mix"
    assert observed["denoising_mask"] == out_dir / "regression_mask.nii.gz"
    assert (out_dir / "melodic.complete").read_text().startswith(
        "MELODIC decomposition"
    )


def test_dilated_anatomical_mask_uses_mm_and_intersects_epi_support(tmp_path: Path) -> None:
    import nibabel as nib

    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    anatomical = np.zeros((7, 7, 7), dtype=np.uint8)
    anatomical[3, 3, 3] = 1
    support = np.ones_like(anatomical)
    support[3, 4, 3] = 0

    anatomical_path = tmp_path / "anatomical.nii.gz"
    support_path = tmp_path / "support.nii.gz"
    output_path = tmp_path / "dilated.nii.gz"
    nib.save(nib.Nifti1Image(anatomical, affine), anatomical_path)
    nib.save(nib.Nifti1Image(support, affine), support_path)

    make_dilated_anatomical_epi_mask(
        anatomical_mask=anatomical_path,
        epi_support_mask=support_path,
        dilation_mm=1.1,
        out_mask=output_path,
    )

    result = np.asarray(nib.load(output_path).dataobj)
    assert result[3, 3, 3] == 1
    assert result[3, 2, 3] == 1
    assert result[3, 3, 2] == 1
    assert result[2, 3, 3] == 0  # Two millimetres away along x.
    assert result[3, 4, 3] == 0  # Excluded by EPI support.


def test_shared_regression_outputs_do_not_require_second_melodic(tmp_path: Path) -> None:
    runner = _runner()
    aroma_dir = tmp_path / "space-MNI/aroma"
    outputs = (aroma_dir / "denoised_func_data_aggr.nii.gz",)
    step = func_module._create_shared_aroma_regression_step(
        runner=runner,
        epi=tmp_path / "mni_bold.nii.gz",
        input_space="MNI152NLin2009cAsym",
        regression_mask=tmp_path / "mni_mask.nii.gz",
        mixing_matrix=tmp_path / "space-T1w/aroma/melodic.ica/melodic_mix",
        classified_components=tmp_path / "space-T1w/aroma/classified_motion_ICs.txt",
        shared_policy=tmp_path / "space-T1w/ica_aroma_policy.json",
        aroma_dir=aroma_dir,
        outputs=outputs,
        env={},
        force=False,
        denoise_type="aggr",
    )

    assert step.outputs == outputs
    assert not any("melodic" in str(path) for path in step.outputs)
    assert not any("classified_motion_ICs" in str(path) for path in step.outputs)


def test_shared_regression_becomes_stale_when_t1w_classification_changes(tmp_path: Path) -> None:
    work_dir = tmp_path / "space-MNI"
    shared_work = tmp_path / "space-T1w"
    shared_mix = shared_work / "aroma/melodic.ica/melodic_mix"
    shared_classified = shared_work / "aroma/classified_motion_ICs.txt"
    shared_policy = shared_work / "ica_aroma_policy.json"
    epi = tmp_path / "mni_bold.nii.gz"
    mean = tmp_path / "mni_mean.nii.gz"
    mask = tmp_path / "mni_mask.nii.gz"
    inputs = [epi, mask, shared_mix, shared_classified, shared_policy]
    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input")

    out_4d = tmp_path / "cleaned.nii"
    out_mean = tmp_path / "cleaned_mean.nii.gz"
    runner = _runner()
    regression = func_module._create_shared_aroma_regression_step(
        runner=runner,
        epi=epi,
        input_space="MNI152NLin2009cAsym",
        regression_mask=mask,
        mixing_matrix=shared_mix,
        classified_components=shared_classified,
        shared_policy=shared_policy,
        aroma_dir=work_dir / "aroma",
        outputs=(out_4d,),
        env={},
        force=False,
        denoise_type="aggr",
    )
    outputs = list(regression.outputs)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output")
    policy = func_module._ica_aroma_shared_regression_policy_payload(
        input_space="MNI152NLin2009cAsym",
        denoise_type="aggr",
        repetition_time=1.0,
        shared_work_dir=shared_work,
    )
    func_module.write_json(work_dir / "ica_aroma_policy.json", policy)

    should_run, _ = artifact_decision(outputs, False, inputs=regression.inputs)
    assert not should_run

    newest_output_ns = max(path.stat().st_mtime_ns for path in outputs)
    os.utime(shared_classified, ns=(newest_output_ns + 1_000_000_000,) * 2)
    should_run, reason = artifact_decision(outputs, False, inputs=regression.inputs)
    assert should_run
    assert "newer" in reason.lower()


def test_shared_regression_uses_t1w_mixing_matrix_without_melodic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import nibabel as nib

    epi = tmp_path / "mni_bold.nii.gz"
    mean = tmp_path / "mni_mean.nii.gz"
    brain_mask = tmp_path / "mni_brain_mask.nii.gz"
    data = np.arange(2 * 2 * 2 * 4, dtype=np.float32).reshape((2, 2, 2, 4))
    nib.save(nib.Nifti1Image(data, np.eye(4)), epi)
    nib.save(nib.Nifti1Image(data.mean(axis=3), np.eye(4)), mean)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), brain_mask)

    shared_work = tmp_path / "space-T1w"
    shared_aroma = shared_work / "aroma"
    shared_melodic = shared_aroma / "melodic.ica"
    shared_melodic.mkdir(parents=True)
    np.savetxt(shared_melodic / "melodic_mix", np.arange(8, dtype=float).reshape((4, 2)))
    (shared_aroma / "classified_motion_ICs.txt").write_text("2\n", encoding="utf-8")
    (shared_work / "ica_aroma_policy.json").write_text("{}\n", encoding="utf-8")

    commands: list[list[str]] = []

    class FakeRunner(Runner):
        def __init__(self) -> None:
            super().__init__(
                module_name="ICA-AROMA Regression Test",
                container=None,
                binds=(),
                logger=logging.getLogger("test.ica-aroma.fake-runner"),
                next_step=count(1).__next__,
                )

        def require_cmds(self, commands_required) -> None:
            assert "melodic" not in commands_required

        def run(self, command, **kwargs) -> None:
            command = list(command)
            commands.append(command)
            with self.python_step(
                step_name=kwargs.get("step_name") or "Recorded command",
                outputs=kwargs.get("outputs"),
                reason=kwargs.get("reason"),
            ):
                prepare = kwargs.get("prepare")
                if prepare is not None:
                    prepare()
                if command[0] == "bet":
                    support = Path(command[2])
                    nib.save(nib.Nifti1Image(data.mean(axis=3), np.eye(4)), support)
                    support_mask = support.with_name(support.name.replace(".nii.gz", "_mask.nii.gz"))
                    nib.save(
                        nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)),
                        support_mask,
                    )

        def log_skip(self, command, **kwargs) -> None:
            commands.append(list(command))
            super().log_skip(command, **kwargs)

    observed: dict[str, object] = {}

    def fake_mask(*, out_mask: Path, **kwargs) -> None:
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), out_mask)

    def fake_denoising(**kwargs) -> None:
        observed["in_file"] = kwargs["in_file"]
        observed["melmix"] = kwargs["melmix"]
        observed["indices"] = np.asarray(kwargs["denoise_indices"])
        output = kwargs["out_dir"] / "denoised_func_data_aggr.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), output)

    def fake_copy(source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(func_module, "make_dilated_anatomical_epi_mask", fake_mask)
    monkeypatch.setattr(func_module, "run_ica_aroma_denoising", fake_denoising)
    monkeypatch.setattr(func_module, "copy_or_convert_nifti", fake_copy)
    monkeypatch.setattr(
        func_module,
        "_resolve_container_command_for_wrapper",
        lambda **kwargs: "fsl_regfilt",
    )

    work_dir = tmp_path / "space-MNI"
    out_4d = tmp_path / "cleaned_mni.nii.gz"
    out_mean = tmp_path / "cleaned_mni_mean.nii.gz"
    runner = FakeRunner()
    aroma_dir = work_dir / "aroma"
    denoised = aroma_dir / "denoised_func_data_aggr.nii.gz"
    runner.add_step(func_module._create_shared_aroma_regression_step(
        runner=runner,
        epi=epi,
        input_space="MNI152NLin2009cAsym",
        regression_mask=brain_mask,
        mixing_matrix=shared_melodic / "melodic_mix",
        classified_components=shared_aroma / "classified_motion_ICs.txt",
        shared_policy=shared_work / "ica_aroma_policy.json",
        aroma_dir=aroma_dir,
        outputs=(denoised,),
        env={},
        force=False,
        denoise_type="aggr",
    ))
    runner.add_step(create_copy_nifti_step(
        src=denoised,
        dst=out_4d,
        force=False,
        step_name="Install Shared ICA-AROMA Output",
    ))
    runner.add_step(func_module._create_temporal_mean_step(
        in_4d=out_4d,
        out_3d=out_mean,
        env={},
        force=False,
        chunk_vols=8,
    ))
    with runner.run_context():
        runner.execute()

    assert observed["in_file"] == epi
    assert observed["melmix"] == shared_melodic / "melodic_mix"
    np.testing.assert_array_equal(observed["indices"], np.array([1]))
    assert func_module._ica_aroma_shared_regression_policy_payload(
        input_space="MNI152NLin2009cAsym",
        denoise_type="aggr",
        repetition_time=1.0,
        shared_work_dir=shared_work,
    )["estimation_space"] == "T1w"
    assert out_4d.exists()
    assert out_mean.exists()
    assert not any(command and command[0] == "melodic" for command in commands)
