from __future__ import annotations

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from nro.configuration.markup import SubjectMarkup
from nro.configuration.store import ConfigStore
from nro.modules.anat import planning as anat_planning
from nro.modules.anat.msmall import add_msmall_plan, resolve_msmall_calibration
from nro.modules.anat.msmall_validate_atlas import validate as validate_atlas
from nro.modules.anat.msmall_validate_subcortical import validate as validate_subcortical
from nro.orchestration.catalog import module_descriptor
from nro.orchestration.planning_context import SubjectPlanningContext
from nro.orchestration.registry import Registry
from nro.orchestration.runner import Runner

PARAMETERS = {
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


def _image(path: Path, metadata: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5), np.float32), np.eye(4)), path)
    sidecar = path.with_suffix("").with_suffix(".json")
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _calibration_subject(tmp_path: Path):
    subject = tmp_path / "BIDS/demo/sub-01"
    t1w = _image(subject / "ses-a/anat/sub-01_ses-a_T1w.nii.gz", {})
    t2w = _image(subject / "ses-a/anat/sub-01_ses-a_T2w.nii.gz", {})
    intended = "ses-a/func/sub-01_ses-a_task-Rest_run-01_bold.nii.gz"
    bold = _image(
        subject / intended,
        {
            "AcquisitionTime": "12:00:00",
            "B0FieldSource": "pair-a",
            "EffectiveEchoSpacing": 0.0006,
            "NROReferencePolicy": "explicit",
            "PhaseEncodingDirection": "j",
            "RepetitionTime": 0.72,
            "TotalReadoutTime": 0.06,
        },
    )
    common = {
        "AcquisitionTime": "11:59:00",
        "B0FieldIdentifier": "pair-a",
        "EffectiveEchoSpacing": 0.0006,
        "IntendedFor": intended,
        "TotalReadoutTime": 0.06,
    }
    negative = _image(
        subject / "ses-a/fmap/sub-01_ses-a_dir-AP_epi.nii.gz",
        {**common, "PhaseEncodingDirection": "j-"},
    )
    positive = _image(
        subject / "ses-a/fmap/sub-01_ses-a_dir-PA_epi.nii.gz",
        {**common, "PhaseEncodingDirection": "j"},
    )
    markup = SubjectMarkup("main", "demo", subject, msmall_rest=(bold,))
    return subject, t1w, t2w, bold, negative, positive, markup


def test_msmall_calibration_resolves_fixed_runs_and_fieldmaps(tmp_path: Path) -> None:
    subject, t1w, t2w, bold, negative, positive, markup = _calibration_subject(tmp_path)

    calibration = resolve_msmall_calibration(
        markup=markup,
        t1w=(t1w,),
        t2w=(t2w,),
        parameters=PARAMETERS,
        surface_engine="freesurfer",
        selection_strategy="first",
    )

    assert calibration is not None
    assert calibration.runs[0].bold == bold
    assert calibration.runs[0].se_negative == negative
    assert calibration.runs[0].se_positive == positive
    assert calibration.runs[0].unwarp_direction == "y"
    assert calibration.contract(subject)["calibration_runs"][0]["bold"] == (
        "ses-a/func/sub-01_ses-a_task-Rest_run-01_bold.nii.gz"
    )


def test_msmall_calibration_is_absent_without_explicit_markup(tmp_path: Path) -> None:
    subject = tmp_path / "BIDS/demo/sub-01"
    markup = SubjectMarkup("main", "demo", subject)

    assert (
        resolve_msmall_calibration(
            markup=markup,
            t1w=(),
            t2w=(),
            parameters=PARAMETERS,
            surface_engine="freesurfer",
            selection_strategy="first",
        )
        is None
    )


def test_msmall_anatomy_follows_selection_strategy(tmp_path: Path) -> None:
    subject, t1w, t2w, _, _, _, markup = _calibration_subject(tmp_path)
    second_t1w = _image(subject / "ses-b/anat/sub-01_ses-b_T1w.nii.gz", {})

    first = resolve_msmall_calibration(
        markup=markup,
        t1w=(t1w, second_t1w),
        t2w=(t2w,),
        parameters=PARAMETERS,
        surface_engine="freesurfer",
        selection_strategy="first",
    )
    averaged = resolve_msmall_calibration(
        markup=markup,
        t1w=(t1w, second_t1w),
        t2w=(t2w,),
        parameters=PARAMETERS,
        surface_engine="freesurfer",
        selection_strategy="robust_average",
    )

    assert first is not None and first.t1w == (t1w,)
    assert averaged is not None and averaged.t1w == (t1w, second_t1w)


def test_msmall_rejects_lesion_and_fastsurfer_routes(tmp_path: Path) -> None:
    _, t1w, t2w, _, _, _, markup = _calibration_subject(tmp_path)
    with pytest.raises(ValueError, match="lesion-aware"):
        resolve_msmall_calibration(
            markup=SubjectMarkup(
                markup.markup_id,
                markup.project,
                markup.subject_dir,
                lesion=True,
                msmall_rest=markup.msmall_rest,
            ),
            t1w=(t1w,),
            t2w=(t2w,),
            parameters=PARAMETERS,
            surface_engine="freesurfer",
            selection_strategy="first",
        )
    with pytest.raises(ValueError, match="requires FreeSurfer"):
        resolve_msmall_calibration(
            markup=markup,
            t1w=(t1w,),
            t2w=(t2w,),
            parameters=PARAMETERS,
            surface_engine="fastsurfer",
            selection_strategy="first",
        )


def test_msmall_plan_declares_runner_stages_and_publication(tmp_path: Path, monkeypatch) -> None:
    _, t1w, t2w, _, _, _, markup = _calibration_subject(tmp_path)
    calibration = resolve_msmall_calibration(
        markup=markup,
        t1w=(t1w,),
        t2w=(t2w,),
        parameters=PARAMETERS,
        surface_engine="freesurfer",
        selection_strategy="first",
    )
    assert calibration is not None
    license_path = tmp_path / "license.txt"
    license_path.write_text("license")
    spheres = {hemi: tmp_path / f"{hemi}.sphere.gii" for hemi in ("lh", "rh")}
    for sphere in spheres.values():
        sphere.write_text("sphere")
    structural_t1w = tmp_path / "out/sub-01_desc-preproc_T1w.nii.gz"
    structural_t2w = tmp_path / "out/sub-01_space-T1w_desc-preproc_T2w.nii.gz"
    _image(structural_t1w, {})
    _image(structural_t2w, {})
    runner = Runner(
        module_name="Anatomical Module",
        container=None,
        binds=(),
        logger=logging.getLogger("test.anat.msmall"),
    )

    outputs = add_msmall_plan(
        runner,
        calibration=calibration,
        subject="sub-01",
        out_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        license_path=license_path,
        structural_t1w=structural_t1w,
        structural_t2w=structural_t2w,
        native_registration_spheres=spheres,
        env={},
        force=False,
    )

    names = [step.name for step in runner._graph.steps]
    assert "Estimate MSMAll Registration" in names
    assert "MSMAll PreFreeSurfer" in names
    assert "MSMAll FreeSurfer Reconstruction" in names
    assert "Validate MSMAll Subcortical Models" in names
    assert "MSMAll fMRI Volume rfMRI_REST001" in names
    assert "MSMAll fMRI Surface rfMRI_REST001" in names
    assert "MSMAll Multi-Run ICA-FIX" in names
    assert "MSMAll Dedrift and Resample" in names
    assert "Validate MSMAll Registration" in names
    assert outputs["manifest"].endswith("sub-01_desc-msmall_manifest.json")
    assert outputs["input_identities"].endswith("sub-01_desc-msmallInputs_provenance.json")

    runner._graph.freeze()
    by_name = {step.name: step for step in runner._graph.steps}
    prefree_dependencies = runner._graph.dependencies(by_name["MSMAll PreFreeSurfer"])
    assert by_name["Write MSMAll Structural Configuration"].id in prefree_dependencies
    assert by_name["Write MSMAll Surface Configuration"].id not in prefree_dependencies
    assert by_name["Write MSMAll Calibration Configuration"].id not in prefree_dependencies
    post_dependencies = runner._graph.dependencies(by_name["MSMAll PostFreeSurfer"])
    assert by_name["Write MSMAll Surface Configuration"].id in post_dependencies
    assert by_name["Write MSMAll Calibration Configuration"].id not in post_dependencies
    assert by_name["MSMAll Mask-Aware Atlas Registration"].id in post_dependencies
    freesurfer_dependencies = runner._graph.dependencies(
        by_name["MSMAll FreeSurfer Reconstruction"]
    )
    assert by_name["MSMAll PreFreeSurfer"].id in freesurfer_dependencies
    assert by_name["MSMAll Mask-Aware Atlas Registration"].id not in freesurfer_dependencies
    volume_dependencies = runner._graph.dependencies(by_name["MSMAll fMRI Volume rfMRI_REST001"])
    assert by_name["Write MSMAll Calibration Configuration"].id in volume_dependencies
    assert by_name["Write MSMAll Surface Configuration"].id in volume_dependencies
    assert by_name["Validate MSMAll Subcortical Models"].id in volume_dependencies
    prefree = by_name["MSMAll PreFreeSurfer"]
    assert all(path.name != "msmall_driver.sh" for path in prefree.inputs)
    assert prefree.scientific_signature

    structural_configuration_step = next(
        step for step in runner._graph.steps if step.name == "Write MSMAll Structural Configuration"
    )
    calibration_configuration_step = next(
        step
        for step in runner._graph.steps
        if step.name == "Write MSMAll Calibration Configuration"
    )
    assert structural_t1w in structural_configuration_step.inputs
    assert structural_t2w in structural_configuration_step.inputs
    assert t1w in structural_configuration_step.inputs
    assert t2w in structural_configuration_step.inputs
    assert structural_configuration_step.action is not None
    assert calibration_configuration_step.action is not None
    monkeypatch.setattr("nro.modules.anat.msmall.write_public_json", lambda *_args, **_kwargs: None)
    structural_configuration_step.action()
    calibration_configuration_step.action()
    structural_configuration = (tmp_path / "work/msmall/structural_configuration.sh").read_text()
    calibration_configuration = (tmp_path / "work/msmall/calibration_configuration.sh").read_text()
    assert f"t1w=({structural_t1w})" in structural_configuration
    assert f"t2w=({structural_t2w})" in structural_configuration
    assert f"t1w=({t1w})" not in structural_configuration
    assert f"t2w=({t2w})" not in structural_configuration
    assert "run_names=(rfMRI_REST001)" in calibration_configuration

    structural_mtime = (tmp_path / "work/msmall/structural_configuration.sh").stat().st_mtime_ns
    structural_configuration_step.action()
    assert (
        tmp_path / "work/msmall/structural_configuration.sh"
    ).stat().st_mtime_ns == structural_mtime


def test_msmall_driver_has_no_independent_checkpoint_graph() -> None:
    driver = Path(anat_planning.__file__).with_name("msmall_driver.sh").read_text()

    assert "run_stage" not in driver
    assert "verify_checkpoint" not in driver
    assert "clear_after" not in driver
    assert 'case "$stage" in' in driver


def test_msmall_planning_uses_long_cpu_profile(tmp_path: Path, monkeypatch) -> None:
    subject, _, _, bold, negative, positive, markup = _calibration_subject(tmp_path)
    bids = tmp_path / "BIDS"
    workflow = ConfigStore().resolve("main")
    registry = Registry.for_project("demo", bids_root=bids)
    registered = registry.register_workflow(workflow)
    context = SubjectPlanningContext(
        project="demo",
        participant="01",
        sub_id="sub-01",
        bids_root=bids,
        project_root=bids / "demo",
        subject_dir=subject,
        workflow=workflow,
        registered=registered,
        registry=registry,
        runs=(),
        aggregate_source_inputs=(),
        target_pairs=(),
        memory_gb=32,
        max_memory_gb=256,
        definitions_roots=(),
        gradient_coefficients_root=tmp_path / "gradients",
        source_markup=markup,
    )

    work_item = anat_planning.plan_work_items(context, {}, module_descriptor("anat"))[0]

    assert work_item.resource_class == "long"
    assert work_item.memory_gb == 64
    assert work_item.work_item_contract["processing"]["msmall"]["calibration_runs"]
    assert {bold, negative, positive}.issubset(work_item.input_paths)

    work_item_id = registry.register_work_items((work_item,))[work_item.key]
    before = next(row for row in registry.work_item_rows() if row["id"] == work_item_id)
    with registry.connection(write=True) as db:
        stored = db.execute(
            "SELECT resolved_yaml FROM module_lineages WHERE id=?",
            (before["module_lineage_id"],),
        ).fetchone()[0]
        historical = yaml.safe_load(stored)
        historical.pop("msmall")
        db.execute(
            "UPDATE module_lineages SET resolved_yaml=? WHERE id=?",
            (yaml.safe_dump(historical), before["module_lineage_id"]),
        )
    monkeypatch.setattr(
        "nro.configuration.markup.MarkupStore.subject",
        lambda _store, _markup_id, _project, _subject_dir: markup,
    )
    from nro.orchestration.manifests import assess_registry

    assessment = assess_registry(registry, work_item_ids=(work_item_id,))[work_item_id]
    assert "Could not reassess selected anatomical inputs" not in assessment[1]
    after = next(row for row in registry.work_item_rows() if row["id"] == work_item_id)
    assert after["input_paths_json"] == before["input_paths_json"]
    assert after["artifact_fingerprint"] == before["artifact_fingerprint"]


def test_msmall_atlas_validation_accepts_plausible_registration(tmp_path: Path) -> None:
    shape = (12, 13, 14)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.uint8)
    paths = {}
    for name, data in {
        "subject": values,
        "reference": values * 2 + 1,
        "subject_mask": mask,
        "reference_mask": mask,
        "jacobian": np.ones(shape, dtype=np.float32),
    }.items():
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), paths[name])
    paths["affine"] = tmp_path / "affine.mat"
    np.savetxt(paths["affine"], np.eye(4))

    report = validate_atlas(
        affine_path=paths["affine"],
        subject_path=paths["subject"],
        subject_mask_path=paths["subject_mask"],
        reference_path=paths["reference"],
        reference_mask_path=paths["reference_mask"],
        jacobian_path=paths["jacobian"],
    )

    assert report["valid"] is True
    assert report["mask_dice"] == 1.0


def test_msmall_atlas_validation_rejects_folded_transform(tmp_path: Path) -> None:
    shape = (12, 13, 14)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.uint8)
    jacobian = np.ones(shape, dtype=np.float32)
    jacobian[0, 0, 0] = 0
    paths = {}
    for name, data in {
        "subject": values,
        "reference": values,
        "subject_mask": mask,
        "reference_mask": mask,
        "jacobian": jacobian,
    }.items():
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), paths[name])
    paths["affine"] = tmp_path / "affine.mat"
    np.savetxt(paths["affine"], np.eye(4))

    report = validate_atlas(
        affine_path=paths["affine"],
        subject_path=paths["subject"],
        subject_mask_path=paths["subject_mask"],
        reference_path=paths["reference"],
        reference_mask_path=paths["reference_mask"],
        jacobian_path=paths["jacobian"],
    )

    assert report["valid"] is False
    assert report["jacobian_nonpositive_fraction"] > 0


def test_msmall_atlas_validation_rejects_scaled_affine(tmp_path: Path) -> None:
    shape = (12, 13, 14)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.ones(shape, dtype=np.uint8)
    paths = {}
    for name, data in {
        "subject": values,
        "reference": values,
        "subject_mask": mask,
        "reference_mask": mask,
        "jacobian": np.ones(shape, dtype=np.float32),
    }.items():
        paths[name] = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), paths[name])
    affine = np.eye(4)
    affine[0, 0] = 1.5
    paths["affine"] = tmp_path / "affine.mat"
    np.savetxt(paths["affine"], affine)

    report = validate_atlas(
        affine_path=paths["affine"],
        subject_path=paths["subject"],
        subject_mask_path=paths["subject_mask"],
        reference_path=paths["reference"],
        reference_mask_path=paths["reference_mask"],
        jacobian_path=paths["jacobian"],
    )

    assert report["valid"] is False
    assert report["affine_singular_values"][0] == 1.5


def test_msmall_subcortical_validation_requires_all_hcp_labels(tmp_path: Path) -> None:
    labels = np.array(
        [26, 58, 18, 54, 16, 11, 50, 8, 47, 28, 60, 17, 53, 13, 52, 12, 51, 10, 49],
        dtype=np.int16,
    ).reshape(19, 1, 1)
    subject = tmp_path / "subject.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.eye(4)), subject)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), reference)

    assert validate_subcortical(subject_path=subject, reference_path=reference)["valid"] is True

    missing = labels.copy()
    missing[0] = 0
    nib.save(nib.Nifti1Image(missing, np.eye(4)), subject)
    report = validate_subcortical(subject_path=subject, reference_path=reference)
    assert report["valid"] is False
    assert report["failures"] == ["subject labels are empty: ACCUMBENS_LEFT"]


def test_msmall_freesurfer_restarts_cleanly_and_bypasses_legacy_talairach_gate() -> None:
    driver = (Path(__file__).parents[1] / "nro/modules/anat/msmall_driver.sh").read_text()

    assert 'rm -rf "$session_root/T1w/$session"' in driver
    assert "--extra-reconall-arg=-notal-check" in driver
    assert "flirt -interp spline -dof 7" in driver
