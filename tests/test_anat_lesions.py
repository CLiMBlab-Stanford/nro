from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from nro.modules.anat.contract import anatomical_output_contract
from nro.modules.anat.lesion_policy import (
    MASKER_MODEL,
    MASKER_REVISION,
    NEUROLIT_CHECKPOINTS,
    lesion_reconstruction_contract,
)
from nro.modules.anat.lesions import (
    create_cut_surfaces_step,
    create_fastsurfer_lit_step,
    create_inpainted_metadata_step,
    create_lesion_excluded_mask_step,
    create_lesion_mask_step,
    create_lesion_qc_step,
    retained_surface_vertices,
    validate_lesion_mask,
    validate_lesion_probability,
)
from nro.modules.anat.steps import _write_json_step


def _nifti(path, data):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), str(path))


def _surface(path):
    points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [1, 4, 2]], dtype=np.int32)
    nib.save(
        nib.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
                nib.gifti.GiftiDataArray(triangles, intent="NIFTI_INTENT_TRIANGLE"),
            ]
        ),
        str(path),
    )


def _metric(path, values):
    nib.save(
        nib.GiftiImage(
            darrays=[
                nib.gifti.GiftiDataArray(
                    np.asarray(values, dtype=np.float32), intent="NIFTI_INTENT_SHAPE"
                )
            ]
        ),
        str(path),
    )


def test_anatomical_metadata_declares_sources_before_they_exist(tmp_path) -> None:
    source = tmp_path / "future-mask.nii.gz"
    step = _write_json_step(
        tmp_path / "mask.json",
        {"Type": "ROI mask", "Sources": [str(source)]},
    )

    assert step.inputs == (source,)


def test_lesion_mask_validation_requires_reference_geometry_and_nonempty_mask(tmp_path) -> None:
    reference = tmp_path / "t1.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    _nifti(reference, np.ones((4, 4, 4)))
    values = np.zeros((4, 4, 4))
    values[1, 1, 1] = 1
    _nifti(mask, values)
    probability = tmp_path / "probability.nii.gz"
    _nifti(probability, values * 0.75)

    summary = validate_lesion_mask(mask, reference)
    assert summary["voxel_count"] == 1
    assert summary["volume_mm3"] == 1.0
    assert summary["component_count"] == 1
    assert summary["component_volumes_mm3"] == [1.0]
    validate_lesion_probability(probability, mask, reference, 0.5)

    _nifti(mask, np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="empty"):
        validate_lesion_mask(mask, reference)


def test_lesion_qc_renders_atomic_png(tmp_path) -> None:
    reference = tmp_path / "t1.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    output = tmp_path / "qc.png"
    anatomy = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    lesion = np.zeros_like(anatomy)
    lesion[2:4, 2:4, 3:5] = 1
    _nifti(reference, anatomy)
    _nifti(mask, lesion)

    step = create_lesion_qc_step(source=reference, mask=mask, output=output, force=False)
    step.action()

    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / ".partial-qc.png").exists()
    assert step.validate is not None and step.validate()[0]


def test_lesion_is_removed_from_scaffold_volumetric_mask(tmp_path) -> None:
    scaffold = tmp_path / "scaffold.nii.gz"
    lesion = tmp_path / "lesion.nii.gz"
    output = tmp_path / "public.nii.gz"
    _nifti(scaffold, np.ones((4, 4, 4)))
    lesion_values = np.zeros((4, 4, 4))
    lesion_values[1:3, 1:3, 1:3] = 1
    _nifti(lesion, lesion_values)

    step = create_lesion_excluded_mask_step(
        scaffold_mask=scaffold,
        lesion_mask=lesion,
        output=output,
        force=False,
    )
    step.action()

    assert step.validate is not None and step.validate()[0]
    values = np.asanyarray(nib.load(str(output)).dataobj)
    assert int(values.sum()) == 56


def test_builtin_lesion_adapter_runs_as_a_python_module(tmp_path) -> None:
    reference = tmp_path / "t1.nii.gz"
    probability = tmp_path / "probability.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    metadata = tmp_path / "mask.json"
    command = tmp_path / "python"
    command.write_text("python")
    values = np.zeros((4, 4, 4))
    values[1, 1, 1] = 0.75
    _nifti(reference, np.ones((4, 4, 4)))
    calls = []

    def run_child(invocation, **options):
        calls.append(invocation)
        probability_path = invocation[invocation.index("--probability") + 1]
        mask_path = invocation[invocation.index("--mask") + 1]
        _nifti(probability_path, values)
        _nifti(mask_path, values >= 0.5)

    step = create_lesion_mask_step(
        run_child=run_child,
        command=command,
        module="nro.modules.anat.synthstroke",
        source=reference,
        probability=probability,
        mask=mask,
        metadata=metadata,
        model_directory=tmp_path / "models",
        model=MASKER_MODEL,
        revision=MASKER_REVISION,
        threshold=0.5,
        test_time_augmentation=True,
        use_gpu=True,
        force=False,
    )
    step.action()

    assert calls[0][:3] == [str(command), "-m", "nro.modules.anat.synthstroke"]
    assert calls[0][calls[0].index("--device") + 1] == "cuda"
    assert calls[0][calls[0].index("--model-directory") + 1] == str(tmp_path / "models")
    assert step.validate is not None and step.validate()[0]


def test_fastsurfer_lit_uses_pinned_read_only_model_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    image = tmp_path / "fastsurfer.sif"
    t1w = tmp_path / "input" / "t1.nii.gz"
    lesion = tmp_path / "input" / "lesion.nii.gz"
    license_file = tmp_path / "license.txt"
    data = tmp_path / "models"
    for path in (image, t1w, lesion, license_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")
    for name in NEUROLIT_CHECKPOINTS:
        checkpoint = data / "LIT" / "weights" / name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("model")
    calls = []

    def run_child(command, **options):
        calls.append((command, options))
        for output in step.outputs[:-1]:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("result")

    step = create_fastsurfer_lit_step(
        run_child=run_child,
        runtime="singularity",
        image=image,
        data_directory=data,
        t1w=t1w,
        lesion_mask=lesion,
        subjects_dir=tmp_path / "subjects",
        subject="sub-test",
        license_file=license_file,
        version="2.5.4",
        use_gpu=True,
        threads=2,
        force=False,
    )
    step.action()

    inpaint, inpaint_options = calls[0]
    reconstruct, reconstruct_options = calls[1]
    assert inpaint[:3] == ["singularity", "exec", "--nv"]
    assert f"{data}:/nro-lit-data:ro" in inpaint
    assert "XDG_DATA_HOME=/nro-lit-data" in inpaint
    assert inpaint[inpaint.index("--device") + 1] == "cuda"
    assert inpaint[inpaint.index("--batch_size") + 1] == "8"
    assert "CUDA_VISIBLE_DEVICES=3" in inpaint
    assert "--lesion_mask" in inpaint
    assert "/fastsurfer/run_fastsurfer.sh" in reconstruct
    assert "--lesion_mask" not in reconstruct
    assert reconstruct[reconstruct.index("--t1") + 1] == (
        "/output/sub-test/mri/inpainted.lit.nii.gz"
    )
    assert reconstruct[reconstruct.index("--vox_size") + 1] == "1.0"
    assert lesion_reconstruction_contract()["fastsurfer_voxel_size_mm"] == 1.0
    assert "--no_cereb" in reconstruct
    assert "--no_hypothal" in reconstruct
    assert "--no_cc" not in reconstruct
    assert inpaint_options == {"direct": True, "stream_output": True}
    assert reconstruct_options == {"direct": True, "stream_output": True}
    assert set(data / "LIT" / "weights" / name for name in NEUROLIT_CHECKPOINTS).issubset(
        step.inputs
    )
    assert step.validate is not None and step.validate()[0]


def test_surface_cut_removes_lesion_faces_and_compacts_vertices(tmp_path) -> None:
    surface = tmp_path / "surface.surf.gii"
    lesion = tmp_path / "lesion.shape.gii"
    _surface(surface)
    _metric(lesion, [0, 0, 0, 1, 0])

    retained, triangles = retained_surface_vertices(surface, lesion)

    assert retained.tolist() == [0, 1, 2, 4]
    assert triangles.tolist() == [[0, 1, 2], [1, 3, 2]]

    output_surface = tmp_path / "cut.surf.gii"
    source_metric = tmp_path / "thickness.shape.gii"
    output_metric = tmp_path / "cut-thickness.shape.gii"
    mapping = tmp_path / "mapping.tsv"
    summary = tmp_path / "validity.json"
    _metric(source_metric, [10, 11, 12, 13, 14])
    step = create_cut_surfaces_step(
        scaffold_surface=surface,
        lesion_metric=lesion,
        surfaces={surface: output_surface},
        metrics={source_metric: output_metric},
        mapping=mapping,
        summary=summary,
        force=False,
    )
    step.action()

    cut_points = (
        nib.load(str(output_surface)).get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
    )
    cut_metric = nib.load(str(output_metric)).darrays[0].data
    assert cut_points.shape == (4, 3)
    assert cut_metric.tolist() == [10, 11, 12, 14]
    assert mapping.read_text().splitlines()[-1] == "3\t4"
    assert step.validate is not None and step.validate()[0]
    assert '"ExcludedVertexCount": 1' in summary.read_text()


def test_inpainted_metadata_binds_synthetic_image_to_mask(tmp_path) -> None:
    image = tmp_path / "inpainted.nii.gz"
    observed = tmp_path / "observed.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    lesion_metadata = tmp_path / "mask.json"
    reconstruction_summary = tmp_path / "reconstruction.yaml"
    output = tmp_path / "inpainted.json"
    for path in (image, observed, mask, reconstruction_summary):
        path.write_bytes(b"image")
    lesion_metadata.write_text(
        '{"Model": "example", "ModelRevision": "revision"}', encoding="utf-8"
    )

    step = create_inpainted_metadata_step(
        image=image,
        observed_t1w=observed,
        lesion_mask=mask,
        lesion_metadata=lesion_metadata,
        reconstruction_summary=reconstruction_summary,
        output=output,
        force=False,
    )
    step.action()

    assert step.validate is not None and step.validate()[0]
    assert '"LesionMaskSHA256"' in output.read_text(encoding="utf-8")


def test_lesion_output_contract_extends_only_lesion_artifacts() -> None:
    ordinary = anatomical_output_contract()
    lesion = anatomical_output_contract(lesion=True)

    assert "outputs.lesion_mask" not in ordinary["publication_manifest_fields"]
    assert lesion["publication_manifest_fields"]["outputs.lesion_mask"] == "string"
    assert lesion["publication_manifest_fields"]["outputs.lesion_qc"] == "string"
    assert lesion["publication_manifest_fields"]["outputs.surface_validity"] == "mapping"
