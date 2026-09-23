from __future__ import annotations

import json

from nro.engine.anatomical_domain import load_anatomical_domain
from nro.modules.anat.contract import anatomical_output_contract


def _file(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("data")
    return path


def _manifest(tmp_path, *, lesion: bool):
    observed = _file(tmp_path, "observed.nii.gz")
    brain_mask = _file(tmp_path, "brain-mask.nii.gz")
    surface = _file(tmp_path, "lh.white.surf.gii")
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    outputs = {
        "t1w": str(observed),
        "t2w": None,
        "myelin_map": None,
        "brain_image": str(observed),
        "brain_mask": str(brain_mask),
        "gray_matter_mask": str(_file(tmp_path, "gray.nii.gz")),
        "cortical_ribbon_mask": str(_file(tmp_path, "ribbon.nii.gz")),
        "subcortical_masks": {},
        "surfaces": {"lh.white": str(surface)},
        "mni_qc_images": {},
        "pose_qc": str(_file(tmp_path, "pose.json")),
        "xfms": {},
    }
    document = {
        "subject": "sub-test",
        "fs_subject": "sub-test",
        "fsaverage_template": "fsaverage6",
        "selection_strategy": "first",
        "gradient_unwarping": {},
        "bias_correction": {},
        "surface_reconstruction": {},
        "inputs": {"t1w": ["source.nii.gz"], "t2w": []},
        "copied_session_files": [],
        "outputs": outputs,
        "freesurfer_subjects_dir": str(subjects_dir),
        "mni_template": "template.nii.gz",
        "options": {
            "synthstrip_image": None,
            "configuration": {},
            "configuration_fingerprint": None,
        },
        "output_metadata_contract": anatomical_output_contract(lesion=lesion),
        "complete": True,
    }
    if lesion:
        inpainted = _file(tmp_path, "inpainted.nii.gz")
        lesion_mask = _file(tmp_path, "lesion-mask.nii.gz")
        left_mapping = _file(tmp_path, "left.tsv")
        right_mapping = _file(tmp_path, "right.tsv")
        outputs.update(
            inpainted_t1w=str(inpainted),
            inpainted_t1w_metadata=str(_file(tmp_path, "inpainted.json")),
            lesion_mask=str(lesion_mask),
            lesion_metadata=str(_file(tmp_path, "lesion.json")),
            lesion_probability=str(_file(tmp_path, "probability.nii.gz")),
            lesion_qc=str(_file(tmp_path, "lesion.png")),
            lesion_reconstruction_summary=str(_file(tmp_path, "reconstruction.yaml")),
            surface_vertex_mappings={"lh": str(left_mapping), "rh": str(right_mapping)},
            surface_validity={
                "lh": str(_file(tmp_path, "left.json")),
                "rh": str(_file(tmp_path, "right.json")),
            },
        )
        document["lesion"] = {
            "enabled": True,
            "masker": {},
            "reconstruction": {},
            "boundary_margin_mm": 0.0,
        }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return path, observed


def test_ordinary_anatomy_uses_observed_t1w_for_registration(tmp_path) -> None:
    manifest, observed = _manifest(tmp_path, lesion=False)

    domain, _ = load_anatomical_domain(manifest)

    assert not domain.lesion_aware
    assert domain.registration_t1w == observed
    assert domain.vertex_mappings == {}


def test_lesion_anatomy_uses_inpainted_t1w_and_cut_surface_mappings(tmp_path) -> None:
    manifest, observed = _manifest(tmp_path, lesion=True)

    domain, _ = load_anatomical_domain(manifest)

    assert domain.lesion_aware
    assert domain.observed_t1w == observed
    assert domain.registration_t1w.name == "inpainted.nii.gz"
    assert set(domain.vertex_mappings) == {"lh", "rh"}
