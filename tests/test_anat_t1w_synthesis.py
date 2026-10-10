from pathlib import Path

import nibabel as nib
import numpy as np

from nro.modules.anat.policy import t1w_synthesis_contract
from nro.modules.anat.t1w_synthesis import (
    create_t1w_support_mask_step,
    create_t1w_synthesis_step,
)


def test_t1w_synthesis_uses_pinned_cpu_synthsr_contract(tmp_path: Path) -> None:
    work = tmp_path / "subject_reference"
    t2w = work / "sub-1_desc-selected_T2w.nii.gz"
    output = work / "sub-1_desc-synthetic_T1w.nii.gz"
    image = tmp_path / "freesurfer.sif"
    license_file = tmp_path / "license.txt"
    for path in (t2w, image, license_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run_child(command, **kwargs):
        calls.append((command, kwargs))

    step = create_t1w_synthesis_step(
        run_child=run_child,
        runtime="singularity",
        image=image,
        license_file=license_file,
        t2w=t2w,
        output=output,
        threads=3,
        env={"OMP_NUM_THREADS": "3"},
        force=False,
    )

    assert step.scientific_signature == step._scientific_signature(t1w_synthesis_contract())
    assert step.action is not None
    step.action()
    command, options = calls.pop()
    assert command[0:3] == ["singularity", "exec", "--cleanenv"]
    assert "mri_synthsr" in command
    assert command[-4:] == [
        "/work/sub-1_desc-synthetic_T1w.nii.gz",
        "--threads",
        "3",
        "--cpu",
    ]
    assert "--cpu" in command
    assert options == {
        "direct": True,
        "env": {"OMP_NUM_THREADS": "3"},
        "discard_stdout": True,
    }


def test_synthetic_t1w_mask_uses_t2w_support_not_synthetic_background(tmp_path: Path) -> None:
    t2w = tmp_path / "selected_T2w.nii.gz"
    synthetic_t1w = tmp_path / "synthetic_T1w.nii.gz"
    output = tmp_path / "synthetic_T1w_mask.nii.gz"

    source_data = np.zeros((7, 7, 7), dtype=np.float32)
    source_data[2:5, 1:6, 2:5] = 10
    nib.save(nib.Nifti1Image(source_data, np.eye(4)), t2w)
    synthetic_data = np.full((9, 9, 9), 0.01, dtype=np.float32)
    nib.save(nib.Nifti1Image(synthetic_data, np.eye(4)), synthetic_t1w)

    step = create_t1w_support_mask_step(
        t2w=t2w,
        synthetic_t1w=synthetic_t1w,
        output=output,
        force=False,
    )
    assert step.action is not None
    step.action()

    mask = nib.load(output)
    values = np.asarray(mask.dataobj)
    assert mask.shape == synthetic_data.shape
    assert np.array_equal(mask.affine, np.eye(4))
    assert set(np.unique(values)) == {0, 1}
    assert int(values.sum()) == 45
    assert not np.any(values[0])
    assert step.inputs == (t2w, synthetic_t1w)
