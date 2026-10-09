from pathlib import Path

from nro.modules.anat.policy import t1w_synthesis_contract
from nro.modules.anat.t1w_synthesis import create_t1w_synthesis_step


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
