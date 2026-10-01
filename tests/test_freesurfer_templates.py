import os
import shutil
from pathlib import Path

from nro.modules.anat.freesurfer_templates import ensure_portable_fsaverage


def _fake_template(root: Path) -> Path:
    source = root / "container/fsaverage"
    for relative in ("mri/T1.mgz", "surf/lh.sphere.reg", "surf/rh.sphere.reg"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    return source


def test_fsaverage_is_copied_once_and_linked_relatively(tmp_path):
    source = _fake_template(tmp_path)
    subjects = tmp_path / "project/derivatives/nro/anat/main/code/freesurfer"
    template_root = tmp_path / "project/derivatives/nro/.nro/templates/freesurfer"
    commands = []

    def execute(command):
        commands.append(tuple(command))
        destination = Path(
            command[-1].replace("/nro-template-output", str(template_root / "build"))
        )
        shutil.copytree(source, destination)

    target, created = ensure_portable_fsaverage(
        runtime="apptainer",
        image=tmp_path / "freesurfer.sif",
        subjects_dir=subjects,
        template_root=template_root,
        build="build",
        container_source="/usr/local/freesurfer/subjects/fsaverage",
        execute=execute,
    )
    second_target, second_created = ensure_portable_fsaverage(
        runtime="apptainer",
        image=tmp_path / "freesurfer.sif",
        subjects_dir=subjects,
        template_root=template_root,
        build="build",
        container_source="/usr/local/freesurfer/subjects/fsaverage",
        execute=execute,
    )

    link = subjects / "fsaverage"
    assert created is True
    assert second_created is False
    assert second_target == target
    assert len(commands) == 1
    assert link.is_symlink()
    assert not Path(os.readlink(link)).is_absolute()
    assert link.resolve() == target.resolve()
    assert (target / "nro-template.json").is_file()
