import shutil
from pathlib import Path

from nro.modules.microparcellation.scene import write_workbench_scene


class _WorkbenchStub:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run_child(self, command: list[str]) -> None:
        self.commands.append(command)
        if command[1] == "-surface-average":
            shutil.copyfile(command[4], command[2])
        elif command[1] == "-surface-generate-inflated":
            shutil.copyfile(command[2], command[3])
            shutil.copyfile(command[2], command[4])
        else:  # pragma: no cover - protects the test stub's contract
            raise AssertionError(f"Unexpected Workbench command: {command}")


def test_scene_generates_missing_template_display_surfaces(tmp_path: Path) -> None:
    sources = []
    for hemi in ("L", "R"):
        pial = tmp_path / f"tpl-test_hemi-{hemi}_pial.surf.gii"
        white = tmp_path / f"tpl-test_hemi-{hemi}_white.surf.gii"
        pial.write_text(f"{hemi} pial")
        white.write_text(f"{hemi} white")
        sources.append(pial)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    runner = _WorkbenchStub()
    scene, surfaces = write_workbench_scene(
        output_dir,
        "sub-test",
        tuple(sources),
        output_dir / "labels.dlabel.nii",
        output_dir / "connectivity.pconn.nii",
        runner=runner,
        executable="wb_command",
    )

    assert scene.is_file()
    assert len(surfaces) == 8
    assert all(path.is_file() for path in surfaces)
    assert [command[1] for command in runner.commands] == [
        "-surface-average",
        "-surface-generate-inflated",
        "-surface-average",
        "-surface-generate-inflated",
    ]
    assert not tuple(output_dir.glob(".*veryInflated*"))
