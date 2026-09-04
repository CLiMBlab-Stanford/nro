import shutil
import tempfile
import unittest
import logging
from itertools import count
from pathlib import Path

from nro.func.synbold_disco import create_synthetic_reference_step
from nro.orchestration.runner import Runner


class _FakeRunner(Runner):
    def __init__(self, *, preserve_transform: bool = True) -> None:
        super().__init__(
            module_name="SynBOLD Test",
            container=None,
            binds=(),
            logger=logging.getLogger("test.synbold.fake-runner"),
            next_step=count(1).__next__,
        )
        self.commands: list[list[str]] = []
        self.preserve_transform = preserve_transform

    def run_child(self, command, **_kwargs) -> None:
        self.commands.append(list(command))
        inputs = next(Path(value.split(":", 1)[0]) for value in command if value.endswith(":/INPUTS:ro"))
        outputs = next(Path(value.split(":", 1)[0]) for value in command if value.endswith(":/OUTPUTS"))
        (outputs / "BOLD_s_3D.nii.gz").write_bytes(b"synthetic")
        if self.preserve_transform:
            shutil.copy2(inputs / "epi_reg_d.mat", outputs / "epi_reg_d.mat")
        else:
            (outputs / "epi_reg_d.mat").write_text("different\n", encoding="utf-8")

    def log_skip(self, *_args, **_kwargs) -> None:
        raise AssertionError("unexpected skip")


class SynboldDiscoTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        bold = root / "bold.nii.gz"
        t1 = root / "t1.nii.gz"
        matrix = root / "epi_to_t1.mat"
        image = root / "synbold.sif"
        license_file = root / "license.txt"
        bold.write_bytes(b"bold")
        t1.write_bytes(b"t1")
        matrix.write_text("1 0 0 2\n0 1 0 3\n0 0 1 4\n0 0 0 1\n", encoding="utf-8")
        image.write_bytes(b"image")
        license_file.write_text("license\n", encoding="utf-8")
        return bold, t1, matrix, image, license_file

    def test_epi_reg_shim_is_bound_over_the_executable_and_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bold, t1, matrix, image, license_file = self._inputs(root)
            runner = _FakeRunner()
            step = create_synthetic_reference_step(
                run_child=runner.run_child,
                distorted_reference=bold,
                skull_stripped_t1=t1,
                epi_to_t1_mat=matrix,
                image=image,
                license_file=license_file,
                engine="singularity",
                work_dir=root / "work",
                force=False,
            )
            runner.add_step(step)
            with runner.run_context():
                runner.execute()
            self.assertTrue(step.outputs[0].is_file())
            command = runner.commands[0]
            self.assertTrue(any(value.endswith(":/opt/fsl/bin/epi_reg:ro") for value in command))
            self.assertNotIn("PREPEND_PATH=/opt/nro/bin", command)
            self.assertTrue((root / "work/outputs/.nro_epi_reg_shim_v2").is_file())

    def test_synthesis_fails_if_container_replaces_precomputed_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bold, t1, matrix, image, license_file = self._inputs(root)
            with self.assertRaisesRegex(SystemExit, "did not honor"):
                runner = _FakeRunner(preserve_transform=False)
                step = create_synthetic_reference_step(
                    run_child=runner.run_child,
                    distorted_reference=bold,
                    skull_stripped_t1=t1,
                    epi_to_t1_mat=matrix,
                    image=image,
                    license_file=license_file,
                    engine="singularity",
                    work_dir=root / "work",
                    force=False,
                )
                runner.add_step(step)
                with runner.run_context():
                    runner.execute()


if __name__ == "__main__":
    unittest.main()
