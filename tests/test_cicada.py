from __future__ import annotations

import json
from pathlib import Path

import pytest

from nro.modules.func import denoising_steps
from nro.modules.func.cicada import (
    parse_component_indices,
    prepare_melodic_adapter,
    write_result_manifest,
)


def test_parse_component_indices_accepts_empty_and_one_based_lists(tmp_path: Path) -> None:
    path = tmp_path / "components.txt"
    path.write_text("\n", encoding="utf-8")
    assert parse_component_indices(path) == ()
    path.write_text("1,3,7\n", encoding="utf-8")
    assert parse_component_indices(path) == (1, 3, 7)


@pytest.mark.parametrize("value", ["0,2\n", "1,1\n", "one\n"])
def test_parse_component_indices_rejects_invalid_lists(tmp_path: Path, value: str) -> None:
    path = tmp_path / "components.txt"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_component_indices(path)


def test_melodic_adapter_orders_probability_maps_by_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    stats = source / "stats"
    stats.mkdir(parents=True)
    for name in ("melodic_mix", "melodic_FTmix", "melodic_ICstats", "melodic_IC.nii.gz"):
        (source / name).write_text("input\n", encoding="utf-8")
    for number in (10, 2, 1):
        (stats / f"probmap_{number}.nii.gz").write_text("map\n", encoding="utf-8")
    commands: list[list[str]] = []

    def run(command) -> None:
        commands.append(list(command))
        for argument in command:
            if argument.startswith("--out="):
                Path(argument.split("=", 1)[1]).write_text("output\n", encoding="utf-8")
        if command[0] == "fslmerge":
            Path(command[2]).write_text("output\n", encoding="utf-8")

    prepare_melodic_adapter(
        run_command=run,
        source_directory=source,
        output_directory=tmp_path / "adapter",
        reference=tmp_path / "reference.nii.gz",
        warp=tmp_path / "warp.nii.gz",
        premat=tmp_path / "identity.mat",
    )
    assert [Path(path).name for path in commands[0][3:]] == [
        "probmap_1.nii.gz",
        "probmap_2.nii.gz",
        "probmap_10.nii.gz",
    ]


def test_result_manifest_preserves_external_warnings_and_indexing(tmp_path: Path) -> None:
    classification = tmp_path / "cicada_python"
    classification.mkdir()
    (classification / "noise_components.txt").write_text("2,4\n", encoding="utf-8")
    (classification / "signal_components.txt").write_text("1,3\n", encoding="utf-8")
    (classification / "provenance.json").write_text(
        json.dumps({"warnings": ["classification fail-safe applied"]}), encoding="utf-8"
    )
    output = tmp_path / "classification.json"
    write_result_manifest(
        output=output,
        executable=tmp_path / "cicada-python",
        classification_directory=classification,
        tolerance=5,
        smoothing_retention_mode="revised",
        mixing_matrix=tmp_path / "melodic_mix",
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["component_indexing"] == "one-based"
    assert result["noise_components"] == [2, 4]
    assert result["signal_components"] == [1, 3]
    assert result["warnings"] == ["classification fail-safe applied"]


def test_classifier_step_uses_external_command_and_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "cicada-python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    source_melodic = tmp_path / "source/melodic.ica"
    source_melodic.mkdir(parents=True)
    (source_melodic / "melodic_mix").write_text("1 0\n0 1\n", encoding="utf-8")

    class Runner:
        command: list[str] | None = None

        def run_child(self, command, *, env=None):
            raise AssertionError(command)

        def run_direct(self, command, *, env=None):
            self.command = list(command)
            output = Path(command[command.index("--output-dir") + 1]) / "cicada_python"
            output.mkdir(parents=True)
            (output / "noise_components.txt").write_text("2\n", encoding="utf-8")
            (output / "signal_components.txt").write_text("1\n", encoding="utf-8")
            (output / "component_labels.tsv").write_text("component\tlabel\n", encoding="utf-8")
            (output / "provenance.json").write_text(json.dumps({"warnings": []}), encoding="utf-8")

    def prepare(**kwargs) -> None:
        kwargs["output_directory"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(denoising_steps, "prepare_melodic_adapter", prepare)
    runner = Runner()
    inputs = {
        name: tmp_path / name
        for name in (
            "bold.nii.gz",
            "mask.nii.gz",
            "confounds.tsv",
            "melodic.complete",
            "thresholded.nii.gz",
            "warp.nii.gz",
            "identity.mat",
            "fsl.sif",
        )
    }
    for path in inputs.values():
        path.write_text("input\n", encoding="utf-8")
    result_manifest = tmp_path / "classification.json"
    step = denoising_steps._create_cicada_classification_step(
        runner=runner,  # type: ignore[arg-type]
        executable=executable,
        epi_mni=inputs["bold.nii.gz"],
        mask_mni=inputs["mask.nii.gz"],
        confounds=inputs["confounds.tsv"],
        source_melodic=source_melodic,
        melodic_complete=inputs["melodic.complete"],
        thresholded_components_mni=inputs["thresholded.nii.gz"],
        t1_to_mni_warp=inputs["warp.nii.gz"],
        identity_transform=inputs["identity.mat"],
        task_directory=tmp_path / "task",
        adapter_directory=tmp_path / "adapter",
        result_manifest=result_manifest,
        tolerance=5,
        smoothing_retention_mode="revised",
        repetition_time=1.0,
        fsl_image=inputs["fsl.sif"],
        fsl_runtime="singularity",
        fsl_binds=("/data:/data",),
        fsl_setup="source /opt/fsl/etc/fslconf/fsl.sh",
        env={},
        force=False,
    )
    assert step.action is not None
    step.action()
    assert runner.command is not None
    assert runner.command[:2] == [str(executable), "run"]
    assert "--no-denoise" in runner.command
    assert json.loads(result_manifest.read_text(encoding="utf-8"))["noise_components"] == [2]
