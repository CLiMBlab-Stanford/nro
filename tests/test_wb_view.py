from pathlib import Path

import yaml

from nro.bin.wb_view import _artifact_scene, build_parser


def test_artifact_scene_resolves_the_manifest_inventory(tmp_path: Path) -> None:
    artifact = tmp_path / "space-T1w_smoothing-2mm" / "sub-01"
    artifact.mkdir(parents=True)
    scene = artifact / "sub-01_space-T1w_smoothing-2mm_desc-networks_scene.scene"
    scene.write_text("scene\n", encoding="utf-8")
    manifest = artifact / "sub-01_space-T1w_smoothing-2mm_desc-networks_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump({"outputs": {"scene": scene.name}}),
        encoding="utf-8",
    )

    assert _artifact_scene(manifest) == scene


def test_wb_view_accepts_shared_subject_level_selectors() -> None:
    args = build_parser().parse_args(
        ["networks", "-p", "01", "-P", "demo", "-s", "fsnative", "-S", "2"]
    )
    assert args.derivative_type == "networks"
    assert args.participant == ["01"]
    assert args.space == ["fsnative"]
