"""Build a lightweight Workbench audit of functional-to-anatomical registration."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from nro.configuration.paths import BIDS_PATH
from nro.configuration.runtime import load_runtime_configuration
from nro.orchestration.runtime import resolve_workflow_runtime

_SCENE_TEMPLATE = Path(__file__).with_name("registration_audit.scene.in")
_TEMPLATE_VOLUME = "sub-c001_desc-firstvolsAcrossRuns_leftSagSlab_bold.nii.gz"
_TEMPLATE_DYNAMIC_VOLUME = "sub-c001_desc-firstvolsAcrossRuns_leftSagSlab_bold.vol_dynconn"
_TEMPLATE_SURFACES = {
    ("L", "pial"): "sub-c001_space-fsnative_hemi-L_pial.surf.gii",
    ("L", "white"): "sub-c001_space-fsnative_hemi-L_white.surf.gii",
    ("R", "pial"): "sub-c001_space-fsnative_hemi-R_pial.surf.gii",
    ("R", "white"): "sub-c001_space-fsnative_hemi-R_white.surf.gii",
}


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)
    )


def _run_sort_key(path: Path) -> tuple[object, ...]:
    name = path.name
    entities = {
        key: match.group(1) if (match := re.search(rf"(?:^|_){key}-([^_]+)", name)) else ""
        for key in ("ses", "task", "run", "echo")
    }
    return (
        _natural_key(entities["ses"]),
        _natural_key(entities["task"]),
        _natural_key(entities["run"]),
        _natural_key(entities["echo"]),
        _natural_key(str(path)),
    )


def find_registered_bold(subject_dir: Path) -> list[Path]:
    """Find final preprocessed BOLD series registered to the subject T1w space."""
    files = {
        path
        for path in subject_dir.glob("**/func/*_space-T1w_desc-preproc_bold.nii*")
        if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
    }
    return sorted(files, key=_run_sort_key)


def find_surfaces(subject_dir: Path, subject: str) -> dict[tuple[str, str], Path]:
    """Locate the registration surfaces required by the QC montage."""
    anat_dirs = [subject_dir / "anat", *sorted(subject_dir.glob("ses-*/anat"))]
    result: dict[tuple[str, str], Path] = {}
    for key in _TEMPLATE_SURFACES:
        hemi, surface = key
        matches = sorted(
            {
                path
                for anat_dir in anat_dirs
                for path in anat_dir.glob(f"{subject}*_hemi-{hemi}_{surface}.surf.gii")
                if path.is_file()
            }
        )
        if not matches:
            raise FileNotFoundError(
                f"Missing {hemi} {surface} surface under {subject_dir}; "
                "run anatomical preprocessing first"
            )
        if len(matches) > 1:
            listed = "\n  ".join(str(path) for path in matches)
            raise RuntimeError(f"Multiple candidate {hemi} {surface} surfaces found:\n  {listed}")
        result[key] = matches[0]
    return result


def find_anatomical_reference(subject_dir: Path, subject: str) -> Path:
    """Find the preprocessed T1w volume whose coordinates contain the surfaces."""
    anat_dirs = [subject_dir / "anat", *sorted(subject_dir.glob("ses-*/anat"))]
    matches = sorted(
        {
            path
            for anat_dir in anat_dirs
            for path in anat_dir.glob(f"{subject}*_desc-preproc_T1w.nii*")
            if path.is_file() and path.name.endswith((".nii", ".nii.gz"))
        }
    )
    if not matches:
        raise FileNotFoundError(f"Missing preprocessed T1w reference under {subject_dir}")
    if len(matches) > 1:
        listed = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple candidate preprocessed T1w references found:\n  {listed}")
    return matches[0]


def _first_volume(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    if len(image.shape) == 3:
        return np.asarray(image.dataobj, dtype=np.float32)
    if len(image.shape) == 4 and image.shape[3] > 0:
        return np.asarray(image.dataobj[..., 0], dtype=np.float32)
    raise ValueError(f"Expected a nonempty 3D or 4D image, got shape {image.shape}")


def _world_bounds(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, np.ndarray]:
    shape = np.asarray(image.shape[:3], dtype=float)
    corners = np.array(
        [
            (i, j, k)
            for i in (0.0, shape[0] - 1.0)
            for j in (0.0, shape[1] - 1.0)
            for k in (0.0, shape[2] - 1.0)
        ]
    )
    xyz = nib.affines.apply_affine(image.affine, corners)
    return np.min(xyz, axis=0), np.max(xyz, axis=0)


def _world_aligned_sagittal_grid(
    anatomical: nib.spatialimages.SpatialImage,
    *,
    sagittal_coordinate: float,
    slab_thickness: int,
    voxel_size: float,
) -> tuple[tuple[int, int, int], np.ndarray]:
    if slab_thickness < 1:
        raise ValueError("Slab thickness must be at least one voxel")
    if voxel_size <= 0:
        raise ValueError("QC voxel size must be positive")
    lower, upper = _world_bounds(anatomical)
    y0 = float(lower[1])
    z0 = float(lower[2])
    ny = int(np.ceil((upper[1] - y0) / voxel_size)) + 1
    nz = int(np.ceil((upper[2] - z0) / voxel_size)) + 1
    x0 = float(sagittal_coordinate) - 0.5 * (int(slab_thickness) - 1) * voxel_size
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    affine[:3, 3] = (x0, y0, z0)
    return (int(slab_thickness), ny, nz), affine


def _write_spec(
    path: Path,
    surfaces: dict[tuple[str, str], Path],
    volume: Path,
) -> None:
    entries = []
    for hemi, structure in (("L", "CortexLeft"), ("R", "CortexRight")):
        for surface in ("white", "pial"):
            entries.append(
                f'   <DataFile Structure="{structure}" DataFileType="SURFACE" Selected="true">'
                f"\n      {surfaces[(hemi, surface)].name}\n   </DataFile>"
            )
    entries.append(
        '   <DataFile Structure="Invalid" DataFileType="VOLUME" Selected="true">'
        f"\n      {volume.name}\n   </DataFile>"
    )
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<CaretSpecFile Version="1.0">\n'
        "   <MetaData>\n   </MetaData>\n" + "\n".join(entries) + "\n</CaretSpecFile>\n",
        encoding="utf-8",
    )


def _write_scene(
    path: Path,
    volume: Path,
    surfaces: dict[tuple[str, str], Path],
    slab_shape: tuple[int, int, int],
    slab_affine: np.ndarray,
) -> None:
    text = _SCENE_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        _TEMPLATE_VOLUME: volume.name,
        _TEMPLATE_DYNAMIC_VOLUME: volume.name.removesuffix(".nii.gz") + ".vol_dynconn",
    }
    replacements.update({_TEMPLATE_SURFACES[key]: value.name for key, value in surfaces.items()})
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"Registration QC scene template is missing placeholder {old}")
        text = text.replace(old, new)

    center_ijk = np.array([0.5 * (size - 1) for size in slab_shape] + [1.0])
    center_xyz = np.asarray(slab_affine) @ center_ijk
    coordinates = {
        "m_sliceCoordinateParasagittal": center_xyz[0],
        "m_sliceCoordinateCoronal": center_xyz[1],
        "m_sliceCoordinateAxial": center_xyz[2],
    }
    for name, coordinate in coordinates.items():
        text = re.sub(
            rf'(<Object Type="float" Name="{name}">)[^<]+(</Object>)',
            rf"\g<1>{float(coordinate):.6f}\g<2>",
            text,
        )
    # The template's preview depicts its original example subject, not this audit.
    text = re.sub(r'<Image Encoding="Base64" Format="png">.*?</Image>', "", text, flags=re.DOTALL)
    text = text.replace("New Scene 1", "Functional registration audit")
    path.write_text(text, encoding="utf-8")


def create_registration_audit(
    *,
    subject_dir: Path,
    output_dir: Path,
    subject: str,
    sagittal_coordinate: float = -20.0,
    slab_thickness: int = 3,
    bold_files: list[Path] | None = None,
) -> dict[str, Path]:
    """Create a scene using anatomy from subject_dir and optional resolved BOLD inputs."""
    bold_files = find_registered_bold(subject_dir) if bold_files is None else bold_files
    if not bold_files:
        raise FileNotFoundError(
            f"No *_space-T1w_desc-preproc_bold.nii[.gz] files found under {subject_dir}"
        )
    source_surfaces = find_surfaces(subject_dir, subject)
    anatomical_path = find_anatomical_reference(subject_dir, subject)
    anatomical = nib.load(str(anatomical_path))

    images: list[tuple[Path, nib.spatialimages.SpatialImage]] = []
    for path in bold_files:
        image = nib.load(str(path))
        _first_volume(image)  # Fail before creating an incomplete output bundle.
        images.append((path, image))

    source_voxel_sizes = [
        float(np.min(nib.affines.voxel_sizes(image.affine))) for _, image in images
    ]
    qc_voxel_size = float(np.median(source_voxel_sizes))
    slab_shape, slab_affine = _world_aligned_sagittal_grid(
        anatomical,
        sagittal_coordinate=sagittal_coordinate,
        slab_thickness=slab_thickness,
        voxel_size=qc_voxel_size,
    )

    slabs: list[np.ndarray] = []
    run_records: list[dict[str, object]] = []
    for map_index, (path, image) in enumerate(images, start=1):
        volume = _first_volume(image)
        source = nib.Nifti1Image(volume, image.affine, image.header)
        slab = resample_from_to(source, (slab_shape, slab_affine), order=1)
        slabs.append(np.asarray(slab.dataobj, dtype=np.float32))
        run_records.append(
            {
                "map": map_index,
                "source_file": str(path.resolve()),
                "resampled_for_qc": True,
                "source_shape": "x".join(str(value) for value in image.shape[:3]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}_desc-registrationAudit"
    volume_path = output_dir / f"{prefix}_firstvols_bold.nii.gz"
    stacked = nib.Nifti1Image(np.stack(slabs, axis=3), slab_affine)
    stacked.header.set_data_dtype(np.float32)
    stacked.header.set_zooms((qc_voxel_size, qc_voxel_size, qc_voxel_size, 1.0))
    stacked.header.set_xyzt_units("mm", "sec")
    nib.save(stacked, volume_path)

    packaged_surfaces: dict[tuple[str, str], Path] = {}
    for key, source in source_surfaces.items():
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        packaged_surfaces[key] = destination

    index_path = output_dir / f"{prefix}_index.tsv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("map", "source_file", "resampled_for_qc", "source_shape"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(run_records)

    spec_path = output_dir / f"{prefix}.spec"
    scene_path = output_dir / f"{prefix}.scene"
    _write_spec(spec_path, packaged_surfaces, volume_path)
    _write_scene(scene_path, volume_path, packaged_surfaces, slab_shape, slab_affine)

    actual_center = np.array([0.5 * (size - 1) for size in slab_shape] + [1.0])
    actual_xyz = slab_affine @ actual_center
    metadata_path = output_dir / f"{prefix}.json"
    metadata = {
        "Description": "Workbench audit of final functional-to-T1w registrations.",
        "AnatomicalReference": str(anatomical_path.resolve()),
        "QCGridOrientation": "world-aligned RAS sagittal slab",
        "QCVoxelSizeMm": qc_voxel_size,
        "RequestedSagittalCoordinateMm": float(sagittal_coordinate),
        "DisplayedSagittalCoordinateMm": float(actual_xyz[0]),
        "SlabVoxelAxis": 0,
        "SlabThicknessVoxels": slab_shape[0],
        "RunCount": len(run_records),
        "Runs": run_records,
        "Surfaces": {
            f"{hemi}.{surface}": path.name for (hemi, surface), path in packaged_surfaces.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "scene": scene_path,
        "spec": spec_path,
        "volume": volume_path,
        "index": index_path,
        "metadata": metadata_path,
    }


def registration_output_dir(
    derivative_root: Path,
    subject: str,
    session: str | None = None,
) -> Path:
    """Return the QC directory parallel to the evaluated derivative entities."""
    output_dir = derivative_root / "derivatives" / "qc" / "registration" / subject
    if session is not None:
        output_dir /= session
    return output_dir


def build_parser(*, prog: str = "python -m nro.qc registration") -> argparse.ArgumentParser:
    """Build the registration quality-control parser."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("participant", help="BIDS participant label, with or without sub-")
    parser.add_argument(
        "-p", "--project", required=True, help="Project directory under the Climblab BIDS root"
    )
    parser.add_argument(
        "-w",
        "--workflow",
        default="main",
        help="Workflow ID whose preprocessing lineage should be audited",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default nested registration-QC derivative directory",
    )
    parser.add_argument(
        "--sagittal-coordinate",
        type=float,
        default=-20.0,
        metavar="MM",
        help="World-space left/right coordinate to display (default: -20 mm)",
    )
    parser.add_argument("--slab-thickness", type=int, default=3, metavar="VOXELS")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    prog: str = "python -m nro.qc registration",
) -> None:
    """Create registration montages for one participant's derivatives."""
    args = build_parser(prog=prog).parse_args(argv)
    subject = args.participant if args.participant.startswith("sub-") else f"sub-{args.participant}"
    from nro.orchestration.branch_views import registered_rows

    rows = registered_rows(Path(BIDS_PATH))
    if rows is not None:
        from nro.configuration.site import CHECKOUT, settings
        from nro.orchestration.branch_store import BranchStore
        from nro.orchestration.branches import BranchPaths

        values = settings()[0]
        branches = BranchStore(Path(values["registry"]))
        name = branches.read().topology.require_checkout(CHECKOUT)
        if name == "main":
            from nro.orchestration.releases import ReleaseStore

            ReleaseStore(branches).require_approved(CHECKOUT)
        selected = [
            row
            for row in rows
            if row["project"] == args.project
            and row["participant"] == subject.removeprefix("sub-")
            and args.workflow in row.get("workflow_ids", "").split(",")
        ]
        anatomicals = [row for row in selected if row["module"] == "anat"]
        if len(anatomicals) != 1:
            raise SystemExit(
                "Registration QC requires one registered anatomical instance for the selected workflow"
            )
        paths = BranchPaths(name, *(Path(values[key]) for key in ("bids", "work", "development")))
        anatomical = anatomicals[0]
        subject_dir = Path(anatomical["output_root"]).parent
        derivative_root = (
            paths.output_project(args.project)
            / "derivatives/preprocessing"
            / anatomical["directory_label"]
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else registration_output_dir(derivative_root, subject)
        )
        paths.require_output(output_dir, args.project)
        bold = sorted(
            {
                path
                for row in selected
                if row["module"] == "func"
                for path in Path(row["output_root"]).glob(
                    f"{row['output_prefix']}_space-T1w_desc-preproc_bold.nii*"
                )
                if path.is_file()
            },
            key=_run_sort_key,
        )
        outputs = create_registration_audit(
            subject_dir=subject_dir,
            output_dir=output_dir,
            subject=subject,
            sagittal_coordinate=args.sagittal_coordinate,
            slab_thickness=args.slab_thickness,
            bold_files=bold,
        )
        print(
            f"Registration audit includes {len(bold)} functional runs.\nScene: {outputs['scene']}\nRun index: {outputs['index']}"
        )
        return
    preprocessing_id, _ = load_runtime_configuration(
        resolve_workflow_runtime(
            project=args.project,
            workflow_id=args.workflow,
            derivative_class="preprocessing",
        ),
        "preprocessing",
    )
    derivative_root = (
        Path(BIDS_PATH) / args.project / "derivatives" / "preprocessing" / preprocessing_id
    ).resolve()
    subject_dir = derivative_root / subject
    if not subject_dir.is_dir():
        raise SystemExit(f"Missing preprocessing subject directory: {subject_dir}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else registration_output_dir(derivative_root, subject)
    )
    try:
        outputs = create_registration_audit(
            subject_dir=subject_dir,
            output_dir=output_dir,
            subject=subject,
            sagittal_coordinate=args.sagittal_coordinate,
            slab_thickness=args.slab_thickness,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Registration audit includes {len(find_registered_bold(subject_dir))} functional runs.")
    print(f"Scene: {outputs['scene']}")
    print(f"Run index: {outputs['index']}")
    print(f"Open with: wb_view {outputs['scene']}")


if __name__ == "__main__":
    main()
