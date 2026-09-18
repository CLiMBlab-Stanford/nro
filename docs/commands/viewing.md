# Viewing and quality control

## `nro scene`

```bash
nro scene -P nptl -p t20 -s fsnative -S 2
nro scene -P nptl -p t20 -m dynconn networks --open
nro scene -P nptl -p t20 -s fsnative -S 2 --publish
```

Common selectors choose completed derivatives. The command combines matching
images, CIFTIs, and surface files without rerunning scientific processing. It
creates one scene for each selected participant, space, smoothing level, and
requested session or run group. The command prints every generated scene path.
Add `--open` to launch the `wb_view` executable beside the configured
`wb_command`. The first call starts a private `srun --x11` viewer broker on the
site's `viewing_partition`. The broker has 2 CPUs and 32 GB of memory and remains
available for its full 12-hour allocation. Later calls reuse it and may open
several independent Workbench processes without waiting for another allocation.
Inside an existing Slurm allocation, the viewer runs directly because Slurm
cannot add X11 forwarding to a nested job step.

Workbench loads the first scene state directly and hides its scene-loader
dialog. Because this operation selects one scene state, `--open` requires
selectors that generate exactly one scene. The viewer broker does not register
as an nro worker or count toward the shared concurrency limit. It remains tied
to the X11 connection that created it. If that display disappears, the next
request replaces the broker. Opening a GUI requires an SSH connection with X
forwarding and a Slurm installation configured to support it.
Broker and Workbench output is written under the current user's private
`viewers/uid-<UID>/` directory in the central nro store.

Linked scenes are the default. They refer directly to source derivatives and
do not duplicate large CIFTIs or surface geometry. `--publish` instead copies
every input beneath the scene directory, rewrites references to those copies,
and records SHA-256 checksums. Published scenes can be moved as a unit.

Scenes are visualization caches, not module artifacts. They live at
`PROJECT/derivatives/scenes/space-SPACE_smoothing-Nmm/sub-ID/SCENE_ID/`.
Regenerating a scene replaces only a directory carrying an nro scene manifest;
the command refuses to replace an unmanaged directory.

Surface scenes provide white, pial, midthickness, and inflated geometry.
Midthickness is the default view. The cerebral montage places the left lateral,
left medial, right medial, and right lateral views in one row.
Dynamic-connectivity scenes expose Workbench's on-demand correlation layer over
concatenated retained frames. Network CIFTIs contain named maps so users can step
through networks. Volume scenes use volume data without carrying surface files.
`--module` can restrict the derivative layers while nro still adds the geometry
needed to display them. See each
[module's output guide](../modules/index.md).

## `nro render`

`render` creates static images from the same combined views as `scene`:

```bash
nro render -P nptl -p t20 -m networks firstlevels
nro render -P nptl -p t20 -m dynconn microparcellation --seeds seeds.yml
```

The command renders every named map in the selected finite-map files. For
example, it creates one image for each network and for each first-level
contrast and statistic. It does not render every frame of a time series.
Outputs normally go into a managed `renders/` directory beside the generated
scene. `--output-dir PATH` chooses another location. When one request creates
several scenes, each scene receives a separate directory beneath that path.

Dynamic connectivity and parcel connectivity need a seed. Without `--seeds`,
the command skips `dynconn` time series and microparcellation connectivity
matrices. A seed file assigns participant ACPC world coordinates, in millimeters, to
subjects:

```yaml
projects:
  nptl:
    t20:
      - xyz_mm: [-24, -4, -18]
      - xyz_mm: [42, -56, 20]
```

For `fsnative` data, nro finds the nearest eligible cortical vertex or volume
voxel. For `ACPC` data, it finds the nearest eligible voxel. The render manifest
records both coordinates, their distance, and the resolved vertex, voxel, or
parcel. A distance above 10 mm produces a warning; change that threshold with
`--warn-seed-distance`. Seed rendering is currently limited to `fsnative` and
`ACPC`, where coordinates and imaging data share the participant's ACPC
coordinate system.

Rendering uses Workbench's headless OSMesa renderer. It does not start the X11
viewer broker or consume an nro worker slot. `--width` sets the image width;
`--format` accepts `png`, `jpg`, or `tiff`. Each managed render directory
contains `render_manifest.yaml`, which links every image to its source map and
map metadata. Regenerating renders replaces only a directory with a matching
nro render manifest.

## `nro qc registration`

Registration QC retains its own parser:

```bash
nro qc registration t20 -p nptl -w main
```

Here the participant is positional and `-p` means **project**, unlike the
common selectors used by `run` and `status`. `-w`/`--workflow` chooses the
anatomical and functional outputs; the default is `main`. `--output-dir` overrides the QC
destination. `--sagittal-coordinate MM` selects the displayed world-space
left/right coordinate (default −20 mm); `--slab-thickness VOXELS` defaults to 3.

The command assembles registration views from completed `anat` and `func`
outputs for visual inspection. It is ad hoc QC, not a scheduled scientific
module and not a numerical pass/fail classifier. Outputs normally live under
the selected `func` derivative's nested
`derivatives/qc/registration/sub-ID/` directory, not as a BIDS datatype beneath
the original subject. Inspect both anatomy and EPI alignment; a plausible image
alone does not validate every transform or motion estimate.

The engine form `python -m nro.qc registration ...` is also supported.
