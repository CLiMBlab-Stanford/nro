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
`wb_command`. Outside a Slurm allocation, the viewer runs through `srun --x11`
on the site's `viewing_partition`. Inside an allocation, it runs directly on
the allocated node because Slurm cannot add X11 forwarding to a nested job
step. Workbench loads the first scene state directly and hides its scene-loader
dialog. Because this operation selects one scene state, `--open` requires
selectors that generate exactly one scene. The viewer remains attached to the
terminal until it closes. A dedicated viewing allocation does not register as
an nro worker or count toward the shared concurrency limit. Opening a GUI
requires an SSH connection with X forwarding and a Slurm installation
configured to support it.

Linked scenes are the default. They refer directly to source derivatives and
do not duplicate large CIFTIs or surface geometry. `--publish` instead copies
every input beneath the scene directory, rewrites references to those copies,
and records SHA-256 checksums. Published scenes can be moved as a unit.

Scenes are visualization caches, not module artifacts. They live at
`PROJECT/derivatives/scenes/space-SPACE_smoothing-Nmm/sub-ID/SCENE_ID/`.
Regenerating a scene replaces only a directory carrying an nro scene manifest;
the command refuses to replace an unmanaged directory.

Surface scenes provide white, pial, midthickness, and inflated geometry.
Midthickness is the default view. Dynamic-connectivity scenes expose Workbench's
on-demand correlation layer over concatenated retained frames. Network CIFTIs contain named maps so users
can step through networks. Volume scenes use volume data without carrying
surface files. `--module` can restrict the derivative layers while nro still
adds the geometry needed to display them. See each
[module's output guide](../modules/index.md).

## `nro qc registration`

Registration QC retains its own parser:

```bash
nro qc registration t20 -p nptl -w main
```

Here the participant is positional and `-p` means **project**, unlike the
common selectors used by `run` and `status`. `-w`/`--workflow` chooses the
preprocessing lineage; the default is `main`. `--output-dir` overrides the QC
destination. `--sagittal-coordinate MM` selects the displayed world-space
left/right coordinate (default −20 mm); `--slab-thickness VOXELS` defaults to 3.

The command assembles registration views from completed preprocessing outputs
for visual inspection. It is ad hoc QC, not an automatically scheduled sixth
scientific module and not a numerical pass/fail classifier. Outputs normally
live under the preprocessing lineage's nested
`derivatives/qc/registration/sub-ID/` directory, not as a BIDS datatype beneath
the original subject. Inspect both anatomy and EPI alignment; a plausible image
alone does not validate every transform or motion estimate.

The engine form `python -m nro.qc registration ...` is also supported.
