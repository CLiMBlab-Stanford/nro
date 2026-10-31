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
This behavior is the same from login hosts and existing Slurm allocations.
Use `--serverless` with `--open` to run Workbench directly on the current host.
The current host must already have a usable X11 display.

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

Scenes omit parcel-connectivity matrices (`.pconn.nii`) and microparcellation
label files (`.dlabel.nii`) by default. These files can take several minutes and
substantial memory to load in Workbench. Add `--include-heavy` when the scene
needs them.

Scenes are visualization caches, not module artifacts. They live at
`PROJECT/derivatives/scenes/space-SPACE_smoothing-Nmm/sub-ID/SCENE_ID/`.
Regenerating a scene replaces only a directory carrying an nro scene manifest;
the command refuses to replace an unmanaged directory.

Surface scenes provide white, pial, midthickness, and inflated geometry.
Inflated is the default view. The cerebral montage places the left lateral,
left medial, right medial, and right lateral views in one row.
For lesion-aware anatomy, `nro scene -m anat` also loads the lesion mask, the
synthetic SynthSR T1w image, and a surface family labeled as the intact
synthetic scaffold. The cut surfaces remain the scene's primary display
geometry.
Dynamic-connectivity scenes expose Workbench's on-demand correlation layer over
concatenated retained frames. Network CIFTIs contain named maps so users can step
through networks. Volume scenes use volume data without carrying surface files.
`--module` can restrict the derivative layers while nro still adds the geometry
needed to display them. See each
[module's output guide](../modules/index.md).

## `nro wb_view`

```bash
nro wb_view sub-01_space-T1w_desc-preproc_T1w.nii.gz lh.white.surf.gii
nro wb_view -help
```

`nro wb_view` runs the Workbench viewer that `nro scene --open` uses: the
`wb_view` executable beside the configured `wb_command`. Every argument goes to
Workbench unchanged, so files and options follow Workbench's own syntax. Only a
leading `-h` or `--help` prints nro's description of the command; Workbench's
help option is `-help`.

The viewer runs in the current shell and needs an X11 display. Unlike
`scene --open`, it does not start or reuse the Slurm viewer broker. Start an X11
allocation first when the current host should not run interactive programs.
Workbench reads NIfTI, GIFTI, and CIFTI files. Convert FreeSurfer `.mgz`
volumes and native surfaces before opening them.

## `nro viz`

```bash
nro viz
nro viz -P climb -p c026 --branch main
nro viz --stop
```

`nro viz` serves completed surface derivatives to a web browser. It is an
early prototype that runs beside Workbench and does not replace `nro scene`.
Selectors are optional. `-P`, `-p`, and `--branch` add the initial project,
participant, and branch to the printed URL; the page can change all three.

Each user has one server for each nro checkout. The first call submits it as a
Slurm batch job on the site's `viewing_partition`, with the same CPU, memory,
and wall-time settings as the Workbench viewer broker. The command waits until
the server is running, prints an SSH tunnel command and a URL, and exits:

```text
ssh -N -L 44252:NODE:44252 USER@LOGIN_HOST
http://localhost:44252/?token=TOKEN
```

Run the tunnel command on your own computer and open the URL there. The server
keeps running until `nro viz --stop` or the end of its allocation. Later calls
find the running server and print the same instructions without submitting
another job, so several terminals and browser tabs share one allocation.
`--no-wait` returns as soon as the job is queued; run `nro viz` again once it
starts. Pressing Ctrl-C while waiting leaves the job queued.

The server port is derived from your user ID, so it stays the same from one
server to the next. A server moves to a nearby port only when that one is busy
on its node. The node can differ between servers. `--local-port` changes the
local port named in the printed commands. `--serverless` runs a private server
in the current terminal on the current host, with `--port` to fix its port.

The server listens on every interface of its host, so other cluster users can
reach the port. A request without your access token receives no data. The token
is stored in your private state directory and persists across servers, so a
saved URL keeps working when a new server lands on the same node. Opening the
URL stores the token in a browser cookie. Treat the URL like a password. Server
state and logs live in `viewers/uid-<UID>/viz-<CHECKOUT_DIGEST>/` in the central
nro store.

The server reads each branch's registry on first use. The page's **Refresh
data** button rereads the selected branch, so derivatives that finished later
appear without a restart. **Restart server** replaces the server process inside
the same job so it loads the checkout's current code; `scancel --batch
--signal=USR1 JOB_ID` does the same. The restarted server keeps its port and
token, so tunnels and open pages keep working, but it must read time series
again. Changes to the page's own HTML, CSS, and JavaScript need only a browser
reload.

The page lists every registered branch. It starts on the server checkout's
branch, or on `main` when that branch has no completed derivatives. For the
selected branch it lists every project and participant with completed
derivatives. For each participant, the viewer draws complete
bilateral display-surface families and lists maps whose vertex counts match the
selected surfaces. These maps include GIFTI metrics and labels, CIFTI dense
scalars and labels, and dense time series. Volume images and parcel-connectivity
matrices are not shown.

Hemispheres are splayed 180 degrees by default; they can also be separated or
shown in anatomical position. The splayed layout turns each hemisphere 90 degrees about the vertical axis, so
both lateral surfaces face the viewer side by side with the base of the brain
down. **Reset view** returns to that view, or to the left lateral view in the
other layouts. The view uses an orthographic projection, as Workbench does, so distant parts
of a surface are not drawn smaller. Lateral and medial views show only the
facing hemisphere. After a change of layout, surface, or view, the
camera keeps its direction and rescales so the surfaces fill the canvas width
with a small margin, or its height when that is the tighter limit. Dragging
rotates like a turntable, keeping the superior axis vertical. Select **Free rotation (trackball)** to rotate about any axis.

Layers are drawn over sulcal shading, with the newest layer on top. Shading is
continuous by default; the panel also offers two-tone sulcus and gyrus shading,
as in Workbench, or none. Each layer has its own colors, range, threshold, and
opacity. The red, green, blue, yellow, cyan, and magenta schemes keep one color
and map the value to opacity between the layer's minimum and maximum. The signed
scheme colors positive values up to the maximum and negative values down to the
minimum. A dense time series, such as the `dynconn` output, becomes a
seed-correlation layer. Click the cortex to place a seed, and the server
correlates that vertex with every other vertex. The first seed request reads the
complete series into memory; a 1.2 GB `dynconn` series takes about 20 seconds to
load, and later seeds take under a second. The server keeps loaded series for
later seeds in up to 60 percent of its memory allocation, dropping the least
recently used series first.

**Open** adds a file that is not a registered derivative, such as a
diagnostic or ad hoc map. Enter its path relative to the selected project's
BIDS directory, for example `derivatives/diagnostics/sub-01/NAME.func.gii`. The
viewer accepts GIFTI metrics and labels and CIFTI dense scalars, labels, and
time series. A GIFTI file with a `hemi-L` or `hemi-R` entity opens with its
other hemisphere when that file exists. The file's vertex counts must match the
displayed surfaces. Paths that resolve outside the project directory, including
through symbolic links, are refused.

The browser loads the three.js rendering library from `cdn.jsdelivr.net`, so
the computer running the browser needs internet access. Imaging data travels
only through the SSH tunnel.

## `nro render`

`render` creates static images from the same combined views as `scene`:

```bash
nro render -P nptl -p t20 -m networks contrasts
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
matrices. A seed file assigns participant T1w world coordinates, in millimeters, to
subjects:

```yaml
projects:
  nptl:
    t20:
      - xyz_mm: [-24, -4, -18]
      - xyz_mm: [42, -56, 20]
```

For `fsnative` data, nro finds the nearest eligible cortical vertex or volume
voxel. For `T1w` data, it finds the nearest eligible voxel. The render manifest
records both coordinates, their distance, and the resolved vertex, voxel, or
parcel. A distance above 10 mm produces a warning; change that threshold with
`--warn-seed-distance`. Seed rendering is currently limited to `fsnative` and
`T1w`, where coordinates and imaging data share the participant's T1w
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
