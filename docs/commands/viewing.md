# Viewing and quality control

These commands display completed derivatives. `nro scene` builds Connectome
Workbench scenes and can open them, `nro wb_view` runs the Workbench viewer
directly, `nro viz` serves surfaces to a web browser, and `nro render` writes
static images. `nro qc registration` assembles registration views for visual
checks. None of them rerun scientific processing.

## `nro scene`

```bash
nro scene -P nptl -p t20 -s fsnative -S 2
nro scene -P nptl -p t20 -m dynconn networks --open
nro scene -P nptl -p t20 -s fsnative -S 2 --publish
```

`nro scene` combines the images, CIFTIs, and surface files of matching
completed derivatives into Workbench scenes. The shared selectors choose the
derivatives. The command creates one scene for each selected participant,
space, smoothing level, and requested session or run group, and prints the path
of every scene it generates. `--wb-command` overrides the configured
`wb_command` path.

### What a scene contains

Surface scenes include white, pial, midthickness, and inflated geometry, and
open on the inflated surface. The cerebral montage shows the left lateral, left
medial, right medial, and right lateral views in one row. Volume scenes use
volume data and carry no surface files.

`--module` restricts the derivative layers, and nro still adds the geometry
needed to display them. Network CIFTIs contain named maps, so you can step
through the networks. Dynamic-connectivity scenes expose Workbench's on-demand
correlation layer over the concatenated retained frames. For lesion-aware
anatomy, `nro scene -m anat` also loads the lesion mask, the synthetic SynthSR
T1w image, and a surface family labeled as the intact synthetic scaffold. The
cut surfaces remain the primary display geometry. Each
[module's output guide](../modules/index.md) describes its layers.

By default, scenes omit parcel-connectivity matrices (`.pconn.nii`) and
microparcellation label files (`.dlabel.nii`). These files can take several
minutes and a lot of memory to load in Workbench. Add `--include-heavy` to
include them.

### Linked and published scenes

Scenes are linked by default. They refer directly to the source derivatives
and do not duplicate large CIFTIs or surface geometry. `--publish` instead
copies every input beneath the scene directory, rewrites the references to
point at the copies, and records SHA-256 checksums. A published scene can be
moved as a unit.

Scenes are visualization caches, not module artifacts. They live at
`PROJECT/derivatives/scenes/space-SPACE_smoothing-Nmm/sub-ID/SCENE_ID/`.
Regenerating a scene replaces only a directory that carries an nro scene
manifest. The command refuses to replace an unmanaged directory.

### Opening a scene

`--open` launches the `wb_view` executable that sits beside the configured
`wb_command`. Workbench loads the scene state directly and hides its
scene-loader dialog. Because it opens one scene state, `--open` requires
selectors that produce exactly one scene.

Opening a GUI requires an SSH connection with X forwarding and a Slurm
installation that supports it. The first `--open` starts a private
`srun --x11` viewer broker on the site's `viewing_partition`. The broker has 2
CPUs and 32 GB of memory and stays available for its full 12-hour allocation.
Later calls reuse it and can open several independent Workbench processes
without waiting for another allocation. This works the same from login hosts
and from inside existing Slurm allocations.

The broker does not register as an nro worker or count toward the shared
concurrency limit. It is tied to the X11 connection that created it. If that
display goes away, the next request replaces the broker. Broker and Workbench
output goes to your private `viewers/uid-<UID>/` directory in the central nro
store.

Add `--serverless` to `--open` to run Workbench directly on the current host
instead. The host must already have a usable X11 display.

## `nro wb_view`

```bash
nro wb_view sub-01_space-T1w_desc-preproc_T1w.nii.gz lh.white.surf.gii
nro wb_view -help
```

`nro wb_view` runs the same Workbench viewer that `nro scene --open` uses, the
`wb_view` executable beside the configured `wb_command`. Every argument goes to
Workbench unchanged, so files and options follow Workbench's own syntax. Only a
leading `-h` or `--help` prints nro's description of the command. Workbench's
own help option is `-help`.

The viewer runs in the current shell and needs an X11 display. Unlike
`scene --open`, it does not start or reuse the Slurm viewer broker, so start an
X11 allocation first if the current host should not run interactive programs.
Workbench reads NIfTI, GIFTI, and CIFTI files. Convert FreeSurfer `.mgz`
volumes and native surfaces before opening them.

## `nro viz`

```bash
nro viz
nro viz -P climb -p c026 --branch main
nro viz --stop
```

`nro viz` serves completed surface derivatives to a web browser. It is an early
prototype. It runs alongside Workbench and does not replace `nro scene`. It has
its own parser, and its selectors are optional. `-P`, `-p`, and `--branch` set
the project, participant, and branch shown when the page opens. You can change
all three in the page.

### Starting and reaching the server

Each user has one server per nro checkout. The first call submits it as a
Slurm batch job on the site's `viewing_partition`, with the same CPU, memory,
and wall-time settings as the Workbench viewer broker. The command waits until
the server is running, prints an SSH tunnel command and a URL, and exits:

```text
ssh -N -L 44252:NODE:44252 USER@LOGIN_HOST
http://localhost:44252/?token=TOKEN
```

Run the tunnel command on your own computer and open the URL there. The server
runs until `nro viz --stop` or the end of its allocation. Later calls find the
running server and print the same instructions without submitting another job,
so several terminals and browser tabs share one allocation. `--no-wait`
returns as soon as the job is queued, and you can run `nro viz` again once it
starts. Pressing Ctrl-C while waiting leaves the job queued.

The server port is derived from your user ID, so it stays the same from one
server to the next. A server moves to a nearby port only when its port is busy
on its node, and the node can differ between servers. `--local-port` changes
the local port named in the printed commands. `--serverless` runs a private
server in the current terminal on the current host until interrupted, and
`--port` sets its port.

### Access token

The server listens on every interface of its host, so other cluster users can
reach the port. A request without your access token receives no data. The token
is stored in your private state directory and persists across servers, so a
saved URL keeps working when a new server lands on the same node. Opening the
URL stores the token in a browser cookie. Treat the URL like a password.
Server state and logs live in `viewers/uid-<UID>/viz-<CHECKOUT_DIGEST>/` in the
central nro store.

The browser loads the three.js rendering library from `cdn.jsdelivr.net`, so
the computer running the browser needs internet access. Imaging data travels
only through the SSH tunnel.

### Refreshing and restarting

The server reads each branch's registry the first time it is used. The page's
**Refresh data** button rereads the selected branch, so derivatives that
finished later appear without a restart. **Restart server** replaces the server
process inside the same job, so it loads the checkout's current code.
`scancel --batch --signal=USR1 JOB_ID` does the same. The restarted server
keeps its port and token, so tunnels and open pages keep working, but it must
read time series again. Changes to the page's own HTML, CSS, and JavaScript
need only a browser reload.

### What the page shows

The page lists every registered branch. It starts on the branch of the
server's checkout, or on `main` when that branch has no completed derivatives.
For the selected branch it lists every project and participant with completed
derivatives. For each participant, the viewer draws complete bilateral families
of display surfaces and lists the maps whose vertex counts match the selected
surfaces. These maps include GIFTI metrics and labels, CIFTI dense scalars and
labels, and dense time series. Volume images and parcel-connectivity matrices
are not shown.

The layer list is grouped by module. Contrast results from `contrasts` appear
as one layer per node and statistic, such as `node-subject stat-t`. Each
contrast in the model is a map within the layer. Subject-level layers come
first, then session and run layers, and a node's t map precedes its effect,
variance, and degrees-of-freedom maps. A signed map, such as a t map, opens
with the signed color scheme and a range symmetric about zero.

**Open** adds a file that is not a registered derivative, such as a diagnostic
or ad hoc map. Enter its path relative to the selected project's BIDS
directory, for example `derivatives/diagnostics/sub-01/NAME.func.gii`. The
viewer accepts GIFTI metrics and labels and CIFTI dense scalars, labels, and
time series. A GIFTI file with a `hemi-L` or `hemi-R` entity opens together
with its other hemisphere when that file exists. The file's vertex counts must
match the displayed surfaces. Paths that resolve outside the project
directory, including through symbolic links, are refused.

### Layout and camera

By default the hemispheres are splayed 180 degrees. They can also be separated
or shown in anatomical position. The splayed layout turns each hemisphere 90
degrees about the vertical axis, so both lateral surfaces face you side by side
with the base of the brain down. **Reset view** returns to that view, or to the
left lateral view in the other layouts. Lateral and medial views show only the
facing hemisphere.

The view uses an orthographic projection, as Workbench does, so distant parts
of a surface are not drawn smaller. After a change of layout, surface, or view,
the camera keeps its direction and rescales so the surfaces fill the canvas
width with a small margin, or its height when that is the tighter limit.
Dragging rotates like a turntable and keeps the superior axis vertical. Select
**Free rotation (trackball)** to rotate about any axis.

### Layers and seed correlation

Layers are drawn over sulcal shading, with the newest layer on top. Shading is
continuous by default. The panel also offers two-tone sulcus and gyrus shading,
as in Workbench, or no shading. Each layer has its own colors, range,
threshold, and opacity. The red, green, blue, yellow, cyan, and magenta schemes
keep one color and map the value to opacity between the layer's minimum and
maximum. The signed scheme colors positive values up to the maximum and
negative values down to the minimum.

A dense time series, such as the `dynconn` output, becomes a seed-correlation
layer. Click the cortex to place a seed, and the server correlates that vertex
with every other vertex. The first seed request reads the complete series into
memory. A 1.2 GB `dynconn` series takes about 20 seconds to load, and later
seeds take under a second. The server keeps loaded series for later seeds in
up to 60 percent of its memory allocation, and drops the least recently used
series first.

## `nro render`

`nro render` writes static images of the same combined views that `nro scene`
builds:

```bash
nro render -P nptl -p t20 -m networks contrasts
nro render -P nptl -p t20 -m dynconn microparcellation --seeds seeds.yml
```

The command renders every named map in the selected finite-map files, for
example one image per network and one per first-level contrast and statistic.
It does not render the frames of a time series. Images normally go into a
managed `renders/` directory beside the generated scene. `--output-dir PATH`
chooses another location. When one request creates several scenes, each scene
gets its own subdirectory beneath that path.

| Option | Behavior and default |
| --- | --- |
| `--seeds FILE` | YAML file of seed coordinates for connectivity maps. |
| `--warn-seed-distance MM` | Warn when a resolved seed is farther than this. Default 10 mm. |
| `--width PIXELS` | Image width. Default 2400. |
| `--format` | `png`, `jpg`, or `tiff`. Default `png`. |
| `--no-scene-colors` | Use Workbench's rendering colors instead of the colors stored in the scene. |
| `--output-dir PATH` | Write images here instead of the managed `renders/` directory. |
| `--wb-command PATH` | Override the configured `wb_command`. |

### Seeds for connectivity maps

Dynamic connectivity and parcel connectivity need a seed. Without `--seeds`,
the command skips `dynconn` time series and microparcellation connectivity
matrices. A seed file assigns coordinates to subjects, in millimeters in each
participant's T1w world space:

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
parcel. A distance above the `--warn-seed-distance` threshold produces a
warning. Seed rendering is limited to `fsnative` and `T1w`, where coordinates
and imaging data share the participant's T1w coordinate system.

### Renderer and outputs

Rendering uses Workbench's headless OSMesa renderer. It does not start the X11
viewer broker or take an nro worker slot. Each managed render directory
contains `render_manifest.yaml`, which links every image to its source map and
the map's metadata. Regenerating renders replaces only a directory with a
matching nro render manifest.

## `nro qc registration`

`nro qc registration` assembles registration views from completed `anat` and
`func` outputs for visual inspection:

```bash
nro qc registration t20 -p nptl -w main
```

This command has its own parser. The participant is positional and `-p` means
**project**, unlike the shared selectors used by `run` and `status`.
`-w`/`--workflow` chooses which anatomical and functional outputs to show and
defaults to `main`. `--output-dir` overrides the QC destination.
`--sagittal-coordinate MM` sets the world-space left-right coordinate of the
displayed slab, default −20 mm, and `--slab-thickness VOXELS` defaults to 3.

The output is for ad hoc inspection. It is not a scheduled scientific module
and does not classify registrations as pass or fail. Outputs normally go under
the selected `func` derivative, in its nested
`derivatives/qc/registration/sub-ID/` directory, and not as a BIDS datatype
beneath the original subject. Check both the anatomy and the EPI alignment. One
plausible image does not validate every transform or motion estimate.

The engine form `python -m nro.qc registration ...` is also supported.
