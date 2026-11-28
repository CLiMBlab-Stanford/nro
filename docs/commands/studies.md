# Study analyses

A study analysis is your own Python code that reads nro outputs, computes
something per participant or per run, and writes files that nro tracks for
freshness. Each study is a Git repository under the site `study_code`
directory. Its results go under the site `study_results` directory, at the same
relative path, and never into a project's `derivatives/`.

| Setting | CLIMBLAB default |
| --- | --- |
| `study_code` | `/juice6/u/nlp/climblab/STUDIES/CODE` |
| `study_results` | `/juice6/u/nlp/climblab/STUDIES/RESULTS` |

Change them with `nro site set --maintain storage.study_code=DIR
storage.study_results=DIR`. Existing studies and results are not moved.

`nro run --study-analysis STUDY/ANALYSIS` plans an analysis through the shared
scheduler, together with any upstream work its inputs need, and creates one
request per member project. `--study-ref` names the study branch, tag, or
commit to plan from; the default is `main`. `--project` and `--participant`
narrow the request. `nro study run` instead executes analyses directly from the
command line, one work item after another. `query`, `check`, `try`, and `run`
read upstream state from the scheduler's last published status snapshot, which
can trail the registry by one scheduler update. A registered checkout reads its
own branch's snapshot, and an unregistered development checkout reads `main`.
Only a standalone installation without branch registries reads the
per-project registry directly.

## Create a study and an analysis

```bash
nro study init a001 --project a001
nro study add a001/between_run_similarity
```

`init` creates the repository, writes `study.yml`, `pyproject.toml`, and
`common/`, creates the results directory, and makes the first commit. `add`
writes `analysis.yml`, `analysis.py`, and `test_analysis.py` under
`analyses/between_run_similarity/` and does not commit them.

```text
STUDIES/CODE/a001/
  study.yml
  pyproject.toml
  common/a001_common/__init__.py
  analyses/between_run_similarity/
    analysis.yml
    analysis.py
    test_analysis.py
```

`study.yml` names the study, lists its member projects, and records the
`nro.analysis` API version (`api: 1`). It may also name a default `container`.
Code shared by several analyses goes in `common/`. When an analysis runs, only
`common/` and its own directory are added to `sys.path`.

nro reads committed code only. `nro study run` and `nro study check` resolve
`main`, or the ref given with `--study-ref`, to a commit. They read the manifest
and code from that commit, so uncommitted edits have no effect on them. The
repository accepts pushes to its checked-out branch
(`receive.denyCurrentBranch=updateInstead`), so you can work in a personal
clone and push to `main`.

## Manifest

This analysis computes, for each participant, the Fisher-transformed spatial
correlation of one contrast's t map between every pair of runs:

```yaml
version: 1
scope: subject
inputs:
  tmaps:
    from: contrasts
    workflow: main
    select: {model: main, task: lang, node: run, stat: t, suffix: statmap}
    many: true
    min: 2
outputs:
  similarity: "sub-{participant}_model-main_task-lang_contrast-{contrast}_desc-betweenrun_similarity.tsv"
parameters:
  contrast: {expand: [SvN, SvW]}
  exclude_medial_wall: true
resources: {class: small}
```

| Key | Meaning |
| --- | --- |
| `version` | Integer you control. Changing it makes every result stale. |
| `entry` | Function name in `analysis.py`. Defaults to `run`. |
| `environment.container` | Image to run in. Defaults to the study's `container`. Without one, the analysis runs in nro's Python environment. |
| `scope` | `subject` (one work item per participant), `run` (one per raw BOLD run), or `project` (one per member project, reading many participants). |
| `participants` | Project scope only: which participants the work item reads. See [project scope](#project-scope). |
| `inputs` | Named inputs, described below. |
| `outputs` | Slot name to filename pattern. Every work item has exactly these slots. |
| `parameters` | Values passed to the function. |
| `resources` | `class` (`small`, `medium`, or `large`) and optional `memory_gb`. Recorded only for now. |
| `viz` | A viewer page for `nro viz`, such as `viz/index.html`. See [viewer pages](#viewer-pages). |

### Parameters and expansion

A parameter is passed to the function unchanged, including lists. A mapping
whose only key is `expand` makes one work item per element:

```yaml
parameters:
  contrast: {expand: [S-N, W-N]}       # params["contrast"] is "S-N" or "W-N"
  rois: [IFG, MFG, AntTemp]            # the same list for every work item
  pairs: {expand: {SN: [S, N], WN: [W, N]}}   # labelled list values
```

Several expanded parameters form a Cartesian product. Every output pattern must
contain every expanded parameter. In a pattern, a value is replaced by its BIDS
label: non-alphanumeric characters are removed, so `S-N` becomes `SN`. List
values need the labelled mapping form, whose keys are the labels. Two values
with the same label are rejected.

Patterns may also use `{project}` and `{participant}`. Subject-scope patterns
must contain `{participant}`. Run-scope patterns must contain `{stem}`, the raw
run's BIDS stem, and may use the run's entities such as `{task}` and `{run}`.

### Inputs

Each input names a source, an upstream configuration, and a selector:

| Key | Meaning |
| --- | --- |
| `from` | An nro module, such as `contrasts`, `raw` for source BIDS files, or `analysis:NAME` for another analysis in the same study. |
| `workflow` | Use the lineage that this workflow selects for the module. Defaults to `main`. |
| `config` | Use this configuration ID of the module instead. Fails if workflows that select it lead to more than one lineage. |
| `select` | Entity filter on filenames. A scalar requires equality, a list accepts any listed value, and `null` requires the entity to be absent. `suffix` and `extension` (such as `.dscalar.nii`) can be selected too. |
| `many` | Accept several files, optionally bounded by `min` and `max`. |
| `optional` | Pass `None` when nothing matches. |
| `path` | For `raw` only: a glob below the project root, which may use `{participant}`. Without it, `raw` matches the participant's BOLD runs. |

`workflow` and `config` may name an expanded parameter, as in
`workflow: "{pipeline}"`, to run the analysis once per workflow.

Candidates are the files that fresh upstream work items wrote for the
participant, as recorded in the registry, for the workflow's latest registered
revision. JSON sidecars are left out unless `select` names an `extension`.
In run scope, a candidate that carries a run entity must match the work item's
run.

A single-file input that matches nothing, or more than one file, is an error
that lists the candidates. A `many` input with fewer than `min` matches makes
the work item not applicable. When the files are missing because upstream work
items are not fresh yet, the work item is reported as waiting.

All inputs are resolved and passed together in one call. Only `scope` and
expanded parameters create more work items.

### Inputs from another analysis

`from: analysis:NAME` reads the output slots of another subject- or run-scope
analysis in the same study, at the same commit. `select` filters the upstream
filenames, and `workflow`, `config`, and `path` are not allowed. Planning the
reading analysis also plans the upstream one. A participant whose upstream work
item is ready or waiting is waiting too, and rerunning the upstream analysis
makes the reading analysis stale. An upstream work item that is an error or not
applicable supplies no files. Inputs that form a cycle are an error.

With `nro study run`, upstream freshness is judged the way `nro study check`
judges it, so run the upstream analysis with `nro study run` too. A scheduled
run plans the upstream analysis in the same request; see
[scheduled runs](#scheduled-runs).

### Project scope

A project-scope analysis has one work item per member project. It reads the
inputs of the participants that its `participants` block selects:

```yaml
version: 1
scope: project
participants:
  match: "re:.*[02468]"   # default "*"
  exclude: ["044"]
  pending: wait           # or exclude
inputs:
  summaries:
    from: analysis:subject_summary
    select: {suffix: summary, extension: .tsv}
outputs:
  table: "project-{project}_desc-group_summary.tsv"
```

`match` takes the same patterns as command-line [selectors](index.md#shared-selectors).
Planning expands it against the project's BIDS participants and resolves each
input once per selected participant, with the usual rules. A participant is
included when every input resolves. A participant is excluded when an input
does not apply to it, including a single-file input that matches nothing. With
`pending: wait`, the default, any participant whose upstream work is not fresh
holds the whole work item. With `pending: exclude`, that participant is left
out instead. Any other resolution problem is an error.

The included and excluded participants are listed in full in the work item's
signature and provenance. The result therefore becomes stale when a matching
participant is added or removed, or when a participant's inputs appear or
disappear. The code receives the lists as `params["participants"]` and
`params["excluded_participants"]`, so those parameter names are reserved. Each
input arrives as one `InputSet` holding every included participant's files in
participant order. Each artifact has a `participant` attribute.

Project-scope output patterns may use `{project}` and expanded parameters, but
not `{participant}` or run entities. Outputs go to
`STUDIES/RESULTS/STUDY/analyses/ANALYSIS/project-PROJECT/`. `--participant`
is rejected for a project-scope analysis because it would change the selected
set.

## Analysis code

```python
import numpy as np
from nro.analysis import Inputs, Outputs


def run(inputs: Inputs, params: dict, outputs: Outputs) -> None:
    maps = {
        f"ses-{artifact.entities['ses']}_run-{artifact.entities['run']}":
            artifact.load().map(Contrast=params["contrast"])
        for artifact in inputs["tmaps"]
    }
    runs = sorted(maps)
    rows = []
    for i, first in enumerate(runs):
        for second in runs[i + 1 :]:
            x, y = maps[first], maps[second]
            finite = np.isfinite(x) & np.isfinite(y)
            r = np.corrcoef(x[finite], y[finite])[0, 1]
            rows.append({"run_a": first, "run_b": second, "z": np.arctanh(r)})
    outputs["similarity"].write_tsv(rows)
```

`inputs` maps each input name to an `Artifact`, an `InputSet` (for `many`), or
`None`. An `Artifact` has `path`, `entities`, `module`, `signature`, and
`load()`. An `InputSet` is a sequence of artifacts with `paths` and
`group_by(entity)`.

`load()` chooses a reader from the extension:

| File | Result |
| --- | --- |
| `.dscalar.nii` | `CiftiMaps`, with `map(**fields)`, `indices(**fields)`, `names`, `records`, `array`, and `brain_axis`. Fields come from the indexed sidecar, such as `Contrast`. |
| `.nii`, `.nii.gz`, other CIFTI | nibabel image |
| `.func.gii`, `.shape.gii` | numpy array |
| `.tsv` | pandas `DataFrame` |
| `.json` | `dict` |
| `.npy` | memory-mapped numpy array |

`outputs` maps each slot to an `OutputHandle` with a fixed `path` and the
writers `write_tsv(rows)`, `write_json(value)`, `write_text(text)`,
`write_nifti(data, like=...)`, and `write_cifti(maps, like=..., names=...)`.
Writers replace the file atomically. Code with its own writer can write to
`handle.path`.

In a notebook, `nro.analysis.query` resolves files the same way a work item
does and raises `ResolutionError` when the match is missing, ambiguous, too
small, or not fresh:

```python
from nro.analysis import query

tmaps = query(project="a001", participant="004", module="contrasts",
              workflow="main", select={"model": "main", "stat": "t", "node": "run"},
              many=True)
```

## Viewer pages

An analysis can ship a browser page that `nro viz` shows as its own tab, next
to the default **Surfaces** viewer. Name the page with `viz`. It must be an
HTML file in a subdirectory of the analysis, and `nro study validate` fails
when the committed file is missing. A tab appears for each analysis on its
study's `main` branch that declares a page and has at least one published
output. Each tab is a separate frame, so switching tabs keeps every view's
state.

The server reads the page and every other file in its directory from the
extracted commit, at `/analyses/STUDY/ANALYSIS/PATH`. The page reads its data
through these endpoints:

| Endpoint | Returns |
| --- | --- |
| `/api/analysis/outputs?analysis=STUDY/ANALYSIS` | Each published output: `project`, `participant` (`null` for project scope), `slot`, `file`, `size`, `commit`, `published`, and `parameters`. |
| `/api/analysis/file?analysis=...&project=...&file=...[&participant=...]` | One listed output file. |
| `/api/template-mesh?space=fsaverage6&hemi=L&kind=midthickness` | Template geometry as float32 coordinates followed by uint32 triangles, with `X-Vertices` and `X-Triangles` headers. `space` is `fsaverage6` or `fsaverage`, and `kind` is `midthickness`, `pial`, `white`, or `sphere`. |

Only files that carry a provenance sidecar are listed or served, so a page
cannot read other files.

`/static/host.js` wraps these endpoints. It exports `outputs()`,
`outputFile(entry, {as})`, `parseNpy(buffer)`, `templateMesh(space, hemi,
kind)`, `inflate(mesh)` for a smoothed display surface, the default viewer's
colormaps through `color(value, scale)`, and `SurfaceView`, which draws both
hemispheres side by side. A page needs its own import map for three.js:

```html
<!doctype html>
<meta charset="utf-8">
<link rel="stylesheet" href="/static/viz.css">
<script type="importmap">
  {"imports": {"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<main><canvas id="canvas"></canvas></main>
<script type="module">
  import { outputs, outputFile, SurfaceView, templateMesh } from "/static/host.js";
  const view = new SurfaceView(document.getElementById("canvas"));
  view.setMeshes({
    L: await templateMesh("fsaverage6", "L"),
    R: await templateMesh("fsaverage6", "R"),
  });
  const [first] = await outputs();
  const summary = await outputFile(first, { as: "json" });
</script>
```

Viewer pages show results from the site `study_results` root. Results that
scheduled runs write for a development branch are not shown yet.

## Commands

| Command | Effect |
| --- | --- |
| `nro study init STUDY [--project P ...]` | Create the repository and results directory, and commit. |
| `nro study add STUDY/ANALYSIS [--scope run\|project]` | Write an analysis skeleton without committing it. |
| `nro study list [STUDY]` | List studies and analyses on `main`, with scope, last commit, and validity. |
| `nro study validate [STUDY[/ANALYSIS]] [--worktree]` | Check `study.yml`, manifests, and that `analysis.py` defines the entry function. Code is parsed, not run. |
| `nro study check STUDY/ANALYSIS [KEY=VALUE ...]` | Plan and print each work item's status, inputs, and output paths. `--worktree` plans from uncommitted files; `--json` prints JSON. |
| `nro study try STUDY/ANALYSIS [KEY=VALUE ...]` | Run exactly one work item from the working tree into a scratch directory. Nothing is published. `--scratch DIR` chooses the directory; the default is `try/ANALYSIS` below the study's work root. |
| `nro study run STUDY/ANALYSIS [KEY=VALUE ...]` | Run every ready work item and publish its outputs. `--dry-run` only reports; `--force` reruns fresh items. |

`check`, `try`, and `run` accept `--project` and `--participant`, which can be
repeated and accept the glob and `re:` patterns described under
[shared selectors](index.md#shared-selectors). `KEY=VALUE` selects an expanded parameter by value or label, or a run
entity such as `task=lang`.

Work-item states are `ready`, `fresh`, `waiting`, `not_applicable`, and
`error`. `run` exits nonzero when any item failed, waits, or has an error.

## Freshness and outputs

A work item is fresh when each of its outputs has a provenance sidecar,
`NAME.provenance.json`, whose `Signature` matches the planned one. The
signature covers the Git tree hashes of the analysis directory and `common/`,
the manifest, the container, the parameters, the work item's entities, and the
signatures of its inputs. A commit that changes only another analysis leaves
results fresh. Rerunning the upstream module or adding a run makes them stale.

Before running, nro extracts the planned commit with `git archive` under
`WORK/studies/STUDY/code/COMMIT`. The function writes into a staging directory
under `WORK/studies/STUDY/runs/`, which also holds `spec.json` and `log.txt`. The
work item fails if the function raises or leaves a slot empty. Otherwise each
slot is copied to

```text
STUDIES/RESULTS/STUDY/analyses/ANALYSIS/project-PROJECT/sub-PARTICIPANT/FILENAME
```

with its provenance sidecar. Files written outside the slots stay in the
staging directory.

## Scheduled runs

A scheduled analysis is one registry lineage, named `STUDY/ANALYSIS`, that no
workflow selects. Each work item depends on the upstream work items of every
input's workflow whose recorded entities agree with the selector. Input files
are chosen when the work item runs, from those work items' outputs, so a
`many` input with fewer than `min` files fails the attempt instead of being
reported as not applicable. `config:` inputs cannot be scheduled yet; use
`workflow:`.

An `analysis:` input makes the request plan the upstream analysis too, and each
work item depends on the upstream analysis's work items for its participant.
When it runs, it reads the files that their completion manifests list.

A project-scope analysis has one work item per project and parameter
combination. Its participant column is empty and its scope is `project`.
Planning expands the `participants` selector against BIDS. A selected
participant is included when every required input has planned upstream work,
and is excluded otherwise. Both lists are part of the work item's contract, so
adding or removing a participant makes the item stale. The work item depends on
the upstream work of every included participant. One failed upstream item
therefore holds it, as for any other dependency. Purging a participant's
upstream work invalidates the project item too.

`--participant` never selects a project-scope work item, in `nro run`,
`status`, `purge`, `gc`, `freshen`, or `log`. `nro status` lists such items
after the project's participants, with `(project)` in the participant column,
and its JSON output gives `scope`.

A development branch writes study results below
`DEVELOPMENT/BRANCH/STUDIES/RESULTS` instead of the shared results root. Each
work item also writes a completion manifest,
`STEM_analysis-NAME_PARAMETER-LABEL_manifest.json`, which lists its outputs and
provenance sidecars. Here `STEM` is `sub-PARTICIPANT`, the run's stem, or
`project-PROJECT` for a project-scope item.

Scheduled analysis work items appear in `nro status`; `--study STUDY` or
`--study STUDY/ANALYSIS` limits the report to them. Their ownership receipts are
in `.nro/` inside each `project-PROJECT` results directory. `nro purge` removes
an analysis's outputs, sidecars, completion manifest, and receipt. Purging an
upstream module leaves study results in place and marks the analysis stale.
`nro gc` also removes files in `STUDY/analyses/NAME/project-PROJECT` that no
registered analysis work item claims, such as the results of an analysis deleted
from its study repository. It keeps the receipts in `.nro/`.

`nro run --repair` restores scheduled analysis work items from these receipts,
together with the upstream work items they depend on, for each work item whose
completion manifest remains. If the extracted study code under
`WORK/studies/STUDY/code/` has been removed, a rerun extracts the recorded
commit again.
