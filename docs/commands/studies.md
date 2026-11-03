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

`nro study run` executes analyses directly from the command line, one work item
after another. The shared scheduler does not run them yet, so `nro status`,
`nro purge`, and `nro gc` do not report study results. In a development
installation, `query`, `check`, `try`, and `run` read upstream state from the
scheduler's last published status snapshot of `main`, which can trail the
registry by one scheduler update.

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
| `scope` | `subject` (one work item per participant) or `run` (one per raw BOLD run). |
| `inputs` | Named inputs, described below. |
| `outputs` | Slot name to filename pattern. Every work item has exactly these slots. |
| `parameters` | Values passed to the function. |
| `resources` | `class` (`small`, `medium`, or `large`) and optional `memory_gb`. Recorded only for now. |

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
| `from` | An nro module, such as `contrasts`, or `raw` for source BIDS files. |
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

## Commands

| Command | Effect |
| --- | --- |
| `nro study init STUDY [--project P ...]` | Create the repository and results directory, and commit. |
| `nro study add STUDY/ANALYSIS [--scope run]` | Write an analysis skeleton without committing it. |
| `nro study list [STUDY]` | List studies and analyses on `main`, with scope, last commit, and validity. |
| `nro study validate [STUDY[/ANALYSIS]] [--worktree]` | Check `study.yml`, manifests, and that `analysis.py` defines the entry function. Code is parsed, not run. |
| `nro study check STUDY/ANALYSIS [KEY=VALUE ...]` | Plan and print each work item's status, inputs, and output paths. `--worktree` plans from uncommitted files; `--json` prints JSON. |
| `nro study try STUDY/ANALYSIS [KEY=VALUE ...]` | Run exactly one work item from the working tree into a scratch directory. Nothing is published. |
| `nro study run STUDY/ANALYSIS [KEY=VALUE ...]` | Run every ready work item and publish its outputs. `--dry-run` only reports; `--force` reruns fresh items. |

`check`, `try`, and `run` accept `--project` and `--participant`, which can be
repeated. `KEY=VALUE` selects an expanded parameter by value or label, or a run
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
