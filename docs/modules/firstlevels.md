# First-level models

`firstlevels` estimates task effects within a participant. It fits each run
independently, then forms session and subject summaries from the available runs. It produces effect, variance, t, and degrees-of-freedom
maps in either volume or surface space. It never combines participants or fits
population-level models.

The module branches from `func`; it does not use `clean`. Its nuisance model
excludes global signal. Where preprocessing applies ICA-AROMA, firstlevels uses
the corresponding `desc-preprocNoAROMA` outputs. Otherwise it uses `desc-preproc`.
This choice comes from the preprocessing workflow, not from whichever files
happen to exist. AROMA's fitted temporal operator is not propagated by this GLM,
so AROMA-denoised inputs are not supported.

## Requesting fits

Requests without a model selector use model set `main`. The following example
assumes the site has registered a `mytask/main` model in that set. New
definitions stores start without models:

```bash
nro models list
nro models show mytask/main
nro run -P example -p 01 -m firstlevels -s fsnative -S 2
```

A bare `nro run` requests `dynconn`, `networks`, and `firstlevels`, including GLMs for
matching tasks in model set `main`. Use `-m` to restrict the requested branches.
Multiple spaces and smoothing values create independent instances.
One instance covers a participant, model variant, space, and smoothing value.
Its run fits are separate steps within that instance, not separate worker jobs.

Models live in `DEFINITIONS/models/TASK/VARIANT.yml` in the selected external store. They define
event predictors and contrasts; the firstlevels configuration defines denoising,
estimation, and run aggregation. See [task models](../task-models.md) for the YAML format,
transformations, and model sets, and [model commands](../commands/models.md)
for registration.

```bash
nro run -m firstlevels --task langlocSN --model main
nro run -m firstlevels --model-set development
```

Task and model selection are independent of workflow configuration. A model
variant can be fitted under different denoising configurations by selecting
different workflows. Each instance includes all runs of its task for that
participant. Partial run selectors other than task are rejected because they
would redefine the same subject-level artifact.

## Processing and branches

1. Match the registered task. Resolve events
   using BIDS inheritance. A more specific events table replaces a general
   table; rows are not merged. `Run` grouping must identify each BOLD run.
2. At positive FWHM, smooth each volume with SciPy's Gaussian filter, using
   voxel dimensions to convert mm to standard deviations. The grid axes must
   be orthogonal; boundaries are zero-padded. For surfaces, use Workbench
   `-metric-smoothing -fwhm` with midthickness geometry. Native geometry comes
   directly from `anat`; fsaverage geometry comes from TemplateFlow. Zero FWHM
   skips smoothing. Anatomy is an explicit dependency and is not copied.
3. Compile the task YAML and module configuration into Stats Models. Select
   continuous nuisance columns from confounds using `confounds_regex`. Event convolution uses Nilearn's SPM or Glover canonical
   HRF. Exclude the union of frames marked by `temporal_mask_regex` columns;
   outlier columns do not enter the continuous design.
4. Select nuisance PCs within the residual-rank budget of the retained frames,
   then fit task and nuisance jointly. Scientific contrasts remain in the
   original task parameterization. No temporal filtering, response
   standardization, or percent signal conversion is applied.
5. Fit OLS or grouped AR(1) GLS. AR covariance respects the original gaps
   between retained frames. Save compact run coefficients and covariance
   information, requested maps, a labeled design SVG, a numerical design, and
   design/rank metadata.
6. Pool available effects across original runs, then evaluate the requested
   linear combinations. Create session summaries where sessions exist and
   subject summaries directly from runs, not from session summaries. Track
   each contribution back to its original run to retain covariance between
   effects sharing data. Never average t-values.
7. Publish node inventories and the instance completion manifest. Missing
   conditions and non-estimable contrasts have omission records, not zero
   effect maps. If censoring makes the temporal model unidentifiable or leaves
   no residual dimensions, omit the entire run's estimates with a reason.
   These decisions do not alter the preconstructed runner graph. Invalid inputs
   or unsupported model instructions still raise errors.

If the post-task rank is below the absolute nuisance-protection floor, nuisance
regression is skipped. This alone does not exclude the run. Lower-rank fits
retain their actual residual degrees of freedom. Locations with zero variance
have NaN t/DOF values and must be excluded from inference.

See [estimation and inference](../methods/firstlevels.md) for equations,
assumptions, weighting, and the limitations of conditional AR inference.

## Artifacts

```text
derivatives/firstlevels/FIRSTLEVEL_CONFIG_ID/
  space-SPACE_smoothing-Nmm/TASK/
    node-run/sub-ID/
    node-session/sub-ID/
    node-subject/sub-ID/
```

The configuration directory uses nro's lineage rules: it is normally the
firstlevels configuration ID, with a distinct label when upstream choices would
otherwise collide. Filenames include model variant, task, space, smoothing,
Stats Models node name, and applicable run/session entities. Statistical maps
also include `contrast-NAME_stat-{effect,variance,t,dof}`. Volumes use
`_statmap.nii.gz`; each surface hemisphere uses `_statmap.shape.gii`. Internal primitive-effect records support aggregation without publishing
additional maps for effects that the user did not request.

Each map's JSON sidecar identifies the model, contrast, estimation choices, and
the linear recipe on source runs. Each fitted run saves a coefficient array, residual
variance map, noise-group map, small group covariance matrices, and AR grid as
NumPy arrays. These permit later covariance-aware summaries without loading
time courses or storing a map for every coefficient pair. They are public
artifacts and must be retained with the contrast maps.

`_design.tsv` contains the fitted columns on retained acquisition frames, with
zero placeholders on excluded rows; the SVG marks those rows. The
`_design.json` records the mask, column names, coefficient mapping,
absence of temporal filtering, rank budget, and residual DOF. The array used by GLS whitening
can differ by noise group; the shared design and group covariance records
describe that fit without producing a separate plot per group.

Each fitted run also saves `_statsmodel.json` with its resolved event HRFs,
selected nuisance columns, exact outlier columns, and estimator rules. The
instance saves the scientific task YAML, resolved configuration, and compiled
model template. Model-set membership is excluded from these scientific records.
The numerical design and its metadata describe the realized PCA fit.

Node manifests list maps, compact fits, design files, contributing records, and
omissions. The fixed `_desc-firstlevels_manifest.json` under `node-run/sub-ID`
collects completion evidence for the instance. Deleting a declared output
invalidates completion. Resumption checks the model/configuration and selected
run set as well as file freshness. Artifacts from different variants or targets
are isolated during purge. These names are BIDS-like, not a claim of validator
compliance.

## Configuration

| Key | Effect |
| --- | --- |
| `confounds_regex` | Select continuous nuisance candidates from `confounds.tsv`; selected global signal columns are rejected. |
| `temporal_mask_regex` | One-hot columns whose union excludes frames from estimation. |
| `nuisance_variance_explained` | Desired fraction of standardized nuisance variance, subject to the rank cap. |
| `minimum_temporal_rank` | Absolute number of post-task dimensions protected from nuisance removal. |
| `minimum_temporal_rank_fraction` | Protected fraction of post-task dimensions. |
| `aggregation_weighting` | Pool runs by inverse estimated marginal variance (`precision`) or arithmetic mean (`equal`). |
| `noise_model` | `ar1` or `ols`. |
| `ar_grid` | Finite AR(1) correlation candidates strictly between -1 and 1. Ignored for OLS. |
| `spatial_block_size` | Locations evaluated together during fitting and summary calculation. One run's input array is loaded at a time. |
| `wb_command` | Installed Workbench executable used for surface smoothing. |

```{literalinclude} ../../nro/configuration/starters/configs/firstlevels/main_firstlevels.yml
:language: yaml
```

## Stats Models execution

The [task-model compiler](../task-models.md) generates the Stats Models document
used for fitting. The estimator implements a bounded subset of
[BIDS Stats Models 1.0.0](https://bids-standard.github.io/stats-models/), without
a FitLins dependency. Unsupported task controls fail validation.

Adaptive nuisance PCA, exact outlier handling, rank safeguards, noise estimation,
and aggregation weighting are recorded under `Model.Software.nro`. These
software-specific rules must be honored to reproduce the estimator; schema
validity alone does not imply identical results from another engine.

Dataset nodes, cross-subject aggregation, summary-level GLMs, formulas, F tests,
random effects, HRF derivatives, and temporal filtering are not supported.
Effect units must be comparable across runs; no automatic acquisition-gain
normalization is applied.

Implementation: [compiler](../autoapi/nro/modules/firstlevels/compiler/index.rst),
[design](../autoapi/nro/modules/firstlevels/design/index.rst),
[statistics](../autoapi/nro/modules/firstlevels/statistics/index.rst),
[module](../autoapi/nro/modules/firstlevels/module/index.rst).
