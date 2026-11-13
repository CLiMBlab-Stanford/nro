# First-level models

A first-level model tells the [contrasts](modules/contrasts.md) module which
event predictors to fit and which contrasts to report. This guide shows how to
write one, starting from a minimal model and adding options as they are needed.

## A minimal model

Store each model as `DEFINITIONS/models/ANALYSIS/VARIANT.yml` in the selected
[definitions store](definitions.md). Its identifier is `ANALYSIS/VARIANT`. For
an ordinary single-task model, the analysis label is also the source BIDS task,
so the YAML does not repeat it.

```{literalinclude} examples/task-model.yml
:language: yaml
```

`conditions` names a categorical column in `events.tsv`. nro includes every
category it observes in that column, even categories that no contrast uses. In
this short form, a contrast key such as `S` resolves to `trial_type.S`.
Contrast weights are finite numbers or fraction strings such as `"1/2"`.
Output contrast names may contain only letters, digits, dots, underscores, and
hyphens.

`nro def edit model ANALYSIS [VARIANT]` opens an existing model, or drafts a
missing one from matching BIDS events. See
[definition authoring](commands/authoring.md) for discovery selectors, local
drafts, and save behavior.

### HRF and event timing

`hrf` defaults to `glover`. The other accepted values are `spm` and `null`.
The selected HRF applies to event-derived predictors. Confounds and the
intercept are never convolved.

Timing comes from the `onset` and `duration` columns, adjusted for the
functional input's `StartTime`. These timing columns are not predictors
themselves.

### What the model does not control

The contrasts configuration supplies nuisance regression, temporal outlier
handling, rank protection, run aggregation, and the noise model. A first-level
model cannot override those settings or add confounds. This separation lets the
same scientific model run under different contrasts configurations.

For example, the configuration's `aggregation_weighting` setting controls how
the available original-run estimates are pooled. The default, `precision`,
weights by inverse estimated marginal variance. `equal` takes an arithmetic
mean. See [first-level estimation and inference](methods/contrasts.md) for the
covariance and DOF calculations.

At execution time, contrasts reads event tables from the BIDS project. Task
repositories and shared event templates can help with BIDSification and model
authoring. Once the corresponding `events.tsv` files are published, they are no
longer runtime dependencies.

## Models that pool several tasks

A model that pools several source tasks lists them in `tasks`:

```yaml
tasks: [ActPassBlock, ActPassEvent]
conditions: trial_type
contrasts:
  plausibility:
    trial_type.activeImplausible: 0.25
    trial_type.passiveImplausible: 0.25
    trial_type.activePlausible: -0.25
    trial_type.passivePlausible: -0.25
```

Each run is still fitted on its own. Session and subject summaries then pool
the available primitive effects across tasks, so a contrast can become
estimable even when no single run contains all of its conditions. In the
compiled Stats Models document, `Input.task` lists the declared source tasks.

The derivative layout predates multi-task models and encodes the analysis label
as `task-ANALYSIS`. This is an on-disk compatibility convention, and the
analysis label need not be a source BIDS task. The public model snapshot and
the compiled Stats Models document record the actual source tasks.

## Models from stimulus annotations

A naturalistic stimulus, such as a story, can be modeled at several temporal
grains: sentences, words, or phonemes. Instead of repeating that timing in
every run's `events.tsv`, store it once per stimulus in the project's
`stimuli/` directory. The run events then record only when each stimulus was
presented, and a model chooses the grain with `annotation`.

The layout follows the draft BIDS stimuli extension (BEP044), which is not part
of a released BIDS version:

```text
stimuli/
  stim-story1_annot-words_events.tsv
  stim-story1_annot-phonemes_events.tsv
sub-01/func/sub-01_task-story_run-1_events.tsv
```

In the run events, a row with a `stim_id` value such as `stim-story1` is a
presentation row. Its `onset` is the run time at which the stimulus started.
Each annotation file has `onset` and `duration` columns, with onsets in seconds
from the start of the stimulus, and any other columns the model needs.

```yaml
annotation: words
conditions: word_class
contrasts:
  content: {content: 1}
  CvF: {content: 0.5, function: -0.5}
```

`annotation` names the `annot-<label>` to use, in letters and digits. Before
fitting, `contrasts` replaces each presentation row with the rows of
`stimuli/<stim_id>_annot-<annotation>_events.tsv`. It adds the presentation
onset to each annotation onset and copies the presentation row's other columns,
such as `trial_type`, onto every expanded row. Rows without a `stim_id`, such as
button presses, are kept. The `StartTime` adjustment applies after expansion.

Planning checks every selected run before any work is scheduled. `nro run`
exits with an error that names the project, participant, model, and missing
file, lists the annotations that are present for the stimulus, and suggests
`nro stimuli sync PROJECT`. Planning or fitting stops with an error when:

- the run events have no `stim_id` values;
- a presented stimulus has no file for the named annotation;
- an event column other than `onset` and `duration` appears in both the run
  events and the annotation file;
- an annotation event ends after the end of a presentation with a positive
  `duration`;
- the stimulus has `part-<label>` annotation files. Parts are not supported yet.

Models without `annotation` use the run events unchanged. The annotation label
is part of the scientific model, so variants that differ only in `annotation`
are separate models. Annotation files are inputs of the fit, and editing one
makes the fits that use it stale.

## Predictors and transformations

Use `predictors` instead of `conditions` when the design needs numeric
variables, interactions, or only some category indicators. Predictors and
transformation inputs can refer only to event variables, never to confounds.
Wildcards match event variable names. Transformations cannot run arbitrary
Python code.

```yaml
model_set: development
predictors: [trial_type.*, sentence_difficulty]
hrf: spm
transformations:
  - Name: Factor
    Input: trial_type
  - Name: Demean
    Input: difficulty
    Groupby: [trial_type]
    Output: centered_difficulty
  - Name: Product
    Input: [trial_type.S, centered_difficulty]
    Output: sentence_difficulty
contrasts:
  sentences: {trial_type.S: 1}
  difficulty: {sentence_difficulty: 1}
```

Instructions run in order on event amplitudes, before HRF convolution and
sampling. A `Product` therefore builds event-level modulation and does not
multiply convolved predictors. Grouping for centering must be stated
explicitly. nro does not orthogonalize modulators automatically.

| Instruction | Supported options and behavior |
| --- | --- |
| `Factor` | Expand categories as `COLUMN.VALUE`. `Constraint: drop_one` requires an explicit `RefLevel`; default `none` keeps all indicators. |
| `Demean` | Subtract the event mean, within each `Groupby` cell if supplied. Optional `Output` creates a new variable. |
| `Scale` | `Demean` and `Rescale` default to true. Rescaling uses sample SD. Constant groups raise an error. Optional `Groupby` and `Output`. |
| `Sum` | Sum aligned event variables into required `Output`; optional `Weights` must match inputs. |
| `Product` | Multiply aligned event variables into required `Output`. |
| `Select` | Keep only the named variables. Select indicators after `Factor` to restrict modeled categories. |
| `Convolve` | Apply explicit `Model: spm` or `glover`. No amplitude transformations may follow convolution. |

`Input` accepts one variable or a list. `Output` accepts one name. When
`Demean` or `Scale` has an `Output`, its `Input` must resolve to one variable.
`Groupby` is the transformation option. Stats Models node grouping uses the
separate `GroupBy` field. Unsupported options are errors and are never silently
ignored.

To change the HRF for individual event predictors, use `hrf_overrides`:

```yaml
hrf_overrides:
  sentence_difficulty: glover
  already_modeled_response: null
```

Each override must match an event predictor. Overlapping overrides, repeated
convolution, and an override that conflicts with an explicit `Convolve` are
errors.

A category indicator for a category that never occurs contributes no estimable
effect. Within event algebra, an explicitly named category that is absent has
zero amplitude. A selected predictor with nonfinite values is an error.

## Advanced Stats Models blocks

The `statsmodels` key accepts `Transformations` and `Contrasts` blocks in BIDS
Stats Models form. Each block replaces the corresponding shorthand section
instead of patching the generated model. Advanced `Transformations` also
replaces the automatic `conditions` expansion, so use an explicit `predictors`
list with it.

```yaml
predictors: [trial_type.*]
hrf: spm
statsmodels:
  Transformations:
    Transformer: pybids-transforms-v1
    Instructions:
      - {Name: Factor, Input: trial_type}
  Contrasts:
    - Name: sentences
      ConditionList: [trial_type.S]
      Weights: [1]
      Test: t
```

Both forms support the same transformation semantics. Advanced contrasts must
be linear t contrasts. Raw `Nodes`, `Edges`, `Model`, `Software`, and `Options`
blocks are not accepted. A Stats Models field becomes supported only when the
estimator implements it and tests cover it; a schema-valid JSON document is not
enough.

## Selecting models and model sets

Registering a model creates no work by itself. Bare `nro run` requests
`dynconn`, `networks`, and `contrasts`, and fits the models in set `main` for
matching tasks. Use `-m contrasts` to request only the modeling module and its
upstream dependencies.

`model_set` assigns a model to one named set or a list of sets. A model without
`model_set` belongs to no set. To keep development models out of default
requests, put them in a separate set or leave them unassigned.

```yaml
model_set: [development, language-experiment]
```

```bash
nro run -m contrasts                         # set main
nro run -m contrasts --analysis language     # set main within this analysis
nro run -m contrasts --task langlocSN        # models that consume this source task
nro run -m contrasts --model alternative    # explicit variants, any set
nro run -m contrasts --model language/main  # one qualified model
nro run -m contrasts --model-set development language-experiment
```

Values given to one selector are alternatives, and different selectors
intersect. `--analysis` matches the first component of `ANALYSIS/VARIANT`.
`--task` matches a model's declared source BIDS tasks and is the same source-task
entity that `--run task=...` constrains. Both selectors choose whole models, so
selecting one task does not drop the model's other declared tasks.

`--model main` matches that variant in every analysis. Use a qualified
`--model ANALYSIS/VARIANT` to select exactly one model. An explicit `--model`
bypasses the default set restriction, while an explicit `--model-set` still
intersects with it.

Status, logs, stop, and purge accept the same model and set filters. On those
commands, omitting the selectors means all registered work, including
development models. Registry discovery considers existing artifacts from every
model set.

Model sets are execution metadata. Changing membership does not alter work item
identity, scientific fingerprints, or completion checks, and work that is
already registered is unaffected by later membership changes.

## Levels, missing effects, and saved definitions

nro always plans run and subject outputs, and adds session outputs when the
dataset has sessions. Session summaries group original run estimates, and
subject summaries also pool original runs directly. Saving session results
therefore does not give a one-run session the same weight as a four-run
session. A summary with one contributing run keeps that run's uncertainty.

A missing condition produces omission metadata instead of a zero-valued map. A
contrast can become estimable after conditions from different runs are pooled,
and shared-run covariance is retained. Numerical estimability changes which
contrasts are emitted. It does not change the execution graph.

The compiler produces the Stats Models document that the estimator uses. At run
time it resolves the selected nuisance and outlier columns and the event HRFs.
The saved run document records those choices. Numerical design files record
the fitted PCs, retained rows, parameter mapping, and rank budget. The work
item also saves its scientific model YAML, the resolved contrasts
configuration, and the compiled template. Execution-only set membership is left
out.

## Model edits and freshness

nro decides whether existing results are still current by comparing canonical
compiled task definitions. YAML text and file timestamps play no part. Edits
that only change authoring style keep the same contract:

- short condition names versus their qualified forms;
- shorthand blocks versus equivalent advanced blocks;
- omitted defaults versus their explicit values;
- equivalent numeric contrast weights;
- mapping key order, comments, `description`, and `model_set`.

Predictor lists, transformation sequences, and explicit advanced contrast lists
stay ordered, so reordering them can change the contract. So can changes to
HRFs, contrast weights, or aggregation rules. nro does not try to prove
arbitrary algebraic equivalence or equivalence that depends on event values.

A change invalidates only the artifacts of the model it touches. Changing
predictors, contrasts, transformations, or HRFs affects that one model and not
every model fitted with the same contrasts configuration. Changing a scientific
setting in the contrasts configuration affects every model fitted with it.

Changing only contrasts or aggregation rules republishes the affected maps and
summaries without refitting the time series. Each run manifest keeps the
compact coefficient and covariance fit for every design predictor. nro reuses
that fit only while its input data are fresh and its predictors, HRF,
transformations, nuisance model, response scaling, and estimator settings are
unchanged. Changes to predictors, transformations, HRF, denoising, or the
estimator require a new GLM fit. Explicit overwrite requests also refit.

The comparison compiles metadata only. It does not read event tables or images,
build design matrices, or fit a model. A bounded in-process cache reuses the
compilation for identical model contents. Edited contents get a new cache
entry, and file timestamps are not cache keys.

Planning, status assessment, public-output adoption, and execution all use this
normalization. Recorded contracts are normalized in the same way as current
ones, so a change in authoring style alone does not invalidate an existing
derivative. Resumed steps also compare the compiled model and configuration,
which keeps column and internal contrast identifiers stable. Summary completion
checks retain the selected run set. Input, dependency, and output checks still
apply.

## Compatibility with other Stats Models engines

Adaptive denoising and covariance-aware estimation are described under
[`Model.Software.nro`](https://bids-standard.github.io/stats-models/_autosummary/bsmschema.models.Model.html).
Another Stats Models engine must implement those rules to reproduce the fit.
Transformation instructions follow the supported subset of the
[named transformer interface](https://bids-standard.github.io/stats-models/_autosummary/bsmschema.models.Transformations.html).
The generated format does not claim unrestricted PyBIDS or FitLins
compatibility.
