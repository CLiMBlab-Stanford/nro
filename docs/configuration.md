# Workflows and configuration

This page is for users who want to change how nro processes data. It explains
how workflows select module configurations, how nro validates and compiles
them, and which settings affect whether results are reused.

## Workflows and module configurations

A workflow selects one configuration ID for each configuration class. The
default workflow is `main`. This is the packaged `main` workflow:

```{literalinclude} ../nro/definitions/starters/workflows/main_workflow.yml
:language: yaml
```

nro ships a `main` configuration for each configuration class, and these hold
the scientific defaults. The external [definitions store](definitions.md) may
hold small site-specific overrides of those defaults and named alternatives.
A named configuration is merged over its class's `main` configuration. Module
pages show the packaged configurations and explain which steps use each key.
A site's active values can differ when it defines an override.

Some configurations select a different upstream source. The `main`,
`ica-correlation`, and `clustering` network configurations read from dynconn.
The `oslom` configuration changes the estimator and selects microparcellation,
which OSLOM requires. The named `dynconn-networks` workflow spells out the default source
explicitly. Whichever source a configuration selects becomes a fixed
dependency of the workflow.

To create or change a definition, run `nro def edit MODULE ID` or
`nro def edit workflow ID`. The editor works on a draft and validates it before
publishing. [Definition authoring](commands/authoring.md) covers local drafts
and the safeguards on the shared store.

Configurations are one kind of definition. Everything authored through
`nro def` is implemented by `nro.definitions`: module configurations,
workflows, models, markup, hardware descriptions, and bidsification policy. A
resolved configuration is the validated runtime form compiled from those
definitions. It is not a separate kind of site metadata.

The definitions store holds other material too. Its
[standard event-file catalog](event-files.md) supplies task-name suggestions
during bidsification. It may contain [source markup](definitions.md#source-markup)
for manual anatomical selection and for excluding known-bad sources. A site can
also define [gradient-unwarping hardware](definitions.md#gradient-unwarping-hardware).

Protected operational settings, such as shared paths and installation state,
belong to `nro.site`. They are set during site setup, not through `nro def`.

## Configuration classes

| Configuration class | Module | Derivative directory | Parameter reference |
| --- | --- | --- | --- |
| `anat` | anat | `anat` | [Anat](modules/anat.md) |
| `func` | func | `func` | [Func](modules/func.md) |
| `clean` | clean | `clean` | [Clean](modules/clean.md) |
| `dynconn` | dynconn | `dynconn` | [Dynamic connectivity](modules/dynconn.md) |
| `microparcellation` | microparcellation | `microparcellation` | [Microparcellation](modules/microparcellation.md) |
| `networks` | networks | `networks` | [Networks](modules/networks.md) |
| `contrasts` | contrasts | `contrasts` | [First-level models](modules/contrasts.md) |

[First-level models](task-models.md) are separate from contrasts
configurations. Models define event predictors, contrasts, and HRFs. The
contrasts configuration supplies denoising, estimation, and run-aggregation
choices. Analysis, task, model, and model-set selectors choose which work to
run. Membership in a model set does not contribute to scientific fingerprints.

## Site resources

Configurations refer to site resources with `site:KEY` references. nro
resolves these before it fingerprints a configuration. The protected
`site/site.yml` in the definitions store owns filesystem locations,
executables, binds, and Slurm routing. The installation's TOML file only
records where that store is.

`nro site` edits the protected site document. It does not edit module
configuration YAML. Changing a resolved resource path can change freshness,
even when you believe the content at the new path is identical.

## Validation and compilation

Loading a configuration and running `nro def edit` use the same compiler. It
parses strict YAML, resolves site references, merges named overrides over
`main`, and validates the result. A workflow resolves each configuration it
selects, and a class the workflow omits selects `main`. Missing references and
unknown classes are errors. Private runtime snapshots are validated too, but
nro does not apply today's defaults to settings that were already resolved.

The [field schema](autoapi/nro/definitions/schema/index.rst) defines types,
nullable fields, allowed values, bounds, and which fields are execution-only.
Packaged `main` YAML files must supply every required field. An external
`main_CLASS.yml` is a partial override, and fields it omits keep the packaged
default. An override cannot change a field's type or make an unknown key valid.

Validation rejects:

- duplicate mapping keys at any depth;
- unknown override keys, so a misspelled setting fails instead of being
  silently added;
- malformed regular expressions, unsupported strategies, and out-of-range
  values; and
- conflicting settings such as `high_pass >= low_pass`.

Errors name the source file and the offending field. Every declared algorithm
group must contain valid values, including groups the selected strategy does
not use. Requirements tied to one strategy are checked only when that strategy
is selected. Config IDs are validated filename identifiers. Scientific code
should look configurations up through the definitions store instead of
building paths.

Workflow validation also checks the interfaces between modules. The selected
`anat` configuration defines which fsaverage target is available. `space` and
`smoothing` are request selectors. They are not module configuration fields
and do not appear in module directory names. Every module configuration that a
workflow selects must use the same `markup` ID, so the workflow has one
consistent view of source BIDS.

### When two configurations count as equivalent

The compiler normalizes values so that formatting differences do not create
new results. Numbers convert to the field's declared type: `50` and `50.0` are
equivalent for an integer count, while `50.5`, `"50"`, and `true` are
rejected. Comments, mapping order, and explicit values equal to inherited
defaults do not affect the compiled result. Ordered lists stay ordered.

The alternatives in a BIDS `input_filter` are treated as sets. Scalar and list
forms become sorted, unique strings, and `null` still requires the entity to be
absent. The compiler does not rewrite regular expressions or paths to guess
whether they are equivalent. A resolved `site:KEY` reference compiles the same
as its literal value.

### Limits of compilation

Parsing and compilation use bounded content caches within each process.
Callers receive independent copies. When file contents change, nro recompiles
them even if timestamps did not change.

Compilation reads no images or event tables, and it does not require future
derivative files to exist. Spatial validation, sampling-rate constraints, and
whether a model can be estimated from the data remain runtime checks. A valid
configuration is not evidence that the settings suit the data scientifically.

## Scientific settings and execution snapshots

Most settings are scientific: changing them changes results, so nro treats
artifacts built with the old value as stale. A few settings only control how
work runs. Changing one of these does not by itself make a completed
derivative stale.

The execution-only fields are:

- the `overwrite` setting of anat, func, clean, dynconn, contrasts,
  microparcellation, and networks;
- the `verbose` setting of anat, func, and clean, and func `io_chunk_vols`;
- contrasts `spatial_block_size`;
- microparcellation `connectivity.temporal_block_size`; and
- networks `oslom.timeout_seconds`.

Every other setting is scientific. This includes random seeds, solver
tolerances, iteration counts, clustering batch size, and software resource
paths. An explicit `overwrite` instruction still controls execution when the
module runs. Attempts that have already started keep the execution snapshot
they started with.

nro keeps two fingerprints. The full resolved snapshot, including execution
settings, is kept for provenance. Its fingerprint changes when an execution
setting changes, which creates a new workflow revision with the updated runtime
configuration. Artifact contracts use a separate scientific fingerprint that
leaves out the execution-only fields.

### Comparing recorded specifications

nro normalizes recorded specifications for comparison without rewriting the
derivatives. When a database completion record binds a full configuration
snapshot to its fingerprint, that snapshot supplies the scientific comparison.
Missing or invalid evidence never counts as proof of equivalence.
Normalization does not merge differently named configurations: configuration
IDs, workflow IDs, directory ownership, and dependency identities stay
distinct.

When the schema changes, explicit
[artifact-contract migrations](artifact-contract-migrations.md) keep older
artifacts comparable. A new field declares both its default for new work and
the value that older snapshots represent, and these values may differ. If the
historical value is unknown, the older artifact becomes stale. Keys explicitly
retired as scientifically irrelevant are ignored. Configurations written by
authors stay strict, so unknown keys are still errors. Changing a retained
value, a field type, a configuration ID, or a dependency still changes the
scientific identity.

The registry saves the resolved runtime configuration for each attempt. A work
item's contract includes the relevant configuration and its upstream
dependencies. Worker CPU and memory limits and the spelling of execution
commands are not part of that contract. Historical lineages can remain owned
and usable even when the current workflow catalog can no longer request them.

### Rules for execution settings

An execution setting may control scheduling, logging, or how work is divided.
It must not define scientific groups or change the estimator. For example,
`connectivity.temporal_block_size` only sets I/O chunk size, so it is
execution-only. `quality.split_half_block_frames` defines the split used for
single-run microparcellation QC, so it belongs in the scientific fingerprint.
Clustering `batch_size` is also scientific, because it changes the stochastic
fitting procedure.

Regression tests vary execution block sizes for temporal means, streaming
correlations and QC, first-level fits, and network sparsification. Temporal
means and streaming moment and variance reductions accumulate in float64 to
limit rounding that depends on chunk size. Saved float32 maps and large
connectome accumulators keep their existing precision. These tests check
numerical agreement. They do not promise bitwise equality across hardware,
BLAS libraries, or thread counts in third-party software.

If an artifact is rebuilt, its saved provenance can show a changed execution
setting. That difference alone must not change its scientific identity.

Timeouts behave differently depending on their source. When a scientific
command's own timeout expires, the attempt fails and publishes nothing
partial. OSLOM, for example, writes its directory completion marker only after
it runs successfully and its outputs pass validation. When a reusable worker
allocation expires, nro interrupts the attempt and returns its work to the
ready queue. CPU, memory, concurrency, and Slurm routing are orchestration
settings and stay separate from scientific contracts.

## Changing configurations during development

To try a change, edit a named configuration and add a workflow that selects
it. Review the effect on artifacts with `status --update` or a narrowly scoped
`run --no-submit`. Both commands write to the registry, so neither guarantees
that registry state stays unchanged. Keep shared workers stopped while you
change shared code.

Adding a setting requires a schema field and a value in the packaged class
`main` YAML. Add tests for its valid range, any cross-field rules, and its
effect on scientific comparison.

For Python callers, `ResolvedConfiguration.fingerprint` identifies the full
snapshot, and `scientific_fingerprint` identifies the scientific settings.
Execution receives `values`, which includes execution controls. The Python
entry points are [ConfigStore](autoapi/nro/definitions/store/index.rst) and
[runtime configuration](autoapi/nro/definitions/runtime/index.rst).
