# Workflows and configuration

Scientific defaults live in `DEFINITIONS/configs`, grouped by derivative class.
`DEFINITIONS` is the external root selected through `nro paths`; see
[definitions stores](definitions.md). A workflow selects configuration IDs;
`main` is the default workflow. Module pages show the starter configurations
and explain which steps consume their keys. A lab's active values may differ.

Use `nro create config CLASS/ID`, `nro create workflow ID`, or the corresponding
`nro edit` commands to review and validate changes before saving. See
[definition authoring](commands/authoring.md) for local drafts and shared-store
safeguards.

The same definitions store contains a [standard event-file catalog](event-files.md)
used for task-name suggestions during bidsification.

```{literalinclude} ../nro/configuration/starters/workflows/main_workflow.yml
:language: yaml
```

The `clustering` and `oslom` workflows change the networks strategy while
sharing upstream lineages. Named configurations are merged over their class's
`main` configuration. Unknown override keys are rejected rather than silently
creating misspelled settings. Config IDs are validated filename identifiers;
use the configuration store instead of assembling paths in scientific code.

## Validation and compilation

Configuration loading and `create`/`edit` use the same compiler. It parses
strict YAML, resolves site references, merges named overrides over `main`, and
validates the resulting settings. Workflows resolve each selected configuration;
an omitted class selects `main`. Missing references and unknown classes are
errors. Private runtime snapshots are also validated, without applying today's
defaults to previously resolved settings.

The [field schema](autoapi/nro/configuration/schema/index.rst) defines types,
nullable fields, allowed values, bounds, and execution-only roles. Defaults
remain exclusively in the `main` YAML files. Those files must supply every
required field and pass the same validation as overrides. Editing a default
does not redefine its type or make an unknown key valid.

Duplicate mapping keys are rejected at every depth. Errors identify the source
file and offending field. Validation catches malformed regular expressions,
unsupported strategies, out-of-range values, and conflicting settings such as
`high_pass >= low_pass`. All declared algorithm groups must contain valid
values, including groups not used by the selected strategy. Requirements tied
to a selected strategy are checked when that strategy is selected.

Equivalent numeric forms normalize to the field's declared type: `50` and
`50.0` are equivalent for an integer count, while `50.5`, `"50"`, and `true`
are rejected. Comments, mapping order, and explicit values equal to inherited
defaults do not affect the compiled result. Ordered lists remain ordered.
The alternatives within a BIDS `input_filter` are sets: scalar and list forms
normalize to sorted, unique strings; `null` still requires an absent entity.
Regexes and paths are not rewritten to guess equivalence. A resolved `site:KEY` reference
and its literal value compile identically.

Parsing and compilation use bounded, process-local content caches. Callers
receive independent copies. Changed file contents are recompiled even when
timestamps are unchanged. Compilation reads no images or event tables and
does not require future derivative files to exist. Spatial validation,
sampling-rate constraints, and data-dependent estimability remain runtime
checks; valid configuration does not establish scientific suitability.

## Scientific settings and execution snapshots

The full resolved snapshot includes execution settings and is retained for
provenance. Its fingerprint can change when an execution setting changes,
creating a new workflow revision with the updated runtime configuration.
Artifact contracts use a separate scientific fingerprint that excludes fields
explicitly marked as execution-only:

- Preprocessing thread settings, `force`, `verbose`, and `func.io_chunk_vols`.
- Clean `force` and `verbose`.
- Firstlevels `spatial_block_size`.
- Microparcellation `overwrite`, `connectivity.temporal_block_size`, and
  `connectivity.reliability_vertex_block_size`.
- Networks `overwrite` and `oslom.timeout_seconds`.

Other settings remain scientific by default. In particular, random seeds,
solver tolerances, iteration counts, clustering batch size, and software
resource paths remain part of scientific comparison. Changing an execution
setting does not by itself make a completed derivative stale. An explicit
force/overwrite instruction still controls execution when the module is run.
Already-started attempts retain their selected execution snapshot.

Recorded specifications are normalized for comparison without rewriting the
derivatives. When a completion certificate binds a full configuration snapshot
to its fingerprint, that snapshot supplies the scientific comparison. Missing
or invalid evidence is not treated as proof of equivalence. Configuration IDs,
workflow IDs, directory ownership, and dependency identities remain distinct;
normalization does not merge differently named configurations.

For Python callers, `ResolvedConfiguration.fingerprint` identifies the full
snapshot and `scientific_fingerprint` identifies the named scientific settings.
Execution receives `values`, including execution controls. Adding a setting
requires a schema field and a value in the class's `main` YAML. Add tests for
its valid range, any cross-field rules, and its effect on scientific comparison.

## Execution settings and numerical reproducibility

Execution settings must control scheduling, logging, or work partitioning
without defining scientific groups or changing the estimator. For example,
`connectivity.temporal_block_size` controls I/O chunks, while
`quality.split_half_block_frames` defines the single-run microparcellation QC
split and belongs in the scientific fingerprint. Clustering `batch_size` also
remains scientific because it changes the stochastic fitting procedure.

Regression tests vary execution block sizes for temporal means, streaming
correlations and QC, first-level fits, and network sparsification. Temporal
means and streaming moment and variance reductions accumulate in float64 to
limit chunk-dependent rounding; saved float32 maps and large connectome
accumulators retain their existing precision. These tests check numerical
agreement, not universal bitwise equality across hardware, BLAS libraries,
or thread counts in third-party software.

An execution change can appear in saved provenance if an artifact is rebuilt.
It must not alter scientific identity merely because that record differs.
Timeouts must fail the attempt rather than publish partial results: OSLOM's
directory completion marker is written only after successful execution and
output validation. CPU, memory, concurrency, and Slurm routing remain
orchestration settings, separate from scientific contracts.

## Configuration classes

| Derivative class | Modules | Parameter reference |
| --- | --- | --- |
| `preprocessing` | anat, func | [Anat](modules/anat.md), [func](modules/func.md) |
| `clean` | clean | [Clean](modules/clean.md) |
| `dynconn` | dynconn | [Dynamic connectivity](modules/dynconn.md) |
| `microparcellation` | microparcellation | [Microparcellation](modules/microparcellation.md) |
| `networks` | networks | [Networks](modules/networks.md) |
| `firstlevels` | firstlevels | [First-level models](modules/firstlevels.md) |

[Task models](task-models.md) are separate from firstlevels configurations.
They define event predictors, contrasts, HRFs, and aggregation weighting.
The firstlevels configuration supplies denoising and estimation choices.
Task/model/model-set selectors choose work; model-set membership does not
contribute to scientific fingerprints.

Site resources use `site:KEY` references, resolved before configuration
fingerprinting. Site TOML handles filesystem locations, executables, binds,
and Slurm routing. `nro paths` edits that layer; it does not edit scientific
YAML. Changing a resolved resource path may affect freshness even when a human
believes the content is identical.

The registry saves resolved runtime configuration for attempts. An instance's
contract includes the relevant configuration and its upstream dependencies.
Worker CPU/memory limits and execution command spelling are distinct from that
scientific contract. Existing historical lineages can remain owned and usable
without being requestable from the current workflow catalog.

For development, change a named configuration and add a workflow that selects
it. Review its effect on artifacts with `status --verify` or a deliberately
scoped `run --no-submit`. Neither is a guarantee that no registry state changes:
both perform writes. Keep shared workers stopped while changing shared code.

The Python entry points are [ConfigStore](autoapi/nro/configuration/store/index.rst)
and [runtime configuration](autoapi/nro/configuration/runtime/index.rst).
