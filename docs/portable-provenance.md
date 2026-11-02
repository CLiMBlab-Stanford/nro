# Portable derivative provenance

This page is for developers and maintainers who work with nro derivative
metadata or move projects between storage locations. It describes how nro
records input and output locations without host-specific paths, and how to
migrate datasets written in the older format.

## Location references

nro public derivatives never use host-specific absolute paths to describe their
inputs and outputs. JSON and YAML metadata use BIDS URIs instead:

- `bids::anat/...` identifies a file in the current `derivatives/nro` dataset.
- `bids:raw:sub-01/...` identifies a file in the containing raw BIDS project.
- `nro-site:templates:...` identifies a resource under a named site root, for
  resources that belong to neither BIDS dataset.

Each `derivatives/nro` directory is a BIDS Derivatives dataset with its own
`dataset_description.json`. Its `DatasetLinks.raw` value links the derivative
dataset to the containing raw project. Before running scientific software, nro
resolves each reference against the active raw, derivative, and site roots.

## Relation to freshness

Portable metadata describes how a derivative was produced and supports
recovery, but freshness decisions do not come from it. The scheduler database
still records direct-source inventories, upstream generations, public output
integrity, and any existing private intermediates. After registry loss, public
ownership receipts rebuild the graph between modules, and Runner ledgers
rebuild each module's internal steps. Moving a complete project therefore
changes neither its scientific contracts nor its dependency graph.

Source images remain byte-level freshness inputs. Inherited JSON metadata is
different. Work-item contracts store a canonical projection of only the fields
the relevant module uses. Changing `RepetitionTime`, for example, changes a
functional contract, while changing an unused descriptive field does not. The
public ownership receipt stores the same projection, so registry repair does not
depend on a private metadata index.

## Migrating existing datasets

`nro migrate` converts existing datasets to these references. Run it
interactively:

```bash
nro migrate -P PROJECT
```

To migrate several projects together, list their IDs after `-P`. A bare
`nro migrate` selects every project.

The command pages through the complete plan and asks for confirmation before it
writes anything. Use `--dry-run` to stop after the preview, or `-f`/`--force` to
skip confirmation. Repeating the migration is a no-op.

The command is available only from the central installation. Before the
preview, it starts or joins the central Slurm scheduler without creating
scientific demand, so the preview and the migration run on the scheduler
allocation instead of the login node. It refuses to run while attempts are
active.

### What the migration changes

The migration coordinates changes across the selected source BIDS projects,
nro-owned public and private derivative trees, registered development branches,
ownership receipts, and scheduler evidence. It makes these changes:

- converts version-4 ownership receipts to version 5;
- removes obsolete `EventsFile` fields from source imaging sidecars; and
- rewrites supported nro JSON and YAML metadata to use the references above.

The read-only preflight reports separate source, derivative, and
work-item-contract phases with file counts.

On the source side, the migration covers BIDS participant data and the opaque
`sourcedata`, `stimuli`, and `phenotype` namespaces. Source links are
materialized throughout that boundary. Imaging-sidecar rewrites are limited to
`sub-*`, because files under `sourcedata` are not publication metadata.

On the derivative side, public metadata discovery is limited to
`derivatives/nro`. Private WORK trees take part in project moves and scheduler
state, but they are not rewritten as public BIDS metadata.

The migration leaves these alone: the project's `code` directory, third-party
derivatives, external code products such as FreeSurfer subject directories,
content outside the recognized BIDS namespaces, and source fields that have no
explicit migration rule.

Metadata reads are bounded and concurrent, while the migration plan, journal,
and writes stay ordered. Large datasets can still take several minutes, because
every selected metadata record is parsed and validated before the first write.

### Failures and recovery

The read-only preflight stops the migration if it finds an unknown ownership
version, invalid metadata, path traversal, or a path outside the known BIDS and
site roots.

During execution, every replacement is recorded in a durable private journal
before it is written. An ordinary failure rolls back immediately. If an
interruption happens after the public files have passed validation, a later
invocation resumes the registry phase without rewriting them. An interruption
before that point is rolled back before another migration begins.
