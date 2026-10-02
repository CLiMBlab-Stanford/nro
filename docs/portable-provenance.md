# Portable derivative provenance

nro public derivatives do not use host-specific absolute paths to describe their
inputs and outputs. JSON and YAML metadata use BIDS URIs:

- `bids::anat/...` identifies a file in the current `derivatives/nro` dataset;
- `bids:raw:sub-01/...` identifies a file in the containing raw BIDS project; and
- `nro-site:templates:...` identifies a resource under a named site root when the
  resource is not part of either BIDS dataset.

Each `derivatives/nro` directory is a BIDS Derivatives dataset with a
`dataset_description.json`. Its `DatasetLinks.raw` value links the derivative
dataset to the containing raw project. Runtime code resolves references against the
active raw, derivative, and site roots before invoking scientific software.

Portable metadata is descriptive and recoverable, but it is not a second freshness
authority. The scheduler database still records direct-source inventories, upstream
generations, public output integrity, and any existing private intermediates. Public
ownership receipts reconstruct the inter-module DAG after registry loss. Runner
ledgers reconstruct each module's internal steps. Moving a complete project therefore
changes neither its scientific contracts nor its dependency graph.

## Existing datasets

Run the migration interactively:

```bash
nro migrate -P PROJECT
```

List multiple IDs after `-P` to migrate several projects together. A bare
`nro migrate` selects every project.

The command pages through the complete plan and asks for confirmation before it
writes anything. Use `--dry-run` to stop after the preview or `-f`/`--force` to skip
confirmation. The command is available only from the central installation. Before
the preview, it starts or joins the central Slurm scheduler without creating
scientific demand. Preview and execution therefore run on the scheduler allocation,
not the login node. It coordinates changes to the selected source BIDS projects,
nro-owned public and private derivative trees, registered development branches,
ownership receipts, and scheduler evidence. It refuses active attempts, converts
version-4 ownership receipts to version 5, removes obsolete `EventsFile` fields from
source imaging sidecars, and rewrites supported nro JSON and YAML metadata. Repeating
the migration is a no-op.

The preflight reports separate source, derivative, and work-item-contract phases with
file counts. Its source boundary includes BIDS participant data and the opaque
`sourcedata`, `stimuli`, and `phenotype` namespaces, but excludes the project's
`code` directory and unrecognized content. Source-link materialization applies
throughout that boundary. Imaging-sidecar rewrites remain limited to `sub-*` because
files under `sourcedata` are not publication metadata. Public metadata discovery is
limited to `derivatives/nro`; third-party derivatives and external code products such
as FreeSurfer subject directories are not traversed. Private WORK trees participate
in project moves and scheduler state but are not rewritten as public BIDS metadata.
Metadata reads are bounded and concurrent, while the migration plan, journal, and
writes remain ordered. Large datasets can still take several minutes because every
selected metadata record is parsed and validated before any write begins.

Source images remain byte-level freshness inputs. Inherited JSON metadata is stored
in work-item contracts as a canonical projection of the fields used by the relevant
module. Changing `RepetitionTime`, for example, changes a functional contract;
changing an unused descriptive field does not. The same projection is stored in the
public ownership receipt, so registry repair does not depend on a private metadata
index.

Unknown ownership versions, invalid metadata, path traversal, and paths outside the
known BIDS and site roots stop the migration during its read-only preflight. Execution
records every replacement in a durable private journal before writing. An ordinary
failure rolls back immediately. If public files have passed validation, a later
invocation resumes the registry phase without rewriting them; earlier interruptions
roll back before another migration begins.

Third-party derivatives, the project `code` directory, and content outside the
recognized BIDS namespaces remain outside this migration. Source fields without an
explicit migration rule remain untouched.
