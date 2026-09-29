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

## Existing derivatives

Preview the representation migration before applying it:

```bash
nro migrate provenance -P PROJECT
nro migrate provenance -P PROJECT --execute
```

The command is available only from the central installation. It covers the main
derivative tree and every registered development branch. It refuses active attempts,
converts version-4 ownership receipts to version 5, rewrites supported nro JSON and
YAML metadata, and refreshes database integrity evidence for the exact files it
changed. It does not change work-item identities, generations, artifact states, or
scientific contracts. Repeating the migration is a no-op.

Unknown ownership versions, invalid metadata, path traversal, and paths outside the
known BIDS and site roots stop the migration during its read-only preflight. Execution
records every replacement in a durable private journal before writing. An ordinary
failure rolls back immediately; a later invocation rolls back an interrupted
transaction before starting another migration.

Source BIDS metadata and third-party derivatives are outside this migration. Their
absolute paths, if any, require a source-dataset-specific review because changing raw
metadata can legitimately affect descendant freshness.
