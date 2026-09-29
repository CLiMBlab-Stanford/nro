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
nro migrate dataset -P PROJECT
```

The command pages through the complete plan and asks for confirmation before it
writes anything. Use `--dry-run` to stop after the preview or `-f`/`--force` to skip
confirmation. The command is available only from the central installation. It
coordinates changes to the selected raw BIDS projects, the main derivative tree,
registered development branches, ownership receipts, and scheduler evidence. It
refuses active attempts, converts version-4 ownership receipts to version 5, removes
obsolete `EventsFile` fields from source imaging sidecars, and rewrites supported nro
JSON and YAML metadata. Repeating the migration is a no-op.

Source images remain byte-level freshness inputs. Inherited JSON metadata is stored
in work-item contracts as a canonical projection of the fields used by the relevant
module. Changing `RepetitionTime`, for example, changes a functional contract;
changing an unused descriptive field does not. The same projection is stored in the
public ownership receipt, so registry repair does not depend on a private metadata
index.

Unknown ownership versions, invalid metadata, path traversal, and paths outside the
known BIDS and site roots stop the migration during its read-only preflight. Execution
records every replacement in a durable private journal before writing. An ordinary
failure rolls back immediately; a later invocation rolls back an interrupted
transaction before starting another migration.

Third-party derivatives remain outside this migration. Source fields without an
explicit migration rule remain untouched.
