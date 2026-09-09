# Standard event files

The event catalog lives in `DEFINITIONS/events` in the selected
[definitions store](definitions.md).
It supplies reusable event TSVs to bidsification without changing an original
source collection. These files describe stimulus timing and condition
labels, not firstlevels models or execution requests.

Each task directory contains an `index.yml`. For example:

```yaml
tasks: [langlocSN]
files:
  set1_run1:
    path: langlocSN/set1_run1.tsv
    source_names: [langlocSN_set1_run1_events.tsv]
```

The logical event ID is `langlocSN/set1_run1`. `tasks` lists the task names
that should suggest this index's entries. `path` is relative to the catalog
root and must point to a TSV directly inside the index's own task directory.
`source_names` records names from the one-off import. Runtime lookup
uses the index, not the original naming conventions. Add explicitly supported
task names to `tasks`; an empty list excludes an index from automatic matching.

## Bidsification

After the user supplies a BOLD task name, `nro bidsify` lists every matching
catalog variant. Matching compares the complete name after removing separators
and folding case: `langloc-SN` matches `langlocSN`, but not `langlocSNW`.
Versions and digits remain significant. No substring or fuzzy match chooses
a scientifically different task.

Choose a displayed `TASK/VARIANT` ID or supply a TSV path. A sole candidate is
offered as a default, still requiring confirmation. Several candidates leave
the choice open. The BIDS `run` entity is not assumed to identify the stimulus
run or set. Digital scanplan parsing remains a separate future integration.

The selected TSV is validated and copied into the session's immutable review
snapshots under its review lease. The request records the catalog ID and
snapshot SHA-256; published BOLD metadata includes them as `NROEventsSource`.
Later catalog edits do not alter an accepted session snapshot or published
events. Explicit event paths and the profile's additional `event_rules` remain
available. The catalog path follows the selected definitions root; it is not
an ingestion-profile setting.

## Editing and adding files

Add a TSV and its entry to the appropriate task index. An entry needs `path`
and `source_names`; use an empty source-name list for a new file. The TSV must
have finite `onset` and `duration` values, nonnegative durations, and at least
one row. Other columns are preserved. Scientific timing and condition labels
still require human review. No `create events` or `edit events` command is
provided yet.

Within one task, several logical IDs may point to one physical TSV when their
imported contents were identical. Editing that TSV affects every referencing
ID in that task. To change only one variant, create a separate TSV and update
only that entry's path. Check the task's index before deleting a shared TSV.
Distinct tasks keep separate TSVs even when their contents match, so editing
one task's events does not change another's. Cross-task references are rejected.

The `unassigned` index preserves files whose names did not identify a task.
Its empty `tasks` list prevents automatic suggestions; its IDs can still be
selected explicitly after review.

## Initial import

The following counts describe this lab's one-off import, not the contents of
a new nro installation. New definitions stores have empty event catalogs.

The one-off import from `/juice6/u/nlp/climblab/eventfiles` accepted 395 source
TSVs into 343 logical IDs, stored in 289 physical TSVs across 54 task directories.
It avoided 106 duplicate copies within tasks. Deduplication compared complete text after
normalizing UTF-8 BOMs, line endings, and terminal newlines; it did not round
numbers, reorder rows, or remove columns. The stored TSVs occupy about 12.1 MB.

Twelve naming conflicts had different contents under names that would otherwise
produce the same ID. Both were kept with distinct IDs. These were story files
with `_events.tsv` and plain `.tsv` variants; the suffix is not sufficient to
establish that their contents are interchangeable.

The import excluded 18 backups and one provenance/audit table that lacks event
timing columns. Another 43 TSVs could not be read because of filesystem
permissions and have not been imported. Source bytes and permissions were not
changed. The detailed local audit, including hashes, duplicate groups, conflicts,
and exclusions, is saved as `plans/evidence/EVENT_FILES_IMPORT.json`.
