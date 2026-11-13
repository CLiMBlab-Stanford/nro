# Standard event files

The event catalog holds reusable events TSVs for bidsification. When you
review a task BOLD run in `nro bidsify`, you can pick a catalog entry instead
of supplying a file. The catalog lives in `DEFINITIONS/events` in the selected
[definitions store](definitions.md). Its files describe stimulus timing and
condition labels. They do not define contrasts models or request processing.

Fine-grained timing within a stimulus, such as word onsets in a story, can
instead live in stimulus annotation files in the project's `stimuli/`
directory. The run events then mark only when each stimulus started. See
[models from stimulus annotations](task-models.md#models-from-stimulus-annotations).
Those annotations have their own [catalog](stimulus-files.md).

## Catalog layout

Each task directory contains an `index.yml`:

```yaml
tasks: [langlocSN]
files:
  set1_run1:
    path: langlocSN/set1_run1.tsv
    source_names: [langlocSN_set1_run1_events.tsv]
```

This entry has the event ID `langlocSN/set1_run1`. `tasks` lists the task
names that should suggest the index's entries. An empty list keeps the index
out of automatic suggestions. `path` is relative to the catalog root and must
point to a TSV directly inside the index's own task directory.
`source_names` records the original filenames from the one-off import. Lookup
uses only the index, not those naming conventions.

## Selecting events during bidsification

After you give a BOLD task name, `nro bidsify` lists every catalog variant for
that task. Matching compares the whole name after removing separators and
ignoring case. `langloc-SN` matches `langlocSN`, but not `langlocSNW`. Versions
and digits count, and no substring or fuzzy match can pick a scientifically
different task.

Enter a displayed `TASK/VARIANT` ID or the path to a TSV. A single candidate is
offered as the default but still needs confirmation. With several candidates,
you must choose. nro does not assume that the BIDS `run` entity identifies the
stimulus run or set. A [site scan-plan parser](commands/scanplans.md) can
supply the task ID, but you still select the event file during review.

nro validates the chosen TSV and copies it into the session's immutable review
snapshots. The request records the catalog ID and the snapshot's SHA-256, and
published BOLD metadata includes both as `NROEventsSource`. Later catalog edits
do not change an accepted snapshot or published events.

You can still supply an explicit event path, and the protected site's
`event_rules` can add candidates from outside the catalog. The catalog path
always follows the selected definitions root and is not a conversion-profile
setting.

## Adding and editing files

To add a file, put the TSV in its task directory and add an entry to that
task's index. Each entry needs `path` and `source_names`. Use an empty
`source_names` list for a new file. The TSV must have at least one row, finite
`onset` and `duration` values, and no negative durations. Other columns are
kept. A person still needs to review the timing and condition labels. There is
no `create events` or `edit events` command yet.

Within one task, several event IDs may point to the same TSV when their
imported contents were identical. Editing that TSV changes every ID that refers
to it. To change only one variant, create a separate TSV and update only that
entry's `path`. Check the index before deleting a shared TSV.

Different tasks always keep separate TSVs, even when the contents match, so
editing one task's events never changes another task's. References across
tasks are rejected.

The `unassigned` index holds files whose names did not identify a task. Its
empty `tasks` list keeps them out of suggestions, but you can still select
their IDs explicitly after reviewing them.

## Initial import

New definitions stores start with an empty catalog. The counts below describe
this lab's one-off import only.

The import from `/juice6/u/nlp/climblab/eventfiles` accepted 395 source TSVs
as 343 event IDs, stored in 289 TSVs across 54 task directories. It avoided 106
duplicate copies within tasks. Duplicates were found by comparing the complete
text after normalizing UTF-8 BOMs, line endings, and final newlines. Numbers
were not rounded, and rows and columns were not reordered or removed. The
stored TSVs take about 12.1 MB.

Twelve name collisions had different contents under names that would have
produced the same ID. Both files were kept, with distinct IDs. These were story
files with `_events.tsv` and plain `.tsv` variants, and the suffix alone does
not show that their contents are interchangeable.

The import skipped 18 backups and one provenance table that has no event
timing columns. Another 43 TSVs could not be read because of filesystem
permissions and were not imported. Source files and their permissions were not
changed. The detailed audit, with hashes, duplicate groups, conflicts, and
exclusions, is saved as `plans/evidence/EVENT_FILES_IMPORT.json`.
