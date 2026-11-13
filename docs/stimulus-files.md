# Stimulus annotations

A stimulus such as a story is often presented in several projects. Its
annotations, such as word or phoneme timings, are stored once in the
definitions store and copied into each project whose events name the stimulus.
First-level models then read the project copy. See
[models from stimulus annotations](task-models.md#models-from-stimulus-annotations)
for how a model uses one.

The layout follows the draft BIDS stimuli extension (BEP044), which is not part
of a released BIDS version.

## Catalog layout

The catalog lives in `DEFINITIONS/stimuli` in the selected
[definitions store](definitions.md). It uses the same flat layout as a
project's `stimuli/` directory:

```text
DEFINITIONS/stimuli/
  stimuli.tsv
  annotations.tsv
  stim-story1_annot-words_events.tsv
  stim-story1_annot-words_events.json
  stim-story1_annot-phonemes_events.tsv
  stim-story1_annot-phonemes_events.json
```

`stimuli.tsv` has one row per stimulus. The `stimulus_id`, `type`, `license`,
and `present` columns are required. `stimulus_id` has the form `stim-<label>`,
and `type` is `audio`, `image`, `video`, or `audiovideo`. The catalog holds no
media files, so `present` must be `false`. Other columns are kept.

```text
stimulus_id	type	license	present	description
stim-story1	audio	n/a	false	A ten-minute narrative
```

Each annotation is an `_events.tsv` file with `onset` and `duration` columns,
in seconds from the start of the stimulus, and a JSON sidecar. Every annotation
file needs a `stimuli.tsv` row. `annotations.tsv` is optional. When it exists,
it lists every annotation label in its `annot_id` column, usually with a
`description`.

Media files, `part-<label>` annotations, and other files in `stimuli/` are
rejected. A stimulus ID may appear in only one store of the definitions chain,
so the same `stim_id` never resolves to different timing in different
projects.

## Versions

Each annotation sidecar carries `NROVersion`, a positive integer:

```json
{
  "Description": "Word onsets from forced alignment",
  "NROVersion": 2
}
```

A new annotation starts at version 1. When you publish a changed annotation
TSV with `nro def apply`, raise its `NROVersion`. Publishing changed timing
under the same version, or a higher version with unchanged timing, is an error.
A sidecar-only change, such as a new description, keeps the version.

The catalog keeps only the current version. Earlier versions stay in the
definitions store's Git history. If an annotation was changed directly in the
published store, nro compares it with its version at the store's Git `HEAD`.

## Copying annotations into a project

Bidsification copies the annotations of the stimuli that reviewed events name.
Review snapshots them from the catalog, and publication adds missing files and
rows to the project's `stimuli/` directory. It never replaces an existing copy.
See [event files in bidsify](commands/bidsify.md#event-files).

For projects that already exist, or to apply a newer catalog version, use
`nro stimuli sync`:

```bash
nro stimuli sync PROJECT --dry-run
nro stimuli sync PROJECT
nro stimuli sync PROJECT --replace
```

`sync` finds every `stim_id` value in the project's run events files and
copies all annotations of those stimuli into `PROJECT/stimuli/`, including
annotations that no current model uses. It adds missing rows to the project's
`stimuli.tsv` and `annotations.tsv`, and never rewrites rows that are already
there.

Each copied sidecar records its source in `NROStimulusSource`: the catalog
path, the copied `NROVersion`, and the SHA-256 of the copied TSV. Sync uses
that record to classify every project annotation:

| Status | Meaning | Action |
| --- | --- | --- |
| `missing` | The project has no copy. | Copy it. |
| `current` | The copy matches its record and the catalog version. | None. |
| `stale` | The copy matches its record, but the catalog has a newer version. | Replace it with `--replace`. |
| `local edit` | The TSV no longer matches its recorded hash. | Report it and leave it. |
| `not written by nro` | The sidecar has no `NROStimulusSource`. | Report it and leave it. |
| `ahead of catalog` | The recorded version is newer than the catalog's. | Report it and leave it. |

Replacing a stale copy makes every fit that uses it out of date, so sync
replaces stale files only with `--replace`. `--dry-run` reports the same
statuses without changing files.

Sync refuses to change files while the project has queued or running
derivative attempts, and holds the registry lock while it copies so that no
new attempt starts until it finishes. Sync exits nonzero when the project's
events name a stimulus that is not in the catalog, or when a project file is
blocked from update. A development
installation can run only `--dry-run`, because it cannot change shared source
BIDS.
