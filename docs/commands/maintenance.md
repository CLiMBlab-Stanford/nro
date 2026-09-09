# Installation, cleanup, and publication

## `./install` and `nro setup`

Both use the same installer. First setup accepts `--mode personal`,
`--mode shared`, or `--mode branch`, or asks interactively. A new checkout uses
branch mode automatically when the user's dispatcher has a default installation.
An existing shared checkout connects
the caller's launcher unless `--maintain` is given. See the
[installation guide](../installation.md) for effects and permissions.

| Option | Meaning |
| --- | --- |
| `--mode personal/shared/branch` | Declare the first installation's role; branch mode reuses an existing site read-only. |
| `--maintain` | Permit shared environment/resource maintenance. |
| `--site PATH` | Choose the site settings file during first setup. |
| `--bin-dir PATH` | Place the user's launcher in this directory. |
| `--non-interactive` | Use supplied settings without prompts. |
| `--offline` | Forbid new resource downloads; require cached/installed dependencies. |
| `--without-oslom` | Omit OSLOM and its Python dependencies for this invocation. |
| `--dev` | Include locked test dependencies. |
| `--accept-qunex-license` | Acknowledge terms for unattended image acquisition. |
| `--local` | Do not require Slurm. |

Ctrl-C exits with status 130 and one cancellation message. Incomplete shared
setup resumes with `./install --maintain`; accepting defaults does not eliminate
the need for license files and suitable host permissions.

## `nro paths`

With no subcommand, show proposed defaults and ask whether to accept all. If
declined, prompt independently for every setting. `show` prints resolved values
and their sources. `set key=value ...` validates and atomically saves updates.
`--maintain` is required to edit a shared site's settings. This command never
moves datasets, artifacts, or registries. See the installation guide for all keys.

## `nro doctor`

Check Python imports, runtime availability, configured resources, filesystem
access, and Slurm executables. `--deep` additionally starts containers and
checks tool availability, template checksums, and image receipts. `--local`
makes Slurm optional; `--without-oslom` makes OSLOM optional. `--json` emits
structured results. Failed required checks exit nonzero. Doctor does not install
resources or process subject data.

## `nro purge`

Delete owned public derivatives, private intermediates, and associated logs.
Selectors narrow scope; absent selectors mean all. Ownership receipts and
module contracts distinguish nro outputs from unrelated derivatives.

```bash
nro purge -P nptl -p t20 -m clean --dry-run
nro purge -P nptl -p t20 -m clean
```

The normal invocation displays a pageable list of public/private removal paths
and asks for confirmation. `-f`/`--force` skips confirmation. `--dry-run` reports
without deleting. `-l`/`--logs` removes matching attempt logs and inactive-worker
logs only. `--json` supplies a structured result; `--work-root` overrides the
private-file root. Selected instances must have no active attempts.

Deleting an upstream artifact also invalidates its consumers and requests
cancellation of their active attempts, even when those consumers were not selected
for deletion. Their files are not purged. The command reserves the selected
outputs and waits up to 30 seconds for confirmed consumer shutdown without holding
the registry lock while waiting. If shutdown is not confirmed, it deletes no
outputs; retry after the attempts stop. Invalidation and cancellation remain in
effect, and demand is preserved. `--dry-run` does not invalidate or cancel work.

After an interrupted purge, repeat the operation to recover its mutation lock and
reservation. Registry repair refuses unresolved mutation reservations. This
prevents repair from discarding the barrier while files might still be changing.

Deletion is destructive and does not move files to trash. A bare invocation
selects every controlled derivative owned by the current branch across all
projects, including historical lineages. Inherited outputs are excluded. Force
does not mean it is safe to delete a shared user's needed work.

Use `--cache` to remove unused executable-source snapshots and captured site
settings across the shared installation:

```bash
nro purge --cache --dry-run
nro purge --cache
nro purge --cache --force
```

This mode ignores all artifact selectors, including project, participant,
module, and workflow. `--bids-root` still selects the registry context. It does
not remove derivatives, logs, definitions, container images, or dependency
downloads, and cannot be combined with `--logs`.

Cache cleanup is conservative: outstanding demand, attempts, workers, scheduler
submissions, or queued/running ingestion defer collection across the whole
pool. Unknown registry state also preserves the cache. `--force` skips the
prompt; it never overrides these checks. State is checked again after
confirmation. Old snapshots are not needed to assess derivative freshness.

Normal requests and worker shutdown also attempt cleanup automatically. A
process running from a snapshot retains its own source and site file; a later
command can reclaim those after it exits. Unknown entries and partial capture
directories are not automatically removed.

## `nro publish`

```bash
nro publish REQUEST_ID /destination -P PROJECT
```

Assess a completed request, copy terminal public files to a staging directory,
verify checksums, embed recursive provenance, and recheck source generations
before publishing. `destination` must not already exist. `--bids-root` overrides
the source root. `--no-validate` skips the optional external BIDS validator.
Otherwise the validator runs only if available on PATH.

Publication writes `dataset_description.json` and `.nro-publication.json`.
It is a separate snapshot operation, not an assertion that every working
derivative format meets the BIDS validator. Inspect the published files and
validation output before distributing the dataset; the command does not upload
anything or create a release.
