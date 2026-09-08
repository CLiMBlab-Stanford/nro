# Installation, cleanup, and publication

## `./install` and `nro setup`

Both use the same installer. First setup needs `--mode personal` or
`--mode shared`, or asks interactively. An existing shared checkout connects
the caller's launcher unless `--maintain` is given. See the
[installation guide](../installation.md) for effects and permissions.

| Option | Meaning |
| --- | --- |
| `--mode personal/shared` | Declare the first installation's role. |
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
private-file root. Active work is protected by the command's worker checks.

Deletion is destructive and does not move files to trash. A bare invocation
selects every controlled derivative across all projects, including historical
lineages. Force does not mean it is safe to delete a shared user's needed work.

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
