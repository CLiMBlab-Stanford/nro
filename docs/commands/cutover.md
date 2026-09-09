# Private-state layout cutover

`nro cutover` converts the previous flat control store into the shared/branch
layout. It is a one-off maintenance operation for the currently supported
registry schema, not a schema migration or an alternate registry reader.

The command does not stop workers, cancel requests, move derivatives, or modify
source BIDS. Instance identities, generations, scientific contracts, and producer
metadata are preserved. It updates operational references to relocated private
files, including manifest locations and execution configuration paths.

## Prepare and preview

Coordinate a maintenance window. All users and services must stop issuing
commands against the store, including clients using older nro code. Finish or
cancel active demand, stop or drain workers, and resolve pending allocations
and ingestion/review leases using the installation that owns that work.
The cutover refuses active registry records even when their leases have expired;
it does not infer inactivity from age or contact Slurm to override them.

Preview the configured site:

```bash
nro cutover --dry-run
```

Use `--registry /absolute/control/root` to inspect a different site. Preview
reads and hashes private files, reports the number of files and their total size,
and lists directory mappings. It performs no writes. `--json` provides a
structured report. Source and derivative images outside the control tree are
not scanned or changed.

Provide enough free space for a second copy plus temporary overhead. The
maintainer needs permission to read the source, write beside the control root,
and preserve file ownership and group membership. Copying state owned by other
users may require administrator assistance; the command fails before publication
if it cannot preserve those permissions. Unknown entries, symlinks, broken
database references, and an incompatible schema are errors, not files to discard.

## Convert

```bash
nro cutover
```

The command displays the plan and asks for confirmation. `-f`/`--force` skips
only that prompt. It never overrides quiescence, integrity, or changed-source
checks. Development installations cannot perform this site-wide operation.

The conversion locks the old store, copies the recognized contents into a
sibling staging directory, updates private path references there, and verifies
that the source remained unchanged. It keeps cached source bundles byte-identical
and preserves historical producer commands. Scientific configuration containing
private paths that would need substantive rewriting requires review; it is not
silently changed.

Publication retains the entire original tree as a sibling rollback copy, then
places the verified new tree at the configured control path. A sibling journal
blocks new-layout clients while publication is incomplete, including the brief
interval when the control root has been renamed. Existing old-code clients do
not know this journal; the maintenance window remains necessary.

On success the command prints the rollback-copy path. Inspect the new store and
retain that copy until the site maintainer has verified the transition. It is
not an execution cache and automatic cache cleanup will not delete it. No
backup or staging files are silently removed by this utility.

The resulting layout is described under
[branch registration](branches.md#storage-and-safeguards). This change does not
enable multi-branch scientific execution. Historical worker scripts and public
ownership receipts retain their original recorded commands; create new demand
through the current planner rather than replaying archived scripts. If an older
development installation recorded the previous branch-catalog path, reconnect
it before use; cutover does not edit checkout-local installation records.

## Recover an interruption

If copying completed and publication was interrupted:

```bash
nro cutover --resume
```

Resume verifies the journal, source copy, and staged contents before completing
publication. It never starts a new cutover when no journal exists. If copying
itself was interrupted, restore the original layout first:

```bash
nro cutover --rollback
```

Rollback restores an unfinished cutover and retains staged files as evidence.
It can also undo publication while its journal still blocks normal clients,
provided the published files have not changed. Completed cutovers cannot be
rolled back with this command: new work may have changed the new registry, so
restoring the backup requires a separately coordinated recovery.

Both recovery actions ask for confirmation unless `--force` is supplied.
Ctrl-C exits with status 130 and leaves recovery information in place.
