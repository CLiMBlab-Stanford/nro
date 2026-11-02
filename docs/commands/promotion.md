# Accepting development artifacts

After a reviewed change is merged, `nro promote` copies the development
branch's outputs that are scientifically equivalent into the accepting
branch's directories, so they do not have to be recomputed. Run it from the
checkout of the accepting ancestor. It does not merge code or delete the
source outputs.

```bash
nro promote --from feature/example --pr lab/nro#42 --attest-merged \
  -P nptl -p t20 -m networks --dry-run
nro promote --from feature/example --pr lab/nro#42 --attest-merged \
  -P nptl -p t20 -m networks
```

`--from` names the development branch that supplies the artifacts, and `--pr`
names the accepted pull request. `--attest-merged` confirms that the reviewed
implementation was merged into this checkout. The PR reference and the
attestation record your statement. nro does not contact the hosting service or
merge Git branches. Main must have an approved release.

The [shared selectors](index.md#shared-selectors) choose the work. An omitted
`--module` selects every workflow endpoint. The accepting checkout compiles its
current workflow and scientific contracts and registers the selection without
requesting computation. Registration happens even with `--dry-run`, but that
mode writes no derivatives.

## What gets copied

Fresh target artifacts are kept. Every missing dependency must have an
equivalent, fresh source artifact, and existing target inputs must agree with
the source inputs. The report lists the copies, the retained artifacts, and the
replacements. Replacing incompatible target files requires `--replace` and a
confirmation. `-f`/`--force` skips the prompt, and `--json` prints the final
result as JSON.

Stop active target jobs before promoting. Transfers are staged and
checksummed. Text references to the source branch are relocated. A reference
that cannot be resolved, or an opaque file that would need a path edit, causes
rejection. External symlinks are not copied.

Before publishing, nro checks the dependency generations and source files
again. Publication invalidates downstream readers and waits for them to shut
down, then commits the target completion records once the files are in place.

## Provenance

The completion metadata keeps the original producing branch and source digest.
It also records the accepting branch, its release, the PR reference, and the
promotion event. Events are recorded under `CONTROL/shared/promotions`. The
accepting release is never presented as the version that computed the data.

## Interruptions and recovery

Each destination replacement has a durable publication phase and a rollback
copy. If publication stops before the registry transaction commits, recovery
restores the earlier target files or removes newly added ones. If the
transaction committed but the ownership metadata was not finished, recovery
completes the metadata without replacing the committed result.

After an interruption, repeat the same command. nro recovers unfinished
journals before it reassesses the request. The source artifacts stay available
throughout. Once you have checked that they are no longer needed, retire or
purge the source separately.
