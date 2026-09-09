# Accepting development artifacts

Run `nro promote` from the accepting ancestor's checkout after merging a reviewed
change. The command copies scientifically equivalent development outputs into
that branch's directories. It does not merge code or delete the source outputs.

```bash
nro promote --from feature/example --pr lab/nro#42 --attest-merged \
  -P nptl -p t20 -m networks --dry-run
nro promote --from feature/example --pr lab/nro#42 --attest-merged \
  -P nptl -p t20 -m networks
```

The normal planner selectors apply. The accepting checkout compiles its current
workflow and scientific contracts, registering the selection without requesting
computation. Registration occurs even with `--dry-run`; no derivatives are
written in that mode. Main must have an approved release. The PR reference and
`--attest-merged` record the operator's statement; nro does not contact the hosting
service or merge Git branches.

Fresh target artifacts are kept. Every missing dependency must have an equivalent,
fresh source artifact. Existing target inputs must agree with the source inputs.
The report lists copies, retained artifacts, and replacements. Replacing
incompatible target files requires `--replace` as well as confirmation.
`-f`/`--force` skips the prompt. `--json` emits the final result as JSON.

Transfers are staged and checksummed. Text references are relocated; unresolved
source-branch references or opaque files requiring path edits cause rejection.
External symlinks are not copied. Active target jobs must be stopped first.
Publication invalidates downstream readers and waits for their shutdown, then
writes completion manifests last. Dependency generations and source files are
checked again before publication.

The completion metadata retains the original producing branch and source digest,
plus the accepting branch, release, PR reference, and promotion event. Events are
recorded under `CONTROL/shared/promotions`. The accepting release is not presented
as the version that originally computed the data.

Each destination replacement has a durable publication phase and rollback copy.
If publication stops before the registry transaction commits, recovery restores
the prior target files or removes newly introduced files. If the transaction
committed but ownership metadata did not finish, recovery completes that metadata
without replacing the committed result. Repeat the same command after an
interruption; nro recovers nonterminal journals before reassessing the request.
Source artifacts remain available throughout. Retire or purge the source
separately after checking that it is no longer needed.
