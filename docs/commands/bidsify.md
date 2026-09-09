# Bidsify

`nro bidsify` downloads selected Flywheel acquisitions, prepares a BIDS session,
and asks before publishing it. A terminal wizard collects decisions. Cluster
workers perform the downloads and conversions without holding the terminal
open. No AI service is used.

Ingestion shares the [worker pool](../orchestration.md) and concurrency limit
with derivative processing. It does not create derivative instances or request
preprocessing. After publication, use `nro run` to request derivatives.

## Before first use

Install the Python dependencies with `./install --with-bidsify`. For an existing
shared installation, its maintainer runs `./install --maintain --with-bidsify`
and confirms an interactive drain if work is active. The extra includes the
Flywheel SDK, dcm2bids, and pydicom. Existing processing installations without
the extra can still report ingestion state with `nro status`.

The default conversion commands use the installation's QuNex image for
dcm2niix and its SynthStrip image for skull stripping. Supply a BIDS validator
on the workers' PATH, or configure an absolute command in the ingestion profile.
The default is `bids-validator`; this external program is not installed by the
Python extra. Worker nodes need network access to Flywheel and shared staging.

Each site defines its Flywheel servers and remote project scopes in the
definitions store. Configure credentials outside the repository and command
arguments. For example, the CLIMBLAB definitions store contains:

| Server | Host | Remote project scope | Credential environment variable |
| --- | --- | --- | --- |
| `cni` | `cni.flywheel.io` | `cashain/climblab` | `FW_API_KEY_CNI` |
| `lucas` | `lucascenter.flywheel.io` | `shain/shain1` | `FW_API_KEY_LUCAS` |

The credential may be a token or a matching `HOST:TOKEN` value. nro creates an
isolated SDK client; it does not change Flywheel CLI login state. Credentials
must be present in the environment of the user and workers that inspect or
download sessions. Workers started before a credential was supplied do not
gain it automatically. Stop those workers and submit from the authenticated
environment if needed. Tokens are never written into request records or worker
scripts. Use the cluster's approved credential-management procedure.

## Select, review, and publish

```bash
nro bidsify --server mysite --flywheel-project group/study -P example
nro status -P example
nro bidsify --request REQUEST_ID
```

`-P` names one destination BIDS project. `--flywheel-project GROUP/PROJECT`
names its source on Flywheel. These names need not match:

```bash
nro bidsify --server cni --flywheel-project cashain/climblab -P climblab_multisession
```

A configured `project_sources` mapping lists the sources allowed for the
BIDS project, including sources at different scanning sites. Nro selects the
sole matching source or asks the user to choose one by server/project name or
number. `--server` and `--flywheel-project` narrow that choice. This selection happens
before any sessions are listed. Only that Flywheel project is queried; sessions
from other projects are not pooled into the list. Explicit selectors that
match none of the configured sources are rejected. Without a mapping, all
configured server/project pairs are available for selection.

Saved requests are offered for the selected server and destination BIDS
project; the source selector controls new session discovery. `--request`
resumes one request directly without source selection or remote discovery.
Existing-session matching searches all raw projects under the BIDS root, so
data already stored in another project are also omitted by default.

1. The wizard offers unfinished requests first. Resume selected requests or
   choose new sessions. Before listing new work, nro omits known existing BIDS
   sessions using publication records and the profile's `session_rules`. It
   reports how many were omitted. The remote list shows session IDs and labels. Select
   numbers or `all`. Matching server rules supply the BIDS session label when
   they agree; otherwise the wizard asks for it. Supply the participant label
   if known, or press Enter to leave it pending. Confirm the inspection
   request. No images are downloaded during inspection.
2. A worker inventories DICOM files. On the next invocation, confirm each
   acquisition's type and BIDS entities. For task BOLD, select a suggested
   `TASK/VARIANT` from the [standard event store](../event-files.md), or supply an events TSV, and
   confirm that it contains no identifying text. Uncertain acquisitions can be
   deferred or explicitly ignored. Type `skip` to leave the current session
   for later, or `q` to exit; completed acquisition decisions remain saved.
3. A worker downloads and converts the selected images. Anatomy is stripped
   locally before reaching shared staging. The next review shows converted
   metadata, asks for any missing BIDS labels, and requests an SBRef and opposite-encoding fieldmap pair for each
   BOLD acquisition. Numbered candidates must have compatible geometry,
   encoding, and readout time. Explicit `none` is allowed. Task events can be
   corrected at this review, including timing that exceeds the converted run.
   Missing identity does not prevent image preparation. Leave a label pending
   to retain the sanitized images and resume review later.
4. A worker organizes the sanitized images through dcm2bids, writes the reviewed
   reference associations, and runs the validator. Failure prevents approval.
5. The next invocation lists the validated files and their SHA-256 hashes.
   Approve those exact files, edit the decisions, or leave them staged. A worker
   publishes only after approval and after rechecking both staged and existing
   file hashes.

Participant and session labels must both be resolved before step 4. Until then,
images remain under request-specific staging paths outside BIDS; no placeholder
subject is published. `nro status` displays unresolved labels as `(pending)`.
Use an unfiltered invocation or `--request` to resume a request whose participant
is still unknown. Completing its identity does not repeat image preparation.
Each supplied label is saved under the review lease. Resolved labels cannot be
reassigned by creating another request or using `--rebidsify`.

A source session is identified by its server and immutable remote session ID.
Its first request reserves one destination BIDS project, even if the participant
is still pending. Later requests must retain that project and any resolved
participant/session labels. Re-bidsification creates a new attempt at the same
destination. Cancellation stops work but does not release the identity mapping.
There is no destination-reassignment command yet; correcting a mistaken mapping
requires a separate, explicit maintenance operation.

The wizard leases only the session it is currently reviewing, including its
publication-approval prompts. Selecting multiple sessions does not lock the
whole selection. Other users can review other sessions concurrently. An
occupied session is skipped with the reviewer's user, host, and process ID.
The lease is released before moving to the next session, including after
`skip`, `q`, an interrupt, or an error.

Leases renew every 30 seconds while the terminal waits for input and expire
after two minutes without renewal. A terminal that loses its lease cannot
renew expired ownership or save more decisions. Reopen the session to resume.
The global registry lock is held only for short operations, not while answering
prompts. Registered requests remain in place even when nobody holds a review
lease. A request reserves its BIDS destination once both labels are known;
filling a missing label checks for another request or an existing destination.

Decision saves require both the active lease and the current record revision.
Event TSV snapshots are written under the same checks and use content-addressed
filenames; a rejected edit cannot overwrite an accepted file. Unfinished
acquisition answers stay in memory until that acquisition's review is saved.
Running requests cannot be edited, and workers cannot claim a leased request.
Changes made during review clear publication approval. Stages release their
workers while waiting for input.

For events, nro checks required `onset` and `duration` columns, finite values,
nonnegative durations, and event onsets relative to the stored run duration
when known. Negative onsets are allowed. Rest does not require events. The
operator remains responsible for synchronization with the first stored volume,
condition labels, and the absence of identifying text. See the
[BIDS events specification](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html).

Reviewed fieldmaps receive `B0FieldIdentifier` and `IntendedFor`; BOLD receives
`B0FieldSource`. The additional `NROReferencePolicy` and `NROSBRef` fields make
both an assigned SBRef and an explicit absence authoritative for `func`.
Preprocessing does not replace these choices with temporal matching.

## Options

| Option | Meaning |
| --- | --- |
| `--server NAME` | Profile name, such as `cni` or `lucas`. |
| `-P`, `--project NAME` | One destination BIDS project. |
| `--flywheel-project GROUP/PROJECT` | One configured source Flywheel project for new sessions. |
| `-p`, `--participant LABEL ...` | Filter saved requests; a single label also supplies the default for a new mapping. Remote participant labels are not assumed to be BIDS labels. |
| `--session ID ...` | Filter by remote session IDs, not BIDS session labels. |
| `--request ID` | Resume one saved request without listing remote sessions. |
| `--config PATH` | Complete ingestion profile YAML for new requests. |
| `--rebidsify` | Include sessions already in BIDS, regardless of which tool produced them, and permit a replacement proposal. Approval is still required. |
| `--no-submit` | Do not supply new workers. Existing workers may still claim queued stages. |
| `--bids-root PATH` | Override the directory containing destination projects. |

Matching a remote session to an existing `sub-…/ses-…` directory is sufficient to
treat it as already bidsified. nro does not validate externally produced data,
inspect image contents, or create publication receipts for them. An existing
destination is never silently treated as safe to replace.

Subject-only intake layouts, such as `climblab/sub-ex123/`, do not suppress
bidsification. Properly organized sessions, such as
`climblab_multisession/sub-t20/ses-ex123/`, do. The old intake files are not
removed or migrated. An explicitly requested re-bidsification of an external
session must retain its matched project, participant, and session. Ambiguous
matches must be resolved before a replacement request can be registered.

Unfinished nro requests retain their recovery workflow even if their BIDS
destination exists. They are not offered again as new sessions, including with
`--rebidsify`. A count points to `nro status` when unfinished requests belong to
another project. Multiple possible BIDS destinations, or multiple remote
sessions matching the same destination, remain visible as ambiguous mappings.
Their candidate paths are shown for review.

`nro status` displays unfinished requests in a separate Bidsification section;
JSON reports include a `bidsification` list. Project, participant, and session
filters apply. Derivative-only selectors suppress the section. Status never
contacts Flywheel or starts a review.

Failed and interrupted requests offer retry, decision review, cancellation, or
leave-as-is. To interrupt executing work, use the existing worker controls
described under [stop](work.md). Requests whose workers are confirmed dead are
marked interrupted; a stale heartbeat alone is not proof that work stopped.

## Configuration

The default profile is `DEFINITIONS/bidsify/main.yml` in the selected
[definitions store](../definitions.md). `--config FILE` selects another complete
profile. New stores have no configured servers; add your Flywheel hosts,
credential environment-variable names, and project lists before ingestion.
Copy the complete main document when creating another profile.
Unknown keys and missing required keys are rejected. `project_sources` is
optional and defaults to an empty mapping. Each request saves its resolved profile,
so later profile edits apply to new requests, not an in-progress conversion.

| Key | Purpose |
| --- | --- |
| `servers` | Named `host`, `credential_env`, and `projects` lists. No credential values. |
| `project_sources` | Optional BIDS project names mapped to nonempty lists of `{server, project}` sources. Each source must appear in that server's `projects` list; duplicate pairs are rejected. |
| `staging` | Absolute shared directory outside BIDS. `null` uses `WORK/bidsify`. |
| `dcm2niix` | Command argument list; `null` resolves the configured QuNex container command. |
| `synthstrip` | Command argument list; `null` resolves the configured SynthStrip container command. |
| `validator` | Command argument list. The staged dataset path is appended. Nonzero exit blocks publication. |
| `memory_gb`, `cpus`, `hours` | Worker allocation settings; defaults 32 GiB, 2 CPUs, 12 hours. |
| `concurrency` | Requested shared limit; default 50. `nro set concurrency=N` updates active ingestion and derivative requests. |
| `protocols` | Ordered regular-expression suggestions with `pattern`, `datatype`, and `suffix`. The first match is proposed; the operator must confirm it. |
| `event_rules` | Additional `task` and absolute glob `pattern` pairs for candidates outside the catalog. All matches are shown; no ambiguous candidate is selected automatically. |
| `session_rules` | Explicit remote-identity rules for recognizing existing raw BIDS subjects/sessions. Empty in new stores. |

The standard event catalog is always `DEFINITIONS/events`. Requests record its
resolved path as runtime metadata, alongside the selected event snapshots.

### Default source projects

For a BIDS dataset receiving sessions at both sites, the mapping can be:

```yaml
project_sources:
  climblab_multisession:
    - server: cni
      project: cashain/climblab
    - server: lucas
      project: shain/shain1
```

Then `nro bidsify -P climblab_multisession` offers the two sources.
Adding `--server cni` selects the CNI source without another prompt.
Mappings are site-specific; new stores leave them empty. Multiple BIDS datasets
may draw different sessions from the same source project, but a source session
has only one destination. An unmapped BIDS dataset requires source selection
when multiple sources match its selectors. Selecting a source in the
wizard does not edit the profile; add a mapping to retain that default.

### Existing-session rules

Each rule applies to one configured `server` and `remote_project`. `match`
contains full-match regular expressions for one or more remote fields:
`id`, `label` (session label), or `subject_code`. Every pattern must match;
repeated named groups must have the same value. The resulting named groups
populate BIDS label templates. nro does not use fuzzy matching or silently
sanitize labels. Labels that cannot produce valid BIDS identifiers do not match.

For example, this lab's CNI session label `31381` corresponds to exam `ex31381`.
The following rule recognizes that session under any participant:

```yaml
session_rules:
  - server: cni
    remote_project: cashain/climblab
    match: {label: '(?:ex)?(?P<exam>[0-9]+)'}
    participant: null
    session: 'ex{exam}'
```

`participant: null` searches for the exact session label under any participant.
Rules without a session label cannot match a completed destination.
Supplying both labels matches that exact participant and
session. `derivatives`, `sourcedata`, and directories without a valid nro project
name (including hidden and underscore-prefixed directories) are excluded.
For new work, nro also uses the session label from these rules when all matching
non-null session labels agree. In the example, selecting remote session `31381`
supplies `ses-ex31381` without another prompt. Conflicting or absent session
rules leave the label for review. The wizard does not copy an unconfigured
remote display label, strip punctuation, or infer a stable participant from
an exam identifier. The proposed destination appears before the user
confirms the request.

Configure rules for other server naming conventions explicitly, then run
`nro definitions validate`. New stores supply no naming assumptions. CLIMBLAB's
CNI rules do not apply to Lucas. Publication always uses a subject/session
hierarchy.

Command lists are executed without a shell. In conversion and skull-stripping
commands, `{staging}` expands to the acquisition's working directory. Default
container commands bind that directory. Custom wrappers must preserve the raw
anatomy boundary and return nonzero on failure. A validator wrapper must really
validate the dataset; approval relies on its exit status.

## Staging, privacy, and replacement

Shared staging is organized as `STAGING/REQUEST_ID/`, with sanitized helpers,
reviewed events, generated dcm2bids configurations, and a proposed `bids/` tree.
Raw anatomical archives, DICOMs, converted full-head images, and skull-stripping
intermediates use only:

```text
/tmp/nro/bidsify/SERVER/REMOTE_SESSION_ID/REQUEST_ID/ACQUISITION_ID/
```

The acquisition classification is a privacy decision made before transfer.
Unknown types are not downloaded. Incorrectly classifying anatomy as functional
would route its source to shared staging; review this step carefully.
Anatomical staging rejects symlinks, world access, and ownership outside the
executing user or shared group. Administrators must provision a suitable shared
group if several users need access. Other raw acquisition files use shared
staging and are removed after conversion.

Successful preparation and handled failures remove raw acquisition files.
Cancellation runs cleanup where the process can handle the termination signal.
A forced kill or node failure can leave anatomy on that node's `/tmp`; a later
attempt on another node cannot clean the old node. The saved request records
the worker hostname. Site-managed temporary cleanup is still required. `/tmp`
does not guarantee a retention interval, exclusion from backups, or secure
erasure. Verify those properties with the cluster administrator.

Archive members receive opaque temporary filenames. NIfTI text fields and
extensions are cleared, and JSON output retains an acquisition-metadata
allowlist. This reduces identifying metadata; it is not a certification of
DICOM de-identification. Raw converter/provider diagnostics are suppressed
because they can contain identifying labels. Safe failures appear in
`REGISTRY/ingestion/REQUEST_ID.log`; unexpected failures report their exception
type without the original message.

Publication copies the approved session to a hidden sibling on the destination
filesystem, verifies it, and renames it into place. Replacing a session uses
Linux atomic directory exchange. Filesystems that cannot perform that exchange
fail without a two-rename fallback. The existing session is removed only after
the exchange, and a receipt supports recovery after an interrupted commit.
Other sessions are untouched. Existing raw `dataset_description.json` metadata
is preserved; an absent description is created.

Replacement requires no active nro derivative attempts in the destination
project at commit time. Stop external readers before approving replacement;
nro cannot coordinate programs outside its registry. Re-publication does not
rebuild derivatives automatically.

Records, logs, and publication receipts live under `REGISTRY/ingestion/` and
survive derivative registry repair. Sanitized staging also remains available
for review and recovery. Cancellation does not delete this shared evidence.

## Current limits

This version accepts individual Flywheel acquisition files of type `dicom`,
with one converted NIfTI per selected file. Supported outputs are T1w, T2w,
BOLD, SBRef, and opposite-encoding EPI fieldmaps. Other source formats,
multi-output archives, multiecho conversion, and behavioral-log/scanplan parsers
need explicit adapters. A failure in those cases does not publish a partial
session. An SBRef currently belongs to one BOLD acquisition; a fieldmap pair
may serve several runs. Unassigned SBRefs are not published.

The wizard suggests catalog entries from complete task-name matches, ignoring
case and separators. It does not infer stimulus set/run from BIDS run numbering
or parse a digital scanplan. Multiple variants require an explicit choice.
The wizard validates supplied event files but does not infer their scientific
meaning or invent missing event times. Skull stripping is performed locally;
this version does not consume Flywheel gear output as an alternative source.

Tests cover the state machine, scheduler accounting, privacy routing,
publication recovery, and dcm2bids organization with synthetic images. Live
Flywheel access, cluster conversion, and the site's validator require an
operator-supervised first session before routine use.
