# Bidsify

`nro bidsify` turns Flywheel scanning sessions into published raw BIDS
sessions. You select sessions and answer review questions in a terminal
wizard. Cluster workers download and convert the images in the background, so
the terminal does not need to stay open. nro publishes a session only after you
approve the exact validated files. No AI service is involved.

Ingestion uses the same [worker pool](../orchestration.md) and concurrency
limit as derivative processing. It does not create derivative work items or
request preprocessing. After a session is published, use `nro run` to request
its derivatives.

## Before first use

### Install the dependencies

Install the ingestion extra with `./install --with-bidsify`. For an existing
shared installation, the maintainer runs `./install --maintain --with-bidsify`
and confirms an interactive drain if work is active. The extra adds the
Flywheel SDK, dcm2bids, pydicom, and the Google Drive client. An installation
without the extra can still report ingestion state with `nro status`.

The default conversion commands run dcm2niix from the installation's QuNex
image and skull stripping from its SynthStrip image. The BIDS validator is an
external program that the extra does not install. Put `bids-validator` on the
workers' PATH, or configure an absolute command in the conversion profile.
Worker nodes also need network access to Flywheel and to shared staging.

### Configure Flywheel servers

Each site lists its Flywheel servers and the remote projects nro may read in
the protected `site/site.yml` document. For example, the CLIMBLAB definitions
store contains:

| Server | Host | Remote project scope |
| --- | --- | --- |
| `cni` | `cni.flywheel.io` | `cashain/climblab` |
| `lucas` | `lucascenter.flywheel.io` | `shain/shain1` |

### Store your Flywheel key

Each user stores their own API key for each server they use. The key is read
from a hidden prompt:

```bash
nro fw addkey cni
nro fw list
```

The key may be a token or a matching `HOST:TOKEN` value of the kind `fw login`
accepts. `nro fw addkey SERVER KEY` also works, but the key may then be
recorded in shell history or visible in a process listing, so omit it in
normal use. `nro fw removekey SERVER` removes your key for that server.

Keys live in the ignored `.definition-secrets` directory of the shared
definitions store, separated by Unix user. The store's group can write key
files but not read them, so store maintainers can replace or delete a key
without learning it. Keys never appear in Git, the definitions manifest,
request records, worker scripts, or branch definitions. Changing a key does not
affect derivative freshness.

If you have no key for a selected server, `nro bidsify` asks for one and stores
it before contacting Flywheel. Workers never prompt. A worker that cannot read
the submitting user's key stops and prints the `nro fw addkey SERVER` command to
run. nro uses its own SDK client and does not change your Flywheel CLI login.

Older definitions stores referred to Flywheel credentials through environment
variables. The definitions migration removes those references without reading
or copying the values. After upgrading from that layout, each user runs
`nro fw addkey SERVER` once for each server they use.

### Optional scan plans

A site can also connect a local or Google Drive directory of scan plans,
together with a site-owned parser. See [configure scan plans](scanplans.md) for
the parser API, the TSV fallback, authentication, and reconciliation rules.

## Choose sources and sessions

```bash
nro bidsify ls
nro bidsify ls -f mysite -F group/study
nro bidsify -f mysite -F group/study -P example
nro status -P example
nro bidsify --request REQUEST_ID
nro bidsify --cancel REQUEST_ID
```

`nro bidsify ls` lists the configured server and project pairs and the sessions
your key can see. `-f` and `-F` narrow the listing. Listing does not create a
request, read the registry, or start scheduler coordination.

To start ingestion, name one destination BIDS project with `-P`. The source
Flywheel project, given with `-F`/`--flywheel-project GROUP/PROJECT`, may have a
different name:

```bash
nro bidsify -f cni -F cashain/climblab -P climblab_multisession
```

nro chooses a single source Flywheel project before it lists any sessions.
If the site maps the BIDS project to sources in `project_sources`, only those
sources are candidates, and they may come from different scanning sites.
Without a mapping, every configured server and project pair is a candidate.
`-f`/`--flywheel-server` and `-F` narrow the candidates. An omitted option
falls back to its site default when one is configured. If one candidate
remains, nro uses it. Otherwise it asks you to choose by server and project
name or by number. Selectors that match no configured source are rejected.
nro queries only the chosen project and does not pool sessions from others.

The wizard also offers saved requests for the selected server and destination
project. `--request` resumes one request directly, without choosing a source
or listing remote sessions.

nro hides sessions that already exist in BIDS. To find them, it searches every
raw project under the BIDS root, so a session already stored in a different
project is also hidden. See [Sessions already in BIDS](#sessions-already-in-bids).

## Review and publish a session

A request moves through the following stages. Stages that need your input
release their workers while they wait.

1. **Select sessions.** The wizard first offers unfinished requests. Resume
   them or choose new sessions. nro reports how many known BIDS sessions it
   omitted, then lists the remaining remote sessions by ID and label. Enter
   numbers or `all`. If the server's session rules agree on a BIDS session
   label, nro uses it. Otherwise the wizard asks for one. Enter the participant
   label if you know it, or press Enter to leave it pending. Confirm the
   request. Nothing is downloaded yet.
2. **Prepare images.** Workers inventory the DICOM files, then download and
   convert them in node-local temporary storage. dcm2niix proposes an initial
   type for each acquisition. Anatomical images are skull stripped before they
   reach shared staging. This stage needs no input.
3. **Choose a scan plan.** If a scan-plan source is configured, choose one
   unassigned plan while the images are prepared. nro parses it and later
   compares its acquisition sequence with the prepared images. Fix any
   difference in the source plan itself. If the parser is not implemented,
   nro asks for the TSV fallback instead.
4. **Review acquisitions.** The next invocation shows the converted metadata
   and proposed type of each acquisition. Supported non-derived types are
   accepted automatically. Confirm each proposed ignore or replace it with a
   supported type, then supply the BIDS entities. For task BOLD, pick a
   suggested `TASK/VARIANT` from the [standard event store](../event-files.md),
   or supply an events TSV and confirm that it contains no identifying text.
   For each BOLD acquisition, the wizard also asks for an SBRef and a pair of
   opposite-encoding fieldmaps. Numbered candidates must have compatible
   geometry, encoding, and readout time. You may answer `none`. Type `skip` to
   leave the session for later or `q` to exit. Decisions you have completed
   stay saved.
5. **Organize and validate.** A worker organizes the sanitized images with
   dcm2bids, writes the reviewed reference associations, and runs the
   validator. If validation fails, the session cannot be approved.
6. **Approve.** The next invocation lists the validated files with their
   SHA-256 hashes. Approve those exact files, edit your decisions, or leave the
   session staged. A worker publishes only after approval, and only after it
   rechecks the hashes of both the staged files and any existing files.

### Participant and session labels

Both the participant and session labels must be known before stage 5. A
missing label does not block image preparation. Until both are known, the
images stay in request-specific staging outside BIDS, and no placeholder
subject is published. `nro status` shows a missing label as `(pending)`.

To finish a request whose participant is unknown, run `nro bidsify` without
filters or with `--request`. Supplying the label later does not repeat image
preparation. Each label is saved under the review lease when you enter it. Once resolved, a
label cannot be changed by creating another request or by using `--rebidsify`.

nro identifies a source session by its server and its immutable remote session
ID. The first request for a session reserves one destination BIDS project, even
while the participant is pending. Later requests for that session must keep the
same project and any resolved labels. Re-bidsifying creates a new attempt at
the same destination. Cancelling a request stops its work but keeps the
mapping. There is no command yet to change a destination. Correcting a mistaken
mapping requires a separate, explicit maintenance operation.

When both labels are known, the request reserves its BIDS destination. Filling
in a missing label checks for a conflicting request or an existing
destination.

### Event files

For each events table, nro checks that the `onset` and `duration` columns
exist, that their values are finite, that durations are not negative, and that
onsets fit within the stored run duration when it is known. Negative onsets
are allowed. Rest runs do not need events. You remain responsible for
synchronizing events with the first stored volume, for the condition labels,
and for removing identifying text. See the
[BIDS events specification](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html).

### Fieldmap and SBRef associations

Reviewed fieldmaps receive `B0FieldIdentifier` and `IntendedFor`, and BOLD
runs receive `B0FieldSource`. The `NROReferencePolicy` and `NROSBRef` fields
record the SBRef choice for `func`. Both an assigned SBRef and an explicit
`none` are authoritative, and preprocessing does not replace them with matching
by acquisition time.

### Reviewing alongside other users

Several users can review different sessions at the same time. The wizard
leases only the session it is reviewing, including its approval prompts.
Selecting several sessions does not lock all of them. If another user holds a
session, nro skips it and shows that reviewer's user, host, and process ID.
The lease is released before the wizard moves to the next session, including
after `skip`, `q`, an interrupt, or an error.

A lease renews every 30 seconds while the terminal waits for input and expires
after two minutes without renewal. A terminal that loses its lease cannot
reclaim it or save more decisions; reopen the session to continue. Requests stay
registered when nobody holds a lease. Workers cannot claim a leased request,
and you cannot edit a request while a stage is running.

Each save checks both the active lease and the current record revision. Event
TSV snapshots follow the same checks and use content-addressed filenames, so a
rejected edit cannot overwrite an accepted file. Answers for an acquisition stay
in memory until you finish reviewing that acquisition. Saves take a dedicated,
short-lived ingestion lock. They never take a scheduler database lock, and no
lock is held while you answer prompts. Any change made during review clears a
pending publication approval.

## Sessions already in BIDS

nro treats a remote session as already bidsified when it matches an existing
`sub-…/ses-…` directory. Matching uses publication records and the protected
site's [session rules](#existing-session-rules). nro does not validate data
produced elsewhere, inspect its images, or create publication receipts for it.
It never silently treats an existing destination as safe to replace.

Subject-only intake layouts, such as `climblab/sub-ex123/`, do not count as
bidsified. Sessions organized by subject and session, such as
`climblab_multisession/sub-t20/ses-ex123/`, do. nro does not remove or migrate
the old intake files.

`--rebidsify` includes sessions that already exist in BIDS, whichever tool
produced them, and allows a replacement proposal. Approval is still required. A
re-bidsified external session must keep its matched project, participant, and
session. Ambiguous matches must be resolved before nro registers a replacement
request.

Unfinished nro requests keep their own recovery workflow even when their BIDS
destination exists. They are never offered again as new sessions, even with
`--rebidsify`. When unfinished requests belong to another project, nro prints a
count and points to `nro status`. A remote session with several possible BIDS
destinations, or several remote sessions that match one destination, is shown
as an ambiguous mapping with its candidate paths.

## Status, cancellation, and failures

`nro status` shows unfinished requests in a separate Bidsification section,
and JSON reports include a `bidsification` list. Project, participant, and
session filters apply, and selectors that apply only to derivatives hide the
section. Status never contacts Flywheel or starts a review.

For a failed or interrupted request, the wizard offers to retry, review
decisions, cancel, or leave it as is. `nro bidsify --cancel REQUEST_ID` also
cancels queued or partly reviewed work and asks the worker to stop an executing
stage. nro marks a request interrupted only when its worker is confirmed dead.
A stale heartbeat alone does not prove that the work stopped.

## Options

| Option | Meaning |
| --- | --- |
| `ls` | List configured servers, projects, and accessible remote sessions without creating work. |
| `-f`, `--flywheel-server NAME` | Configured server, such as `cni` or `lucas`. |
| `-P`, `--project NAME` | One destination BIDS project. |
| `-F`, `--flywheel-project GROUP/PROJECT` | One configured source Flywheel project for new sessions. |
| `-p`, `--participant LABEL ...` | Filter saved requests by participant label. Accepts globs and `re:REGEX`, matched against saved labels because new participants are not yet in BIDS. A single exact label also becomes the default participant for a new mapping. Remote participant labels are not assumed to be BIDS labels. |
| `--session ID ...` | Filter by remote session IDs, not BIDS session labels. |
| `--request ID` | Resume one saved request without listing remote sessions. |
| `--cancel ID` | Cancel one saved request. An executing stage stops once its worker acknowledges the cancellation. |
| `--config PATH` | Use another complete conversion profile for new requests. |
| `--rebidsify` | Include sessions already in BIDS and allow a replacement proposal. Approval is still required. |
| `--no-submit` | Do not submit new workers. Existing workers may still claim queued stages. |
| Slurm options | `--partition`, `--account`, the `--scheduler-*` options, and the `--worker-*` options. Worker options override the ingestion profile. See [Slurm options](index.md#slurm-options). |

`--request` and `--cancel` cannot be combined. The BIDS root always comes from
the global site configuration, so one invocation cannot publish to a different
root.

## Configuration

Ingestion settings come from two places. The protected site document holds the
sources and routing that every development branch shares. A conversion profile
holds settings that a branch may change.

### Protected site settings

The shared `DEFINITIONS/site/site.yml` document lists the Flywheel servers,
maps destination projects to source projects, and holds scan-plan locations,
session rules, and event lookup rules. Private Flywheel keys are stored
outside this tracked document.

| Protected site key | Purpose |
| --- | --- |
| `servers` | Named entries with a `host` and a `projects` list. Keys are managed by `nro fw`. |
| `project_sources` | Maps BIDS project names to nonempty lists of `{server, project}` sources. Each source must appear in that server's `projects` list. Duplicate pairs are rejected. |
| `scanplans.location`, `scanplans.credential_env` | Shared scan-plan source and its authentication variable, described in [configure scan plans](scanplans.md). A null location turns scan plans off. |
| `event_rules` | Extra `task` and absolute glob `pattern` pairs for event files outside the catalog. All matches are shown, and an ambiguous candidate is never chosen automatically. |
| `session_rules` | Explicit rules that recognize existing raw BIDS subjects and sessions from remote identities. |

### Default source projects

For a BIDS dataset that receives sessions from both sites, the mapping can be:

```yaml
project_sources:
  climblab_multisession:
    - server: cni
      project: cashain/climblab
    - server: lucas
      project: shain/shain1
```

`nro bidsify -P climblab_multisession` then offers both sources, and adding
`-f cni` selects the CNI source without a prompt.

Mappings are specific to each site, and new stores leave them empty. Several
BIDS datasets may take different sessions from the same source project, but
each source session has only one destination. Choosing a source in the wizard
does not edit the protected site document. To keep a choice as the default,
use `nro site set`, or edit and validate the document through site maintenance.

### Existing-session rules

A session rule tells nro how a remote session's identity maps to BIDS labels.
Each rule applies to one configured `server` and `remote_project`. Its `match`
mapping holds full-match regular expressions for one or more remote fields:
`id`, `label` (the session label), or `subject_code`. Every pattern must match,
and a named group that appears more than once must capture the same value each
time. The named groups fill the BIDS label templates.

For example, this lab's CNI session label `31381` corresponds to exam
`ex31381`. This rule recognizes that session under any participant:

```yaml
session_rules:
  - server: cni
    remote_project: cashain/climblab
    match: {label: '(?:ex)?(?P<exam>[0-9]+)'}
    participant: null
    session: 'ex{exam}'
```

`participant: null` looks for the exact session label under any participant. A
rule that supplies both labels matches only that participant and session. A
rule without a session label cannot match a completed destination. The search
skips `derivatives`, `sourcedata`, and directories that are not valid nro
project names, including hidden and underscore-prefixed directories.

nro also uses these rules to propose labels for new work. When all matching
rules that give a session label agree, nro uses that label. In the example,
selecting remote session `31381` supplies `ses-ex31381` without a prompt. If
the rules conflict or none match, you enter the label during review. nro does
not use fuzzy matching or sanitize labels. Labels that would not be valid BIDS
identifiers do not match. The wizard also does not copy an unconfigured remote
label, strip punctuation, or infer a stable participant from an exam
identifier. The proposed destination is shown before you confirm the request.

New stores contain no naming rules. Add rules for each server's naming
convention, then run `nro def validate`. CLIMBLAB's CNI rules do not apply to
Lucas. Published sessions always use a subject and session hierarchy.

### Conversion profile

The default conversion profile is `DEFINITIONS/bidsify/main.yml`, and
`--config FILE` selects another complete profile. Unknown keys and missing
required keys are rejected. Each request saves its resolved profile, so later
profile edits affect only new requests. If the selected scan-plan file changes,
nro reparses it when the request resumes.

| Conversion profile key | Purpose |
| --- | --- |
| `staging` | Absolute shared directory outside BIDS. `null` uses `WORK/bidsify`. |
| `dcm2niix` | Command argument list. `null` uses the configured QuNex container command. |
| `synthstrip` | Command argument list. `null` uses the configured SynthStrip container command. |
| `validator` | Command argument list. nro appends the staged dataset path. A nonzero exit blocks publication. |
| `memory_gb`, `cpus`, `hours` | Worker allocation. The defaults are 32 GiB, 2 CPUs, and 12 hours. |
| `concurrency` | Requested shared limit. Defaults to `execution.concurrency`. `nro site set execution.concurrency=N` changes the durable site limit and passes it to a running scheduler. |
| `protocols` | Ordered rules with `pattern`, `datatype`, and `suffix` that refine types. After dcm2niix supplies a supported `BidsGuess`, the first rule that matches `SeriesDescription`, `ProtocolName`, or `SequenceName` applies. |
| `scanplans.parser` | Parser implementation, which a branch may change. Null selects the packaged stub. |

Commands run without a shell. In the conversion and skull-stripping commands,
`{staging}` expands to the acquisition's working directory, and the default
container commands bind that directory. A custom wrapper must keep raw anatomy
out of shared staging and must exit nonzero on failure. A validator wrapper must
actually validate the dataset, because approval relies on its exit status.

The standard event catalog is always `DEFINITIONS/events` and is not a profile
setting. Each request records its resolved path as runtime metadata, along with
the selected event snapshots.

### Acquisition type proposals

During conversion, dcm2niix derives a `BidsGuess` from the DICOM metadata. nro
accepts supported non-derived guesses, then applies the `protocols` rules for
known refinements, such as telling an SBRef apart from BOLD data. Missing,
derived, discarded, and unsupported guesses are proposed as ignored. The wizard
reports each of these and asks you to confirm or replace it. An override runs
preparation again before conversion continues.

### Vendor metadata enrichment

Preparation keeps valid dcm2niix metadata and fills approved missing fields
from bounded vendor protocol data. It never silently replaces a converter
value. A conflict among converter, DICOM, series, or site values blocks
publication. An optional field that cannot be derived produces a warning and is
left out.

For GE data, fractional `NEX` supplies the phase-direction `PartialFourier`
when fractional echo is absent. `PHASEFOV` describes spatial coverage and is
not used in that calculation. A fractional-echo acquisition gets no derived
fraction, because the exported metadata do not establish its readout fraction.

GE `THREEDGW` supplies `GradientCorrectionMode` when present. Otherwise, a
matching [hardware profile](../definitions.md#gradient-unwarping-hardware) may
assert the mode. The published values are `2D`, `3D`, and `none`, and unknown
modes are left out. Private preparation records keep bounded provenance for
each field and the reason a value could not be derived, without keeping vendor
protocol text.

## Staging, privacy, and replacement

### Where files live

Raw archives, DICOMs, initial converted images, and skull-stripping
intermediates stay in node-local temporary storage:

```text
/tmp/nro/bidsify/SERVER/REMOTE_SESSION_ID/REQUEST_ID/ACQUISITION_ID/
```

Shared staging is organized as `STAGING/REQUEST_ID/`. It holds sanitized helper
images, reviewed events, generated dcm2bids configurations, and the proposed
`bids/` tree. Raw acquisition files never enter shared staging. Accepted helper
images and an allowlist of metadata enter it only after conversion,
sanitization, and skull stripping where needed.

Temporary staging rejects symlinks, world access, and ownership by anyone other
than the executing user or the shared group. If several users need access,
administrators must provision a suitable shared group.

### Cleanup

Successful preparation and handled failures remove the raw acquisition files.
Cancellation runs cleanup when the process can handle the termination signal. A
forced kill or node failure can leave raw data in that node's `/tmp`, and a
later attempt on another node cannot clean it. The saved request records the
worker hostname, but the site must still manage
temporary cleanup. `/tmp` does not guarantee a retention period, exclusion from
backups, or secure erasure. Check those properties with the cluster
administrator.

### Identifying information

Archive members get opaque temporary filenames. NIfTI text fields and
extensions are cleared, and JSON output keeps only an allowlist of acquisition
metadata. This reduces identifying metadata but does not certify DICOM
de-identification. Raw converter and provider diagnostics are suppressed
because they can contain identifying labels. Safe failure messages appear in
`REGISTRY/ingestion/REQUEST_ID.log`. Unexpected failures report only their
exception type, without the original message.

### Publication and replacement

Publication copies the approved session to a hidden sibling directory on the
destination filesystem, verifies it, and renames it into place, so a partly
copied session never appears at the destination. Other sessions are untouched.
An existing raw `dataset_description.json` is preserved, and a missing one is
created.

Replacing a session uses a durable transaction journal and two renames on the
same filesystem. The old session moves to a hidden backup, then the complete
new session moves into place. If execution stops between the renames, the next
attempt uses the approved file hashes to either finish publication or restore
the old session from the backup. The backup is removed only after a durable
receipt records the new session.

Replacement requires that no nro derivative attempts be active in the
destination project when it commits. nro cannot coordinate programs outside
its registry, so stop external readers before approving a replacement.
Publishing again does not rebuild derivatives automatically.

Records, logs, and publication receipts live under `REGISTRY/ingestion/` and
survive derivative registry repair. Sanitized staging also stays available for
review and recovery. Cancellation does not delete either.

## Current limits

This version accepts individual Flywheel acquisition files of type `dicom` and
produces one converted NIfTI per selected file. It supports T1w, T2w, BOLD,
SBRef, and opposite-encoding EPI fieldmaps. Other source formats, archives with
several outputs, multiecho conversion, and behavioral-log parsers need explicit
adapters. Scan-plan input goes through the site-parser contract. In all of
these cases, a failure never publishes a partial session.

An SBRef belongs to one BOLD acquisition, while a fieldmap pair may serve
several runs. Unassigned SBRefs are not published.

The wizard suggests catalog event files only when the whole task name matches,
ignoring case and separators. It does not infer the stimulus set or run from
the BIDS run number. When a scan plan aligns exactly, it supplies the task ID,
and the wizard then looks up event-file candidates for that task. If several
variants match, you must choose one. The wizard validates event files but does
not interpret their scientific meaning or invent missing event times.

Skull stripping always runs locally. This version does not use Flywheel gear
output as an alternative source.

Tests cover the request state machine, scheduler accounting, privacy routing,
publication recovery, and dcm2bids organization with synthetic images. Live
Flywheel access, cluster conversion, and the site's validator need an
operator-supervised first session before routine use.
