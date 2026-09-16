# Configure scan plans

Sites may connect bidsification to a directory of scan plans. The directory
may be local or stored in Google Drive. Each site supplies a parser because
scan-plan formats are not standardized. nro defines the parser output and
handles source selection, validation, and reconciliation.

This setup belongs in the definitions system. Parser code is trusted code:
`nro definitions validate` imports it, and `nro bidsify` runs it as the current
user. Review parser changes through the definitions store's normal version
control process.

## Configure the source

Set the source and optional credential variable in the protected
`site/site.yml` document:

```yaml
bidsify:
  scanplans:
    location: /data/lab/scanplans
    credential_env: null
```

Select the parser in `bidsify/main.yml`:

```yaml
scanplans:
  parser: scanplans/parser.py
```

The source is site-wide. Development branches may change parser code or select
another parser, but they cannot redirect the shared scan-plan source or its
authentication setting.

`location` accepts an absolute directory, a path relative to the definitions
store, or a Google Drive folder URL. A local source includes every regular file
below the directory, recursively. Symbolic links are ignored.
Keep source plans outside `DEFINITIONS/scanplans`; that directory contains the
version-controlled parser only.

For Google Drive, install the bidsification dependencies and use one of these
authentication methods:

* Set `credential_env` to the name of an environment variable whose value is
  an authorized-user or service-account credential JSON file.
* Leave `credential_env` null and configure Google application-default
  credentials for the account running `nro bidsify`.

nro requests read-only Drive access, traverses the configured folder, and
downloads only the selected file into a temporary directory for parsing.
Google-native documents are exported as DOCX. Credentials and source contents
are not copied into ingestion records.

Set `location: null` to use the ordinary acquisition-review wizard without a
scan plan. The parser may remain installed.

## Implement the parser

The definitions starter includes `scanplans/parser.py`. Its stub raises
`NotImplementedError`. Replace its body with a parser for the site's source
format while retaining this interface:

```python
from pathlib import Path

from nro.bidsify.scanplans import ScanPlan, ScanPlanRow


def parse_scanplan(source: Path) -> ScanPlan:
    return ScanPlan(
        rows=(
            ScanPlanRow(ordinal=1, acquisition_type="localizer", include=False),
            ScanPlanRow(
                ordinal=2,
                acquisition_type="bold",
                phase_encoding="j",
                task="language",
            ),
        ),
    )
```

The input file may use any format. The return value must be a `ScanPlan` whose
`rows` are in scanner acquisition order. Each `ScanPlanRow` has these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `ordinal` | positive `int` | Stable row order within this plan. Values must be unique and ascending. |
| `acquisition_type` | `T1w`, `T2w`, `bold`, `sbref`, `fmap`, `localizer`, `shim`, or `other` | Normalized type used for sequence alignment. |
| `phase_encoding` | `i`, `i-`, `j`, `j-`, `k`, `k-`, or `None` | BIDS direction. `None` leaves this field out of matching. |
| `include` | `bool` | Whether the aligned acquisition should be published. |
| `task` | `str` or `None` | BIDS task label transferred to an aligned BOLD acquisition. |

`task` is valid only for `bold` rows. Source identity fields and event-file
assignments do not belong in this output. The BIDS destination selected in the
wizard remains authoritative.

Run `nro definitions validate` after changing the parser. Validation checks
that the module loads and exports a callable `parse_scanplan`. Parser output is
validated when a source file is selected.

## Machine-readable fallback

If the configured parser raises `NotImplementedError`, `nro bidsify` asks for a
tab-separated scan plan before review can continue. The TSV must have this
exact header:

```text
ordinal	acquisition_type	phase_encoding	include	task
```

Use `true` or `false` for `include`. Empty optional fields are allowed. The same
field rules as the Python API apply. This fallback lets a site use scan plans
before it has automated its parser; it does not infer missing rows.

## Reconciliation

Image inspection and preparation are submitted before interactive scan-plan
selection, so conversion can run while the operator works on metadata. After
preparation, nro aligns the plan and image sequences. The alignment uses type,
phase encoding, and order. Task names do not affect matching because DICOM
metadata do not establish them.

Review proceeds only when the alignment is unique and has no differences.
Otherwise, nro reports missing, extra, or conflicting rows. Edit the source
scan plan and rerun `nro bidsify`; nro notices the changed source revision and
parses it again. The wizard does not store a private correction layer. Once the
sequences align, task identifiers transfer to their corresponding BOLD
acquisitions. Event-file selection then follows the ordinary bidsification
review, using the task ID to find candidates in the event definitions store.
Excluded rows are omitted from publication.

Changes to the selected source or parser are detected when the request resumes
and clear any pending publication approval.

One source file can be assigned to only one ingestion session within a
configured source location. This prevents concurrent requests from silently
using the same plan.
