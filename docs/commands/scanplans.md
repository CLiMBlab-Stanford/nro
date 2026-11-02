# Configure scan plans

A site can connect [bidsification](bidsify.md) to a directory of scan plans so
that task names and include decisions come from the plan written for each
session. The directory may be local or in Google Drive. Scan-plan formats are
not standardized, so each site supplies a parser. nro defines what the parser
returns and handles source selection, validation, and reconciliation with the
prepared images.

The parser belongs to the definitions store, and it is trusted code.
`nro def validate` imports it, and `nro bidsify` runs it as the current user.
Review parser changes through the store's normal version-control process.

## Configure the source

Set the source and an optional credential variable in the protected
`site/site.yml` document:

```yaml
bidsify:
  scanplans:
    location: /data/lab/scanplans
    credential_env: null
```

Select the parser in the conversion profile, `bidsify/main.yml`:

```yaml
scanplans:
  parser: scanplans/parser.py
```

The source is shared by the whole site. A development branch may change the
parser code or select another parser, but it cannot redirect the scan-plan
source or change its authentication setting.

`location` accepts an absolute directory, a path relative to the definitions
store, or a Google Drive folder URL. A local source includes every regular file
below the directory, recursively, and ignores symbolic links. Keep the plans
themselves outside `DEFINITIONS/scanplans`, which holds only the
version-controlled parser.

Set `location: null` to review acquisitions without a scan plan. The parser can
stay installed.

### Google Drive

For a Google Drive source, install the bidsification dependencies and choose
one way to authenticate:

* Set `credential_env` to the name of an environment variable whose value is
  the path to an authorized-user or service-account credential JSON file.
* Leave `credential_env` null and configure Google application-default
  credentials for the account that runs `nro bidsify`.

nro requests read-only Drive access, walks the configured folder, and downloads
only the selected file into a temporary directory for parsing. Google-native
documents are exported as DOCX. Credentials and plan contents are not copied
into ingestion records.

## Write the parser

The definitions starter includes `scanplans/parser.py`, whose stub raises
`NotImplementedError`. Replace the body with a parser for your site's format
and keep this interface:

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

The input file may be in any format. The function must return a `ScanPlan`
whose `rows` are in scanner acquisition order. Each `ScanPlanRow` has these
fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `ordinal` | positive `int` | Row order within the plan. Values must be unique and ascending. |
| `acquisition_type` | `T1w`, `T2w`, `bold`, `sbref`, `fmap`, `localizer`, `shim`, or `other` | Normalized type used to align the plan with the images. |
| `phase_encoding` | `i`, `i-`, `j`, `j-`, `k`, `k-`, or `None` | BIDS phase-encoding direction. `None` leaves it out of matching. |
| `include` | `bool` | Whether the matching acquisition should be published. |
| `task` | `str` or `None` | BIDS task label given to the matching BOLD acquisition. Allowed only on `bold` rows. |

Do not return source identity fields or event-file assignments. The BIDS
destination chosen in the wizard always takes precedence.

Run `nro def validate` after changing the parser. Validation checks that the
module loads and exports a callable `parse_scanplan`. The parser's output is
validated later, when a plan is selected.

## TSV fallback

While the parser still raises `NotImplementedError`, `nro bidsify` asks for a
tab-separated scan plan before review can continue. This lets a site use scan
plans before it has written a parser. The TSV must have exactly this header:

```text
ordinal	acquisition_type	phase_encoding	include	task
```

Write `true` or `false` for `include`. Optional fields may be empty. The field
rules are the same as for the Python API, and nro does not infer missing rows.

## Reconciliation

nro submits image inspection and preparation before it asks you to pick a scan
plan, so conversion runs while you work on metadata. After preparation, nro
aligns the plan with the image sequence using acquisition type, phase encoding,
and order. Task names are not used for matching because DICOM metadata do not
establish them.

Review continues only when the alignment is unique and has no differences.
Otherwise nro reports the missing, extra, or conflicting rows. Edit the source
plan and run `nro bidsify` again. nro notices the new revision and parses it
again. The wizard does not keep its own correction layer.

Once the sequences align, each task ID moves to its matching BOLD acquisition,
and rows with `include` false are left out of publication. Event-file selection
then continues in the ordinary review, using the task ID to find candidates in
the [standard event catalog](../event-files.md).

If the selected plan or the parser changes, nro detects it when the request
resumes and clears any pending publication approval.

Within one source location, each plan file can be assigned to only one
ingestion session. This keeps concurrent requests from using the same plan.
