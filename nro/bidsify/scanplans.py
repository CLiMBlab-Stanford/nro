"""Load site scan plans through a typed parser and configurable file source."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
import tokenize
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Literal, Sequence

AcquisitionType = Literal["T1w", "T2w", "bold", "sbref", "fmap", "localizer", "shim", "other"]
ACQUISITION_TYPES = frozenset({"T1w", "T2w", "bold", "sbref", "fmap", "localizer", "shim", "other"})
_DRIVE_FOLDER = re.compile(r"^https://drive\.google\.com/drive/(?:u/\d+/)?folders/")
_DRIVE_FOLDER_ID = re.compile(r"/folders/([A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class ScanPlanRow:
    """Describe one acquisition in scanner order using nro's shared vocabulary.

    A task ID annotates an aligned BOLD acquisition. It does not participate in
    sequence matching because DICOM metadata does not establish it.
    """

    ordinal: int
    acquisition_type: AcquisitionType
    phase_encoding: str | None = None
    include: bool = True
    task: str | None = None


@dataclass(frozen=True)
class ScanPlan:
    """Return the normalized, ordered acquisitions extracted from one source plan."""

    rows: tuple[ScanPlanRow, ...]


@dataclass(frozen=True)
class ScanPlanFile:
    """Identify one selectable file without embedding its potentially sensitive content."""

    id: str
    name: str
    revision: str
    local_path: Path | None = None
    mime_type: str | None = None


def parse_scanplan(source: Path) -> ScanPlan:
    """Define the site-parser interface and fail when no implementation is configured.

    A site parser accepts any file format it understands and returns a
    :class:`ScanPlan`. Parser implementations belong in the selected definitions
    store and may import :class:`ScanPlan` and :class:`ScanPlanRow` from this
    module.
    """
    del source
    raise NotImplementedError("No site scan-plan parser is configured")


def validate_scanplan(plan: ScanPlan) -> ScanPlan:
    """Validate parser output before it can influence an ingestion request."""
    if not isinstance(plan, ScanPlan):
        raise TypeError("A scan-plan parser must return nro.bidsify.scanplans.ScanPlan")
    if not plan.rows:
        raise ValueError("A scan plan must contain at least one acquisition row")
    ordinals = []
    for row in plan.rows:
        if not isinstance(row, ScanPlanRow):
            raise TypeError("ScanPlan.rows must contain ScanPlanRow values")
        if type(row.ordinal) is not int or row.ordinal < 1:
            raise ValueError("Scan-plan ordinals must be positive integers")
        if row.acquisition_type not in ACQUISITION_TYPES:
            raise ValueError(f"Unsupported scan-plan acquisition type: {row.acquisition_type}")
        if row.phase_encoding not in {None, "i", "i-", "j", "j-", "k", "k-"}:
            raise ValueError("Scan-plan phase_encoding must use a BIDS axis direction")
        if type(row.include) is not bool:
            raise ValueError("Scan-plan include values must be boolean")
        if row.acquisition_type != "bold" and row.task is not None:
            raise ValueError("Only BOLD rows may define task")
        for label, value in (
            ("phase_encoding", row.phase_encoding),
            ("task", row.task),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Scan-plan {label} must be a nonempty string or null")
        if row.task is not None and not re.fullmatch(r"[A-Za-z0-9]+", row.task):
            raise ValueError("Scan-plan task values must be valid BIDS labels")
        ordinals.append(row.ordinal)
    if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
        raise ValueError("Scan-plan rows must have unique ascending ordinals")
    return plan


def scanplan_document(plan: ScanPlan) -> dict[str, object]:
    """Convert validated parser output into JSON-compatible request metadata."""
    plan = validate_scanplan(plan)
    return {"rows": [asdict(row) for row in plan.rows]}


TSV_COLUMNS = (
    "ordinal",
    "acquisition_type",
    "phase_encoding",
    "include",
    "task",
)


def read_scanplan_tsv(path: Path) -> ScanPlan:
    """Read the documented machine-readable fallback format."""
    with path.expanduser().open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TSV_COLUMNS:
            raise ValueError(
                "Machine-readable scan plan must have columns: " + ", ".join(TSV_COLUMNS)
            )
        rows = []
        for source in reader:
            include = source["include"].strip().lower()
            if include not in {"true", "false"}:
                raise ValueError("Scan-plan include values must be true or false")
            rows.append(
                ScanPlanRow(
                    ordinal=int(source["ordinal"]),
                    acquisition_type=source["acquisition_type"],  # type: ignore[arg-type]
                    phase_encoding=source["phase_encoding"].strip() or None,
                    include=include == "true",
                    task=source["task"].strip() or None,
                )
            )
    return validate_scanplan(ScanPlan(tuple(rows)))


def load_parser(path: Path | None) -> ModuleType | None:
    """Load and validate a definitions-store parser module without parsing source data."""
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Scan-plan parser must be a regular Python file: {path}")
    name = "_nro_site_scanplan_" + hashlib.sha256(str(path).encode()).hexdigest()
    module = ModuleType(name)
    module.__file__ = str(path)
    try:
        with tokenize.open(path) as stream:
            code = compile(stream.read(), str(path), "exec")
        exec(code, module.__dict__)
    except Exception as error:
        raise ValueError(f"Cannot import scan-plan parser ({type(error).__name__})") from None
    if not callable(getattr(module, "parse_scanplan", None)):
        raise ValueError("Site scan-plan parser must define parse_scanplan(source: Path)")
    return module


def run_parser(source: Path, parser: Path | None) -> ScanPlan:
    """Run the configured parser or the public unimplemented stub."""
    module = load_parser(parser)
    result = parse_scanplan(source) if module is None else module.parse_scanplan(source)
    return validate_scanplan(result)


def local_files(directory: Path) -> tuple[ScanPlanFile, ...]:
    """List every regular file below a configured local scan-plan directory."""
    root = directory.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Scan-plan location is not a regular directory: {root}")
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            stat = path.stat()
            relative = str(path.relative_to(root))
            revision = hashlib.sha256(
                f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
            ).hexdigest()
            identity = "local-" + hashlib.sha256(relative.encode()).hexdigest()
            result.append(ScanPlanFile(identity, relative, revision, path))
    return tuple(result)


def _drive_folder_id(url: str) -> str:
    match = _DRIVE_FOLDER_ID.search(url)
    if match is None:
        raise ValueError("Google Drive scan-plan location must be a folder URL")
    return match[1]


def _drive_service(credential_env: str | None):
    try:
        import google.auth
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        raise ValueError(
            "Google Drive scan plans require the bidsify dependencies; rerun ./install --with-bidsify"
        ) from None
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    if credential_env and not os.environ.get(credential_env):
        raise ValueError(f"Set {credential_env} to a Google credential JSON file")
    credential_path = os.environ.get(credential_env, "") if credential_env else ""
    try:
        if credential_path:
            path = Path(credential_path).expanduser()
            if not path.is_file():
                raise ValueError(f"{credential_env} does not name a credential file")
            try:
                credentials = Credentials.from_authorized_user_file(str(path), scopes)
            except ValueError:
                credentials, _ = google.auth.load_credentials_from_file(str(path), scopes=scopes)
        else:
            credentials, _ = google.auth.default(scopes=scopes)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except ValueError:
        raise
    except Exception:
        raise ValueError(
            "Google Drive authentication failed; configure application-default credentials "
            "or the scanplans credential environment variable"
        ) from None


def drive_files(url: str, *, credential_env: str | None = None) -> tuple[ScanPlanFile, ...]:
    """List files recursively below a Google Drive folder using read-only credentials."""
    service = _drive_service(credential_env)
    result: list[ScanPlanFile] = []
    pending = [(_drive_folder_id(url), "")]
    while pending:
        folder, prefix = pending.pop()
        token = None
        while True:
            try:
                response = (
                    service.files()
                    .list(
                        q=f"'{folder}' in parents and trashed = false",
                        fields="nextPageToken,files(id,name,mimeType,modifiedTime,md5Checksum)",
                        pageToken=token,
                        pageSize=1000,
                    )
                    .execute()
                )
            except Exception:
                raise ValueError(
                    "Google Drive scan-plan listing failed; check folder access and credentials"
                ) from None
            for item in response.get("files", []):
                name = f"{prefix}/{item['name']}" if prefix else item["name"]
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    pending.append((item["id"], name))
                    continue
                revision = item.get("md5Checksum") or item.get("modifiedTime") or item["id"]
                result.append(
                    ScanPlanFile(item["id"], name, str(revision), mime_type=item["mimeType"])
                )
            token = response.get("nextPageToken")
            if not token:
                break
    return tuple(sorted(result, key=lambda item: item.name))


def scanplan_files(config: dict) -> tuple[ScanPlanFile, ...]:
    """List files from the profile's local directory or Google Drive folder."""
    settings = config["scanplans"]
    location = settings["location"]
    if location is None:
        return ()
    if _DRIVE_FOLDER.match(location):
        return drive_files(location, credential_env=settings["credential_env"])
    return local_files(Path(location))


@contextmanager
def materialize_scanplan(config: dict, candidate: ScanPlanFile) -> Iterator[Path]:
    """Provide one selected source as a local file and remove cloud downloads afterward."""
    if candidate.local_path is not None:
        yield candidate.local_path
        return
    service = _drive_service(config["scanplans"]["credential_env"])
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        raise ValueError("Google Drive support is unavailable") from None
    suffix = Path(candidate.name).suffix
    with tempfile.TemporaryDirectory(prefix="nro-scanplan-") as temporary:
        target = Path(temporary) / ("scanplan" + suffix)
        mime = candidate.mime_type or ""
        try:
            if mime.startswith("application/vnd.google-apps."):
                exports = {
                    "application/vnd.google-apps.document": (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        ".docx",
                    ),
                    "application/vnd.google-apps.spreadsheet": (
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        ".xlsx",
                    ),
                    "application/vnd.google-apps.presentation": (
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        ".pptx",
                    ),
                }
                if mime not in exports:
                    raise ValueError(f"Unsupported Google-native scan-plan type: {mime}")
                export_type, extension = exports[mime]
                request = service.files().export_media(
                    fileId=candidate.id,
                    mimeType=export_type,
                )
                target = target.with_suffix(extension)
            else:
                request = service.files().get_media(fileId=candidate.id)
            with target.open("wb") as stream:
                downloader = MediaIoBaseDownload(stream, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                "Google Drive scan-plan download failed; check file access and credentials"
            ) from None
        yield target


def choose_scanplan(files: Sequence[ScanPlanFile], answer: str) -> ScanPlanFile:
    """Resolve one displayed scan-plan number without accepting an implicit first match."""
    if not answer.isdigit() or not 1 <= int(answer) <= len(files):
        raise ValueError("Select one displayed scan-plan number")
    return files[int(answer) - 1]


def parser_identity(path: Path | None) -> dict[str, str] | None:
    """Record the configured parser's relative identity without retaining its source."""
    if path is None:
        return None
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_selection(config: dict, candidate: ScanPlanFile) -> dict[str, object]:
    """Parse one selected source and return sanitized, hash-bound request metadata."""
    parser_value = config["scanplans"]["parser"]
    parser = Path(parser_value) if parser_value is not None else None
    with materialize_scanplan(config, candidate) as source:
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        plan = run_parser(source, parser)
    return {
        "source": {
            "id": candidate.id,
            "revision": candidate.revision,
            "sha256": content_hash,
        },
        "parser": parser_identity(parser),
        "plan": scanplan_document(plan),
        "alignment": None,
    }


def manual_selection(path: Path, *, candidate: ScanPlanFile | None = None) -> dict[str, object]:
    """Parse a user-authored TSV when the site parser is unavailable."""
    path = path.expanduser().resolve()
    plan = read_scanplan_tsv(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = hashlib.sha256(str(path).encode()).hexdigest()
    return {
        "source": {
            "id": candidate.id if candidate is not None else "manual-" + identity,
            "revision": candidate.revision if candidate is not None else digest,
            "sha256": digest,
        },
        "parser": {"path": "machine-readable-tsv", "sha256": digest},
        "plan": scanplan_document(plan),
        "alignment": None,
    }


def _prepared_type(item: dict) -> AcquisitionType:
    kind = item.get("datatype"), item.get("suffix")
    known = {
        ("anat", "T1w"): "T1w",
        ("anat", "T2w"): "T2w",
        ("func", "bold"): "bold",
        ("func", "sbref"): "sbref",
        ("fmap", "epi"): "fmap",
    }
    if kind in known:
        return known[kind]  # type: ignore[return-value]
    evidence = " ".join(item.get("classification", {}).get("bids_guess") or ()).lower()
    protocol = str(item.get("metadata", {}).get("ProtocolName", "")).lower()
    text = evidence + " " + protocol
    if "loc" in text or "scout" in text:
        return "localizer"
    if "shim" in text:
        return "shim"
    return "other"


def _prepared_rows(acquisitions: Sequence[dict]) -> list[tuple[int, dict, AcquisitionType]]:
    indexed = list(enumerate(acquisitions))

    def order(value):
        index, item = value
        metadata = item.get("metadata", {})
        series = metadata.get("SeriesNumber")
        try:
            series = float(series)
        except (TypeError, ValueError):
            series = float("inf")
        return series, str(metadata.get("AcquisitionTime", "")), index

    return [(index, item, _prepared_type(item)) for index, item in sorted(indexed, key=order)]


def align_scanplan(plan: dict, acquisitions: Sequence[dict]) -> dict[str, object]:
    """Align normalized plan rows to prepared acquisitions and describe every difference.

    Matching uses acquisition type and phase-encoding direction. Task labels
    never affect alignment because prepared DICOM metadata does not establish
    them.
    """
    planned = plan["rows"]
    prepared = _prepared_rows(acquisitions)
    gap = 2

    def substitution(row, actual):
        if row["acquisition_type"] != actual[2]:
            return 5
        wanted = row.get("phase_encoding")
        found = actual[1].get("metadata", {}).get("PhaseEncodingDirection")
        return 0 if wanted is None or wanted == found else 3

    costs = [[0] * (len(prepared) + 1) for _ in range(len(planned) + 1)]
    paths = [[1] * (len(prepared) + 1) for _ in range(len(planned) + 1)]
    for i in range(1, len(planned) + 1):
        costs[i][0] = i * gap
    for j in range(1, len(prepared) + 1):
        costs[0][j] = j * gap
    for i in range(1, len(planned) + 1):
        for j in range(1, len(prepared) + 1):
            options = (
                (
                    costs[i - 1][j - 1] + substitution(planned[i - 1], prepared[j - 1]),
                    paths[i - 1][j - 1],
                ),
                (costs[i - 1][j] + gap, paths[i - 1][j]),
                (costs[i][j - 1] + gap, paths[i][j - 1]),
            )
            best = min(value for value, _ in options)
            costs[i][j] = best
            paths[i][j] = min(2, sum(count for value, count in options if value == best))
    rows = []
    i, j = len(planned), len(prepared)
    while i or j:
        diagonal = (
            costs[i - 1][j - 1] + substitution(planned[i - 1], prepared[j - 1]) if i and j else None
        )
        if i and j and diagonal == costs[i][j]:
            expected = planned[i - 1]
            _, item, actual = prepared[j - 1]
            phase = item.get("metadata", {}).get("PhaseEncodingDirection")
            status = "match"
            if expected["acquisition_type"] != actual:
                status = "type_mismatch"
            elif expected.get("phase_encoding") not in {None, phase}:
                status = "phase_encoding_mismatch"
            rows.append(
                {
                    "plan_ordinal": expected["ordinal"],
                    "acquisition_id": item["id"],
                    "plan_type": expected["acquisition_type"],
                    "prepared_type": actual,
                    "status": status,
                }
            )
            i -= 1
            j -= 1
        elif i and costs[i - 1][j] + gap == costs[i][j]:
            expected = planned[i - 1]
            rows.append(
                {
                    "plan_ordinal": expected["ordinal"],
                    "acquisition_id": None,
                    "plan_type": expected["acquisition_type"],
                    "prepared_type": None,
                    "status": "missing_prepared_acquisition",
                }
            )
            i -= 1
        else:
            _, item, actual = prepared[j - 1]
            rows.append(
                {
                    "plan_ordinal": None,
                    "acquisition_id": item["id"],
                    "plan_type": None,
                    "prepared_type": actual,
                    "status": "unplanned_acquisition",
                }
            )
            j -= 1
    rows.reverse()
    complete = (
        costs[-1][-1] == 0
        and paths[-1][-1] == 1
        and len(planned) == len(prepared)
        and all(row["status"] == "match" for row in rows)
    )
    return {
        "complete": complete,
        "distance": costs[-1][-1],
        "unique": paths[-1][-1] == 1,
        "rows": rows,
    }


def apply_scanplan(record: dict) -> dict:
    """Apply annotations only after the source plan matches every prepared acquisition."""
    selected = record.get("scanplan")
    if not selected:
        return record
    alignment = align_scanplan(selected["plan"], record["acquisitions"])
    selected["alignment"] = alignment
    if not alignment["complete"]:
        return record
    by_id = {
        row["acquisition_id"]: plan_row
        for row, plan_row in zip(alignment["rows"], selected["plan"]["rows"])
    }
    for item in record["acquisitions"]:
        row = by_id[item["id"]]
        item["scanplan_ordinal"] = row["ordinal"]
        item["scanplan_include"] = row["include"]
        if row.get("task") is not None:
            item.setdefault("entities", {})["task"] = row["task"]
            item["scanplan_task"] = row["task"]
        else:
            item.pop("scanplan_task", None)
    return record
