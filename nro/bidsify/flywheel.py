"""Read-only Flywheel access with isolated credentials and opaque saved IDs."""

import hashlib
import logging
import os
from pathlib import Path

from .errors import BidsificationError


def _revision(file) -> dict:
    return {
        "version": getattr(file, "version", None),
        "hash": getattr(file, "hash", None),
        "modified": str(getattr(file, "modified", "") or ""),
    }


class FlywheelSource:
    """Enumerate configured projects and download individually selected files.

    Credentials are read in the executing process and never put in commands,
    configuration snapshots, or returned records. SDK errors are not echoed
    because they may include source labels or authenticated URLs.
    """

    def __init__(self, profile: dict, *, client=None):
        """Create an isolated client; client injection supports offline tests."""
        self.profile = profile
        if client is not None:
            self.client = client
            return
        key = os.environ.get(profile["credential_env"])
        if not key:
            raise BidsificationError(
                f"Set {profile['credential_env']} in your environment before submission"
            )
        if ":" in key:
            host, key = key.split(":", 1)
            if host != profile["host"]:
                raise BidsificationError("Credential host does not match selected server")
        try:
            import flywheel
        except ImportError:
            raise BidsificationError(
                "Could not import the Flywheel SDK. Install bidsification's optional "
                "Python dependencies (Flywheel, dcm2bids, and pydicom).\n"
                "From the nro repository, run: ./install --with-bidsify\n"
                "For a shared installation, ask its maintainer to run instead: "
                "./install --maintain --with-bidsify\n"
                "Stop or drain the worker pool before running the installer. "
                "It updates nro's managed environment and remembers this option "
                "for future updates."
            ) from None
        logging.getLogger("flywheel").setLevel(logging.CRITICAL)
        try:
            self.client = flywheel.Client(f"{profile['host']}:{key}")
        except Exception:
            raise BidsificationError(
                "Could not initialize Flywheel client; check server credentials"
            ) from None

    def sessions(self) -> list[dict]:
        """List session IDs and transient identity labels for BIDS matching and review."""
        result = []
        try:
            for path in self.profile["projects"]:
                project = self.client.lookup(path)
                for session in project.sessions.iter():
                    result.append(
                        {
                            "id": session.id,
                            "label": session.label,
                            "subject_code": getattr(
                                getattr(session, "subject", None), "code", None
                            ),
                            "remote_project": path,
                        }
                    )
        except Exception:
            raise BidsificationError(
                "Flywheel inventory failed; check authentication and project access"
            ) from None
        return result

    def inventory(self, session_id: str) -> list[dict]:
        """List DICOM files without guessing their BIDS types from source labels."""
        result = []
        try:
            session = self.client.get_session(session_id)
            for acquisition in session.acquisitions.iter():
                for file in acquisition.files or []:
                    if file.type != "dicom":
                        continue
                    result.append(
                        {
                            "id": acquisition.id
                            + "-"
                            + hashlib.sha256(file.name.encode()).hexdigest()[:16],
                            "acquisition": acquisition.id,
                            "file_token": hashlib.sha256(file.name.encode()).hexdigest(),
                            "bytes": int(file.size or 0),
                            "datatype": None,
                            "suffix": None,
                            "source_revision": _revision(file),
                            "confirmed": False,
                            "entities": {},
                            "entities_confirmed": False,
                            "events": None,
                        }
                    )
        except Exception:
            raise BidsificationError("Could not inventory the selected Flywheel session") from None
        if not result:
            raise BidsificationError(
                "No acquisition DICOM files found; this source needs an explicit file adapter"
            )
        return result

    def describe(self, item: dict) -> str:
        """Return the acquisition label for terminal review, never persisted in records."""
        try:
            return str(self.client.get_acquisition(item["acquisition"]).label)
        except Exception:
            raise BidsificationError("Cannot retrieve acquisition label for review") from None

    def download(self, item: dict, destination: Path) -> None:
        """Download one unchanged file into the caller's approved staging directory."""
        try:
            acquisition = self.client.get_acquisition(item["acquisition"])
            matches = [
                f
                for f in acquisition.files or []
                if hashlib.sha256(f.name.encode()).hexdigest() == item["file_token"]
            ]
            if (
                len(matches) != 1
                or int(matches[0].size or 0) != item["bytes"]
                or _revision(matches[0]) != item["source_revision"]
            ):
                raise BidsificationError("source changed")
            matches[0].download(str(destination))
            if destination.stat().st_size != item["bytes"]:
                raise BidsificationError("incomplete download")
        except Exception:
            raise BidsificationError(
                "Flywheel download failed or source changed; inspect the source before retrying"
            ) from None
