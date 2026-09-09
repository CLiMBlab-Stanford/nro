"""Convert a quiescent shared checkout into a registered development installation."""

import json
import os
import sqlite3
import subprocess
import time
from pathlib import Path

from nro.configuration.site import settings
from nro.orchestration.branches import checkout_identity
from nro.orchestration.control_paths import ControlPaths


def _require_quiescent(site: Path) -> None:
    paths = ControlPaths(Path(settings(path=site)[0]["registry"]))
    paths.require_current_layout()
    if paths.database.exists():
        with sqlite3.connect(paths.database.as_uri() + "?mode=ro", uri=True) as db:
            for table, states in (
                ("requests", "'active'"),
                ("attempts", "'queued','running','cancel_requested'"),
                ("workers", "'idle','running','draining','shutdown_requested'"),
                ("scheduler_submissions", "'prepared','submitted','running','cancel_requested'"),
            ):
                if db.execute(
                    f"SELECT 1 FROM {table} WHERE state IN ({states}) LIMIT 1"
                ).fetchone():
                    raise RuntimeError(
                        f"Finish or cancel active {table} before converting the installation"
                    )
            if db.execute("SELECT 1 FROM metadata WHERE key='maintenance_mode'").fetchone():
                raise RuntimeError("Shared registry maintenance is in progress")
    for path in paths.ingestion.glob("*.json"):
        if json.loads(path.read_text()).get("state") in {"queued", "running"}:
            raise RuntimeError(
                "Finish or cancel active ingestion before converting the installation"
            )
    for path in (paths.ingestion / "reviews").glob("*.json"):
        if json.loads(path.read_text()).get("expires", 0) > time.time():
            raise RuntimeError("End ingestion review leases before converting the installation")


def convert_shared(root: Path, replacement: dict | None, *, site: Path | None = None) -> dict:
    """Convert or resume a shared-to-branch transition without modifying its environment.

    Requires a different ready shared default and an unchanged resolved site.
    The caller holds the installation lock and coordinates a maintenance window.
    After conversion begins, failures leave a blocked branch record for retry;
    the previous shared record is retained, never automatically reactivated.
    """
    from nro.engine.bootstrap import RECORD, write_record

    record_path = root / RECORD
    current = json.loads(record_path.read_text())
    journal_path = root / ".nro-installation-transition.json"
    if (
        not replacement
        or replacement.get("checkout") == str(root)
        or replacement.get("mode") != "shared"
    ):
        raise ValueError(
            "Connect a different shared installation as your default before conversion"
        )
    if (
        not replacement.get("ready")
        or not (Path(replacement["environment"]) / "bin/python").is_file()
        or json.loads((Path(replacement["checkout"]) / RECORD).read_text()) != replacement
    ):
        raise ValueError("Replacement shared installation is incomplete or changed")
    checkout, branch, _ = checkout_identity(root)
    if checkout != root or branch == "main":
        raise ValueError("Conversion requires this checkout on a named development branch")
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else None
    original = journal["original"] if journal else current
    if (
        original.get("checkout") != str(root)
        or original.get("mode") != "shared"
        or not original.get("ready")
    ):
        raise ValueError("Conversion requires a ready shared installation record")
    target_site = site.expanduser().resolve() if site else Path(replacement["site"])
    if not target_site.is_file() or not Path(original["site"]).is_file():
        raise ValueError("Both existing and replacement site settings must exist")
    if target_site.is_relative_to(root):
        raise ValueError("Select site settings outside the development checkout")
    original_values = settings(path=Path(original["site"]))[0]
    if (
        settings(path=target_site)[0] != original_values
        or settings(path=Path(replacement["site"]))[0] != original_values
    ):
        raise ValueError("Conversion cannot change resolved site paths or settings")
    if journal and (journal.get("branch") != branch or journal.get("site") != str(target_site)):
        raise ValueError("Resume conversion with its original branch and site settings")
    if (
        journal
        and current != original
        and (
            current.get("mode") != "branch"
            or current.get("branch") != branch
            or current.get("checkout") != str(root)
            or current.get("site") != str(target_site)
            or current.get("environment") != original["environment"]
        )
    ):
        raise ValueError("Installation record changed outside the pending conversion")
    _require_quiescent(Path(original["site"]))
    if not journal:
        journal = dict(original=original, branch=branch, site=str(target_site))
        write_record(journal_path, journal)
    record = dict(original, mode="branch", branch=branch, site=str(target_site), ready=False)
    write_record(record_path, record)
    paths = ControlPaths(Path(original_values["registry"]))
    catalog = json.loads(paths.catalog.read_text()) if paths.catalog.exists() else {}
    action = "attach" if branch in catalog else "register"
    python = str(Path(original["environment"]) / "bin/python")
    env = {
        key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(NRO_SITE_CONFIG=str(target_site), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [python, "-I", "-B", "-m", "nro.bin.branch", action, "--checkout", str(root)],
        check=True,
        cwd=root,
        env=env,
    )
    binding = json.loads(paths.catalog.read_text())[branch]
    if binding.get("retired") is not False or str(root) not in binding.get("checkouts", []):
        raise ValueError("The new branch registration does not authorize this checkout")
    _require_quiescent(target_site)
    record.update(ready=True, registry_id=binding["registry_id"], branch_catalog=str(paths.catalog))
    write_record(record_path, record)
    print(f"Converted {root} to branch {branch}; isolated processing remains disabled.")
    print(f"Previous shared installation record retained in {journal_path}")
    return record
