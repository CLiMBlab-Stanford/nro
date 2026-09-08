"""Bootstrap an editable environment using only the Python standard library."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import tomllib

UV_VERSION = "0.8.22"
ROOT = Path(__file__).resolve().parents[2]
RECORD = ".nro-installation.json"


def write_record(path: Path, record: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, indent=2)
        stream.write("\n")
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def maintenance_lock(root: Path, mode: str):
    with (root / ".nro-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = os.umask(0o022) if mode == "shared" else None
        try:
            yield
        finally:
            if previous is not None:
                os.umask(previous)


def check_workers(site: Path) -> None:
    """Block maintenance unless recorded workers and allocations have ended."""
    values = tomllib.loads(site.read_text()) if site.exists() else {}
    database = Path(values.get("registry", "/juice6/u/nlp/climblab/.nro")) / "registry.sqlite3"
    if not database.exists():
        return
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        try:
            workers = db.execute(
                "SELECT slurm_job_id FROM workers WHERE state IN "
                "('idle','running','draining','shutdown_requested')"
            ).fetchall()
            submissions = db.execute(
                "SELECT slurm_job_id FROM scheduler_submissions WHERE state IN "
                "('prepared','submitted','running','cancel_requested')"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            raise RuntimeError(f"Cannot establish whether workers are active: {error}") from error
    records = workers + submissions
    if not records:
        return
    if any(not row[0] for row in records):
        raise RuntimeError("Stop or drain the shared worker pool before maintaining this installation.")
    job_ids = sorted({str(row[0]) for row in records})
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%T"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot confirm that recorded Slurm allocations have ended; "
            "maintenance is blocked. Check scheduler access and retry."
        ) from error
    if result.stdout.strip():
        raise RuntimeError("Stop or drain the shared worker pool before maintaining this installation.")


def connect_user(record: dict, *, bin_dir: Path | None = None) -> Path:
    if not record.get("ready"):
        raise RuntimeError("The installation is incomplete. Ask its maintainer to run ./install --maintain.")
    python = Path(record["environment"]) / "bin/python"
    if not python.is_file() or not Path(record["site"]).is_file():
        raise RuntimeError("The shared interpreter or site configuration is missing.")
    destination = (bin_dir or Path.home() / ".local/bin") / "nro"
    launcher = (
        "#!/bin/sh\n# nro installation launcher\n"
        + "export PYTHONDONTWRITEBYTECODE=1\n"
        + "export NRO_SITE_CONFIG=" + shlex.quote(record["site"]) + "\n"
        + "exec " + shlex.quote(str(python)) + ' -m nro.cli "$@"\n'
    )
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_text() != launcher:
            raise RuntimeError(f"{destination} already selects another command. Choose --bin-dir or move it explicitly.")
        print(f"Already connected: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(launcher)
    destination.chmod(0o755)
    print(f"Installed {destination}")
    if str(destination.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Add this directory to PATH: {destination.parent}")
    return destination


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="./install", description=__doc__)
    parser.add_argument("--mode", choices=("personal", "shared"))
    parser.add_argument("--maintain", action="store_true")
    parser.add_argument("--site", type=Path)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--without-oslom", action="store_true", help="Skip OSLOM and its Python dependencies")
    parser.add_argument("--with-bidsify", action="store_true", help="Install Flywheel, dcm2bids, and DICOM Python dependencies")
    parser.add_argument("--dev", action="store_true", help="Include the locked test dependencies")
    parser.add_argument("--accept-qunex-license", action="store_true")
    parser.add_argument("--local", action="store_true", help="Do not require Slurm")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("nro supports Linux only")
    record_path = ROOT / RECORD
    existing = json.loads(record_path.read_text()) if record_path.exists() else None
    if existing and existing.get("checkout") != str(ROOT):
        parser.error("This installation record belongs to a different checkout; configure a fresh installation")
    if existing and args.mode and args.mode != existing["mode"]:
        parser.error("An existing installation's role cannot be changed implicitly")
    if existing and args.site and args.site.expanduser().resolve() != Path(existing["site"]):
        parser.error("Use nro paths to edit the existing site configuration")
    if existing and existing["mode"] == "shared" and not args.maintain:
        connect_user(existing, bin_dir=args.bin_dir)
        return
    mode = existing["mode"] if existing else args.mode
    if not mode:
        if args.non_interactive or not sys.stdin.isatty():
            parser.error("First installation requires --mode personal or --mode shared")
        mode = input("Installation role [personal/shared]: ").strip().lower()
        if mode not in {"personal", "shared"}:
            parser.error("Choose personal or shared")
    site = Path(existing["site"]) if existing else (
        args.site.expanduser().resolve() if args.site else
        ROOT / ".nro-site.toml" if mode == "shared" else
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nro/site.toml"
    )
    check_workers(site)
    with maintenance_lock(ROOT, mode):
        environment = Path(existing["environment"]) if existing else ROOT / ".nro-env"
        record = {
            "mode": mode, "checkout": str(ROOT), "environment": str(environment),
            "site": str(site), "ready": False,
            "with_oslom": not args.without_oslom,
            "with_bidsify": args.with_bidsify or bool(existing and existing.get("with_bidsify")),
            "dev": args.dev or bool(existing and existing.get("dev")),
            "local": args.local or bool(existing and existing.get("local")),
        }
        write_record(record_path, record)
        uv_env = ROOT / ".nro-bootstrap"
        uv = uv_env / "bin/uv"
        if not uv.is_file():
            if args.offline:
                raise RuntimeError("Offline setup needs an existing bootstrap environment")
            subprocess.run([sys.executable, "-m", "venv", str(uv_env)], check=True)
            subprocess.run([str(uv_env / "bin/python"), "-m", "pip", "install", f"uv=={UV_VERSION}"], check=True)
        sync = [str(uv), "sync", "--frozen", "--python", "3.12"]
        if record["with_oslom"]:
            sync += ["--extra", "oslom"]
        if record["with_bidsify"]:
            sync += ["--extra", "bidsify"]
        if not record["dev"]:
            sync += ["--no-dev"]
        if args.offline:
            sync += ["--offline"]
        env = {
            **os.environ, "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_PYTHON_INSTALL_DIR": str(ROOT / ".nro-python"),
            "UV_CACHE_DIR": str(ROOT / ".nro-cache"),
            "NRO_SITE_CONFIG": str(site),
            "NRO_SETUP_CHILD": "1",
        }
        subprocess.run(sync, cwd=ROOT, env=env, check=True)
        command = [str(environment / "bin/python"), "-m", "nro.bin.setup", "--maintain", "--resources-only"]
        if args.non_interactive:
            command += ["--non-interactive"]
        if args.offline:
            command += ["--offline"]
        if not record["with_oslom"]:
            command += ["--without-oslom"]
        if args.accept_qunex_license:
            command += ["--accept-qunex-license"]
        if record["local"]:
            command += ["--local"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        record["ready"] = True
        write_record(record_path, record)
        connect_user(record, bin_dir=args.bin_dir)


def cancel_setup() -> None:
    """Report cancellation only from the outermost setup process."""
    if os.environ.get("NRO_SETUP_CHILD") != "1":
        print("\nSetup cancelled. Rerun ./install to resume; use --maintain for shared setup.", file=sys.stderr)
    raise SystemExit(130) from None


def main(argv=None) -> None:
    try:
        _main(argv)
    except (KeyboardInterrupt, EOFError):
        cancel_setup()
    except subprocess.CalledProcessError as error:
        if error.returncode in {-2, 130}:
            cancel_setup()
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Setup incomplete: {error}")
