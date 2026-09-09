"""Bootstrap an editable environment using only the Python standard library."""

import argparse
import fcntl
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from nro.configuration.site import settings
from nro.engine import user_launcher
from nro.orchestration.control_paths import ControlPaths

UV_VERSION = "0.8.22"
ROOT = Path(__file__).resolve().parents[2]
RECORD = ".nro-installation.json"


def write_record(path: Path, record: dict) -> None:
    """Atomically publish an installation record and sync its directory."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def maintenance_lock(root: Path, mode: str):
    """Hold the checkout maintenance lock and apply its installation umask."""
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
    values = settings(path=site)[0]
    paths = ControlPaths(Path(values["registry"]))
    paths.require_current_layout()
    database = paths.database
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
        raise RuntimeError(
            "Stop or drain the shared worker pool before maintaining this installation."
        )
    job_ids = sorted({str(row[0]) for row in records})
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(job_ids), "--format", "%T"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot confirm that recorded Slurm allocations have ended; "
            "maintenance is blocked. Check scheduler access and retry."
        ) from error
    if result.stdout.strip():
        raise RuntimeError(
            "Stop or drain the shared worker pool before maintaining this installation."
        )


def check_branch_environment(site: Path, environment: Path) -> None:
    """Ask central Python about environment pins while holding the installation lock."""
    values = settings(path=site)[0]
    binding = ControlPaths(Path(values["registry"])).scheduler / "implementation.json"
    if not binding.exists():
        return
    central = json.loads(binding.read_text())
    result = subprocess.run(
        [central["python"], "-I", "-B", "-m", "nro.orchestration.scheduler_service"],
        cwd=central["checkout"],
        env={**os.environ, "NRO_SITE_CONFIG": central["site"]},
        text=True,
        input=json.dumps(
            dict(operation="environment_idle", checkout=str(ROOT), environment=str(environment))
        ),
        stdout=subprocess.PIPE,
        check=False,
    )
    try:
        response = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("Cannot verify environment safety with the central scheduler") from error
    if result.returncode or response.get("error"):
        raise RuntimeError(response.get("error", "Central environment check failed"))


def _fixed_launcher_record(text: str) -> dict | None:
    lines = text.splitlines()
    if len(lines) != 5 or lines[:3] != [
        "#!/bin/sh",
        "# nro installation launcher",
        "export PYTHONDONTWRITEBYTECODE=1",
    ]:
        return None
    try:
        exported, command = shlex.split(lines[3]), shlex.split(lines[4])
        if (
            len(exported) != 2
            or exported[0] != "export"
            or not exported[1].startswith("NRO_SITE_CONFIG=")
            or len(command) != 5
            or command[0] != "exec"
            or command[2:] != ["-m", "nro.cli", "$@"]
        ):
            return None
        python = Path(command[1])
        if not python.is_absolute() or python.name != "python" or python.parent.name != "bin":
            return None
        root = python.parents[2]
        record = json.loads((root / RECORD).read_text())
        if (
            record.get("checkout") != str(root)
            or not record.get("ready")
            or Path(record.get("environment", "")) / "bin/python" != python
            or record.get("site") != exported[1].split("=", 1)[1]
            or not python.is_file()
            or not Path(record["site"]).is_file()
            or record.get("mode") not in {"shared", "personal", "branch"}
        ):
            return None
        return record
    except (OSError, ValueError, IndexError, TypeError):
        return None


def connect_user(
    record: dict,
    *,
    bin_dir: Path | None = None,
    set_default: bool = False,
    replace_launcher: bool = False,
) -> Path:
    """Connect this checkout; change the user's default only when explicitly requested.

    Replacing a recognized fixed-path nro launcher requires explicit permission
    and retains a backup. Unrelated commands and symlinks are never replaced.
    """
    if not record.get("ready"):
        raise RuntimeError(
            "The installation is incomplete. Ask its maintainer to run ./install --maintain."
        )
    python = Path(record["environment"]) / "bin/python"
    if not python.is_file() or not Path(record["site"]).is_file():
        raise RuntimeError("The shared interpreter or site configuration is missing.")
    destination = (bin_dir or Path.home() / ".local/bin") / "nro"
    marker = "# nro directory-aware launcher\n"
    launcher = "#!" + sys.executable + "\n" + marker + Path(user_launcher.__file__).read_text()
    root = str(Path(record["checkout"]).resolve())
    if record["checkout"] != root or record.get("mode") not in {"personal", "shared", "branch"}:
        raise RuntimeError("Invalid installation checkout or role")
    if json.loads((Path(root) / RECORD).read_text()) != record:
        raise RuntimeError("Installation record changed; reconnect after setup completes")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / ".nro-launchers.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_symlink():
            raise RuntimeError(
                f"{destination} is a symlink; choose --bin-dir or move it explicitly"
            )
        previous = destination.read_text() if destination.exists() else None
        fixed = None
        if previous is not None and marker not in previous.splitlines(keepends=True)[:2]:
            fixed = _fixed_launcher_record(previous)
            if fixed is None:
                raise RuntimeError(
                    f"{destination} already selects another command; no launcher was replaced"
                )
            if not replace_launcher:
                raise RuntimeError(
                    f"{destination} is a fixed-path nro launcher; use --replace-launcher to back it up and replace it"
                )
        index_path = destination.with_name(user_launcher.INDEX_NAME)
        index = (
            user_launcher.read_index(index_path)
            if index_path.exists()
            else {"default": None, "checkouts": {}}
        )
        original_index = index_path.read_bytes() if index_path.exists() else None
        if fixed is not None:
            old_root = fixed["checkout"]
            index["checkouts"][old_root] = str(Path(old_root) / RECORD)
            if index["default"] is None:
                index["default"] = old_root
        index["checkouts"][root] = str(Path(root) / RECORD)
        if set_default or (index["default"] is None and record["mode"] != "branch"):
            index["default"] = root
        if set_default:
            user_launcher.select_installation(index, Path(root))
        backup = None
        if fixed is not None:
            backup = destination.with_name(f"nro.previous-{uuid.uuid4().hex}")
            with backup.open("x") as stream:
                stream.write(previous)
                stream.flush()
                os.fsync(stream.fileno())
            backup.chmod(destination.stat().st_mode & 0o777)
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(launcher)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            temporary.chmod(0o755)
            write_record(index_path, index)
            temporary.replace(destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if original_index is None:
                index_path.unlink(missing_ok=True)
            else:
                write_record(index_path, json.loads(original_index))
            raise
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Installed {destination}")
    print(f"Default installation: {index['default'] or '(none)'}")
    if backup is not None:
        print(f"Previous launcher retained at {backup}")
    if str(destination.parent) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"Add this directory to PATH: {destination.parent}")
    return destination


def _main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="./install", description=__doc__)
    parser.add_argument("--mode", choices=("personal", "shared", "branch"))
    parser.add_argument("--maintain", action="store_true")
    parser.add_argument("--site", type=Path)
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument(
        "--default",
        dest="set_default",
        action="store_true",
        help="Make this checkout the invoking user's default outside installed checkouts",
    )
    parser.add_argument(
        "--replace-launcher",
        action="store_true",
        help="Back up and replace a recognized fixed-path nro launcher",
    )
    parser.add_argument(
        "--convert-to-branch",
        action="store_true",
        help="Convert a quiescent shared checkout to branch mode, retaining its environment",
    )
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--without-oslom", action="store_true", help="Skip OSLOM and its Python dependencies"
    )
    parser.add_argument(
        "--with-bidsify",
        action="store_true",
        help="Install Flywheel, dcm2bids, and DICOM Python dependencies",
    )
    parser.add_argument("--dev", action="store_true", help="Include the locked test dependencies")
    parser.add_argument("--accept-qunex-license", action="store_true")
    parser.add_argument("--local", action="store_true", help="Do not require Slurm")
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("nro supports Linux only")
    record_path = ROOT / RECORD
    existing = json.loads(record_path.read_text()) if record_path.exists() else None
    default_record = None
    index_path = (args.bin_dir or Path.home() / ".local/bin") / user_launcher.INDEX_NAME
    if index_path.exists():
        index = user_launcher.read_index(index_path)
        if index["default"]:
            default_record = json.loads(Path(index["checkouts"][index["default"]]).read_text())
    if existing and existing.get("checkout") != str(ROOT):
        parser.error(
            "This installation record belongs to a different checkout; configure a fresh installation"
        )
    if args.convert_to_branch:
        if not existing or args.mode or args.maintain or args.set_default:
            parser.error(
                "--convert-to-branch requires an existing installation and cannot be combined with --mode, --maintain, or --default"
            )
        from nro.engine.installation_transition import convert_shared

        with maintenance_lock(ROOT, "branch"):
            record = convert_shared(ROOT, default_record, site=args.site)
        connect_user(record, bin_dir=args.bin_dir, replace_launcher=args.replace_launcher)
        return
    if existing and args.mode and args.mode != existing["mode"]:
        parser.error("An existing installation's role cannot be changed implicitly")
    if existing and args.site and args.site.expanduser().resolve() != Path(existing["site"]):
        parser.error("Use nro paths to edit the existing site configuration")
    if existing and existing["mode"] == "shared" and not args.maintain:
        connect_user(
            existing,
            bin_dir=args.bin_dir,
            set_default=args.set_default,
            replace_launcher=args.replace_launcher,
        )
        return
    mode = existing["mode"] if existing else args.mode
    if not mode and default_record and default_record["checkout"] != str(ROOT):
        mode = "branch"
    if not mode:
        if args.non_interactive or not sys.stdin.isatty():
            parser.error("First installation requires --mode personal, shared, or branch")
        mode = input("Installation role [personal/shared/branch]: ").strip().lower()
        if mode not in {"personal", "shared", "branch"}:
            parser.error("Choose personal, shared, or branch")
    site = (
        Path(existing["site"])
        if existing
        else (
            args.site.expanduser().resolve()
            if args.site
            else Path(default_record["site"])
            if mode == "branch" and default_record
            else ROOT / ".nro-site.toml"
            if mode == "shared"
            else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nro/site.toml"
        )
    )
    branch_name = None
    if mode == "branch":
        if not site.is_file():
            parser.error(
                "Branch installation must select an existing site using --site or the default installation"
            )
        if args.maintain:
            parser.error("A branch installation cannot maintain shared resources")
        from nro.orchestration.branches import checkout_identity

        _, branch_name, _ = checkout_identity(ROOT)
        if branch_name == "main":
            parser.error(
                "Main must be installed through the approved production deployment, not branch setup"
            )
        paths = ControlPaths(Path(settings(path=site)[0]["registry"]))
        paths.require_current_layout()
        if existing and existing.get("branch") != branch_name:
            parser.error("Checkout branch changed; return to its installed branch")
    else:
        check_workers(site)
    with maintenance_lock(ROOT, mode):
        environment = Path(existing["environment"]) if existing else ROOT / ".nro-env"
        if mode in {"branch", "shared"} and existing:
            check_branch_environment(site, environment)
        record = {
            "mode": mode,
            "checkout": str(ROOT),
            "environment": str(environment),
            "site": str(site),
            "ready": False,
            "with_oslom": not args.without_oslom,
            "with_bidsify": args.with_bidsify or bool(existing and existing.get("with_bidsify")),
            "dev": args.dev or bool(existing and existing.get("dev")),
            "local": args.local or bool(existing and existing.get("local")),
        }
        if branch_name:
            record["branch"] = branch_name
        write_record(record_path, record)
        uv_env = ROOT / ".nro-bootstrap"
        uv = uv_env / "bin/uv"
        if not uv.is_file():
            if args.offline:
                raise RuntimeError("Offline setup needs an existing bootstrap environment")
            subprocess.run([sys.executable, "-m", "venv", str(uv_env)], check=True)
            subprocess.run(
                [str(uv_env / "bin/python"), "-m", "pip", "install", f"uv=={UV_VERSION}"],
                check=True,
            )
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
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_PYTHON_INSTALL_DIR": str(ROOT / ".nro-python"),
            "UV_CACHE_DIR": str(ROOT / ".nro-cache"),
            "NRO_SITE_CONFIG": str(site),
            "NRO_SETUP_CHILD": "1",
        }
        subprocess.run(sync, cwd=ROOT, env=env, check=True)
        command = [
            str(environment / "bin/python"),
            "-I",
            "-B",
            "-m",
            "nro.bin.setup",
            "--resources-only",
        ]
        if mode != "branch":
            command += ["--maintain"]
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
        if branch_name:
            catalog = json.loads(paths.catalog.read_text()) if paths.catalog.exists() else {}
            action = "attach" if branch_name in catalog else "register"
            command = [
                str(environment / "bin/python"),
                "-I",
                "-B",
                "-m",
                "nro.bin.branch",
                action,
                "--checkout",
                str(ROOT),
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            binding = json.loads(paths.catalog.read_text())[branch_name]
            record.update(registry_id=binding["registry_id"], branch_catalog=str(paths.catalog))
        record["ready"] = True
        write_record(record_path, record)
        connect_user(
            record,
            bin_dir=args.bin_dir,
            set_default=args.set_default,
            replace_launcher=args.replace_launcher,
        )


def cancel_setup() -> None:
    """Report cancellation only from the outermost setup process."""
    if os.environ.get("NRO_SETUP_CHILD") != "1":
        print(
            "\nSetup cancelled. Rerun ./install to resume; use --maintain for shared setup.",
            file=sys.stderr,
        )
    raise SystemExit(130) from None


def main(argv=None) -> None:
    """Run setup and translate interactive cancellation into exit status 130."""
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
    except (
        OSError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(f"Setup incomplete: {error}")
