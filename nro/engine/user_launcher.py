"""Select a user-registered installation by working directory using only stdlib.

Installation copies this script into the user's bin directory. It does not
import nro before selecting the interpreter, site, and checkout.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

INDEX_NAME = ".nro-launchers.json"
RECORD_NAME = ".nro-installation.json"


def read_index(path: Path) -> dict:
    """Read explicit checkout bindings; reject malformed or relative locations."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"default", "checkouts"}:
        raise ValueError(f"Invalid launcher index: {path}")
    if not isinstance(value["checkouts"], dict):
        raise ValueError(f"Invalid launcher checkouts: {path}")
    for root, record in value["checkouts"].items():
        if not Path(root).is_absolute() or Path(record) != Path(root) / RECORD_NAME:
            raise ValueError(f"Invalid launcher binding: {root}")
    if value["default"] is not None and value["default"] not in value["checkouts"]:
        raise ValueError("Default installation is not registered with the launcher")
    return value


def select_installation(index: dict, cwd: Path) -> dict:
    """Resolve the nearest installed checkout, or the outside-checkout default.

    Recognized but unregistered nro checkouts stop selection. They cannot fall
    through to a production installation or another ancestor checkout.
    """
    root = None
    cwd = cwd.resolve()
    for parent in (cwd, *cwd.parents):
        if str(parent) in index["checkouts"]:
            root = str(parent)
            break
        if (parent / RECORD_NAME).exists() or (
            (parent / "install").is_file() and (parent / "nro/cli.py").is_file()
        ):
            raise ValueError(f"nro checkout {parent} is not connected; run ./install there")
    root = root or index["default"]
    if root is None:
        raise ValueError(
            "No default nro installation; enter an installed checkout or connect a default installation"
        )
    record = json.loads(Path(index["checkouts"][root]).read_text())
    if (Path(root) / ".nro-installation-transition.json").exists() and record.get(
        "mode"
    ) != "branch":
        raise ValueError(
            "Shared-to-branch conversion is incomplete; resume ./install --convert-to-branch"
        )
    if record.get("checkout") != root or Path(root).resolve() != Path(root):
        raise ValueError("Installation checkout binding changed; reconnect explicitly")
    if not record.get("ready"):
        raise ValueError(f"Installation is incomplete: {root}; rerun ./install")
    if record.get("mode") not in {"personal", "shared", "branch"}:
        raise ValueError("Unknown installation role; reconnect explicitly")
    if record["mode"] == "branch" and not all(
        record.get(key) for key in ("branch", "branch_catalog", "registry_id")
    ):
        raise ValueError(
            "Development installation has no complete branch binding; reconnect explicitly"
        )
    if "branch" in record:
        validate_branch(record)
    python = Path(record["environment"]) / "bin/python"
    if not python.is_file() or not Path(record["site"]).is_file():
        raise ValueError(f"Installation interpreter or site settings are missing: {root}")
    return record


def validate_branch(record: dict) -> None:
    """Check the current Git branch and central binding before selecting its code."""
    result = subprocess.run(
        ["git", "-C", record["checkout"], "symbolic-ref", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or result.stdout.strip() != record["branch"]:
        raise ValueError("Checkout branch changed or HEAD is detached; use the registered branch")
    catalog = json.loads(Path(record["branch_catalog"]).read_text())
    binding = catalog.get(record["branch"], {})
    if (
        binding.get("retired") is not False
        or binding.get("registry_id") != record["registry_id"]
        or record["checkout"] not in binding.get("checkouts", [])
    ):
        raise ValueError("Checkout no longer matches its central branch registration")


def main(argv=None, *, index_path: Path | None = None) -> None:
    """Replace the launcher with the selected interpreter, excluding ambient imports."""
    try:
        index = read_index(index_path or Path(__file__).with_name(INDEX_NAME))
        record = select_installation(index, Path.cwd())
        python = str(Path(record["environment"]) / "bin/python")
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        env.update(NRO_SITE_CONFIG=record["site"], PYTHONDONTWRITEBYTECODE="1")
        os.execve(
            python,
            [python, "-I", "-B", "-m", "nro.cli", *(sys.argv[1:] if argv is None else argv)],
            env,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"nro: {error}")


if __name__ == "__main__":
    main()
