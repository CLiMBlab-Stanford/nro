"""Create and validate lab-owned definitions independently of the installation."""

import ctypes
import errno
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from nro.configuration.events import EventStore, task_key
from nro.configuration.site import definitions_root
from nro.configuration.store import DERIVATIVE_CLASSES, ConfigStore, validate_config_id

STARTERS = Path(__file__).parent / "starters"
CATEGORIES = ("configs", "workflows", "models", "events", "bidsify")


def validate_store(root: Path | None = None) -> dict[str, int]:
    """Check every definition and reference without changing files or registry state.

    Validate filenames, compiled configurations and models, workflow references,
    event tables, and ingestion profiles. Empty model and event catalogs are
    allowed. Raise ValueError with all discovered problems. This does not check
    scientific suitability, credentials, or availability of external resources.
    """
    from nro.bidsify.config import load_config
    from nro.modules.firstlevels.task_models import load_task_model, scientific_model

    root = Path(root).expanduser().resolve() if root is not None else definitions_root()
    store = ConfigStore(root)
    errors = []
    counts = dict(configs=0, workflows=0, models=0, event_ids=0, event_tsvs=0, bidsify=0)

    def check(path, operation):
        try:
            return operation()
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            errors.append(f"{path}: {error}")
            return None

    files = {}
    for category in CATEGORIES:
        directory = root / category
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")
        paths = []
        if directory.is_symlink():
            errors.append(f"Definition directories must not be symlinks: {directory}")
        else:
            for parent, directories, names in os.walk(directory, followlinks=False):
                for name in list(directories):
                    path = Path(parent) / name
                    if path.is_symlink():
                        errors.append(f"Definition directories must not be symlinks: {path}")
                        directories.remove(name)
                    elif name.startswith("."):
                        directories.remove(name)
                for name in names:
                    if name.startswith("."):
                        continue
                    path = Path(parent) / name
                    if path.is_symlink():
                        errors.append(f"Definition files must not be symlinks: {path}")
                    elif path.is_file():
                        paths.append(path)
        files[category] = sorted(paths)

    for kind in DERIVATIVE_CLASSES:
        check(store.configs / kind, lambda kind=kind: store.load_configuration(kind, "main"))
    for path in files["configs"]:
        relative = path.relative_to(store.configs)
        kind = relative.parts[0]
        suffix = f"_{kind}.yml"
        if (
            len(relative.parts) != 2
            or kind not in DERIVATIVE_CLASSES
            or not path.name.endswith(suffix)
        ):
            errors.append(f"Unexpected configuration filename: {path}")
            continue
        check(path, lambda: store.load_configuration(kind, path.name.removesuffix(suffix)))
        counts["configs"] += 1

    if not (root / "workflows/main_workflow.yml").is_file():
        errors.append("Missing main workflow")
    for path in files["workflows"]:
        if path.parent != root / "workflows" or not path.name.endswith("_workflow.yml"):
            errors.append(f"Unexpected workflow filename: {path}")
            continue
        check(path, lambda: store.resolve(path.name.removesuffix("_workflow.yml")))
        counts["workflows"] += 1

    for path in files["models"]:
        if len(path.relative_to(root / "models").parts) != 2 or path.suffix != ".yml":
            errors.append(f"Expected models/TASK/VARIANT.yml: {path}")
            continue
        identifier = f"{path.parent.name}/{path.stem}"
        check(path, lambda: scientific_model(load_task_model(identifier, root / "models")))
        counts["models"] += 1

    if (root / "events").is_dir():
        events = EventStore(root / "events")
        referenced, tasks = set(), {}
        for path in files["events"]:
            if path.name == "index.yml" and len(path.relative_to(events.root).parts) == 2:
                entries = check(path, lambda: events._entries(path))
                if entries is None:
                    continue
                for task in entries[0]:
                    key = task_key(task)
                    if key in tasks and tasks[key] != path:
                        errors.append(f"Task {task!r} is indexed in both {tasks[key]} and {path}")
                    tasks[key] = path
                for entry in entries[1]:
                    counts["event_ids"] += 1
                    if entry.path not in referenced:
                        check(entry.path, entry.snapshot)
                        referenced.add(entry.path)
            elif path.suffix != ".tsv":
                errors.append(f"Unexpected event catalog file: {path}")
        for path in files["events"]:
            if path.suffix == ".tsv" and path not in referenced:
                errors.append(f"Unindexed event table: {path}")
        counts["event_tsvs"] = len(referenced)

    if not (root / "bidsify/main.yml").is_file():
        errors.append("Missing bidsify/main.yml")
    for path in files["bidsify"]:
        if path.parent != root / "bidsify" or path.suffix != ".yml":
            errors.append(f"Expected bidsify/PROFILE.yml: {path}")
            continue
        check(path, lambda: validate_config_id(path.stem, kind="ingestion profile"))
        check(path, lambda: load_config(path, root=root))
        counts["bidsify"] += 1
    if errors:
        raise ValueError("Definitions store is invalid:\n" + "\n".join(errors))
    return counts


def _publish(staged: Path, destination: Path) -> None:
    # Linux renameat2 prevents replacement even if another creator wins the race.
    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        if error not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise OSError(error, os.strerror(error), str(destination))
        # Some shared filesystems lack RENAME_NOREPLACE. Reserve the directory
        # exclusively and mark it incomplete until every child has been moved.
        destination.mkdir(mode=0o755)
        marker = destination / ".nro-incomplete"
        marker.touch(exist_ok=False)
        for path in staged.iterdir():
            path.rename(destination / path.name)
        marker.unlink()


def create_store(root: Path | None = None) -> Path:
    """Publish a new validated starter store; refuse every existing destination.

    Create parent directories as needed. Leave existing stores and site settings
    unchanged. Models and events start empty. Do not run Git or access the network.
    """
    from nro.configuration.site import require_definition_write

    destination = Path(root).expanduser().absolute() if root is not None else definitions_root()
    require_definition_write(destination, creating_store=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Definitions destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".nro-definitions-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "store"
        shutil.copytree(
            STARTERS,
            staged,
            ignore=shutil.ignore_patterns("*.swp", "*.swo", "*~", ".DS_Store", "__pycache__"),
        )
        (staged / "gitignore").rename(staged / ".gitignore")
        for kind in DERIVATIVE_CLASSES:
            directory = staged / "configs" / kind
            (directory / f"main_{kind}.yml").unlink()
            if not any(directory.iterdir()):
                (directory / ".gitkeep").touch()
        for category in CATEGORIES:
            (staged / category).mkdir(exist_ok=True)
        for category in ("models", "events"):
            (staged / category / ".gitkeep").touch()
        validate_store(staged)
        _publish(staged, destination)
    return destination


def ensure_store(root: Path | None = None) -> Path:
    """Create a missing installation store, or validate an existing one without overwriting it."""
    root = Path(root).expanduser().absolute() if root is not None else definitions_root()
    if not root.exists():
        return create_store(root)
    validate_store(root)
    return root
