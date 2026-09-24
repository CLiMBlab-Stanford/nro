"""Create and validate lab-owned definitions independently of the installation."""

import ctypes
import errno
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from nro.configuration.events import EventStore, task_key
from nro.configuration.site import (
    DERIVED,
    definitions_root,
    read_site_definition,
    site_definition_path,
    write_site_definition,
)
from nro.configuration.store import CONFIGURATION_CLASSES, ConfigStore, validate_config_id

STARTERS = Path(__file__).parent / "starters"
CATEGORIES = (
    "site",
    "configs",
    "workflows",
    "models",
    "events",
    "markup",
    "hardware",
    "bidsify",
    "scanplans",
)


def validate_store(
    root: Path | None = None,
    *,
    require_site: bool = False,
    inherited_site: Path | None = None,
    inherited_roots: tuple[Path, ...] = (),
) -> dict[str, int]:
    """Check every definition and reference without changing files or registry state.

    Validate filenames, compiled configurations and models, workflow references,
    event tables, and ingestion profiles. Empty model and event catalogs are
    allowed. Raise ValueError with all discovered problems. This does not check
    scientific suitability, credentials, or availability of external resources.
    """
    from nro.bidsify.config import load_config
    from nro.configuration.definition_migrations import validate_store_integrity
    from nro.modules.firstlevels.task_models import load_task_model, scientific_model

    root = Path(root).expanduser().resolve() if root is not None else definitions_root()
    validate_store_integrity(root)
    inherited_roots = tuple(Path(path).expanduser().resolve() for path in inherited_roots)
    if inherited_site is not None:
        inherited_site = Path(inherited_site).expanduser().resolve()
        if inherited_site not in inherited_roots:
            inherited_roots = (*inherited_roots, inherited_site)
    errors = []
    counts = dict(
        site=0,
        configs=0,
        workflows=0,
        models=0,
        event_ids=0,
        event_tsvs=0,
        markup=0,
        hardware_profiles=0,
        bidsify=0,
        scanplan_parsers=0,
    )

    def check(path, operation):
        try:
            return operation()
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            errors.append(f"{path}: {error}")
            return None

    files = {}
    for category in CATEGORIES:
        directory = root / category
        # Markup and scan-plan parsing are optional site capabilities. Missing
        # directories preserve the default behavior for either capability.
        optional = {"markup", "hardware", "scanplans"}
        if not require_site:
            optional.add("site")
        if not directory.is_dir() and category not in optional:
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

    expected_site = site_definition_path(root)
    site_values = None
    for path in files["site"]:
        if path != expected_site:
            errors.append(f"Expected site/site.yml: {path}")
    if not expected_site.is_file():
        if require_site:
            errors.append(f"Missing protected site definition: {expected_site}")
        if inherited_site is not None:
            inherited_document = check(
                site_definition_path(inherited_site),
                lambda: read_site_definition(inherited_site),
            )
            if inherited_document is not None:
                site_values = inherited_document[0]
    else:
        site_document = check(expected_site, lambda: read_site_definition(root))
        if site_document is not None:
            counts["site"] = 1
            site_values = site_document[0]

    if site_values is not None:
        site_values = dict(site_values)
        for key, (parent, suffix) in DERIVED.items():
            site_values.setdefault(key, str(Path(site_values[parent]) / suffix))
    store = ConfigStore(
        roots=(root, *inherited_roots),
        site_values=(
            site_values
            if site_values is not None
            else {}
            if require_site or inherited_site is not None
            else None
        ),
    )

    for kind in CONFIGURATION_CLASSES:
        check(store.configs / kind, lambda kind=kind: store.load_configuration(kind, "main"))
    for path in files["configs"]:
        relative = path.relative_to(store.configs)
        kind = relative.parts[0]
        suffix = f"_{kind}.yml"
        if (
            len(relative.parts) != 2
            or kind not in CONFIGURATION_CLASSES
            or not path.name.endswith(suffix)
        ):
            errors.append(f"Unexpected configuration filename: {path}")
            continue
        check(path, lambda: store.load_configuration(kind, path.name.removesuffix(suffix)))
        counts["configs"] += 1

    check(root / "workflows/main_workflow.yml", lambda: store.resolve("main"))
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

    from nro.configuration.markup import MarkupStore

    markup_store = MarkupStore(root)
    for path in files["markup"]:
        if path.parent != root / "markup" or not path.name.endswith("_markup.yml"):
            errors.append(f"Expected markup/ID_markup.yml: {path}")
            continue
        identifier = path.name.removesuffix("_markup.yml")
        check(path, lambda identifier=identifier: markup_store.load(identifier))
        counts["markup"] += 1

    from nro.configuration.hardware import validate_gradient_unwarping_catalog

    hardware_files = files["hardware"]
    expected_hardware = root / "hardware" / "gradient_unwarping.yml"
    for path in hardware_files:
        if path != expected_hardware:
            errors.append(f"Expected hardware/gradient_unwarping.yml: {path}")
    if expected_hardware.is_file():
        count = check(
            expected_hardware,
            lambda: validate_gradient_unwarping_catalog(root),
        )
        if count is not None:
            counts["hardware_profiles"] = count

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

    if not any((candidate / "bidsify/main.yml").is_file() for candidate in store.roots):
        errors.append("Missing bidsify/main.yml in the definitions chain")
    for path in files["bidsify"]:
        if path.parent != root / "bidsify" or path.suffix != ".yml":
            errors.append(f"Expected bidsify/PROFILE.yml: {path}")
            continue
        check(path, lambda: validate_config_id(path.stem, kind="ingestion profile"))
        check(path, lambda: load_config(path, root=root, site_root=inherited_site))
        counts["bidsify"] += 1
    from nro.bidsify.scanplans import load_parser

    for path in files["scanplans"]:
        if path.parent != root / "scanplans" or path.name != "parser.py":
            errors.append(f"Expected scanplans/parser.py: {path}")
            continue
        check(path, lambda path=path: load_parser(path))
        counts["scanplan_parsers"] += 1
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


def create_store(
    root: Path | None = None,
    *,
    include_site: bool = True,
    site_values: dict | None = None,
    site_bidsify: dict | None = None,
    inherited_site: Path | None = None,
) -> Path:
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
        for kind in CONFIGURATION_CLASSES:
            directory = staged / "configs" / kind
            if include_site:
                (directory / f"main_{kind}.yml").unlink()
            else:
                for path in directory.glob("*.yml"):
                    path.unlink()
            if not any(directory.iterdir()):
                (directory / ".gitkeep").touch()
        for category in CATEGORIES:
            (staged / category).mkdir(exist_ok=True)
        for category in ("models", "events"):
            (staged / category / ".gitkeep").touch()
        if not include_site:
            for category in ("workflows", "markup", "hardware", "bidsify", "scanplans"):
                directory = staged / category
                for path in directory.iterdir():
                    if path.is_file() and not path.name.startswith("."):
                        path.unlink()
                if not any(directory.iterdir()):
                    (directory / ".gitkeep").touch()
        if include_site:
            if site_values is None:
                from nro.configuration import site

                site_values = site.settings()[0]
            write_site_definition(staged, site_values, bidsify=site_bidsify)
        from nro.configuration.definition_migrations import migrate_store

        inherited = (inherited_site,) if inherited_site is not None else ()
        migrate_store(
            staged,
            validate=lambda candidate: validate_store(
                candidate,
                require_site=include_site,
                inherited_site=inherited_site,
                inherited_roots=inherited,
            ),
        )
        from nro.configuration.definition_migrations import ensure_group_maintainable

        ensure_group_maintainable(staged)
        _publish(staged, destination)
    return destination


def ensure_store(root: Path | None = None) -> Path:
    """Create a missing store, or migrate and validate an existing store."""
    root = Path(root).expanduser().absolute() if root is not None else definitions_root()
    if not root.exists():
        return create_store(root)
    from nro.configuration.definition_migrations import migrate_store

    migrate_store(root, validate=lambda candidate: validate_store(candidate, require_site=True))
    validate_store(root, require_site=True)
    return root
