"""Review and publish text definitions without exposing partially edited files."""

import difflib
import fcntl
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

DRAFT_DIRECTORY = ".definition-drafts"


def read_definition(path: Path) -> bytes | None:
    """Snapshot a regular definition file; reject symbolic links."""
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path}")
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


@contextmanager
def _definition_lock(path: Path, expected: bytes | None):
    from nro.configuration.site import definition_write

    with definition_write(path), _file_lock(path, expected):
        yield


@contextmanager
def _file_lock(path: Path, expected: bytes | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.edit.lock")
    mode = stat.S_IMODE(path.stat().st_mode) if expected is not None else 0o644
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, mode | 0o600)
        os.fchmod(descriptor, mode | 0o600)
    except FileExistsError:
        descriptor = os.open(lock, os.O_RDWR | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another writer is updating {path}; try again") from error
        if read_definition(path) != expected:
            raise ValueError(f"Definition changed during editing: {path}; no changes saved")
        yield


def save_definition(
    path: Path,
    text: str,
    *,
    expected: bytes | None,
    store_root: Path | None = None,
    validate_store: Callable[[Path], None] | None = None,
) -> None:
    """Atomically publish text if the target still matches its snapshot.

    Serialize cooperating writers with a persistent sibling lock. Direct edits
    do not honor that lock, but their content is checked before replacement.
    Existing permissions are preserved. New files are readable by other users.
    """
    with _definition_lock(path, expected):
        if store_root is not None:
            from nro.configuration.definition_migrations import update_store

            update_store(
                store_root,
                {path.relative_to(store_root): text.encode("utf-8")},
                validate=validate_store,
            )
            return
        mode = stat.S_IMODE(path.stat().st_mode) if expected is not None else 0o644
        with tempfile.TemporaryDirectory(prefix=".definition-", dir=path.parent) as directory:
            staged = Path(directory) / "definition.yml"
            with staged.open("w", encoding="utf-8") as output:
                output.write(text)
                output.flush()
                os.fsync(output.fileno())
            staged.chmod(mode)
            if expected is None:
                os.link(staged, path)
            else:
                os.replace(staged, path)


def delete_definition(
    path: Path,
    *,
    expected: bytes,
    store_root: Path | None = None,
    validate_store: Callable[[Path], None] | None = None,
) -> Path:
    """Remove an unchanged definition and return a private recovery-copy path.

    Use the same lock as authoring writes. Only the named file is removed;
    parent directories and the persistent lock remain. Recovery copies live in
    the system temporary directory and may be removed by its cleanup policy.
    """
    with _definition_lock(path, expected):
        directory = Path(tempfile.mkdtemp(prefix="nro-deleted-definition-"))
        backup = directory / path.name
        with backup.open("wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
        if store_root is None:
            path.unlink()
        else:
            from nro.configuration.definition_migrations import update_store

            update_store(
                store_root,
                {path.relative_to(store_root): None},
                validate=validate_store,
            )
        return backup


def _private_directory(path: Path) -> None:
    """Require a real directory owned by this user and inaccessible to others."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Invalid private definition-draft directory: {path}")
    status = path.stat()
    if status.st_uid != os.geteuid() or stat.S_IMODE(status.st_mode) & 0o077:
        raise ValueError(f"Definition-draft directory is not private to this user: {path}")


def definition_draft_path(path: Path, store_root: Path) -> Path:
    """Return the private, ignored draft path for one published definition."""
    root = Path(store_root).expanduser().resolve()
    target = Path(path).expanduser().absolute()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Definition is outside its store: {target}") from error
    if len(relative.parts) < 2:
        raise ValueError(f"Definition has no managed category: {target}")
    category = root / relative.parts[0]
    if category.is_symlink() or not category.is_dir():
        raise ValueError(f"Invalid definition category: {category}")

    shared = category / DRAFT_DIRECTORY
    try:
        shared.mkdir(mode=0o3770)
        shared.chmod(0o3770)
    except FileExistsError:
        if shared.is_symlink() or not shared.is_dir():
            raise ValueError(f"Invalid definition-draft directory: {shared}") from None

    private = shared / f"user-{os.geteuid()}"
    try:
        private.mkdir(mode=0o700)
        private.chmod(0o700)
    except FileExistsError:
        _private_directory(private)

    parent = private
    for component in relative.parts[1:-1]:
        parent = parent / component
        try:
            parent.mkdir(mode=0o700)
            parent.chmod(0o700)
        except FileExistsError:
            _private_directory(parent)
    return parent / relative.name


@contextmanager
def _draft_lock(draft: Path):
    """Prevent two commands from editing one user's saved draft concurrently."""
    lock = draft.with_name(f".{draft.name}.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"This definition draft is already open: {draft}") from error
        yield


def _prepare_draft(
    draft: Path,
    initial: str,
    *,
    interactive: bool,
    explicit_source: bool,
) -> str | None:
    """Create a draft or let an interactive user recover an existing one."""
    if draft.is_symlink():
        raise ValueError(f"Refusing a symbolic-link definition draft: {draft}")
    if draft.exists() and not draft.is_file():
        raise ValueError(f"Invalid definition draft: {draft}")
    if draft.is_file():
        if not interactive:
            raise ValueError(
                "A saved draft is available for this definition. Rerun interactively to "
                "recover it or start over."
            )
        source = "the supplied file" if explicit_source else "the published definition"
        print("A saved draft is available for this definition.")
        while True:
            answer = (
                input(f"Recover it [r], start over from {source} [s], or cancel [q]? [r] ")
                .strip()
                .lower()
            )
            if answer in {"", "r", "recover"}:
                print("Recovered the saved draft.")
                return "recovered"
            if answer in {"s", "start", "start over"}:
                break
            if answer in {"q", "quit", "cancel"}:
                print("Cancelled; the stored definition and saved draft are unchanged.")
                return None
            print("Choose r to recover, s to start over, or q to cancel.")
    draft.write_text(initial, encoding="utf-8")
    draft.chmod(0o600)
    return "fresh"


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Return metadata that changes when an editor writes or replaces a draft."""
    status = path.stat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _arm_write_detection(path: Path) -> tuple[int, int, int, int, int]:
    """Backdate a draft before editing so even an identical write is visible."""
    status = path.stat()
    os.utime(
        path,
        ns=(status.st_atime_ns, max(0, status.st_mtime_ns - 2_000_000_000)),
    )
    return _file_identity(path)


def review_definition(
    path: Path,
    initial: str,
    *,
    expected: bytes | None,
    validate: Callable[[str], None],
    source: Path | None = None,
    store_root: Path | None = None,
    validate_store: Callable[[Path], None] | None = None,
) -> bool:
    """Edit a private draft and publish it after a successful editor write.

    source supplies a local file instead of opening an editor. Interactive edits
    publish after the editor changes or rewrites the draft; exiting without a
    write leaves the stored definition unchanged. Invalid or interrupted work
    remains in a private draft and is offered on the next edit.
    """
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive and source is None:
        raise ValueError("Use an interactive terminal or --file FILE")
    command = None
    if source is None:
        editor = (
            os.environ.get("VISUAL")
            or os.environ.get("EDITOR")
            or shutil.which("nano")
            or shutil.which("vi")
        )
        if not editor or not shlex.split(editor):
            raise ValueError("Set VISUAL or EDITOR, or supply --file FILE")
        command = shlex.split(editor)
    initial_text = source.read_text(encoding="utf-8") if source else initial
    if store_root is None:
        raise ValueError("Definition review requires its definitions-store root")
    draft = definition_draft_path(path, store_root)
    saved = False
    with _draft_lock(draft):
        draft_state = _prepare_draft(
            draft,
            initial_text,
            interactive=interactive,
            explicit_source=source is not None,
        )
        if draft_state is None:
            return False
        try:
            while True:
                if command:
                    before_edit = _arm_write_detection(draft)
                    subprocess.run([*command, str(draft)], check=True)
                    if _file_identity(draft) == before_edit:
                        print("No editor write detected; the stored definition is unchanged.")
                        saved = draft_state == "fresh"
                        return False
                    draft_state = "modified"
                text = draft.read_text(encoding="utf-8")
                from nro.configuration.definition_migrations import normalize_managed_text

                text = normalize_managed_text(path, text)
                draft.write_text(text, encoding="utf-8")
                draft.chmod(0o600)
                try:
                    validate(text)
                except ValueError as error:
                    print(f"Validation failed: {error}", file=sys.stderr)
                    if not command or input("Reopen the draft? [Y/n] ").strip().lower() not in {
                        "",
                        "y",
                        "yes",
                    }:
                        raise
                    continue
                if expected is not None and text.encode("utf-8") == expected:
                    print("No changes.")
                    saved = True
                    return False
                before = expected.decode("utf-8") if expected is not None else ""
                print(
                    "".join(
                        difflib.unified_diff(
                            before.splitlines(keepends=True),
                            text.splitlines(keepends=True),
                            fromfile=str(path),
                            tofile="proposed definition",
                        )
                    ),
                    end="",
                )
                print("Scientific changes may affect artifact freshness on the next assessment.")
                save_definition(
                    path,
                    text,
                    expected=expected,
                    store_root=store_root,
                    validate_store=validate_store,
                )
                saved = True
                print(f"Saved {path}")
                return True
        finally:
            if saved:
                draft.unlink()
            elif draft.exists():
                print("Draft retained for the next edit of this definition.", file=sys.stderr)
