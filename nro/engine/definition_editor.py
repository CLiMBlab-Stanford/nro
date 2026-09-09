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


def save_definition(path: Path, text: str, *, expected: bytes | None) -> None:
    """Atomically publish text if the target still matches its snapshot.

    Serialize cooperating writers with a persistent sibling lock. Direct edits
    do not honor that lock, but their content is checked before replacement.
    Existing permissions are preserved. New files are readable by other users.
    """
    with _definition_lock(path, expected):
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


def delete_definition(path: Path, *, expected: bytes) -> Path:
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
        path.unlink()
        return backup


def review_definition(
    path: Path,
    initial: str,
    *,
    expected: bytes | None,
    validate: Callable[[str], None],
    source: Path | None = None,
    yes: bool = False,
) -> bool:
    """Edit a private draft, validate it, show a diff, and confirm publication.

    source supplies a local file instead of opening an editor. Noninteractive
    publication requires source and yes. Unpublished drafts are retained and
    their location is printed, including on cancellation or errors.
    """
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not interactive and (source is None or not yes):
        raise ValueError(
            "Use an interactive terminal, or --file FILE --yes; create --output FILE saves a local draft"
        )
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
    directory = Path(tempfile.mkdtemp(prefix="nro-definition-"))
    draft = directory / path.name
    saved = False
    try:
        draft.write_text(initial_text, encoding="utf-8")
        while True:
            if command:
                subprocess.run([*command, str(draft)], check=True)
            text = draft.read_text(encoding="utf-8")
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
            if not yes and input(f"Save {path}? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Cancelled; the stored definition is unchanged.")
                return False
            save_definition(path, text, expected=expected)
            saved = True
            print(f"Saved {path}")
            return True
    finally:
        if saved:
            draft.unlink()
            directory.rmdir()
        elif draft.exists():
            print(f"Draft retained at {draft}", file=sys.stderr)
        else:
            directory.rmdir()
