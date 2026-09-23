"""Store per-user Flywheel credentials beside tracked site definitions."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from nro.engine.io import atomic_write_text

PRIVATE_DIRECTORY = ".definition-secrets"
_SERVER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _server_id(value: str) -> str:
    if not isinstance(value, str) or not _SERVER.fullmatch(value):
        raise ValueError("Flywheel server IDs may contain letters, digits, underscores, or hyphens")
    return value


def _private_directory(path: Path, *, mode: int, owner_only: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"Flywheel credential directories must not be symbolic links: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Invalid Flywheel credential directory: {path}")
    else:
        path.mkdir(mode=mode)
        path.chmod(mode)
    metadata = path.stat()
    if owner_only and metadata.st_uid != os.geteuid():
        raise ValueError(f"Flywheel credential directory belongs to another user: {path}")
    forbidden = 0o007
    if stat.S_IMODE(metadata.st_mode) & forbidden:
        raise ValueError(f"Flywheel credential directory has unsafe permissions: {path}")
    if (stat.S_IMODE(metadata.st_mode) & 0o070) != 0o070:
        raise ValueError(f"Flywheel credential directory is not group-maintainable: {path}")


def credential_path(root: Path, server: str, *, uid: int | None = None) -> Path:
    """Return one user's private key path without reading or creating it."""
    server = _server_id(server)
    user_id = os.geteuid() if uid is None else uid
    if not isinstance(user_id, int) or user_id < 0:
        raise ValueError("Invalid operating-system user ID")
    return (
        Path(root).expanduser().resolve()
        / PRIVATE_DIRECTORY
        / f"user-{user_id}"
        / "flywheel"
        / f"{server}.key"
    )


def normalize_key(value: str, *, host: str) -> str:
    """Validate an ASCII API key and remove an optional matching host prefix."""
    if not isinstance(value, str):
        raise ValueError("Flywheel API keys must be text")
    key = value.strip()
    if ":" in key:
        supplied_host, key = key.split(":", 1)
        if supplied_host != host:
            raise ValueError("Flywheel API key host does not match the selected server")
    try:
        encoded = key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("Flywheel API keys must contain only ASCII characters") from None
    if not encoded or len(encoded) > 4096 or any(byte < 33 or byte > 126 for byte in encoded):
        raise ValueError("Flywheel API key is empty or contains invalid characters")
    return key


def store_key(root: Path, server: str, value: str, *, host: str) -> Path:
    """Atomically replace the current user's key with private permissions."""
    from nro.configuration.definition_migrations import store_lock

    root = Path(root).expanduser().resolve()
    path = credential_path(root, server)
    key = normalize_key(value, host=host)
    with store_lock(root):
        shared = path.parents[2]
        user = path.parents[1]
        flywheel = path.parent
        _private_directory(shared, mode=0o2770)
        _private_directory(user, mode=0o2770, owner_only=True)
        _private_directory(flywheel, mode=0o2770, owner_only=True)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Invalid Flywheel credential path: {path}")
        atomic_write_text(path, key + "\n", mode=0o620, durable=True)
    return path


def read_key(root: Path, server: str, *, host: str) -> str:
    """Read the current user's key after checking ownership and permissions."""
    path = credential_path(root, server)
    shared = path.parents[2]
    user = path.parents[1]
    flywheel = path.parent
    if not shared.exists() or not user.exists() or not flywheel.exists():
        raise ValueError(
            f"No Flywheel key is configured for server {server!r}; run `nro fw addkey {server}`"
        )
    _private_directory(shared, mode=0o2770)
    _private_directory(user, mode=0o2770, owner_only=True)
    _private_directory(flywheel, mode=0o2770, owner_only=True)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise ValueError(
            f"No Flywheel key is configured for server {server!r}; run `nro fw addkey {server}`"
        ) from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Invalid Flywheel credential file: {path}")
        permissions = stat.S_IMODE(metadata.st_mode)
        if metadata.st_uid != os.geteuid() or permissions & 0o057 or (permissions & 0o020) != 0o020:
            raise ValueError(
                f"Flywheel credential file has unsafe ownership or permissions: {path}"
            )
        with os.fdopen(descriptor, "r", encoding="ascii", closefd=False) as stream:
            value = stream.read(4097)
    finally:
        os.close(descriptor)
    if len(value) > 4096:
        raise ValueError(f"Flywheel credential file is unexpectedly large: {path}")
    return normalize_key(value, host=host)


def has_key(root: Path, server: str, *, host: str) -> bool:
    """Return whether the current user has a readable, valid key."""
    try:
        read_key(root, server, host=host)
    except ValueError:
        return False
    return True


def remove_key(root: Path, server: str) -> bool:
    """Remove the current user's key without touching another user's credentials."""
    from nro.configuration.definition_migrations import store_lock

    root = Path(root).expanduser().resolve()
    path = credential_path(root, server)
    with store_lock(root):
        if not path.exists() and not path.is_symlink():
            return False
        _private_directory(path.parents[2], mode=0o2770)
        _private_directory(path.parents[1], mode=0o2770, owner_only=True)
        _private_directory(path.parent, mode=0o2770, owner_only=True)
        if path.is_symlink():
            raise ValueError(f"Refusing a symbolic-link Flywheel credential: {path}")
        metadata = path.stat()
        if not path.is_file() or metadata.st_uid != os.geteuid():
            raise ValueError(f"Invalid Flywheel credential file: {path}")
        path.unlink()
    return True
