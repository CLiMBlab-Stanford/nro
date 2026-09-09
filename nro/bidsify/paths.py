"""Create ingestion directories without importing image-processing libraries."""

import os
import stat
from pathlib import Path

from .errors import BidsificationError


def secure_directory(path: Path) -> Path:
    """Create central group-owned staging; reject symlinks and unsafe permissions."""
    if not path.is_absolute():
        raise BidsificationError("Staging path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            current.mkdir(mode=0o2770)
            current.chmod(0o2770)
        except FileExistsError:
            pass
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise BidsificationError("Staging path contains a symlink or non-directory")
        if current in (Path("/tmp"), Path("/")):
            continue
        if info.st_mode & stat.S_IWOTH:
            raise BidsificationError("Staging parent is writable by everyone")
        if current.is_relative_to("/tmp/nro"):
            if info.st_mode & stat.S_IRWXO:
                raise BidsificationError(
                    "Anatomical staging must not be accessible outside its owner and group"
                )
            if info.st_uid not in {0, os.getuid()} and info.st_gid not in {
                os.getgid(),
                *os.getgroups(),
            }:
                raise BidsificationError(
                    "Anatomical staging is not owned by your user or a shared group"
                )
    return path
