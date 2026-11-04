"""Resolved analysis inputs and the standard readers for them."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def file_extension(path: str | Path) -> str:
    """Return a BIDS-style extension, including compound ones such as ``.dscalar.nii``."""
    name = Path(path).name
    for extension in (
        ".dscalar.nii",
        ".dtseries.nii",
        ".dlabel.nii",
        ".pscalar.nii",
        ".ptseries.nii",
        ".func.gii",
        ".shape.gii",
        ".label.gii",
        ".surf.gii",
        ".nii.gz",
        ".tsv.gz",
    ):
        if name.endswith(extension):
            return extension
    return Path(name).suffix


def file_entities(path: str | Path) -> dict[str, str]:
    """Parse BIDS entities plus ``suffix`` and ``extension`` from a filename."""
    name = Path(path).name
    extension = file_extension(name)
    stem = name[: len(name) - len(extension)] if extension else name
    entities: dict[str, str] = {}
    parts = stem.split("_")
    for part in parts:
        key, separator, value = part.partition("-")
        if separator and key and value:
            entities[key] = value
    if parts and "-" not in parts[-1]:
        entities["suffix"] = parts[-1]
    entities["extension"] = extension
    return entities


@dataclass(frozen=True)
class CiftiMaps:
    """A multi-map dense-scalar CIFTI with its indexed sidecar."""

    path: Path
    names: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]
    image: Any

    @property
    def array(self) -> Any:
        """Return every map as a ``(maps, grayordinates)`` float32 array."""
        import numpy as np

        return np.asarray(self.image.dataobj, dtype=np.float32)

    @property
    def brain_axis(self) -> Any:
        """Return the CIFTI brain-model axis."""
        return self.image.header.get_axis(1)

    def indices(self, **fields: object) -> tuple[int, ...]:
        """Return the indices of maps whose sidecar records match every field."""
        if not fields:
            raise TypeError("Give at least one sidecar field, such as Contrast='S-N'")
        result = []
        for index, record in enumerate(self.records):
            for key, expected in fields.items():
                actual = record.get(key)
                values = actual if isinstance(actual, list) else [actual]
                if expected not in values:
                    break
            else:
                result.append(index)
        return tuple(result)

    def map(self, **fields: object) -> Any:
        """Return the one map whose sidecar record matches every field."""
        import numpy as np

        indices = self.indices(**fields)
        if len(indices) != 1:
            described = ", ".join(f"{key}={value!r}" for key, value in fields.items())
            raise KeyError(f"Expected one map for {described} in {self.path}, found {len(indices)}")
        return np.asarray(self.image.dataobj[indices[0]], dtype=np.float32)


def read_cifti_maps(path: Path) -> CiftiMaps:
    """Read a dense-scalar CIFTI and its indexed sidecar."""
    import nibabel as nib

    from nro.engine.cifti import indexed_cifti_sidecar, validate_indexed_cifti_sidecar

    image = nib.load(str(path))
    axis = image.header.get_axis(0)
    names = tuple(str(name) for name in axis.name)
    sidecar = indexed_cifti_sidecar(path)
    if sidecar.is_file():
        document = validate_indexed_cifti_sidecar(path)
        index = document["IndexToMetadata"]
        records = tuple(dict(index[str(position)]) for position in range(len(names)))
    else:
        records = tuple({"Name": name} for name in names)
    return CiftiMaps(Path(path), names, records, image)


def load_path(path: Path) -> Any:
    """Read one file with the reader chosen from its extension."""
    extension = file_extension(path)
    if extension == ".dscalar.nii":
        return read_cifti_maps(path)
    if extension in {".nii", ".nii.gz"} or extension.endswith(".nii"):
        import nibabel as nib

        return nib.load(str(path))
    if extension.endswith(".gii"):
        import nibabel as nib
        import numpy as np

        image = nib.load(str(path))
        arrays = [np.asarray(item.data) for item in image.darrays]
        return arrays[0] if len(arrays) == 1 else np.stack(arrays)
    if extension in {".tsv", ".tsv.gz"}:
        import pandas as pd

        return pd.read_csv(path, sep="\t")
    if extension == ".json":
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if extension == ".npy":
        import numpy as np

        return np.load(path, mmap_mode="r", allow_pickle=False)
    raise ValueError(f"No standard reader for {path.name}; read artifact.path directly")


@dataclass(frozen=True)
class Artifact:
    """One resolved input file."""

    path: Path
    entities: Mapping[str, str]
    module: str | None
    signature: str
    project: str
    participant: str
    lineage: str | None = None

    def load(self) -> Any:
        """Read the file with the standard reader for its extension."""
        return load_path(self.path)

    def as_spec(self) -> dict[str, Any]:
        """Serialize for an execution specification."""
        return {
            "path": str(self.path),
            "entities": dict(self.entities),
            "module": self.module,
            "signature": self.signature,
            "project": self.project,
            "participant": self.participant,
            "lineage": self.lineage,
        }

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> "Artifact":
        """Decode one serialized artifact."""
        return cls(
            path=Path(value["path"]),
            entities=dict(value["entities"]),
            module=value.get("module"),
            signature=str(value["signature"]),
            project=str(value["project"]),
            participant=str(value["participant"]),
            lineage=value.get("lineage"),
        )


@dataclass(frozen=True)
class InputSet(Sequence[Artifact]):
    """An ordered sequence of artifacts resolved for one ``many: true`` input."""

    artifacts: tuple[Artifact, ...] = field(default=())

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return InputSet(self.artifacts[index])
        return self.artifacts[index]

    def __len__(self) -> int:
        return len(self.artifacts)

    def __iter__(self) -> Iterator[Artifact]:
        return iter(self.artifacts)

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the artifact paths in order."""
        return tuple(item.path for item in self.artifacts)

    def group_by(self, entity: str) -> dict[str | None, "InputSet"]:
        """Group artifacts by one entity value, in order of first appearance."""
        groups: dict[str | None, list[Artifact]] = {}
        for artifact in self.artifacts:
            groups.setdefault(artifact.entities.get(entity), []).append(artifact)
        return {key: InputSet(tuple(value)) for key, value in groups.items()}


Inputs = Mapping[str, "Artifact | InputSet | None"]
