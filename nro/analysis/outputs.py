"""Output slots that analysis code writes to."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nro.engine.io import atomic_output_path, atomic_write_text


def _plain(value: Any) -> Any:
    """Convert numpy scalars and arrays to JSON and TSV values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class OutputHandle:
    """One fixed output path. Writers stage the file and move it into place."""

    slot: str
    path: Path

    def _prepare(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_text(self, text: str) -> Path:
        """Write text atomically."""
        self._prepare()
        atomic_write_text(self.path, text)
        return self.path

    def write_json(self, value: Any) -> Path:
        """Write a JSON document atomically."""
        return self.write_text(json.dumps(value, indent=2, sort_keys=True, default=_plain) + "\n")

    def write_tsv(self, rows: Any, columns: Sequence[str] | None = None) -> Path:
        """Write rows (mappings) or a pandas DataFrame as a tab-separated table."""
        if hasattr(rows, "to_csv"):
            buffer = io.StringIO()
            rows.to_csv(buffer, sep="\t", index=False, na_rep="n/a")
            return self.write_text(buffer.getvalue())
        rows = [dict(row) for row in rows]
        if columns is None:
            columns = []
            for row in rows:
                columns.extend(key for key in row if key not in columns)
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer, fieldnames=list(columns), delimiter="\t", lineterminator="\n", restval="n/a"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _plain(value) for key, value in row.items()})
        return self.write_text(buffer.getvalue())

    def write_nifti(self, data: Any, *, like: Any) -> Path:
        """Write a NIfTI image with the affine and header of ``like``."""
        import nibabel as nib

        reference = like.load() if hasattr(like, "load") else like
        image = nib.Nifti1Image(data, reference.affine, reference.header)
        self._prepare()
        with atomic_output_path(self.path) as staged:
            nib.save(image, str(staged))
        return self.path

    def write_cifti(self, maps: Any, *, like: Any, names: Sequence[str] | None = None) -> Path:
        """Write a dense-scalar CIFTI on the brain axis of ``like``."""
        import nibabel as nib
        import numpy as np

        reference = like.load() if hasattr(like, "load") else like
        image = reference.image if hasattr(reference, "brain_axis") else reference
        brain_axis = image.header.get_axis(1)
        values = np.atleast_2d(np.asarray(maps, dtype=np.float32))
        names = (
            list(names) if names is not None else [f"map{index}" for index in range(len(values))]
        )
        header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), brain_axis))
        self._prepare()
        with atomic_output_path(self.path) as staged:
            nib.save(nib.Cifti2Image(values, header=header, dtype=np.float32), str(staged))
        return self.path


Outputs = Mapping[str, OutputHandle]
