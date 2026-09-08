"""Dependency-free SVG design plots and numerical design exports."""

from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from nro.engine.io import atomic_write_text, atomic_write_json


def write_design(prefix: Path, design) -> tuple[Path, ...]:
    """Save a resolved design, its coefficient mapping, and its labeled plot."""
    numerical = prefix.with_name(prefix.name + "_design.tsv")
    plot = prefix.with_name(prefix.name + "_design.svg")
    info = prefix.with_name(prefix.name + "_design.json")
    names = design.metadata["FitColumns"]
    expanded = np.zeros((len(design.retained), design.matrix.shape[1]))
    expanded[design.retained] = design.matrix
    atomic_write_text(numerical, pd.DataFrame(expanded, columns=names).to_csv(sep="\t", index=False))
    atomic_write_json(info, {**design.metadata, "CoefficientMap": design.coefficient_map.tolist()})
    design_plot(plot, expanded, names, design.retained)
    return numerical, plot, info


def design_plot(path: Path, matrix: np.ndarray, names: list[str], retained: np.ndarray) -> None:
    """Plot actual retained-frame fit columns and mark excluded frames in gray."""
    values = np.asarray(matrix, dtype=float)
    width = max(640, 18 * values.shape[1] + 100)
    height = max(360, min(1200, 2 * values.shape[0]))
    left, top = 65, 160
    dx, dy = (width - left - 20) / max(1, values.shape[1]), height / len(values)
    scale = np.maximum(np.max(np.abs(values), axis=0), np.finfo(float).eps)
    normalized = values / scale
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + top + 40}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="12" y="22" font-size="14">Fit design before voxel-specific whitening</text>']
    for column, name in enumerate(names):
        x = left + (column + .5) * dx
        parts.append(f'<text transform="translate({x:.2f},150) rotate(-65)" font-size="10">{escape(name)}</text>')
    for row in range(len(values)):
        for column in range(values.shape[1]):
            v = normalized[row, column]
            fade = int(255 * (1 - abs(v)))
            color = f"rgb(255,{fade},{fade})" if v >= 0 else f"rgb({fade},{fade},255)"
            if not retained[row]:
                color = "#888888"
            parts.append(f'<rect x="{left + column * dx:.2f}" y="{top + row * dy:.2f}" width="{dx + .1:.2f}" height="{dy + .1:.2f}" fill="{color}"/>')
    parts.extend([f'<text x="12" y="{top + height + 25}" font-size="11">Rows: original frames. Gray: censored. Columns scaled only for display.</text>', '</svg>'])
    atomic_write_text(path, "\n".join(parts))
