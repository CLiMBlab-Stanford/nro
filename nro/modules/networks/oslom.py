"""Prepare, run, and parse OSLOM network partitions."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np
from scipy import sparse

from nro.orchestration.runner import Runner, shlex_quote

from .config import OslomConfig

LOG = logging.getLogger(__name__)


def resolve_oslom_executable(configured: Path | None) -> Path:
    """Resolve an existing OSLOM executable."""
    if configured is not None:
        path = configured.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured OSLOM executable not found: {path}. "
                "Provide the executable or update oslom.executable."
            )
        return path
    on_path = shutil.which("oslom_undir")
    if on_path:
        return Path(on_path).resolve()
    raise FileNotFoundError(
        "OSLOM executable 'oslom_undir' was not found on PATH and "
        "oslom.executable is not configured."
    )


def write_oslom_graph(path: Path, adjacency: sparse.spmatrix) -> int:
    """Stream a lower-triangular sparse matrix to OSLOM's edge-list format."""
    lower = sparse.tril(adjacency, k=-1, format="csc")
    lower.sort_indices()
    columns_per_block = 128
    blocks = max(1, (lower.shape[1] + columns_per_block - 1) // columns_per_block)
    LOG.info("Writing OSLOM graph with %d edges in %d blocks", lower.nnz, blocks)
    with path.open("w", encoding="utf8") as f:
        for block_index, start in enumerate(range(0, lower.shape[1], columns_per_block), start=1):
            stop = min(start + columns_per_block, lower.shape[1])
            first = int(lower.indptr[start])
            last = int(lower.indptr[stop])
            counts = np.diff(lower.indptr[start : stop + 1])
            columns = np.repeat(np.arange(start, stop, dtype=np.int32), counts)
            rows = lower.indices[first:last]
            weights = lower.data[first:last]
            values = np.column_stack((columns, rows, weights))
            np.savetxt(f, values, fmt=("%d", "%d", "%.9g"))
            if block_index == 1 or block_index == blocks or block_index % max(1, blocks // 10) == 0:
                LOG.info("OSLOM graph write: %d/%d blocks", block_index, blocks)
    return lower.nnz


def parse_tp(path: Path) -> list[set[int]]:
    """Parse an OSLOM partition file into vertex sets."""
    lines = path.read_text().splitlines()
    groups: list[set[int]] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#") and i + 1 < len(lines):
            values = {int(x) for x in re.findall(r"-?\d+", lines[i + 1])}
            if values:
                groups.append(values)
    return groups


def run_oslom(graph: Path, workdir: Path, cfg: OslomConfig, *, runner: Runner) -> list[set[int]]:
    """Run one OSLOM fit and return its lowest-level communities."""
    workdir.mkdir(parents=True, exist_ok=False)
    local_graph = workdir / "graph.dat"
    os.symlink(graph.resolve(), local_graph)
    cmd = [
        str(cfg.executable),
        "-f",
        local_graph.name,
        "-t",
        str(cfg.significance),
        "-r",
        str(cfg.internal_runs),
        "-hr",
        str(cfg.hierarchical_runs),
    ]
    if cfg.weighted:
        cmd.append("-w")
    else:
        cmd.append("-uw")
    if cfg.initial_partition is not None:
        local_hint = workdir / "hint.dat"
        shutil.copy2(cfg.initial_partition, local_hint)
        cmd.extend(("-hint", local_hint.name))
    cmd.extend(cfg.extra_args)
    stdout = workdir / "stdout.log"
    stderr = workdir / "stderr.log"
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL", *cmd]
    command_text = " ".join(shlex_quote(arg) for arg in cmd)
    runner.run_child(
        [
            "bash",
            "-lc",
            (
                f"{command_text} "
                f"> >(tee {shlex_quote(str(stdout))}) "
                f"2> >(tee {shlex_quote(str(stderr))} >&2)"
            ),
        ],
        cwd=workdir,
        stream_output=True,
        timeout_seconds=cfg.timeout_seconds,
    )
    candidates = [
        workdir / "graph.dat_oslo_files" / "tp_without_singletons",
        workdir / "graph.dat_oslo_files" / "tp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return parse_tp(candidate)
    raise RuntimeError(f"OSLOM produced no tp output in {workdir}")
