"""Validation shared by event catalogs and ingestion review."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def validate_events(path: Path | StringIO, *, duration: float | None = None) -> None:
    """Require finite onset/duration columns and nonnegative event durations.

    Negative onsets are allowed. When run duration in seconds is known,
    reject events beginning at or after its end. Other columns are retained.
    """
    events = pd.read_csv(path, sep='\t')
    if not {'onset', 'duration'} <= set(events) or not len(events):
        raise ValueError('Events need onset and duration columns and at least one event')
    timing = events[['onset', 'duration']].to_numpy(dtype=float)
    if not np.isfinite(timing).all() or (timing[:, 1] < 0).any():
        raise ValueError('Events timing must be finite and durations nonnegative')
    if duration is not None and (timing[:, 0] >= duration).any():
        raise ValueError('An event starts after the recorded run ends; provide corrected events')
