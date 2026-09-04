"""Repository and site paths used by supported nro modules."""

from __future__ import annotations

import os
from pathlib import Path


LAB_PATH = Path("/juice6/u/nlp/climblab")
CONFIG_PATH = Path(__file__).parent / "files"
BIDS_PATH = Path(os.environ.get("NRO_BIDS_PATH", LAB_PATH / "BIDS"))
WORK_PATH = Path(os.environ.get("NRO_WORK_PATH", LAB_PATH / "WORK"))
REGISTRY_PATH = LAB_PATH / ".nro"
