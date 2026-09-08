"""Repository and site paths used by supported nro modules."""

from __future__ import annotations

from pathlib import Path
from nro.configuration.site import settings


LAB_PATH = Path("/juice6/u/nlp/climblab")
_SITE, _ = settings()
BIDS_PATH = Path(_SITE["bids"])
WORK_PATH = Path(_SITE["work"])
REGISTRY_PATH = Path(_SITE["registry"])
WB_COMMAND_PATH = Path(_SITE["workbench"])
