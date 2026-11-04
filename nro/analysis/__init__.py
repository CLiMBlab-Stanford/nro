"""Public interface for study analysis code.

Study analyses import only this package. ``query`` resolves files the way a planned
work item does; ``Inputs`` and ``Outputs`` are the arguments of an analysis entry
function. Importing the package does not read site configuration, so the analysis
runner can use it in any environment that has nro's dependencies.
"""

from nro.analysis.artifacts import Artifact, CiftiMaps, Inputs, InputSet
from nro.analysis.outputs import OutputHandle, Outputs

API_VERSION = 1

__all__ = [
    "API_VERSION",
    "Artifact",
    "CiftiMaps",
    "InputSet",
    "Inputs",
    "OutputHandle",
    "Outputs",
    "ResolutionError",
    "query",
]


def __getattr__(name: str):
    if name in {"query", "ResolutionError"}:
        from nro.analysis import resolve

        return getattr(resolve, name)
    raise AttributeError(f"module 'nro.analysis' has no attribute {name!r}")
