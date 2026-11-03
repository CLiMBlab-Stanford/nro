"""Execute one analysis work item from its JSON specification.

Run as ``python -m nro.analysis.runner SPEC``. The specification names the study
code snapshot, the entry function, the resolved inputs, the parameters, and the
staging path of each output slot.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from nro.analysis.artifacts import Artifact, InputSet
from nro.analysis.outputs import OutputHandle


def build_inputs(value: Mapping[str, Any]) -> dict[str, Artifact | InputSet | None]:
    """Decode the ``inputs`` section of a specification."""
    result: dict[str, Artifact | InputSet | None] = {}
    for name, item in value.items():
        if item is None:
            result[name] = None
        elif isinstance(item, list):
            result[name] = InputSet(tuple(Artifact.from_spec(entry) for entry in item))
        else:
            result[name] = Artifact.from_spec(item)
    return result


def load_entry(code_root: Path, analysis: str, entry: str):
    """Import ``analysis.py`` with only the study's ``common/`` and analysis directory added."""
    directory = code_root / "analyses" / analysis
    common = code_root / "common"
    for path in (common, directory):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module_spec = importlib.util.spec_from_file_location(
        f"nro_study_analysis_{analysis}", directory / "analysis.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load {directory / 'analysis.py'}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    function = getattr(module, entry, None)
    if not callable(function):
        raise AttributeError(f"{directory / 'analysis.py'} does not define {entry}()")
    return function


def run_specification(path: Path) -> None:
    """Run the entry function of one specification."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    function = load_entry(Path(spec["code_root"]), spec["analysis"].split("/", 1)[1], spec["entry"])
    outputs = {slot: OutputHandle(slot, Path(value)) for slot, value in spec["outputs"].items()}
    function(build_inputs(spec["inputs"]), dict(spec["parameters"]), outputs)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point used inside the analysis environment."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m nro.analysis.runner SPEC")
    run_specification(Path(arguments[0]))


if __name__ == "__main__":
    main()
