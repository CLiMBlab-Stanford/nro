"""Guardrails for the single runner-execution implementation."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "nro"
_MODULES = _ROOT / "modules"
_RUNNER_AREAS = (
    _MODULES / "anat",
    _MODULES / "func",
    _MODULES / "clean",
    _MODULES / "dynconn",
    _MODULES / "microparcellation",
    _MODULES / "networks",
)


def test_modules_do_not_launch_subprocesses_outside_runner() -> None:
    """All tool execution must be logged and failure-managed by Runner."""
    offenders = []
    for area in _RUNNER_AREAS:
        for path in area.rglob("*.py"):
            if "subprocess.run" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(_ROOT))
    assert not offenders, f"Runner subprocess bypasses Runner: {offenders}"


def test_module_steps_do_not_use_untracked_none_outputs() -> None:
    """Resumable module steps must declare concrete artifacts or breadcrumbs."""
    offenders = []
    for area in _RUNNER_AREAS:
        for path in area.rglob("*.py"):
            if "outputs=None" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(_ROOT))
    assert not offenders, f"Runner steps with untracked outputs: {offenders}"


def test_module_steps_do_not_declare_empty_output_tuples() -> None:
    """Every DAG step is an artifact node with a fixed nonempty output set."""
    offenders = []
    for area in _RUNNER_AREAS:
        for path in area.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                for keyword in call.keywords:
                    if (
                        keyword.arg == "outputs"
                        and isinstance(keyword.value, (ast.Tuple, ast.List))
                        and not keyword.value.elts
                    ):
                        offenders.append(f"{path.relative_to(_ROOT)}:{call.lineno}")
    assert not offenders, f"Runner steps with empty output contracts: {offenders}"


def test_scientific_modules_do_not_discover_derivative_members_with_globs() -> None:
    """Downstream modules consume fixed publication manifests."""
    checked = (
        _MODULES / "clean" / "module.py",
        _MODULES / "dynconn" / "module.py",
        _MODULES / "microparcellation" / "module.py",
        _MODULES / "microparcellation" / "__main__.py",
        _MODULES / "networks" / "module.py",
        _MODULES / "networks" / "__main__.py",
    )
    offenders = []
    for path in checked:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr in {"glob", "rglob"}:
                if isinstance(call.func.value, ast.Name) and call.func.value.id == "fsaverage_dir":
                    continue  # Fixed TemplateFlow resource lookup, not a derivative.
                offenders.append(f"{path.relative_to(_ROOT)}:{call.lineno}")
    assert not offenders, f"Derivative discovery remains in scientific modules: {offenders}"


def test_opaque_directory_producers_use_shared_directory_lifecycle() -> None:
    """Directory-producing steps must not implement cleanup ad hoc."""
    expected = {
        _MODULES / "anat" / "steps.py": {"_create_recon_all_step"},
        _MODULES / "func" / "steps.py": {
            "_create_robust_bold_reference_step",
            "_create_topup_dfout_step",
            "_create_ica_aroma_workflow_step",
            "_create_ants_registration_step",
        },
        _MODULES / "networks" / "module.py": {"build_module"},
    }
    offenders: list[str] = []
    for path, names in expected.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in names:
            node = functions.get(name)
            uses_directory_step = node is not None and any(
                isinstance(candidate, ast.Call)
                and (
                    isinstance(candidate.func, ast.Attribute)
                    and candidate.func.attr == "directory_step"
                )
                for candidate in ast.walk(node)
            )
            if not uses_directory_step:
                offenders.append(f"{path.relative_to(_ROOT)}:{name}")
    assert not offenders, f"Opaque directory producers bypass shared lifecycle: {offenders}"
