"""Static checks for the seams between planning, orchestration, and execution."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "nro/modules"
ORCHESTRATION = ROOT / "nro/orchestration"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return tuple(names)


def test_scientific_modules_do_not_reach_into_scheduler_state() -> None:
    allowed = {
        "nro.orchestration.catalog",
        "nro.orchestration.contracts",
        "nro.orchestration.execution_context",
        "nro.orchestration.planning_context",
        "nro.orchestration.runner",
        "nro.orchestration.runner_graph",
        "nro.orchestration.runner_support",
        "nro.orchestration.runtime",
    }
    violations = []
    for path in MODULES.rglob("*.py"):
        for name in _imports(path):
            if name.startswith("nro.orchestration") and name not in allowed:
                violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert violations == []


def test_scientific_modules_do_not_import_other_module_packages() -> None:
    violations = []
    for path in MODULES.rglob("*.py"):
        owner = path.relative_to(MODULES).parts[0]
        for name in _imports(path):
            if not name.startswith("nro.modules."):
                continue
            imported = name.split(".", 3)[2]
            if imported != owner and not name.endswith(".contract"):
                violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert violations == []


def test_orchestration_imports_only_module_planning_interfaces() -> None:
    violations = []
    for path in ORCHESTRATION.rglob("*.py"):
        for name in _imports(path):
            if name == "nro.modules":
                continue
            if not name.startswith("nro.modules."):
                continue
            parts = name.split(".")
            if len(parts) < 4 or parts[3] not in {"contract", "planning", "task_models"}:
                violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert violations == []


def test_planning_catalog_does_not_load_array_or_image_stacks(
    tmp_path: Path, definitions_fixture: Path
) -> None:
    site_config = tmp_path / "site.toml"
    site_config.write_text(f'definitions = "{definitions_fixture}"\n', encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    code = (
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import nro.configuration.site as site\n"
        f"site.CHECKOUT = Path({str(checkout)!r})\n"
        "site.installation_record = lambda root=site.CHECKOUT: {}\n"
        f"os.environ['NRO_SITE_CONFIG'] = {str(site_config)!r}\n"
        "import nro.orchestration.catalog\n"
        "forbidden = {'numpy', 'scipy', 'pandas', 'nibabel'}\n"
        "loaded = sorted(forbidden.intersection(sys.modules))\n"
        "if loaded:\n"
        "    raise SystemExit(', '.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_functional_stage_planners_do_not_import_or_mutate_runner() -> None:
    path = MODULES / "func" / "construction.py"
    source = path.read_text(encoding="utf-8")
    assert "nro.orchestration.runner" not in _imports(path)
    assert ".add_step(" not in source
    assert ".add_steps(" not in source


def test_registry_operations_leave_connection_ownership_to_registry() -> None:
    operation_files = (
        ORCHESTRATION / "registry_status.py",
        ORCHESTRATION / "registry_work_items.py",
    )
    violations: list[str] = []
    for path in operation_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"connect", "commit", "rollback"}:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} calls {node.func.attr}"
                    )
    assert violations == []
