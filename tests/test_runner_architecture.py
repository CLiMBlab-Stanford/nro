"""Guard the unified module-runner contract against architectural regression."""

from pathlib import Path
import ast


ROOT = Path(__file__).parents[1]
MODULE_FILES = (
    ROOT / "nro/anat/module.py",
    ROOT / "nro/func/module.py",
    ROOT / "nro/clean/module.py",
    ROOT / "nro/microparcellation/module.py",
    ROOT / "nro/networks/module.py",
)


def test_every_module_uses_the_shared_runner_context() -> None:
    for path in MODULE_FILES:
        source = path.read_text(encoding="utf-8")
        assert "Runner" in source, path
        assert ".run_context(" in source, path


def test_modules_construct_declarative_runner_graphs_without_freshness_checks() -> None:
    forbidden = (
        "artifact_decision(",
        "RunnerNode(",
        ".step_decision(",
        ".needs_update(",
        ".python_artifact(",
        ".command_artifact(",
        ".directory_artifact(",
        ".mark_stale(",
        "runner._log_command",
        "runner.log_python_step(",
        "runner.log_python_success(",
        "runner.log_python_failure(",
        "subprocess.run(",
        "subprocess.Popen(",
    )
    failures: list[str] = []
    for path in MODULE_FILES:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                failures.append(f"{path.relative_to(ROOT)} contains {token!r}")
    assert not failures, "\n".join(failures)

    for path in MODULE_FILES:
        source = path.read_text(encoding="utf-8")
        assert "RunnerGraph(" not in source, path
        assert "def build_module(" in source, path


def test_runner_owns_graph_and_module_builders_own_dag_mutation() -> None:
    runner_source = (ROOT / "nro/orchestration/runner.py").read_text(encoding="utf-8")
    assert "self._graph = RunnerGraph(module_name)" in runner_source

    offenders: list[str] = []
    for path in MODULE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_step"
            ):
                continue
            owner: ast.AST = node
            while owner in parents and not isinstance(owner, ast.FunctionDef):
                owner = parents[owner]
            if not isinstance(owner, ast.FunctionDef) or owner.name != "build_module":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, f"DAG mutation occurs outside build_module(): {offenders}"


def test_no_local_module_freshness_helpers_remain() -> None:
    forbidden = ("def _step_decision(", "def _would_rerun(", "def _invalidate_paths(")
    failures = [
        f"{path.relative_to(ROOT)} contains {token!r}"
        for path in MODULE_FILES
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]
    assert not failures, "\n".join(failures)


def test_repository_defines_exactly_one_runner_graph_engine() -> None:
    path = ROOT / "nro/orchestration/runner_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    graph_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("RunnerGraph")
    ]
    assert graph_classes == ["RunnerGraph"]


def test_repository_defines_exactly_one_runner_class() -> None:
    definitions = []
    for path in (ROOT / "nro").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions.extend(
            path.relative_to(ROOT)
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Runner"
        )
    assert definitions == [Path("nro/orchestration/runner.py")]


def test_module_steps_declare_paths_not_display_basenames() -> None:
    failures: list[str] = []
    for path in MODULE_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            for keyword in call.keywords:
                if keyword.arg != "outputs":
                    continue
                for value in ast.walk(keyword.value):
                    if isinstance(value, ast.Attribute) and value.attr == "name":
                        failures.append(
                            f"{path.relative_to(ROOT)}:{value.lineno} passes a basename as an output"
                        )
    assert not failures, "\n".join(failures)
