"""Keep the public API documented and its introductory example executable."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_has_docstrings() -> None:
    missing = []
    for path in sorted((ROOT / "nro").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{path.relative_to(ROOT)}:1 module")
        for declaration in tree.body:
            if isinstance(declaration, ast.ClassDef) and not declaration.name.startswith("_"):
                public = [declaration]
                public.extend(
                    method
                    for method in declaration.body
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and (not method.name.startswith("_") or method.name == "__init__")
                )
            elif isinstance(
                declaration, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and not declaration.name.startswith("_"):
                public = [declaration]
            else:
                continue
            for item in public:
                if not ast.get_docstring(item):
                    missing.append(f"{path.relative_to(ROOT)}:{item.lineno} {item.name}")
    assert not missing, "Missing API docstrings:\n" + "\n".join(missing)


def test_runner_documentation_example(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "NRO_STEP_LEDGER",
        "NRO_RUNNER_GRAPH_SIGNATURE",
    ):
        monkeypatch.delenv(name, raising=False)
    page = (ROOT / "docs" / "api.md").read_text()
    examples = re.findall(r"```python\n(.*?)\n```", page, flags=re.DOTALL)
    assert len(examples) == 1
    namespace = {"__name__": "__documentation_example__"}
    exec(compile(examples[0], "docs/api.md", "exec"), namespace)
    assert (tmp_path / "result.txt").read_text() == "complete\n"
