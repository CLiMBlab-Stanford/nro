"""Pinned context delivery rejects changed payloads and unresolved inputs."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nro.orchestration.attempt_entry import encode_payload, execute_payload, validate_payload
from nro.orchestration.branches import BranchPaths
from nro.orchestration.execution_context import ExecutionContext, InputBinding


def payload(tmp_path, *, generation=2):
    paths = BranchPaths("dev", tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
    root = paths.bids / "demo/derivatives/parent"
    context = ExecutionContext(
        paths, "demo", "child", (InputBinding("main", "parent", generation, root, root, None),)
    )
    data, digest = encode_payload(
        module="nro.example",
        argv=["--project", "demo"],
        runtime_config=tmp_path / "runtime.yml",
        configuration="science",
        context=context,
    )
    path = tmp_path / "attempt.json"
    path.write_bytes(data)
    return path, digest, context


def test_delivery_sets_configuration_before_import(tmp_path, monkeypatch):
    import os

    import nro.orchestration.attempt_entry as entry

    path, digest, context = payload(tmp_path)
    monkeypatch.setenv("NRO_RUNTIME_CONFIG", "unrelated")
    monkeypatch.setenv("NRO_CONFIGURATION_FINGERPRINT", "unrelated")

    def import_module(name):
        assert name == "nro.example.__main__"
        assert os.environ["NRO_RUNTIME_CONFIG"] == str(tmp_path / "runtime.yml")
        assert os.environ["NRO_CONFIGURATION_FINGERPRINT"] == "science"
        return SimpleNamespace(main=lambda argv, execution_context: (argv, execution_context))

    monkeypatch.setattr(entry.importlib, "import_module", import_module)
    assert execute_payload(path, digest) == (["--project", "demo"], context)


def test_payload_changes_fail_before_import(tmp_path, monkeypatch):
    import nro.orchestration.attempt_entry as entry

    path, digest, _ = payload(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(
        entry.importlib, "import_module", lambda _: pytest.fail("imported changed payload")
    )
    with pytest.raises(ValueError, match="changed"):
        execute_payload(path, digest)


def test_unresolved_local_generation_cannot_launch(tmp_path):
    with pytest.raises(ValueError, match="nonnegative generations"):
        payload(tmp_path, generation=None)


def test_adopted_native_generation_can_launch(tmp_path):
    payload(tmp_path, generation=0)


@pytest.mark.parametrize("bad", ["module", "unknown", "boolean_generation", "debug_input"])
def test_invalid_transport_is_rejected(tmp_path, bad):
    path, _, _ = payload(tmp_path)
    value = json.loads(path.read_bytes())
    if bad == "module":
        value["module"] = "os"
    elif bad == "unknown":
        value["context"]["authority"] = True
    elif bad == "boolean_generation":
        value["context"]["inputs"][0]["generation"] = True
    else:
        value["context"]["inputs"][0]["physical_root"] = str(tmp_path / "DEV/dev/BIDS/demo/sub-01")
    with pytest.raises(ValueError):
        validate_payload(value)


def test_symlinked_transport_is_rejected(tmp_path):
    path, digest, _ = payload(tmp_path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="location"):
        execute_payload(link, digest)


def test_branch_runtime_does_not_open_production_registry(tmp_path, monkeypatch):
    import nro.configuration.site as site
    import nro.orchestration.runtime as runtime
    from nro.orchestration.control_paths import ControlPaths

    _, _, context = payload(tmp_path)
    control = tmp_path / "control"
    config = ControlPaths(control).branch("dev") / "workflows/example.yml"
    config.parent.mkdir(parents=True)
    config.write_text("example: true")
    monkeypatch.setattr(site, "settings", lambda: ({"registry": str(control)}, {}))
    monkeypatch.setattr(
        runtime.Registry, "for_project", lambda *a, **kw: pytest.fail("production registry opened")
    )
    monkeypatch.setenv("NRO_RUNTIME_CONFIG", str(config))
    assert (
        runtime.select_runtime_config(
            project="demo",
            workflow_id="main",
            derivative_class="example",
            execution_context=context,
        )
        == config
    )
    monkeypatch.delenv("NRO_RUNTIME_CONFIG")
    with pytest.raises(ValueError, match="pinned"):
        runtime.select_runtime_config(
            project="demo",
            workflow_id="main",
            derivative_class="example",
            execution_context=context,
        )


def test_two_pinned_catalogs_execute_with_distinct_contexts(tmp_path):
    from nro.orchestration.source_snapshots import SourceStore

    source = Path(__file__).resolve().parents[1] / "nro/orchestration"
    main = tmp_path / "BIDS/demo/derivatives/parent"
    main.mkdir(parents=True)
    (main / "input.txt").write_text("shared input")
    for branch in ("one", "two"):
        checkout = tmp_path / branch
        package = checkout / "nro"
        orchestration = package / "orchestration"
        orchestration.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (orchestration / "__init__.py").write_text("")
        for name in (
            "source_launcher.py",
            "attempt_entry.py",
            "execution_context.py",
            "branches.py",
        ):
            shutil.copyfile(source / name, orchestration / name)
        (orchestration / "runtime.py").write_text(
            'CONFIGURATION_FINGERPRINT_ENV = "NRO_CONFIGURATION_FINGERPRINT"\n'
        )
        module = package / ("extension_" + branch)
        module.mkdir()
        (module / "__init__.py").write_text("")
        implementation = module / "__main__.py"
        implementation.write_text(
            "from pathlib import Path\n"
            "def main(argv, *, execution_context):\n"
            "    source = execution_context.input_path(Path(argv[0]))\n"
            "    target = execution_context.output_path(Path(argv[1]))\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            f'    target.write_text({branch!r} + ":" + source.read_text())\n'
        )
        (checkout / "pyproject.toml").write_text('[project]\nname="nro"\n')
        snapshot = SourceStore(tmp_path / "snapshots").capture(checkout)
        paths = BranchPaths(branch, tmp_path / "BIDS", tmp_path / "WORK", tmp_path / "DEV")
        context = ExecutionContext(
            paths, "demo", "child", (InputBinding("main", "parent", 1, main, main, None),)
        )
        logical = paths.bids / "demo/derivatives/child/output.txt"
        data, digest = encode_payload(
            module="nro.extension_" + branch,
            argv=[str(main / "input.txt"), str(logical)],
            runtime_config=tmp_path / "runtime.yml",
            configuration="science",
            context=context,
        )
        attempt = tmp_path / (branch + ".json")
        attempt.write_bytes(data)
        implementation.write_text('raise RuntimeError("live edits must not be executed")\n')
        command = snapshot.command(
            (
                sys.executable,
                "-m",
                "nro.orchestration.attempt_entry",
                "--payload",
                str(attempt),
                "--digest",
                digest,
            )
        )
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert context.output_path(logical).read_text() == branch + ":shared input"
        assert not logical.exists()
    assert (main / "input.txt").read_text() == "shared input"
