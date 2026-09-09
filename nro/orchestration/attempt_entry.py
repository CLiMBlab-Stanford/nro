"""Deliver a pinned execution context to a module in its own Python process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
from pathlib import Path

from nro.orchestration.execution_context import ExecutionContext


def validate_payload(value: dict) -> ExecutionContext:
    """Check transport shape and require resolved input generations.

    A payload is execution data, not a grant of authority. The scheduler must
    authorize ownership, pin its digest, and fence the selected generations
    before launching this entry point.
    """
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "protocol",
            "module",
            "argv",
            "runtime_config",
            "configuration",
            "context",
        }
        or type(value["protocol"]) is not int
        or value["protocol"] != 1
    ):
        raise ValueError("Unsupported attempt payload")
    if (
        not isinstance(value["module"], str)
        or not re.fullmatch(
            r"nro\.[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", value["module"]
        )
        or not isinstance(value["argv"], list)
        or not all(isinstance(arg, str) for arg in value["argv"])
        or not isinstance(value["runtime_config"], str)
        or not Path(value["runtime_config"]).is_absolute()
        or not isinstance(value["configuration"], str)
        or not value["configuration"]
    ):
        raise ValueError("Invalid attempt execution fields")
    context = ExecutionContext.from_dict(value["context"])
    if context.as_dict() != value["context"]:
        raise ValueError("Invalid execution context fields")
    for binding in context.inputs:
        if type(binding.generation) is not int or binding.generation < 0:
            raise ValueError("Attempt inputs require resolved nonnegative generations")
        member = (binding.prefix + "_binding") if binding.prefix else "_binding"
        context.input_path(binding.logical_root / member)
    return context


def encode_payload(
    *,
    module: str,
    argv: list[str],
    runtime_config: Path,
    configuration: str,
    context: ExecutionContext,
) -> tuple[bytes, str]:
    """Return canonical payload bytes and their digest for scheduler-owned storage."""
    value = dict(
        protocol=1,
        module=module,
        argv=argv,
        runtime_config=str(runtime_config),
        configuration=configuration,
        context=context.as_dict(),
    )
    validate_payload(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return data, hashlib.sha256(data).hexdigest()


def execute_payload(path: Path, digest: str) -> object:
    """Verify immutable execution data before importing and invoking its module."""
    path = Path(path).expanduser().absolute()
    if path.resolve() != path or not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid attempt payload location or digest")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Attempt payload changed after scheduling")
    value = json.loads(data)
    context = validate_payload(value)
    # Set these before module import: configuration defaults may load on import.
    from nro.orchestration.runtime import CONFIGURATION_FINGERPRINT_ENV

    os.environ["NRO_RUNTIME_CONFIG"] = value["runtime_config"]
    os.environ[CONFIGURATION_FINGERPRINT_ENV] = value["configuration"]
    implementation = importlib.import_module(value["module"] + ".__main__")
    return implementation.main(value["argv"], execution_context=context)


def main(argv=None) -> None:
    """Run the module selected by a scheduler-pinned payload and propagate failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--digest", required=True)
    args = parser.parse_args(argv)
    result = execute_payload(args.payload, args.digest)
    if isinstance(result, int) and result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
