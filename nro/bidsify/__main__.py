"""Private noninteractive entry point used by central workers."""

import argparse
import json
import logging
import signal
from pathlib import Path
from types import SimpleNamespace

from nro.engine.io import atomic_write_text

from .errors import BidsificationError
from .stages import run_stage
from .store import IngestionStore


def main():
    """Execute a claimed stage and save a sanitized result for its supervising worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--execution")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def interrupted(_signum, _frame):
        raise InterruptedError("Worker cancellation")

    signal.signal(signal.SIGTERM, interrupted)
    pin = json.loads(args.execution) if args.execution else None
    paths = None
    if pin is not None:
        from nro.configuration.site import settings
        from nro.orchestration.branches import BranchPaths

        values = settings(path=Path(pin["site"]))[0]
        paths = BranchPaths(
            pin["branch"],
            *(Path(values[key]) for key in ("bids", "work", "development")),
        )
        if (
            paths.bids != Path(args.bids_root).resolve()
            or Path(args.control).resolve() != Path(values["registry"]).resolve()
        ):
            raise ValueError("Ingestion paths differ from the scheduler-approved execution")
        if pin["branch"] != "main":
            # The central worker validated the execution pin before launch. The
            # development process receives paths only and never opens scheduler
            # tables through its own Registry implementation.
            registry = SimpleNamespace(
                paths=SimpleNamespace(
                    control=Path(args.control).resolve(),
                    bids_root=paths.bids,
                ),
                detached_ingestion=True,
            )
        else:
            from nro.orchestration.registry import Registry

            registry = Registry.for_project(
                "", bids_root=args.bids_root, registry_path=args.control
            )
    else:
        from nro.orchestration.registry import Registry

        registry = Registry.for_project("", bids_root=args.bids_root, registry_path=args.control)
    store = IngestionStore(registry, branch_paths=paths, execution=pin)
    record = store.get(args.request)
    if record.get("execution") != pin:
        raise ValueError("Claimed ingestion execution differs from the stage invocation")
    if record["state"] != "running":
        raise SystemExit("Request has not been claimed")
    try:
        result = run_stage(record, registry, branch_paths=paths)
    except BidsificationError as error:
        result = {"state": "failed", "issues": [str(error)]}
        print(str(error))
    atomic_write_text(store.root / f"{record['id']}.result", json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Converter and provider wrappers remove their raw diagnostics. Other
        # failures expose only their type, never a DICOM header or API response.
        print(f"Bidsification failed ({type(error).__name__}); staged BIDS was not approved.")
        raise SystemExit(1) from None
