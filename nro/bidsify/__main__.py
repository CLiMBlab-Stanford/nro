"""Private noninteractive entry point used by central workers."""

import argparse
import json
import logging
import signal

from nro.engine.io import atomic_write_text
from nro.orchestration.registry import Registry
from .store import IngestionStore
from .pipeline import run_stage
from .errors import BidsificationError


def main():
    """Execute a claimed stage and save a sanitized result for its supervising worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--bids-root', required=True)
    parser.add_argument('--control', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    def interrupted(_signum, _frame):
        raise InterruptedError('Worker cancellation')
    signal.signal(signal.SIGTERM, interrupted)
    registry = Registry.for_project('', bids_root=args.bids_root, registry_path=args.control)
    store = IngestionStore(registry)
    record = store.get(args.request)
    if record['state'] != 'running':
        raise SystemExit('Request has not been claimed')
    try:
        result = run_stage(record, registry)
    except BidsificationError as error:
        result = {'state': 'failed', 'issues': [str(error)]}
        print(str(error))
    atomic_write_text(store.root / f"{record['id']}.result", json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Converter and provider wrappers remove their raw diagnostics. Other
        # failures expose only their type, never a DICOM header or API response.
        print(f'Bidsification failed ({type(error).__name__}); staged BIDS was not approved.')
        raise SystemExit(1) from None
