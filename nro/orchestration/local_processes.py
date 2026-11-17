"""Check whether a process on this host still exists."""

from __future__ import annotations

import os
from typing import Literal

ProcessState = Literal["absent", "present", "unknown"]


def process_state(pid: int) -> ProcessState:
    """Probe a local PID without sending it a signal.

    Return "absent" only when the kernel reports no such process. A process
    owned by another user is "present". Any other error is "unknown", and each
    caller decides how to treat it. A present PID may have been reused by an
    unrelated process.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "present"
    except OSError:
        return "unknown"
    return "present"
