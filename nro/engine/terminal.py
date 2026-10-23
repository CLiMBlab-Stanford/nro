"""Shared terminal rendering primitives."""

import sys
import threading

BRAILLE_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
SPINNER_INTERVAL_SECONDS = 0.08
CLEAR_LINE = "\r\x1b[2K"


def spinner_frame(frame: int) -> str:
    """Return the Braille spinner character for a zero-based frame number."""
    return BRAILLE_SPINNER_FRAMES[frame % len(BRAILLE_SPINNER_FRAMES)]


class SpinnerLine:
    """Animate one replaceable status line on standard error.

    A background thread advances the spinner at a fixed rate, so the animation
    does not depend on how often the caller polls. Nothing is written when
    standard error is not a terminal.
    """

    def __init__(self, *, interval: float = SPINNER_INTERVAL_SECONDS) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._message = ""
        self._frame = 0
        self._thread: threading.Thread | None = None

    def update(self, message: str) -> None:
        """Start the display or replace its current message in place."""
        if not sys.stderr.isatty():
            return
        with self._lock:
            self._message = message
            if self._thread is not None:
                return
            self._stop.clear()
            self._draw()
            self._thread = threading.Thread(
                target=self._animate,
                name="nro-spinner",
                daemon=True,
            )
            self._thread.start()

    def clear(self) -> None:
        """Stop rendering and remove the transient terminal line."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop.set()
        thread.join(timeout=max(1.0, self._interval * 2))
        with self._lock:
            sys.stderr.write(CLEAR_LINE)
            sys.stderr.flush()
            self._thread = None

    def _draw(self) -> None:
        sys.stderr.write(f"{CLEAR_LINE}{spinner_frame(self._frame)} {self._message}")
        sys.stderr.flush()
        self._frame += 1

    def _animate(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                self._draw()
