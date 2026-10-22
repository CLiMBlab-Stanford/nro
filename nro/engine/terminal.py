"""Shared terminal rendering primitives."""

BRAILLE_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def spinner_frame(frame: int) -> str:
    """Return the Braille spinner character for a zero-based frame number."""
    return BRAILLE_SPINNER_FRAMES[frame % len(BRAILLE_SPINNER_FRAMES)]
