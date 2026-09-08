"""Errors whose messages are safe to persist in shared ingestion records."""


class BidsificationError(ValueError):
    """Report an operator-actionable failure without raw provider or DICOM content."""
