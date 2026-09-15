"""Implement this site hook to convert a selected scan-plan file into nro rows."""

from pathlib import Path

from nro.bidsify.scanplans import ScanPlan


def parse_scanplan(source: Path) -> ScanPlan:
    """Parse one site-defined source file into nro's normalized scan-plan contract."""
    del source
    raise NotImplementedError("This definitions store has no scan-plan parser")
