"""Read branch-visible outputs without loading the shared scheduler schema."""

from pathlib import Path


def registered_rows(bids_root: Path) -> list[dict] | None:
    """Return authorized cached rows, or None before central activation."""
    from nro.configuration.site import CHECKOUT, installation_record, settings
    from nro.orchestration.scheduler_client import status
    from nro.orchestration.scheduler_implementation import implementation_path

    values = settings()[0]
    control = Path(values["registry"])
    if installation_record().get("mode") != "branch" and not implementation_path(control).is_file():
        return None
    if bids_root.resolve() != Path(values["bids"]).resolve():
        raise ValueError("Branch inspection uses the shared site BIDS root")
    report = status(control, bids_root, checkout=CHECKOUT, mode="cached")
    visible = set(report["visible_ids"])
    return [row for row in report["rows"] if row["id"] in visible]
