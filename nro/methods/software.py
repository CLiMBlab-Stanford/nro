"""Collect software versions for the supplementary software table."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from nro.methods.context import SoftwareUse

SOFTWARE_COLUMNS = (
    "software",
    "version",
    "source",
    "container",
    "container_digest",
    "modules",
    "citation",
)

# Software provided by each pinned image. Versions apply only when the site image's
# acquisition receipt names the pinned source in ``nro.site.dependencies.IMAGES``.
IMAGE_SOFTWARE: Mapping[str, tuple[str, str, str]] = {
    "qunex": ("QuNex", "1.5.1", ""),
    "synthstrip": ("SynthStrip", "1.7", "hoopes2022synthstrip"),
    "synbold": ("SynBOLD-DisCo", "1.4", "yu2023synbold"),
    "freesurfer": ("FreeSurfer", "7.4.1", "fischl2012freesurfer"),
    "fastsurfer": ("FastSurfer", "2.5.4", "henschel2022fastsurfervinn"),
}

PYTHON_PACKAGES: Mapping[str, str] = {
    "numpy": "",
    "scipy": "",
    "nibabel": "",
    "nitransforms": "",
    "nilearn": "",
    "scikit-learn": "",
}


@dataclass(frozen=True)
class ImageRecord:
    """The configured image for one site key and its acquisition receipt source."""

    key: str
    path: str
    source: str | None


def site_images(keys: Iterable[str]) -> dict[str, ImageRecord]:
    """Read configured image paths and receipt sources from the site settings."""
    from nro.site.configuration import settings

    values = settings()[0]
    records: dict[str, ImageRecord] = {}
    for key in sorted(set(keys)):
        path = values.get(key)
        source = None
        if path:
            receipt = Path(str(path)).with_name(Path(str(path)).name + ".receipt.json")
            if receipt.is_file():
                source = json.loads(receipt.read_text(encoding="utf-8")).get("source")
        records[key] = ImageRecord(key, str(path or ""), source)
    return records


def pinned_sources() -> dict[str, str]:
    """Return the pinned image sources used by installation."""
    from nro.site.dependencies import IMAGES

    return dict(IMAGES)


def software_rows(
    *,
    image_modules: Mapping[str, set[str]],
    observed: Mapping[tuple[str, str, str], set[str]],
    images: Mapping[str, ImageRecord],
    pins: Mapping[str, str],
    nro_version: str,
    package_versions: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Build sorted software-table rows.

    ``image_modules`` maps an image key to the modules that ran it. ``observed``
    maps (software, version, citation) from manifests to modules. A configured
    image whose receipt does not match the pin is listed with an unknown version.
    """
    rows: set[tuple[str, ...]] = set()
    for key, modules in image_modules.items():
        software, version, citation = IMAGE_SOFTWARE.get(key, (key, "", ""))
        record = images.get(key)
        source = record.source if record else None
        pinned = pins.get(key)
        digest = source.split("@", 1)[1] if source and "@" in source else ""
        origin = "pinned image"
        if not source or source != pinned:
            version = "unknown"
            origin = "site image"
        container = source.split("@", 1)[0] if source else (record.path if record else "")
        rows.add(
            (software, version, origin, container, digest, ", ".join(sorted(modules)), citation)
        )
    for (software, version, citation), modules in observed.items():
        match = next((row for row in rows if row[0] == software and row[1] == version), None)
        if match is not None:
            merged = sorted(set(match[5].split(", ")) | modules)
            rows.discard(match)
            rows.add(
                (*match[:2], match[2] + "; manifest", *match[3:5], ", ".join(merged), match[6])
            )
            continue
        rows.add((software, version, "manifest", "", "", ", ".join(sorted(modules)), citation))
    rows.add(("nro", nro_version, "reporting installation", "", "", "", ""))
    for package, version in package_versions.items():
        rows.add(
            (
                package,
                version,
                "reporting installation",
                "",
                "",
                "",
                PYTHON_PACKAGES.get(package, ""),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row[0].casefold(), row[1], row[2])))


def installed_package_versions() -> dict[str, str]:
    """Return versions of the scientific Python packages in this installation."""
    versions = {}
    for package in PYTHON_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def observed_software(
    uses: Iterable[tuple[str, SoftwareUse]],
) -> dict[tuple[str, str, str], set[str]]:
    """Group manifest-observed software by name, version, and citation."""
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for module, use in uses:
        grouped.setdefault((use.software, use.version, use.citation), set()).add(module)
    return grouped
