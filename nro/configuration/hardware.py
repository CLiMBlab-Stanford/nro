"""Match BIDS acquisition metadata to site gradient-unwarping policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nro.configuration.parsing import parse_mapping
from nro.configuration.site import definitions_root, settings

CATALOG_RELATIVE_PATH = Path("hardware/gradient_unwarping.yml")
SUPPORTED_ACTIONS = frozenset({"unwarp", "already_corrected"})
GRADIENT_UNWARP_METHOD = "HCP GradientDistortionUnwarp.sh"
GRADIENT_UNWARP_IMAGE = (
    "docker://flywheel/hcp-base@"
    "sha256:34b37286be1a1c3be5d763ae1910631b0c266f40992ff60b58659871041db68c"
)


@dataclass(frozen=True)
class GradientUnwarpingResolution:
    """Resolved acquisition policy and, when needed, its coefficient resource."""

    mode: str
    applied: bool
    reason: str
    profile: str | None
    action: str | None
    matched_metadata: dict[str, object]
    coefficients: Path | None
    coefficient_sha256: str | None
    override_existing_correction: bool

    def scientific_record(self) -> dict[str, object]:
        """Return path-independent provenance suitable for artifact contracts."""
        return {
            "mode": self.mode,
            "applied": self.applied,
            "reason": self.reason,
            "profile": self.profile,
            "action": self.action,
            "matched_metadata": dict(self.matched_metadata),
            "coefficient_sha256": self.coefficient_sha256,
            "override_existing_correction": self.override_existing_correction,
            "method": GRADIENT_UNWARP_METHOD if self.applied else None,
            "tool_image": GRADIENT_UNWARP_IMAGE if self.applied else None,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_catalog(root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    base = definitions_root() if root is None else Path(root).expanduser().resolve()
    path = base / CATALOG_RELATIVE_PATH
    if root is None and not path.is_file():
        shared = Path(settings()[0]["definitions"]).expanduser().resolve()
        shared_path = shared / CATALOG_RELATIVE_PATH
        if shared_path.is_file():
            path = shared_path
    if not path.is_file():
        packaged = Path(__file__).parent / "starters" / CATALOG_RELATIVE_PATH
        path = packaged
    document = parse_mapping(path.read_text(encoding="utf-8"), source=str(path))
    if document.get("version") != 1:
        raise ValueError(f"{path}: version must be 1")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{path}: profiles must be a mapping")
    return path, profiles


def _matches(expected: object, actual: object) -> bool:
    if isinstance(expected, Mapping):
        unknown = set(expected) - {"minimum", "maximum"}
        if unknown or not expected:
            raise ValueError("Numeric hardware matches may contain only minimum and maximum")
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        if "minimum" in expected and float(actual) < float(expected["minimum"]):
            return False
        if "maximum" in expected and float(actual) > float(expected["maximum"]):
            return False
        return True
    if isinstance(expected, list):
        if not expected:
            raise ValueError("Hardware match lists cannot be empty")
        return actual in expected
    return actual == expected


def _validate_profile(identifier: str, value: object, *, source: Path) -> dict[str, Any]:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError(f"{source}: profile IDs must be nonempty strings")
    if not isinstance(value, dict):
        raise ValueError(f"{source}: profile {identifier!r} must be a mapping")
    unknown = set(value) - {
        "match",
        "action",
        "coefficients",
        "override_existing_correction",
    }
    if unknown:
        raise ValueError(
            f"{source}: profile {identifier!r} has unknown fields: {', '.join(sorted(unknown))}"
        )
    match = value.get("match")
    action = value.get("action")
    if (
        not isinstance(match, dict)
        or not match
        or any(not isinstance(key, str) or not key for key in match)
    ):
        raise ValueError(f"{source}: profile {identifier!r} requires a nonempty match mapping")
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(
            f"{source}: profile {identifier!r} action must be unwarp or already_corrected"
        )
    coefficients = value.get("coefficients")
    if action == "unwarp" and (not isinstance(coefficients, str) or not coefficients.strip()):
        raise ValueError(f"{source}: unwarp profile {identifier!r} requires coefficients")
    if action == "already_corrected" and coefficients is not None:
        raise ValueError(
            f"{source}: already_corrected profile {identifier!r} cannot define coefficients"
        )
    override = value.get("override_existing_correction", False)
    if not isinstance(override, bool):
        raise ValueError(
            f"{source}: profile {identifier!r} override_existing_correction must be boolean"
        )
    return {
        "match": dict(match),
        "action": action,
        "coefficients": coefficients,
        "override_existing_correction": override,
    }


def validate_gradient_unwarping_catalog(root: Path | None = None) -> int:
    """Validate the selected hardware catalog and return its profile count."""
    source, profiles = _load_catalog(root)
    for identifier, profile in profiles.items():
        _validate_profile(identifier, profile, source=source)
    return len(profiles)


def gradient_unwarping_configured(root: Path | None = None) -> bool:
    """Report whether the site catalog contains a profile that applies correction."""
    source, profiles = _load_catalog(root)
    return any(
        _validate_profile(identifier, profile, source=source)["action"] == "unwarp"
        for identifier, profile in profiles.items()
    )


def gradient_unwarping_catalog_path(root: Path | None = None) -> Path:
    """Return the external catalog when present, otherwise the packaged empty catalog."""
    return _load_catalog(root)[0]


def gradient_unwarping_records(
    images: list[Path] | tuple[Path, ...],
    *,
    mode: str,
    markup=None,
) -> tuple[list[dict[str, object]], dict[Path, GradientUnwarpingResolution]]:
    """Resolve path-independent processing records for selected BIDS images."""
    if str(mode).strip().lower() == "off":
        return [], {}
    from nro.engine.bids import resolve_bids_metadata

    records: list[dict[str, object]] = []
    resolutions: dict[Path, GradientUnwarpingResolution] = {}
    sources = sorted({Path(image).expanduser().absolute() for image in images})
    for source in sources:
        try:
            metadata = resolve_bids_metadata(source, markup=markup).values
        except FileNotFoundError:
            metadata = {}
        resolution = resolve_gradient_unwarping(metadata, mode=mode)
        resolutions[source] = resolution
        records.append({"source": str(source), **resolution.scientific_record()})
    return records, resolutions


def resolve_gradient_unwarping(
    metadata: Mapping[str, object],
    *,
    mode: str,
    definitions: Path | None = None,
    coefficient_root: Path | None = None,
) -> GradientUnwarpingResolution:
    """Resolve one acquisition or reject ambiguous and unsupported metadata."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "off"}:
        raise ValueError(f"Gradient unwarping mode must be auto or off, found {mode!r}")
    if normalized_mode == "off":
        return GradientUnwarpingResolution(
            mode="off",
            applied=False,
            reason="disabled_by_configuration",
            profile=None,
            action=None,
            matched_metadata={},
            coefficients=None,
            coefficient_sha256=None,
            override_existing_correction=False,
        )

    source, raw_profiles = _load_catalog(definitions)
    matches: list[tuple[str, dict[str, Any], dict[str, object]]] = []
    for identifier, raw_profile in raw_profiles.items():
        profile = _validate_profile(identifier, raw_profile, source=source)
        matched = {key: metadata.get(key) for key in profile["match"]}
        if all(_matches(expected, matched[key]) for key, expected in profile["match"].items()):
            matches.append((identifier, profile, matched))
    if len(matches) > 1:
        raise ValueError(
            "Acquisition metadata matches multiple gradient-unwarping profiles: "
            + ", ".join(identifier for identifier, _, _ in matches)
        )
    reports_corrected = metadata.get("NonlinearGradientCorrection") is True
    if not matches:
        return GradientUnwarpingResolution(
            mode="auto",
            applied=False,
            reason=(
                "bids_metadata_reports_corrected"
                if reports_corrected
                else "no_matching_site_profile"
            ),
            profile=None,
            action="already_corrected" if reports_corrected else None,
            matched_metadata=({"NonlinearGradientCorrection": True} if reports_corrected else {}),
            coefficients=None,
            coefficient_sha256=None,
            override_existing_correction=False,
        )

    identifier, profile, matched = matches[0]
    action = profile["action"]
    override = bool(profile["override_existing_correction"])
    if action == "already_corrected" or reports_corrected and not override:
        return GradientUnwarpingResolution(
            mode="auto",
            applied=False,
            reason=(
                "site_profile_marks_already_corrected"
                if action == "already_corrected"
                else "bids_metadata_reports_corrected"
            ),
            profile=identifier,
            action=action,
            matched_metadata=matched,
            coefficients=None,
            coefficient_sha256=None,
            override_existing_correction=override,
        )

    root = (
        (
            Path(settings()[0]["gradient_coefficients"])
            if coefficient_root is None
            else Path(coefficient_root)
        )
        .expanduser()
        .resolve()
    )
    relative = Path(str(profile["coefficients"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{source}: coefficient paths must be relative to {root}")
    coefficients = (root / relative).resolve()
    if not coefficients.is_relative_to(root):
        raise ValueError(f"{source}: coefficient path escapes {root}")
    if not coefficients.is_file() or coefficients.stat().st_size == 0:
        raise FileNotFoundError(
            f"Gradient coefficients for profile {identifier!r} are missing: {coefficients}"
        )
    return GradientUnwarpingResolution(
        mode="auto",
        applied=True,
        reason="site_profile_requires_unwarping",
        profile=identifier,
        action=action,
        matched_metadata=matched,
        coefficients=coefficients,
        coefficient_sha256=_sha256(coefficients),
        override_existing_correction=override,
    )
