from __future__ import annotations

from pathlib import Path

import pytest

from nro.configuration import site
from nro.configuration.hardware import (
    gradient_unwarping_configured,
    resolve_acquisition_metadata,
    resolve_gradient_unwarping,
    validate_gradient_unwarping_catalog,
)


def write_catalog(root: Path, body: str) -> None:
    path = root / "hardware/gradient_unwarping.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_protected_site_fingerprint_covers_hardware_assertions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(site, "protected_site", lambda _definitions: ({"bids": "/bids"}, {}))
    write_catalog(tmp_path, "version: 1\nprofiles: {}\n")
    first = site.protected_site_fingerprint(tmp_path)
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  ge:
    match: {Manufacturer: GE}
    action: already_corrected
    acquisition_metadata:
      nonlinear_gradient_correction: true
      gradient_correction_mode: 2D
""",
    )
    assert site.protected_site_fingerprint(tmp_path) != first


def test_auto_is_off_without_a_matching_site_profile(tmp_path: Path) -> None:
    write_catalog(tmp_path, "version: 1\nprofiles: {}\n")
    assert not gradient_unwarping_configured(tmp_path)
    resolution = resolve_gradient_unwarping(
        {"MagneticFieldStrength": 7},
        mode="auto",
        definitions=tmp_path,
        coefficient_root=tmp_path / "coefficients",
    )
    assert not resolution.applied
    assert resolution.reason == "no_matching_site_profile"


def test_matching_profile_uses_coefficient_content_identity(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  impulse_7t:
    match:
      Manufacturer: Siemens
      MagneticFieldStrength:
        minimum: 6.5
    action: unwarp
    coefficients: impulse.grad
""",
    )
    coefficients = tmp_path / "coefficients/impulse.grad"
    coefficients.parent.mkdir()
    coefficients.write_bytes(b"coefficient data")
    assert gradient_unwarping_configured(tmp_path)
    resolution = resolve_gradient_unwarping(
        {"Manufacturer": "Siemens", "MagneticFieldStrength": 7},
        mode="auto",
        definitions=tmp_path,
        coefficient_root=coefficients.parent,
    )
    assert resolution.applied
    assert resolution.profile == "impulse_7t"
    assert resolution.scientific_record()["coefficient_sha256"]
    assert str(coefficients) not in str(resolution.scientific_record())


def test_prior_correction_prevents_repeat_unwarping(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  impulse_7t:
    match: {MagneticFieldStrength: 7}
    action: unwarp
    coefficients: impulse.grad
""",
    )
    resolution = resolve_gradient_unwarping(
        {"MagneticFieldStrength": 7, "NonlinearGradientCorrection": True},
        mode="auto",
        definitions=tmp_path,
        coefficient_root=tmp_path / "missing",
    )
    assert not resolution.applied
    assert resolution.reason == "bids_metadata_reports_corrected"


def test_matching_unwarp_profile_requires_its_coefficients(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  impulse_7t:
    match: {MagneticFieldStrength: 7}
    action: unwarp
    coefficients: impulse.grad
""",
    )
    with pytest.raises(FileNotFoundError, match="impulse.grad"):
        resolve_gradient_unwarping(
            {"MagneticFieldStrength": 7},
            mode="auto",
            definitions=tmp_path,
            coefficient_root=tmp_path / "missing",
        )


def test_catalog_rejects_overlapping_profiles_at_resolution(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  first:
    match: {MagneticFieldStrength: 7}
    action: already_corrected
  second:
    match: {Manufacturer: Siemens}
    action: already_corrected
""",
    )
    assert validate_gradient_unwarping_catalog(tmp_path) == 2
    with pytest.raises(ValueError, match="multiple"):
        resolve_gradient_unwarping(
            {"MagneticFieldStrength": 7, "Manufacturer": "Siemens"},
            mode="auto",
            definitions=tmp_path,
        )


def test_hardware_profile_can_assert_missing_acquisition_metadata(tmp_path: Path) -> None:
    write_catalog(
        tmp_path,
        """version: 1
profiles:
  ge_scanner:
    match:
      Manufacturer: GE
      ManufacturersModelName: SIGNA UHP
    action: already_corrected
    acquisition_metadata:
      nonlinear_gradient_correction: true
      gradient_correction_mode: 2D
      gradient_coil_model: SIGNA UHP gradient
""",
    )
    profile, matched, assertions = resolve_acquisition_metadata(
        {"Manufacturer": "GE", "ManufacturersModelName": "SIGNA UHP"},
        definitions=tmp_path,
    )
    assert profile == "ge_scanner"
    assert matched == {"Manufacturer": "GE", "ManufacturersModelName": "SIGNA UHP"}
    assert assertions["gradient_correction_mode"] == "2D"


@pytest.mark.parametrize(
    "metadata",
    [
        "gradient_correction_mode: unknown",
        "nonlinear_gradient_correction: false\n      gradient_correction_mode: 2D",
        "nonlinear_gradient_correction: true\n      gradient_correction_mode: none",
    ],
)
def test_hardware_profile_rejects_inconsistent_correction_metadata(
    tmp_path: Path, metadata: str
) -> None:
    write_catalog(
        tmp_path,
        f"""version: 1
profiles:
  invalid:
    match: {{Manufacturer: GE}}
    action: already_corrected
    acquisition_metadata:
      {metadata}
""",
    )
    with pytest.raises(ValueError):
        validate_gradient_unwarping_catalog(tmp_path)
