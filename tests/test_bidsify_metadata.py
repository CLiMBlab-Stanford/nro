from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

from nro.bidsify import metadata as enrichment


class Header(dict):
    """Minimal pydicom-like mapping for vendor metadata tests."""

    def add(self, tag: tuple[int, int], value: object) -> None:
        self[tag] = SimpleNamespace(value=value)


def ge_header(**protocol: str) -> Header:
    header = Header()
    text = "\n".join(f'{key} "{value}"' for key, value in protocol.items()).encode()
    header.add((0x0025, 0x0010), "GEMS_SERS_01")
    header.add((0x0025, 0x101B), b"\0\0\0\0" + gzip.compress(text))
    header.add((0x0019, 0x0010), "GEMS_ACQU_01")
    header.add((0x0019, 0x10D5), 0)
    header.add((0x0043, 0x0010), "GEMS_PARM_01")
    return header


def no_site(monkeypatch) -> None:
    monkeypatch.setattr(
        enrichment,
        "resolve_acquisition_metadata",
        lambda _metadata, **_kwargs: (None, {}, {}),
    )


def test_ge_fractional_nex_is_phase_partial_fourier(monkeypatch, tmp_path: Path) -> None:
    no_site(monkeypatch)
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE"},
        [(ge_header(NEX="0.75"), tmp_path / "unused.dcm")],
    )
    assert result.metadata["PartialFourier"] == 0.75
    assert result.metadata["PartialFourierDirection"] == "PHASE"
    assert result.conflicts == ()


def test_converter_value_wins_but_disagreement_blocks_publication(
    monkeypatch, tmp_path: Path
) -> None:
    no_site(monkeypatch)
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE", "PartialFourier": 0.8},
        [(ge_header(NEX="0.75"), tmp_path / "unused.dcm")],
    )
    assert result.metadata["PartialFourier"] == 0.8
    assert any(
        item["field"] == "PartialFourier" and item["reason"] == "converter_derivation_disagreement"
        for item in result.conflicts
    )


def test_ge_fractional_echo_prevents_inventing_a_fraction(monkeypatch, tmp_path: Path) -> None:
    no_site(monkeypatch)
    header = ge_header(NEX="0.75")
    header[(0x0019, 0x10D5)].value = 1
    result = enrichment.enrich_metadata({"Manufacturer": "GE"}, [(header, tmp_path / "unused.dcm")])
    assert "PartialFourier" not in result.metadata
    assert any("fractional echo" in warning for warning in result.warnings)
    assert result.underivable["PartialFourier"] == "ge_fractional_echo_fraction_unavailable"


def test_site_profile_fills_missing_ge_correction_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        enrichment,
        "resolve_acquisition_metadata",
        lambda _metadata, **_kwargs: (
            "ge_scanner",
            {"Manufacturer": "GE"},
            {
                "nonlinear_gradient_correction": True,
                "gradient_correction_mode": "2D",
            },
        ),
    )
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE"},
        [(ge_header(NEX="1.00"), tmp_path / "unused.dcm")],
    )
    assert result.metadata["NonlinearGradientCorrection"] is True
    assert result.metadata["GradientCorrectionMode"] == "2D"
    assert result.derivations["GradientCorrectionMode"]["source"].endswith("ge_scanner")


def test_known_correction_without_mode_records_private_reason(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        enrichment,
        "resolve_acquisition_metadata",
        lambda _metadata, **_kwargs: (
            "ge_scanner",
            {"Manufacturer": "GE"},
            {"nonlinear_gradient_correction": True},
        ),
    )
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE"},
        [(ge_header(NEX="1.00"), tmp_path / "unused.dcm")],
    )
    assert "GradientCorrectionMode" not in result.metadata
    assert result.underivable["GradientCorrectionMode"] == "missing_direct_and_site_evidence"


def test_direct_ge_mode_disagreement_with_site_is_a_conflict(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        enrichment,
        "resolve_acquisition_metadata",
        lambda _metadata, **_kwargs: (
            "ge_scanner",
            {"Manufacturer": "GE"},
            {
                "nonlinear_gradient_correction": True,
                "gradient_correction_mode": "2D",
            },
        ),
    )
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE"},
        [(ge_header(NEX="1.00", THREEDGW="1"), tmp_path / "unused.dcm")],
    )
    assert result.metadata["GradientCorrectionMode"] == "3D"
    assert any(item["reason"] == "dicom_site_disagreement" for item in result.conflicts)


def test_ge_gradient_coil_code_is_decoded(monkeypatch, tmp_path: Path) -> None:
    no_site(monkeypatch)
    header = ge_header(NEX="1.00")
    header.add((0x0043, 0x1082), ["SRMode=200", "GCoilType=16"])
    result = enrichment.enrich_metadata({"Manufacturer": "GE"}, [(header, tmp_path / "unused.dcm")])
    assert result.metadata["GradientCoilModel"] == "SIGNA UHP gradient"


def test_ge_private_fields_require_the_expected_creator(monkeypatch, tmp_path: Path) -> None:
    no_site(monkeypatch)
    header = ge_header(NEX="0.75")
    header[(0x0025, 0x0010)].value = "UNRELATED_CREATOR"
    result = enrichment.enrich_metadata({"Manufacturer": "GE"}, [(header, tmp_path / "unused.dcm")])
    assert "PartialFourier" not in result.metadata


def test_series_level_disagreement_blocks_publication(monkeypatch, tmp_path: Path) -> None:
    no_site(monkeypatch)
    result = enrichment.enrich_metadata(
        {"Manufacturer": "GE"},
        [
            (ge_header(NEX="0.75"), tmp_path / "first.dcm"),
            (ge_header(NEX="1.00"), tmp_path / "second.dcm"),
        ],
    )
    assert any(item["reason"] == "inconsistent_series_values" for item in result.conflicts)


def test_siemens_protocol_repairs_partial_fourier_and_echo_spacing(
    monkeypatch, tmp_path: Path
) -> None:
    no_site(monkeypatch)
    path = tmp_path / "image.dcm"
    path.write_bytes(
        b"prefix\n### ASCCONV BEGIN ###\n"
        b"sKSpace.ucPhasePartialFourier = 0x4\n"
        b"sFastImaging.lEchoSpacing = 500\n"
        b"### ASCCONV END ###\nsuffix"
    )
    result = enrichment.enrich_metadata({"Manufacturer": "Siemens"}, [(Header(), path)])
    assert result.metadata["PartialFourier"] == 0.75
    assert result.metadata["PartialFourierDirection"] == "PHASE"
    assert result.metadata["VendorReportedEchoSpacing"] == 0.0005
