"""Enrich converted MR metadata from bounded vendor protocol fields."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nro.configuration.hardware import (
    gradient_unwarping_catalog_path,
    resolve_acquisition_metadata,
)
from nro.configuration.site import settings

from .errors import BidsificationError

GE_PDB = (0x0025, 0x101B)
GE_SYSTEM_CONFIG = (0x0043, 0x1082)
GE_ASSET_FACTORS = (0x0043, 0x1083)
GE_FRACTIONAL_ECHO = (0x0019, 0x10D5)
GE_PRIVATE_CREATORS = {
    0x0019: "GEMS_ACQU_01",
    0x0025: "GEMS_SERS_01",
    0x0043: "GEMS_PARM_01",
}
MAX_PROTOCOL_BYTES = 8 * 1024 * 1024
SIEMENS_PARTIAL_FOURIER = {1: 0.5, 2: 0.625, 4: 0.75, 8: 0.875, 16: 1.0}
GE_GRADIENT_COILS = {"13": "SIGNA Premier gradient", "16": "SIGNA UHP gradient"}

PUBLIC_FIELDS = frozenset(
    {
        "NonlinearGradientCorrection",
        "PartialFourier",
        "PartialFourierDirection",
        "ReceiveCoilName",
        "ReceiveCoilActiveElements",
        "ParallelAcquisitionTechnique",
        "ParallelReductionFactorInPlane",
        "VendorReportedEchoSpacing",
        "BandwidthPerPixelPhaseEncode",
        "DwellTime",
        "EchoTrainLength",
        "PhaseEncodingSteps",
        "TxRefAmp",
        "ShimSetting",
        "GradientCorrectionMode",
        "GradientCoilModel",
        "CoilCombinationMethod",
        "BaseResolution",
        "PhaseResolution",
        "AcquisitionMatrixPE",
        "PulseSequenceDetails",
        "RefLinesPE",
        "ConsistencyInfo",
        "PercentPhaseFOV",
    }
)

_POLICY = {
    "version": 1,
    "ge_partial_fourier": "pdb_nex_when_no_fractional_echo",
    "ge_partial_fourier_direction": "phase",
    "ge_correction_mode": "threedgw_then_site_profile_then_omit",
    "merge": "dcm2niix_then_dicom_then_site_conflict_on_disagreement",
    "series": "all_observed_values_must_agree",
}


@dataclass(frozen=True)
class Derived:
    """A candidate metadata value and its bounded provenance description."""

    value: object
    source: str
    transform: str | None = None


@dataclass(frozen=True)
class Enrichment:
    """Sanitized metadata plus private derivation and review records."""

    metadata: dict[str, object]
    derivations: dict[str, dict[str, str]]
    underivable: dict[str, str]
    warnings: tuple[str, ...]
    conflicts: tuple[dict[str, object], ...]
    hardware_profile: str | None


def policy_identity(metadata: Mapping[str, object]) -> str:
    """Identify the rules and protected hardware assertion relevant to a series."""
    shared_definitions = Path(settings()[0]["definitions"])
    catalog = gradient_unwarping_catalog_path(shared_definitions)
    profile, matched, assertions = resolve_acquisition_metadata(
        metadata, definitions=shared_definitions
    )
    payload = {
        "policy": _POLICY,
        "hardware": (
            {
                "profile": profile,
                "matched": matched,
                "assertions": assertions,
            }
            if profile is not None
            else {"catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest()}
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _element_value(dataset: Any, tag: tuple[int, int]) -> object | None:
    element = dataset.get(tag)
    return None if element is None else getattr(element, "value", element)


def _private_value(dataset: Any, tag: tuple[int, int]) -> object | None:
    """Read one GE private element only under its documented private creator."""
    group, element = tag
    creator = _element_value(dataset, (group, (element >> 8) & 0xFF))
    expected = GE_PRIVATE_CREATORS.get(group)
    if expected is None or str(creator or "").strip() != expected:
        return None
    return _element_value(dataset, tag)


def _bounded_gzip(payload: bytes) -> str:
    for offset in (4, 0):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload[offset:])) as stream:
                expanded = stream.read(MAX_PROTOCOL_BYTES + 1)
            if len(expanded) > MAX_PROTOCOL_BYTES:
                raise BidsificationError("GE protocol data exceed the accepted size")
            return expanded.decode("latin-1")
        except (OSError, EOFError, UnicodeError):
            continue
    raise BidsificationError("GE protocol data could not be decoded")


def parse_ge_protocol(payload: object) -> dict[str, str]:
    """Decode the bounded GE Protocol Data Block into unique key-value pairs."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return {}
    text = _bounded_gzip(bytes(payload))
    result: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r'(\w+)\s+"(.*)"\s*', line)
        if match is None:
            continue
        key, value = match.groups()
        if key in result and result[key] != value:
            raise BidsificationError("GE protocol data contain conflicting duplicate fields")
        result[key] = value
    return result


def parse_ascconv(raw: bytes) -> dict[str, str]:
    """Extract unique Siemens ASCCONV assignments from one DICOM file."""
    begin = raw.find(b"### ASCCONV BEGIN")
    end = raw.find(b"### ASCCONV END", begin + 1)
    if begin < 0 or end < 0:
        return {}
    if end - begin > MAX_PROTOCOL_BYTES:
        raise BidsificationError("Siemens protocol data exceed the accepted size")
    text = raw[begin:end].decode("latin-1", errors="replace")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key or key.startswith("#"):
            continue
        if value.startswith('""') and value.endswith('""') and len(value) >= 4:
            value = value[2:-2]
        elif value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if key in result and result[key] != value:
            raise BidsificationError("Siemens protocol data contain conflicting duplicate fields")
        result[key] = value
    return result


def _number(value: object, cast: type[int] | type[float]) -> int | float | None:
    try:
        parsed = cast(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(float(parsed)) else None


def _integer(value: object) -> int | None:
    try:
        return int(str(value), 0)
    except (TypeError, ValueError):
        return None


def _put(
    output: dict[str, Derived],
    field: str,
    value: object | None,
    source: str,
    transform: str | None = None,
) -> None:
    if value is not None:
        _validate_candidate(field, value)
        output[field] = Derived(value, source, transform)


def _validate_candidate(field: str, value: object) -> None:
    """Reject malformed derived values before they enter a public sidecar."""
    if field == "NonlinearGradientCorrection":
        valid = isinstance(value, bool)
    elif field == "GradientCorrectionMode":
        valid = value in {"2D", "3D", "none"}
    elif field == "PartialFourier":
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0 < float(value) <= 1
        )
    elif field == "PartialFourierDirection":
        valid = value in {"PHASE", "FREQUENCY", "SLICE_SELECT", "COMBINATION"}
    elif field in {"VendorReportedEchoSpacing", "DwellTime"}:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
        )
    elif field in {"ParallelReductionFactorInPlane"}:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 1
        )
    elif field in {"BaseResolution", "AcquisitionMatrixPE", "RefLinesPE"}:
        valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
    elif field == "PhaseResolution":
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0 < float(value) <= 1
        )
    elif field == "ShimSetting":
        valid = (
            isinstance(value, list)
            and bool(value)
            and all(
                not isinstance(item, bool)
                and isinstance(item, (int, float))
                and math.isfinite(float(item))
                for item in value
            )
        )
    elif isinstance(value, str):
        valid = (
            bool(value.strip())
            and len(value) <= 1024
            and not any(ord(character) < 32 and character not in "\t" for character in value)
        )
    else:
        valid = True
    if not valid:
        raise BidsificationError(f"Derived {field} metadata are invalid")


def _siemens_candidates(
    metadata: Mapping[str, object], protocol: Mapping[str, str]
) -> dict[str, Derived]:
    output: dict[str, Derived] = {}
    tokens = metadata.get("ImageTypeText", metadata.get("ImageType", []))
    if isinstance(tokens, str):
        tokens = re.split(r"[\\, ]+", tokens)
    upper = {str(token).upper() for token in tokens if token}
    mode = (
        "3D"
        if "DIS3D" in upper
        else "2D"
        if "DIS2D" in upper
        else "none"
        if "ND" in upper
        else None
    )
    if mode is not None:
        _put(output, "GradientCorrectionMode", mode, "Siemens ImageTypeText")
        _put(output, "NonlinearGradientCorrection", mode != "none", "Siemens ImageTypeText")

    code = _integer(protocol.get("sKSpace.ucPhasePartialFourier"))
    if code in SIEMENS_PARTIAL_FOURIER:
        _put(
            output,
            "PartialFourier",
            SIEMENS_PARTIAL_FOURIER[int(code)],
            "Siemens ASCCONV sKSpace.ucPhasePartialFourier",
        )
        if SIEMENS_PARTIAL_FOURIER[int(code)] < 1:
            _put(output, "PartialFourierDirection", "PHASE", "Siemens ASCCONV")

    offsets = [
        _number(protocol.get(f"sGRADSPEC.asGPAData[0].lOffset{axis}"), int) for axis in "XYZ"
    ]
    if any(value is None for value in offsets):
        offsets = [_number(protocol.get(f"sGRADSPEC.lOffset{axis}"), int) for axis in "XYZ"]
    currents = [
        _number(protocol.get(f"sGRADSPEC.alShimCurrent[{index}]"), int) for index in range(5)
    ]
    if all(value is not None for value in (*offsets, *currents)):
        _put(output, "ShimSetting", [*offsets, *currents], "Siemens ASCCONV shim fields")

    scalar_fields = {
        "TxRefAmp": ("sTXSPEC.asNucleusInfo[0].flReferenceAmplitude", float),
        "BaseResolution": ("sKSpace.lBaseResolution", int),
        "PhaseResolution": ("sKSpace.dPhaseResolution", float),
        "AcquisitionMatrixPE": ("sKSpace.lPhaseEncodingLines", int),
        "RefLinesPE": ("sPat.lRefLinesPE", int),
        "ParallelReductionFactorInPlane": ("sPat.lAccelFactPE", float),
    }
    for field, (key, cast) in scalar_fields.items():
        value = _number(protocol.get(key), cast)
        if field == "ParallelReductionFactorInPlane" and value is not None and value <= 1:
            continue
        _put(output, field, value, f"Siemens ASCCONV {key}")
    sequence = protocol.get("tSequenceFileName")
    _put(output, "PulseSequenceDetails", sequence, "Siemens ASCCONV tSequenceFileName")
    coil = protocol.get("sCoilSelectMeas.aRxCoilSelectData[0].asList[0].sCoilElementID.tCoilID")
    _put(output, "ReceiveCoilName", coil, "Siemens ASCCONV receive coil")
    _put(
        output,
        "ReceiveCoilActiveElements",
        protocol.get("sCoilSelectMeas.sCoilStringForConversion"),
        "Siemens ASCCONV active coil elements",
    )
    spacing = _number(protocol.get("sFastImaging.lEchoSpacing"), int)
    _put(
        output,
        "VendorReportedEchoSpacing",
        None if spacing is None else round(spacing / 1_000_000, 9),
        "Siemens ASCCONV sFastImaging.lEchoSpacing",
        "microseconds_to_seconds",
    )
    combine = _number(protocol.get("ucCoilCombineMode"), int)
    _put(
        output,
        "CoilCombinationMethod",
        {1: "Sum of Squares", 2: "Adaptive Combine"}.get(combine),
        "Siemens ASCCONV ucCoilCombineMode",
    )
    return output


def _ge_candidates(dataset: Any) -> tuple[dict[str, Derived], list[str]]:
    output: dict[str, Derived] = {}
    warnings: list[str] = []
    protocol = parse_ge_protocol(_private_value(dataset, GE_PDB))
    three_d = protocol.get("THREEDGW")
    if three_d is not None:
        numeric = _number(three_d, float)
        if numeric in {0.0, 1.0}:
            _put(
                output,
                "GradientCorrectionMode",
                "3D" if numeric == 1 else "2D",
                "GE Protocol Data Block THREEDGW",
            )
            _put(
                output,
                "NonlinearGradientCorrection",
                True,
                "GE Protocol Data Block THREEDGW",
            )
        else:
            warnings.append("GradientCorrectionMode: GE THREEDGW has an unsupported value")

    system_config = _private_value(dataset, GE_SYSTEM_CONFIG)
    values = (
        [system_config]
        if isinstance(system_config, (str, int, float))
        else list(system_config or [])
    )
    coil_code = next(
        (
            str(value).split("=", 1)[1].strip()
            for value in values
            if str(value).startswith("GCoilType=")
        ),
        None,
    )
    if coil_code in GE_GRADIENT_COILS:
        _put(
            output,
            "GradientCoilModel",
            GE_GRADIENT_COILS[coil_code],
            "GE private element (0043,1082) GCoilType",
        )
    elif coil_code is not None:
        warnings.append("GradientCoilModel: GE GCoilType is not in the approved mapping")

    fractional_echo_raw = _private_value(dataset, GE_FRACTIONAL_ECHO)
    fractional_echo = _number(fractional_echo_raw, int)
    has_fractional_echo = fractional_echo is not None and fractional_echo != 0
    nex = _number(protocol.get("NEX"), float)
    if nex is not None and not has_fractional_echo:
        fraction = round(float(nex), 4) if nex < 1 else 1
        _put(output, "PartialFourier", fraction, "GE Protocol Data Block NEX")
        if fraction < 1:
            _put(output, "PartialFourierDirection", "PHASE", "GE Protocol Data Block NEX")
    elif has_fractional_echo:
        warnings.append(
            "PartialFourier: GE fractional echo is present but its acquired fraction is unavailable"
        )

    _put(output, "ReceiveCoilName", protocol.get("COIL"), "GE Protocol Data Block COIL")
    _put(
        output,
        "ReceiveCoilActiveElements",
        protocol.get("COILPLUGS"),
        "GE Protocol Data Block COILPLUGS",
    )
    acceleration = None
    acceleration_source = None
    raw_factors = _private_value(dataset, GE_ASSET_FACTORS)
    if raw_factors is not None:
        factors = [raw_factors] if isinstance(raw_factors, (str, int, float)) else list(raw_factors)
        first = _number(factors[0], float) if factors else None
        if first is not None and first > 0:
            acceleration = 1 / float(first)
            acceleration_source = "GE private element (0043,1083)"
    if acceleration is None:
        acceleration = _number(protocol.get("PHASEACCEL"), float)
        acceleration_source = "GE Protocol Data Block PHASEACCEL"
    if acceleration is not None and acceleration > 1 + 1e-6:
        acceleration = round(float(acceleration), 4)
        _put(
            output,
            "ParallelReductionFactorInPlane",
            int(acceleration) if acceleration.is_integer() else acceleration,
            str(acceleration_source),
        )
        options = protocol.get("IOPT", "").upper()
        technique = "ARC" if "ARC" in options else "ASSET" if "ASSET" in options else None
        _put(output, "ParallelAcquisitionTechnique", technique, "GE Protocol Data Block IOPT")
        if technique is None:
            warnings.append(
                "ParallelAcquisitionTechnique: acceleration is present but IOPT names no supported technique"
            )
    return output, warnings


def _equivalent(first: object, second: object) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-9)
    return first == second


def _series_candidates(
    candidates: Iterable[tuple[dict[str, Derived], Sequence[str]]],
) -> tuple[dict[str, Derived], list[str], list[dict[str, object]]]:
    merged: dict[str, Derived] = {}
    warnings: list[str] = []
    conflicts: list[dict[str, object]] = []
    for fields, item_warnings in candidates:
        warnings.extend(item_warnings)
        for field, candidate in fields.items():
            prior = merged.get(field)
            if prior is None:
                merged[field] = candidate
            elif not _equivalent(prior.value, candidate.value):
                conflicts.append(
                    {
                        "field": field,
                        "reason": "inconsistent_series_values",
                        "sources": sorted({prior.source, candidate.source}),
                    }
                )
    return merged, list(dict.fromkeys(warnings)), conflicts


def _site_candidates(metadata: Mapping[str, object]) -> tuple[str | None, dict[str, Derived]]:
    shared_definitions = Path(settings()[0]["definitions"])
    profile, _matched, assertions = resolve_acquisition_metadata(
        metadata, definitions=shared_definitions
    )
    names = {
        "nonlinear_gradient_correction": "NonlinearGradientCorrection",
        "gradient_correction_mode": "GradientCorrectionMode",
        "gradient_coil_model": "GradientCoilModel",
    }
    return profile, {
        names[key]: Derived(value, f"site hardware profile {profile}")
        for key, value in assertions.items()
    }


def _merge(
    baseline: Mapping[str, object],
    direct: Mapping[str, Derived],
    site: Mapping[str, Derived],
    conflicts: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    metadata = dict(baseline)
    derivations: dict[str, dict[str, str]] = {}
    candidates: dict[str, Derived] = dict(direct)
    for field, candidate in site.items():
        prior = candidates.get(field)
        if prior is not None and not _equivalent(prior.value, candidate.value):
            conflicts.append(
                {
                    "field": field,
                    "reason": "dicom_site_disagreement",
                    "sources": [prior.source, candidate.source],
                }
            )
            continue
        candidates.setdefault(field, candidate)
    for field, candidate in candidates.items():
        _validate_candidate(field, candidate.value)
        if field in metadata:
            if not _equivalent(metadata[field], candidate.value):
                conflicts.append(
                    {
                        "field": field,
                        "reason": "converter_derivation_disagreement",
                        "sources": ["dcm2niix", candidate.source],
                    }
                )
            continue
        metadata[field] = candidate.value
        record = {"source": candidate.source}
        if candidate.transform is not None:
            record["transform"] = candidate.transform
        derivations[field] = record
    return metadata, derivations


def enrich_metadata(
    metadata: Mapping[str, object],
    dicoms: Sequence[tuple[Any, Path]],
) -> Enrichment:
    """Fill approved metadata from a validated, internally consistent MR series."""
    manufacturer = str(metadata.get("Manufacturer", "")).upper()
    observations: list[tuple[dict[str, Derived], Sequence[str]]] = []
    if "SIEMENS" in manufacturer:
        for _header, path in dicoms:
            observations.append(
                (_siemens_candidates(metadata, parse_ascconv(path.read_bytes())), ())
            )
    elif manufacturer in {"GE", "GE MEDICAL SYSTEMS", "GE HEALTHCARE"}:
        observations.extend(_ge_candidates(header) for header, _path in dicoms)
    direct, warnings, conflicts = _series_candidates(observations)
    underivable: dict[str, str] = {}
    if any("fractional echo" in warning for warning in warnings):
        underivable["PartialFourier"] = "ge_fractional_echo_fraction_unavailable"
    if any("IOPT" in warning for warning in warnings):
        underivable["ParallelAcquisitionTechnique"] = "ge_iopt_technique_unavailable"
    if any("GCoilType" in warning for warning in warnings):
        underivable["GradientCoilModel"] = "ge_gradient_coil_code_unmapped"
    profile, site = _site_candidates(metadata)
    enriched, derivations = _merge(metadata, direct, site, conflicts)

    mode = enriched.get("GradientCorrectionMode")
    corrected = enriched.get("NonlinearGradientCorrection")
    if mode in {"2D", "3D"} and corrected is not True:
        conflicts.append(
            {"field": "GradientCorrectionMode", "reason": "mode_requires_correction_true"}
        )
    if mode == "none" and corrected is not False:
        conflicts.append(
            {"field": "GradientCorrectionMode", "reason": "none_requires_correction_false"}
        )
    if corrected is True and mode is None:
        warnings.append("GradientCorrectionMode: correction is known but its mode is underivable")
        underivable["GradientCorrectionMode"] = "missing_direct_and_site_evidence"

    return Enrichment(
        metadata=enriched,
        derivations=derivations,
        underivable=underivable,
        warnings=tuple(dict.fromkeys(warnings)),
        conflicts=tuple(conflicts),
        hardware_profile=profile,
    )
