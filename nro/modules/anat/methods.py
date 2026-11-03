"""Methods-report context for anatomical preprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nro.methods.context import ManifestRecord, ModuleMethods, RunFact, SoftwareUse, lookup

TITLE = "Anatomical preprocessing"
CONTEXT_VERSION = 1

FSAVERAGE_VERTICES = {
    "fsaverage": 163842,
    "fsaverage6": 40962,
    "fsaverage5": 10242,
    "fsaverage4": 2562,
}

CONTEXT_FIELDS = {
    "observed": "boolean",
    "participants": "integer",
    "selection_strategy": "string",
    "t1w_fallback": "string",
    "engine": "string",
    "fsaverage_template": "string",
    "fsaverage_vertices": "integer",
    "mni_space": "string",
    "gradient_unwarping.mode": "string",
    "gradient_unwarping.applied_images": "integer",
    "gradient_unwarping.total_images": "integer",
    "counts.multiple_t1w": "integer",
    "counts.t2w": "integer",
    "counts.synthetic_t1w": "integer",
    "counts.lesion": "integer",
    "counts.msmall": "integer",
    "counts.fastsurfer": "integer",
    "counts.freesurfer": "integer",
}

CONFIGURATION_FIELDS = {
    "selection_strategy": "selection_strategy",
    "t1w_fallback": "t1w_fallback",
    "surface_reconstruction_engine": "engine",
    "fsaverage_template": "fsaverage_template",
    "gradient_unwarping": "gradient_unwarping.mode",
    "mni_template": "mni_space",
}

METHODS_SILENT = {
    "container.bind": "Execution environment; images are listed in the software table.",
    "container.cleanenv": "Execution environment.",
    "container.home": "Execution environment.",
    "container.image": "Reported in the software table.",
    "container.inner_setup": "Execution environment.",
    "container.no_container": "Execution environment.",
    "freesurfer_subjects_dir": "Storage location.",
    "fs_subject": "Participant naming.",
    "markup": "Selects source markup; its effects are reported through observed counts.",
    "synthstrip_container": "Reported in the software table.",
}


def _engine(record: ManifestRecord, configured: str) -> str:
    backend = str(lookup(record.data, "surface_reconstruction.backend", configured))
    return "fastsurfer" if backend.casefold().startswith("fastsurfer") else "freesurfer"


def methods(
    configuration: Mapping[str, Any],
    manifests: Sequence[ManifestRecord],
    *,
    workflow: Mapping[str, Mapping[str, Any]],
) -> ModuleMethods:
    """Summarize the anatomical configuration and its fresh participant manifests."""
    del workflow
    configured_engine = str(configuration["surface_reconstruction_engine"])
    template = str(configuration["fsaverage_template"])
    applied = total = 0
    counts = dict.fromkeys(
        ("multiple_t1w", "t2w", "synthetic_t1w", "lesion", "msmall", "fastsurfer", "freesurfer"), 0
    )
    facts: list[RunFact] = []
    software: set[SoftwareUse] = set()
    for record in manifests:
        data = record.data
        entities = {name: record.entities.get(name, "") for name in ("project", "participant")}
        t1w = lookup(data, "inputs.t1w", []) or []
        t2w = lookup(data, "inputs.t2w", []) or []
        counts["multiple_t1w"] += len(t1w) > 1
        counts["t2w"] += bool(t2w)
        synthetic = isinstance(data.get("t1w_synthesis"), Mapping)
        lesion = lookup(data, "lesion.enabled", False) is True
        msmall = lookup(data, "msmall.enabled", False) is True and (
            lookup(data, "outputs.msmall") is not None
        )
        counts["synthetic_t1w"] += synthetic
        counts["lesion"] += lesion
        counts["msmall"] += msmall
        engine = _engine(record, configured_engine)
        counts[engine] += 1
        for image in (data.get("gradient_unwarping") or {}).values():
            total += 1
            applied += bool(isinstance(image, Mapping) and image.get("applied"))
        version = lookup(data, "surface_reconstruction.version")
        backend = lookup(data, "surface_reconstruction.backend")
        if backend and version:
            citation = (
                "fischl2012freesurfer" if engine == "freesurfer" else "henschel2022fastsurfervinn"
            )
            software.add(SoftwareUse(str(backend), str(version), citation))
        facts.extend(
            (
                RunFact(entities, "t1w_sources", str(len(t1w))),
                RunFact(entities, "t2w_sources", str(len(t2w))),
                RunFact(entities, "synthetic_t1w", str(synthetic).lower()),
                RunFact(entities, "surface_engine", engine),
                RunFact(entities, "lesion_aware", str(lesion).lower()),
                RunFact(entities, "msmall", str(msmall).lower()),
            )
        )
    context = {
        "observed": bool(manifests),
        "participants": len(manifests),
        "selection_strategy": str(configuration["selection_strategy"]),
        "t1w_fallback": str(configuration["t1w_fallback"]),
        "engine": configured_engine,
        "fsaverage_template": template,
        "fsaverage_vertices": FSAVERAGE_VERTICES.get(template, 0),
        "mni_space": "MNI152NLin2009cAsym",
        "gradient_unwarping": {
            "mode": str(configuration["gradient_unwarping"]),
            "applied_images": applied,
            "total_images": total,
        },
        "counts": counts,
    }
    images = (
        "qunex",
        "synthstrip",
        "fastsurfer" if configured_engine == "fastsurfer" else "freesurfer",
    )
    return ModuleMethods(
        context=context,
        run_facts=tuple(facts),
        software=tuple(sorted(software, key=lambda use: (use.software, use.version))),
        images=images,
    )
