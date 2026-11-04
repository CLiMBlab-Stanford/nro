"""Methods-report rendering, citations, coverage, and golden outputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from nro.definitions.schema import scientific_values
from nro.definitions.store import ConfigStore
from nro.methods.bibliography import (
    BibliographyError,
    citation_labels,
    load_bibliography,
    parse_bibliography,
    plain_text,
    reference_runs,
)
from nro.methods.context import flatten_keys, participants
from nro.methods.document import Run
from nro.methods.report import (
    SUPPORTED_MODULES,
    ManifestSelection,
    MethodsReportError,
    build_report,
    configuration_coverage,
    provider,
    registry_manifests,
    workflow_lineages,
)
from nro.methods.software import ImageRecord
from nro.methods.templates import (
    TEMPLATE_ROOT,
    MethodsTemplateError,
    module_templates,
    parse_template,
    render_template,
)
from nro.methods.writers import _rtf_escape

GOLDEN = Path(__file__).parent / "fixtures" / "methods"

BIB = """
@article{smith2020a, author = {Smith, Jane and Doe, John}, title = {Beta}, journal = {J}, year = {2020}}
@article{smith2020b, author = {Smith, Jane and Doe, John}, title = {Alpha}, journal = {J}, year = {2020}}
@article{solo2019, author = {M{\\"u}ller, Ann}, title = {Solo}, journal = {J}, year = {2019},
  volume = {3}, number = {2}, pages = {1--9}, doi = {10.1/x}}
@article{many2018, author = {One, A. and Two, B. and Three, C.}, title = {Many}, journal = {J}, year = {2018}}
@article{partial2023, author = {Yu, Tian and others}, title = {Partial}, journal = {J}, year = {2023}}
"""


def _report(output_format: str = "rtf", **kwargs):
    defaults = dict(
        output_format=output_format,
        nro_version="0.0.0",
        images=lambda keys: {key: ImageRecord(key, "", None) for key in keys},
        pins={},
        package_versions={"numpy": "1.0"},
    )
    defaults.update(kwargs)
    return build_report(ConfigStore().resolve("main"), **defaults)


def _template(tmp_path: Path, body: str, *, requires: str, when: str = "true", name="01_test.md"):
    directory = tmp_path / "func"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(
        f'---\nsection: Test\ncontext_version: 1\nrequires: {requires}\nwhen: "{when}"\n---\n{body}'
    )
    return path


# Bibliography


def test_author_year_labels_disambiguate_and_shorten_author_lists() -> None:
    entries = parse_bibliography(BIB)
    labels = citation_labels(entries, entries)

    assert labels.parenthetical["smith2020b"] == "Smith & Doe, 2020a"
    assert labels.parenthetical["smith2020a"] == "Smith & Doe, 2020b"
    assert labels.narrative["smith2020b"] == "Smith and Doe (2020a)"
    assert labels.parenthetical["solo2019"] == "Müller, 2019"
    assert labels.parenthetical["many2018"] == "One et al., 2018"
    assert labels.parenthetical["partial2023"] == "Yu et al., 2023"
    with pytest.raises(BibliographyError, match="missing"):
        citation_labels(entries, ["missing"])


def test_reference_list_item_is_apa_like() -> None:
    entries = parse_bibliography(BIB)
    text = "".join(run.value for run in reference_runs(entries["solo2019"], "2019"))
    partial = "".join(run.value for run in reference_runs(entries["partial2023"], "2023"))

    assert text == "Müller, A. (2019). Solo. J, 3(2), 1–9. https://doi.org/10.1/x"
    assert partial.startswith("Yu, T., et al. (2023).")


def test_bibliography_rejects_unsupported_markup() -> None:
    with pytest.raises(BibliographyError, match="Unsupported LaTeX"):
        plain_text(r"\textbf{x}")
    with pytest.raises(BibliographyError, match="Duplicate"):
        parse_bibliography(BIB + BIB)


def test_packaged_bibliography_parses_and_covers_every_template_citation() -> None:
    entries = load_bibliography()
    for entry in entries.values():
        entry.authors
        entry.field("title")
    cited = set()
    for path in TEMPLATE_ROOT.glob("*/*.md"):
        cited.update(re.findall(r"@([A-Za-z][\w:-]*)", path.read_text().split("---", 2)[2]))
    assert cited and cited <= set(entries)


# Templates


def test_requires_must_match_template_variables(tmp_path) -> None:
    path = _template(tmp_path, "{{ runs }} and {{ other }}", requires="[runs]")
    with pytest.raises(MethodsTemplateError, match="requires"):
        parse_template("func", path)


def test_rendering_parses_citations_emphasis_and_paragraphs(tmp_path) -> None:
    body = (
        "{% if observed %}\nWe used *tool* [@a; @b] on {{ value | pct }}.\n{% endif %}\n"
        "Still the first paragraph, after @c.\n\nSecond paragraph, p < .05.\n"
    )
    path = _template(tmp_path, body, requires="[observed, value]")
    section = render_template(parse_template("func", path), {"observed": True, "value": 0.5})

    assert section is not None
    first, second = section.paragraphs
    assert first == (
        Run.text("We used "),
        Run.emphasis("tool"),
        Run.text(" "),
        Run.citation(("a", "b")),
        Run.text(" on 50%. Still the first paragraph, after "),
        Run.citation(("c",), narrative=True),
        Run.text("."),
    )
    assert second == (Run.text("Second paragraph, p < .05."),)


def test_false_condition_omits_section_and_markdown_is_rejected(tmp_path) -> None:
    hidden = _template(tmp_path, "{{ runs }}", requires="[runs]", when="runs > 3")
    assert render_template(parse_template("func", hidden), {"runs": 1}) is None

    heading = _template(tmp_path, "# Heading {{ runs }}", requires="[runs]", name="02_bad.md")
    with pytest.raises(MethodsTemplateError, match="headings"):
        render_template(parse_template("func", heading), {"runs": 1})


def test_override_replaces_template_and_must_match_context_version(tmp_path) -> None:
    packaged = module_templates("func", context_version=1)
    replaced = packaged[0].name
    _template(tmp_path, "Override {{ runs }}.", requires="[runs]", name=replaced)

    templates = module_templates("func", context_version=1, overrides=[tmp_path])

    assert templates[0].path == tmp_path / "func" / replaced
    assert len(templates) == len(packaged)
    with pytest.raises(MethodsTemplateError, match="context version"):
        module_templates("func", context_version=2, overrides=[tmp_path])


def test_sandbox_blocks_attribute_escape(tmp_path) -> None:
    path = _template(tmp_path, "{{ runs.__class__.__mro__ }}", requires="[runs]")
    with pytest.raises(Exception):
        render_template(parse_template("func", path), {"runs": 1})


# Writers


def test_rtf_escapes_control_and_non_ascii_characters() -> None:
    assert _rtf_escape("a{b}\\ é 𝛼") == "a\\{b\\}\\\\ \\u233? \\u-10187?\\u-8452?"


def test_latex_uses_natbib_and_escapes_text() -> None:
    report = _report("latex")
    text = report.files["methods.tex"]

    assert "\\citep{" in text and "\\citet{afyouni2018dvars}" in text
    assert "5\\%" in text
    assert "\\begin{longtable}" in text
    assert "\\section*{Supplementary tables}" in text


# Coverage and contexts


# Starter workflows whose lineages are all supported by methods reporting.
REPORTED_WORKFLOWS = ("main", "clustering")


def _reported_variants(module: str) -> list[dict]:
    """Return each reported starter workflow's values for one module."""
    return [
        dict(ConfigStore().resolve(workflow).configuration(module).values)
        for workflow in REPORTED_WORKFLOWS
    ]


@pytest.mark.parametrize("module", SUPPORTED_MODULES)
def test_every_substantive_parameter_is_reported_or_excused(module) -> None:
    methods = provider(module)
    substantive: set[str] = set()
    for values in _reported_variants(module):
        assert configuration_coverage(module, values) == [], values
        substantive |= set(flatten_keys(scientific_values(module, dict(values))))

    assert set(methods.CONFIGURATION_FIELDS).isdisjoint(methods.METHODS_SILENT)
    assert set(methods.CONFIGURATION_FIELDS) | set(methods.METHODS_SILENT) <= substantive
    assert set(methods.CONFIGURATION_FIELDS.values()) <= set(methods.CONTEXT_FIELDS)


@pytest.mark.parametrize("module", SUPPORTED_MODULES)
def test_packaged_templates_declare_current_context_version(module) -> None:
    assert module_templates(module, context_version=provider(module).CONTEXT_VERSION)


def test_lineages_follow_dependencies_and_reject_unsupported_modules() -> None:
    workflow = ConfigStore().resolve("main")

    assert [lineage.module for lineage in workflow_lineages(workflow, ["func"])] == ["anat", "func"]
    assert [lineage.module for lineage in workflow_lineages(workflow, ["networks"])] == [
        "anat",
        "func",
        "clean",
        "dynconn",
        "networks",
    ]
    oslom = ConfigStore().resolve("oslom")
    with pytest.raises(MethodsReportError, match="not yet available"):
        workflow_lineages(oslom, ["networks"])
    assert "networks" not in {lineage.module for lineage in workflow_lineages(oslom)}


def _func_manifest(sdc: str, *, fallback: str | None = None, marss: bool = False) -> dict:
    return {
        "registration": {
            "sdc_method": sdc,
            "sdc_fallback_reason": fallback,
            "reference_selection": {"Selected": sdc == "topup_sbref_pe"},
            "gradient_unwarping": {"epi": {"applied": False}},
        },
        "denoising": {
            "applied": False,
            "simultaneous_slice_artifact": {"Applied": marss, "OfficialPackageVersion": "1.0.2"},
        },
        "options": {"configuration": {"output_spaces": ["T1w", "fsnative"]}},
    }


class FakeRegistry:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.rows: dict[str, list[dict]] = {"anat": [], "func": []}
        for index, (sdc, fallback, state) in enumerate(
            [
                ("topup_sbref_pe", None, "fresh"),
                ("syn", "no readout metadata", "fresh"),
                ("synbold_disco", None, "stale"),
            ]
        ):
            path = tmp_path / f"sub-01_run-{index}_desc-preprocessFunc_manifest.json"
            path.write_text(json.dumps(_func_manifest(sdc, fallback=fallback, marss=index == 0)))
            self.rows["func"].append(
                {
                    "participant": "01",
                    "entities": {"ses": "a", "task": "rest", "run": str(index)},
                    "artifact_state": state,
                    "expected_outputs": [str(path)],
                }
            )

    def lineage_artifacts(self, project, module, directory):
        del project, directory
        return self.rows[module]


def test_report_counts_fresh_manifests_and_fills_supplementary_tables(tmp_path) -> None:
    report = _report(
        modules=["func"],
        manifests=registry_manifests({"demo": FakeRegistry(tmp_path)}),
    )
    text = report.files["methods.rtf"]
    runs = report.files["supplementary_runs.tsv"].splitlines()

    assert "A total of 2 blood-oxygen-level-dependent (BOLD) runs from 1 participant" in text
    assert (
        "Of the 2 runs, 1 used TOPUP with acquired references, 0 used SynBOLD-DisCo, and 1 used"
        in text
    )
    assert "This applied to 1 of 2 runs" in text
    assert "demo\t01\ta\trest\t\t\t\t1\t\t\tfunc\tsdc_fallback_reason\tno readout metadata" in runs
    assert report.trace["lineages"][1]["excluded_work_items"] == {"stale": 1}
    assert "MARSS\t1.0.2\tmanifest" in report.files["supplementary_software.tsv"]
    assert "SynBOLD-DisCo" not in report.files["supplementary_software.tsv"]
    assert "Run-level processing" not in text
    assert "no readout metadata" not in text
    assert "Supplementary Table S1. Software" in text


def test_registry_source_pools_projects_and_applies_selectors(tmp_path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    anat = {
        "participant": "01",
        "entities": {"ses": "a"},
        "artifact_state": "stale",
        "expected_outputs": [],
    }
    registries = {"demo": FakeRegistry(first), "other": FakeRegistry(second)}
    for registry in registries.values():
        registry.rows["anat"] = [anat]
    registries["other"].rows["func"][0]["entities"]["task"] = "story"
    source = registry_manifests(registries, ManifestSelection(runs={"task": ("rest",)}))
    lineages = {
        item.module: item for item in workflow_lineages(ConfigStore().resolve("main"), ["func"])
    }

    func = source(lineages["func"], tuple(lineages.values()))
    anat_found = source(lineages["anat"], tuple(lineages.values()))

    assert [(r.entities["project"], r.entities["run"]) for r in func.records] == [
        ("demo", "0"),
        ("demo", "1"),
        ("other", "1"),
    ]
    assert func.excluded == {"stale": 2}
    assert anat_found.excluded == {"stale": 2}

    registries["demo"].rows["anat"] = [
        {**anat, "participant": participant, "entities": {"ses": session}}
        for participant, session in (("01", "a"), ("01", "b"), ("02", "a"))
    ]
    registries["other"].rows["anat"] = []
    for row in registries["demo"].rows["func"]:
        row["entities"]["task"] = "story"
    restricted = registry_manifests(
        {"demo": registries["demo"]}, ManifestSelection(runs={"task": ("story",)})
    )
    assert restricted(lineages["anat"], tuple(lineages.values())).excluded == {"stale": 1}
    with pytest.raises(MethodsReportError, match="do not apply to anat"):
        restricted(lineages["anat"], (lineages["anat"],))
    assert participants(func.records) == 2

    only = registry_manifests(registries, ManifestSelection(participants=("02",)))
    assert only(lineages["func"], tuple(lineages.values())).records == ()


def test_bibliography_output_contains_only_cited_entries() -> None:
    report = _report()
    cited = set(report.trace["citations"])
    written = set(parse_bibliography(report.files["references.bib"]))

    assert written == cited
    assert "satterthwaite2013confound" in cited
    assert "afni3dtproject" not in written
    assert "pruim2015aroma" not in cited


def test_software_rows_report_pinned_versions_only_for_matching_receipts() -> None:
    def images(keys):
        return {
            key: ImageRecord(key, f"/{key}.sif", f"docker://example/{key}@sha256:{key}")
            for key in keys
        }

    report = _report(images=images, pins={"qunex": "docker://example/qunex@sha256:qunex"})
    rows = report.files["supplementary_software.tsv"]

    assert "QuNex\t1.5.1\tpinned image\tdocker://example/qunex\tsha256:qunex" in rows
    assert "SynthStrip\tunknown\tsite image" in rows


@pytest.mark.parametrize(
    "output_format,name", [("rtf", "main_preview.rtf"), ("latex", "main_preview.tex")]
)
def test_main_workflow_preview_matches_golden_file(output_format, name) -> None:
    text = _report(output_format).files[f"methods.{'rtf' if output_format == 'rtf' else 'tex'}"]
    golden = GOLDEN / name
    if os.environ.get("NRO_UPDATE_GOLDEN"):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text, encoding="utf-8")
    assert text == golden.read_text(encoding="utf-8")


def test_report_is_deterministic() -> None:
    assert _report().files == _report().files


# Command


def test_methods_workflow_defaults_to_main() -> None:
    from nro.bin.methods import build_parser

    args = build_parser().parse_args(["--preview"])
    assert args.workflow == "main"
    assert args.output_format == "rtf"
    assert build_parser().parse_args(["other", "--preview", "--format", "rtf"]).workflow == "other"


def test_methods_command_writes_preview_files(tmp_path, capsys) -> None:
    from nro.bin.methods import main

    main(["main", "--preview", "--format", "latex", "-o", str(tmp_path)])

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "methods.json",
        "methods.tex",
        "references.bib",
        "supplementary_runs.tsv",
        "supplementary_software.tsv",
    ]
    assert str(tmp_path / "methods.tex") in capsys.readouterr().out


def test_methods_command_rejects_selectors_in_preview(tmp_path) -> None:
    from nro.bin.methods import main

    with pytest.raises(SystemExit):
        main(["main", "--preview", "--format", "rtf", "--task", "rest", "-o", str(tmp_path)])


def test_methods_command_requires_project_or_preview(tmp_path) -> None:
    from nro.bin.methods import main

    with pytest.raises(SystemExit):
        main(["main", "--format", "rtf", "-o", str(tmp_path)])


def test_connectivity_contexts_count_runs_and_report_strategy() -> None:
    from nro.methods.context import ManifestRecord
    from nro.modules.dynconn import methods as dynconn_methods
    from nro.modules.networks import methods as networks_methods

    workflow = ConfigStore().resolve("main")
    values = {name: config.values for name, config in workflow.configurations.items()}

    def run(name: str) -> dict:
        return {"files": [f"/clean/{name}.dtseries.nii"]}

    records = [
        ManifestRecord(
            {"project": "p", "participant": "01", "space": space},
            {
                "space": space,
                "smoothing_fwhm_mm": 2,
                "functional_runs": {"included": [run("a")], "skipped": [run("b")]},
                "low_rank": {"dimensions": dimensions},
            },
        )
        for space, dimensions in (("fsnative", 900), ("T1w", 950))
    ]
    context = dynconn_methods.methods(values["dynconn"], records, workflow=values).context
    assert (context["participants"], context["included_runs"], context["excluded_runs"]) == (
        1,
        1,
        1,
    )
    assert context["fitted_dimensions"] == ["900", "950"]

    networks = networks_methods.methods(values["networks"], records, workflow=values).context
    assert networks["strategy"] == "ica" and networks["low_rank_source"]
    assert networks["spaces"] == ["T1w", "fsnative"] and networks["smoothing_mm"] == ["2"]
    with pytest.raises(MethodsReportError, match="strategy 'oslom'"):
        networks_methods.methods(
            {**values["networks"], "parcellation_strategy": "oslom"}, records, workflow=values
        )
