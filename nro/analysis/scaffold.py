"""Create study repositories and analysis skeletons."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from nro.analysis.layout import API_VERSION, StudyError, check_study_name
from nro.analysis.manifest import _ANALYSIS_NAME, SUPPORTED_SCOPES
from nro.engine.paths import study_code_root, study_results_root


def _module_name(study: str) -> str:
    name = "".join(character if character.isalnum() else "_" for character in study.lower())
    return f"{name if name[0].isalpha() else 's' + name}_common"


def _git(repository: Path, *arguments: str) -> None:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments), capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise StudyError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")


def init_study(study: str, *, projects: tuple[str, ...] = ()) -> Path:
    """Create a study repository and its results directory, and make the first commit."""
    check_study_name(study)
    repository = study_code_root(study)
    if repository.exists():
        raise StudyError(f"{repository} already exists")
    common = _module_name(study)
    (repository / "common" / common).mkdir(parents=True)
    (repository / "analyses").mkdir()
    (repository / "analyses" / ".gitkeep").write_text("", encoding="utf-8")
    (repository / "study.yml").write_text(
        yaml.safe_dump(
            {"name": study, "projects": list(projects), "api": API_VERSION}, sort_keys=False
        ),
        encoding="utf-8",
    )
    (repository / "pyproject.toml").write_text(
        f"""[project]
name = "{study}-study"
version = "0"
description = "Analyses for study {study}, run by nro"
requires-python = ">=3.11"
# Dependencies are not installed at run time. Every package listed here must
# already be installed in the nro environment or in the study's container.
dependencies = []

[tool.nro]
analysis_api = {API_VERSION}
""",
        encoding="utf-8",
    )
    (repository / "common" / common / "__init__.py").write_text(
        f'"""Code shared by the analyses of study {study}."""\n', encoding="utf-8"
    )
    (repository / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    study_results_root(study).mkdir(parents=True, exist_ok=True)
    _git(repository, "init", "--initial-branch=main", "--quiet")
    _git(repository, "config", "receive.denyCurrentBranch", "updateInstead")
    _git(repository, "add", "--all")
    _git(repository, "commit", "--quiet", "--message", f"Initialize study {study}")
    return repository


def _manifest(scope: str) -> str:
    placeholder = "{participant}" if scope == "subject" else "{stem}"
    name = f"sub-{placeholder}" if scope == "subject" else placeholder
    return f"""version: 1
scope: {scope}
inputs:
  maps:
    from: contrasts
    workflow: main
    select: {{node: run, stat: t, suffix: statmap}}
    many: true
outputs:
  summary: "{name}_desc-example_summary.tsv"
parameters: {{}}
resources: {{class: small}}
"""


_CODE = '''"""{analysis}: describe what this analysis computes."""

from nro.analysis import Inputs, Outputs


def run(inputs: Inputs, params: dict, outputs: Outputs) -> None:
    """Compute the analysis for one work item and write every output slot."""
    rows = [{{"path": artifact.path.name}} for artifact in inputs["maps"]]
    outputs["summary"].write_tsv(rows)
'''

_TEST = '''"""Interactive check of {analysis} against real data.

Run with ``python -m pytest`` from the study repository after setting PROJECT
and PARTICIPANT to a processed participant.
"""

import pytest

from nro.analysis import query

PROJECT = None
PARTICIPANT = None


@pytest.mark.skipif(PROJECT is None, reason="set PROJECT and PARTICIPANT")
def test_inputs_resolve():
    maps = query(
        project=PROJECT,
        participant=PARTICIPANT,
        module="contrasts",
        workflow="main",
        select={{"node": "run", "stat": "t", "suffix": "statmap"}},
        many=True,
    )
    assert len(maps) > 0
'''


def add_analysis(study: str, analysis: str, *, scope: str = "subject") -> Path:
    """Write an analysis skeleton without committing it."""
    check_study_name(study)
    if not _ANALYSIS_NAME.match(analysis):
        raise StudyError(
            f"Analysis name {analysis!r} must use lowercase letters, digits, and underscores"
        )
    if scope not in SUPPORTED_SCOPES:
        raise StudyError(f"--scope must be one of {', '.join(SUPPORTED_SCOPES)}")
    repository = study_code_root(study)
    if not (repository / ".git").exists():
        raise StudyError(f"No study repository at {repository}; run nro study init {study}")
    directory = study_code_root(study, analysis)
    if directory.exists():
        raise StudyError(f"{directory} already exists")
    directory.mkdir(parents=True)
    (directory / "analysis.yml").write_text(_manifest(scope), encoding="utf-8")
    (directory / "analysis.py").write_text(_CODE.format(analysis=analysis), encoding="utf-8")
    (directory / "test_analysis.py").write_text(_TEST.format(analysis=analysis), encoding="utf-8")
    return directory
