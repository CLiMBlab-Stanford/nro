"""Scan-plan contracts, site parsers, sources, and assignment boundaries."""

from types import SimpleNamespace

import pytest

from nro.bidsify.config import load_config
from nro.bidsify.scanplans import (
    ScanPlan,
    ScanPlanRow,
    align_scanplan,
    apply_scanplan,
    choose_scanplan,
    drive_files,
    local_files,
    manual_selection,
    run_parser,
    scanplan_document,
)
from nro.bidsify.store import IngestionStore
from nro.orchestration.registry import Registry


def test_contract_validates_order_and_bold_annotations():
    plan = ScanPlan(
        (
            ScanPlanRow(1, "localizer", include=False),
            ScanPlanRow(2, "bold", phase_encoding="j", task="language"),
        ),
    )
    document = scanplan_document(plan)
    assert document["rows"][1]["task"] == "language"
    with pytest.raises(ValueError, match="Only BOLD"):
        scanplan_document(ScanPlan((ScanPlanRow(1, "T1w", task="language"),)))
    with pytest.raises(ValueError, match="ascending"):
        scanplan_document(ScanPlan((ScanPlanRow(2, "T1w"), ScanPlanRow(1, "T2w"))))


def test_unconfigured_parser_raises_not_implemented(tmp_path):
    source = tmp_path / "anything"
    source.write_text("site-defined")
    with pytest.raises(NotImplementedError, match="No site"):
        run_parser(source, None)


def test_external_parser_uses_published_types(tmp_path):
    source = tmp_path / "plan.custom"
    source.write_text("input")
    parser = tmp_path / "parser.py"
    parser.write_text(
        "from nro.bidsify.scanplans import ScanPlan, ScanPlanRow\n"
        "def parse_scanplan(source):\n"
        "    return ScanPlan((ScanPlanRow(1, 'bold', task=source.read_text()),))\n"
    )
    assert run_parser(source, parser).rows[0].task == "input"


def test_local_source_recurses_and_ignores_symlinks(tmp_path):
    root = tmp_path / "plans"
    (root / "nested").mkdir(parents=True)
    (root / "a.docx").write_bytes(b"a")
    (root / "nested/b.pdf").write_bytes(b"b")
    (root / "link").symlink_to(root / "a.docx")
    files = local_files(root)
    assert [item.name for item in files] == ["a.docx", "nested/b.pdf"]
    assert choose_scanplan(files, "2") == files[1]
    with pytest.raises(ValueError, match="displayed"):
        choose_scanplan(files, "0")


def test_machine_readable_fallback_has_fixed_columns(tmp_path):
    path = tmp_path / "plan.tsv"
    path.write_text(
        "ordinal\tacquisition_type\tphase_encoding\tinclude\ttask\n"
        "1\tshim\t\tfalse\t\n"
        "2\tbold\tj\ttrue\tlanguage\n"
    )
    selected = manual_selection(path)
    assert selected["plan"]["rows"][1]["task"] == "language"
    assert selected["source"]["id"].startswith("manual-")


def test_alignment_uses_type_and_phase_encoding_but_not_task():
    plan = scanplan_document(
        ScanPlan(
            (
                ScanPlanRow(1, "localizer", include=False),
                ScanPlanRow(2, "bold", phase_encoding="j", task="language"),
            )
        )
    )
    acquisitions = [
        {
            "id": "loc",
            "datatype": "ignore",
            "suffix": "ignore",
            "classification": {"bids_guess": ["discard", "_localizer"]},
            "metadata": {"SeriesNumber": 1},
            "entities": {},
        },
        {
            "id": "bold",
            "datatype": "func",
            "suffix": "bold",
            "classification": {"bids_guess": ["func", "_task-rest_bold"]},
            "metadata": {"SeriesNumber": 2, "PhaseEncodingDirection": "j"},
            "entities": {"task": "unknown"},
        },
    ]
    alignment = align_scanplan(plan, acquisitions)
    assert alignment["complete"] is True
    record = {"scanplan": {"plan": plan, "alignment": None}, "acquisitions": acquisitions}
    apply_scanplan(record)
    assert record["acquisitions"][0]["scanplan_include"] is False
    assert record["acquisitions"][1]["entities"]["task"] == "language"


def test_alignment_reports_sequence_differences_without_guessing():
    plan = scanplan_document(ScanPlan((ScanPlanRow(1, "bold"),)))
    acquisitions = [
        {
            "id": "t1",
            "datatype": "anat",
            "suffix": "T1w",
            "classification": {},
            "metadata": {"SeriesNumber": 1},
            "entities": {},
        },
        {
            "id": "bold",
            "datatype": "func",
            "suffix": "bold",
            "classification": {},
            "metadata": {"SeriesNumber": 2},
            "entities": {},
        },
    ]
    alignment = align_scanplan(plan, acquisitions)
    assert alignment["complete"] is False
    assert any(row["status"] == "unplanned_acquisition" for row in alignment["rows"])


def test_drive_source_recurses_and_paginates(monkeypatch):
    responses = {
        "root": {
            "files": [
                {
                    "id": "file",
                    "name": "a.docx",
                    "mimeType": "application/docx",
                    "md5Checksum": "x",
                },
                {
                    "id": "child",
                    "name": "nested",
                    "mimeType": "application/vnd.google-apps.folder",
                },
            ]
        },
        "child": {
            "files": [
                {"id": "pdf", "name": "b.pdf", "mimeType": "application/pdf", "modifiedTime": "now"}
            ]
        },
    }

    class Files:
        def list(self, **kwargs):
            folder = kwargs["q"].split("'", 2)[1]
            return SimpleNamespace(execute=lambda: responses[folder])

    monkeypatch.setattr(
        "nro.bidsify.scanplans._drive_service",
        lambda _: SimpleNamespace(files=lambda: Files()),
    )
    files = drive_files("https://drive.google.com/drive/folders/root")
    assert [(item.id, item.name) for item in files] == [
        ("file", "a.docx"),
        ("pdf", "nested/b.pdf"),
    ]


def test_scanplan_assignment_is_unique_within_source_location(tmp_path):
    registry = Registry.for_project("", bids_root=tmp_path / "bids")
    store = IngestionStore(registry)
    config = load_config()
    config["staging"] = str(tmp_path / "staging")
    config["scanplans"]["location"] = str(tmp_path / "plans")
    first = store.create(server="cni", remote_session="one", project="demo", config=config)
    second = store.create(server="cni", remote_session="two", project="demo", config=config)
    selected = {
        "source": {"id": "source", "revision": "1", "sha256": "hash"},
        "parser": None,
        "plan": scanplan_document(ScanPlan((ScanPlanRow(1, "bold"),))),
        "alignment": None,
    }
    with store.review_session(first["id"]) as token:
        store.update(first, expected_revision=first["revision"], review_token=token)
    assigned = store.set_scanplan(
        first["id"],
        selected,
        expected_revision=first["revision"],
        expected_scanplan=None,
    )
    assert assigned["scanplan"] == selected
    with pytest.raises(ValueError, match="already assigned"):
        store.set_scanplan(second["id"], selected, expected_revision=second["revision"])
