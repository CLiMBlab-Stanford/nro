"""Repair plans from existing outputs, not the source-data universe."""

from nro.configuration.store import ConfigStore
from nro.orchestration.discovery import register_existing_artifacts
from nro.orchestration.ownership import write_instance_ownership
from nro.orchestration.planner import Planner
from nro.orchestration.registry import Registry


def write(path, text="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sources(bids, participant="01", runs=("1",)):
    subject = bids / "demo" / f"sub-{participant}"
    write(subject / "anat" / f"sub-{participant}_T1w.nii.gz")
    for run in runs:
        prefix = f"sub-{participant}_task-rest_run-{run}_bold"
        write(subject / "func" / f"{prefix}.nii.gz")
        write(subject / "func" / f"{prefix}.json")


def spy_planning(monkeypatch):
    calls = []
    original = Planner.plan_subject

    def plan(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(Planner, "plan_subject", plan)
    return calls


def test_empty_derivative_tree_never_plans_sources(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    sources(bids)
    write(bids / "demo/derivatives/unmanaged/main/sub-01/sub-01_result.txt")
    write(bids / "demo/derivatives/preprocessing/unknown/sub-01/anat/sub-01_partial.txt")
    write(bids / "demo/derivatives/preprocessing/main/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz")
    registry = Registry.for_project("demo", bids_root=bids)
    calls = spy_planning(monkeypatch)
    result = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01",)})
    assert result.instances == 0
    assert calls == []


def test_anatomy_recovery_skips_functional_discovery_and_duplicate_workflows(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    sources(bids)
    sources(bids, "02")
    write(bids / "demo/derivatives/preprocessing/main/sub-01/anat/sub-01_partial.txt")
    registry = Registry.for_project("demo", bids_root=bids)
    calls = spy_planning(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Anatomy recovery inspected functional runs")

    monkeypatch.setattr("nro.orchestration.planner.discover_raw_runs", forbidden)
    result = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01", "02")})
    assert result.instances == 1
    assert [(call["module"], call["participant"]) for call in calls] == [("anat", "01")]


def test_clean_recovery_selects_only_existing_run_and_exact_target_pairs(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    sources(bids, runs=("1", "2"))
    (bids / "demo/sub-01/func/sub-01_task-rest_run-2_bold.json").unlink()
    for space, smoothing in [("fsnative", 2), ("T1w", 0)]:
        write(
            bids
            / "demo/derivatives/clean/main/sub-01"
            / f"sub-01_task-rest_run-1_space-{space}_smoothing-{smoothing}mm_partial.txt"
        )
    registry = Registry.for_project("demo", bids_root=bids)
    calls = spy_planning(monkeypatch)
    result = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01",)})
    assert result.artifacts == 2
    assert result.instances == 4
    assert len(calls) == 2
    assert {(call["spaces"], call["smoothing_levels"]) for call in calls} == {
        (("fsnative",), (2,)),
        (("T1w",), (0,)),
    }
    assert all(call["module"] == "clean" and call["selectors"]["run"] == ("1",) for call in calls)
    assert registry.request_rows() == []


def test_owned_records_restore_without_planning_but_new_run_is_discovered(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    sources(bids, runs=("1", "2"))
    registry = Registry.for_project("demo", bids_root=bids)
    workflow = ConfigStore().resolve("main")
    registered = registry.register_workflow(workflow)
    specs = Planner(registry, bids_root=bids).plan_subject(
        project="demo",
        participant="01",
        module="func",
        workflow=workflow,
        registered=registered,
        selectors={"run": ("1",)},
    )
    ids = registry.register_instances(specs)
    for spec in specs:
        write(
            spec.output_root
            / ("func" if spec.module == "func" else "")
            / f"{spec.output_prefix}_partial.txt"
        )
        write_instance_ownership(registry, ids[spec.key])
    registry.reinitialize()
    calls = spy_planning(monkeypatch)
    first = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01",)})
    assert first.instances == 2
    assert calls == []

    write(
        bids / "demo/derivatives/preprocessing/main/sub-01/func/sub-01_task-rest_run-2_partial.txt"
    )
    registry.reinitialize()
    second = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01",)})
    assert second.instances == 3
    assert len(calls) == 1
    assert calls[0]["module"] == "func"
    assert calls[0]["selectors"]["run"] == ("2",)


def test_microparcellation_recovery_does_not_plan_networks(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    sources(bids)
    write(
        bids
        / "demo/derivatives/microparcellation/main/sub-01"
        / "sub-01_space-fsnative_smoothing-2mm_partial.txt"
    )
    registry = Registry.for_project("demo", bids_root=bids)
    calls = spy_planning(monkeypatch)
    result = register_existing_artifacts(registry, bids_root=bids, inventory={"demo": ("01",)})
    assert result.instances == 4
    assert [call["module"] for call in calls] == ["microparcellation"]
    assert {row["module"] for row in registry.instance_rows()} == {
        "anat",
        "func",
        "clean",
        "microparcellation",
    }
