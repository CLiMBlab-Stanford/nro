"""Unit tests for compact branch-to-scheduler request handoff."""

from dataclasses import replace

from nro.orchestration.branch_requests import _request_groups
from nro.orchestration.contracts import WorkItemSpec
from nro.orchestration.planner import RequestPlan
from nro.orchestration.workflow_registry import RegisteredWorkflow


def _work_item(key: str) -> WorkItemSpec:
    return WorkItemSpec.create(
        key=key,
        module="anat",
        project="demo",
        participant=key,
        entities={},
        scope="subject",
        module_lineage_id=1,
        config_fingerprint="science",
        directory_label="main-lineage",
        runtime_config="runtime.yml",
        command=("nro", "anat", key),
        dependencies=(),
        input_paths=(),
        output_root=f"/derivatives/{key}",
        output_prefix=key,
        expected_outputs=(f"/derivatives/{key}/{key}.nii.gz",),
        resource_class="small",
    )


def test_request_groups_combine_independent_endpoints_per_workflow():
    registered = RegisteredWorkflow(
        workflow_id="main",
        revision=1,
        revision_id=1,
        fingerprint="workflow",
        lineages={"anat": 1},
        lineage_fingerprints={"anat": "lineage"},
        directories={"anat": "main-lineage"},
        created=False,
    )
    first = _work_item("sub-01")
    second = _work_item("sub-02")
    base = RequestPlan(
        project="demo",
        workflow_id="main",
        registered=registered,
        module="anat",
        participants=("01",),
        work_items=(first,),
        terminal_keys=(first.key,),
    )

    groups = _request_groups(
        (
            base,
            replace(
                base,
                participants=("02",),
                work_items=(second,),
                terminal_keys=(second.key,),
            ),
            replace(base, project="other"),
        )
    )

    assert len(groups) == 2
    combined = next(group for group in groups if group.project == "demo")
    assert tuple(combined.work_items) == (first.key, second.key)
    assert combined.terminal_keys == [first.key, second.key]
