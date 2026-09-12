"""Run the scheduled stages of one reviewed ingestion request."""

import json
import logging
import shutil
from itertools import count
from pathlib import Path

from nro.engine.io import atomic_write_text
from nro.orchestration.runner import Runner
from nro.orchestration.runner_graph import Step

from .errors import BidsificationError
from .flywheel import FlywheelSource
from .images import command, prepare_image
from .paths import secure_directory
from .publication import inventory, publish
from .review import issues


def prefix(record: dict, item: dict) -> str:
    """Build a BIDS prefix from the approved acquisition entities."""
    entities = {
        "sub": record["participant"],
        "ses": record["session"],
        **item["entities"],
    }
    return "_".join(
        f"{key}-{entities[key]}"
        for key in ("sub", "ses", "task", "acq", "dir", "run", "echo")
        if key in entities
    )


def convert(record: dict, registry) -> dict:
    """Organize sanitized helper images with dcm2bids and write approved associations."""
    unresolved = issues(record, prepared=True)
    if unresolved:
        return {"state": "needs_input", "issues": unresolved}
    root = Path(record["config"]["staging"]) / record["id"]
    bids = root / "bids"
    if bids.exists():
        if bids.is_symlink():
            raise BidsificationError("BIDS staging cannot be a symlink")
        shutil.rmtree(bids)
    secure_directory(bids)
    acquisitions = {a["id"]: a for a in record["acquisitions"]}
    sbref_owners = {}
    for item in acquisitions.values():
        if item.get("sbref"):
            if item["sbref"] in sbref_owners:
                raise BidsificationError(
                    "A shared SBRef needs an explicit sharing representation; assign distinct references or none"
                )
            sbref_owners[item["sbref"]] = item
    destinations = {}
    for item in acquisitions.values():
        if (
            item["datatype"] == "ignore"
            or item["suffix"] == "sbref"
            and item["id"] not in sbref_owners
        ):
            continue
        selected = sbref_owners.get(item["id"], item)
        entities = "_".join(
            f"{key}-{selected['entities'][key]}"
            for key in ("task", "acq", "dir", "run", "echo")
            if key in selected["entities"]
        )
        description = {
            "datatype": item["datatype"],
            "suffix": item["suffix"],
            "criteria": {"SidecarFilename": "image.json"},
        }
        if entities:
            description["custom_entities"] = entities
        if selected["entities"].get("task"):
            description["sidecar_changes"] = {"TaskName": selected["entities"]["task"]}
        config = root / f"{item['id']}-dcm2bids.json"
        atomic_write_text(config, json.dumps({"descriptions": [description]}))
        helper = root / "helpers" / item["id"]
        from dcm2bids.dcm2bids_gen import Dcm2BidsGen

        Dcm2BidsGen(
            dicom_dir=[str(helper)],
            participant=record["participant"],
            session=record["session"],
            config=str(config),
            output_dir=str(bids),
            skip_dcm2niix=True,
            force_dcm2bids=True,
        ).run()
        directory = (
            bids / f"sub-{record['participant']}" / f"ses-{record['session']}" / item["datatype"]
        )
        expected = directory / f"{prefix(record, selected)}_{item['suffix']}.nii.gz"
        if not expected.is_file():
            raise BidsificationError(
                "dcm2bids did not produce the approved filename; publication stopped"
            )
        sidecar = expected.with_name(expected.name[:-7] + ".json")
        metadata = json.loads(sidecar.read_text())
        if item["suffix"] == "bold":
            metadata["NROReferencePolicy"] = "explicit"
            metadata["NROSBRef"] = (
                f"{prefix(record, selected)}_sbref.nii.gz" if item.get("sbref") else None
            )
            metadata["B0FieldSource"] = (
                ["nro" + min(item["fieldmaps"]).replace("-", "")] if item.get("fieldmaps") else []
            )
            if item["events"]:
                shutil.copyfile(
                    item["events"], directory / f"{prefix(record, selected)}_events.tsv"
                )
                if item.get("events_source"):
                    metadata["NROEventsSource"] = item["events_source"]
        atomic_write_text(sidecar, json.dumps(metadata, indent=2))
        destinations[item["id"]] = (expected, sidecar)
    pairs = {}
    for item in acquisitions.values():
        if item["suffix"] == "bold" and item.get("fieldmaps"):
            group = tuple(sorted(item["fieldmaps"]))
            pairs.setdefault(group, []).append(item["id"])
    used = set()
    for pair, runs in pairs.items():
        if used.intersection(pair):
            raise BidsificationError("A fieldmap belongs to conflicting pairs")
        used.update(pair)
        for key in pair:
            path, sidecar = destinations[key]
            metadata = json.loads(sidecar.read_text())
            metadata["B0FieldIdentifier"] = "nro" + min(pair).replace("-", "")
            metadata["IntendedFor"] = [
                "bids::" + str(destinations[r][0].relative_to(bids)) for r in runs
            ]
            atomic_write_text(sidecar, json.dumps(metadata, indent=2))
    atomic_write_text(
        bids / "dataset_description.json",
        json.dumps({"Name": record["project"], "BIDSVersion": "1.11.0", "DatasetType": "raw"}),
    )
    command([*record["config"]["validator"], str(bids)])
    from importlib.metadata import version

    return {
        "state": "awaiting_approval",
        "issues": [],
        "validation": "passed",
        "software": {name: version(name) for name in ("dcm2bids", "pydicom", "nibabel")},
        "output_hashes": inventory(
            bids / f"sub-{record['participant']}" / f"ses-{record['session']}"
        ),
    }


def run_stage(record: dict, registry, *, source=None, branch_paths=None) -> dict:
    """Run one fixed stage; return its next state without asking questions."""
    root = secure_directory(Path(record["config"]["staging"]) / record["id"])
    if record["stage"] in {"inspect", "prepare"}:
        source = source or FlywheelSource(record["config"]["servers"][record["server"]])
    if record["stage"] == "inspect":
        acquisitions = source.inventory(record["remote_session"])
        return {
            "state": "queued",
            "stage": "prepare",
            "acquisitions": acquisitions,
            "issues": [],
        }
    if record["stage"] == "prepare":
        runner = Runner(
            module_name="Bidsification preparation",
            container=None,
            binds=(),
            logger=logging.getLogger("bidsify"),
            next_step=count(1).__next__,
        )
        for item in record["acquisitions"]:
            if (
                item["datatype"] == "ignore"
                and item.get("confirmed")
                and not item.get("classification_override")
            ):
                continue

            def action(item=item):
                item["metadata"] = prepare_image(record, item, root / "helpers", source)

            runner.add_step(
                Step.python(
                    name=f"Prepare acquisition {item['id']}",
                    action=action,
                    outputs=(root / "helpers" / item["id"] / ".prepared.json",),
                    force=True,
                )
            )
        with runner.run_context():
            runner.execute()
        return {
            "state": "needs_input",
            "stage": "convert",
            "acquisitions": record["acquisitions"],
            "issues": issues(record, prepared=True),
        }
    if record["stage"] == "convert":
        return convert(record, registry)
    if record["stage"] == "publish":
        return {"state": "published", **publish(record, registry, branch_paths=branch_paths)}
    raise BidsificationError("Unknown ingestion stage")
