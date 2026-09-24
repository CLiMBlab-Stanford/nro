"""Closed catalog of scientific modules understood by the planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

from nro.modules import MODULE_NAMES
from nro.modules.anat.contract import (
    anatomical_output_contract,
    bias_correction_contract,
    pose_normalization_contract,
    surface_reconstruction_contract,
)
from nro.modules.anat.planning import plan_work_items as plan_anat_work_items
from nro.modules.clean.contract import clean_output_contract
from nro.modules.clean.planning import plan_work_items as plan_clean_work_items
from nro.modules.dynconn.contract import dynconn_output_contract
from nro.modules.dynconn.planning import plan_work_items as plan_dynconn_work_items
from nro.modules.firstlevels.contract import firstlevels_output_contract, validate_public_definition
from nro.modules.firstlevels.planning import direct_inputs as firstlevels_direct_inputs
from nro.modules.firstlevels.planning import plan_work_items as plan_firstlevels_work_items
from nro.modules.firstlevels.planning import refresh_command as refresh_firstlevels_command
from nro.modules.firstlevels.planning import select_model_runs
from nro.modules.firstlevels.task_models import canonical_processing, model_contract, select_models
from nro.modules.func.contract import final_resampling_contract, functional_output_contract
from nro.modules.func.planning import plan_work_items as plan_func_work_items
from nro.modules.microparcellation.contract import microparcellation_output_contract
from nro.modules.microparcellation.planning import (
    plan_work_items as plan_microparcellation_work_items,
)
from nro.modules.networks.contract import networks_output_contract
from nro.modules.networks.planning import plan_work_items as plan_networks_work_items
from nro.orchestration.dependencies import direct_dependencies

if TYPE_CHECKING:
    from nro.orchestration.contracts import WorkItemSpec
    from nro.orchestration.planning_context import SubjectPlanningContext


PlanFunction = Callable[
    ["SubjectPlanningContext", Mapping[str, tuple["WorkItemSpec", ...]], "ModuleDescriptor"],
    tuple["WorkItemSpec", ...],
]
ProcessingContractFunction = Callable[[], Mapping[str, object]]


def _anat_processing_contract() -> Mapping[str, object]:
    return {
        "bias_correction": bias_correction_contract(),
        "output_metadata": anatomical_output_contract(),
        "pose_normalization": pose_normalization_contract(),
        "surface_reconstruction": surface_reconstruction_contract(),
    }


def _func_processing_contract() -> Mapping[str, object]:
    return {
        "final_resampling": final_resampling_contract(),
        "output_metadata": functional_output_contract(),
    }


def _clean_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": clean_output_contract()}


def _dynconn_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": dynconn_output_contract()}


def _microparcellation_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": microparcellation_output_contract()}


def _networks_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": networks_output_contract()}


def _firstlevels_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": firstlevels_output_contract()}


@dataclass(frozen=True)
class ModuleDescriptor:
    """Planner-facing properties of one built-in scientific module."""

    name: str
    scope: str
    output_format: str
    resource_class: str
    plan: PlanFunction
    processing_contract: ProcessingContractFunction
    select_runs: Callable | None = None
    direct_inputs: Callable | None = None
    work_item_processing: Callable | None = None
    refresh_command: Callable | None = None
    select_models: Callable | None = None
    validate_public_definition: Callable | None = None
    canonical_processing: Callable | None = None
    dynamic_processing_keys: tuple[str, ...] = ()

    @property
    def configuration_class(self) -> str:
        """Return the module's configuration namespace."""
        return self.name

    @property
    def execution_module(self) -> str:
        """Return the Python module used to execute this scientific module."""
        return f"nro.modules.{self.name}"

    def processing_for(self, entities: dict) -> dict:
        """Combine module policy with any work item-specific scientific definition."""
        return {
            **self.processing_contract(),
            **(self.work_item_processing(entities) if self.work_item_processing else {}),
        }

    def dependencies_for(self, configuration: Mapping[str, object]) -> tuple[str, ...]:
        """Return direct dependencies for one resolved configuration."""

        return direct_dependencies(self.name, configuration)


BUILTIN_MODULES = (
    ModuleDescriptor(
        name="anat",
        scope="subject",
        output_format="BIDS-like anatomical images, surfaces, transforms, and module manifest",
        resource_class="large",
        plan=plan_anat_work_items,
        processing_contract=_anat_processing_contract,
        dynamic_processing_keys=(
            "gradient_unwarping",
            "lesion_reconstruction",
            "surface_reconstruction",
        ),
    ),
    ModuleDescriptor(
        name="func",
        scope="run",
        output_format="BIDS-like functional images, confounds, transforms, and module manifest",
        resource_class="large",
        plan=plan_func_work_items,
        processing_contract=_func_processing_contract,
        dynamic_processing_keys=("gradient_unwarping", "final_resampling"),
    ),
    ModuleDescriptor(
        name="clean",
        scope="run",
        output_format="BIDS cleaned functional images and module manifest",
        resource_class="medium",
        plan=plan_clean_work_items,
        processing_contract=_clean_processing_contract,
    ),
    ModuleDescriptor(
        name="dynconn",
        scope="subject",
        output_format="Full or low-rank dynamic-connectivity time series and metadata",
        resource_class="large",
        plan=plan_dynconn_work_items,
        processing_contract=_dynconn_processing_contract,
    ),
    ModuleDescriptor(
        name="microparcellation",
        scope="subject",
        output_format="Subject-level BIDS-like CIFTI microparcellation products",
        resource_class="large",
        plan=plan_microparcellation_work_items,
        processing_contract=_microparcellation_processing_contract,
    ),
    ModuleDescriptor(
        name="networks",
        scope="subject",
        output_format="Subject-level BIDS-like CIFTI network maps, labels, and metadata",
        resource_class="medium",
        plan=plan_networks_work_items,
        processing_contract=_networks_processing_contract,
    ),
    ModuleDescriptor(
        name="firstlevels",
        scope="subject",
        output_format="Task/model run, session and subject GLM maps and compact covariance",
        resource_class="medium",
        plan=plan_firstlevels_work_items,
        processing_contract=_firstlevels_processing_contract,
        select_runs=select_model_runs,
        direct_inputs=firstlevels_direct_inputs,
        work_item_processing=model_contract,
        refresh_command=refresh_firstlevels_command,
        select_models=select_models,
        validate_public_definition=validate_public_definition,
        canonical_processing=canonical_processing,
    ),
)

MODULE_CATALOG = {descriptor.name: descriptor for descriptor in BUILTIN_MODULES}
if tuple(MODULE_CATALOG) != MODULE_NAMES:
    raise RuntimeError("The orchestration catalog does not match nro.modules.MODULE_NAMES")
MODULES = MODULE_NAMES


def canonical_contract(contract: dict, configuration: dict | None = None) -> dict:
    """Normalize recorded scientific syntax without consulting mutable definitions."""
    from nro.configuration.store import configuration_fingerprint
    from nro.engine.artifact_metadata import metadata_contract_compatible
    from nro.orchestration.contract_migrations import migrate_contract

    recorded_contract = contract
    resolved = configuration.get("resolved") if isinstance(configuration, dict) else None
    contract, migrated_configuration = migrate_contract(recorded_contract, resolved)
    descriptor = module_descriptor(contract["module"])
    configuration_was_migrated = recorded_contract.get("contract_schema") != contract.get(
        "contract_schema"
    )
    configuration_matches_contract = isinstance(configuration, dict) and contract.get(
        "configuration"
    ) == configuration.get("fingerprint")
    if isinstance(configuration, dict) and (
        configuration_was_migrated or configuration_matches_contract
    ):
        kind = descriptor.configuration_class
        values = resolved
        identifier = configuration.get("id")
        if isinstance(values, dict) and isinstance(identifier, str):
            if configuration_fingerprint(kind, identifier, values) == configuration["fingerprint"]:
                assert migrated_configuration is not None
                contract = {
                    **contract,
                    "configuration": configuration_fingerprint(
                        kind,
                        identifier,
                        migrated_configuration,
                        scientific=True,
                    ),
                }
    processing = contract.get("processing")
    if not isinstance(processing, dict):
        return contract
    processing = dict(processing)
    current_metadata = descriptor.processing_contract().get("output_metadata")
    if current_metadata is not None and metadata_contract_compatible(
        processing.get("output_metadata"), current_metadata
    ):
        processing["output_metadata"] = current_metadata
    normalize = descriptor.canonical_processing
    if normalize is not None:
        processing = normalize(processing)
    return {**contract, "processing": processing}


def terminal_modules(workflow=None) -> tuple[str, ...]:
    """Return modules with no downstream consumers for one resolved workflow."""

    if workflow is None:
        upstream = {
            name
            for descriptor in MODULE_CATALOG.values()
            for name in descriptor.dependencies_for({})
        }
    else:
        upstream = {
            name
            for descriptor in MODULE_CATALOG.values()
            for name in descriptor.dependencies_for(
                workflow.configuration(descriptor.configuration_class).values
            )
        }
    return tuple(name for name in MODULE_CATALOG if name not in upstream)


def module_descriptor(name: str) -> ModuleDescriptor:
    """Return one built-in descriptor or reject an unsupported module."""
    try:
        return MODULE_CATALOG[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown derivative module {name!r}; choose from {', '.join(MODULES)}"
        ) from error


def normalize_module(value: str) -> str:
    """Validate and return a built-in module name."""
    return module_descriptor(value).name


def modules_through(target: str, workflow) -> tuple[ModuleDescriptor, ...]:
    """Return one workflow's dependency closure in catalog order."""
    target = normalize_module(target)
    required: set[str] = set()

    def visit(name: str) -> None:
        if name in required:
            return
        descriptor = module_descriptor(name)
        configuration = workflow.configuration(descriptor.configuration_class).values
        for upstream in descriptor.dependencies_for(configuration):
            visit(upstream)
        required.add(name)

    visit(target)
    return tuple(item for item in BUILTIN_MODULES if item.name in required)
