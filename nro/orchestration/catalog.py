"""Closed catalog of scientific modules understood by the planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

from nro.anat.contract import anatomical_output_contract
from nro.anat.planning import plan_instances as plan_anat_instances
from nro.clean.contract import clean_output_contract
from nro.clean.planning import plan_instances as plan_clean_instances
from nro.func.contracts import final_resampling_contract, functional_output_contract
from nro.func.planning import plan_instances as plan_func_instances
from nro.firstlevels.contract import firstlevels_output_contract, validate_public_definition
from nro.firstlevels.planning import plan_instances as plan_firstlevels_instances
from nro.firstlevels.planning import direct_inputs as firstlevels_direct_inputs, select_model_runs
from nro.firstlevels.task_models import canonical_processing, model_contract, select_models
from nro.firstlevels.planning import refresh_command as refresh_firstlevels_command
from nro.microparcellation.contract import microparcellation_output_contract
from nro.microparcellation.planning import plan_instances as plan_microparcellation_instances
from nro.networks.contract import networks_output_contract
from nro.networks.planning import plan_instances as plan_networks_instances

if TYPE_CHECKING:
    from nro.orchestration.contracts import InstanceSpec
    from nro.orchestration.planning_context import SubjectPlanningContext


PlanFunction = Callable[
    ["SubjectPlanningContext", Mapping[str, tuple["InstanceSpec", ...]], "ModuleDescriptor"],
    tuple["InstanceSpec", ...],
]
ProcessingContractFunction = Callable[[], Mapping[str, object]]


def _anat_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": anatomical_output_contract()}


def _func_processing_contract() -> Mapping[str, object]:
    return {
        "final_resampling": final_resampling_contract(),
        "output_metadata": functional_output_contract(),
    }


def _clean_processing_contract() -> Mapping[str, object]:
    return {"output_metadata": clean_output_contract()}


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
    configuration_class: str
    scope: str
    output_format: str
    resource_class: str
    upstream_modules: tuple[str, ...]
    plan: PlanFunction
    processing_contract: ProcessingContractFunction
    select_runs: Callable | None = None
    direct_inputs: Callable | None = None
    instance_processing: Callable | None = None
    refresh_command: Callable | None = None
    select_models: Callable | None = None
    validate_public_definition: Callable | None = None
    canonical_processing: Callable | None = None

    def processing_for(self, entities: dict) -> dict:
        """Combine module policy with any instance-specific scientific definition."""
        return {**self.processing_contract(), **(self.instance_processing(entities) if self.instance_processing else {})}


BUILTIN_MODULES = (
    ModuleDescriptor(
        name="anat",
        configuration_class="preprocessing",
        scope="subject",
        output_format="BIDS anatomical images, surfaces, transforms, and module manifest",
        resource_class="large",
        upstream_modules=(),
        plan=plan_anat_instances,
        processing_contract=_anat_processing_contract,
    ),
    ModuleDescriptor(
        name="func",
        configuration_class="preprocessing",
        scope="run",
        output_format="BIDS functional images, confounds, transforms, and module manifest",
        resource_class="large",
        upstream_modules=("anat",),
        plan=plan_func_instances,
        processing_contract=_func_processing_contract,
    ),
    ModuleDescriptor(
        name="clean",
        configuration_class="clean",
        scope="run",
        output_format="BIDS cleaned functional images and module manifest",
        resource_class="medium",
        upstream_modules=("func",),
        plan=plan_clean_instances,
        processing_contract=_clean_processing_contract,
    ),
    ModuleDescriptor(
        name="microparcellation",
        configuration_class="microparcellation",
        scope="subject",
        output_format="Subject-level BIDS CIFTI microparcellation products",
        resource_class="large",
        upstream_modules=("clean",),
        plan=plan_microparcellation_instances,
        processing_contract=_microparcellation_processing_contract,
    ),
    ModuleDescriptor(
        name="networks",
        configuration_class="networks",
        scope="subject",
        output_format="Subject-level BIDS CIFTI network maps, labels, and metadata",
        resource_class="medium",
        upstream_modules=("microparcellation", "anat"),
        plan=plan_networks_instances,
        processing_contract=_networks_processing_contract,
    ),
)

BUILTIN_MODULES += (
    ModuleDescriptor(
        name="firstlevels", configuration_class="firstlevels", scope="subject",
        output_format="Task/model run, session and subject GLM maps and compact covariance",
        resource_class="medium", upstream_modules=("func", "anat"),
        plan=plan_firstlevels_instances, processing_contract=_firstlevels_processing_contract,
        select_runs=select_model_runs,
        direct_inputs=firstlevels_direct_inputs,
        instance_processing=model_contract,
        refresh_command=refresh_firstlevels_command,
        select_models=select_models,
        validate_public_definition=validate_public_definition,
        canonical_processing=canonical_processing,
    ),
)

MODULE_CATALOG = {descriptor.name: descriptor for descriptor in BUILTIN_MODULES}
MODULES = tuple(MODULE_CATALOG)


def canonical_contract(contract: dict, configuration: dict | None = None) -> dict:
    """Normalize recorded scientific syntax without consulting mutable definitions."""
    from nro.configuration.store import configuration_fingerprint

    descriptor = module_descriptor(contract["module"])
    if isinstance(configuration, dict) and contract.get("configuration") == configuration.get("fingerprint"):
        kind = descriptor.configuration_class
        values = configuration.get("resolved")
        identifier = configuration.get("id")
        if isinstance(values, dict) and isinstance(identifier, str):
            if configuration_fingerprint(kind, identifier, values) == configuration["fingerprint"]:
                contract = {**contract, "configuration": configuration_fingerprint(kind, identifier, values, scientific=True)}
    normalize = descriptor.canonical_processing
    if normalize is None or "processing" not in contract:
        return contract
    return {**contract, "processing": normalize(contract["processing"])}


def terminal_modules() -> tuple[str, ...]:
    """Return modules with no downstream consumers, in catalog order.

    Workflows select configurations for this shared graph; they do not change
    its topology. Requesting these endpoints covers every branch.
    """
    upstream = {
        name
        for descriptor in MODULE_CATALOG.values()
        for name in descriptor.upstream_modules
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
    return module_descriptor(value).name


def modules_through(target: str) -> tuple[ModuleDescriptor, ...]:
    """Return the dependency closure of a target in catalog order."""
    target = normalize_module(target)
    required: set[str] = set()

    def visit(name: str) -> None:
        if name in required:
            return
        descriptor = module_descriptor(name)
        for upstream in descriptor.upstream_modules:
            visit(upstream)
        required.add(name)

    visit(target)
    return tuple(item for item in BUILTIN_MODULES if item.name in required)
