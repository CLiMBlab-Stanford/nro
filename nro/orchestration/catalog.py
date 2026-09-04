"""Closed catalog of scientific modules understood by the planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

from nro.anat.planning import plan_instances as plan_anat_instances
from nro.clean.planning import plan_instances as plan_clean_instances
from nro.func.planning import plan_instances as plan_func_instances
from nro.microparcellation.planning import plan_instances as plan_microparcellation_instances
from nro.networks.planning import plan_instances as plan_networks_instances

if TYPE_CHECKING:
    from nro.orchestration.contracts import InstanceSpec
    from nro.orchestration.planning_context import SubjectPlanningContext


PlanFunction = Callable[
    ["SubjectPlanningContext", Mapping[str, tuple["InstanceSpec", ...]], "ModuleDescriptor"],
    tuple["InstanceSpec", ...],
]


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


BUILTIN_MODULES = (
    ModuleDescriptor(
        name="anat",
        configuration_class="preprocessing",
        scope="subject",
        output_format="BIDS anatomical images, surfaces, transforms, and module manifest",
        resource_class="large",
        upstream_modules=(),
        plan=plan_anat_instances,
    ),
    ModuleDescriptor(
        name="func",
        configuration_class="preprocessing",
        scope="run",
        output_format="BIDS functional images, confounds, transforms, and module manifest",
        resource_class="large",
        upstream_modules=("anat",),
        plan=plan_func_instances,
    ),
    ModuleDescriptor(
        name="clean",
        configuration_class="clean",
        scope="run",
        output_format="BIDS cleaned functional images and module manifest",
        resource_class="medium",
        upstream_modules=("func",),
        plan=plan_clean_instances,
    ),
    ModuleDescriptor(
        name="microparcellation",
        configuration_class="microparcellation",
        scope="subject",
        output_format="CIFTI microparcellation products and Workbench scene",
        resource_class="large",
        upstream_modules=("clean",),
        plan=plan_microparcellation_instances,
    ),
    ModuleDescriptor(
        name="networks",
        configuration_class="networks",
        scope="subject",
        output_format="CIFTI network maps, labels, metadata, and Workbench scene",
        resource_class="medium",
        upstream_modules=("microparcellation", "anat"),
        plan=plan_networks_instances,
    ),
)

MODULE_CATALOG = {descriptor.name: descriptor for descriptor in BUILTIN_MODULES}
MODULES = tuple(MODULE_CATALOG)


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
