"""Define worker capabilities used by planning and scheduler capacity checks."""

from __future__ import annotations

GPU_RESOURCE_CLASS = "gpu"
GENERAL_RESOURCE_CLASS = "large"

WORKER_COMPATIBILITY = {
    GPU_RESOURCE_CLASS: (GPU_RESOURCE_CLASS,),
    GENERAL_RESOURCE_CLASS: (GENERAL_RESOURCE_CLASS, "medium", "small"),
    "medium": ("medium", "small"),
    "small": ("small",),
}

# The shared scheduler launches these worker classes. Medium and small describe
# work-item requirements that a general worker can satisfy.
SCHEDULABLE_RESOURCE_CLASSES = (GPU_RESOURCE_CLASS, GENERAL_RESOURCE_CLASS)
WORK_ITEM_RESOURCE_CLASSES = tuple(
    dict.fromkeys(item for worker_class in WORKER_COMPATIBILITY.values() for item in worker_class)
)


def compatible_work_item_classes(resource_class: str) -> tuple[str, ...]:
    """Return the work-item classes that a worker class may claim."""
    return WORKER_COMPATIBILITY.get(resource_class, (resource_class,))


def memory_tiers(initial_gb: int, maximum_gb: int) -> tuple[int, ...]:
    """Return the scheduler's doubling sequence through the memory ceiling."""
    tiers = [initial_gb]
    while tiers[-1] < maximum_gb:
        tiers.append(min(maximum_gb, tiers[-1] * 2))
    return tuple(tiers)
