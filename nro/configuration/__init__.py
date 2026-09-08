"""Workflow configuration resolution."""

__all__ = [
    "ConfigStore",
    "DERIVATIVE_CLASSES",
    "ResolvedConfiguration",
    "ResolvedWorkflow",
    "WorkflowError",
    "fingerprint",
]


def __getattr__(name):
    # Path setup must be importable before scientific configurations resolve.
    if name not in __all__:
        raise AttributeError(name)
    from nro.configuration import store
    return getattr(store, name)
