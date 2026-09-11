from __future__ import annotations

import functools
import importlib
import sys
import types
from typing import Any


@functools.cache
def _configure_litelm_defaults(litelm: types.ModuleType) -> None:
    """Apply DSPy's global LiteLM defaults once when LiteLM is first imported."""
    litelm.telemetry = False
    litelm.cache = None  # By default we disable LiteLM cache and use DSPy on-disk cache.
    if not getattr(litelm, "_dspy_logging_configured", False):
        litelm.suppress_debug_info = True
        litelm._dspy_logging_configured = True


def _materialize_litelm(litelm: types.ModuleType) -> None:
    """Force LiteLM's lazy module to execute, or raise the missing dependency error."""
    # `require()` returns either an importlib LazyLoader-backed module or a _MissingModule.
    # Accessing a real LiteLM attribute forces LazyLoader execution; on _MissingModule it raises
    # the helpful install-hint ImportError immediately at the DSPy call site.
    _completion = litelm.completion


@functools.cache
def get_litelm(*, feature: str) -> Any:
    """Import LiteLM, apply DSPy's defaults once, and return the module."""
    litelm = importlib.import_module("litelm")
    _materialize_litelm(litelm)
    _configure_litelm_defaults(litelm)
    return litelm


def is_litelm_context_window_error(error: Exception) -> bool:
    """Return whether an exception is LiteLM's context-window error, if LiteLM is loaded."""
    litelm_module = sys.modules.get("litelm")
    context_window_error = getattr(litelm_module, "ContextWindowExceededError", None)
    return context_window_error is not None and isinstance(error, context_window_error)
