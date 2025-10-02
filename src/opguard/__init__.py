"""Utilities for opguard Inference."""

from . import model_guard_base as _base
from . import model_guard_util as _util

# Gather all public names from both modules
_public_names: list[str] = []
for _mod in (_util, _base):
    names = getattr(_mod, "__all__", None)
    if names is None:
        names = [n for n in dir(_mod) if not n.startswith("_")]
    # Update globals and collect
    globals().update({n: getattr(_mod, n) for n in names})
    _public_names.extend(names)

# Deduplicate (preserve order)
__all__ = list(dict.fromkeys(_public_names))
