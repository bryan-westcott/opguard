"""Utilities for opguard Inference."""

from . import base as _base
from . import util as _util
from .sitecustomize import package_remove_init

# avoid calling __init__.py in controlnet_aux as it loads
# all detectors, even those uninstalled or with compatibility issues
# Note: this should be called automatically since it is in sitecustomize.py
#       but edge cases in uv can prevent this, so call it here manually
package_remove_init("controlnet_aux")

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
