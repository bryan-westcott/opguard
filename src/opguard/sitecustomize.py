"""Customization to fix quirks of certain modules."""

import importlib.machinery
import importlib.util
import sys
import types


def package_remove_init(pkg: str = "controlnet_aux") -> None:
    """Override the __init__.py call on import."""
    # ruff: noqa: BLE001  # this is intended to be broad
    try:
        spec = importlib.util.find_spec(pkg)
        m = types.ModuleType(pkg)
        m.__path__ = list(spec.submodule_search_locations)  # let submodules resolve
        m.__spec__ = importlib.machinery.ModuleSpec(pkg, loader=None, is_package=True)
        m.__spec__.submodule_search_locations = m.__path__
        sys.modules[pkg] = m
    except Exception as e:
        # Never print to stdout; due to uv issues with sitecustomize
        print(f"[sitecustomize] noop failed for {pkg}: {e}", file=sys.__stderr__)


# The controlnet_aux top level __init__.py imports all detectors, many of
# which we do not intend to use and are not installed, which triggers several
# warnings.  The use of package_remove_init avoids the call to __init__.py and
# you can call a submodule directly like:
#   >>> from controlnet_aux.hed import HEDdetector
package_remove_init("controlnet_aux")
