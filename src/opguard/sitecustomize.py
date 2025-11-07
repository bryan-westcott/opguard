"""Customization to fix quirks of certain modules."""

import importlib.machinery
import importlib.util
import sys
import types
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable


def package_remove_init(pkg: str = "controlnet_aux") -> None:
    """Override the __init__.py call on import."""
    # ruff: noqa: BLE001  # this is intended to be broad
    try:
        spec = importlib.util.find_spec(pkg)
        # Guard bot# Guard both None cases so mypy knows we're safe below
        if spec is None or spec.submodule_search_locations is None:
            return

        # submodule_search_locations is a NamespacePath (Iterable[str]) at runtime,
        # but typed as Optional[...] → cast for mypy
        search_locs = list(cast("Iterable[str]", spec.submodule_search_locations))

        m = types.ModuleType(pkg)
        m.__path__ = search_locs  # list[str] is fine for __path__
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
