"""Tests for model guard."""

# module/tests/__init__.py
from __future__ import annotations

# Import all marked tests in test.py
# Note: this allows import from opguard.tests, such as:
# >>> from opguard.tests import smoke
from . import test as _test_mod

__all__: list[str] = []

# Walk attributes of tests.test and re-export anything with pytest marks
for name, obj in vars(_test_mod).items():
    # Only care about callables (functions)
    if not callable(obj):
        continue

    # pytest marks attach a `pytestmark` attribute to the function
    marks = getattr(obj, "pytestmark", None)
    if not marks:
        continue

    # Re-export on this package
    globals()[name] = obj
    __all__ += [name]
