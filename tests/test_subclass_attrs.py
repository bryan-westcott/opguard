"""Step 4 TDD: __init_subclass__ accepts DEFAULT_DEVICE_MAP = None (COS-4).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_subclass_attrs.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import pytest

from opguard.tests.trivial import PassthroughDetector
from opguard.tests.util import load_test_image


def test_default_device_map_none_is_allowed() -> None:
    """A subclass may disable device mapping with DEFAULT_DEVICE_MAP = None."""

    class NoDeviceMapGuard(PassthroughDetector):
        NAME = "test-no-device-map"
        DEFAULT_DEVICE_MAP = None

    guard = NoDeviceMapGuard(device_override="cpu")
    assert guard.device_map is None
    out = guard(input_raw=load_test_image(use_blank=True, final_size=(64, 64)))
    assert out is not None


def test_other_attrs_still_reject_none() -> None:
    """The non-optional class attrs (e.g., NAME) still reject None."""
    with pytest.raises(TypeError, match="non-empty class attr"):

        class NoNameGuard(PassthroughDetector):  # noqa: F841  (defined for the raise)
            NAME = None  # type: ignore[assignment]


def test_other_attrs_still_reject_empty_string() -> None:
    """The non-optional class attrs (e.g., MODEL_ID) still reject empty strings."""
    with pytest.raises(TypeError, match="non-empty class attr"):

        class EmptyModelIdGuard(PassthroughDetector):  # noqa: F841  (defined for the raise)
            MODEL_ID = ""
