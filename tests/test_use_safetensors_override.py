"""Step 5 TDD: _load passes the use_safetensors property to load_guard (COS-5).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_use_safetensors_override.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

# ruff: noqa: SLF001  # tests legitimately access private members

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opguard.tests.trivial import PassthroughDetector
from opguard.tests.util import load_test_image

if TYPE_CHECKING:
    from collections.abc import Callable


class RecordingSafetensorsGuard(PassthroughDetector):
    """Guard with USE_SAFETENSORS=False that records the in-load value.

    cache_guard routes the use_safetensors value received by load_guard
    back onto the instance for the duration of the load, so recording
    the property inside _load_detector observes exactly what _load
    passed down.
    """

    NAME = "test-use-safetensors"
    USE_SAFETENSORS = False

    def __init__(self, **kwargs: Any) -> None:
        self.seen_use_safetensors: bool | None = None
        super().__init__(device_override="cpu", **kwargs)

    def _load_detector(self) -> Callable:
        self.seen_use_safetensors = self.use_safetensors
        return super()._load_detector()


def test_instance_override_reaches_load_guard() -> None:
    """Setting the instance override makes the loader see True, not the class False."""
    guard = RecordingSafetensorsGuard()
    guard.use_safetensors = True
    guard(input_raw=load_test_image(use_blank=True, final_size=(64, 64)))
    assert guard.seen_use_safetensors is True


def test_class_false_without_override_loads_cleanly() -> None:
    """A USE_SAFETENSORS=False guard with no override loads without ValueError.

    Pre-fix, cache_guard's refresh branch routed use_safetensors=False
    back through the property setter, which rejects False.
    """
    guard = RecordingSafetensorsGuard()
    guard(input_raw=load_test_image(use_blank=True, final_size=(64, 64)))
    assert guard.seen_use_safetensors is False
    # the guard's own value is untouched after the load
    assert guard.use_safetensors is False


def test_class_default_true_still_reaches_load_guard() -> None:
    """Without an override, the class default (True) flows through unchanged."""

    class DefaultSafetensorsGuard(RecordingSafetensorsGuard):
        NAME = "test-use-safetensors-default"
        USE_SAFETENSORS = True

    guard = DefaultSafetensorsGuard()
    guard(input_raw=load_test_image(use_blank=True, final_size=(64, 64)))
    assert guard.seen_use_safetensors is True
