"""Step 1 TDD: lazy-load gate loads on first use (COS-1).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_lifecycle_lazy_load.py \
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
    from PIL.Image import Image as PILImage


class CountingPassthrough(PassthroughDetector):
    """Passthrough guard that counts _load_detector calls, pinned to CPU."""

    NAME = "test-lifecycle-passthrough"

    def __init__(self, **kwargs: Any) -> None:
        # set before super().__init__ since keep_warm loads during init
        self.load_count = 0
        super().__init__(device_override="cpu", **kwargs)

    def _load_detector(self) -> Callable:
        self.load_count += 1
        return super()._load_detector()


def _blank_image() -> PILImage:
    return load_test_image(use_blank=True, final_size=(64, 64))


def test_first_bare_call_succeeds() -> None:
    """A fresh instance (keep_warm=False, no context) loads on the FIRST call."""
    guard = CountingPassthrough()
    out = guard(input_raw=_blank_image())
    assert out is not None
    assert guard.load_count == 1


def test_first_detector_access_not_none() -> None:
    """A fresh instance returns a real detector on first .detector access."""
    guard = CountingPassthrough()
    assert guard.detector is not None
    assert guard.load_count == 1


def test_keep_warm_not_freed_between_calls() -> None:
    """A keep_warm instance loads once and is not freed between calls."""
    guard = CountingPassthrough(keep_warm=True)
    assert guard.load_count == 1
    guard(input_raw=_blank_image())
    assert guard._detector is not None
    guard(input_raw=_blank_image())
    assert guard._detector is not None
    assert guard.load_count == 1


def test_context_manager_no_double_load() -> None:
    """Context-manager enter followed by a call must not double-load."""
    with CountingPassthrough() as guard:
        guard(input_raw=_blank_image())
        assert guard.load_count == 1
    assert guard._is_freed is True
    assert guard._detector is None


def test_reload_after_free() -> None:
    """An explicitly freed instance reloads on the next call."""
    guard = CountingPassthrough(keep_warm=True)
    guard._free(reason="test reload-after-free")
    assert guard._detector is None
    out = guard(input_raw=_blank_image())
    assert out is not None
    assert guard.load_count == 2
