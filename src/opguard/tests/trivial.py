"""Test with trivial passthrough detector."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from collections.abc import Callable
from typing import cast

from opguard.base import DetectorFactory, OpGuardBase


class PassthroughDetector(OpGuardBase):
    """Tile controlnet."""

    NAME = "passthrough-detector"
    MODEL_ID = "xinsir/controlnet-tile-sdxl-1.0"
    REVISION = "main"
    DETECTOR_TYPE = cast("DetectorFactory", lambda *_, **__: ...)  # one-off coercion

    def _load_detector(self) -> Callable:
        return lambda img: img.copy()


def trivial() -> None:
    """Test trivial (no CPU or GPU) pass through detector."""
    from opguard.tests.util import load_test_image

    with PassthroughDetector() as passthrough:
        passthrough(input_raw=load_test_image(use_blank=True, final_size=(256, 256)))


if __name__ == "__main__":
    trivial()
