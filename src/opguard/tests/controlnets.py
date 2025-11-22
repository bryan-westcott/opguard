"""Controlnets tests."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def control(test_image: PILImage | None = None) -> dict[str, Any]:
    """Test ControlNets objects."""
    import torch

    from opguard.controlnets import (
        DepthControlnet,
        DwposeDetector,
        HedDetector,
        MarigoldDepthDetector,
        MarigoldNormalsDetector,
        TileControlnet,
        TileDetector,
        UnionControlnet,
        UnionPromaxControlnet,
    )
    from opguard.tests.util import load_test_image

    test_image = test_image or load_test_image()
    display_size = (512, 512)
    control_types = (
        TileDetector,
        TileControlnet,
        HedDetector,
        MarigoldDepthDetector,
        DepthControlnet,
        UnionControlnet,
        UnionPromaxControlnet,
        MarigoldNormalsDetector,
        DwposeDetector,
    )
    control_outputs = []
    return_control = {}
    for control_type in control_types:
        logger.info(f"Running controlnet test: {control_type}")
        try:
            if control_type in (MarigoldDepthDetector, MarigoldNormalsDetector):
                logger.warning("Skipping marigold control tests due lack of CUDA/GPU")
                continue
            with control_type() as control:
                control_output = None
                if control.IS_CALLABLE:
                    control_output = control(input_raw=test_image).resize(display_size)
                control_outputs.append(control_output)
                return_control[control.NAME] = control_output
        except torch.OutOfMemoryError:
            logger.error(f"OUT OF MEMORY FOR {control_type}, skipping")
    return return_control


if __name__ == "__main__":
    control()
