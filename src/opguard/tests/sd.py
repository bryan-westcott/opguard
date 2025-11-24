"""Stable Diffusion tests."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def sd(*, run_tiny: bool = False) -> PILImage:
    """Run SDXL or SD Tiny."""
    if run_tiny:
        return sd_tiny()
    return sdxl()


def sdxl() -> PILImage:
    """Test SDXL."""
    import numpy as np
    from PIL.Image import Image as PILImage

    from opguard.sd import SdxlTextToImage

    logger.debug("Running SDXL")
    prompt = "An astronaut on a horse on the moon."
    with SdxlTextToImage() as sdxl:
        image = sdxl(input_raw=prompt)
    # check for image output, with some width/height and not all zeros.
    assert isinstance(image, PILImage)
    assert all(dim > 0 for dim in image.size)
    assert len(np.unique(np.array(image).ravel())) > 1
    return image


def sd_tiny() -> PILImage:
    """Test SD 21 Nano 4-Bit Quantized that runs in 2 GiB VRAM.

    Warning: designed for demo tests purposes only!
    """
    import numpy as np
    from PIL.Image import Image as PILImage

    from opguard.sd import SdTinyNanoTextToImage

    logger.debug("Running SD Nano")
    prompt = "An astronaut on a horse on the moon."
    with SdTinyNanoTextToImage() as sd:
        image = sd(input_raw=prompt, height=128, width=128)
    # check for image output, with some width/height and not all zeros.
    assert isinstance(image, PILImage)
    assert all(dim > 0 for dim in image.size)
    assert len(np.unique(np.array(image).ravel())) > 1
    image.save("sd21-test.png")
    return image


if __name__ == "__main__":
    sd()
