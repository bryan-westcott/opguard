"""Test inversion."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def sdxl_inversion(
    test_image: PILImage | None = None,
    generated_caption: str | None = None,
) -> dict[str, Any]:
    """Test Inversion and Reconstruction objects."""
    from PIL.Image import blend

    from opguard.inversion import InversionSdxl, InversionSdxlReconstruct
    from opguard.nlp import Blip1
    from opguard.tests.util import hstack_numpy, load_test_image

    display_size = (512, 512)
    test_image = test_image or load_test_image()
    if not generated_caption:
        with Blip1() as blip1:
            generated_caption = blip1(input_raw=test_image)
    with InversionSdxl() as inversion:
        inverse_latent, inverse_config, inverse_caption = inversion(
            input_raw=test_image,
            generated_caption=generated_caption,
        )
        logger.debug(inverse_latent.shape)
        logger.debug(inversion.extra_info["sdxl_model_name"])
        return_inverse = {
            "inverse_latent": inverse_latent,
            "inverse_config": inverse_config,
            "inverse_caption": inverse_caption,
        }
    with InversionSdxlReconstruct() as recon:
        reconstructed_image, reconstructed_config = recon(
            input_raw=inverse_latent,
            inverse_config=inverse_config,
            inverse_caption=inverse_caption,
        )
        logger.debug(reconstructed_image.size)
        return_reconstructed = {
            "reconstructed_image": reconstructed_image,
            "reconstructed_config": reconstructed_config,
        }
    blend_image = blend(reconstructed_image.resize(display_size), test_image.resize(display_size), 0.5)
    image_grid = [
        reconstructed_image.resize(display_size),
        test_image.resize(display_size),
        blend_image.resize(display_size),
    ]
    stacked_images = hstack_numpy(image_grid)
    stacked_images.save("inversion-test.png")
    return return_inverse | return_reconstructed | {"test_image": test_image} | {"image_grid": image_grid}


if __name__ == "__main__":
    sdxl_inversion()
