"""Test methods."""

from typing import Any

import PIL
import requests
from diffusers.utils import make_image_grid
from loguru import logger
from PIL.Image import Image as PILImage
from PIL.Image import blend

from .controlnets import Anyline, Dwpose, Hed, MarigoldDepth, MarigoldNormals, Tile
from .inversion import Inversion, InversionReconstruct
from .nlp import Blip1, Blip2

# ruff: noqa: PT028 - defaults are fine, these are light tests


def load_test_image(
    test_image: PILImage | None = None,
    test_image_url: str | None = None,
) -> PILImage:
    """Shared test image loader."""
    if not test_image:
        test_image_url = (
            test_image_url or "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png"
        )
        test_image = PIL.Image.open(requests.get(test_image_url, stream=True, timeout=(3.0, 30.0)).raw)
    return test_image


def test_blip(test_image: PILImage | None = None) -> dict[str, Any]:
    """Test NLP objects."""
    test_image = test_image or load_test_image()
    test_image = load_test_image()
    with Blip2() as blip1:
        caption_blip2 = blip1(input_raw=test_image)
        logger.info(f"blip2 (blip1): {caption_blip2}")
        caption_blip2_conditional = blip1(
            input_raw=test_image,
            text="Question: What color is her hair? Answer:",
        )
        logger.info(f"blip2 (question): {caption_blip2_conditional}")
        return_blip2 = {
            "caption_blip2": caption_blip2,
            "caption_blip2_conditional": caption_blip2_conditional,
        }
    with Blip1() as blip2:
        caption_blip1 = blip2(input_raw=test_image)
        logger.info(f"blip1 (blip2): {caption_blip1}")
        text_blip1 = "Hair Color"
        caption_blip1_conditional = blip2(input_raw=test_image, text=text_blip1)
        logger.info(f'blip1 (text="{text_blip1}"): {caption_blip1_conditional}')
        return_blip1 = {
            "caption_blip1": caption_blip1,
            "caption_blip1_conditional": caption_blip1_conditional,
        }
    return return_blip1 | return_blip2


def test_inversion(
    test_image: PILImage | None = None,
    generated_caption: str | None = None,
) -> dict[str, Any]:
    """Test Inversion and Reconstruction objects."""
    display_size = (512, 512)
    test_image = test_image or load_test_image()
    if not generated_caption:
        with Blip1() as blip1:
            generated_caption = blip1(input_raw=test_image)
    with Inversion() as inversion:
        inverse_latent, inverse_config, inverse_caption = inversion(
            input_raw=test_image,
            generated_caption=generated_caption,
        )
        logger.info(inverse_latent.shape)
        logger.info(inversion.extra_info["sdxl_model_name"])
        return_inverse = {
            "inverse_latent": inverse_latent,
            "inverse_config": inverse_config,
            "inverse_caption": inverse_caption,
        }
    with InversionReconstruct() as recon:
        reconstructed_image, reconstructed_config = recon(
            input_raw=inverse_latent,
            inverse_config=inverse_config,
            inverse_caption=inverse_caption,
        )
        logger.info(reconstructed_image.size)
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
    make_image_grid(image_grid, 1, len(image_grid))
    return return_inverse | return_reconstructed | {"test_image": test_image} | {"image_grid": image_grid}


def test_control(test_image: PILImage | None = None) -> dict[str, Any]:
    """Test ControlNets objects."""
    test_image = test_image or load_test_image()
    display_size = (512, 512)
    control_types = (Tile, Hed, MarigoldDepth, MarigoldNormals, Dwpose, Anyline)
    control_outputs = []
    return_control = {}
    for control_type in control_types:
        with control_type() as control:
            logger.info(control.name, control.model_id)
            control_output = control(input_raw=test_image).resize(display_size)
            control_outputs.append(control_output)
            return_control[control.name] = control_output
    make_image_grid(control_outputs, 1, len(control_outputs))
    return return_control
