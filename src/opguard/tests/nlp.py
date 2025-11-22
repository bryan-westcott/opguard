"""NLP Tests."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def blip(
    *,
    test_image: PILImage | None = None,
    test_blip1: bool = True,
    test_blip2_4bit: bool = True,
    test_blip2_16bit: bool = True,
) -> dict[str, Any]:
    """Test NLP objects."""
    from opguard.tests.util import load_test_image

    test_image = test_image or load_test_image()
    return_blip1 = {}
    return_blip2 = {}
    if test_blip2_16bit:
        from opguard.nlp import Blip2_16Bit

        with Blip2_16Bit() as blip2:
            caption_blip2 = blip2(input_raw=test_image)
            logger.debug(f"blip2: {caption_blip2}")
            caption_blip2_conditional = blip2(
                input_raw=test_image,
                text="Question: What color is her hair? Answer:",
            )
            logger.debug(f"blip2 (question): {caption_blip2_conditional}")
            return_blip2 = {
                "caption_blip2": caption_blip2,
                "caption_blip2_conditional": caption_blip2_conditional,
            }
    if test_blip2_4bit:
        from opguard.nlp import Blip2_4Bit

        with Blip2_4Bit() as blip2:
            caption_blip2 = blip2(input_raw=test_image)
            logger.debug(f"blip2: {caption_blip2}")
            caption_blip2_conditional = blip2(
                input_raw=test_image,
                text="Question: What color is her hair? Answer:",
            )
            logger.debug(f"blip2 (question): {caption_blip2_conditional}")
            return_blip2 = {
                "caption_blip2": caption_blip2,
                "caption_blip2_conditional": caption_blip2_conditional,
            }
    if test_blip1:
        from opguard.nlp import Blip1

        with Blip1() as blip1:
            caption_blip1 = blip1(input_raw=test_image)
            logger.debug(f"blip1 (blip1): {caption_blip1}")
            text_blip1 = "Hair Color"
            caption_blip1_conditional = blip1(input_raw=test_image, text=text_blip1)
            logger.debug(f'blip1 (text="{text_blip1}"): {caption_blip1_conditional}')
            return_blip1 = {
                "caption_blip1": caption_blip1,
                "caption_blip1_conditional": caption_blip1_conditional,
            }
    return return_blip1 | return_blip2


if __name__ == "__main__":
    blip()
