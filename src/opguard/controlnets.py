"""ControlNets classes."""

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

from collections.abc import Callable

import torch
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, MarigoldDepthPipeline, MarigoldNormalsPipeline
from diffusers.pipelines.marigold.pipeline_marigold_depth import MarigoldDepthOutput
from diffusers.pipelines.marigold.pipeline_marigold_normals import MarigoldNormalsOutput
from easy_dwpose import DWposeDetector
from PIL.Image import Image as PILImage

from .opguard_base import OpGuardBase


class Tile(OpGuardBase):
    """Tile controlnet."""

    NAME = "tile"
    MODEL_ID = "xinsir/controlnet-tile-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> Callable:
        return lambda image: image.copy()

    def controlnet(self) -> ControlNetModel:
        """Return a loaded Tile controlnet to attach to external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)


class Hed(OpGuardBase):
    """HED softline detector."""

    NAME = "hed"
    MODEL_ID = "lllyasviel/Annotators"
    REVISION = "main"
    # Will fail on bfloat16 due to use of numpy detach
    DTYPE_PREFERENCE = torch.float16

    def _load_detector(self) -> HEDdetector:
        return HEDdetector.from_pretrained(
            self.model_id,
        ).to(self.device)

    def controlnet(self) -> ControlNetModel:
        """Return a loaded HED controlnet to attach to external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            "xinsir/controlnet-tile-sdxl-1.0",
            torch_dtype=self.dtype,
        ).to(self.device)


class MarigoldDepth(OpGuardBase):
    """Marigold depth detector."""

    NAME = "marigold_depth"
    MODEL_ID = "prs-eth/marigold-depth-v1-1"
    REVISION = "main"

    def _load_detector(self) -> MarigoldDepthPipeline:
        return MarigoldDepthPipeline.from_pretrained(
            self.model_id,
            variant=self.variant,
            torch_dtype=self.dtype,
        ).to(self.device)

    def _postprocess(self, output_raw: MarigoldDepthOutput) -> PILImage:
        # Note: must be inverted to match depth convention of zoe, dpt, etc.
        # Note: colormap_binary inverts the 16-bit depth map and converts to 8-bit
        # See: https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage#marigold-for-controlnet
        return self._detector.image_processor.visualize_depth(
            output_raw.prediction,
            color_map="binary",
        )[0]

    def controlnet(self) -> ControlNetModel:
        """Return a loaded Marigold Depth controlnet for external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            "xinsir/controlnet-depth-sdxl-1.0",
            torch_dtype=self.dtype,
        ).to(self.device)


class MarigoldNormals(OpGuardBase):
    """Marigold normals detector."""

    NAME = "marigold_normals"
    MODEL_ID = "prs-eth/marigold-normals-v1-1"
    REVISION = "main"

    def _load_detector(self) -> MarigoldNormalsPipeline:
        return MarigoldNormalsPipeline.from_pretrained(
            self.model_id,
            variant=self.variant,
            torch_dtype=self.dtype,
        ).to(self.device)

    def _postprocess(self, output_raw: MarigoldNormalsOutput) -> PILImage:
        return self._detector.image_processor.visualize_normals(output_raw.prediction)[0]


class Dwpose(OpGuardBase):
    """DWPose detector."""

    NAME = "dwpose"
    MODEL_ID = "RedHash/DWPose"
    REVISION = "main"

    def _load_detector(self) -> DWposeDetector:
        return DWposeDetector(device=self.device)

    def _predict(self, input_proc: PILImage) -> PILImage:
        return self._detector(
            input_proc,
            detect_resolution=input_proc.width,
            output_type="pil",
            include_hands=True,
            include_face=True,
        )
