"""ControlNets classes."""

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

from collections.abc import Callable
from typing import Final

import torch
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, MarigoldDepthPipeline, MarigoldNormalsPipeline
from diffusers import ControlNetModel as ControlNetModel_Union
from diffusers.pipelines.marigold.pipeline_marigold_depth import MarigoldDepthOutput
from diffusers.pipelines.marigold.pipeline_marigold_normals import MarigoldNormalsOutput
from easy_dwpose import DWposeDetector
from PIL.Image import Image as PILImage

from .opguard_base import OpGuardBase


class ControlnetBase(OpGuardBase):
    """Abstract class for controlnets."""

    CALLABLE: Final[bool] = False

    # Stub for callers
    def _caller(self, *args: object, **kwargs: object) -> None:
        # ruff: noqa: ARG002  (this is to chatch accidental calls)
        message = "Controlnets are not meant to be called, but rather attached to a pipe"
        raise RuntimeError(message)


class TileDetector(OpGuardBase):
    """Tile controlnet."""

    NAME = "tile-detector"
    MODEL_ID = "xinsir/controlnet-tile-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> Callable:
        return lambda image: image.copy()

    def controlnet(self) -> ControlNetModel:
        """Return a loaded Tile controlnet to attach to external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            torch_dtype=self.dtype,
        ).to(self.device)


class TileControlnet(ControlnetBase):
    """Tile controlnet."""

    NAME = "tile-controlnet"
    MODEL_ID = "xinsir/controlnet-tile-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> ControlNetModel:
        """Return a loaded Tile controlnet to attach to external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            torch_dtype=self.dtype,
        ).to(self.device)


class HedDetector(OpGuardBase):
    """HED softline detector."""

    NAME = "hed-detector"
    MODEL_ID = "lllyasviel/Annotators"
    REVISION = "main"
    # Will fail on bfloat16 due to use of numpy detach
    DTYPE_PREFERENCE = torch.float16

    def _load_detector(self) -> HEDdetector:
        return HEDdetector.from_pretrained(
            self.model_id,
        ).to(self.device)


class MarigoldDepthDetector(OpGuardBase):
    """Marigold depth detector."""

    NAME = "marigold-depth-detector"
    MODEL_ID = "prs-eth/marigold-depth-v1-1"
    REVISION = "main"

    def _load_detector(self) -> MarigoldDepthPipeline:
        return MarigoldDepthPipeline.from_pretrained(
            self.model_id,
            revision=self.REVISION,
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


class DepthControlnet(ControlnetBase):
    """Depth controlnet."""

    NAME = "marigold-depth-controlnet"
    MODEL_ID = "xinsir/controlnet-depth-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> ControlNetModel:
        """Return a loaded Marigold Depth controlnet for external diffusion pipeline."""
        return ControlNetModel.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            torch_dtype=self.dtype,
        ).to(self.device)


class UnionControlnet(ControlnetBase):
    """Union controlnet for various control methods.

    See: https://github.com/xinsir6/ControlNetPlus/tree/main

    Supports:
        Openpose (now DWPose), Depth, Canny, Lineart, AnimeLineart, Mlsd, Scribble,
        Hed, Pidi (Softedge), Teed, Segment, Normal
    """

    NAME = "marigold-depth-controlnet"
    MODEL_ID = "xinsir/controlnet-union-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> ControlNetModel:
        """Return a loaded Marigold Depth controlnet for external diffusion pipeline."""
        return ControlNetModel_Union.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            torch_dtype=self.dtype,
        ).to(self.device)


class UnionPromaxControlnet(ControlnetBase):
    """Union Promax controlnet for various control methods.

    See: https://github.com/xinsir6/ControlNetPlus/tree/main

    Supports:
        Same as Union, and
        Inpatining, Outpainting, Tile Superresolution, Tile Variation, Tile Deblur,
    """

    NAME = "marigold-depth-controlnet"
    MODEL_ID = "xinsir/controlnet-union-sdxl-1.0"
    REVISION = "main"

    def _load_detector(self) -> ControlNetModel:
        """Return a loaded Marigold Depth controlnet for external diffusion pipeline."""
        return ControlNetModel_Union.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            torch_dtype=self.dtype,
            config_name="config_promax.json",
        ).to(self.device)


class MarigoldNormalsDetector(OpGuardBase):
    """Marigold normals detector."""

    NAME = "marigold-normals-detector"
    MODEL_ID = "prs-eth/marigold-normals-v1-1"
    REVISION = "main"

    def _load_detector(self) -> MarigoldNormalsPipeline:
        return MarigoldNormalsPipeline.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            variant=self.variant,
            torch_dtype=self.dtype,
        ).to(self.device)

    def _postprocess(self, output_raw: MarigoldNormalsOutput) -> PILImage:
        return self._detector.image_processor.visualize_normals(output_raw.prediction)[0]


class DwposeDetector(OpGuardBase):
    """DWPose detector."""

    NAME = "dwpose-detector"
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
