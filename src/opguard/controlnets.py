"""ControlNets classes."""

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

from collections.abc import Callable
from typing import Any, ClassVar, cast

import torch

# The root __init__.py in controlnet_aux is suppressed in opguard.__int__.py since most detectors not needed
# which causes noisy warnings but also unnecessarily adds computation time
from controlnet_aux.hed import HEDdetector  # Import only from .hed, do not import from base module
from diffusers import ControlNetModel, MarigoldDepthPipeline, MarigoldNormalsPipeline
from diffusers import ControlNetModel as ControlNetModel_Union
from diffusers.pipelines.marigold.pipeline_marigold_depth import MarigoldDepthOutput
from diffusers.pipelines.marigold.pipeline_marigold_normals import MarigoldNormalsOutput
from easy_dwpose import DWposeDetector
from PIL.Image import Image as PILImage

from .base import DetectorFactory, OpGuardBase


class ControlnetBase(OpGuardBase):
    """Abstract class for controlnets."""

    # All controlnets are non-callable, only associated detectors are
    IS_CALLABLE: ClassVar[bool] = False
    DETECTOR_TYPE = ControlNetModel

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
    DETECTOR_TYPE = cast("DetectorFactory", lambda *_, **__: ...)  # one-off coercion

    def _load_detector(self) -> Callable:
        return lambda img: img.copy()


class TileControlnet(ControlnetBase):
    """Tile controlnet."""

    NAME = "tile-controlnet"
    MODEL_ID = "xinsir/controlnet-tile-sdxl-1.0"
    REVISION = "main"


class HedDetector(OpGuardBase):
    """HED softline detector."""

    NAME = "hed-detector"
    MODEL_ID = "lllyasviel/Annotators"
    REVISION = "main"
    # Will fail on bfloat16 due to use of numpy detach
    DETECTOR_TYPE = HEDdetector
    FROM_PRETRAINED_SKIP_KWARGS = ("variant", "use_safetensors", "revision", "torch_dtype", "device", "device_map")
    ACCEPTS_TO_KWARGS = False
    # This model does not accept dtype as a .to() arg or kwarg and it
    # is lightweight enough that it should run in full (32-bit) precision
    SKIP_TO_DTYPE = True  # This model
    DTYPE_PREFERENCE = torch.float32


class MarigoldDepthDetector(OpGuardBase):
    """Marigold depth detector."""

    NAME = "marigold-depth-detector"
    MODEL_ID = "prs-eth/marigold-depth-v1-1"
    REVISION = "main"
    DETECTOR_TYPE = MarigoldDepthPipeline

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
    DETECTOR_TYPE = ControlNetModel_Union


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
    DETECTOR_TYPE = ControlNetModel_Union
    FROM_PRETRAINED_ADDITIONAL_KWARGS: ClassVar[dict[str, Any]] = {"config_name": "config_promax.json"}


class MarigoldNormalsDetector(OpGuardBase):
    """Marigold normals detector."""

    NAME = "marigold-normals-detector"
    MODEL_ID = "prs-eth/marigold-normals-v1-1"
    REVISION = "main"
    DETECTOR_TYPE = MarigoldNormalsPipeline

    def _postprocess(self, output_raw: MarigoldNormalsOutput) -> PILImage:
        return self._detector.image_processor.visualize_normals(output_raw.prediction)[0]


class DwposeDetector(OpGuardBase):
    """DWPose detector."""

    NAME = "dwpose-detector"
    MODEL_ID = "RedHash/DWPose"
    REVISION = "main"
    DETECTOR_TYPE = DWposeDetector

    # Note: only device supported as a kwarg, and no positional model_id
    # This has an atypical signature, so load manually
    def _load_detector(self) -> object:
        return self.DETECTOR_TYPE(device=self.device)

    def _predict(self, input_proc: PILImage) -> PILImage:
        return self._detector(
            input_proc,
            detect_resolution=input_proc.width,
            output_type="pil",
            include_hands=True,
            include_face=True,
        )
