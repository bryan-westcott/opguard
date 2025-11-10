"""Examples of Stable Diffusion."""

from typing import Any, TypeAlias

import torch
from diffusers import (
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from PIL.Image import Image as PILImage
from transformers import CLIPTextModel, CLIPTokenizer

from .base import OpGuardBase
from .vae import VaeSdxlFp16Fix, VaeTinyForSd

StableDiffusionPipelineLike: TypeAlias = StableDiffusionPipeline | StableDiffusionXLPipeline
StableDiffusionPipelineOutputLike: TypeAlias = StableDiffusionPipelineOutput | StableDiffusionXLPipelineOutput

# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override


class StableDiffusionBase(OpGuardBase):
    """Abstract class for Stable Diffusion (variants including 2.1 and XL)."""

    def _predict(
        self,
        *,
        prompt: str,
        **kwargs: object,
    ) -> StableDiffusionPipelineOutputLike:
        return self._detector(prompt=prompt, **kwargs)

    def _postprocess(
        self,
        output_raw: StableDiffusionPipelineOutputLike,
    ) -> PILImage:
        # Simple postproc on the raw output, but return the config also
        return output_raw.images[0]

    def _caller(
        self,
        *,
        input_raw: str,
        **kwargs: object,
    ) -> PILImage:
        output_raw = self._predict(prompt=input_raw, **kwargs)
        return self._postprocess(output_raw=output_raw)  # outupt_proc

    def _load_detector(self, **kwargs: dict[str, Any]) -> StableDiffusionPipelineLike:
        pipe: StableDiffusionPipelineLike = super()._load_detector(**kwargs)
        return pipe


class SdTinyNanoTextToImage(StableDiffusionBase):
    """A 4-bit quantized SD 2.1 Nano that uses around 2 GiB of VRAM.

    Note: primarily designed for demonstration and test purposes.
    """

    NAME = "sd-nano"
    MODEL_ID = "bguisard/stable-diffusion-nano-2-1"
    REVISION = "main"
    DETECTOR_TYPE = StableDiffusionPipeline
    DEFAULT_DEVICE_MAP = "cuda"
    DEFAULT_QUANT_TYPE = "nf4"
    DEFAULT_QUANT_USE_DOUBLE = True
    SKIP_TO_DEVICE = True
    SKIP_TO_DTYPE = True
    FROM_PRETRAINED_SKIP_KWARGS = ("quantization_config",)  # use individual components

    def _load_detector(self) -> object:
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            quantization_config=self.quant_config,
            device_map="balanced",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            quantization_config=self.quant_config,
            device_map="balanced",
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
        )
        scheduler = DDPMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
        )
        # Load SD base pipeline
        # Note: individual components must be quantized,
        #       as it doesn't accept quantization_config argument
        pipe = super()._load_detector(
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        # Load TinyVAE using OpGuard wrapped version
        # Note: it may have different variant
        #       it also has its own model_id that it tracks
        # Note: have vae match the device/dtype/device_map of the pipeline since
        #       by design VaeTinyForSd defaults to cpu
        with VaeTinyForSd(
            device_override=self.device,
            dtype_override=self.dtype,
            device_map_override=self.device_map,
        ) as vae:
            pipe.vae = vae.detector
        return pipe


class SdxlTextToImage(StableDiffusionBase):
    """Stable diffusion XL with fixed VAE."""

    NAME = "sdxl"
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "main"
    DEFAULT_DEVICE = "cuda"
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_DEVICE_MAP = "cuda"
    DETECTOR_TYPE = StableDiffusionXLPipeline
    SKIP_TO_DEVICE = True
    SKIP_TO_DTYPE = True

    def _load_detector(self) -> object:
        # Note: OpGuard will still free references on exit
        pipe = super()._load_detector()
        with VaeSdxlFp16Fix(
            device_override=self.device,
            dtype_override=self.dtype,
            device_map_override=self.device_map,
        ) as vae:
            pipe.vae = vae.detector
        return pipe
