"""Examples of Stable Diffusion."""

from typing import Any, ClassVar, TypeAlias

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import (
    PipelineQuantizationConfig,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from PIL.Image import Image as PILImage

from opguard.base import OpGuardBase
from opguard.vae import VaeSdxlFp16Fix, VaeTinyForSd

StableDiffusionPipelineLike: TypeAlias = StableDiffusionPipeline | StableDiffusionXLPipeline
StableDiffusionPipelineOutputLike: TypeAlias = StableDiffusionPipelineOutput | StableDiffusionXLPipelineOutput


# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override


def make_pipeline_quant_config(bnb_cfg: DiffusersBitsAndBytesConfig) -> PipelineQuantizationConfig:
    """Convert diffusers object BitsAndBytesConfig to PipelineQuantizationConfig."""
    backend = "bitsandbytes_4bit" if bnb_cfg.load_in_4bit else "bitsandbytes_8bit"
    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs={
            "load_in_4bit": bnb_cfg.load_in_4bit,
            "load_in_8bit": bnb_cfg.load_in_8bit,
            "bnb_4bit_quant_type": bnb_cfg.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": bnb_cfg.bnb_4bit_compute_dtype or torch.bfloat16,
            "bnb_4bit_use_double_quant": bnb_cfg.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_storage": bnb_cfg.bnb_4bit_quant_storage,
        },
        # For SDXL detectors, usually these are the heavy bits:
        components_to_quantize=["unet", "text_encoder", "text_encoder_2"],
    )


class StableDiffusionBase(OpGuardBase):
    """Abstract class for Stable Diffusion (variants including 2.1 and XL)."""

    # Placeholder for VAE override (e.g., madebyollin/vae)
    # Note: leave as None for default VAE
    VAE_TYPE: ClassVar[type[OpGuardBase] | None] = None

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
        if self.VAE_TYPE is not None:
            with self.VAE_TYPE(
                device_override=self.device,
                dtype_override=self.dtype,
                device_map_override=self.device_map,
            ) as vae:
                pipe.vae = vae.detector
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
    VAE_TYPE = VaeTinyForSd


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
    VAE_TYPE = VaeSdxlFp16Fix
