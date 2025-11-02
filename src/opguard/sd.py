"""Examples of Stable Diffusion."""

import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
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

from .opguard_base import OpGuardBase
from .vae import SdxlVaeFp16Fix, TinyVaeForSd

# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override


class StableDiffusionBase(OpGuardBase):
    """Abstract class for Stable Diffusion (variants including 2.1 and XL)."""

    def _predict(
        self,
        *,
        prompt: str,
        **kwargs: object,
    ) -> StableDiffusionXLPipelineOutput | StableDiffusionPipelineOutput:
        return self._detector(prompt=prompt, **kwargs)

    def _postprocess(
        self,
        output_raw: StableDiffusionXLPipelineOutput | StableDiffusionPipelineOutput,
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

    def _load_detector(self) -> StableDiffusionXLPipeline | StableDiffusionPipeline:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            variant=self.variant,
            dtype=self.dtype,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        # Final device/dtype placement, with to() applied after VRAM savers
        return pipe.to(self.device)


class SdTinyNanoTextToImage(StableDiffusionBase):
    """A 4-bit quantized SD 2.1 Nano that uses around 2 GiB of VRAM.

    Note: primarily designed for demonstration and test purposes.
    """

    NAME = "sd-nano"
    MODEL_ID = "bguisard/stable-diffusion-nano-2-1"
    REVISION = "main"

    def _load_detector(self) -> StableDiffusionPipeline:
        # Swap in TinyVAE for SD1.x
        quant_config = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Enables double quantization
            bnb_4bit_compute_dtype=torch.float16,
        )

        unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            quantization_config=quant_config,
            device_map="balanced",
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            quantization_config=quant_config,
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
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        # Load TinyVAE using OpGuard wrapped version
        # Note: it may have different variant
        #       it also has its own model_id that it tracks
        with TinyVaeForSd() as vae:
            pipe.vae = vae.detector

        # Memory and performance tweaks
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        # Final device/dtype placement, with to() applied after VRAM savers
        return pipe.to(self.device)


class SdxlTextToImage(StableDiffusionBase):
    """Stable diffusion XL with fixed VAE."""

    NAME = "sdxl"
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "main"

    def _load_detector(self) -> StableDiffusionXLPipeline:
        with SdxlVaeFp16Fix() as vae:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                revision=self.REVISION,
                variant=self.variant,
                dtype=self.dtype,
                vae=vae.detector,
            )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        # Final device/dtype placement, with to() applied after VRAM savers
        return pipe.to(self.device)
