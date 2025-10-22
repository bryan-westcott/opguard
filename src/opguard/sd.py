"""Examples of Stable Diffusion."""

import torch
from diffusers import (
    AutoencoderTiny,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from PIL.Image import Image as PILImage
from transformers import CLIPTextModel, CLIPTokenizer

from .opguard_base import OpGuardBase
from .vae import TinyVaeForSd


class SdTinyNanoTextToImage(OpGuardBase):
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
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )
        # Load TinyVAE using OpGuard wrapped version
        # Note: it may have different variant
        #       it also has its own model_id that it tracks
        with TinyVaeForSd(keep_warm=True) as vae:
            pipe.vae = vae.detector

        # Memory and performance tweaks
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        # Final device/dtype placement, with to() applied after VRAM savers
        return pipe.to(self.device)

    def _predict(self, *, prompt: str, **kwargs: object) -> StableDiffusionPipelineOutput:
        # You can pass height/width (multiples of 8; 512x512 default for SD1.5)
        return self._detector(prompt=prompt, **kwargs)

    def _postprocess(self, output_raw: StableDiffusionPipelineOutput) -> PILImage:
        return output_raw.images[0]

    def _caller(self, *, prompt: str, **kwargs: object) -> PILImage:
        out = self._predict(prompt=prompt, **kwargs)
        return self._postprocess(out)


class SdxlTextToImage(OpGuardBase):
    """Stable diffusion XL with fixed VAE."""

    NAME = "sdxl"
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "main"

    def _load_detector(self) -> StableDiffusionXLPipeline:
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
        quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            variant=self.variant,
            dtype=self.dtype,
            vae=vae,
            quantization_config=quant_config,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

        # Final device/dtype placement, with to() applied after VRAM savers
        return pipe.to(self.device)

    def _predict(self, *, prompt: str, **kwargs: object) -> StableDiffusionXLPipelineOutput:
        return self._detector(prompt=prompt, **kwargs)

    def _postprocess(
        self,
        output_raw: StableDiffusionXLPipelineOutput,
    ) -> PILImage:
        # Simple postproc on the raw output, but return the config also
        return output_raw.images[0]

    def _caller(
        self,
        *,
        prompt: str,
        **kwargs: object,
    ) -> PILImage:
        output_raw = self._predict(prompt=prompt, **kwargs)
        return self._postprocess(output_raw=output_raw)  # outupt_proc
