"""Inversion and reconstruction classes."""

# ruff: noqa: RET504  (Explicit is better here)

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

from types import SimpleNamespace
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from instantstyle_plus.config import RunConfig
from instantstyle_plus.eunms import Model_Type, Scheduler_Type
from instantstyle_plus.inversion import run as isp_inversion
from instantstyle_plus.pipes.sdxl_inversion_pipeline import SDXLDDIMPipeline
from instantstyle_plus.utils.enums_utils import get_pipes
from PIL.Image import Image as PILImage
from torch import Tensor

from .base import OpGuardBase
from .nlp import Blip1
from .vae import VaeSdxlFp16Fix


class InversionSdxl(OpGuardBase):
    """InstantStyle-Plus inversion predictor of latents."""

    # Note: not a detector strictly speaking, but produces an estiamte of the latent inversion
    NAME = "isp-inversion"
    # Note: it is an inversion pipeline on top of the
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "main"
    NEED_GRADS = True
    DETECTOR_TYPE = SDXLDDIMPipeline

    def _load_detector(self) -> object:
        # For tracking what is loaded"
        self.extra_info["sdxl_model_name"] = self.MODEL_ID
        self.extra_info["model_type"] = Model_Type.SDXL
        self.extra_info["scheduler_type"] = Scheduler_Type.DDIM
        # Load pipeline
        pipe_inversion, _ = get_pipes(
            model_name=self.extra_info["sdxl_model_name"],
            model_type=self.extra_info["model_type"],
            scheduler_type=self.extra_info["scheduler_type"],
            device=self.device,
        )
        pipe_inversion.enable_xformers_memory_efficient_attention()
        pipe_inversion.enable_attention_slicing()
        return pipe_inversion

    def _preprocess(
        self,
        *,
        input_raw: PILImage,
        config_override: dict[str, Any] | None,
    ) -> tuple[PILImage, RunConfig]:
        # These are the default *inputs* to RunConfig, used in _predict
        inversion_config_input_default = {
            "num_inference_steps": 50,
            "num_inversion_steps": 50,
            "num_renoise_steps": 1,
            "perform_noise_correction": False,
            "seed": 0,
            "model_type": self.extra_info["model_type"],
            "scheduler_type": self.extra_info["scheduler_type"],
        }
        # Override them here if you want all runs to have the same config
        # and/or override them in _predict for per-run config
        # Not: actual config will be self.inv_config set after _predict() call
        inversion_config_input = inversion_config_input_default | (config_override or {})
        inversion_config = RunConfig(**inversion_config_input)
        # Note: input image is not changed, just config
        return input_raw, inversion_config

    def _predict(
        self,
        *,
        input_proc: tuple[PILImage, RunConfig],
        generated_caption: str | None,
    ) -> tuple[Tensor, RunConfig, str]:
        init_image, inversion_config = input_proc

        # Generate caption if needed
        if not generated_caption:
            with Blip1() as blip1:
                generated_caption = blip1(input_raw=init_image)
        # Note: caption should be generated from image, not changed by user!
        self.caption_used = generated_caption

        # library expects pipe_inference even if do_reconstruction=False
        # so dummy up a "dot settable" namespace
        pipe_inference_dummy = SimpleNamespace()

        with torch.enable_grad():  # re-enable just here
            _, inverse_latent, _, _s = isp_inversion(
                init_image=init_image,
                prompt=generated_caption,
                cfg=inversion_config,
                pipe_inversion=self._detector,
                pipe_inference=pipe_inference_dummy,
                do_reconstruction=False,
            )  # torch.Size([1, 4, 128, 128])
        # Note: returns a latent-space tensor, not an image
        return inverse_latent, inversion_config, generated_caption

    def _caller(
        self,
        *,
        input_raw: PILImage,
        generated_caption: str | None,
        config_override: dict[str, Any] | None = None,
    ) -> tuple[Tensor, RunConfig, str]:
        input_proc = self._preprocess(input_raw=input_raw, config_override=config_override)
        output_raw = self._predict(
            input_proc=input_proc,
            generated_caption=generated_caption,
        )
        return self._postprocess(output_raw=output_raw)  # output_proc


class InversionSdxlReconstruct(OpGuardBase):
    """Reconstruction pipeline for Inversion object.

    Note: not intended for prompt or param variation.
    """

    NAME = "inversion-reconstruct"
    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "main"
    DETECTOR_TYPE = StableDiffusionXLPipeline

    def _load_detector(self) -> object:
        vae = VaeSdxlFp16Fix()
        ddim_scheduler = DDIMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
        )
        pipe = self.DETECTOR_TYPE.from_pretrained(
            self.model_id,
            vae=vae,
            variant=self.variant,
            scheduler=ddim_scheduler,
        ).to(self.device, self.dtype)
        # Allows it to fit on a 24GB GPU
        # Note: final vae and upsample will cause OOM on inference only
        #       inversion outputs latents so it fits as is
        # Note: for more extreme cases, also use enable_cpu_offload
        pipe.enable_xformers_memory_efficient_attention()
        return pipe

    def _preprocess(self, *, input_raw: Tensor, inverse_config: RunConfig, inverse_caption: str) -> dict[str, Any]:
        # No image processing, but read inversion config
        # Match exactly what was used for inverse
        inverse_latent = input_raw.to(self.device, self.dtype)
        # arguments to sdxl
        diffusion_args = {
            "num_inference_steps": inverse_config.num_inversion_steps,
            "seed": 0,
            "latents": inverse_latent,  # from Inversion
            "strength": 0,  # 0 means no noise added
            "denoising_start": 0.001,
            "guidance_scale": 0,  # high cfg increase style, 0 matches latent
            "negative_prompt": None,
            "prompt": inverse_caption,
        }
        return diffusion_args

    def _predict(self, *, input_proc: dict[str, Any]) -> tuple[StableDiffusionXLPipelineOutput, dict[str, Any]]:
        diffusion_args = input_proc
        diffusion_output_raw = self._detector(**diffusion_args)
        return diffusion_output_raw, diffusion_args

    def _postprocess(
        self,
        output_raw: tuple[StableDiffusionXLPipelineOutput, dict[str, Any]],
    ) -> tuple[PILImage, dict[str, Any]]:
        # Simple postproc on the raw output, but return the config also
        diffusion_output_raw, diffusion_args = output_raw
        reconstructed_image = diffusion_output_raw.images[0]
        return reconstructed_image, diffusion_args

    def _caller(
        self,
        *,
        input_raw: PILImage,
        inverse_config: RunConfig,
        inverse_caption: str,
    ) -> tuple[PILImage, dict[str, Any]]:
        inverse_latent = input_raw
        diffusion_args = self._preprocess(
            input_raw=inverse_latent,
            inverse_config=inverse_config,
            inverse_caption=inverse_caption,
        )
        output_raw = self._predict(input_proc=diffusion_args)
        return self._postprocess(output_raw=output_raw)  # outupt_proc
