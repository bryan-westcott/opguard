"""An example TinyVAE specialization of OpGuardBase."""

# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override

import torch
from diffusers import AutoencoderKL, AutoencoderTiny
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image as PILImage

from .base import OpGuardBase


class AutoencoderBase(OpGuardBase):
    """Tiny VAE and Autoencoder KL base class for both SD and SDXL.

    By passing mode of "encode", "decode" or "encode-decode"
    different aspects (or both aspects) of the autoencoder can
    be exercised

    Note: the encode-decode is more useful for testsing purposes
    """

    def _load_processor(self) -> VaeImageProcessor:
        return VaeImageProcessor(vae_scale_factor=8)

    def _preprocess(self, *, input_raw: PILImage) -> torch.FloatTensor:
        return self._processor.preprocess(input_raw).to(self.device, self.dtype)  # (B,C,H,W)

    def _postprocess(self, *, output_raw: torch.FloatTensor) -> PILImage:
        return self._processor.postprocess(output_raw, output_type="pil")[0]

    def _encode(self, *, image: torch.FloatTensor) -> torch.FloatTensor:
        """Encode image.

        Input:
          input_proc:
            represents: batch (B) of color (3-channel) images
            shape: (B, 3, H, W),
            values ∈ [-1, 1]
        Output:
          output_raw:
            represents: batch (B) of downsampled latents (4-channel)
            shape: (B, 4, H//factor, W//factor),
            values ∈ R (unbounded),
            factor ∈ {8,16}
        """
        # (N, latent_channels, H/8, W/8) for SD1.x TAEs
        return self._detector.encode(image.to(self.device, self.dtype)).latents

    def _decode(self, *, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode image.

        Input:
          output_raw:
            represents: batch (B) of downsampled latents (4-channel)
            shape: (B, 4, H//factor, W//factor),
            values ∈ R (unbounded),
            factor ∈ {8,16}
        Output:
          input_proc:
            represents: batch (B) of color (3-channel) images
            shape: (B, 3, H, W),
            values ∈ [-1, 1]
        """
        # (N, latent_channels, H/8, W/8) for SD1.x TAEs
        return self._detector.decode(latent.to(self.device, self.dtype)).sample

    def _predict(self, *, input_proc: torch.FloatTensor, mode: str) -> torch.FloatTensor:
        """Apply model in encode-only, decod-only, or encode-then-decode mode.

        input_proc: a torch.FloatTensor representing input image
                    (for 'encode' and 'encode-decode' mode)
                    or a torch.FloatTensor representing latent embedding
                    (for 'decode' mode)
        mode: valid modes are 'encode', 'decode', or 'encode-decode'

        Note: output should closely match input on 'encode-decode' mode
        """
        if mode == "encode":
            return self._encode(image=input_proc)
        if mode == "decode":
            return self._decode(latent=input_proc)
        if mode == "encode-decode":
            return self._decode(latent=self._encode(image=input_proc))
        valid_modes = ("encode", "decode", "encode-decode")
        message = f"Prediction mode must be in: {valid_modes}"
        raise ValueError(message)


class AutoencoderTinyBase(AutoencoderBase):
    """AutoencoderTiny VAE base class for both SD and SDXL."""

    DETECTOR_TYPE = AutoencoderTiny


class AutoencoderKLBase(AutoencoderBase):
    """AutoencoderKL VAE base class for both SD and SDXL."""

    DETECTOR_TYPE = AutoencoderKL

    def _encode(self, *, image: torch.FloatTensor) -> torch.FloatTensor:
        return self._detector.encode(image.to(self.device, self.dtype)).latent_dist.sample()


class VaeTinyForSd(AutoencoderTinyBase):
    """Tiny VAE for SD."""

    NAME = "vae-tiny-sd"
    MODEL_ID = "madebyollin/taesd"
    REVISION = "main"
    DEFAULT_DEVICE = "cuda"
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_DEVICE_MAP = "cuda"  # will ignore for cpu


class VaeTinyForSdxl(AutoencoderTinyBase):
    """Tiny VAE for SDXL."""

    NAME = "vae-tiny-sdxl"
    MODEL_ID = "madebyollin/taesdxl"
    REVISION = "main"
    DEFAULT_DEVICE = "cuda"
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_DEVICE_MAP = "cuda"  # will ignore for cpu


class VaeSdxlFp16Fix(AutoencoderKLBase):
    """SDXL fp-16 fixed VAE for SDXL."""

    NAME = "vae-sdxl-fp16-fix"
    MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
    REVISION = "main"
    DEFAULT_DEVICE = "cuda"
    DEFAULT_DTYPE = torch.bfloat16
    DEFAULT_DEVICE_MAP = "cuda"
    FROM_PRETRAINED_SKIP_KWARGS = ("variant",)
