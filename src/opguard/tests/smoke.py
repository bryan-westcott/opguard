"""Smoke tests for ModelGuard exercising CPU, GPU, and BFloat modes."""

# We do not care about LSP substitutability, ModelGuardBase is not used directly
# mypy: disable-error-code=override

# ruff: noqa: PLR0913  (intentionally explicit about all loader options)

import torch
from diffusers import AutoencoderTiny
from loguru import logger

from opguard import ModelGuardBase


class Smoke(ModelGuardBase):
    """Tiny VAE for SD."""

    NAME = "tiny-vae"
    MODEL_ID = "madebyollin/taesd"
    REVISION = "main"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_DTYPE = torch.float32

    def _load_detector(
        self,
        *,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
        local_files_only: bool,
        revision: str,
        variant: str,
    ) -> AutoencoderTiny:
        _ = variant  # accepted but unused on purpose
        return AutoencoderTiny.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device)

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

    def _predict(self, *, input_proc: torch.FloatTensor) -> torch.FloatTensor:
        """Encode-then-decode round trip.

        Note: output should closely match input
        Note: two subsequent calls (encode-decode) is atypical for this library,
              typically the _preprocess would do the input.to(device, dtype)
        """
        return self._decode(latent=self._encode(image=input_proc))


def _tiny_vae_roundtrip(device: str, dtype: torch.device) -> None:
    """Tiny round-trip VAE that exercises ModelGuardBase with various device/dtypes."""
    batch: int = 2
    with Smoke(device_override=device, dtype_override=dtype) as smoke:
        size = (batch, 3, 512, 512)
        input_tensor: torch.FloatTensor = torch.rand(size=size, device=smoke.device, dtype=smoke.dtype) * 2 - 1
        logger.info(f"{smoke.device=}, {smoke.dtype=}, {input_tensor.dtype}")
        output_tensor: torch.FloatTensor = smoke(input_raw=input_tensor)

    assert input_tensor.shape == output_tensor.shape


def cpu() -> None:
    """Test CPU mode, full precision."""
    _tiny_vae_roundtrip(device="cpu", dtype=torch.float32)


def gpu() -> None:
    """Test GPU mode, half-precision."""
    _tiny_vae_roundtrip(device="cuda", dtype=torch.float16)


def bfloat() -> None:
    """Test GPU mode, with bfloat16 dtype.

    Note: will fail if no GPU, and fallback to float16 if insufficient compute capability.
    """
    _tiny_vae_roundtrip(device="cuda", dtype=torch.bfloat16)
