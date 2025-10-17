"""Smoke tests for OpGuardBase exercising CPU, GPU, and BFloat modes.

To run, with debugging:     uv run pytest --log-cli-level=DEBUG --capture=no
"""

# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override

import pytest
import torch
from diffusers import AutoencoderTiny
from loguru import logger

from opguard.model_guard_base import OpGuardBase


class Smoke(OpGuardBase):
    """Tiny VAE for SD."""

    NAME = "tiny-vae"
    MODEL_ID = "madebyollin/taesd"
    REVISION = "main"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_DTYPE = torch.float32

    def _load_detector(self, **kwargs: object) -> AutoencoderTiny:
        # ruff: noqa: ARG002
        return AutoencoderTiny.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            revision=self.revision,
        ).to(self.device)

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


def _tiny_vae_roundtrip(*, device: str | torch.device, dtype: torch.dtype, **kwargs) -> None:
    """Tiny round-trip VAE that exercises OpGuard with various device/dtypes."""
    # ruff: noqa: ANN003  (too restrictive on tests)
    batch: int = 2
    logger.info(f"Running smoke test with {device=}, {dtype=}, {kwargs=}")
    with Smoke(device_override=device, dtype_override=dtype, **kwargs) as smoke:
        size = (batch, 3, 512, 512)
        input_tensor: torch.FloatTensor = torch.rand(size=size, device=smoke.device, dtype=smoke.dtype) * 2 - 1
        output_tensor: torch.FloatTensor = smoke(input_raw=input_tensor)

    assert input_tensor.shape == output_tensor.shape


def _tiny_vae_roundtrip_sequence(*, device: str | torch.device, dtype: torch.dtype) -> None:
    """Test tiny VAE round trip, but also exercise local export caching."""
    # run it once, forcing refresh of export and remote variant check
    _tiny_vae_roundtrip(device=device, dtype=dtype, local_hfhub_variant_check_only=False, force_export_refresh=True)
    # then run with defaults
    _tiny_vae_roundtrip(device=device, dtype=dtype)
    # then explicitly force use of local cache export and variant check
    _tiny_vae_roundtrip(device=device, dtype=dtype, local_hfhub_variant_check_only=True, only_load_export=True)


def cpu() -> None:
    """Test CPU mode, full precision."""
    _tiny_vae_roundtrip_sequence(device="cpu", dtype=torch.float32)


def gpu() -> None:
    """Test GPU mode, half-precision."""
    if torch.cuda.is_available():
        _tiny_vae_roundtrip_sequence(device="cuda", dtype=torch.float16)
    else:
        logger.warning("Unable to run GPU tests due to loack of CUDA/GPU")


def bfloat() -> None:
    """Test GPU mode, with bfloat16 dtype.

    Note: will fail if no GPU, and fallback to float16 if insufficient compute capability.
    """
    if torch.cuda.is_available():
        _tiny_vae_roundtrip_sequence(device="cuda", dtype=torch.bfloat16)
    else:
        logger.warning("Unable to run BFLOAT16 tests due to loack of CUDA/GPU")


@pytest.mark.smoke
def smoke() -> None:
    """Simplest run (no CUDA/GPU needed)."""
    _tiny_vae_roundtrip(
        device="cpu",
        dtype=torch.float32,
        local_hfhub_variant_check_only=False,
        force_export_refresh=True,
    )


@pytest.mark.slow
def slow() -> None:
    """Slower tests, checks basic caching and multiple precision/devices."""
    cpu()
    gpu()
    bfloat()
