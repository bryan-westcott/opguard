"""Smoke tests for OpGuardBase exercising CPU, GPU, and BFloat modes.

To run, with debugging:     uv run pytest --log-cli-level=DEBUG --capture=no
"""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling


# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

if TYPE_CHECKING:
    import torch

    TorchDeviceLike = str | torch.device
    DTypeLike = torch.dtype
else:
    # At runtime, we don't actually care about the precise type;
    # this keeps annotations happy and avoids importing torch.
    TorchDeviceLike = str | Any
    DTypeLike = Any


@pytest.mark.trivial
def trivial() -> None:
    """Simplest run without inference (no CPU/GPU needed)."""
    # Note: same as single vae run but always CPU
    from opguard.tests.trivial import trivial as _trival

    _trival()


@pytest.mark.smoke
def smoke() -> None:
    """Simplest run with inference (no CUDA/GPU needed)."""
    # Note: same as single vae run but always CPU
    from opguard.tests.smoke import smoke as _smoke

    _smoke()


@pytest.mark.cpu
def cpu() -> None:
    """Test CPU mode, full precision."""
    from opguard.tests.vae import tiny_vae_roundtrip_sequence

    logger.info("Running 'cpu' tests")
    tiny_vae_roundtrip_sequence(device="cpu", dtype="float32")


@pytest.mark.gpu
def gpu() -> None:
    """Test GPU mode, half-precision."""
    import torch

    from opguard.tests.vae import tiny_vae_roundtrip_sequence

    logger.info("Running 'gpu' tests")
    if torch.cuda.is_available():
        tiny_vae_roundtrip_sequence(device="cuda", dtype="float16")
    else:
        logger.warning("Unable to run GPU tests due to loack of CUDA/GPU")


@pytest.mark.bfloat
def bfloat() -> None:
    """Test GPU mode, with bfloat16 dtype.

    Note: will fail if no GPU, and fallback to float16 if insufficient compute capability.
    """
    import torch

    from opguard.tests.vae import tiny_vae_roundtrip_sequence

    logger.info("Running 'bfloat' tests")
    if torch.cuda.is_available():
        tiny_vae_roundtrip_sequence(device="cuda", dtype="bfloat16")
    else:
        logger.warning("Unable to run BFLOAT16 tests due to loack of CUDA/GPU")


@pytest.mark.fp16vae
def fp16vae() -> None:
    """Test SDXL vae with FP16 fix, with bfloat16 dtype.

    Note: will fail if no GPU, and fallback to float16 if insufficient compute capability.
    """
    import torch

    from opguard.tests.vae import sdxl_vae_roundtrip

    logger.info("Running 'fp16vae' tests")
    if torch.cuda.is_available():
        sdxl_vae_roundtrip(
            local_hfhub_variant_check_only=False,
            force_export_refresh=True,
        )
    else:
        logger.warning("Unable to run sdxl_vae_fp16_fix tests due to loack of CUDA/GPU")


@pytest.mark.nlp
def nlp() -> None:
    """Test BLIP1 captioner."""
    from opguard.tetsts.nlp import blip

    logger.info("Running 'nlp' smaller tests")
    blip(test_blip1=True, test_blip2_4bit=True, test_blip2_16bit=False)


@pytest.mark.nlpxl
def nlpxl() -> None:
    """Test BLIP2 captioner."""
    from opguard.tetsts.nlp import blip

    logger.info("Running 'nlp' large tests")
    blip(test_blip1=False, test_blip2_4bit=False, test_blip2_16bit=True)


@pytest.mark.sd
def sd() -> None:
    """Test tiny SD."""
    import torch

    from opguard.tests.sd import sd_tiny

    logger.info("Running 'sd' tests")
    if torch.cuda.is_available():
        sd_tiny()
    else:
        logger.warning("Unable to run sd tests due to loack of CUDA/GPU")


@pytest.mark.control
def control() -> None:
    """Test various controlnets."""
    from opguard.tests.controlnets import control

    logger.info("Running 'control' tests")
    control()


@pytest.mark.inversion
def inversion() -> None:
    """Test inversion."""
    from opguard.tests.inversion import sdxl_inversion

    logger.info("Running 'inversion' tests")
    sdxl_inversion(test_image=None, generated_caption=None)


@pytest.mark.vae
def vae() -> None:
    """VAE test."""
    logger.info("Running 'vae' meta test set")
    cpu()
    gpu()
    bfloat()
    fp16vae()


@pytest.mark.slow
def slow() -> None:
    """Slower tests, checks basic caching and multiple precision/devices."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("Some tests will be skipped due to lack of CUDA/GPU")
    logger.info("Running 'slow' meta test set")
    nlp()
    sd()
    control()


@pytest.mark.large
def large() -> None:
    """Large VRAM tests, runs large models."""
    import torch

    from opguard.tests.sd import sdxl

    if not torch.cuda.is_available():
        logger.warning("Some tests will be skipped due to lack of CUDA/GPU")
    logger.info("Running 'slow' meta test set")
    nlpxl()
    sdxl()
    inversion()


@pytest.mark.full
def full() -> None:
    """Run smoke, slow and large tests."""
    smoke()
    vae()
    slow()
    large()
