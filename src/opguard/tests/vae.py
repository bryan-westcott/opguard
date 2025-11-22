"""VAE Tests."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from opguard.util import DeviceLike, DTypeLike


def sdxl_vae_roundtrip(*, device: str | DeviceLike = "cuda", dtype: DTypeLike = "bfloat16", **kwargs) -> None:
    """Tiny round-trip VAE that exercises OpGuard with various device/dtypes."""
    # ruff: noqa: ANN003  (too restrictive on tests)
    import torch

    from opguard.tests.util import load_test_image
    from opguard.vae import VaeSdxlFp16Fix

    # avoid importing torch just for defaults
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    with VaeSdxlFp16Fix(device_override=device, dtype_override=dtype, **kwargs) as vae:
        logger.debug(
            f"Running vae-for-sdxl roundtrip test for {type(vae).__name__} with: {device=}, {dtype=}, {kwargs=}",
        )
        input_image = load_test_image(final_size=(512, 512), allow_direct_download=False)
        output_image = vae(input_raw=input_image, mode="encode-decode")
    assert input_image.size == output_image.size


def tiny_vae_roundtrip(*, device: str | DeviceLike, dtype: DTypeLike, **kwargs) -> None:
    """Tiny round-trip VAE that exercises OpGuard with various device/dtypes."""
    # ruff: noqa: ANN003  (too restrictive on tests)
    import torch

    from opguard.tests.util import load_test_image
    from opguard.vae import VaeTinyForSd

    # avoid importing torch just for defaults
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    with VaeTinyForSd(device_override=device, dtype_override=dtype, **kwargs) as vae:
        logger.debug(f"Running vae-tiny roundtrip test for {type(vae).__name__} with: {device=}, {dtype=}, {kwargs=}")
        input_image = load_test_image(final_size=(512, 512), allow_direct_download=False)
        output_image = vae(input_raw=input_image, mode="encode-decode")
    assert input_image.size == output_image.size


def tiny_vae_roundtrip_sequence(*, device: str | DeviceLike, dtype: DTypeLike) -> None:
    """Test tiny VAE round trip, but also exercise local export caching."""
    # run it once, forcing refresh of export and remote variant check
    logger.debug("Running with forced export refresh (1/3)")
    tiny_vae_roundtrip(device=device, dtype=dtype, local_hfhub_variant_check_only=False, force_export_refresh=True)
    # then run with defaults
    logger.debug("Running with default cache handling (2/3)")
    tiny_vae_roundtrip(device=device, dtype=dtype)
    # then explicitly force use of local cache export and variant check
    logger.debug("Running with export only (3/3)")
    tiny_vae_roundtrip(device=device, dtype=dtype, local_hfhub_variant_check_only=True, only_load_export=True)
