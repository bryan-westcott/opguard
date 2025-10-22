"""Smoke tests for OpGuardBase exercising CPU, GPU, and BFloat modes.

To run, with debugging:     uv run pytest --log-cli-level=DEBUG --capture=no
"""

# We do not care about LSP substitutability, OpGuard is not used directly
# mypy: disable-error-code=override

from typing import Any

import numpy as np
import PIL
import pytest
import requests
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL.Image import Image as PILImage

from opguard.controlnets import Dwpose, Hed, MarigoldDepth, MarigoldNormals, Tile
from opguard.nlp import Blip1, Blip2
from opguard.sd import SdTinyNanoTextToImage, SdxlTextToImage
from opguard.vae import TinyVaeForSd


def load_test_image(
    *,
    hub_repo_id: str | None = None,
    hub_filename: str | None = None,
    hub_revision: str | None = None,
    final_size: tuple[int, int] | None = None,
    local_files_only: bool = False,
    allow_direct_download: bool = True,
    image_passthrough: PILImage | None = None,
    image_url_override: str | None = None,
) -> PILImage:
    """Test image loader.

    Load a small test image with Hugging Face Hub caching and an optional
    direct-URL fallback. A pre-supplied image can bypass all I/O.

    Resolution order
    1) If `image_passthrough` is provided, return it as-is.
    2) Try `hf_hub_download(repo_type='dataset')` for
       `{hub_repo_id}::{hub_filename}` at `hub_revision`
       (honors `local_files_only`).
    3) If that fails and `allow_direct_download=True`, fetch via HTTPS from
       `image_url_override` or a resolve-URL derived from the Hub fields.

    Parameters
    ----------
    hub_repo_id : str | None, default="YiYiXu/testing-images"
        Dataset repo containing the image.
    hub_filename : str | None, default="women_input.png"
        File path/name within the dataset repo.
    hub_revision : str | None, default="main"
        Branch/tag/commit. For reproducibility, prefer a commit hash.
    final_size : tuple[int, int] | None, optional
        If given, resize to (width, height) before returning.
    local_files_only : bool, default=False
        Passed to `hf_hub_download`. When True, only uses files already in the
        local HF cache (no network).
    allow_direct_download : bool, default=True
        Whether to attempt an HTTPS fallback if the Hub fetch fails.
    image_passthrough : PIL.Image.Image | None, optional
        Pre-loaded image to return immediately (skips Hub/HTTP).
    image_url_override : str | None, optional
        Explicit HTTPS URL to use for the fallback download. If not set, a
        resolve-URL is constructed from the Hub fields.

    Returns
    -------
    PIL.Image.Image
        An RGB image, possibly resized.

    Raises
    ------
    ValueError
        If the image cannot be obtained from any source.

    Notes
    -----
    - To enable offline use (`local_files_only=True`), call at least once with
      `local_files_only=False` so the asset is cached locally
      (typically under `~/.cache/huggingface/hub`).
    - `hub_revision` should be pinned to a commit hash for deterministic tests.

    Examples
    --------
    # Online first run (caches file), offline thereafter
    img = load_test_image(local_files_only=False)
    img_offline = load_test_image(local_files_only=True)

    # Use a specific dataset file and commit
    img = load_test_image(
        hub_repo_id="hf-internal-testing/diffusers-images",
        hub_filename="flag/mini.png",
        hub_revision="7f6d1c1",  # commit hash
    )

    # Provide your own image to bypass I/O
    img = load_test_image(image_passthrough=PIL.Image.new("RGB", (64, 64), "white"))
    """
    # ruff: noqa: PLR0913  (this is a test helper)
    # Huggingface hub repo information
    hub_repo_id = hub_repo_id or "YiYiXu/testing-images"
    hub_filename = hub_filename or "women_input.png"
    hub_revision = hub_revision or "main"
    # Backup method, construct direct URL from hub information
    image_url = (
        image_url_override or f"https://huggingface.co/datasets/{hub_repo_id}/resolve/{hub_revision}/{hub_filename}"
    )

    # Obtain from passthrough if provided
    image = image_passthrough

    if not image:
        # first try huggingface hub module,
        # will download and cache on very first call
        try:
            path = hf_hub_download(
                repo_id=hub_repo_id,
                filename=hub_filename,
                repo_type="dataset",  # important: it's a dataset repo
                revision=hub_revision,  # or pin a commit hash for reproducibility
                local_files_only=local_files_only,
            )
            image = PIL.Image.open(path).convert("RGB")
            logger.debug("Obtained test image from HF Hub download or cache")
        except RuntimeError:
            logger.warning("HF Hub Download failed")

    if not image and allow_direct_download:
        # then try direct download, constructing URL based on hub info
        try:
            image = PIL.Image.open(requests.get(image_url, stream=True, timeout=(3.0, 30.0)).raw)
            logger.debug("Obtained test image from direct URL download")
        except RuntimeError:
            logger.warning("Direct download failed")

    if not image:
        # No alternatives remaining
        message = "Unable to obtain test image"
        raise ValueError(message)

    if final_size:
        # Resize if needed
        image = image.resize(final_size)

    return image


def _tiny_vae_roundtrip(*, device: str | torch.device, dtype: torch.dtype, **kwargs) -> None:
    """Tiny round-trip VAE that exercises OpGuard with various device/dtypes."""
    # ruff: noqa: ANN003  (too restrictive on tests)
    logger.info(f"Running smoke test with {device=}, {dtype=}, {kwargs=}")

    with TinyVaeForSd(device_override=device, dtype_override=dtype, **kwargs) as vae:
        input_image = load_test_image(final_size=(512, 512), allow_direct_download=False)
        output_image = vae(input_raw=input_image, mode="encode-decode")
    assert input_image.size == output_image.size


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


def blip(*, test_image: PILImage | None = None, test_blip1: bool = True, test_blip2: bool = True) -> dict[str, Any]:
    """Test NLP objects."""
    test_image = test_image or load_test_image()
    return_blip1 = {}
    return_blip2 = {}
    if test_blip2:
        with Blip2() as blip2:
            caption_blip2 = blip2(input_raw=test_image)
            logger.info(f"blip2: {caption_blip2}")
            caption_blip2_conditional = blip2(
                input_raw=test_image,
                text="Question: What color is her hair? Answer:",
            )
            logger.info(f"blip2 (question): {caption_blip2_conditional}")
            return_blip2 = {
                "caption_blip2": caption_blip2,
                "caption_blip2_conditional": caption_blip2_conditional,
            }
    if test_blip1:
        with Blip1() as blip1:
            caption_blip1 = blip1(input_raw=test_image)
            logger.info(f"blip1 (blip1): {caption_blip1}")
            text_blip1 = "Hair Color"
            caption_blip1_conditional = blip1(input_raw=test_image, text=text_blip1)
            logger.info(f'blip1 (text="{text_blip1}"): {caption_blip1_conditional}')
            return_blip1 = {
                "caption_blip1": caption_blip1,
                "caption_blip1_conditional": caption_blip1_conditional,
            }
    return return_blip1 | return_blip2


def sdxl() -> None:
    """Test SDXL."""
    prompt = "An astronaut on a horse on the moon."
    with SdxlTextToImage() as sdxl:
        image = sdxl(prompt=prompt)
    # check for image output, with some width/height and not all zeros.
    assert isinstance(image, PILImage)
    assert all(dim > 0 for dim in image.size)
    assert len(np.unique(np.array(image).ravel())) > 1


def sd_tiny() -> None:
    """Test SD 1.5."""
    prompt = "An astronaut on a horse on the moon."
    with SdTinyNanoTextToImage() as sd:
        image = sd(input_raw=None, prompt=prompt, height=128, width=128)
    # check for image output, with some width/height and not all zeros.
    assert isinstance(image, PILImage)
    assert all(dim > 0 for dim in image.size)
    assert len(np.unique(np.array(image).ravel())) > 1
    image.save("test-sd21.png")


def controlnets(test_image: PILImage | None = None) -> dict[str, Any]:
    """Test ControlNets objects."""
    test_image = test_image or load_test_image()
    display_size = (512, 512)
    control_types = (Tile, Hed, MarigoldDepth, MarigoldNormals, Dwpose)
    control_outputs = []
    return_control = {}
    for control_type in control_types:
        logger.debug(f"Running test: {control_type}")

        with control_type() as control:
            control_output = control(input_raw=test_image).resize(display_size)
            control_outputs.append(control_output)
            return_control[control.name] = control_output
    return return_control


@pytest.mark.smoke
def smoke() -> None:
    """Simplest run (no CUDA/GPU needed)."""
    _tiny_vae_roundtrip(
        device="cpu",
        dtype=torch.float32,
        local_hfhub_variant_check_only=False,
        force_export_refresh=True,
    )


@pytest.mark.nlp
def nlp() -> None:
    """Test BLIP1 captioner."""
    blip(test_blip2=False)


@pytest.mark.sd
def sd() -> None:
    """Test BLIP1 captioner."""
    sd_tiny()


@pytest.mark.controlnets
def control() -> None:
    """Test various controlnets."""
    controlnets()


@pytest.mark.slow
def slow() -> None:
    """Slower tests, checks basic caching and multiple precision/devices."""
    cpu()
    gpu()
    bfloat()
    nlp()
    controlnets()
