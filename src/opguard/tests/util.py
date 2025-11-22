"""Test utilities."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def load_test_image(
    *,
    image_passthrough: PILImage | None = None,
    use_blank: bool = False,
    hub_repo_id: str | None = None,
    hub_filename: str | None = None,
    hub_revision: str | None = None,
    final_size: tuple[int, int] | None = None,
    local_files_only: bool = False,
    allow_direct_download: bool = True,
    image_url_override: str | None = None,
) -> PILImage:
    """Test image loader.

    Load a small test image with Hugging Face Hub caching and an optional
    direct-URL fallback. A pre-supplied image can bypass all I/O.

    Resolution order
    1) If `image_passthrough` is provided, return it as-is.
    2) If `use_blank` is provided, generate a white image
       assuming final_size is set
    2) Try `hf_hub_download(repo_type='dataset')` for
       `{hub_repo_id}::{hub_filename}` at `hub_revision`
       (honors `local_files_only`).
    3) If that fails and `allow_direct_download=True`, fetch via HTTPS from
       `image_url_override` or a resolve-URL derived from the Hub fields.

    Parameters
    ----------
    image_passthrough : PIL.Image.Image | None, OPTIONAL
        Pre-loaded image to return immediately (skips Hub/HTTP).
    use_blank: bool, OPTIONAL
        Create a blank image according to final_size
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
    from PIL.Image import new as pil_new
    from PIL.Image import open as pil_open

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

    # Generate blank, if requested
    if not image and use_blank:
        if not final_size:
            message = "Must provide final_size if use_blank==True"
            raise ValueError(message)
        image = pil_new("RGB", final_size, "white")

    if not image:
        # first try huggingface hub module,
        from huggingface_hub import hf_hub_download

        # will download and cache on very first call
        try:
            path = hf_hub_download(
                repo_id=hub_repo_id,
                filename=hub_filename,
                repo_type="dataset",  # important: it's a dataset repo
                revision=hub_revision,  # or pin a commit hash for reproducibility
                local_files_only=local_files_only,
            )
            image = pil_open(path).convert("RGB")
            logger.debug("Obtained test image from HF Hub download or cache")
        except RuntimeError:
            logger.warning("HF Hub Download failed")

    if not image and allow_direct_download:
        # then try direct download, constructing URL based on hub info
        try:
            import requests

            image = pil_open(requests.get(image_url, stream=True, timeout=(3.0, 30.0)).raw)
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


def hstack_numpy(images: list[PILImage]) -> PILImage:
    """Stack images horizontally using NumPy arrays."""
    import numpy as np
    from PIL.Image import fromarray as pil_fromarray

    arrays = [np.asarray(img.convert("RGB")) for img in images]
    stacked = np.hstack(arrays)
    return pil_fromarray(stacked)
