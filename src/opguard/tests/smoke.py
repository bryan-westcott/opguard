"""Smoke test with simple tiny VAE."""

# ruff: noqa: PLC0415  # try to keep heavy imports restricted to when needed for testing/profiling

from loguru import logger


def smoke() -> None:
    """Simplest run with inference (no CUDA/GPU needed)."""
    # Note: same as single vae run but always CPU
    logger.info("Running 'smoke' tests")
    from opguard.tests.vae import tiny_vae_roundtrip

    tiny_vae_roundtrip(
        device="cpu",
        dtype="float32",
        local_hfhub_variant_check_only=False,
        force_export_refresh=True,
    )


if __name__ == "__main__":
    smoke()
