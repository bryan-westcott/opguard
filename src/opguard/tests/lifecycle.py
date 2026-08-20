"""Lifecycle test matrix runners for OpGuardBase (load/free/reload semantics).

Each runner is a plain callable with bare asserts so it can run both
from the standalone pytest file (tests/test_lifecycle.py) and from the
marker-suite entry functions in test.py. No mocks: real guards, real
construction, CPU-pinned by default so dtype resolves to float32 and
variant_guard never probes the network.
"""

# ruff: noqa: SLF001   # lifecycle checks legitimately inspect private state
# ruff: noqa: PLC0415  # keep heavy imports restricted to when needed
# ruff: noqa: ANN401   # passthrough helpers mirror the abstract interfaces
# ruff: noqa: D401     # matrix docstrings describe observed behavior
# ruff: noqa: PLR2004  # small literal load counts are clearest inline

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from opguard.tests.trivial import PassthroughDetector

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from PIL.Image import Image as PILImage


class LifecyclePassthrough(PassthroughDetector):
    """Passthrough guard that counts _load_detector calls, CPU by default."""

    NAME = "lifecycle-passthrough"

    def __init__(self, **kwargs: Any) -> None:
        """Pin the device to CPU and start the load counter."""
        # set before super().__init__ since keep_warm loads during init
        self.load_count = 0
        kwargs.setdefault("device_override", "cpu")
        super().__init__(**kwargs)

    def _load_detector(self) -> Callable:
        self.load_count += 1
        return super()._load_detector()


class _RealPassthroughProcessor:
    """Real minimal processor object (loaded to observe partial state)."""

    def preprocess(self, input_raw: Any) -> Any:
        return input_raw

    def postprocess(self, output_raw: Any) -> Any:
        return output_raw


class FailingDetectorGuard(LifecyclePassthrough):
    """Guard whose processor loads fine but whose detector load raises."""

    NAME = "lifecycle-failing-detector"

    def _load_processor(self) -> _RealPassthroughProcessor:
        return _RealPassthroughProcessor()

    def _load_detector(self) -> Callable:
        message = "intentional detector load failure"
        raise RuntimeError(message)


def _blank_image() -> PILImage:
    from opguard.tests.util import load_test_image

    return load_test_image(use_blank=True, final_size=(64, 64))


def bare_first_call() -> None:
    """A fresh guard loads on the very first bare call, then frees."""
    guard = LifecyclePassthrough()
    out = guard(input_raw=_blank_image())
    assert out is not None
    assert guard.load_count == 1
    # keep_warm=False and not in a context: freed again after the call
    assert guard._detector is None
    assert guard._is_freed is True


def keep_warm_reuse() -> None:
    """A keep_warm guard loads once and stays loaded across calls."""
    guard = LifecyclePassthrough(keep_warm=True)
    assert guard.load_count == 1
    guard(input_raw=_blank_image())
    assert guard._detector is not None
    guard(input_raw=_blank_image())
    assert guard._detector is not None
    assert guard.load_count == 1
    guard._free(reason="lifecycle check cleanup")


def context_manager_roundtrip() -> None:
    """Enter loads exactly once, calls reuse it, exit frees."""
    with LifecyclePassthrough() as guard:
        guard(input_raw=_blank_image())
        guard(input_raw=_blank_image())
        assert guard.load_count == 1
        assert guard._detector is not None
    assert guard._detector is None
    assert guard._is_freed is True


def call_after_exit() -> None:
    """A guard remains usable after its context exits (lazy reload)."""
    with LifecyclePassthrough() as guard:
        guard(input_raw=_blank_image())
    out = guard(input_raw=_blank_image())
    assert out is not None
    assert guard.load_count == 2
    # outside any context and keep_warm=False: freed after the call
    assert guard._detector is None


def enter_failure_leaves_no_partial_state() -> None:
    """A failing __enter__ frees partial state and resets context flags.

    The processor loads before the detector inside _load, so a detector
    failure would otherwise strand a loaded processor on the instance.
    """
    guard = FailingDetectorGuard()
    raised = False
    try:
        with guard:
            pass
    except RuntimeError:
        raised = True
    assert raised, "expected the detector load failure to propagate"
    assert guard._processor is None
    assert guard._detector is None
    assert guard._in_context is False
    assert guard._is_freed is True


def double_free_idempotent() -> None:
    """Freeing twice is safe and leaves the guard freed."""
    guard = LifecyclePassthrough(keep_warm=True)
    guard._free(reason="lifecycle check first free")
    assert guard._is_freed is True
    guard._free(reason="lifecycle check second free")
    assert guard._is_freed is True
    assert guard._detector is None


def free_then_reload_cache_hit(*, device: str = "cpu", dtype: str = "float32") -> None:
    """A load-free-reload cycle keeps model_id and does not rebuild the export."""
    import torch

    from opguard.util import _cache_calc_export_name
    from opguard.vae import VaeTinyForSd

    torch_dtype = getattr(torch, dtype)

    # Build a fresh export so the check does not rely on pre-existing state
    builder = VaeTinyForSd(
        device_override=device,
        dtype_override=torch_dtype,
        keep_warm=True,
        force_export_refresh=True,
    )
    # capture the dtype the guard actually resolved (cuda may fall back)
    effective_dtype = builder.dtype
    builder._free(reason="lifecycle check: export built")
    del builder

    _, _, export_dir = _cache_calc_export_name(
        base_export_name=f"{VaeTinyForSd.NAME}-detector",
        dtype=effective_dtype,
        quant_config=None,
    )
    assert export_dir is not None
    metadata_path: Path = export_dir / "metadata.json"
    assert metadata_path.exists()

    guard = VaeTinyForSd(device_override=device, dtype_override=torch_dtype, keep_warm=True)
    assert guard.detector is not None
    assert guard.model_id == VaeTinyForSd.MODEL_ID

    mtime_before = metadata_path.stat().st_mtime_ns
    content_before = metadata_path.read_bytes()

    guard._free(reason="lifecycle check: free before reload")
    assert guard._detector is None
    assert guard.detector is not None
    assert guard.model_id == VaeTinyForSd.MODEL_ID

    assert metadata_path.stat().st_mtime_ns == mtime_before
    assert metadata_path.read_bytes() == content_before
    guard._free(reason="lifecycle check: cleanup")


def lifecycle_checks() -> None:
    """Run the full CPU lifecycle matrix (for marker-suite wiring)."""
    logger.info("Running lifecycle matrix checks")
    bare_first_call()
    keep_warm_reuse()
    context_manager_roundtrip()
    call_after_exit()
    enter_failure_leaves_no_partial_state()
    double_free_idempotent()
    free_then_reload_cache_hit(device="cpu", dtype="float32")
    logger.info("Lifecycle matrix checks passed")
