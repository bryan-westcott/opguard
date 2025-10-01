"""High-level runtime guard for CUDA-backed inference.

This module defines `ModelGuardBase`, an abstract base class that wraps the
lower-level utilities in `cuda_guard.py` to run VRAM-heavy models safely.
It centralizes device/dtype resolution, AMP/grad modes, cross-device sync,
traceback sanitization, and cache cleanup—while leaving model loading and
the forward pass to subclasses.

`ModelGuardBase` can be used persistently or as a context manager:
    with MyModelGuard(...) as guard:
        out = guard(input_raw=...)

When run persistently, it can optionally:
    * lazy load model on call
    * aggressively free after call

Subclasses specialize loading/freeing and the call path for concrete models
"""

# ruff: noqa: ANN401  (this is an abstract base class)
# ruff: noqa: ANN003 (this is an abstract base class)
# ruff: noqa: ARG002 (this is an abstract base class)

# mypy: disable-error-code=override
#   Why: this module specializes pipeline methods with narrower/extra kwargs.
#   We don't use polymorphic calls through the base, so substitutability doesn't apply.
#   Note: this must be in all derived classe to avoid contravariance errors

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterable

import torch
from huggingface_hub import HfApi
from loguru import logger

from .cuda_guard import DeviceLike, DeviceMapLike, cuda_guard, device_guard, dtype_guard, sync_gc_and_cache_cleanup


class ModelGuardBase(ABC):
    """Abstract guard for CUDA inference with predictable memory behavior.

    This base class owns the execution environment (device list, effective
    dtype, AMP/grad setup, synchronization, traceback scrubbing, GC/cache
    clearing). Subclasses provide model construction and the actual forward
    call, keeping model-specific code minimal.  The class is designged to
    be easily overridable (with sensible defaults) for model loading
    (including preprocessor and/or postprocessor) and also the pre-process,
    caller, and post-process steps themselves.

    Usage patterns:
        1) Persistent instance (keep models warm across calls):
            >>> guard = MyModelGuard(device="cuda:0", dtype=torch.bfloat16, keep_warm=True)
            ... out = guard(input_raw=...)
            ...  out2 = guard(input_raw=...)
        2) Context manager (auto-load on enter, auto-free on exit):
            >>> with MyModelGuard(device_map="auto", dtype=torch.bfloat16) as guard:
            ... out = guard(input_raw=...)

    Notes:
      - Each `__call__` is wrapped in a composed CUDA guard (grad/AMP/sync/
        sanitize/GC/cache clear). Outputs are typically detached to CPU to
        avoid dangling CUDA refs in notebooks or long-lived services.
      - Device/dtype selection is resolved once at init; bf16→fp16→fp32 fallback
        is applied based on capability.

    Subclass responsibilities
        - Required specialization:
            - _load_detector() (specialization required): load the main inference model
        - Optional specialization
            - _load_processor(): load a processor for pre- and post-processing
            - _caller(): special steps for _preprocess, _predict and _postproces
            - _preprocess(): special steps for preprocessing
            - _predict(): special steps for prediction call
            - _postprocess(): special steps for postprocessing

    Class attributes:
        - Required override:
            NAME (str): Human-friendly name for the model
            MODEL_ID (str): Huggingface ID of the model
            REVISION (str): Huggingface revision (git hash or branch like "main")
        - Optional override:
            NEED_GRADS: whether gradients are needed
                - Note: usually false except for techniques like inversion
            DEFAULT_DEVICE (str | int | torch.device): default device to be used (default is "cuda")
            DEFAULT_DEVICE_MAP: default device mapping (for models that use it, default is "auto")
            DTYPE_PREFERENCE: preference for smallest precision
                preference order: torch.bfloat16 -> torch.float16 -> torch.float32
                Note: this is device (compute capability) dependent
                Note: if bfloat16 causes issues, override with wider supported float16
    """

    # --- Specialize Here -----

    # Subclasses must overrdie
    NAME: ClassVar[str] = ""
    MODEL_ID: ClassVar[str] = ""
    REVISION: ClassVar[str] = ""

    # Subclasses may override
    DEFAULT_DEVICE: ClassVar[DeviceLike] = "cuda"
    DEFAULT_DEVICE_MAP: ClassVar[DeviceMapLike] = "auto"
    # Whether honored is compute capability and hardware dependent
    DTYPE_PREFERENCE: ClassVar[torch.dtype] = torch.bfloat16
    # Whether gradients are needed
    NEED_GRADS: ClassVar[bool] = False

    def _caller(self, *, input_raw: Any) -> Any:
        """Actual call line inside __call__ safe wrapper."""
        input_proc = self._preprocess(input_raw=input_raw)
        output_raw = self._predict(input_proc=input_proc)
        return self._postprocess(output_raw=output_raw)  # output_proc

    @abstractmethod
    def _load_detector(self, **kwargs) -> Any:
        """Return an initialized detector model, must be specialized."""
        raise NotImplementedError

    def _load_processor(self, **kwargs) -> Any:
        """Load preprocessor (optional), often also used for post-processing."""
        logger.debug("Running default _load_processor(), no processor loaded")
        return None

    def _preprocess(self, *, input_raw: Any, **kwargs) -> Any:
        """Preprocessing, default is no-op, override for custom behavior."""
        logger.debug("Running default _preprocess(), simple passthrough")
        return input_raw  # default: input_proc = input_raw

    def _predict(self, *, input_proc: Any, **kwargs) -> Any:
        """Call the detector, default is a simple call, may be overridden."""
        if not self._detector:
            message = f"{type(self).__name__} must set _detector via _load_detector"
            raise ValueError(message)
        return self._detector(input_proc, **kwargs)

    def _postprocess(self, *, output_raw: Any, **kwargs) -> Any:
        """Postprocessing, default is no-op, override for custom behavior."""
        logger.debug("Running default _postprocess(), simple passthrough")
        return output_raw  # default: output_proc = output_raw

    # --- Boilerplate After Here ---

    def __init__(
        self,
        *,
        dtype_override: torch.device | None = None,
        device_override: DeviceLike | None = None,
        device_map_override: DeviceMapLike | None = None,
        keep_warm: bool = False,
        sanitize_cuda_errors: bool = True,
    ) -> None:
        """Initialize the model guard and resolve device list and effective dtype.

        Args:
            dtype_override:
                Optional override for the requested compute dtype. If not
                provided, uses the class default `DTYPE_PREFERENCE`. The final
                `self.dtype` is chosen via `dtype_guard` with preference
                **bfloat16 → float16 → float32** based on device support.
            device_override:
                Optional single-device override (e.g., 0, "cuda:1",
                `torch.device("cuda:0")`). If omitted, uses `DEFAULT_DEVICE`.
            device_map_override:
                Optional multi-GPU placement override. If `"auto"`, the device
                list is all visible CUDA devices in index order. If omitted,
                uses `DEFAULT_DEVICE_MAP`. Other mappings are not interpreted
                here.
            keep_warm:
                If True, models are eagerly loaded on init and kept in memory
                across calls. If False, loading may be lazy and `_free()` may
                release memory between calls.
            sanitize_cuda_errors:
                If True, suspected CUDA `RuntimeError`s during guarded calls are
                sanitized (tracebacks detached, devices synchronized, concise
                error re-raised).

        Sets:
            self.device:
                The chosen device spec (override or `DEFAULT_DEVICE`).
            self.device_map:
                The chosen device-map spec (override or `DEFAULT_DEVICE_MAP`).
            self.device_list:
                Normalized CUDA devices resolved by `device_guard(...)`.
            self.dtype_preference:
                The requested dtype (override or `DTYPE_PREFERENCE`).
            self.dtype:
                Effective dtype after validation/fallback by `dtype_guard(...)`.
            self.keep_warm:
                Stored flag controlling eager/lazy lifecycle.
            self.sanitize_cuda_errors:
                Stored flag controlling CUDA error sanitization.
            self._in_context:
                Internal flag indicating context-manager scope.
            self._is_freed:
                Internal flag indicating whether shutdown free has completed.
            self._processor, self._detector:
                Lazy-loaded model/processors (subclasses populate).
            self.extra_info:
                Dict for subclass/config metadata.

        Notes:
            - No heavy CUDA allocation happens here beyond device/dtype resolution.
            - If `keep_warm` is True, `_load()` is invoked to eagerly construct
              model components.
            - Use the context-manager protocol to auto-load/free:
                  with MyGuard(...) as g:
                      out = g(input_raw=...)
        """
        # Set device_list, but keep original device and device_map chosen
        self.device: DeviceLike = device_override or type(self).DEFAULT_DEVICE
        self.device_map: DeviceMapLike = device_map_override or type(self).DEFAULT_DEVICE_MAP
        self.device_list: list[torch.device] = device_guard(device=self.device, device_map=self.device_map)

        # Choose dtype
        self.dtype_preference: torch.dtype = dtype_override or type(self).DTYPE_PREFERENCE
        self.dtype: torch.dtype = dtype_guard(device_list=self.device_list, dtype_desired=self.dtype_preference)
        self.variant = self._variant_selector()

        logger.debug(f"Choices for {type(self).__name__}: {self.dtype=}, {self.variant=}")
        self.keep_warm: bool = keep_warm
        self.sanitize_cuda_errors: bool = True

        # Set this if in context manager to avoid freeing each iteration
        self._in_context: bool = False
        # Whether shutdown free has completed (regardless of keep_warm option)
        self._is_freed: bool = False

        # Default for lazy loading
        self._processor: Any = None
        self._detector: Any = None
        # For extra configuration information to keep track of
        self.extra_info: dict[str, Any] = {}

        # prepare detector now
        if self.keep_warm:
            logger.debug("Preloading models on init due to keep_warm option")
            self._load()

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure all attributes are set properly.

        Note: NAME, MODEL_ID, REVISION are expected to be overridden.
        """
        super().__init_subclass__(**kwargs)
        # Ensure all attributes set
        for attr in [
            "NAME",
            "MODEL_ID",
            "REVISION",
            "NEED_GRADS",
            "DEFAULT_DEVICE",
            "DEFAULT_DEVICE_MAP",
            "DTYPE_PREFERENCE",
        ]:
            val = getattr(cls, attr, None)
            if (val == "") or (val is None):
                message = f"{cls.__name__} must define non-empty class attr {attr!r}"
                raise TypeError(message)

    @property
    def name(self) -> str:
        """Return the class human-friendly identifier for the specialized class."""
        return type(self).NAME

    @property
    def model_id(self) -> str:
        """Return the Huggingface ID for the core model weights."""
        return type(self).MODEL_ID

    @property
    def revision(self) -> str:
        """Return the model revision (git hash or branch name)."""
        return type(self).REVISION

    @property
    def need_grads(self) -> bool:
        """Return whether gradients needed."""
        return type(self).NEED_GRADS

    def _variant_selector(self) -> str | None:
        """Select fp16 based on availability and preference in Huggingface Hub."""
        if self.dtype == torch.float32:
            # if 32-bit, return no variant
            variant = None
        elif self.dtype in (torch.float16, torch.bfloat16):
            # otherwise, half precision requested
            # Heuristic: check for common fp16-variant filenames used by Diffusers/Transformers."""
            # Note: repo_id is the same as model_id
            api = HfApi()
            files: Iterable[str] = api.list_repo_files(repo_id=self.model_id, revision=self.revision)
            # Look for fp16.safetensors, float16.safetensors, fp16.bin or float16.bin suffixes in filename
            has_fp16 = any(("fp16" in f) or ("float16" in f) for f in files)
            variant = "fp16" if has_fp16 else None
        else:
            message = f"Invalid dtype provided: {self.dtype=}"
            raise ValueError(message)
        return variant

    def _load(self) -> None:
        """Load detector and processor (if applicable), unless already loaded."""
        # Always indicate potentially unfreed
        self._is_freed = False
        # Now attemtp to load
        logger.debug("Loading models")

        # reset to empty
        if self.extra_info:
            self.extra_info = {}
        if not self._processor:
            self._processor = self._load_processor()
        if not self._detector:
            self._detector = self._load_detector()

    def _free(self, *, clear_cache: bool = True, reason: str = "unspecified") -> None:
        """Free detector and processor (if applicable).

        Note: will also garbage collect and clear torch cache.
        """
        logger.debug(f"Running free for {type(self).__name__}, reason: {reason}")

        # Don't free twice, unless models still loaded
        if self._is_freed and (not self._processor) and (not self._detector):
            logger.debug("Already freed, exiting early from _free()")
            return

        # free detector and processor, if allocated
        if self._detector:
            self._detector = None
        if self._processor:
            self._processor = None
        # reset to empty
        if self.extra_info:
            self.extra_info = {}

        # Note: gc and cache clear recommended since
        # models themselves are not freed inside cuda_guard
        if not clear_cache:
            logger.debug("Free complete (skipping sync, garbage collection, and torch cache empty)")
            return

        # Apply GC and cache clear as we just freed up the models
        # Note: suppress errors as this is best-effort VRAM freeup
        #       and we already handled RuntimeErrors inside cuda_guard
        # Note: must sync and gc to get proper cache clear
        sync_gc_and_cache_cleanup(
            do_sync=True,
            do_garbage_collect=True,
            do_empty_cache=True,
            suppress_errors=True,
            device_list=self.device_list,
        )

        # Indicate already freed
        logger.debug("Free complete (including sync, garbage collection, and torch cache empty)")
        self._is_freed = True

    def __enter__(self) -> "ModelGuardBase":
        """Support for use as context manager, avoiding agressive per-call freeing."""
        self._in_context = True
        logger.debug("Pre-loading models on enter due to context manager")
        self._load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, this is standard format
        """Free on exit of context always."""
        self._in_context = False
        self._free(reason="Context manager exit")

    def __del__(self) -> None:
        """On object delete, all the free method, handlnig exceptions."""
        self._free(reason="destructor")

    def __call__(self, *, input_raw: Any, **kwargs) -> Any:
        """Call the model, with optional lazy loading and/or agressive free if needed.

        Note: Will also detach (to cpu) all outputs to avoid dangling references tying up VRAM
        """
        # Private stream to keep track of for synchronization
        try:
            # Lazy loader
            if self._is_freed:
                logger.debug("Lazy loading models on call")
                self._load()
            # Autocast for proper precision, accounting for gradient needs, also stream synchronize
            # Note: dtype and device guard are already pre-checked in init
            with cuda_guard(
                dtype=self.dtype,
                need_grads=self.need_grads,
                device=self.device,
                device_map=self.device_map,
                sanitize_cuda_errors=self.sanitize_cuda_errors,
                device_list=self.device_list,  # previously computed, so override
                effective_dtype=self.dtype,  # previously computed, so override
            ) as (_device_list, _effective_dtype, detach):
                # Note: cache_guard will handle detaching outptu and tracebacks
                return detach(self._caller(input_raw=input_raw, **kwargs))
        finally:
            # Agressively free up VRAM if desired
            # Ensure free as expected even on error
            if (not self.keep_warm) and (not self._in_context):
                self._free(reason=f"per-call finally, {self.keep_warm=}, {self._in_context=}")
