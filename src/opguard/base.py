"""High-level runtime guard for inference.

This module defines `OpGuardBase`, an abstract base class that wraps the
lower-level utilities in `opguard_util.py` to run VRAM-heavy models safely.
It centralizes device/dtype resolution, AMP/grad modes, cross-device sync,
traceback sanitization, and cache cleanup—while leaving model loading and
the forward pass to subclasses.

`OpGuardBase` can be used persistently or as a context manager:
    with OpGuardDerived(...) as guard_model:
        out = guarded_model(input_raw=...)

When run persistently, it can optionally:
    * lazy load model on call
    * aggressively free after call

For an example of how simple specialization can be, see
    the smoke test Smoke class in smoke.py

Subclasses specialize loading/freeing and the call path for concrete models

Minimal example: this will apply all the guards (with default options)

>>> import torch
>>> from diffusers import AutoencoderTiny
>>> from opguard import OpGuardBase
>>> class TinyVae(OpGuardBase):
...     NAME = "tiny-vae"
...     MODEL_ID = "madebyollin/taesd"
...     REVISION = "main"
...
...     def _load_detector(self, **kwargs: dict[str, Any]) -> AutoencoderTiny:
...         return AutoencoderTiny.from_pretrained(
...             kwargs["model_id"],
...             torch_dtype=kwargs["dtype"],
...             local_files_only=kwargs["local_files_only"],
...             revision=kwargs["revision"],
...         ).to(kwargs["device"])
...
...     def _encode(self, *, image: torch.FloatTensor) -> torch.FloatTensor:
...         return self._detector.encode(image.to(self.device, self.dtype)).latents
...
...     def _decode(self, *, latent: torch.FloatTensor) -> torch.FloatTensor:
...         return self._detector.decode(latent.to(self.device, self.dtype)).sample
...
...     def _predict(self, *, input_proc: torch.FloatTensor) -> torch.FloatTensor:
...         return self._decode(latent=self._encode(image=input_proc))
>>> with TinyVae() as vae:
...     input_tensor = torch.rand((2, 3, 512, 512), device=vae.device, dtype=vae.dtype) * 2 - 1
...     assert vae(input_raw=input_tensor).shape == input_tensor.shape
"""

# ruff: noqa: ANN401  (this is an abstract base class)
# ruff: noqa: ANN003 (this is an abstract base class)
# ruff: noqa: ARG002 (this is an abstract base class)

# mypy: disable-error-code=override
#   Why: this module specializes pipeline methods with narrower/extra kwargs.
#   We don't use polymorphic calls through the base, so substitutability doesn't apply.
#   Note: this must be in all derived classe to avoid contravariance errors

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from inspect import isabstract
from typing import Any, ClassVar, Literal

import torch
from loguru import logger

from .util import (
    DeviceLike,
    DeviceMapLike,
    call_guard,
    free_guard,
    init_guard,
    load_guard,
)


class OpGuardBase(ABC):
    """Abstract guard for inference with predictable memory behavior.

    This base class owns the execution environment (device list, effective
    dtype, AMP/grad setup, synchronization, traceback scrubbing, GC/cache
    clearing). Subclasses provide model construction and the actual forward
    call, keeping model-specific code minimal.  The class is designged to
    be easily overridable (with sensible defaults) for model loading
    (including preprocessor and/or postprocessor) and also the pre-process,
    caller, and post-process steps themselves.

    Usage patterns:
        1) Persistent instance (keep models warm across calls):
            >>> guard = OpGuardDerived(device="cuda:0", dtype=torch.bfloat16, keep_warm=True)
            ... out = guard(input_raw=...)
            ...  out2 = guard(input_raw=...)
        2) Context manager (auto-load on enter, auto-free on exit):
            >>> with OpGuardDerived(device_map="auto", dtype=torch.bfloat16) as guarded_model:
            ... out = guarded_model(input_raw=...)

    Notes:
      - Each `__call__` is wrapped in a composed guard.  Outputs are typically
        detached to CPU to avoid dangling refs in notebooks or long-lived
        services which can tie up VRAM (and RAM in cpu-only mode).
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
            NEED_GRADS: whether gradients are needed (default is False)
                - Note: usually False except for techniques like inversion
            DEFAULT_DEVICE (str | int | torch.device): default device to be used (default is "cuda")
            DEFAULT_DEVICE_MAP: default device mapping (for models that use it, default is "auto")
            DTYPE_PREFERENCE: preference for smallest precision
                preference order: torch.bfloat16 -> torch.float16 -> torch.float32
                Note: this is device (compute capability) dependent
                Note: if bfloat16 causes issues, override with wider supported float16
            USE_SAFETENSORS: whether to request safe serialized from huggingface hub (default True)
                Note: for cached/exported models, it will always be safetensors
            IS_CALLABLE: whether object is callable (default True)
                Note: only False for rare cases like diffusion pipe components
    """

    # --- Specialize Here -----

    # Subclasses must override

    @property
    @abstractmethod
    def NAME(self) -> str:  # noqa: N802  (becomes a ClassVar when concrete)
        """The name of the object."""

    @property
    @abstractmethod
    def MODEL_ID(self) -> str:  # noqa: N802  (becomes a ClassVar when concrete)
        """The HF hub repo id for the model weights."""

    @property
    @abstractmethod
    def REVISION(self) -> str:  # noqa: N802  (becomes a ClassVar when concrete)
        """The revision identifier for the model weights."""

    # --- Subclasses may override ---

    # Default torch device
    DEFAULT_DEVICE: ClassVar[DeviceLike] = "cuda"
    # default huggingface device_map
    DEFAULT_DEVICE_MAP: ClassVar[DeviceMapLike] = "auto"
    # Whether honored is compute capability and hardware dependent
    DTYPE_PREFERENCE: ClassVar[torch.dtype] = torch.bfloat16
    # Whether to use safetensors for loading (override to False if not available)
    USE_SAFETENSORS: ClassVar[bool] = True
    # Whether gradients are needed
    NEED_GRADS: ClassVar[bool] = False
    # Whether class is callable (only False in special cases, e.g., pipe components)
    IS_CALLABLE: ClassVar[bool] = True

    def _caller(self, *, input_raw: Any, **kwargs) -> Any:
        """Actual call line inside __call__ safe wrapper."""
        input_proc = self._preprocess(input_raw=input_raw, **kwargs)
        output_raw = self._predict(input_proc=input_proc, **kwargs)
        return self._postprocess(output_raw=output_raw, **kwargs)  # output_proc

    @abstractmethod
    def _load_detector(self, *, model_id_overrde: str | None = None, **kwargs) -> Any:
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
            message = f"{self.classname} must set _detector via _load_detector"
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
        sanitize_all_exceptions: bool = True,
        local_hfhub_variant_check_only: bool = False,
        local_files_only: bool = False,
        only_load_export: bool = False,
        force_export_refresh: bool = False,
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
                Optional multi-device placement override. It should be list of
                torch devices, optionally a singleton list of the cpu device.
            keep_warm:
                If True, models are eagerly loaded on init and kept in memory
                across calls. If False, loading may be lazy and `_free()` may
                release memory between calls.
            sanitize_all_exceptions:
                If Ture, any exceptions during guarded calls are
                sanitized (tracebacks detached, cuda devices synchronized, concise
                error re-raised).
            local_hfhub_variant_check_only:
                only look local cache when checking hfhub for variants
            local_files_only:
                only load models from local cache or export
            only_load_export:
                only load model from local export (not the same as
                huggingfacehub cache, see cache_guard)
            force_export_refresh:
                force a refresh of the local export

        Notes:
            - No heavy model allocation happens here beyond device/dtype resolution.
            - If `keep_warm` is True, `_load()` is invoked to eagerly construct
              model components.
            - Use the context-manager protocol to auto-load/free:
                  with MyGuard(...) as g:
                      out = g(input_raw=...)
        """
        logger.info(f"Initializing OpGuard {self.NAME} using {self.classname} from {self.MODEL_ID}@{self.REVISION}")
        # ruff: noqa: PLR0913  (configurable util with sane defaults)
        # A placeholder for overriding MODEL_ID without mutating class
        self._model_id_override: str | None = None
        # For overrideing use_safetensors
        # Note: can only overide with True
        self._use_safetensors_override: Literal[True] | None = None

        # Control logic
        # Set this if in context manager to avoid freeing each iteration
        self._in_context: bool = False
        # Whether shutdown free has completed (regardless of keep_warm option)
        self._is_freed: bool = False

        # Default for lazy loading
        self._processor: Any = None
        self._detector: Any = None
        # For extra configuration information to keep track of
        self.extra_info: dict[str, Any] = {}

        # loader-related params
        self.local_files_only: bool = local_files_only
        self.only_load_export: bool = only_load_export
        self.force_export_refresh: bool = force_export_refresh
        # caller related ptions
        self.keep_warm: bool = keep_warm
        self.sanitize_all_exceptions: bool = True

        # initialize dtype, variant, device_list based on runtime hardware
        self.device_list, self.device, self.dtype, self.variant, self.device_map = init_guard(
            device=device_override or self.DEFAULT_DEVICE,
            device_map=device_map_override or self.DEFAULT_DEVICE_MAP,
            dtype=dtype_override or self.DTYPE_PREFERENCE,
            model_id=self.model_id,
            revision=self.REVISION,
            local_hfhub_variant_check_only=local_hfhub_variant_check_only,
        )
        logger.info(
            f"Choices for {self.NAME}: {self.dtype=}, {self.variant=}, "
            f"{self.device=}, {self.device_map=}, {self.device_list=}",
        )

        # prepare detector now
        if self.keep_warm:
            logger.debug("Preloading models on init due to keep_warm option")
            self._load()

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure all attributes are set properly.

        Note: NAME, MODEL_ID, REVISION are expected to be overridden.
        """
        super().__init_subclass__(**kwargs)
        # Don't enforce attributes if still abstract
        # defer if explicitly allowed or still abstract
        if isabstract(cls):
            return
        # Ensure all attributes set
        for attr in [
            "NAME",
            "MODEL_ID",
            "REVISION",
            "NEED_GRADS",
            "CALLABLE",
            "DEFAULT_DEVICE",
            "DEFAULT_DEVICE_MAP",
            "DTYPE_PREFERENCE",
            "USE_SAFETENSORS",
        ]:
            val = getattr(cls, attr, None)
            if (val == "") or (val is None):
                message = f"{cls.__name__} must define non-empty class attr {attr!r}"
                raise TypeError(message)

    @property
    def model_id(self) -> str:
        """Return the Huggingface ID for the core model weights.

        Note: from type(self).MODEL_ID unles self._model_id_override not None.
        """
        # Note: may b
        return self._model_id_override if self._model_id_override else self.MODEL_ID

    @model_id.setter
    def model_id(self, value: str) -> None:
        """Model id setter, to _model_id_override witout class muatation."""
        self._model_id_override = value

    @property
    def use_safetensors(self) -> bool:
        """Return the Huggingface ID for the core model weights.

        Note: from type(self).USE_SAFETENSORS unles self._use_safetensors_override not None.
        """
        # Note: may b
        return self._use_safetensors_override if self._use_safetensors_override else self.USE_SAFETENSORS

    @use_safetensors.setter
    def use_safetensors(self, value: bool) -> None:
        """Model id setter, to _use_safetensors_override witout class muatation."""
        if value is False:
            message = "Cannot override USE_SAFETENSORS with False"
            raise ValueError(message)
        self._use_safetensors_override = value

    @property
    def classname(self) -> str:
        """Return name of class."""
        return type(self).__name__

    def _load(self) -> None:
        """Load detector and processor (if applicable), unless already loaded."""
        # Always indicate potentially unfreed
        self._is_freed = False
        # Now attemtp to load
        logger.info(f"Loading model(s) for {self.NAME}: model_id={self.model_id}")

        # reset to empty
        if self.extra_info:
            self.extra_info = {}
        if not self._processor:
            # Note: for now load guard is only applied to detector
            self._processor = self._load_processor()
        if not self._detector:
            self._detector = load_guard(
                local_files_only=self.local_files_only,
                train_mode=False,
                loader_fn=self._load_detector,
                loader_kwargs=None,
                base_export_name=f"{self.NAME}-detector",
                only_load_export=self.only_load_export,
                force_export_refresh=self.force_export_refresh,
                use_safetensors=self.USE_SAFETENSORS,
            )
        logger.info(
            f"Loaded detector for {self.model_id}: {type(self._detector)}, "
            f"dtype={getattr(self._detector, 'dtype', None)}, "
            f"device={getattr(self._detector, 'device', None)}",
        )
        if self._processor:
            logger.info(
                f"Loaded processor for {self.model_id}: {type(self._processor)}, "
                f"dtype={getattr(self._processor, 'dtype', None)}, "
                f"device={getattr(self._processor, 'device', None)}",
            )
        else:
            logger.debug("No processor loaded")

    def _free(self, *, clear_cache: bool = True, reason: str = "unspecified") -> None:
        """Free detector and processor (if applicable).

        Note: will also garbage collect and clear torch cache.
        """
        logger.debug(f"Running free for {self.classname}, reason: {reason}")

        # Don't free twice, unless models still loaded
        if self._is_freed and (not self._processor) and (not self._detector):
            logger.debug("Already freed, exiting early from _free()")
            return

        with free_guard(device_list=self.device_list, run_gc_and_clear_cache=clear_cache):
            # free detector and processor, if allocated
            if self._detector:
                self._detector = None
            if self._processor:
                self._processor = None
            # reset to empty
            if self.extra_info:
                self.extra_info = {}

        # Indicate already freed
        logger.debug("Free complete (including sync, garbage collection, and torch cache empty)")
        self._is_freed = True

    def __enter__(self) -> "OpGuardBase":
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
        # ruff: noqa: SIM117  (clarity is better)
        logger.info(f"Running inference (call) for {self.NAME}...")
        with self.lazy_loader_context():
            # Autocast for proper precision, accounting for gradient needs, also stream synchronize
            # Note: dtype and device guard are already pre-checked in init
            with call_guard(
                need_grads=self.NEED_GRADS,
                sanitize_all_exceptions=self.sanitize_all_exceptions,
                caller_fn=self._caller,
                effective_dtype=self.dtype,
                device_list=self.device_list,
                models=(self._processor, self._detector),
                train_mode=False,
            ) as guarded_call:
                # Note: cache_guard will handle detaching outptu and tracebacks
                return guarded_call(input_raw=input_raw, **kwargs)

    @property
    def detector(self) -> Any:
        """Retrieve model, lazy loading if needed.

        Note: Will also detach (to cpu) all outputs to avoid dangling references tying up VRAM
        """
        with self.lazy_loader_context():
            return self._detector

    @contextmanager
    def lazy_loader_context(self) -> Generator[None, None, None]:
        """Lazy loader context manager.

        For calling _load() and _free() in lazy mode based on keep_warm or _in_context.
        Useful for lazy caller and model property getters.
        """
        try:
            # Lazy loader
            if self._is_freed:
                logger.debug(f"Lazy loading {self.NAME} on call")
                self._load()
            yield None
        finally:
            # Agressively free up VRAM if desired
            # Ensure free as expected even on error
            if (not self.keep_warm) and (not self._in_context):
                self._free(reason=f"per-call finally, {self.keep_warm=}, {self._in_context=}")
                logger.debug(f"Lazy freeing {self.NAME} due to {self.keep_warm=}, {self._in_context=}")
