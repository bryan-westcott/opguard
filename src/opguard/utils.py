"""Abastract base class to facilitate Inference with VRAM management."""


# ruff: noqa: ANN401  (this is an abstract base class)
# ruff: noqa: ANN003 (this is an abstract base class)
# ruff: noqa: ARG002 (this is an abstract base class) TODO: try to avoid unused kwargs

# mypy: disable-error-code=override
#   Why: this module specializes pipeline methods with narrower/extra kwargs.
#   We don't use polymorphic calls through the base, so substitutability doesn't apply.
#   Note: this must be in all derived classe to avoid contravariance errors

import gc
import sys
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar, TypeAlias

import torch
from huggingface_hub import HfApi
from loguru import logger

# Minimum compute capability for bfloat16 support
MIN_BFLOAT16_COMPUTE_CAPABILITY = 8

# For specifying (non-normalized) torch devices
DeviceLike: TypeAlias = int | str | torch.device
# For specifying HF device_map
DeviceMapLike: TypeAlias = str | Mapping[str, DeviceLike]


def normalize_device(device: DeviceLike) -> torch.device:
    """Normalize the various methods of specifying device to torch.device."""
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        return torch.device("cuda:0" if device == "cuda" else device)
    if isinstance(device, torch.device):
        return device
    message = f"Not a DeviceLike: {device!r}"
    raise TypeError(message)


def bf16_consensus() -> bool:
    """Check if all visible GPUs support bf16, False if none do, else raise on mixed."""
    if not torch.cuda.is_available():
        return False
    flags = []
    for i in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(i)[0]
        with torch.cuda.device(i):
            rt_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        flags.append(major >= MIN_BFLOAT16_COMPUTE_CAPABILITY and rt_ok)
    if not flags:
        return False
    if all(flags):
        return True
    if any(flags):
        message = "Mixed bfloat16 support, unable to autodetermine dtype"
        raise ValueError(message)
    return False


def has_fp16_variant(*, repo_id: str, revision: str | None = None) -> bool:
    """Heuristic: check for common fp16-variant filenames used by Diffusers/Transformers."""
    api = HfApi()
    files: Iterable[str] = api.list_repo_files(repo_id=repo_id, revision=revision)
    # Look for fp16.safetensors, float16.safetensors, fp16.bin or float16.bin suffixes in filename
    return any(("fp16" in f) or ("float16" in f) for f in files)


def dtype_selector(*, dtype_preference: torch.dtype) -> torch.dtype:
    """Select dytpe based on what is available and by preference.

    Preference order: bfloat16 -> float16 -> float32
    """
    if (not torch.cuda.is_available()) or (dtype_preference == torch.float32):
        return torch.float32
    # Assume half preceision (16-bit types) are preferred and available
    if (dtype_preference == torch.bfloat16) and bf16_consensus():
        return torch.bfloat16
    # Fall back to float16
    return torch.float16


def variant_selector(*, dtype: torch.dtype, repo_id: str, revision: str) -> str | None:
    """Select fp16 based on availability and preference."""
    if dtype == torch.float32:
        return None
    if has_fp16_variant(repo_id=repo_id, revision=revision):
        return "fp16"
    return None


@contextmanager
def cuda_guard(
    *,
    device: DeviceLike,
    dtype: torch.dtype,
    need_grads: bool,
    stream_in: torch.cuda.Stream | None = None,
) -> Generator:
    """CUDA-only guard for device scope, stream, autocast, and grad/inference mode.

    - Errors if CUDA is unavailable or device is not CUDA.
    - Creates a private stream if `stream is None`, and synchronizes it on exit.
    - Disables autocast for float32; enables for other dtypes.
    """
    if not torch.cuda.is_available():
        message = "amp_run requires CUDA, but torch.cuda.is_not_available()."
        raise RuntimeError(message)

    # Handle device
    device_normalized: torch.device = torch.device(device)
    if device_normalized.type != "cuda":
        message = f"amp_run requires a CUDA device, got {device_normalized!s}."
        raise RuntimeError(message)
    # Ensure proper device
    device_ctx = torch.cuda.device(device_normalized)

    # Create a stream if not provided
    stream: torch.cuda.Stream = stream_in or torch.cuda.Stream(device=device_normalized)
    # create the context
    stream_ctx = torch.cuda.stream(stream)

    # ensure either grad_enabled or inference_mode
    grad_ctx = torch.set_grad_enabled(True) if need_grads else torch.inference_mode()
    # autocast based on dtype preferences
    amp_ctx = torch.autocast("cuda", dtype=dtype, enabled=(dtype is not torch.float32))

    # Build the contexts we always want in CUDA

    with device_ctx, grad_ctx, stream_ctx, amp_ctx:
        # Contexts are active across the caller's `with amp_run(...):` body.
        yield stream

    # block only on streams created in this context
    if stream_in is None:
        stream.synchronize()


def to_cpu_detached(x: Any) -> Any:
    """Ensure no outputs stay on the gpu.

    Works on nested dict/list/tuple structures of tensors
    """
    if isinstance(x, torch.Tensor):
        # detach to drop autograd refs; move to CPU (forces sync for that tensor)
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: to_cpu_detached(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_cpu_detached(v) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    return x


class InferenceBase(ABC):
    """An inference wrapper to manage VRAM-expensive inference pipelines.

    Specialization:
    - Specialize _detector_loader() to construct and assign self._detector
    - Optionally specialize preprocessor, postprocessor and call behavior

    Benefis:
    - Unified interface for idiosyncratic models
        - even HF models vary in interfaces, including precision, casting,
          and device handling
        - highly customizable to support virtually any pipe (BLIP, Diffusion, '
          Inversion, Controlnets, etc.)
        - minimal specialization required if not necessary (only the predictor
          loader must be specialized)
    - Avoid VRAM memory leaks
        - automatic detachment avoids safely avoid dangling refrences
        - automatic synchronization to avoid dangling references due to async
        - automatic delete, garbage collection and torch cache free on exit or
          exception (in proper order) with try/cach/finally blocks
        - especially useful in notebook mode
        - in proper order even upon exceptions or within try/catch block
    - Flexible memory management
        - Optional eager/lazy/ephemeral/warm model loading options (including
          context manager)
        - Automatic in torch no_grad
        - Automtic autocast handling (TODO)
    - Flexible checkpoint handling
        - Storage and fast loading of casted models (TODO)

    Modes:
      1) keep_warm=False (default): load on demand, free after each call
      2) keep_warm=True: stay loaded across calls
      3) Context manager: with Inference(...) as inf: ...  (auto-load/free)

    TODO:
      1) include autocast
      2) include write/read from checkpoints after cast
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
        device_override: DeviceLike | None = None,
        device_map_override: DeviceMapLike | None = None,
        keep_warm: bool = False,
    ) -> None:
        """Pre-load models (optionally) and prepare for inference.

        Attributes:
            NAME (str): Human-friendly name for the model
            MODEL_ID (str): Huggingface ID of the model
            REVISION (str): Huggingface revision (git hash or branch like "main")
            NEED_GRADS: whether gradients are needed
                - Note: usually false except for techniques like inversion
            DEFAULT_DEVICE (str | int | torch.device): default device to be used (default is "cuda")
            DEFAULT_DEVICE_MAP: default device mapping (for models that use it, default is "auto")
            DTYPE_PREFERENCE: preference for smallest precision
                preference order: torch.bfloat16 -> torch.float16 -> torch.float32
                Note: this is device (compute capability) dependent
                Note: if bfloat16 causes issues, override with wider supported float16


        Input:
            device_override - override the default DEFAULT_DEVICE
            device_map_override - override the default DEFAULT_DEVICE_MAP
            keep_warm: whether to keep the model ready
                - if true: will eager load and not free upon __call__
                - if false: will lazy load and free after each __call__

        Models loaded are set in _load_detector and _load_preprocessor.
        """
        self.device: DeviceLike = device_override or type(self).DEFAULT_DEVICE or device_override
        self.device_map: DeviceMapLike = device_map_override or type(self).DEFAULT_DEVICE_MAP
        self.dtype_preference: torch.dtype = type(self).DTYPE_PREFERENCE
        self.dtype = dtype_selector(dtype_preference=self.dtype_preference)
        self.variant = variant_selector(dtype=self.dtype, repo_id=self.model_id, revision=self.revision)
        logger.debug(f"Choices for {type(self).__name__}: {self.dtype=}, {self.variant=}")
        self.keep_warm = keep_warm

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
            self.load()

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

    def load(self) -> None:
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

    def free(self, reason: str = "unspecified") -> None:
        """Free detector and processor (if applicable).

        Note: will also garbage collect and clear torch cache.
        """
        logger.debug(f"Running free for {type(self).__name__}, reason: {reason}")

        # Don't free twice, unless models still loaded
        if self._is_freed and (not self._processor) and (not self._detector):
            logger.debug("Already freed, exiting early from free()")
            return

        # free detector and processor, if allocated
        if self._detector:
            self._detector = None
        if self._processor:
            self._processor = None
        # reset to empty
        if self.extra_info:
            self.extra_info = {}

        # NOTE: garbage collect BEFORE torch empty_cache!
        gc.collect()

        # Avoid import at shutdown; use sys.modules
        torch = sys.modules.get("torch")
        if torch:
            # Empty torch cache here
            cuda = getattr(torch, "cuda", None)
            if cuda and hasattr(cuda, "is_available") and cuda.is_available():
                empty_cache = getattr(cuda, "empty_cache", None)
                if callable(empty_cache):
                    empty_cache()  # let errors propagate here

        # Indicate already freed
        logger.debug("Free complete (including garbage collection and torch cache empty)")
        self._is_freed = True

    def __del__(self) -> None:
        """On object delete, all the free method, handlnig exceptions."""
        self.free(reason="destructor")

    def __call__(self, *, input_raw: Any, **kwargs) -> Any:
        """Call the model, with optional lazy loading and/or agressive free if needed.

        Note: Will also detach (to cpu) all outputs to avoid dangling references tying up VRAM
        """
        # Private stream to keep track of for synchronization
        try:
            # Lazy loader
            if self._is_freed:
                logger.debug("Lazy loading models on call")
                self.load()
            # Autocast for proper precision, accounting for gradient needs, also stream synchronize
            with cuda_guard(device=self.device, dtype=self.dtype, need_grads=self.need_grads):
                # Run ensuring no references to models on the GPU
                # Note: a deepcopy here could have unintended consequences for memory use and synchronization
                return to_cpu_detached(self._caller(input_raw=input_raw, **kwargs))
        finally:
            # Agressively free up VRAM if desired
            # Ensure free as expected even on error
            if (not self.keep_warm) and (not self._in_context):
                self.free(reason=f"per-call finally, {self.keep_warm=}, {self._in_context=}")

    def __enter__(self) -> "InferenceBase":
        """Support for use as context manager, avoiding agressive per-call freeing."""
        self._in_context = True
        logger.debug("Pre-loading models on enter due to context manager")
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, this is standard format
        """Free on exit of context always."""
        self._in_context = False
        self.free(reason="Context manager exit")
