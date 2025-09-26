"""Abastract base class to facilitate Inference with VRAM management."""


# ruff: noqa: ANN401  (this is an abstract base class)
# ruff: noqa: ANN003 (this is an abstract base class)
# ruff: noqa: ARG002 (this is an abstract base class) TODO: try to avoid unused kwargs

# mypy: disable-error-code=override
#   Why: this module specializes pipeline methods with narrower/extra kwargs.
#   We don't use polymorphic calls through the base, so substitutability doesn't apply.
#   Note: this must be in all derived classe to avoid contravariance errors

import copy
import gc
import sys
from abc import ABC, abstractmethod
from typing import Any

import torch
from loguru import logger


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
        - automatic deep copy of outputs avoids safely avoid dangling refrences
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

    # Subclasses must overrdie
    NAME: str | None = None
    MODEL_ID: str | None = None

    def __init__(
        self,
        *,
        device: str | None = "cuda",
        variant: str | None = "fp16",
        dtype: torch.dtype | None = torch.float16,
        keep_warm: bool = False,
    ) -> None:
        """Pre-load models (optionally) and prepare for inference.

        Input
        device: the torch device, typically "cuda"
        variant: the precisione variant, typically "fp16"
        dtype: the torch dtype, typically torch.blfloat16 or torch.float16
        keep_warm: whether to keep the model ready
            - if true: will eager load and not free upon __call__
            - if false: will lazy load and free after each __call__

        Models loaded are set in _load_detector and _load_preprocessor.
        """
        logger.debug(f"Initialising {type(self).__name__}, {device=}, {variant=}, {dtype=}, {keep_warm=}")
        self.device = device
        self.variant = variant
        self.dtype = dtype
        self.keep_warm = keep_warm
        # Set this if in context manager to avoid freeing each iteration
        self._in_context: bool = False
        # Whether shutdown free has completed (regardless of keep_warm option)
        self._is_freed: bool = False
        # For extra configuration information to keep track of
        self.extra_info: dict[str, Any] = {}

        # Default for lazy loading
        self._processor: Any = None
        self._detector: Any = None
        # prepare detector now
        if self.keep_warm:
            self.load()

    @property
    def name(self) -> str:
        """The user-defined name identifier for this inference method."""
        name = type(self).NAME
        if not name:
            message = f"{type(self).__name__} must define a non-empty class attribute NAME"
            raise TypeError(message)
        return name

    @property
    def model_id(self) -> str:
        """The huggingace hub model identifier for the primary model used."""
        model_id = type(self).MODEL_ID
        if not model_id:
            message = f"{type(self).__name__} must define a non-empty class attribute MODEL_ID"
            raise TypeError(message)
        return model_id

    def load(self) -> None:
        """Load detector and processor (if applicable), unless already loaded."""
        # Always indicate potentially unfreed
        self._is_freed = False
        # Now attemtp to load

        if not self._processor:
            self._processor = self._load_processor()
        logger.debug("Loaded processor:", self._processor)
        if not self._detector:
            self._detector = self._load_detector()
        logger.debug("Loaded detector:", self._processor)

    def free(self) -> None:
        """Free detector and processor (if applicable).

        Note: will also garbage collect and clear torch cache.
        """
        logger.debug("Running free")

        # Don't free twice, unless models still loaded
        if self._is_freed and (not self._processor) and (not self._detector):
            logger.debug("Already freed, exiting")
            return

        # free detector and processor, if allocated
        if self._detector:
            self._detector = None
        if self._processor:
            self._processor = None

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
        logger.debug("Free complete")
        self._is_freed = True

    def __del__(self) -> None:
        """On object delete, all the free method, handlnig exceptions."""
        self.free()

    def __call__(self, *, input_raw: Any, **kwargs) -> Any:
        """Call the model, with optional lazy loading and/or agressive free if needed.

        Note: Will also deep copy all outputs to avoid dangling references tying up VRAM
        """
        try:
            # Lazy loader (idempotent)
            self.load()
            # Ensure no references
            with torch.no_grad():
                return copy.deepcopy(self._caller(input_raw=input_raw, **kwargs))
        finally:
            # Agressively free up VRAM if desired
            # Ensure free as expected even on error
            if not self.keep_warm and not self._in_context:
                self.free()

    def __enter__(self) -> "InferenceBase":
        """Support for use as context manager, avoiding agressive per-call freeing."""
        self._in_context = True
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, this is standard format
        """Free on exit of context always."""
        self._in_context = False
        self.free()

    # --- Specialize Here -----

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
        return None

    def _preprocess(self, *, input_raw: Any, **kwargs) -> Any:
        """Preprocessing, default is no-op, override for custom behavior."""
        return input_raw  # default: input_proc = input_raw

    def _predict(self, *, input_proc: Any, **kwargs) -> Any:
        """Call the detector, default is a simple call, may be overridden."""
        if not self._detector:
            message = f"{type(self).__name__} must set _detector via _load_detector"
            raise ValueError(message)
        return self._detector(input_proc, **kwargs)

    def _postprocess(self, *, output_raw: Any, **kwargs) -> Any:
        """Postprocessing, default is no-op, override for custom behavior."""
        return output_raw  # default: output_proc = output_raw
