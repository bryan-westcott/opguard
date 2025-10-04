"""CUDA guards and utilities for PyTorch.

This module provides composable context managers and helpers to run CUDA work
safely and predictably.

What this fixes:
- Tying up VRAM on past calculations, especially exceptions and Jupyter notebooks
    - detatch outputs, without copying and even for nested output
    - detach tracebacks on exceptions wihtout losing error messages (strip and reraise)
    - always run garbage collect and cache clean, even on exception (try/finally)
- Difficulties in using reduced precision types:
    - autocast for stable mixed precision
    - handle runtime-determined, hardware-dependent bfloat16 support
    - automatically determine the best supported reduced precision dtype
- Forgetting to use no-gradient mode:
    - assume no-grad unless otherwise specified
    - use more efficient inference_mode, when applicable
- Headaches with various device and device_map specifications
    - normalize all devices to a torch.device
    - convert device_map to a deterministic list
- Synchronization issues
    - synchronize calls without external stream management
    - ensure predictable exceptions, logging, destructors, GC and cache clear
- Forgetting logging
    - all choices including input preferences and overrides are logged
- Remembering to do all of this
    - single convenience guard: cuda_gurad

Composite context manager:
- model_guard: compose device/dtype/grad/AMP/cache guards, with optional multi-GPU support.

Composable context managers:
- device_guard: resolve a deterministic device list (from device or device_map).
- dtype_guard: validate requested dtype (bf16→fp16 fallback if needed).
- variant_guard: choose the Hub download variant (e.g., "fp16") from dtype/repo contents.
- eval_guard: recursively set .eval() or .train() mode on all models provided.
- grad_guard: toggle torch.set_grad_enabled / torch.inference_mode.
- autocast_guard: scope torch.autocast for the chosen device/dtype to reduce compute/memory.
- vram_guard: Prevent VRAM from getting pinned by stale tensors:
                sync streams, detach/move outputs off-GPU as needed, run GC,
                and torch.cuda.empty_cache(); coalesce/surface CUDA errors deterministically.

Utility functions:
- Check support for bfloat16 (based on compute capability)
- Detach CUDA tracebacks (avoiding tying up VRAM on exceptions)
- Device and device_map handling (normalization)
"""

from __future__ import annotations

import ast
import gc
import hashlib
import inspect
import json
import os
import re
import shutil
import textwrap
import traceback
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager, suppress
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable

import torch
from diffusers import DiffusionPipeline
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import LocalEntryNotFoundError
from loguru import logger
from transformers import AutoModel

# ---------- type aliases (Py 3.11) ----------
DeviceLike: TypeAlias = int | str | torch.device
DeviceMapLike: TypeAlias = str | Mapping[str, DeviceLike]

# for detach function
DetachFn = Callable[[object], object]


# ---------- CUDA-ish error detection / scrubbing ----------
"""
Message pattern to search exceptions for evicence of CUDA.
Note: used for avoiding VRAM leaks due to tracebacks.
"""
_CUDA_MESSAGE_PATTERN = re.compile(
    r"(cuda|cublas|cudnn|cufft|cusolver|nccl|device-?side assert|"
    r"illegal memory access|out of memory|no kernel image|invalid device)",
    re.IGNORECASE,
)


def is_cuda_error(exc: BaseException) -> bool:
    """Heuristically determine whether an exception likely originated from CUDA.

    Args:
        exc:
            The exception to inspect.

    Returns:
        bool:
            True if the exception is a known CUDA error, else False.

    Logic:
        - Returns True for `torch.cuda.OutOfMemoryError`.
        - Returns True for `RuntimeError` whose message contains common CUDA /
          library substrings (CUDA/cuBLAS/cuDNN/NCCL, OOM, illegal access, etc.).
        - Returns False otherwise.

    Notes:
        - Many CUDA failures in PyTorch surface as plain `RuntimeError` with
          CUDA-ish text; this helper is designed to catch those.
        - This check is conservative; it may produce false negatives if vendors
          change wording in error messages.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and bool(_CUDA_MESSAGE_PATTERN.search(str(exc)))


def detach_exception_tracebacks(exc: BaseException, *, deep: bool = True) -> None:
    """Detach/clear traceback frames to avoid retaining CUDA refs via frame locals.

    Args:
        exc:
            The exception whose traceback should be detached/cleared.
        deep:
            If True (default), also detaches/clears the traceback chains reachable
            via `exc.__context__` and `exc.__cause__` recursively.

    Returns:
        None. The function mutates `exc` (and its chained exceptions) in place.

    Behavior:
        - On Python 3.11+, uses `traceback.clear_frames(tb)` to drop references
          held by frame locals; on older versions this call is a harmless no-op.
        - Sets `exc.__traceback__ = None` to detach the traceback object itself.
        - When `deep=True`, repeats the process for `__context__` and `__cause__`.

    Why:
        - In Jupyter/long-lived processes, traceback frames keep their locals,
          which often include `self` → modules → CUDA tensors; that can pin VRAM
          even after you think you've freed it. Scrubbing frames breaks that chain.

    Safety:
        - Safe to call multiple times.
        - Intended for use alongside your error sanitization path (e.g., when
          `is_cuda_error(exc)` is True), prior to GC and cache clearing.
    """
    if exc is None:
        return

    # Clear this exception's traceback frames (Py 3.11+) and detach
    tb = exc.__traceback__
    with suppress(Exception):
        if tb is not None:
            traceback.clear_frames(tb)  # Py 3.11+; harmless if not supported
    with suppress(Exception):
        exc.__traceback__ = None

    if not deep:
        return

    # Recursively clear chained exceptions (they can also pin frames)
    # First from context
    ctx: BaseException | None = getattr(exc, "__context__", None)
    if ctx is not None:
        with suppress(Exception):
            tb_ctx = ctx.__traceback__
            if tb_ctx is not None:
                traceback.clear_frames(tb_ctx)
            ctx.__traceback__ = None
        # Avoid infinite loops on self-referential chains
        if ctx is not exc:
            detach_exception_tracebacks(ctx, deep=True)

    # Then from context
    cause: BaseException | None = getattr(exc, "__cause__", None)
    if cause is not None:
        with suppress(Exception):
            tb_cause = cause.__traceback__
            if tb_cause is not None:
                traceback.clear_frames(tb_cause)
            cause.__traceback__ = None
        # Avoid infinite loops on self-referential chains
        if cause is not exc:
            detach_exception_tracebacks(cause, deep=True)


# ---------- device normalization / bf16 support ----------

# Minimum compute capability for bfloat16 support
MIN_BFLOAT16_COMPUTE_CAPABILITY_MAJOR = 8  # Ampere+


def normalize_device(d: DeviceLike) -> torch.device:
    """Normalize a device spec into a `torch.device('cuda:N')`.

    Args:
        d:
            Device spec. Accepted forms:
            - `torch.device('cuda:N')`
            - integer CUDA index (e.g., `0` → `'cuda:0'`)
            - string `'cuda'` (uses current CUDA device if set, else index 0)
            - string `'cuda:N'`

    Returns:
        torch.device: A normalized CUDA device.

    Notes:
        - If `d == 'cuda'`, this resolves to `torch.cuda.current_device()` when
          available, otherwise `'cuda:0'`.
        - This function does **not** validate CUDA availability beyond the
          `torch.device` construction itself; callers should check
          `torch.cuda.is_available()` when needed.
    """
    if isinstance(d, torch.device):
        return d
    if isinstance(d, int):
        return torch.device(f"cuda:{d}")
    s = str(d)
    if s == "cuda":
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(f"cuda:{idx}")
    return torch.device(s)


def device_supports_bfloat16(device: DeviceLike) -> bool:
    """Check whether a CUDA device supports bfloat16 compute.

    Args:
        device:
            Device spec accepted by `normalize_device` (e.g., 0, 'cuda:1',
            `torch.device('cuda:0')`).

    Returns:
        bool: True if the device supports bf16 math, else False.

    Logic:
        1) CUDA must be available.
        2) Device must be CUDA.
        3) Compute capability major ≥ `MIN_BFLOAT16_COMPUTE_CAPABILITY_MAJOR`
           (Ampere+ by default).
        4) If available, `torch.cuda.is_bf16_supported()` scoped to the device
           must return True. If the probe raises, we conservatively return False.

    Notes:
        - This is a **capability** check, not a performance guarantee.
        - Returns False instead of raising for non-CUDA devices or when probing
          fails, so callers can safely use it in feature detection code.
    """
    # If no cuda, no bf16 support
    if not torch.cuda.is_available():
        return False
    # If not a cuda device, no bf16 support
    dev = normalize_device(device)
    if dev.type != "cuda":
        return False
    # Compute capability of the device is another major indicator
    major, _ = torch.cuda.get_device_capability(dev)
    if major < MIN_BFLOAT16_COMPUTE_CAPABILITY_MAJOR:
        return False
    # Query directly
    try:
        # Some builds expose torch.cuda.is_bf16_supported(); call it per-device
        with torch.cuda.device(dev):
            fn = getattr(torch.cuda, "is_bf16_supported", None)
            return bool(fn()) if callable(fn) else True
    except (RuntimeError, AssertionError):
        # If querying fails, be conservative.
        return False


# ----------- detach outputs and cleanup --------------------


def to_cpu_detached(x: object) -> object:
    """Move tensors to CPU and detach them; preserve container structure.

    Args:
        x:
            A tensor or a (possibly nested) Python container of tensors:
            - `torch.Tensor`
            - `dict[str, Any]`
            - `list[Any]`
            - `tuple[Any, ...]`
            Non-tensor leaves are returned unchanged.

    Returns:
        Same type as `x`:
            - For tensors: a CPU tensor with `requires_grad=False` (via
              `.detach().to('cpu')`).
            - For containers: the same container type with tensors converted as
              above, recursively.

    Notes:
        - Detaching drops autograd references to help release GPU memory.
        - `.to('cpu')` may synchronize the producing stream for that tensor.
        - Container reconstruction preserves the original Python type
          (`dict`, `list`, `tuple`); tuple-ness is maintained.
        - This does **not** deep-copy non-tensor leaves.
        - For very large structures, consider applying this near the boundary
          where results cross process/thread/notebook cells to avoid retaining
          CUDA references.

    Attributes that must be overridden:
        -
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


def make_guarded_call(
    call: Callable[..., object],
    wrapper: Callable[[object], object],
) -> Callable:
    """Return a callable that runs `call(*args, **kwargs)` and then applies `wrapper`.

    Args:
        call:
            The original function to wrap. Its signature/metadata are preserved
            via `functools.wraps`.
        wrapper:
            A post-processing function applied to the result of `call`.
            This can be identity (no-op) or a transformation (e.g., move/convert/
            sanitize outputs).

    Returns:
        A new function with the same signature as `call`. Each invocation:
        1) computes `out = call(*args, **kwargs)`
        2) returns `wrapper(out)`.

    Notes:
        - This does not modify `call` in place; it returns a new callable.
        - If `wrapper` is an identity function, the behavior is effectively
          the same as calling `call` directly.
        - Useful with contexts that manage resources around the call while
          normalizing/transforming outputs (e.g., CPU moves, detaching, casting).
    """

    @wraps(call)
    def wrapped(*args: object, **kwargs: object) -> object:
        out = call(*args, **kwargs)
        return wrapper(out)  # type: ignore[return-value]

    return wrapped


def sync_gc_and_cache_cleanup(
    *,
    device_list: list[torch.device],
    do_sync: bool = True,
    do_garbage_collect: bool = True,
    do_empty_cache: bool = True,
    suppress_errors: bool = True,
) -> None:
    """
    Reusable cleanup: sync across devices, garbage collect, clear cache.

    Notes:
    * intended to be run as a best-effort on final cleanup, suppressing errors
    * it is best to sync within a try block *without* suppression, as that are were some errors surface
    * device_list required for sync, even if not used to avoid potentially raising exception in exception
    """
    logger.debug(f"cleanup: {device_list=}, {do_sync=}, {do_garbage_collect=}, {do_empty_cache=}, {suppress_errors=}")
    # Best-effort drain; never clobber the primary error.
    # Note: synchronize all devices
    if do_sync:
        for dev in device_list:
            if suppress_errors:
                with suppress(Exception):
                    torch.cuda.synchronize(dev)
            else:
                torch.cuda.synchronize(dev)

    # Garbage collect before cache empty, but after sync
    if do_garbage_collect:
        if suppress_errors:
            with suppress(Exception):
                gc.collect()
        else:
            gc.collect()

    # Cache empty
    if do_empty_cache:
        for dev in device_list:
            with torch.cuda.device(dev):
                if suppress_errors:
                    with suppress(Exception):
                        torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()


# ---------- helpers for setting .train() attr ----------


def _call_attr_recursive(x: object | None, attr: str, *args: object, **kwargs: object) -> None:
    """Recursively call x.<attr>(*args, **kwargs) on leaves inside dict/list/tuple/set.

    - If x is a Mapping: recurse into values, then return.
    - If x is a list/tuple/set: recurse into items, then return.
    - Else: if x has a callable attribute `attr`, call it strictly (no fallbacks).
      If it doesn't, silently ignore.
    """
    if x is None:
        return
    if isinstance(x, Mapping):
        for v in x.values():
            _call_attr_recursive(v, attr, *args, **kwargs)
        return
    if isinstance(x, (list, tuple, set)):
        for v in x:
            _call_attr_recursive(v, attr, *args, **kwargs)
        return
    fn = getattr(x, attr, None)
    if callable(fn):
        fn(*args, **kwargs)


# ---------- helpers for cached models ----------


def hf_hub_cache_dir() -> Path:
    """Return the absolute path to the Hugging Face Hub cache directory.

    Resolution order (first match wins):
      1. ``HF_HUB_CACHE`` — absolute cache path for Hub repos.
      2. ``HF_HOME``      — base directory; cache lives under ``HF_HOME / "hub"``.
      3. ``XDG_CACHE_HOME`` (non-Windows) — use ``XDG_CACHE_HOME / "huggingface" / "hub"``.
      4. Default          — ``~/.cache/huggingface/hub``.

    The resulting path is expanded (``~``) and normalized via ``Path.resolve()``.

    Returns
    -------
    pathlib.Path
        The resolved cache directory path.

    Notes
    -----
    - This mirrors the precedence used by ``huggingface_hub`` so tools that
      compute paths relative to the Hub cache (e.g., a sibling ``exports`` tree)
      stay consistent with user configuration.
    - This function does not create the directory; it only computes its location.
    - On Windows, ``XDG_CACHE_HOME`` is uncommon; the default branch will place
      the cache under the user's home directory.
    """
    if "HF_HUB_CACHE" in os.environ:
        cache_dir = Path(os.environ["HF_HUB_CACHE"]).expanduser()
    elif "HF_HOME" in os.environ:
        cache_dir = Path(os.environ["HF_HOME"]).expanduser() / "hub"
    elif "XDG_CACHE_HOME" in os.environ:
        cache_dir = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    else:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_dir.resolve()


def _loader_fingerprint(fn: Callable[..., object]) -> str:
    """Compute a stable-ish fingerprint for a loader function.

    The fingerprint prefers a hash of the function's **AST** (source parsed with
    locations stripped) to increase stability across edits like reformatting.
    If source is unavailable (e.g., builtins/C-extensions/dynamic functions),
    it falls back to hashing the function's code object representation. If AST
    parsing fails, the normalized source text is hashed instead.

    Strategy (first successful step is used)
    ---------------------------------------
    1) ``inspect.getsource(fn)`` → normalize with ``textwrap.dedent`` →
       ``ast.parse`` → SHA-256 of ``ast.dump(..., include_attributes=False)``.
    2) If AST parsing fails: SHA-256 of the normalized source text.
    3) If source retrieval fails: SHA-256 of ``repr(fn.__code__)`` (or ``repr(None)``).

    Parameters
    ----------
    fn : Callable[..., object]
        The function to fingerprint.

    Returns
    -------
    str
        A hex SHA-256 digest representing the function's implementation.

    Limitations
    -----------
    - This captures the function *definition* only; it does **not** include
      runtime state such as globals, environment variables, or files on disk.
      If your loader's behavior depends on such inputs, include them in a
      separate signature layer (e.g., hash of ``args``/``kwargs``/config).
    - Bytecode and AST formats can vary across Python versions; the AST path is
      usually more stable, but fingerprints are not guaranteed to be cross-version
      identical in all cases.
    - For nested/closure-heavy functions, the fallback code-object hash may change
      if outer scopes change, even if the inner function body is the same.
    """
    try:
        # get and normalize source (builtins/C-extensions may not have it)
        src = inspect.getsource(fn)
        src_normalized = textwrap.dedent(src)
        try:
            # parse AST; on failure, hash the normalized source text
            tree = ast.parse(src_normalized)
            dump = ast.dump(tree, annotate_fields=True, include_attributes=False)
            base_str = dump
        except (SyntaxError, ValueError):
            base_str = src_normalized
    except (OSError, TypeError):
        # source not available / builtins / C-extensions / dynamic objects
        co = getattr(fn, "__code__", None)  # note: __code__, not "__coded__"
        base_str = repr(co)

    return hashlib.sha256(base_str.encode("utf-8")).hexdigest()


# ---------- guards you can compose ----------


def device_guard(
    *,
    device: DeviceLike | None,
    device_map: DeviceMapLike | None,
    device_list_override: list[torch.device] | None = None,
) -> list[torch.device]:
    """Resolve a deterministic list of CUDA devices from `device` and/or `device_map`.

    Args:
        device:
            A single CUDA device spec (e.g., 0, "cuda:1",
            torch.device("cuda:0")). When provided together with `device_map`,
            this device will be included in the result.
        device_map:
            Placement hint for multi-GPU workloads.
            - "auto": include **all visible CUDA devices** in index order.
            - None  : no multi-device expansion is performed.
            Other mapping forms are not interpreted here.
        device_list_override:
            manual setting of device_list, e.g., from prior call to this context manager

    Returns:
        list[torch.device]: Normalized CUDA devices to use. If both `device`
        and `device_map="auto"` are provided, the result is the **union** of the
        explicit device and all visible devices (de-duplicated).

    Notes:
        - Raises if CUDA is unavailable or if neither `device` nor `device_map`
          is provided.
        - Only CUDA devices are returned; non-CUDA placements are rejected here.
    """
    if device_list_override:
        # Manual override, e.g., from prior call to this manager
        logger.debug(f"Using {device_list_override=}")
        return device_list_override
    if not torch.cuda.is_available():
        message = "CUDA is not available."
        raise RuntimeError(message)
    if device_map is not None:
        if device_map != "auto":
            message = "Only device_map='auto' or None is supported."
            raise ValueError(message)
        n = torch.cuda.device_count()
        if n == 0:
            message = "No CUDA devices visible."
            raise RuntimeError(message)
        device_list = [torch.device(f"cuda:{i}") for i in range(n)]
        logger.debug(f"Setting {device_list=} from {device_map=}")
        return device_list
    if device is not None:
        dev = normalize_device(device)
        if dev.type != "cuda":
            message = f"Expected a CUDA device, got {dev!s}."
            raise RuntimeError(message)
        device_list = [dev]
        logger.debug(f"Setting {device_list=} from {device=}")
        return device_list
    message = "Must specify `device` and/or `device_map`."
    raise ValueError(message)


def dtype_guard(
    *,
    device_list: list[torch.device],
    dtype_desired: torch.dtype,
    dtype_override: torch.dtype | None = None,
) -> torch.dtype:
    """Validate the requested dtype against the selected CUDA devices and select best.

    Args:
        device_list:
            Normalized CUDA devices to be used for computation (e.g.,
            [torch.device('cuda:0'), torch.device('cuda:1')]).
        dtype_desired:
            The preferred compute dtype to try first.
        dtype_override:
            Manually set dtype, e.g., from previous call to this context manager

    Returns:
        torch.dtype: The effective dtype, selected by preference order
        **bfloat16 → float16 → float32** subject to device support:
        - If `dtype_desired` is `torch.bfloat16`, returns bf16 only if **all**
          devices in `device_list` support bf16; otherwise falls back to fp16.
        - If `dtype_desired` is `torch.float16`, returns fp16 (requires CUDA).
        - If `dtype_desired` is `torch.float32`, returns fp32 (CPU allowed).

    Notes:
        - Reduced-precision dtypes (bf16/fp16) require CUDA. When CUDA is not
          available, only `torch.float32` is permitted.
        - bf16 support is determined per device via `device_supports_bfloat16`.
        - The selection is deterministic given `device_list`.

    Raises:
        RuntimeError: If a reduced dtype is requested but CUDA is unavailable.
        ValueError: If `dtype_desired` is not one of {bf16, fp16, fp32}.
    """
    if dtype_override:
        # Manual override, e.g., from prior call to this manager
        logger.debug(f"Using {dtype_override=}")
        return dtype_override
    if dtype_desired is torch.float32:
        dtype = torch.float32
    elif not torch.cuda.is_available():
        message = "Reduced dtypes require CUDA; only float32 on CPU."
        raise RuntimeError(message)
    elif dtype_desired is torch.float16:
        dtype = torch.float16
    elif dtype_desired is torch.bfloat16:
        all_bf16 = all(device_supports_bfloat16(d) for d in device_list)
        dtype = torch.bfloat16 if all_bf16 else torch.float16
    else:
        message = f"Unsupported dtype requested: {dtype_desired!r}"
        raise ValueError(message)
    logger.debug(f"Setting {dtype=} from {dtype_desired=}, for {device_list=}")
    return dtype


def variant_guard(
    *,
    dtype: torch.dtype | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    local_files_only: bool = True,
    variant_override: str | None = None,
) -> str:
    """Decide which Hugging Face Hub *variant* to request for a model download.

    This helper returns a `variant` string suitable for passing to
    `huggingface_hub.snapshot_download(..., variant=...)` based on:
      1) an explicit `variant_override`, or
      2) the requested tensor `dtype` and the repository's files.

    Behavior
    --------
    - If `variant_override` is provided (non-None), it is returned as-is.
    - If `dtype` is `torch.float32`, returns the empty string (no variant).
    - If `dtype` is `torch.float16` or `torch.bfloat16`, the function queries
      the repo file list (optionally at `revision`) and selects `"fp16"` **iff**
      any filename contains `"fp16"` or `"float16"` (e.g., `*.fp16.safetensors`,
      `*float16.bin`, etc.). Otherwise returns the empty string.
    - Any other `dtype` raises `ValueError`.

    Notes
    -----
    - Returning the empty string means “do not set a variant” when calling
      `snapshot_download`; callers may choose to omit the `variant` kwarg in
      that case.
    - This is a filename-heuristic only. It does **not** guarantee that the
      entire repo is organized by formal Hub variants. It simply detects common
      half-precision naming patterns and maps them to `variant="fp16"`.
    - `revision` (branch, tag, or commit) is forwarded to the file listing to
      avoid scanning the default branch unintentionally.

    Parameters
    ----------
    dtype : torch.dtype | None
        Desired tensor precision (e.g., `torch.float32`, `torch.float16`,
        `torch.bfloat16`). Required unless `variant_override` is provided.
    model_id : str | None
        Hub repository ID (e.g., `"org/model"`). Required unless
        `variant_override` is provided.
    revision : str | None
        Optional Hub revision to inspect (branch, tag, or commit SHA).
    local_files_only: bool
        Only look in local HF cache (default), unless False
    variant_override : str | None
        If provided, short-circuits all logic and is returned verbatim
        (useful for caching a previously chosen variant).

    Returns
    -------
    str
        `"fp16"` if a half-precision variant appears available (per heuristic);
        otherwise the empty string.

    Raises
    ------
    ValueError
        If `variant_override` is None and either `dtype` or `model_id` is missing,
        or if `dtype` is not one of {`torch.float32`, `torch.float16`, `torch.bfloat16`}.

    Examples
    --------
    >>> variant = variant_guard(dtype=torch.float16, model_id="bigscience/bloom")
    >>> kwargs = {} if not variant else {"variant": variant}
    >>> local_dir = snapshot_download(
    ...     repo_id="bigscience/bloom",
    ...     revision="main",
    ...     **kwargs,
    ... )

    >>> # Explicit override (skips dtype/model inspection):
    >>> variant = variant_guard(variant_override="int8")
    >>> snapshot_download("org/model", variant=variant)
    """
    no_variant: str = ""
    if variant_override is not None:
        # Manual override, e.g., from prior call to this manager
        logger.debug(f"Using {variant_override=}")
        variant = variant_override
    # Check other args
    elif (not dtype) or (not model_id):
        message = "Arguments dtype and model_id must be provided if variant_override is None"
        raise ValueError(message)
    elif dtype == torch.float32:
        # if 32-bit, return no variant
        variant = no_variant
    elif dtype in (torch.float16, torch.bfloat16):
        # otherwise, half precision requested
        # Prefer offline probe when requested
        if local_files_only:
            try:
                # Probe whether an fp16 variant is cached locally
                # Note: with local_files_only==True it does NOT download
                #       it instead returns a string
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    allow_patterns=["*fp16*.safetensors", "*float16*.safetensors"],
                    local_files_only=True,
                )
                has_fp16 = True
            except LocalEntryNotFoundError:
                has_fp16 = False
                return no_variant
        else:
            # The very first time the model is cached, we must poll HF Hub
            # Online heuristic: look for common fp16/float16 filenames
            api = HfApi()
            files: Iterable[str] = api.list_repo_files(repo_id=model_id, revision=revision)
            has_fp16 = any(("fp16" in f) or ("float16" in f) for f in files)
        variant = "fp16" if has_fp16 else no_variant
    else:
        message = f"Invalid dtype provided: {dtype=}"
        raise ValueError(message)
    return variant


@contextmanager
def local_guard(*, local_files_only: bool = True) -> Generator[None, None, None]:
    """Temporarily toggle Hugging Face *offline* mode via the HF_HUB_OFFLINE env var.

    Use this guard around code that calls ``from_pretrained`` (or other Hub helpers)
    when you want to **prevent unintended network access** and rely strictly on the
    local cache.

    Parameters
    ----------
    local_files_only : bool, default True
        If True, sets ``HF_HUB_OFFLINE=1`` inside the context. If False, removes
        the variable (allowing online access) for the duration.

    Yields
    ------
    None
        Context scope with the desired offline/online setting applied.

    Notes
    -----
    - Some loaders accept ``local_files_only`` directly; this guard is useful for
      libraries/utilities that *ignore* that kwarg but still respect the env var.
    - The previous value of ``HF_HUB_OFFLINE`` (if any) is restored on exit.
    """
    old = os.environ.get("HF_HUB_OFFLINE")
    try:
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        yield
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old


def load_guard(
    *,
    model_id: str,
    loader_fn: Callable[..., Any] | None = None,
    loader_args: tuple[Any, ...] | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    only_export: bool = True,
    force_refresh: bool = False,
) -> object:
    """Load a locally **exported** model by name, or (optionally) refresh that export.

    This helper provides a simple, local-only cache for *assembled* models (e.g.,
    Diffusers pipelines with adapters/LoRAs, or precision-converted weights).
    Each model lives under an ``exports`` directory parallel to the standard
    Hugging Face cache (e.g., ``~/.cache/huggingface/exports/...``).
    Cache validity is determined by a **signature** over:
    - the loader function's fingerprint (source/bytecode),
    - the loader's positional/keyword arguments (excluding ``local_files_only``).

    If a matching export exists, it is loaded directly. Otherwise—when allowed—
    the export is rebuilt via ``loader_fn`` and re-saved alongside a new
    ``metadata.json`` containing the signature.

    Parameters
    ----------
    model_id : str
        A **local** identifier for the assembled model. This does not need to
        correspond to a Hub repo_id and is intended to avoid collisions with
        third-party names.
    loader_fn : Callable[..., Any] | None, optional
        Function that builds and returns an HF object supporting
        ``save_pretrained``. Required when creating or refreshing the export.
    loader_args : tuple[Any, ...] | None, optional
        Positional arguments passed to ``loader_fn`` when building/refreshing.
    loader_kwargs : dict[str, Any] | None, optional
        Keyword arguments passed to ``loader_fn`` when building/refreshing.
        If present, ``variant`` (for Diffusers) is also used on load/save.
        Any ``local_files_only`` entry is ignored for signature purposes.
    only_export : bool, default True
        If True, do **not** build/refresh; raise if the export is missing or stale.
        If False, missing/stale exports will be rebuilt using ``loader_fn``.
    force_refresh : bool, default False
        Rebuild and overwrite the export regardless of the stored signature.
        ``loader_fn`` must be provided.

    Returns
    -------
    object
        The loaded (or newly built) HF object. For Diffusers, this is typically a
        ``DiffusionPipeline``; for Transformers, an ``AutoModel`` (or subclass).

    Raises
    ------
    FileNotFoundError
        When ``only_export=True`` and no valid export exists.
    ValueError
        When a refresh is needed but ``loader_fn`` is not provided; or when the
        export does not look like a supported format (neither Diffusers nor Transformers).
    huggingface_hub.utils.LocalEntryNotFoundError
        Raised with a stricter message when an export is required but absent.

    Notes
    -----
    - The export directory for ``model_id`` is computed under
      ``hf_hub_cache_dir().parent / "exports"`` with a sanitized folder name.
    - For Diffusers exports, the optional ``variant`` in ``loader_kwargs`` is
      forwarded on load/save (e.g., ``"fp16"``). Most Transformers models do not
      use ``variant``.
    - This function does **not** manage online/offline behavior; wrap your calls
      in ``local_guard(...)`` if you need to forbid network access.
    - The stored signature is minimal by design and does *not* include external
      state (e.g., environment, files on disk). If your loader depends on such
      state, include it explicitly in ``loader_args``/``loader_kwargs``.
    """
    # ruff: noqa: PLR0913

    loader_args = loader_args or ()
    loader_kwargs = loader_kwargs or {}

    # work in directory parallel to hub cache dir called exports
    # typically .../huggingface/hub -> ../huggingface/exports
    export_dir = hf_hub_cache_dir().parent / "exports" / repo_folder_name(repo_id=model_id, repo_type="model")

    # get variant from loader_kwargs
    variant = loader_kwargs.get("variant", "")

    if loader_fn:
        # since local_files_only mayb be in loader_kwargs, remove it here before checking metadata
        # We may be wrapping a more generic loader
        # But, don't mutate caller
        loader_kwargs_filtered = {k: v for k, v in loader_kwargs.items() if k != "local_files_only"}

        # expected metadata in export (if exists and fresh)
        metadata_expected = {
            "loader_fn_fingerprint": _loader_fingerprint(loader_fn),
            "loader_args": loader_args,
            "loader_kwargs_filtered": loader_kwargs_filtered,
        }
        signature_expected = hashlib.sha256(
            json.dumps(metadata_expected, sort_keys=True, default=str).encode(),
        ).hexdigest()

        # check for existing export
        metadata_path = export_dir / "metadata.json"
        metadata_export = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

        # detect a match
        match = (metadata_export is not None) and (metadata_export["signature"] == signature_expected)
    else:
        match = None

    if not match or force_refresh:
        # Either it doesn't match or we want to force a refresh
        if only_export and not force_refresh:
            # Raise exception if strict mode, unless force_refresh
            message = f"exported model not found in cache {export_dir=}"
            raise LocalEntryNotFoundError(message)
        if not loader_fn:
            message = "cannot refresh if no loader_fn provided"
            raise ValueError(message)
        # load model from huggingface hub, use original with local_files_only
        model = loader_fn(str(export_dir), *loader_args, **loader_kwargs)
        # Now ensure directory exists and is empty
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        # Write to *export* cache, being sure to (1) use variant, (2) to also pass expected_metadata
        save_kwargs = {"variant": variant} if isinstance(model, DiffusionPipeline) else {}
        model.save_pretrained(export_dir, save_kwargs=save_kwargs)
        (export_dir / "metadata.json").write_text(
            json.dumps({"signature": signature_expected}, indent=2, sort_keys=True),
        )

    elif (export_dir / "model_index.json").exists():
        model = DiffusionPipeline.from_pretrained(str(export_dir), variant=variant or None)
    elif (export_dir / "config.json").exists():
        model = AutoModel.from_pretrained(str(export_dir))
    else:
        message = "Unable to determine if huggingface transformers or diffusers"
        raise ValueError(message)
    return model


@contextmanager
def grad_guard(*, need_grads: bool) -> Generator[None, None, None]:
    """Set PyTorch grad mode for the guarded block.

    Args:
        need_grads:
            If True, enables gradient recording via
            `torch.set_grad_enabled(True)`. If False, wraps the block in
            `torch.inference_mode()` to disable autograd and reduce memory.

    Yields:
        None: This is a context manager; it yields control to the guarded block.

    Notes:
        - `inference_mode()` is more efficient than `no_grad()` because it also
          freezes version counters and avoids autograd book-keeping.
    """
    logger.debug(f"Context for grad_guard: {need_grads=}")
    ctx = torch.set_grad_enabled(True) if need_grads else torch.inference_mode()
    with ctx:
        yield


@contextmanager
def autocast_guard(
    *,
    dtype: torch.dtype,
    enabled_override: bool | None = None,
) -> Generator[None, None, None]:
    """Enable CUDA autocast for the guarded block.

    Args:
        dtype:
            Desired compute dtype for autocast (e.g., `torch.bfloat16`,
            `torch.float16`, or `torch.float32`). When `torch.float32`, AMP is
            disabled by default.
        enabled_override:
            If provided, forces autocast on/off regardless of `dtype`. Use this
            to explicitly disable AMP for code paths that are not numerically
            stable in reduced precision.

    Yields:
        None: This is a context manager; it yields control to the guarded block.

    Notes:
        - Autocast device type is fixed to `"cuda"`. Ensure CUDA is available
          and the active device supports the requested dtype.
        - By default, AMP is enabled for any dtype other than float32.
    """
    enabled = (dtype is not torch.float32) if enabled_override is None else enabled_override
    logger.debug(f"Context for autocast_guard: {enabled=}, {dtype=}, {enabled_override=}")
    with torch.autocast("cuda", dtype=dtype, enabled=enabled):
        yield


@contextmanager
def vram_guard(
    *,
    device_list: list[torch.device],
    sanitize_cuda_errors: bool = True,
    detach_outputs: bool = False,
    call: Callable[..., object] | None = None,
) -> Iterator[Callable]:
    """Sync/sanitize/cleanup wrapper across all provided CUDA devices.

    Features:
    * applies to_cpu/detach for all outputs
        - a deepcopy is problematic for memory use and synchronization
    * synchronizes (and waits on) all devices used and sanitze/re-throw exceptions
    * memory cleanup at the end: garbage collection and torch cache clear (in proper order)
    * handles exceptions gracefully

    Yields:
        if call function provided:
            a local-scope version of call wrapped in detach
        else:
            a callable `detach(x)` that applies `to_cpu_detached(x)` when
            `detach_outputs=True`, else returns `x` unchanged. Use like:

    Usage:
        with vram_guard(device_list=device_list, detach_outputs=True) as detach:
            out = detach(run_model(...))
    """
    try:
        if call is not None:
            # apply the wrapper
            if detach_outputs:
                yield make_guarded_call(call, to_cpu_detached)
            else:
                yield call
        else:
            # return the wrapper itself
            yield to_cpu_detached
        # First sync: surface latent CUDA faults here (propagate if not sanitized).
        # Note: do not suppress, there is one more best effort sync in finally
        for dev in device_list:
            torch.cuda.synchronize(dev)
    except Exception as e:
        # Sanitize and re-raise exception
        logger.debug("Exception inside vram_guard guarded execution")
        if sanitize_cuda_errors and is_cuda_error(e):
            # avoid references in traceback tying up VRAM
            detach_exception_tracebacks(e, deep=True)
            message = "CUDA error during guarded region"
            raise RuntimeError(message) from None
        raise
    finally:
        logger.debug("Performing vram_guard cleanup")
        sync_gc_and_cache_cleanup(
            do_sync=True,
            do_garbage_collect=True,
            do_empty_cache=True,
            suppress_errors=True,
            device_list=device_list,
        )


def eval_guard(models: object, *, train_mode: bool = False) -> None:
    """Recursively set evaluation mode on all models contained in `models`.

    - Calls `train(False)` (equivalent to `eval()`) on any object that defines it.
    - Traverses common containers (dict, list, tuple, set) and applies in place.
    - Does *not* affect gradient computation; pair with `torch.no_grad()` or your
      own grad context for inference.

    This function is one-way (no restoration). Use a context manager if you need
    to save and restore prior training modes.
    """
    # Prefer the boolean API (propagates to children) and keep it strict.
    _call_attr_recursive(models, "train", train_mode)


# ---------- composed model_guard ----------


# ruff: ignore =
@contextmanager
def model_guard(
    *,
    dtype: torch.dtype,
    need_grads: bool,
    model_id: str | None = None,
    revision: str | None = None,
    device: DeviceLike | None = None,
    device_map: DeviceMapLike | None = None,
    sanitize_cuda_errors: bool = True,
    detach_outputs: bool = True,
    call: Callable[..., object] | None = None,
    device_list_override: list[torch.device] | None = None,
    dtype_override: torch.dtype | None = None,
    variant_override: str | None = None,
    models: object | None = None,
    train_mode: bool = False,
) -> Generator[tuple[list[torch.device], torch.dtype, Callable], None, None]:
    """Compose device/dtype/grad/AMP/cache guards and yield resolved settings.

    Args:
        dtype:
            The *requested* compute dtype. Validation rules:
            - torch.float32: always allowed (no CUDA required).
            - torch.float16: requires CUDA but no capability check.
            - torch.bfloat16: requires CUDA; all selected CUDA devices must
              support bf16. If not, we fall back to fp16.
        need_grads:
            If True, enable `torch.set_grad_enabled(True)`; otherwise use
            `torch.inference_mode()` to disable autograd for performance.
        model_id:
            The huggingface hub model id
        revision:
            The huggingface hub model revision
        device:
            A single CUDA device selection (e.g., 0, "cuda:1", torch.device("cuda:0")).
            If provided together with `device_map`, both are honored: the explicit
            `device` is used for single-device operations while `device_map` may
            direct sharded frameworks (e.g., HF Accelerate) across multiple GPUs.
        device_map:
            Placement hint for multi-GPU workloads.
            - "auto": use *all visible CUDA devices* deterministically discovered.
            - None: no multi-GPU placement implied.
            Other mappings are not interpreted here (validated elsewhere).
            Can be provided together with `device` (see above).
        sanitize_cuda_errors:
            If True, CUDA-ish `RuntimeError`s are sanitized: we detach Python
            tracebacks (to avoid VRAM retention via frame locals), sync devices,
            run GC, and re-raise a concise error.
        detach_outputs:
            Setup detach function to optionally run torch detach on all outputs
        call:
            function to call the model, with outputs automatically detached
        models: (optional) provide any torch models to apply eval_guard
        train_mode: (optional) set <model>.train(train_mode) in eval_guard, if models provided
        device_list_override: (optional) previously computed device list from device_guard()
        dtype_override: (optional) previously computed device list from dtype_guard()
        variant_override: (optional) previously computed variant from variant_guard()


    Returns:
        (device_list, effective_dtype)
            device_list: normalized CUDA devices selected by `device` or
                         `device_map="auto"`.
            effective_dtype: the dtype actually used after validation and
                             fallback (bf16 → fp16 → fp32).

    Basic usage:
        >>> with model_guard(dtype=torch.bfloat16, need_grads=False, device="cuda:0") as (device_list, effective_dtype):
        ...     # run your CUDA work here; device_list is like [torch.device('cuda:0')]
        ...     out = pipe(prompt="hello")  # autocast & grad mode already set

    Multi-GPU (HF sharded) usage:
        >>> with model_guard(dtype=torch.bfloat16, need_grads=False, device_map="auto") as (
        ...     device_list,
        ...     effective_dtype,
        ... ):
        ...     # model/pipeline may shard across `device_list` items; guard syncs & sanitizes
        ...     out = pipe(prompt="hello")

    Notes:
    - If bf16 isn't supported on all selected GPUs, `effective_dtype` falls back
      to fp16 automatically.
    - The context surfaces CUDA errors at sync, sanitizes tracebacks to avoid
      VRAM being retained by frame locals, then performs GC and cache clears.
    """
    device_list: list[torch.device] = device_guard(
        device=device,
        device_map=device_map,
        device_list_override=device_list_override,
    )
    effective_dtype: torch.dtype = dtype_guard(
        device_list=device_list,
        dtype_desired=dtype,
        dtype_override=dtype_override,
    )
    variant: str = variant_guard(
        dtype=effective_dtype,
        model_id=model_id,
        revision=revision,
        variant_override=variant_override,
    )
    eval_guard(
        models=models,
        train_mode=train_mode,
    )
    with (
        grad_guard(need_grads=need_grads),
        autocast_guard(dtype=effective_dtype),
        vram_guard(
            device_list=device_list,
            detach_outputs=detach_outputs,
            sanitize_cuda_errors=sanitize_cuda_errors,
            call=call,
        ) as guarded_call,
    ):
        logger.debug(f"Context for model_guard: {device_list=}, {effective_dtype=}, {variant=}, {guarded_call=}")
        yield device_list, effective_dtype, guarded_call
