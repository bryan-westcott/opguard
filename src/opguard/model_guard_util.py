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
- model_guard: convenience of all guards, yields a guarded callable model
    - Note: this is provided for illustration purposes, while it can be
            use for fast experimentaion, the ModelGuardBase calss is still
            preferred

Aggregate context managers - used individually in classes (see ModelGuardBase)
- init_guard - aggregates: device_guard, dtype_guard, variant_guard
- load_guard - aggregates: local_guard, eval_guard, cache_guard
- call_guard - aggregates: autocast_guard, grad_guard, vram_guard
- free_guard - (not an aggregate but only step on free)

Composable context managers:
- Initialization:
    - device_guard: resolve a deterministic device list (from device or device_map).
    - dtype_guard: validate requested dtype (bf16→fp16 fallback if needed).
    - variant_guard: choose the Hub download variant (e.g., "fp16") from dtype/repo contents.
- Model Loading:
    - local_guard: ensure use of huggingface_hub is in local files only mode.
    - eval_guard: recursively set .eval() or .train() mode on all models provided.
    - cache_guard: cache previously prepared and cast models locally
- Model Calling
    - autocast_guard: scope torch.autocast for the chosen device/dtype to reduce compute/memory.
    - grad_guard: toggle torch.set_grad_enabled / torch.inference_mode.
    - vram_guard: Prevent VRAM from getting pinned by stale tensors:
                sync streams, detach/move outputs off-GPU as needed, run GC,
                and torch.cuda.empty_cache(); coalesce/surface CUDA errors deterministically.
- Clean-up
    - free_guard: ensure garbage collection and torch cache clear happen after model delete

Utility functions:
- Check support for bfloat16 (based on compute capability)
- Detach CUDA tracebacks (avoiding tying up VRAM on exceptions)
- Device and device_map handling (normalization)
"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import textwrap
import traceback
import types
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, overload

if TYPE_CHECKING:
    from collections.abc import Iterable
    from contextlib import AbstractContextManager

import torch
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import LocalEntryNotFoundError
from loguru import logger

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
    """Wrap `call` so that an invocation returns `wrapper(call(*args, **kwargs))`.

    Note: preserve metadata and bound-method semantics.

    Behavior
    --------
    - If `call` is a *bound method* (e.g., `self.fn`), the returned callable is
      re-bound to the same `self` so attribute access inside `call` still sees the
      original instance (i.e., `__self__` is preserved).
    - If `call` is a free function or other callable, a regular wrapper is returned.
    - Any decorator layers on `call` are unwrapped (`inspect.unwrap`) for accurate
      metadata and to locate the underlying function for rebinding.
    - Function metadata (`__name__`, `__qualname__`, `__doc__`, annotations, etc.)
      is preserved via `functools.wraps`.

    Args
    ----
    call:
        The original callable to wrap. May be a bound method, free function,
        or any callable. If it is a bound method, the result remains bound
        to the same instance.
    wrapper:
        A post-processing function applied to the *result* of `call`.
        It should accept exactly one argument (the output of `call`) and
        return the transformed value.

    Returns
    -------
    Callable
        A callable with the same call signature as `call`. On each call:
        1) computes `out = call(*args, **kwargs)`
        2) returns `wrapper(out)`.

    Notes
    -----
    - This does not mutate `call`; it returns a new callable.
    - Exceptions raised by `call` or by `wrapper` propagate unchanged.
    - “Preserving bound semantics” means the returned callable has a valid
      `__self__` when `call` was a bound method, which is crucial when
      downstream logic (e.g., guards) inspects or mutates instance attributes.
    - If `wrapper` is an identity/no-op, the behavior matches `call` aside from
      the additional stack frame.

    Examples
    --------
    >>> class C:
    ...     def f(self, x):
    ...         return x + 1
    >>> c = C()
    >>> g = make_guarded_call(c.f, lambda y: y + 10)
    >>> assert g(2) == (c(2) + 10)
    >>> assert getattr(g, "__self__", None) != c
    """
    orig: Callable[..., object] = inspect.unwrap(call)
    self_obj: object | None = getattr(call, "__self__", None)  # present for bound methods
    base: Callable[..., object] = getattr(call, "__func__", call)  # the original function if bound

    if self_obj is not None and inspect.isfunction(base):
        # BOUND METHOD CASE: keep descriptor semantics by defining a function
        # that takes `self` explicitly, then bind it back to `self_obj`.
        @wraps(base)
        def _wrapped_bound(self: object, *args: object, **kwargs: object) -> object:
            out = base(self, *args, **kwargs)  # <-- use the unbound function
            return wrapper(out)

        wrapped_call = types.MethodType(_wrapped_bound, self_obj)  # restores __self__
        if not hasattr(wrapped_call, "__self__"):
            message = "Self object not preserved"
            raise ValueError(message)
        if wrapped_call.__self__.__class__.__name__ != self_obj.__class__.__name__:
            message = "Wrapped caller has self and names do not match"
            raise ValueError(message)
        return wrapped_call

    # FREE FUNCTION / CALLABLE CASE: no binding to preserve
    @wraps(base if inspect.isfunction(base) else orig)
    def _wrapped_unbound(*args: object, **kwargs: object) -> object:
        out = orig(*args, **kwargs)
        return wrapper(out)

    return _wrapped_unbound


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
        co = getattr(fn, "__func__", None)  # note: __code__, not "__coded__"
        base_str = repr(co)

    return hashlib.sha256(base_str.encode("utf-8")).hexdigest()


def _normalize_arg(x: object) -> object:
    """Normalize a Python value into a **JSON-serializable, deterministic** form.

    Purpose
    -------
    Produce a stable, compact representation suitable for hashing and for storing
    a human-readable snapshot of loader arguments.

    Rules (in order)
    ----------------
    - Primitives (`str`, `int`, `float`, `bool`, `None`) are returned as-is.
    - `pathlib.Path` → absolute, expanded string path.
    - `dict` → keys coerced to `str`; values normalized recursively.
    - `list`/`tuple` → each element normalized recursively (tuples become lists).
    - PyTorch:
        * `torch.dtype` → `str(dtype)` (e.g., ``"torch.float16"``)
        * `torch.device` → `str(device)` (e.g., ``"cuda:0"``)
        * `torch.Tensor` → summary dict with shape/dtype/device/requires_grad
          (data is **not** embedded).
    - Callables/classes → summary dict with module and (qual)name.
    - Fallback → `repr(x)`.

    Parameters
    ----------
    x : Any
        Value to normalize.

    Returns
    -------
    Any
        A JSON-serializable structure (primitives, lists, dicts) that is stable
        across runs for the same semantic input.

    Notes
    -----
    - This is **one-way**; it is not intended for reconstruction of `x`.
    - Extend this function if your project uses other non-JSON types that should
      hash consistently (e.g., numpy dtypes/devices).
    """
    # ruff: noqa: PLR0911
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Path):
        return str(Path(x).expanduser().resolve())
    if isinstance(x, dict):
        return {str(k): _normalize_arg(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_normalize_arg(v) for v in x]
    # Torch dtypes/devices/tensors
    if "torch" in type(x).__module__:
        if isinstance(x, torch.dtype):
            return str(x)  # e.g., "torch.float16"
        if isinstance(x, torch.device):
            return str(x)  # e.g., "cuda:0"
        if isinstance(x, torch.Tensor):
            # Don't embed data; summarize deterministically
            return {
                "__tensor__": True,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "requires_grad": bool(x.requires_grad),
            }
    # Functions/methods/classes
    if inspect.isfunction(x) or inspect.ismethod(x) or inspect.isclass(x):
        base = getattr(x, "__func__", x)  # normalize bound methods
        return {
            "__callable__": True,
            "name": getattr(base, "__qualname__", getattr(base, "__name__", str(base))),
            "module": getattr(base, "__module__", None),
        }
    # Fallback: stable repr
    return repr(x)


def _canonicalize_call(*, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a `(args, kwargs)` call into a normalized JSON tree.

    Behavior
    --------
    - Drops non-semantic flags that should not invalidate the cache
      (currently: ``"local_files_only"``).
    - Sorts keyword arguments by key for deterministic ordering.
    - Applies :func:`_normalize_arg` recursively to all values.

    Parameters
    ----------
    args : tuple
        Positional arguments that will be passed to the loader.
    kwargs : dict
        Keyword arguments that will be passed to the loader.

    Returns
    -------
    dict
        ``{"args": [...], "kwargs": {...}}`` where both sides are normalized and
        JSON-serializable.

    Examples
    --------
    >>> _canonicalize_call((Path("~/data"),), {"variant": "fp16", "local_files_only": True})
    {'args': ['/home/user/data'],
     'kwargs': {'variant': 'fp16'}}
    """
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "local_files_only"}
    return {
        "args": [_normalize_arg(a) for a in args],
        "kwargs": {k: _normalize_arg(v) for k, v in sorted(filtered_kwargs.items())},
    }


def _hash_canonical(obj: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hex digest of a normalized JSON tree.

    Behavior
    --------
    - Serializes ``obj`` with ``json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=False)``.
    - Hashes the resulting UTF-8 bytes with SHA-256 and returns the hex digest.

    Parameters
    ----------
    obj : dict
        A JSON-serializable structure, typically the output of
        :func:`_canonicalize_call` or a metadata dict built from normalized parts.

    Returns
    -------
    str
        A 64-character lowercase hex digest.

    Notes
    -----
    - Stability depends on the **normalization** you apply before hashing; use
      :func:`_normalize_arg` to avoid incidental diffs (e.g., differing `Path`
      representations).
    - This is not salted; identical inputs always produce identical digests,
      which is desirable for cache keys.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_variant_override(*, export_name: str | None, dtype: torch.dtype, variant: str) -> tuple[str | None, str]:
    """Set variant based on dtype, even variants not in HF Hub or transformer models.

    Even though diffusers only supports variant in from_pretrained, we
         will also export variants for transformers models indicated
          via directory name and metadata

    Note: will bypass if export_name is None

    Input:
        export_name - export name without variant in it, None indicates
                      no export being used
        dtype - torch dtype
    Returns:
        export_name - appended with "-fp16" if dtype is float16 or bfloat16
        variant - export variant (even if not in HF hub or for transformers),
                  or returns variant input if export_name is None
    """
    # Ensure explictly no variant input
    if not isinstance(variant, str):
        message = "Variant must be string, use '' for no variant"
        raise TypeError(message)
    if export_name:
        # Variant in cache mode depends only on dtype, not what is in HF hub
        if dtype in (torch.float16, torch.bfloat16):
            # Half precision detected, make new variant
            variant = "fp16"
            # Variants in separate directory for maintainability
            export_name += "-" + variant
        elif variant:
            # Full precision mode, no variant needed, but check for problems
            # A half-precision variant input with full-precision dtype
            # should not happen
            message = f"Unexpected variant ({variant}) provided on non-half-precision input"
            raise ValueError(message)
    return export_name, variant


# Determine export dir, checking for issues
def _cache_export_path(*, export_name: str | None, variant: str, subdir: str = "exports") -> Path | None:
    """Compute export directory parallel to huggingface hub cache dir.

    Will produce directory parallel to hub cache dir, with name in subdir argument
    Typically:
        .../.cache/huggingface/hub/... -> .../.cache/huggingface/<subidr>/...
    """
    if not export_name:
        return None
    if variant not in export_name:
        message = f"Variant identifier '{variant}' not in {export_name=}, run _cache_variant_override first"
        raise ValueError(message)
    # export_dir
    export_dir = hf_hub_cache_dir().parent / subdir / repo_folder_name(repo_id=export_name, repo_type="model")
    return export_dir.resolve()


def _cache_signature(
    *,
    export_name: str | None,
    loader_fn: Callable[..., Any],
    loader_kwargs: dict[str, Any],
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    variant: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Compute signature and associated metadata based on laoder, name and arguments."""
    # Lack of export name prevents local export
    if not export_name:
        return None, None

    if loader_fn is None:
        message = "loader_fn must be provided as a callable"
        raise ValueError(message)

    # Which kwargs are used to detect changes
    # Note: do not copy local_files_only, that may change in this gurad
    loader_kwargs_filtered: dict[str, Any] = {}
    signature_kwargs = ("model_id", "device", "dtype", "variant")
    loader_kwargs_filtered = {k: v for k, v in loader_kwargs.items() if k in signature_kwargs}

    # Noirmalize call args/kwargs
    call_norm: dict[str, Any] = _canonicalize_call(args=(), kwargs=loader_kwargs_filtered)
    # Hash for matching
    call_hash: str = _hash_canonical(call_norm)

    # construct signature
    signature_metadata: dict[str, Any] = {
        "name": export_name,
        "loader_fn_fingerprint": _loader_fingerprint(loader_fn),
        "loader_call_hash": call_hash,
        "model_id": model_id,
        "device": str(device),
        "dtype": str(dtype),
        "variant": variant,
    }
    # compute expected signature
    signature_expected = hashlib.sha256(
        json.dumps(
            signature_metadata,
            sort_keys=True,
            default=str,
        ).encode(),
    ).hexdigest()

    # Put normalized call in for debugging, but match on signature
    signature_metadata["loader_call"] = call_norm

    return signature_expected, signature_metadata


def _cache_match(
    *,
    export_dir: Path | None,
    signature_expected: str | None,
    only_load_export: bool,
) -> bool:
    """Check for existing local export that matches expected signature."""
    # Lack of export name prevents local export
    if not export_dir:
        logger.warning("Bypassing load_guard import due to no export_name")
        if only_load_export:
            message = "Must provide export_name to enforce only_load_export==True"
            raise ValueError(message)

    # Check for existing match
    match = False
    if export_dir:
        if not signature_expected:
            message = "Cannot match prior export without signature."
            raise ValueError(message)
        # check for existing export
        metadata_path = export_dir / "metadata.json"
        prior_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

        # detect a match
        signature_found = prior_metadata.get("signature") if isinstance(prior_metadata, dict) else ""
        match = signature_expected == signature_found
        logger.debug(
            f"Export {match=} in cache_guard, signatures: "
            f"expected={signature_expected[0:7]}..., found={str(signature_found)[0:7]}...",
        )
    return match


def _cache_get_loader_overrides(
    *,
    model_id: str,
    export_dir: Path | None,
    match: bool,
    only_load_export: bool,
    force_export_refresh: bool,
    local_files_only_on_refresh: bool,
) -> dict[str, bool | str]:
    """Compute kwargs to provide to loader to use local export, if applicable.

    returns:
        model_id
    """
    overrides: dict[str, Any] = {}
    # determine which to use
    if match and not force_export_refresh:
        logger.debug(f"Importing in cache_guard due to {match=}, {force_export_refresh=}")
        # A match and also no refresh
        if not export_dir:
            message = "Must provide export_dir when loading from export."
            raise ValueError(message)
        # resolve the export directory
        logger.debug(f"Setting {model_id=} to {export_dir=}")
        overrides["model_id"] = export_dir
        logger.debug("Setting local_files_only to True")
        overrides["local_files_only"] = True
    else:
        logger.debug(f"Not importing in cache_guard due to {match=}, {force_export_refresh=}")
        # Either it doesn't match or we want to force a refresh
        if only_load_export and not force_export_refresh:
            # Raise exception if strict mode, unless force_export_refresh
            message = f"exported model not found in cache {export_dir=}"
            raise LocalEntryNotFoundError(message)
        # decision on whether to use local or remote files on refresh
        logger.debug(f"Leaving {model_id=} unchanged")
        logger.debug(f"Overriding local_files_only to {local_files_only_on_refresh=}")
        overrides["local_files_only"] = local_files_only_on_refresh
    # work with a copy, then override
    return overrides


def _cache_apply_loader_overrides(
    *,
    loader_overrides: Mapping[str, Any],
    loader_params_obj: object | None,
    loader_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply cache routing overrides to a loader.

    Overrides may be in the form of a bound method (with `self`) or a free
    function. Overrides are written to `self` when possible and to kwargs
    only if accepted by the loader (has that param or **kwargs) or already
    present.

    Returns a NEW kwargs dict to pass to `loader_fn(**new_kwargs)`.

    Note: will set in both places, provided they are in the original

    Raises:
        TypeError: if any `required_keys` present in loader_overrides cannot be
                   routed to either `self` (attribute exists) or accepted kwargs.
    """
    # Shallow copy to avoid mutating caller's dict
    new_kwargs: dict[str, Any] = dict(loader_kwargs) if (loader_kwargs is not None) else {}

    for key, value in loader_overrides.items():
        if (loader_params_obj is not None) and hasattr(loader_params_obj, key):
            setattr(loader_params_obj, key, value)
        elif (loader_kwargs is not None) and (key in loader_kwargs):
            new_kwargs[key] = value
        else:
            message = f"No target to override {key}"
            raise ValueError(message)

    return new_kwargs


def _cache_export_model(
    *,
    model: object,
    signature_metadata: dict[str, Any] | None,
    signature: str | None,
    export_dir: Path | None,
    match: bool,
    force_export_refresh: bool,
) -> None:
    """Export the model to local export, if applicable."""
    if not export_dir:
        logger.warning("Bypassing load_guard export due to no export_dir")
        if force_export_refresh:
            message = "Must provide export_name to use force_export_refresh==True"
            raise ValueError(message)

    # No attempt to re-export it
    if export_dir and (not match or force_export_refresh):
        if not signature_metadata or not signature:
            message = "Cannot export without signature metadata and signature"
            raise ValueError(message)
        # Sanity check
        model_dtype = getattr(model, "dtype", None)
        if model_dtype and (str(model_dtype) != signature_metadata["dtype"]):
            message = (
                f"Unexpected mismatch of signature dtype ({signature_metadata['dtype']}) "
                f"and model dtype ({model_dtype})"
            )
            raise ValueError(message)

        # Get time
        now = datetime.now().astimezone()
        created_at = now.isoformat(timespec="seconds")
        created_at_pretty = now.strftime("%a, %b %d, %Y %I:%M %p %Z (%z)")
        created_at_unix = int(now.timestamp())
        # Get module, model_class, base module
        module = getattr(model.__class__, "__module__", "None")
        model_class = getattr(model.__class__, "__name__", "None")
        base_module = module.split(".")[0]
        base_module_version = getattr(importlib.import_module(base_module), "__version__", "None")
        torch_version = getattr(torch, "__version__", "None")
        # Write to *export* cache, being sure to (1) use variant, (2) to also pass expected_metadata
        # Note: dtype, variant and name all in signature_metadata
        metadata = signature_metadata | {
            "signature": signature,
            "module": module,
            "model_class": model_class,
            "base_module": base_module,
            "base_module_version": base_module_version,
            "torch_version": torch_version,
            "created_at": created_at,
            "created_at_pretty": created_at_pretty,
            "created_at_unix": created_at_unix,
        }
        # Add variant if appropriate
        save_kwargs: dict[str, Any] = {}
        if (base_module == "diffusers") and ("variant" in metadata):
            save_kwargs = {"variant": metadata["variant"]}
        # Now ensure directory exists and is empty
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        # save and export metadata
        if not hasattr(model, "save_pretrained"):
            message = "Model provided has no save_pretrained method"
            raise ValueError(message)
        model.save_pretrained(export_dir, save_kwargs=save_kwargs)
        (export_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=False),
        )
        logger.debug(f"Export for cache_guard written to: {export_dir}")


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
        logger.debug(f"Setting {device_list=} from {device_map=} in device_guard")
        return device_list
    if device is not None:
        dev = normalize_device(device)
        if dev.type != "cuda":
            message = f"Expected a CUDA device, got {dev!s}."
            raise RuntimeError(message)
        device_list = [dev]
        logger.debug(f"Setting {device_list=} from {device=} in device_guard")
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
    logger.debug(f"Setting {dtype=} from {dtype_desired=}, for {device_list=} in dtype_guard")
    return dtype


def variant_guard(
    *,
    dtype: torch.dtype | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    local_hfhub_variant_check_only: bool = False,
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
    local_hfhub_variant_check_only: bool
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
        if local_hfhub_variant_check_only:
            logger.warning(
                "Checking local huggingface hub cache only for variant "
                f"due to {local_hfhub_variant_check_only=} in variant_guard",
            )
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
            logger.debug(f"Variant result {has_fp16=} in local huggingface_hub cache in variant_guard")
        else:
            logger.debug(
                f"Checking huggingfacehub for variant due to {local_hfhub_variant_check_only=}in variant_guard",
            )
            # The very first time the model is cached, we must poll HF Hub
            # Online heuristic: look for common fp16/float16 filenames
            api = HfApi()
            files: Iterable[str] = api.list_repo_files(repo_id=model_id, revision=revision)
            has_fp16 = any(("fp16" in f) or ("float16" in f) for f in files)
            logger.debug(f"Variant result {has_fp16=} from huggingface_hub in variant_guard")
        variant = "fp16" if has_fp16 else no_variant
    else:
        message = f"Invalid dtype provided: {dtype=}"
        raise ValueError(message)
    logger.debug(f"Setting {variant=} from {dtype=}")
    return variant


@contextmanager
def local_guard(*, local_files_only: bool = True) -> Generator[bool, None, None]:
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
    bool
        The `local_files_only` value in effect for this context.

    Notes
    -----
    - Some loaders accept ``local_files_only`` directly; this guard is useful for
      libraries/utilities that *ignore* that kwarg but still respect the env var.
    - The previous value of ``HF_HUB_OFFLINE`` (if any) is restored on exit.
    """
    old = os.environ.get("HF_HUB_OFFLINE")
    try:
        logger.debug(f"Setting HF_HUB_OFFLINE according to {local_files_only=} in local_guard")
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        yield local_files_only
    finally:
        logger.debug("Restoring HF_HUB_OFFILNE in local_guard exit")
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old


# Overlaods for eval_guard to keep mypy happy
@overload
def eval_guard(
    *,
    models_or_loader: None,
    train_mode: bool = ...,
) -> AbstractContextManager[Callable[[object], object]]: ...
@overload
def eval_guard(
    *,
    models_or_loader: Callable[..., Any],
    train_mode: bool = ...,
) -> AbstractContextManager[Callable[..., Any]]: ...
@overload
def eval_guard(*, models_or_loader: object, train_mode: bool = ...) -> AbstractContextManager[object]: ...


@contextmanager
def eval_guard(
    *,
    models_or_loader: object | Callable[..., Any] | None,
    train_mode: bool = False,
) -> Iterator[Any]:
    """Apply train/eval mode recursively to models or loaders.

    This context manager sets `.train(train_mode)` on a single model or on every
    model contained inside common containers (dict, list, tuple, set). It supports
    three modes depending on the input.

    Parameters
    ----------
    models_or_loader : object | Callable[..., Any] | None
        - `None`:
            Yields a **deferred setter** `Callable[[object], object]` that, when
            called with a model (or container), applies `.train(train_mode)`
            recursively and returns the same object.
        - `Callable[..., Any]` (a loader/factory):
            Yields a **wrapped loader** with the same signature. Calling it returns
            the loader's result after `.train(train_mode)` has been applied
            recursively.
        - Any other `object` (model or container of models):
            Applies `.train(train_mode)` immediately and yields the **same object**.
    train_mode : bool, optional
        If `False` (default), sets eval mode (`.train(False)`). If `True`, sets
        training mode (`.train(True)`).

    Yields
    ------
    object | Callable[..., Any] | Callable[[object], object]
        - Deferred setter: `Callable[[object], object]` when `models_or_loader is None`.
        - Wrapped loader: `Callable[..., Any]` when `models_or_loader` is callable.
        - Same object: `object` when an object was provided.

    Behavior & Notes
    ----------------
    - Traverses dict/list/tuple/set and applies `.train(train_mode)` to leaves that
      define a callable `train`.
    - One-way: this does **not** restore prior modes on exit.
    - Pairs well with `torch.no_grad()` or `torch.inference_mode()` for inference.
    - Designed to work whether your loader returns a single model or a container.

    Examples
    --------
    1) Deferred setter (no model yet; apply later)
    >>> with eval_guard(models_or_loader=None, train_mode=False) as set_eval:
    ...     set_eval(model_or_bundle)  # model_or_bundle is now in eval()

    2) Wrap a loader (apply after building)
    >>> def load_model():
    ...     return build_model()
    >>> with eval_guard(models_or_loader=load_model, train_mode=True) as guarded_load:
    ...     model = guarded_load()  # model is now in train() recursively

    3) Immediate apply (object in hand)
    >>> with eval_guard(models_or_loader=model, train_mode=False) as m:
    ...     assert m is model  # same reference; now in eval()
    """
    logger.debug(f"Applying eval_guard with {train_mode=}, {type(models_or_loader)=}, ")

    # a simple train attr setter with fixed train_mode based on eval_guard argumenmt
    def _apply_train(out: object) -> object:
        _call_attr_recursive(out, "train", train_mode)
        return out

    if models_or_loader is None:
        # return a deferred setter
        yield _apply_train
    elif callable(models_or_loader):
        # a loader was provided, provide a guarded caller
        # that applies .train(train_mode) recursively on outputs
        yield make_guarded_call(models_or_loader, _apply_train)
    else:
        # apply the .train(train_mode) in place and return the same
        # reference (which will likely be ignored)
        yield _apply_train(models_or_loader)


def cache_guard(
    *,
    loader_fn: Callable[..., Any],
    loader_params_obj: object | None = None,
    loader_kwargs: dict[str, Any] | None = None,
    export_name: str | None = None,
    only_load_export: bool = False,
    force_export_refresh: bool = False,
    local_files_only_on_refresh: bool = False,
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
    loader_fn : Callable[..., Any]
        Function that builds and returns an HF object supporting
        ``save_pretrained``.
    loader_kwargs : dict[str, Any]
        Must provide: 'model_id', 'dtype', 'device', 'variant', 'local_files_only'
        For no variant: use "" not None
        Note: if positional args needed in laoder, use them from kwargs
    export_name : str | None
        If None, then no import/export will occur (just huggingface_hub which,
        may or may not be cached locally).
        A **local** identifier for the assembled model. This does not need to
        correspond to a Hub repo_id and is intended to avoid collisions with
        third-party names. Do not include variants in this, they will be
        handled internally.
    only_load_export : bool, default True
        If True, do **not** build/refresh; raise if the export is missing or stale.
        If False, missing/stale exports will be rebuilt using ``loader_fn``.
    force_export_refresh : bool, default False
        Rebuild and overwrite the export regardless of the stored signature.
        ``loader_fn`` must be provided.
    local_files_only_on_refresh: bool, default False
        Force use of local files only for hfhub on refresh
        (exports do not exist anywhere except locally)

    Returns
    -------
    object
        The loaded (or newly built) HF object. For Diffusers, this is typically a
        ``DiffusionPipeline``; for Transformers, an ``AutoModel`` (or subclass).

    Raises
    ------
    FileNotFoundError
        When ``only_load_export=True`` and no valid export exists.
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

    Future Extensions
    -----
    - consider a fast mode and device_list ID check
    """
    # ruff: noqa: PLR0913

    # Check for params in loader self in case it is a bound method
    if (not loader_params_obj and not loader_kwargs) and hasattr(loader_fn, "__self__"):
        logger.debug("Bound method for loader_fn detected, using loader.__self__ for loader_params_obj")
        # Detect bound `self` (None if free function)
        loader_params_obj = loader_fn.__self__
    # Only one source of truth here
    if (loader_params_obj is None) == (loader_kwargs is None):
        message = "Excatly one of two loader_params_obj and loader_kwargs must be non None"
        raise ValueError(message)
    # Ensure non-None now
    loader_kwargs = loader_kwargs or {}

    if (model_id := getattr(loader_params_obj, "model_id", None) or loader_kwargs.get("model_id", None)) is None:
        message = "No source for model_id"
        raise ValueError(message)
    if (device := loader_params_obj.device if loader_params_obj else loader_kwargs.get("device", None)) is None:
        message = "No source for device"
        raise ValueError(message)
    if (dtype := loader_params_obj.dtype if loader_params_obj else loader_kwargs.get("dtype", None)) is None:
        message = "No source for dtype"
        raise ValueError(message)
    # WARNING: if variant is "", then getattr(obj, variant, None) or None will return None!
    if (variant := loader_params_obj.variant if loader_params_obj else loader_kwargs.get("variant")) is None:
        message = "No source for variant"
        raise ValueError(message)

    # get variant and variant-specific export name
    export_name, variant = _cache_variant_override(
        export_name=export_name,
        dtype=dtype,
        variant=variant,
    )
    # path for export
    export_dir = _cache_export_path(export_name=export_name, variant=variant)
    # compute signature and associated metadata used to craete it
    signature_expected, signature_metadata = _cache_signature(
        export_name=export_name,
        loader_fn=loader_fn,
        loader_kwargs=loader_kwargs,
        model_id=model_id,
        device=device,
        dtype=dtype,
        variant=variant,
    )
    # check for existing export
    match = _cache_match(
        export_dir=export_dir,
        signature_expected=signature_expected,
        only_load_export=only_load_export,
    )
    # override kwargs according to arguments
    loader_overrides = _cache_get_loader_overrides(
        model_id=model_id,
        export_dir=export_dir,
        match=match,
        only_load_export=only_load_export,
        force_export_refresh=force_export_refresh,
        local_files_only_on_refresh=local_files_only_on_refresh,
    )

    loader_kwargs = _cache_apply_loader_overrides(
        loader_params_obj=loader_params_obj,
        loader_kwargs=loader_kwargs,
        loader_overrides=loader_overrides,
    )

    # Load the model with same loader either way
    model = loader_fn(**loader_kwargs)
    # Export the model, if applicable
    _cache_export_model(
        model=model,
        signature_metadata=signature_metadata,
        signature=signature_expected,
        export_dir=export_dir,
        match=match,
        force_export_refresh=force_export_refresh,
    )
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
    caller_fn: Callable[..., object] | None = None,
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
        if caller_fn is not None:
            logger.debug(f"Guarding {caller_fn=} with {detach_outputs=} in vram_guard")
            # apply the wrapper
            if detach_outputs:
                yield make_guarded_call(caller_fn, to_cpu_detached)
            else:
                yield caller_fn
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


@contextmanager
def free_guard(*, device_list: list[torch.device], run_gc_and_clear_cache: bool = True) -> Generator[None, None, None]:
    """Ensure garbage collectiona and cache clear happen after model free."""
    # Note: models freed here
    yield
    # Apply GC and cache clear as we just freed up the models
    # Note: suppress errors as this is best-effort VRAM freeup
    #       and we already handled RuntimeErrors inside model_guard
    # Note: must sync and gc to get proper cache clear
    if run_gc_and_clear_cache:
        logger.debug("Applying garbage collection and torch cache clear in free_guard")
        sync_gc_and_cache_cleanup(
            do_sync=True,
            do_garbage_collect=True,
            do_empty_cache=True,
            suppress_errors=True,
            device_list=device_list,
        )
    else:
        logger.warning("Skippnig garbage collection and cache_clear in free_guard")


# ---------- aggregates for init, load, call ----------


def init_guard(
    *,
    dtype: torch.dtype,
    device: DeviceLike | None,
    device_map: DeviceMapLike | None,
    model_id: str | None = None,
    revision: str | None = None,
    local_hfhub_variant_check_only: bool = False,
    device_list_override: list[torch.device] | None = None,
    dtype_override: torch.dtype | None = None,
    variant_override: str | None = None,
) -> tuple[list[str], torch.dtype, str]:
    """Aggregate context manager for init: device_guard, dtype_guard, variant_guard."""
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
        local_hfhub_variant_check_only=local_hfhub_variant_check_only,
    )
    return device_list, effective_dtype, variant


def load_guard(
    *,
    loader_fn: Callable[..., Any],
    local_files_only: bool = False,
    train_mode: bool = False,
    loader_kwargs: dict[str, Any] | None = None,
    export_name: str | None = None,
    only_load_export: bool = False,
    force_export_refresh: bool = False,
) -> object:
    """Aggregate context manager for model load: local_guard, eval_guard, cache_guard."""
    with (
        # extra protection for local files
        local_guard(local_files_only=local_files_only) as _local_files_only,
        eval_guard(models_or_loader=loader_fn, train_mode=train_mode) as guarded_loader_fn,
    ):
        return cache_guard(
            loader_fn=guarded_loader_fn,
            loader_kwargs=loader_kwargs,
            export_name=export_name,
            only_load_export=only_load_export,
            force_export_refresh=force_export_refresh,
            local_files_only_on_refresh=_local_files_only,  # respect local_guard
        )


@contextmanager
def call_guard(
    *,
    need_grads: bool,
    sanitize_cuda_errors: bool = True,
    detach_outputs: bool = True,
    caller_fn: Callable[..., object] | None = None,
    effective_dtype: torch.dtype,
    device_list: list[torch.device],
    models: object | None = None,
    train_mode: bool = False,
) -> Generator[Callable, None, None]:
    """Aggregate context manager for inference.

    Includes: eval_guard, grad_guard, autocast_guard, vram_guard
    """
    with (
        eval_guard(
            models_or_loader=models,
            train_mode=train_mode,
        ) as _,
        grad_guard(need_grads=need_grads),
        autocast_guard(dtype=effective_dtype),
        vram_guard(
            device_list=device_list,
            detach_outputs=detach_outputs,
            sanitize_cuda_errors=sanitize_cuda_errors,
            caller_fn=caller_fn,
        ) as guarded_call,
    ):
        yield guarded_call


# ---------- composed model_guard ----------


@contextmanager
def model_guard(
    *,
    init_guard_kwargs: dict[str, Any],
    load_guard_kwargs: dict[str, Any],
    call_guard_kwargs: dict[str, Any],
    free_guard_kwargs: dict[str, Any],
) -> Generator[Callable, None, None]:
    """Provide guarded caller using all model_guard guards (convenience function).

    Inputs:
        init_gurad_kwargs:
            device, device_map, dtype, model_id, revision,
            local_hfhub_variant_check_only, device_list_override,
            dtype_override, variant_override
        load_guard_kwargs:
            local_files_only, train_mode, loader_fn,
            export_name, only_load_export, force_export_refresh,
        call _gurad_kwargs:
            need_grads, sanitize_cuda_errors, caller_fn, train_mode
        free_guard_kwargs:
            run_gc_and_clear_cache
    """
    try:
        # detect device-specific settings
        device_list, effective_dtype, variant = init_guard(**init_guard_kwargs)

        # safely load the model
        loader_kwargs = {
            "model_id": init_guard_kwargs["model_id"],
            "revision": init_guard_kwargs["revision"],
            "device": init_guard_kwargs["device"],
            "dtype": effective_dtype,
            "variant": variant,
            "local_files_only": init_guard_kwargs["local_files_only"],
        }
        model = load_guard(loader_kwargs=loader_kwargs, **load_guard_kwargs)

        # yield guarded caller
        with call_guard(
            effective_dtype=effective_dtype,
            device_list=device_list,
            models=model,
            train_mode=load_guard_kwargs["train_mode"],
            **call_guard_kwargs,
        ) as guarded_caller:
            yield guarded_caller

    finally:
        # Free at the end
        with free_guard(
            device_list=device_list,
            **free_guard_kwargs,
        ):
            # explicitly delete it so GC/cache-clear can run
            del model
