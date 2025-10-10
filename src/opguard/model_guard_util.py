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

Utility functions:
- Check support for bfloat16 (based on compute capability)
- Detach CUDA tracebacks (avoiding tying up VRAM on exceptions)
- Device and device_map handling (normalization)
"""

# ruff: noqa: TD002, TD003, FIX002

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
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterable

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
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        yield local_files_only
    finally:
        if old is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old


def _cache_signature(
    *,
    export_name: str | None,
    loader_fn: Callable[..., Any],
    loader_kwargs: dict[str, Any],
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
    for kwarg in signature_kwargs:
        if (kwarg not in loader_kwargs) or (loader_kwargs[kwarg] is None):
            message = "Missing or None for kwarg: " + kwarg + " in loader_kwargs, use '' for no variant"
            raise ValueError(message)
        loader_kwargs_filtered[kwarg] = loader_kwargs[kwarg]

    # Noirmalize call args/kwargs
    call_norm: dict[str, Any] = _canonicalize_call(args=(), kwargs=loader_kwargs_filtered)
    # Hash for matching
    call_hash: str = _hash_canonical(call_norm)

    # construct signature
    signature_metadata: dict[str, Any] = {
        "name": export_name,
        "loader_fn_fingerprint": _loader_fingerprint(loader_fn),
        "loader_call_hash": call_hash,
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
    signature_metadata["dtype"] = str(loader_kwargs_filtered["dtype"])
    signature_metadata["variant"] = loader_kwargs_filtered["variant"]

    return signature_expected, signature_metadata


def _cache_match(
    *,
    export_dir: Path,
    export_name: str | None,
    signature_expected: str | None,
    only_export: bool,
) -> bool:
    """Check for existing local export that matches expected signature."""
    # Lack of export name prevents local export
    if not export_name:
        logger.warning("Bypassing load_guard import due to no export_name")
        if only_export:
            message = "Must provide export_name to enforce only_export==True"
            raise ValueError(message)

    # Check for existing match
    match = False
    if export_name:
        if not signature_expected:
            message = "Cannot match prior export without signature."
            raise ValueError(message)
        # check for existing export
        metadata_path = export_dir / "metadata.json"
        prior_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None

        # detect a match
        match = (prior_metadata is not None) and (prior_metadata["signature"] == signature_expected)

    return match


def _cache_import_kwargs(
    *,
    loader_kwargs: dict[str, Any],
    export_dir: Path | None,
    match: bool,
    only_export: bool,
    force_refresh: bool,
) -> dict[str, Any]:
    """Compute kwargs to provide to loader to use local export, if applicable."""
    # work with a copy
    loader_kwargs = {**loader_kwargs}
    # determine which to use
    if match and not force_refresh:
        if not export_dir:
            message = "Must provide export_dir when loading from export."
            raise ValueError(message)
        # provide the export dir in place of huggingface hub model id
        loader_kwargs["model_id"] = str(export_dir.resolve())
        loader_kwargs["local_files_only"] = True
    # Either it doesn't match or we want to force a refresh
    elif only_export and not force_refresh:
        # Raise exception if strict mode, unless force_refresh
        message = f"exported model not found in cache {export_dir=}"
        raise LocalEntryNotFoundError(message)
    return loader_kwargs


def _cache_export(
    *,
    model: object,
    signature_metadata: dict[str, Any] | None,
    signature: str | None,
    export_dir: Path,
    export_name: str | None,
    match: bool,
    force_refresh: bool,
) -> None:
    """Export the model to local export, if applicable."""
    if not export_name:
        logger.warning("Bypassing load_guard export due to no export_name")
        if force_refresh:
            message = "Must provide export_name to use force_refresh==True"
            raise ValueError(message)

    # No attempt to re-export it
    if export_name and (not match or force_refresh):
        if not signature_metadata or not signature:
            message = "Cannot export without signature metadata and signature"
            raise ValueError(message)
        # Sanity check
        model_dtype = getattr(model, "dtype", None)
        if model_dtype and str(model_dtype) != signature_metadata["dtype"]:
            message = "Unexpected mismatch of signature dtype and model dtype"
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


def cache_guard(
    *,
    loader_fn: Callable[..., Any],
    loader_kwargs: dict[str, Any],
    export_name: str | None = None,
    only_export: bool = False,
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
        third-party names.
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

    # TODO: log module and model_type in metadata
    # TODO: ensure eval_guard is applied
    # TODO: make a loader_guard
    # TODO: export everythign to fast_loader function or function args
    # TODO: check GPU ids for all device_list for match!

    loader_kwargs = loader_kwargs or {}

    # Choose a variant.
    # Even though diffusers only supports variant in from_pretrained, we
    #      will still export specialized versions here.
    if export_name:
        # Variant in cache mode depends only on dtype, not what is in HF hub
        if loader_kwargs["dtype"] in (torch.float16, torch.bfloat16):
            loader_kwargs["variant"] = "fp16"
            export_name += "-" + loader_kwargs["variant"]
        else:
            loader_kwargs["variant"] = ""

    # work in directory parallel to hub cache dir called exports
    #   typically: .../huggingface/hub -> ../huggingface/exports
    export_dir = hf_hub_cache_dir().parent / "exports" / repo_folder_name(repo_id=export_name, repo_type="model")
    # compute signature and associated metadata used to craete it
    signature_expected, signature_metadata = _cache_signature(
        export_name=export_name,
        loader_fn=loader_fn,
        loader_kwargs=loader_kwargs,
    )
    # check for existing export
    match = _cache_match(
        export_name=export_name,
        export_dir=export_dir,
        signature_expected=signature_expected,
        only_export=only_export,
    )
    # Load model, using local export if applicable
    override_loader_kwargs = _cache_import_kwargs(
        loader_kwargs=loader_kwargs,
        export_dir=export_dir,
        match=match,
        only_export=only_export,
        force_refresh=force_refresh,
    )
    # Load the model with same loader either way
    model = loader_fn(**override_loader_kwargs)
    # Export the model, if applicable
    _cache_export(
        model=model,
        signature_metadata=signature_metadata,
        signature=signature_expected,
        export_name=export_name,
        export_dir=export_dir,
        match=match,
        force_refresh=force_refresh,
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


@contextmanager
def eval_guard(models: object, *, train_mode: bool = False) -> Generator[object, None, None]:
    """Set train/eval mode across `models`, yield the same reference, no exit restore.

    This helper supports two common use cases:

    1) Load-time chaining (one-shot set + assignment)
       Use when constructing a model and you want it left in eval() for reuse.
       Because this guard does not restore, the mode remains in effect after the
       `with` block.

       Example
       -------
       with eval_guard(call_guard(loader_fn, ...)) as m:
           self.model = m

       # or bind directly (attribute target is valid in `as`):
       with eval_guard(call_guard(loader_fn, ...)) as self.model:
           pass  # model is now in eval() and kept that way

    2) Call-time wrapping (apply mode before a call)
       Use around a single inference call. Since there is no restoration, the
       model stays in the chosen mode after the block. Pair with
       `torch.inference_mode()` or `torch.no_grad()` as needed.

       Example
       -------
       with eval_guard(self.model):             # sets eval()
           with torch.inference_mode():
               out = self.model(inputs)

       # For training loops, request train() explicitly:
       with eval_guard(self.model, train_mode=True):
           loss = self.model(batch)  # stays in train() afterwards

    Parameters
    ----------
    models : object
        A single model or a nested structure (dict/list/tuple/set) containing
        objects that implement `.train(bool)`. `torch.nn.Module` trees are handled.
    train_mode : bool, default False
        If False, applies `train(False)` (equivalent to `eval()`).
        If True, applies `train(True)`.

    Yields
    ------
    object
        The same `models` reference for convenient assignment/chaining.

    Notes
    -----
    - This guard is intentionally one-way: it does **not** restore prior modes.
      If you need restoration, implement a separate context manager.
    - This does not affect autograd; wrap inference in `torch.inference_mode()`
      or `torch.no_grad()` for best performance and safety.
    """
    # Prefer the boolean API (propagates to children) and keep it strict.
    _call_attr_recursive(models, "train", train_mode)
    yield models  # no restore, no cleanup


# ---------- composed model_guard ----------

# TODO: make this a call_guard


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
    with (
        eval_guard(
            models=models,
            train_mode=train_mode,
        ) as _,
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
