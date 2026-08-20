"""Step 12 TDD: the lifecycle test matrix (COS-11).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_lifecycle.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

from opguard.tests import lifecycle


def test_bare_first_call() -> None:
    """Fresh guard loads on the first bare call."""
    lifecycle.bare_first_call()


def test_keep_warm_reuse() -> None:
    """keep_warm guard loads once and stays warm."""
    lifecycle.keep_warm_reuse()


def test_context_manager_roundtrip() -> None:
    """Context manager loads once and frees on exit."""
    lifecycle.context_manager_roundtrip()


def test_call_after_exit() -> None:
    """Guard reloads lazily after its context exits."""
    lifecycle.call_after_exit()


def test_enter_failure_leaves_no_partial_state() -> None:
    """Failing __enter__ frees partial state and resets flags."""
    lifecycle.enter_failure_leaves_no_partial_state()


def test_keep_warm_load_failure_leaves_no_partial_state() -> None:
    """Failed keep_warm construction load frees partial state."""
    lifecycle.keep_warm_load_failure_leaves_no_partial_state()


def test_double_free_idempotent() -> None:
    """Double free is a safe no-op."""
    lifecycle.double_free_idempotent()


def test_free_then_reload_cache_hit() -> None:
    """Free-then-reload is a cache hit with identity intact."""
    lifecycle.free_then_reload_cache_hit(device="cpu", dtype="float32")
