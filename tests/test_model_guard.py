"""Step 3 TDD: model_guard propagates original exceptions (COS-3).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_model_guard.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

# ruff: noqa: ANN401  # test loaders mirror the abstract loader interface

from __future__ import annotations

from typing import Any

import pytest
import torch

from opguard.util import model_guard


def _passthrough_loader(**kwargs: Any) -> Any:
    return lambda x: x


def _raising_loader(**kwargs: Any) -> Any:
    message = "intentional loader failure"
    raise RuntimeError(message)


def _echo(value: Any) -> Any:
    return value


def _init_guard_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device": "cpu",
        "device_map": None,
        "dtype": torch.float32,
        "model_type": _passthrough_loader,
        "model_id": "test/model",
        "revision": "main",
    }
    return kwargs | overrides


def test_loader_failure_propagates_original_exception() -> None:
    """A RuntimeError from the loader reaches the caller, not a NameError."""
    with pytest.raises(RuntimeError, match="intentional loader failure"):  # noqa: SIM117
        with model_guard(
            init_guard_kwargs=_init_guard_kwargs(),
            load_guard_kwargs={
                "loader_fn": _raising_loader,
                "train_mode": False,
                "device_list": [torch.device("cpu")],
                "sanitize_all_exceptions": False,
            },
            call_guard_kwargs={"need_grads": False, "caller_fn": _echo},
            free_guard_kwargs={"run_gc_and_clear_cache": True},
        ):
            pytest.fail("guarded caller must not be reached when the loader raises")


def test_init_failure_propagates_original_exception() -> None:
    """A TypeError from init_guard reaches the caller, not a NameError."""
    with pytest.raises(TypeError, match="Unsupported"):  # noqa: SIM117
        with model_guard(
            init_guard_kwargs=_init_guard_kwargs(dtype=torch.float64),
            load_guard_kwargs={
                "loader_fn": _passthrough_loader,
                "train_mode": False,
                "device_list": [torch.device("cpu")],
            },
            call_guard_kwargs={"need_grads": False, "caller_fn": _echo},
            free_guard_kwargs={"run_gc_and_clear_cache": True},
        ):
            pytest.fail("guarded caller must not be reached when init fails")


def test_success_path_yields_and_frees() -> None:
    """The success path yields a working guarded caller and exits cleanly."""
    with model_guard(
        init_guard_kwargs=_init_guard_kwargs(),
        load_guard_kwargs={
            "loader_fn": _passthrough_loader,
            "train_mode": False,
            "device_list": [torch.device("cpu")],
        },
        call_guard_kwargs={"need_grads": False, "caller_fn": _echo},
        free_guard_kwargs={"run_gc_and_clear_cache": True},
    ) as guarded:
        assert guarded("hello") == "hello"
