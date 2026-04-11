"""Tests for step-00 cleanup fixes (COS-1 through COS-4)."""

# ruff: noqa: SLF001   # tests legitimately access private members
# ruff: noqa: PLC0415  # defer heavy imports (matches existing test convention)
# ruff: noqa: ANN401   # kwargs typing in test helpers

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
import torch

from opguard.base import DetectorFactory, OpGuardBase


class _MinimalGuard(OpGuardBase):
    """Minimal concrete subclass for unit testing.

    Uses DEFAULT_DEVICE="cpu" and patches variant_guard to avoid network calls.
    """

    NAME = "test-minimal"
    MODEL_ID = "test/model"
    REVISION = "main"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_DEVICE_MAP = "cpu"
    DETECTOR_TYPE = cast("DetectorFactory", lambda *_, **__: ...)

    def _load_detector(self) -> Callable:
        return lambda x: x


def _make_guard(**kwargs: Any) -> _MinimalGuard:
    """Create a _MinimalGuard with variant_guard patched to avoid network calls."""
    with patch("opguard.util.variant_guard", return_value=("", None)):
        return _MinimalGuard(**kwargs)


# ---------------------------------------------------------------------------
# COS-1: _postprocess calls processor.postprocess(), not preprocessor()
# ---------------------------------------------------------------------------


class TestPostprocess:
    """COS-1: _postprocess() delegates to processor.postprocess()."""

    def test_postprocess_calls_postprocess_method(self) -> None:
        """When _processor has a postprocess method, _postprocess delegates to it."""
        guard = _MinimalGuard.__new__(_MinimalGuard)
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = "post_result"
        guard._processor = mock_processor

        result = guard._postprocess(output_raw="raw_data")

        mock_processor.postprocess.assert_called_once_with("raw_data")
        assert result == "post_result"

    def test_postprocess_passthrough_when_no_processor(self) -> None:
        """When _processor is None, _postprocess returns input unchanged."""
        guard = _MinimalGuard.__new__(_MinimalGuard)
        guard._processor = None

        result = guard._postprocess(output_raw="raw_data")
        assert result == "raw_data"

    def test_postprocess_passthrough_when_no_postprocess_method(self) -> None:
        """When _processor exists but has no postprocess method, passthrough."""
        guard = _MinimalGuard.__new__(_MinimalGuard)
        # Object with no postprocess attribute
        guard._processor = object()

        result = guard._postprocess(output_raw="raw_data")
        assert result == "raw_data"


# ---------------------------------------------------------------------------
# COS-2: sanitize_all_exceptions and detach_outputs respect constructor args
# ---------------------------------------------------------------------------


class TestInitParams:
    """COS-2: Constructor parameters are not silently overwritten."""

    def test_defaults_are_true(self) -> None:
        """When not passed, both default to True in the signature."""
        import inspect

        sig = inspect.signature(OpGuardBase.__init__)
        assert sig.parameters["sanitize_all_exceptions"].default is True
        assert sig.parameters["detach_outputs"].default is True

    def test_sanitize_all_exceptions_false(self) -> None:
        """sanitize_all_exceptions=False is respected, not overwritten to True."""
        guard = _make_guard(sanitize_all_exceptions=False)
        assert guard.sanitize_all_exceptions is False

    def test_detach_outputs_false(self) -> None:
        """detach_outputs=False is respected, not overwritten to True."""
        guard = _make_guard(detach_outputs=False)
        assert guard.detach_outputs is False

    def test_both_false(self) -> None:
        """Both can be set to False simultaneously."""
        guard = _make_guard(sanitize_all_exceptions=False, detach_outputs=False)
        assert guard.sanitize_all_exceptions is False
        assert guard.detach_outputs is False

    def test_both_default_true(self) -> None:
        """Both default to True when not specified."""
        guard = _make_guard()
        assert guard.sanitize_all_exceptions is True
        assert guard.detach_outputs is True


# ---------------------------------------------------------------------------
# COS-3: FROM_PRETRAINED_SKIP_KWARGS validation raises string message
# ---------------------------------------------------------------------------


class TestSkipKwargsValidation:
    """COS-3: TypeError message is a string, not a tuple."""

    def test_error_message_is_string(self) -> None:
        """TypeError raised for non-tuple SKIP_KWARGS has a string message."""

        class BadSkipKwargs(_MinimalGuard):
            FROM_PRETRAINED_SKIP_KWARGS = "not-a-tuple"  # type: ignore[assignment]

        with patch("opguard.util.variant_guard", return_value=("", None)):
            guard = BadSkipKwargs()
        # Call the base class _load_detector directly — the override in
        # _MinimalGuard skips the validation, but the base class method
        # contains the FROM_PRETRAINED_SKIP_KWARGS type check.
        with pytest.raises(TypeError) as exc_info:
            OpGuardBase._load_detector(guard)
        # The message itself must be a string, not a tuple
        assert isinstance(exc_info.value.args[0], str)
        assert "FROM_PRETRAINED_SKIP_KWARGS" in str(exc_info.value)


# ---------------------------------------------------------------------------
# COS-4: Subclasses use DTYPE_PREFERENCE, not DEFAULT_DTYPE
# ---------------------------------------------------------------------------


class TestDtypePreference:
    """COS-4: vae.py and sd.py use DTYPE_PREFERENCE."""

    def test_no_default_dtype_in_codebase(self) -> None:
        """No class in opguard defines DEFAULT_DTYPE (it should be DTYPE_PREFERENCE)."""
        import opguard.sd as sd_mod
        import opguard.vae as vae_mod

        # Check that none of the VAE subclasses have DEFAULT_DTYPE
        for name, obj in vars(vae_mod).items():
            if isinstance(obj, type) and issubclass(obj, OpGuardBase) and obj is not OpGuardBase:
                assert not hasattr(obj, "DEFAULT_DTYPE") or "DEFAULT_DTYPE" not in obj.__dict__, (
                    f"{name} still defines DEFAULT_DTYPE"
                )

        # Check that none of the SD subclasses have DEFAULT_DTYPE
        for name, obj in vars(sd_mod).items():
            if isinstance(obj, type) and issubclass(obj, OpGuardBase) and obj is not OpGuardBase:
                assert not hasattr(obj, "DEFAULT_DTYPE") or "DEFAULT_DTYPE" not in obj.__dict__, (
                    f"{name} still defines DEFAULT_DTYPE"
                )

    def test_dtype_preference_exists(self) -> None:
        """VAE and SD subclasses have DTYPE_PREFERENCE set."""
        import opguard.sd as sd_mod
        import opguard.vae as vae_mod

        for mod in (vae_mod, sd_mod):
            for name, obj in vars(mod).items():
                if isinstance(obj, type) and issubclass(obj, OpGuardBase) and obj is not OpGuardBase:
                    assert hasattr(obj, "DTYPE_PREFERENCE"), f"{name} missing DTYPE_PREFERENCE"
                    assert torch.bfloat16 == obj.DTYPE_PREFERENCE, f"{name}.DTYPE_PREFERENCE is not torch.bfloat16"
