"""Step 7 TDD: dtype_override is annotated torch.dtype | None (COS-7).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_annotations.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import inspect

from opguard.base import OpGuardBase


def test_dtype_override_annotation() -> None:
    """dtype_override takes a dtype, so its annotation must say dtype."""
    sig = inspect.signature(OpGuardBase.__init__)
    annotation = sig.parameters["dtype_override"].annotation
    assert annotation == "torch.dtype | None"
