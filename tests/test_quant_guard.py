"""Step 9 TDD: quant_guard passes no bogus backend kwarg (COS-9).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_quant_guard.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import pytest
import torch

from opguard.util import quant_guard


def test_nf4_config_has_no_backend() -> None:
    """The built BitsAndBytesConfig carries no backend key or attribute.

    BitsAndBytesConfig has no backend parameter; the value was silently
    swallowed by **kwargs, so this is a regression guard in case a
    future transformers version starts storing unknown kwargs.
    """
    from transformers import BlipForConditionalGeneration

    config = quant_guard(
        compute_dtype=torch.bfloat16,
        model_type=BlipForConditionalGeneration,
        quant_type="nf4",
    )
    assert config is not None
    assert not hasattr(config, "backend")
    assert "backend" not in config.to_dict()
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"


def test_invalid_backend_still_raises() -> None:
    """The bnb-only backend validation check remains in place."""
    from transformers import BlipForConditionalGeneration

    with pytest.raises(ValueError, match="bnb"):
        quant_guard(
            compute_dtype=torch.bfloat16,
            model_type=BlipForConditionalGeneration,
            quant_type="nf4",
            backend="not-bnb",
        )
