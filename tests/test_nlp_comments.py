"""Step 10 TDD: Blip2_4Bit dtype comment matches its setting (COS-10).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_nlp_comments.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import inspect

import torch

from opguard.nlp import Blip2_4Bit


def test_blip2_4bit_comment_agrees_with_setting() -> None:
    """The 4-bit class keeps bfloat16 and carries no 8-bit claim.

    The old comment ('when using 8-bit bnb, must use float16 not
    bfloat16') contradicted the bfloat16 setting on a 4-bit class; the
    8-bit note belongs on the 8-bit class if anywhere.
    """
    assert Blip2_4Bit.DTYPE_PREFERENCE is torch.bfloat16
    source = inspect.getsource(Blip2_4Bit)
    assert "8-bit" not in source
    assert "8bit" not in source.lower().replace("-", "").replace("_", "")
