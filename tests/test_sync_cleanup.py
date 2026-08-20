"""Step 6 TDD: sync loop iterates cuda_devs, not device_list (COS-6).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_sync_cleanup.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import pytest
import torch

from opguard.util import sync_gc_and_cache_cleanup


def test_cpu_only_device_list_no_exception() -> None:
    """A cpu-only device list must clean up without raising."""
    sync_gc_and_cache_cleanup(
        device_list=[torch.device("cpu")],
        suppress_errors=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_device_list_no_exception() -> None:
    """A mixed cuda+cpu device list must only synchronize the cuda devices.

    Pre-fix, the synchronize loop iterated the full device_list, so the
    cpu entry raised inside torch.cuda.synchronize when errors are not
    suppressed.
    """
    sync_gc_and_cache_cleanup(
        device_list=[torch.device("cuda:0"), torch.device("cpu")],
        suppress_errors=False,
    )
