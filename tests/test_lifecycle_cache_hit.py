"""Step 2 TDD: cache_guard must not mutate guard identity (COS-2).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_lifecycle_cache_hit.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

# ruff: noqa: SLF001  # tests legitimately access private members

from __future__ import annotations

import torch

from opguard.util import _cache_calc_export_name
from opguard.vae import VaeTinyForSd

TAESD_ID = "madebyollin/taesd"


def _tiny_vae_export_metadata_path() -> object:
    """Resolve the metadata.json path for the cpu/float32 tiny-vae export."""
    _, _, export_dir = _cache_calc_export_name(
        base_export_name=f"{VaeTinyForSd.NAME}-detector",
        dtype=torch.float32,
        quant_config=None,
    )
    assert export_dir is not None
    return export_dir / "metadata.json"


def test_free_then_reload_is_cache_hit() -> None:
    """After load-free-reload, model_id is original and the export is untouched."""
    # Build a fresh export so this test does not rely on pre-existing state
    builder = VaeTinyForSd(
        device_override="cpu",
        dtype_override=torch.float32,
        keep_warm=True,
        force_export_refresh=True,
    )
    builder._free(reason="test: export built")
    del builder

    metadata_path = _tiny_vae_export_metadata_path()
    assert metadata_path.exists()

    # Fresh instance: first load must be a cache hit and must not
    # permanently rewrite model_id to the export path
    guard = VaeTinyForSd(device_override="cpu", dtype_override=torch.float32, keep_warm=True)
    assert guard.detector is not None
    assert guard.model_id == TAESD_ID

    mtime_before = metadata_path.stat().st_mtime_ns
    content_before = metadata_path.read_bytes()

    # Free-then-reload cycle must be a cache hit (export NOT rebuilt)
    guard._free(reason="test: free before reload")
    assert guard._detector is None
    assert guard.detector is not None
    assert guard.model_id == TAESD_ID

    assert metadata_path.stat().st_mtime_ns == mtime_before
    assert metadata_path.read_bytes() == content_before
    guard._free(reason="test: cleanup")
