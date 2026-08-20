"""Step 8 TDD: variant_guard's local probe checks actual files (COS-8).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_variant_probe.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import shutil
import uuid
from typing import TYPE_CHECKING

import pytest
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from opguard.util import variant_guard

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def synthetic_fp16_repo() -> Iterator[str]:
    """Create a minimal cached repo layout containing a real fp16-named file.

    The plan's B-4 positive case: none of the pre-cached test repos have
    fp16-named files, so a synthetic cached layout exercises the
    variant-exists side of the probe.
    """
    from pathlib import Path

    repo_id = "opguard-test/fp16-probe"
    repo_dir = Path(HF_HUB_CACHE) / "models--opguard-test--fp16-probe"
    fake_hash = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    snapshot_dir = repo_dir / "snapshots" / fake_hash
    snapshot_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(fake_hash)
    (snapshot_dir / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"\0")
    (snapshot_dir / "config.json").write_text("{}")
    try:
        yield repo_id
    finally:
        shutil.rmtree(repo_dir)


def test_local_probe_no_fp16_files_returns_no_variant() -> None:
    """A cached repo with zero fp16/float16 files must not select fp16.

    madebyollin/taesd is cached but has no fp16-named files; the buggy
    probe treated 'repo is cached' as 'variant exists'.
    """
    variant = variant_guard(
        dtype=torch.float16,
        model_id="madebyollin/taesd",
        revision="main",
        local_hfhub_variant_check_only=True,
    )
    assert variant is None


def test_local_probe_with_fp16_files_returns_fp16(synthetic_fp16_repo: str) -> None:
    """A cached repo that really has fp16-named files selects the variant."""
    variant = variant_guard(
        dtype=torch.float16,
        model_id=synthetic_fp16_repo,
        revision="main",
        local_hfhub_variant_check_only=True,
    )
    assert variant == "fp16"


def test_local_probe_uncached_repo_returns_no_variant() -> None:
    """An uncached repo returns no variant instead of raising."""
    variant = variant_guard(
        dtype=torch.float16,
        model_id="opguard-test/definitely-not-cached",
        revision="main",
        local_hfhub_variant_check_only=True,
    )
    assert variant is None


def test_float32_skips_probe_entirely() -> None:
    """float32 never probes and never selects a variant."""
    variant = variant_guard(
        dtype=torch.float32,
        model_id="opguard-test/definitely-not-cached",
        revision="main",
        local_hfhub_variant_check_only=True,
    )
    assert variant is None
