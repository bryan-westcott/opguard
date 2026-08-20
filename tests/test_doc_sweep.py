"""Step 14 TDD: doc/behavior sweep leaves no verified typos (COS-13).

Standalone file, not collected by the marker suite. Run with:
    uv run pytest tests/test_doc_sweep.py \
        --override-ini="python_files=test_*.py" \
        --override-ini="python_functions=test_*" \
        --override-ini="addopts=-q --maxfail=1"
"""

from __future__ import annotations

import re
from pathlib import Path

import opguard

SRC_DIR = Path(opguard.__file__).parent

# Verified typo tokens (2026-08-19 re-verification) plus the doc/behavior
# wording items; regexes use word boundaries where a correct spelling
# contains the broken token
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"designged",
    r"FROM_PRETRIAINED",
    r"\bTure\b",
    r"\bunles\b",
    r"witout",
    r"muatation",
    r"attemtp",
    r"handlnig",
    r"device_list_overide\b",
    r"backedn",
    r"OFFILNE",
    r"sanitze\b",
    r"detatch",
    r"collectiona\b",
    r"Skippnig",
    r"init_gurad_kwargs",
    r"_gurad",
    r"loack",
    # wording items: stale claims that no longer match the code
    r"CPU device requested and CUDA available, falling back",
    r"will detatch ALL exceptions",
    r"'postprocessor' method",
    r"_postprocessor if desired",
)

# Corrected wording that must now be present (file, substring)
REQUIRED_STRINGS: tuple[tuple[str, str], ...] = (
    ("base.py", "Return whether to use safetensors"),
    ("base.py", "Retrieve processor"),
)


def test_no_forbidden_tokens_remain() -> None:
    """Every verified typo and stale wording is gone from src/opguard."""
    offenders: list[str] = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        text = py_file.read_text()
        for pattern in FORBIDDEN_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{py_file.relative_to(SRC_DIR)}:{line}: {pattern!r}")
    assert not offenders, "forbidden tokens remain:\n" + "\n".join(offenders)


def test_corrected_wording_present() -> None:
    """The rewritten docstrings carry the corrected wording."""
    for filename, required in REQUIRED_STRINGS:
        text = (SRC_DIR / filename).read_text()
        assert required in text, f"{filename} missing corrected wording: {required!r}"
