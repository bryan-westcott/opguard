"""Configure pytest."""

import sys

from _pytest.config import Config
from loguru import logger


def pytest_configure(config: Config) -> None:
    """Set default logging level for pytest.

    Note: this will not have an effect unless pytest `-s` option is used,
    for example:
        >>> pytest -s -m smoke
    """
    # mark as intentionally unused (silences Ruff ARG001/ARG002 and mypy)
    _ = config

    logger.remove()
    # Note: set this to TRACE for maximum detail
    logger.add(sys.stderr, level="INFO")
