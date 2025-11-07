"""Configure pytest."""

import sys

from _pytest.config import Config
from loguru import logger


def pytest_configure(_config: Config) -> None:
    """Set default logging level for pytest.

    Note: this will not have an effect unless pytest `-s` option is used,
    for example:
        >>> pytest -s -m smoke
    """
    logger.remove()
    # Note: set this to TRACE for maximum detail
    logger.add(sys.stderr, level="INFO")
