"""Console logging setup shared by all entry points."""

from __future__ import annotations

import logging


def get_logger(name: str = "pathfusion") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger
