"""Logging utilities."""

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str | None = None,
    level: int = logging.INFO,
    log_file: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Setup logger with optional console and file handlers."""
    if name is None:
        name = "tvp"
        if log_file is not None:
            name = Path(log_file).stem
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
