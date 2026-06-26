"""Configurable multi-channel logging utilities."""

import logging
import os
from pathlib import Path
from core.config import settings


def setup_logger(name: str, log_filename: str | None = None) -> logging.Logger:
    """Configures and returns a logger instance.

    If log_filename is provided, output is routed both to console and the specified
    log file inside the logs directory (e.g. logs/training.log).
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)

    # Avoid adding duplicate handlers if the logger has already been configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_filename:
        log_dir = Path(settings.log_dir)
        log_path = log_dir / log_filename
        
        # Ensure parent directories exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
