"""Centralized logging helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict


_LOGGERS: dict[str, logging.Logger] = {}


def configure_logging(config: Dict[str, Any]) -> None:
    """Set up console + file logging according to config settings."""
    logging_config = config.get("logging", {})
    level = logging_config.get("level", "INFO")
    log_dir = Path(logging_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = []
    if logging_config.get("log_to_stdout", True):
        handlers.append(logging.StreamHandler())

    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a cached logger to keep handler configuration consistent."""
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]
