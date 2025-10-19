"""JSON logging utilities."""
from __future__ import annotations

import json
import logging
from logging import Logger
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("extra_"):
                payload[key.removeprefix("extra_")] = value
        return json.dumps(payload)


def setup_logging(level: int = logging.INFO) -> Logger:
    """Configure root logger with JSON formatter."""
    logger = logging.getLogger("traderx")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
