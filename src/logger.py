"""
Production-Grade Structured JSON Logger
========================================
Every log line is valid JSON — parseable by Loki, ELK, CloudWatch.
Includes: timestamp, level, service, trace_id, message, extra fields.
"""

import logging
import json
import sys
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    SERVICE_NAME = "mlops-flask-app"

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "service":   self.SERVICE_NAME,
            "logger":    record.name,
            "message":   record.getMessage(),
            "file":      f"{record.filename}:{record.lineno}",
        }

        # Attach extra fields passed via logger.info("msg", extra={"key": val})
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
        }
        for key, val in record.__dict__.items():
            if key not in skip and not key.startswith("_"):
                log_entry[key] = val

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
