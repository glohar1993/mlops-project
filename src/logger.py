"""
Production-Grade Structured JSON Logger
========================================
Every log line is valid JSON — parseable by Loki, ELK, CloudWatch Insights.
Includes: timestamp, level, service, trace_id, pod_name, message, extra fields.

Correlation ID:
  Set _log_context.trace_id before a request to correlate all logs for that
  request. Flask middleware can do this automatically per request.
  Usage:
      from src.logger import set_trace_id
      set_trace_id(request.headers.get("X-Request-ID", str(uuid4())))
"""

import logging
import json
import sys
import os
import threading
import uuid
from datetime import datetime, timezone

# Thread-local storage for per-request correlation IDs
_log_context = threading.local()

# Pod identity — injected by K8s via fieldRef downward API
_POD_NAME  = os.getenv("POD_NAME",  "unknown-pod")
_NODE_NAME = os.getenv("NODE_NAME", "unknown-node")
_ENV       = os.getenv("ENVIRONMENT", "local")


def set_trace_id(trace_id: str = None) -> str:
    """Set a trace/correlation ID for the current thread (request)."""
    _log_context.trace_id = trace_id or str(uuid.uuid4())
    return _log_context.trace_id


def get_trace_id() -> str:
    """Return current trace ID or generate one."""
    return getattr(_log_context, "trace_id", None) or str(uuid.uuid4())


class JSONFormatter(logging.Formatter):
    """Formats every log record as a single JSON line for CloudWatch/ELK."""

    SERVICE_NAME = "mlops-flask-app"

    _SKIP = {
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "level":      record.levelname,
            "service":    self.SERVICE_NAME,
            "environment": _ENV,
            "pod":        _POD_NAME,
            "node":       _NODE_NAME,
            "logger":     record.name,
            "trace_id":   get_trace_id(),
            "message":    record.getMessage(),
            "file":       f"{record.filename}:{record.lineno}",
        }

        # Attach extra fields passed via logger.info("msg", extra={"key": val})
        for key, val in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_"):
                log_entry[key] = val

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
