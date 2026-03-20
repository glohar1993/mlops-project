"""
Tier 2 — Distributed Tracing (OpenTelemetry)
=============================================
Instruments spans for request tracing.
- Sends to Jaeger/Zipkin via OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set
- No-op (records locally) when OpenTelemetry not installed / endpoint not set
- Use start_span() / finish_span() for manual instrumentation
"""

import os
import time
import uuid
from functools import wraps
from typing import Optional, Dict, Any

OTEL_AVAILABLE = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    pass


class Tracer:
    """Lightweight distributed-tracing wrapper. Falls back to no-op."""

    def __init__(self, service_name: str = "mlops-serving"):
        self.service_name = service_name
        self._otel_tracer = None
        self._local_spans: Dict[str, Dict[str, Any]] = {}
        if OTEL_AVAILABLE:
            self._init_otel()

    def _init_otel(self) -> None:
        try:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)
            if endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
                )
            otel_trace.set_tracer_provider(provider)
            self._otel_tracer = otel_trace.get_tracer(self.service_name)
        except Exception as exc:
            print(f"[Tracing] OpenTelemetry init failed (no-op): {exc}")

    def start_span(self, operation: str,
                   attributes: Optional[Dict[str, Any]] = None) -> str:
        """Begin a span. Returns span_id string."""
        span_id = uuid.uuid4().hex[:10]
        self._local_spans[span_id] = {
            "operation":  operation,
            "started_at": time.time(),
            "attributes": attributes or {},
            "span_id":    span_id,
        }
        return span_id

    def finish_span(self, span_id: str, error: bool = False,
                    extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """End a span and return its data."""
        span = self._local_spans.pop(span_id, None)
        if not span:
            return None
        span["duration_ms"] = round((time.time() - span["started_at"]) * 1000, 2)
        span["error"]       = error
        if extra:
            span["attributes"].update(extra)
        return span

    def trace_function(self, operation_name: str):
        """Decorator to automatically trace a function."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                sid   = self.start_span(operation_name)
                error = False
                try:
                    return f(*args, **kwargs)
                except Exception:
                    error = True
                    raise
                finally:
                    self.finish_span(sid, error=error)
            return wrapper
        return decorator

    @staticmethod
    def new_request_id() -> str:
        """Generate a request correlation ID."""
        return uuid.uuid4().hex[:12]


# Singleton
tracer = Tracer()
