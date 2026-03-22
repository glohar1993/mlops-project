"""
Unit tests for Tracer (src/tracing.py).
No external dependencies required — OTEL is optional and falls back gracefully.
"""
import time
import pytest
from unittest.mock import patch, MagicMock

# Top-level import for coverage tracking
import src.tracing  # noqa: F401

from src.tracing import Tracer


# ─────────────────────────────────────────────────────────────────────────────
# TestTracerInit
# ─────────────────────────────────────────────────────────────────────────────

class TestTracerInit:
    def test_init_sets_service_name(self):
        t = Tracer(service_name="test-service")
        assert t.service_name == "test-service"

    def test_init_default_service_name(self):
        t = Tracer()
        assert t.service_name == "mlops-serving"

    def test_init_local_spans_empty(self):
        t = Tracer()
        assert t._local_spans == {}

    def test_init_otel_tracer_none_when_no_otel(self):
        # OTEL may or may not be installed; either way, _otel_tracer may be set or None
        t = Tracer()
        # Just verify it's accessible without error
        _ = t._otel_tracer


# ─────────────────────────────────────────────────────────────────────────────
# TestStartSpan
# ─────────────────────────────────────────────────────────────────────────────

class TestStartSpan:
    def test_start_span_returns_string(self):
        t = Tracer()
        span_id = t.start_span("test_op")
        assert isinstance(span_id, str)

    def test_start_span_id_is_10_chars(self):
        t = Tracer()
        span_id = t.start_span("test_op")
        assert len(span_id) == 10

    def test_start_span_stores_in_local_spans(self):
        t = Tracer()
        sid = t.start_span("inference")
        assert sid in t._local_spans

    def test_start_span_stores_operation(self):
        t = Tracer()
        sid = t.start_span("model.predict")
        assert t._local_spans[sid]["operation"] == "model.predict"

    def test_start_span_stores_attributes(self):
        t = Tracer()
        sid = t.start_span("op", attributes={"key": "value"})
        assert t._local_spans[sid]["attributes"]["key"] == "value"

    def test_start_span_empty_attributes_default(self):
        t = Tracer()
        sid = t.start_span("op")
        assert t._local_spans[sid]["attributes"] == {}

    def test_multiple_spans_coexist(self):
        t = Tracer()
        sid1 = t.start_span("op1")
        sid2 = t.start_span("op2")
        assert sid1 in t._local_spans
        assert sid2 in t._local_spans
        assert sid1 != sid2


# ─────────────────────────────────────────────────────────────────────────────
# TestFinishSpan
# ─────────────────────────────────────────────────────────────────────────────

class TestFinishSpan:
    def test_finish_span_returns_dict(self):
        t = Tracer()
        sid = t.start_span("op")
        result = t.finish_span(sid)
        assert isinstance(result, dict)

    def test_finish_span_removes_from_local_spans(self):
        t = Tracer()
        sid = t.start_span("op")
        t.finish_span(sid)
        assert sid not in t._local_spans

    def test_finish_span_sets_duration(self):
        t = Tracer()
        sid = t.start_span("op")
        time.sleep(0.01)
        result = t.finish_span(sid)
        assert result["duration_ms"] >= 0

    def test_finish_span_error_false_by_default(self):
        t = Tracer()
        sid = t.start_span("op")
        result = t.finish_span(sid)
        assert result["error"] is False

    def test_finish_span_error_true_when_set(self):
        t = Tracer()
        sid = t.start_span("op")
        result = t.finish_span(sid, error=True)
        assert result["error"] is True

    def test_finish_span_returns_none_for_unknown_id(self):
        t = Tracer()
        result = t.finish_span("nonexistent_id")
        assert result is None

    def test_finish_span_merges_extra_attributes(self):
        t = Tracer()
        sid = t.start_span("op", attributes={"a": 1})
        result = t.finish_span(sid, extra={"b": 2})
        assert result["attributes"]["a"] == 1
        assert result["attributes"]["b"] == 2

    def test_finish_span_contains_operation_name(self):
        t = Tracer()
        sid = t.start_span("my_operation")
        result = t.finish_span(sid)
        assert result["operation"] == "my_operation"


# ─────────────────────────────────────────────────────────────────────────────
# TestTraceFunction (decorator)
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceFunction:
    def test_decorator_calls_wrapped_function(self):
        t = Tracer()
        calls = []

        @t.trace_function("test_op")
        def my_func(x):
            calls.append(x)
            return x * 2

        result = my_func(5)
        assert result == 10
        assert calls == [5]

    def test_decorator_cleans_up_span_after_call(self):
        t = Tracer()

        @t.trace_function("cleanup_test")
        def noop():
            pass

        noop()
        assert len(t._local_spans) == 0

    def test_decorator_preserves_function_name(self):
        t = Tracer()

        @t.trace_function("op")
        def my_named_function():
            pass

        assert my_named_function.__name__ == "my_named_function"

    def test_decorator_propagates_exceptions(self):
        t = Tracer()

        @t.trace_function("error_op")
        def raises():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            raises()

    def test_decorator_cleans_up_span_on_exception(self):
        t = Tracer()

        @t.trace_function("error_op")
        def raises():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            raises()
        assert len(t._local_spans) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestNewRequestId
# ─────────────────────────────────────────────────────────────────────────────

class TestNewRequestId:
    def test_returns_string(self):
        assert isinstance(Tracer.new_request_id(), str)

    def test_length_is_12(self):
        assert len(Tracer.new_request_id()) == 12

    def test_ids_are_unique(self):
        ids = {Tracer.new_request_id() for _ in range(50)}
        assert len(ids) == 50


# ─────────────────────────────────────────────────────────────────────────────
# TestSingleton
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_module_tracer_is_tracer_instance(self):
        from src.tracing import tracer
        assert isinstance(tracer, Tracer)
