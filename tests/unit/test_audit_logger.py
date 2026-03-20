"""
Unit Tests — Audit Logger (src/audit_logger.py)
================================================
Tests: event writing, querying, filtering, thread safety, count
"""

import os
import pytest
import sys
import threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Use a temp DB for tests
os.environ["AUDIT_DB_PATH"] = "/tmp/test_audit.db"

from src.audit_logger import log_event, query_events, count_events, clear_all_events


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe audit log before each test."""
    clear_all_events()
    yield
    clear_all_events()


# ════════════════════════════════════════════════════════════════
#  Basic Write & Read
# ════════════════════════════════════════════════════════════════

class TestLogEvent:

    def test_log_single_event_returns_true(self):
        ok = log_event("PREDICT", client_id="svc", role="operator",
                       endpoint="/predict", method="POST",
                       status_code=200, latency_ms=12.5)
        assert ok is True

    def test_logged_event_appears_in_query(self):
        log_event("PREDICT", client_id="svc-1", endpoint="/predict")
        events = query_events()
        assert any(e["client_id"] == "svc-1" for e in events)

    def test_event_type_stored_correctly(self):
        log_event("MODEL_RETRAIN", client_id="admin")
        events = query_events(event_type="MODEL_RETRAIN")
        assert len(events) == 1
        assert events[0]["event_type"] == "MODEL_RETRAIN"

    def test_all_fields_stored(self):
        log_event(
            event_type="PREDICT",
            client_id="test-svc",
            role="operator",
            endpoint="/v1/predict",
            method="POST",
            status_code=200,
            latency_ms=45.3,
            ip_address="10.0.0.1",
            request_id="req-abc123",
            extra={"model_version": 3, "prediction": "High"},
        )
        evs = query_events()
        ev  = evs[0]
        assert ev["client_id"]   == "test-svc"
        assert ev["role"]        == "operator"
        assert ev["endpoint"]    == "/v1/predict"
        assert ev["method"]      == "POST"
        assert ev["status_code"] == 200
        assert ev["latency_ms"]  == pytest.approx(45.3)
        assert ev["ip_address"]  == "10.0.0.1"
        assert ev["request_id"]  == "req-abc123"
        assert "model_version" in ev["extra"]

    @pytest.mark.parametrize("event_type", [
        "PREDICT", "AUTH_FAIL", "RETRAIN", "DRIFT_CHECK",
        "GOVERNANCE_APPROVE", "BATCH_JOB", "AB_TEST",
    ])
    def test_various_event_types(self, event_type):
        log_event(event_type)
        events = query_events(event_type=event_type)
        assert len(events) == 1
        assert events[0]["event_type"] == event_type


# ════════════════════════════════════════════════════════════════
#  Querying
# ════════════════════════════════════════════════════════════════

class TestQueryEvents:

    def test_filter_by_event_type(self):
        log_event("PREDICT")
        log_event("RETRAIN")
        log_event("PREDICT")
        preds = query_events(event_type="PREDICT")
        retrains = query_events(event_type="RETRAIN")
        assert len(preds)    == 2
        assert len(retrains) == 1

    def test_limit_parameter(self):
        for i in range(20):
            log_event("PREDICT", client_id=f"svc-{i}")
        limited = query_events(limit=5)
        assert len(limited) == 5

    def test_returns_most_recent_first(self):
        log_event("PREDICT", client_id="first")
        log_event("PREDICT", client_id="second")
        events = query_events()
        assert events[0]["client_id"] == "second"   # most recent first

    def test_empty_db_returns_empty_list(self):
        assert query_events() == []

    def test_count_events(self):
        for _ in range(5):
            log_event("PREDICT")
        assert count_events() == 5
        assert count_events("PREDICT") == 5
        assert count_events("RETRAIN") == 0


# ════════════════════════════════════════════════════════════════
#  Thread Safety
# ════════════════════════════════════════════════════════════════

class TestThreadSafety:

    def test_concurrent_writes(self):
        """100 concurrent threads each write one event — all must be recorded."""
        n = 100

        def write(i):
            log_event("CONCURRENT_TEST", client_id=f"worker-{i}")

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert count_events("CONCURRENT_TEST") == n

    def test_graceful_on_null_fields(self):
        """log_event with all optional fields None must not crash."""
        result = log_event("MINIMAL_EVENT")
        assert result is True
