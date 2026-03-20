"""
Integration tests for the Flask prediction API.
These run against the staging pod or a local Flask instance.

Set FLASK_URL env var to override default (localhost):
  FLASK_URL=http://staging-endpoint pytest tests/integration/
"""
import os
import pytest
import requests

FLASK_URL = os.getenv("FLASK_URL", "http://localhost:5001")


def flask_available():
    """Skip integration tests if Flask is not reachable."""
    try:
        r = requests.get(f"{FLASK_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not flask_available(), reason="Flask not reachable")
class TestHealthEndpoint:

    def test_health_returns_200(self):
        r = requests.get(f"{FLASK_URL}/health", timeout=5)
        assert r.status_code == 200

    def test_health_model_loaded(self):
        data = requests.get(f"{FLASK_URL}/health", timeout=5).json()
        assert data["model_loaded"] is True

    def test_health_has_required_fields(self):
        data = requests.get(f"{FLASK_URL}/health", timeout=5).json()
        for field in ["status", "model_loaded", "model_version", "environment", "timestamp"]:
            assert field in data, f"Missing field: {field}"

    def test_health_environment_is_production(self):
        data = requests.get(f"{FLASK_URL}/health", timeout=5).json()
        assert data["environment"] in ("production", "staging", "local")


@pytest.mark.skipif(not flask_available(), reason="Flask not reachable")
class TestPredictEndpoint:

    def test_predict_returns_200(self, sample_feature_vector):
        r = requests.post(f"{FLASK_URL}/predict",
                          json=sample_feature_vector, timeout=10)
        assert r.status_code == 200

    def test_predict_returns_valid_class(self, sample_feature_vector):
        data = requests.post(f"{FLASK_URL}/predict",
                             json=sample_feature_vector, timeout=10).json()
        assert data["prediction"] in ("High", "Low", "Medium")

    def test_predict_probabilities_sum_to_one(self, sample_feature_vector):
        data = requests.post(f"{FLASK_URL}/predict",
                             json=sample_feature_vector, timeout=10).json()
        probs = data.get("probabilities", {})
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_has_model_version(self, sample_feature_vector):
        data = requests.post(f"{FLASK_URL}/predict",
                             json=sample_feature_vector, timeout=10).json()
        assert "model_version" in data
        assert isinstance(data["model_version"], int)

    def test_predict_latency_under_500ms(self, sample_feature_vector):
        data = requests.post(f"{FLASK_URL}/predict",
                             json=sample_feature_vector, timeout=10).json()
        assert data.get("latency_ms", 9999) < 500

    def test_predict_missing_feature_returns_400(self):
        r = requests.post(f"{FLASK_URL}/predict",
                          json={"Temperature_C": 75.0}, timeout=10)
        assert r.status_code == 400

    def test_predict_empty_body_returns_400(self):
        r = requests.post(f"{FLASK_URL}/predict",
                          json={}, timeout=10)
        assert r.status_code in (400, 500)

    def test_predict_anomalous_returns_low_efficiency(self, anomalous_feature_vector):
        """High temp/vibration/error should predict Low efficiency."""
        data = requests.post(f"{FLASK_URL}/predict",
                             json=anomalous_feature_vector, timeout=10).json()
        # Model should consistently classify extreme anomalous inputs as Low
        assert data["prediction"] == "Low"


@pytest.mark.skipif(not flask_available(), reason="Flask not reachable")
class TestDriftEndpoint:

    def test_drift_returns_200(self):
        r = requests.get(f"{FLASK_URL}/drift", timeout=5)
        assert r.status_code == 200

    def test_drift_has_required_fields(self):
        data = requests.get(f"{FLASK_URL}/drift", timeout=5).json()
        for field in ["status", "overall_score", "prediction_drift_score",
                      "feature_drift_score", "window_size"]:
            assert field in data, f"Missing field: {field}"

    def test_drift_status_is_valid(self):
        data = requests.get(f"{FLASK_URL}/drift", timeout=5).json()
        assert data["status"] in ("OK", "WARNING", "CRITICAL")

    def test_drift_scores_are_non_negative(self):
        data = requests.get(f"{FLASK_URL}/drift", timeout=5).json()
        assert data["prediction_drift_score"] >= 0
        assert data["feature_drift_score"]    >= 0
        assert data["overall_score"]          >= 0
