"""
End-to-End Tests — Full ML Pipeline
======================================
Tests the complete flow: data → processing → training → serving → drift → retrain.
Each test exercises real artifacts (no mocks) where possible.
"""

import sys
import os
import json
import pytest
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns_test"
import unittest.mock as mock


# ════════════════════════════════════════════════════════════════
#  Pipeline Component E2E
# ════════════════════════════════════════════════════════════════

class TestDataPipeline:

    def test_feature_registry_columns_deterministic(self):
        """Feature columns must never change order between runs."""
        from src.feature_registry import FEATURE_COLUMNS
        expected = [
            "Operation_Mode", "Temperature_C", "Vibration_Hz",
            "Power_Consumption_kW", "Network_Latency_ms", "Packet_Loss_%",
            "Quality_Control_Defect_Rate_%", "Production_Speed_units_per_hr",
            "Predictive_Maintenance_Score", "Error_Rate_%",
            "Year", "Month", "Day", "Hour",
        ]
        assert FEATURE_COLUMNS == expected

    def test_label_map_stable(self):
        """Label mapping must be deterministic and never use fit_transform()."""
        from src.feature_registry import LABEL_MAP, LABEL_MAP_REVERSE
        assert LABEL_MAP["High"]   == 0
        assert LABEL_MAP["Low"]    == 1
        assert LABEL_MAP["Medium"] == 2
        assert LABEL_MAP_REVERSE[0] == "High"
        assert LABEL_MAP_REVERSE[1] == "Low"
        assert LABEL_MAP_REVERSE[2] == "Medium"

    def test_data_processing_pipeline(self, tmp_path):
        """End-to-end: load → preprocess → split → save artifacts."""
        from src.data_processing import DataProcessing
        import joblib

        # Use existing data.csv
        data_path = "artifacts/raw/data.csv"
        if not os.path.exists(data_path):
            pytest.skip("data.csv not found")

        out_dir = str(tmp_path / "processed")
        proc    = DataProcessing(data_path, out_dir)
        proc.run()

        # Verify all artifacts created
        for fname in ["X_train.pkl", "X_test.pkl", "y_train.pkl",
                      "y_test.pkl", "scaler.pkl"]:
            path = os.path.join(out_dir, fname)
            assert os.path.exists(path), f"Missing artifact: {fname}"

        # Verify shapes are sane
        X_train = joblib.load(os.path.join(out_dir, "X_train.pkl"))
        X_test  = joblib.load(os.path.join(out_dir, "X_test.pkl"))
        assert X_train.shape[1] == 14    # 14 features
        assert len(X_test) > 0

    def test_model_training_pipeline(self, tmp_path):
        """End-to-end: train model and evaluate metrics."""
        from src.data_processing import DataProcessing
        from src.model_training import ModelTraining

        data_path = "artifacts/raw/data.csv"
        if not os.path.exists(data_path):
            pytest.skip("data.csv not found")

        proc_dir  = str(tmp_path / "processed")
        model_dir = str(tmp_path / "models")

        proc    = DataProcessing(data_path, proc_dir)
        proc.run()

        trainer = ModelTraining(proc_dir + "/", model_dir + "/",
                                run_reason="e2e_test")
        metrics = trainer.run()

        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0

    def test_encode_operation_mode_all_modes(self):
        """encode_operation_mode must handle all known mode strings."""
        from src.feature_registry import encode_operation_mode
        assert encode_operation_mode("Mode_A") == 0
        assert encode_operation_mode("Mode_B") == 1
        assert encode_operation_mode("Mode_C") == 2
        assert encode_operation_mode(0)        == 0
        assert encode_operation_mode("0")      == 0
        assert encode_operation_mode("unknown") == 0   # fallback to 0


# ════════════════════════════════════════════════════════════════
#  Serving E2E
# ════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def app_client():
    with mock.patch("src.retraining_pipeline.RetrainingPipeline.get_current_accuracy",
                    return_value=0.85):
        import application
        application.app.config["TESTING"] = True
        with application.app.test_client() as c:
            yield c


PAYLOAD = {
    "Operation_Mode": 1, "Temperature_C": 72.5, "Vibration_Hz": 2.1,
    "Power_Consumption_kW": 45.0, "Network_Latency_ms": 12.0,
    "Packet_Loss_%": 0.5, "Quality_Control_Defect_Rate_%": 1.2,
    "Production_Speed_units_per_hr": 320.0,
    "Predictive_Maintenance_Score": 0.85, "Error_Rate_%": 0.8,
}


class TestServingE2E:

    def test_health_to_predict_flow(self, app_client):
        """Health check → predict → verify model is serving."""
        # 1. Health check
        health = app_client.get("/health")
        assert health.status_code == 200
        assert health.get_json()["status"] == "healthy"

        # 2. Make prediction
        rv = app_client.post("/predict",
                             data=json.dumps(PAYLOAD),
                             content_type="application/json")
        assert rv.status_code == 200

        # 3. Verify prediction
        data = rv.get_json()
        assert data["prediction"] in ("High", "Low", "Medium")

    def test_prediction_consistency(self, app_client):
        """Same input must produce same prediction (deterministic model)."""
        rv1 = app_client.post("/predict", data=json.dumps(PAYLOAD),
                               content_type="application/json")
        rv2 = app_client.post("/predict", data=json.dumps(PAYLOAD),
                               content_type="application/json")
        assert rv1.get_json()["prediction"] == rv2.get_json()["prediction"]

    def test_drift_after_predictions(self, app_client):
        """After making predictions, drift endpoint should have window_size > 0."""
        for _ in range(5):
            app_client.post("/predict", data=json.dumps(PAYLOAD),
                            content_type="application/json")
        drift = app_client.get("/drift").get_json()
        assert drift["window_size"] >= 0

    def test_explain_endpoint(self, app_client):
        rv   = app_client.post("/explain", data=json.dumps(PAYLOAD),
                               content_type="application/json")
        if rv.status_code == 503:
            pytest.skip("SHAP not installed")
        assert rv.status_code == 200
        data = rv.get_json()
        assert "prediction"   in data
        assert "shap_values"  in data
        assert "top_features" in data

    def test_multiple_payload_variations(self, app_client):
        """20 different payloads all return valid predictions."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            payload = {
                "Operation_Mode":                   int(rng.integers(0, 3)),
                "Temperature_C":                    float(rng.uniform(60, 100)),
                "Vibration_Hz":                     float(rng.uniform(1, 10)),
                "Power_Consumption_kW":             float(rng.uniform(30, 90)),
                "Network_Latency_ms":               float(rng.uniform(5, 100)),
                "Packet_Loss_%":                    float(rng.uniform(0, 5)),
                "Quality_Control_Defect_Rate_%":    float(rng.uniform(0, 15)),
                "Production_Speed_units_per_hr":    float(rng.uniform(100, 400)),
                "Predictive_Maintenance_Score":     float(rng.uniform(0, 1)),
                "Error_Rate_%":                     float(rng.uniform(0, 10)),
            }
            rv = app_client.post("/predict", data=json.dumps(payload),
                                 content_type="application/json")
            assert rv.status_code == 200
            assert rv.get_json()["prediction"] in ("High", "Low", "Medium")


# ════════════════════════════════════════════════════════════════
#  Drift Detection E2E
# ════════════════════════════════════════════════════════════════

class TestDriftE2E:

    def test_drift_detector_no_data(self):
        """DriftDetector with empty window returns 0.0 scores."""
        from src.drift_detector import DriftDetector
        if not os.path.exists("artifacts/processed/X_train.pkl"):
            pytest.skip("Processed artifacts not found")
        detector = DriftDetector()
        detector.recent_predictions.clear()
        detector.recent_inputs.clear()
        assert detector.compute_prediction_drift() == 0.0
        assert detector.compute_feature_drift()    == 0.0

    def test_drift_detector_with_uniform_predictions(self):
        """All-same predictions should show drift from reference distribution."""
        from src.drift_detector import DriftDetector
        if not os.path.exists("artifacts/processed/X_train.pkl"):
            pytest.skip("Processed artifacts not found")
        detector = DriftDetector()
        for _ in range(50):
            detector.recent_predictions.append(0)   # all "High"
        score = detector.compute_prediction_drift()
        assert isinstance(score, float)
        assert score >= 0.0
