"""
Contract Tests — Data Schema & API Contracts
=============================================
Validates that:
  - Input schema matches feature registry (training-serving parity)
  - Output schema is stable and versioned
  - Feature types and ranges are enforced
  - Label mappings are deterministic
"""

import sys
import os
import json
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.feature_registry import (
    FEATURE_COLUMNS, LABEL_MAP, LABEL_MAP_REVERSE,
    OPERATION_MODE_MAP, TARGET_COLUMN, REQUIRED_RAW_COLUMNS,
    encode_operation_mode, apply_label_map,
    MIN_ACCURACY, MIN_ACCURACY_GAIN,
    PSI_WARNING, PSI_CRITICAL,
)
import pandas as pd
import numpy as np


# ════════════════════════════════════════════════════════════════
#  Feature Column Contract
# ════════════════════════════════════════════════════════════════

class TestFeatureColumnContract:

    def test_feature_count_is_14(self):
        assert len(FEATURE_COLUMNS) == 14

    def test_required_numeric_features_present(self):
        required = [
            "Temperature_C", "Vibration_Hz", "Power_Consumption_kW",
            "Network_Latency_ms", "Packet_Loss_%",
            "Quality_Control_Defect_Rate_%",
            "Production_Speed_units_per_hr",
            "Predictive_Maintenance_Score", "Error_Rate_%",
        ]
        for f in required:
            assert f in FEATURE_COLUMNS, f"Missing feature: {f}"

    def test_temporal_features_present(self):
        for f in ["Year", "Month", "Day", "Hour"]:
            assert f in FEATURE_COLUMNS

    def test_operation_mode_is_first(self):
        assert FEATURE_COLUMNS[0] == "Operation_Mode"

    def test_temporal_features_are_last_four(self):
        assert FEATURE_COLUMNS[-4:] == ["Year", "Month", "Day", "Hour"]

    def test_no_duplicate_features(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_feature_order_is_stable(self):
        """Import twice — order must be identical (no set/dict non-determinism)."""
        from src.feature_registry import FEATURE_COLUMNS as FC2
        assert FEATURE_COLUMNS == FC2


# ════════════════════════════════════════════════════════════════
#  Label Mapping Contract
# ════════════════════════════════════════════════════════════════

class TestLabelContract:

    def test_label_map_has_three_classes(self):
        assert len(LABEL_MAP) == 3

    def test_label_map_values_are_0_1_2(self):
        assert set(LABEL_MAP.values()) == {0, 1, 2}

    def test_label_map_keys_are_strings(self):
        for k in LABEL_MAP:
            assert isinstance(k, str)

    def test_reverse_map_is_inverse(self):
        for name, idx in LABEL_MAP.items():
            assert LABEL_MAP_REVERSE[idx] == name

    def test_label_names(self):
        assert "High"   in LABEL_MAP
        assert "Low"    in LABEL_MAP
        assert "Medium" in LABEL_MAP

    @pytest.mark.parametrize("name,expected_id", [
        ("High",   0),
        ("Low",    1),
        ("Medium", 2),
    ])
    def test_specific_label_ids(self, name, expected_id):
        assert LABEL_MAP[name] == expected_id


# ════════════════════════════════════════════════════════════════
#  Operation Mode Encoding Contract
# ════════════════════════════════════════════════════════════════

class TestOperationModeContract:

    @pytest.mark.parametrize("input_val,expected", [
        ("Mode_A", 0),
        ("Mode_B", 1),
        ("Mode_C", 2),
        (0,        0),
        (1,        1),
        (2,        2),
        ("0",      0),
        ("1",      1),
        ("2",      2),
    ])
    def test_encode_operation_mode(self, input_val, expected):
        assert encode_operation_mode(input_val) == expected

    @pytest.mark.parametrize("unknown_val", [
        "Mode_D", "Mode_X", "unknown", "null", "", "abc",
    ])
    def test_unknown_mode_falls_back_to_zero(self, unknown_val):
        result = encode_operation_mode(unknown_val)
        assert result == 0

    def test_encoding_is_deterministic(self):
        for _ in range(10):
            assert encode_operation_mode("Mode_A") == 0
            assert encode_operation_mode("Mode_B") == 1
            assert encode_operation_mode("Mode_C") == 2


# ════════════════════════════════════════════════════════════════
#  Apply Label Map Contract
# ════════════════════════════════════════════════════════════════

class TestApplyLabelMap:

    def test_maps_valid_labels(self):
        s      = pd.Series(["High", "Low", "Medium", "High"])
        result = apply_label_map(s)
        assert list(result) == [0, 1, 2, 0]

    def test_raises_on_unknown_label(self):
        s = pd.Series(["High", "Unknown", "Low"])
        with pytest.raises(ValueError, match="Unknown"):
            apply_label_map(s)

    def test_all_three_classes(self):
        s      = pd.Series(["High", "Low", "Medium"])
        result = apply_label_map(s)
        assert set(result.tolist()) == {0, 1, 2}


# ════════════════════════════════════════════════════════════════
#  API Response Schema Contract
# ════════════════════════════════════════════════════════════════

class TestAPIResponseContract:

    @pytest.fixture(scope="class")
    def client(self):
        import unittest.mock as mock
        os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns_test"
        with mock.patch(
            "src.retraining_pipeline.RetrainingPipeline.get_current_accuracy",
            return_value=0.85
        ):
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

    def test_predict_response_fields(self, client):
        rv   = client.post("/predict", data=json.dumps(self.PAYLOAD),
                            content_type="application/json")
        data = rv.get_json()
        required_fields = ["prediction", "class_id", "probabilities",
                           "model_version", "latency_ms"]
        for f in required_fields:
            assert f in data, f"Missing response field: {f}"

    def test_prediction_is_valid_class(self, client):
        rv = client.post("/predict", data=json.dumps(self.PAYLOAD),
                         content_type="application/json")
        assert rv.get_json()["prediction"] in ("High", "Low", "Medium")

    def test_class_id_is_int(self, client):
        rv = client.post("/predict", data=json.dumps(self.PAYLOAD),
                         content_type="application/json")
        assert isinstance(rv.get_json()["class_id"], int)

    def test_probabilities_three_classes(self, client):
        rv = client.post("/predict", data=json.dumps(self.PAYLOAD),
                         content_type="application/json")
        proba = rv.get_json()["probabilities"]
        assert set(proba.keys()) == {"High", "Low", "Medium"}

    def test_health_response_schema(self, client):
        rv   = client.get("/health")
        data = rv.get_json()
        for f in ["status", "model_loaded", "model_version", "environment", "timestamp"]:
            assert f in data

    def test_drift_response_schema(self, client):
        rv   = client.get("/drift")
        data = rv.get_json()
        for f in ["status", "prediction_drift_score", "feature_drift_score",
                  "drift_detected", "window_size"]:
            assert f in data


# ════════════════════════════════════════════════════════════════
#  Threshold Contract
# ════════════════════════════════════════════════════════════════

class TestThresholdContract:

    def test_min_accuracy_threshold(self):
        assert MIN_ACCURACY == 0.75

    def test_min_accuracy_gain_threshold(self):
        assert MIN_ACCURACY_GAIN == 0.02

    def test_psi_warning_threshold(self):
        assert PSI_WARNING == 0.1

    def test_psi_critical_threshold(self):
        assert PSI_CRITICAL == 0.2

    def test_psi_warning_less_than_critical(self):
        assert PSI_WARNING < PSI_CRITICAL
