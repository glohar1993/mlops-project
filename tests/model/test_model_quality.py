"""
Model Quality Gate Tests
=========================
These tests act as a production gate:
  - Accuracy must exceed MIN_ACCURACY (75%)
  - Per-class recall must not fall below 60% (fairness)
  - Inference latency must be under 200ms for a single row
  - Model artifact must exist and load correctly

If any test fails → pipeline blocks promotion to Production.
"""
import os
import time
import pytest
import numpy as np
import joblib
from src.feature_registry import MIN_ACCURACY, LABEL_MAP_REVERSE

MODEL_PATH  = os.getenv("MODEL_PATH",  "artifacts/models/model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/processed/scaler.pkl")
X_TEST_PATH = os.getenv("X_TEST_PATH", "artifacts/processed/X_test.pkl")
Y_TEST_PATH = os.getenv("Y_TEST_PATH", "artifacts/processed/y_test.pkl")


def artifacts_exist():
    return all(os.path.exists(p) for p in
               [MODEL_PATH, SCALER_PATH, X_TEST_PATH, Y_TEST_PATH])


@pytest.mark.skipif(not artifacts_exist(), reason="Artifacts not built yet")
class TestModelQuality:

    @pytest.fixture(autouse=True)
    def load_artifacts(self):
        from sklearn.metrics import accuracy_score, recall_score
        self.model  = joblib.load(MODEL_PATH)
        self.X_test = joblib.load(X_TEST_PATH)
        self.y_test = joblib.load(Y_TEST_PATH)
        self.y_pred = self.model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def test_accuracy_above_threshold(self):
        """Model accuracy must exceed MIN_ACCURACY from feature_registry."""
        assert self.accuracy >= MIN_ACCURACY, (
            f"Accuracy {self.accuracy:.4f} is below threshold {MIN_ACCURACY}. "
            f"Do not promote this model to Production."
        )

    def test_per_class_recall_above_60pct(self):
        """No class should be systematically ignored (min 60% recall per class)."""
        from sklearn.metrics import recall_score
        MIN_CLASS_RECALL = 0.60
        for label_id, label_name in LABEL_MAP_REVERSE.items():
            class_mask = (self.y_test == label_id)
            if class_mask.sum() == 0:
                continue
            class_recall = (self.y_pred[class_mask] == label_id).mean()
            assert class_recall >= MIN_CLASS_RECALL, (
                f"Class '{label_name}' recall {class_recall:.2f} < {MIN_CLASS_RECALL}. "
                f"Model has class imbalance problem."
            )

    def test_inference_latency_under_200ms(self):
        """Single-row inference must complete in under 200ms."""
        single_row = self.X_test[:1]
        MAX_LATENCY_MS = 200
        start = time.perf_counter()
        _ = self.model.predict(single_row)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < MAX_LATENCY_MS, (
            f"Inference latency {elapsed_ms:.1f}ms exceeds {MAX_LATENCY_MS}ms SLA"
        )

    def test_model_outputs_valid_classes(self):
        """Model must only output labels 0, 1, or 2."""
        valid_classes = set(LABEL_MAP_REVERSE.keys())
        actual_classes = set(np.unique(self.y_pred))
        assert actual_classes.issubset(valid_classes), (
            f"Model produced unexpected class labels: {actual_classes - valid_classes}"
        )

    def test_model_has_predict_proba(self):
        """Model must support probability output (required for drift detection)."""
        assert hasattr(self.model, "predict_proba"), (
            "Model does not support predict_proba — drift detection will fail"
        )

    def test_probabilities_sum_to_one(self):
        """predict_proba output must sum to 1.0 for every row."""
        proba = self.model.predict_proba(self.X_test)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"

    def test_model_artifact_is_sklearn(self):
        """Model must be a scikit-learn estimator."""
        from sklearn.base import BaseEstimator
        assert isinstance(self.model, BaseEstimator), (
            f"Expected sklearn estimator, got {type(self.model)}"
        )
