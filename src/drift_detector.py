"""
Model Drift Detector
====================
Detects two types of drift:
1. Prediction Drift  — distribution of High/Medium/Low predictions is shifting
2. Feature Drift     — input data statistics are changing (using PSI score)

PSI (Population Stability Index) thresholds:
  PSI < 0.1  → No drift       (GREEN)
  PSI < 0.2  → Slight drift   (YELLOW - monitor)
  PSI >= 0.2 → Major drift    (RED - retrain!)
"""

import numpy as np
import joblib
import json
import os
from datetime import datetime
from collections import deque


# How many recent predictions to keep in memory for drift analysis
WINDOW_SIZE = 50

# PSI drift thresholds
PSI_WARNING   = 0.1
PSI_CRITICAL  = 0.2


class DriftDetector:
    def __init__(self, reference_data_path="artifacts/processed/X_train.pkl",
                       reference_labels_path="artifacts/processed/y_train.pkl",
                       state_path="artifacts/drift/drift_state.json"):

        self.state_path = state_path
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        # Rolling window of recent predictions and inputs
        self.recent_predictions = deque(maxlen=WINDOW_SIZE)
        self.recent_inputs       = deque(maxlen=WINDOW_SIZE)

        # Load reference (training) data to compare against
        self.reference_features = joblib.load(reference_data_path)
        self.reference_labels   = joblib.load(reference_labels_path)

        # Compute reference prediction distribution (expected %)
        labels, counts = np.unique(self.reference_labels, return_counts=True)
        total = counts.sum()
        self.reference_pred_dist = {int(l): c / total for l, c in zip(labels, counts)}

        # Compute reference feature stats (mean/std per feature)
        self.reference_feature_mean = np.mean(self.reference_features, axis=0)
        self.reference_feature_std  = np.std(self.reference_features, axis=0) + 1e-8

        # Drift scores (updated on each check)
        self.prediction_drift_score = 0.0
        self.feature_drift_score    = 0.0
        self.drift_detected         = False
        self.retraining_triggered   = False

        self._load_state()

    # ------------------------------------------------------------------ #
    #  Record a new prediction
    # ------------------------------------------------------------------ #
    def record(self, input_features: list, prediction: int):
        self.recent_predictions.append(prediction)
        self.recent_inputs.append(input_features)
        self._save_state()

    # ------------------------------------------------------------------ #
    #  Compute PSI for prediction distribution
    # ------------------------------------------------------------------ #
    def compute_prediction_drift(self) -> float:
        if len(self.recent_predictions) < 10:
            return 0.0  # Not enough data yet

        labels = [0, 1, 2]
        total  = len(self.recent_predictions)
        psi    = 0.0

        for label in labels:
            actual_pct   = self.recent_predictions.count(label) / total
            expected_pct = self.reference_pred_dist.get(label, 0.01)

            # Avoid log(0)
            actual_pct   = max(actual_pct, 0.0001)
            expected_pct = max(expected_pct, 0.0001)

            psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

        return round(psi, 4)

    # ------------------------------------------------------------------ #
    #  Compute feature drift using normalized mean shift
    # ------------------------------------------------------------------ #
    def compute_feature_drift(self) -> float:
        if len(self.recent_inputs) < 10:
            return 0.0

        recent_array = np.array(list(self.recent_inputs))
        current_mean = np.mean(recent_array, axis=0)

        # Normalized drift: how many std deviations has the mean shifted
        shift = np.abs(current_mean - self.reference_feature_mean) / self.reference_feature_std
        drift_score = float(np.mean(shift))

        # Scale to PSI-like range (>0.2 = significant)
        return round(drift_score / 10.0, 4)

    # ------------------------------------------------------------------ #
    #  Run full drift check - returns status dict
    # ------------------------------------------------------------------ #
    def check_drift(self) -> dict:
        self.prediction_drift_score = self.compute_prediction_drift()
        self.feature_drift_score    = self.compute_feature_drift()

        overall_score = max(self.prediction_drift_score, self.feature_drift_score)

        if overall_score >= PSI_CRITICAL:
            status = "CRITICAL"
            self.drift_detected = True
        elif overall_score >= PSI_WARNING:
            status = "WARNING"
            self.drift_detected = False
        else:
            status = "OK"
            self.drift_detected = False

        result = {
            "timestamp":              datetime.utcnow().isoformat(),
            "prediction_drift_score": self.prediction_drift_score,
            "feature_drift_score":    self.feature_drift_score,
            "overall_score":          overall_score,
            "status":                 status,
            "drift_detected":         self.drift_detected,
            "window_size":            len(self.recent_predictions),
            "current_pred_dist": {
                "High":   self.recent_predictions.count(0),
                "Low":    self.recent_predictions.count(1),
                "Medium": self.recent_predictions.count(2),
            }
        }

        self._save_state()
        return result

    # ------------------------------------------------------------------ #
    #  Persist state to disk
    # ------------------------------------------------------------------ #
    def _save_state(self):
        state = {
            "recent_predictions": list(self.recent_predictions),
            "recent_inputs":      [x for x in self.recent_inputs],
            "retraining_triggered": self.retraining_triggered,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f)

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            self.recent_predictions = deque(state.get("recent_predictions", []), maxlen=WINDOW_SIZE)
            self.recent_inputs       = deque(state.get("recent_inputs", []),       maxlen=WINDOW_SIZE)
            self.retraining_triggered = state.get("retraining_triggered", False)
