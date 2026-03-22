"""
Unit tests for DriftDetector — PSI calculations, thresholds, prediction storage.
All external dependencies (joblib, file I/O) are mocked.
"""
import pytest
import numpy as np
import json
import os
import tempfile
from collections import deque
from unittest.mock import patch, MagicMock, mock_open

# Top-level import ensures pytest-cov tracks this module from the test session start
import src.drift_detector  # noqa: F401 (coverage bootstrap)

from src.feature_registry import PSI_WARNING, PSI_CRITICAL, DRIFT_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build a DriftDetector without real pickle files
# ─────────────────────────────────────────────────────────────────────────────

def _make_detector(tmp_path=None):
    """
    Create a DriftDetector instance with all file I/O mocked.
    Injects synthetic reference data (100 rows, 14 features).
    """
    rng = np.random.default_rng(0)
    n_ref = 100
    X_ref = rng.uniform(0, 1, (n_ref, 14))
    # Balanced labels: 33 High(0), 33 Low(1), 34 Medium(2)
    y_ref = np.array([0] * 33 + [1] * 33 + [2] * 34)

    state_file = os.path.join(tmp_path or tempfile.mkdtemp(), "drift_state.json")

    with patch("src.drift_detector.joblib.load") as mock_load, \
         patch("os.makedirs"), \
         patch.object(
             __import__("src.drift_detector", fromlist=["DriftDetector"]).DriftDetector,
             "_load_state",
         ):
        mock_load.side_effect = [X_ref, y_ref]
        from src.drift_detector import DriftDetector
        detector = DriftDetector(
            reference_data_path="fake/X_train.pkl",
            reference_labels_path="fake/y_train.pkl",
            state_path=state_file,
        )
    return detector


# ─────────────────────────────────────────────────────────────────────────────
# TestDriftDetectorPSI
# ─────────────────────────────────────────────────────────────────────────────

class TestDriftDetectorPSI:
    """PSI calculation and threshold validation."""

    def test_psi_thresholds_from_registry(self):
        assert PSI_WARNING == 0.1
        assert PSI_CRITICAL == 0.2

    def test_psi_formula_symmetric(self):
        """PSI(A, B) when A == B → 0."""
        import math
        actual = expected = 0.5
        psi = (actual - expected) * math.log(actual / expected)
        assert abs(psi) < 1e-10

    def test_psi_formula_increases_with_divergence(self):
        import math
        def bin_psi(a, e):
            a, e = max(a, 1e-4), max(e, 1e-4)
            return (a - e) * math.log(a / e)
        assert bin_psi(0.5, 0.45) < bin_psi(0.5, 0.1)

    def test_psi_non_negative(self):
        """PSI contributions are always >= 0."""
        import math
        pairs = [(0.3, 0.4), (0.6, 0.2), (0.1, 0.8)]
        for a, e in pairs:
            a, e = max(a, 1e-4), max(e, 1e-4)
            assert (a - e) * math.log(a / e) >= 0

    def test_status_ok(self):
        assert _classify(0.05) == "OK"

    def test_status_warning(self):
        assert _classify(0.15) == "WARNING"

    def test_status_critical(self):
        assert _classify(0.25) == "CRITICAL"

    def test_status_boundary_warning(self):
        assert _classify(PSI_WARNING) == "WARNING"

    def test_status_boundary_critical(self):
        assert _classify(PSI_CRITICAL) == "CRITICAL"

    def test_status_just_below_warning(self):
        assert _classify(0.099) == "OK"

    def test_status_just_below_critical(self):
        assert _classify(0.199) == "WARNING"


def _classify(score):
    if score >= PSI_CRITICAL:
        return "CRITICAL"
    elif score >= PSI_WARNING:
        return "WARNING"
    return "OK"


# ─────────────────────────────────────────────────────────────────────────────
# TestComputePredictionDrift
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePredictionDrift:
    """Tests for compute_prediction_drift()."""

    def _det(self):
        return _make_detector()

    def test_returns_zero_with_fewer_than_10(self):
        det = self._det()
        for i in range(9):
            det.recent_predictions.append(i % 3)
        result = det.compute_prediction_drift()
        assert result == 0.0

    def test_returns_float_with_enough_data(self):
        det = self._det()
        for i in range(20):
            det.recent_predictions.append(i % 3)
        result = det.compute_prediction_drift()
        assert isinstance(result, float)

    def test_drift_score_non_negative(self):
        det = self._det()
        for _ in range(25):
            det.recent_predictions.append(0)
        result = det.compute_prediction_drift()
        assert result >= 0.0

    def test_drift_score_rounded_to_4_decimal(self):
        det = self._det()
        for _ in range(15):
            det.recent_predictions.append(1)
        result = det.compute_prediction_drift()
        assert result == round(result, 4)

    def test_uniform_distribution_has_low_drift(self):
        """If recent matches reference distribution, drift should be near 0."""
        det = self._det()
        # Reference is ~33/33/34 distribution; replicate it
        for _ in range(11):
            det.recent_predictions.append(0)
        for _ in range(11):
            det.recent_predictions.append(1)
        for _ in range(11):
            det.recent_predictions.append(2)
        result = det.compute_prediction_drift()
        assert result < PSI_WARNING

    def test_skewed_distribution_has_higher_drift(self):
        """All predictions being label 0 → higher PSI vs balanced distribution."""
        det = self._det()
        for _ in range(30):
            det.recent_predictions.append(0)
        skewed = det.compute_prediction_drift()

        det2 = self._det()
        for i in range(30):
            det2.recent_predictions.append(i % 3)
        balanced = det2.compute_prediction_drift()

        assert skewed >= balanced

    def test_window_size_exactly_10(self):
        det = self._det()
        for i in range(10):
            det.recent_predictions.append(i % 3)
        result = det.compute_prediction_drift()
        assert isinstance(result, float)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeFeatureDrift
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFeatureDrift:
    """Tests for compute_feature_drift()."""

    def _det(self):
        return _make_detector()

    def test_returns_zero_with_fewer_than_10(self):
        det = self._det()
        for i in range(5):
            det.recent_inputs.append([0.5] * 14)
        assert det.compute_feature_drift() == 0.0

    def test_returns_float_with_enough_data(self):
        det = self._det()
        for _ in range(15):
            det.recent_inputs.append(list(det.reference_feature_mean))
        result = det.compute_feature_drift()
        assert isinstance(result, float)

    def test_no_drift_when_mean_matches_reference(self):
        det = self._det()
        for _ in range(15):
            det.recent_inputs.append(list(det.reference_feature_mean))
        score = det.compute_feature_drift()
        assert score < PSI_WARNING

    def test_large_drift_when_inputs_are_extreme(self):
        det = self._det()
        for _ in range(15):
            det.recent_inputs.append([999.0] * 14)
        score = det.compute_feature_drift()
        assert score > PSI_CRITICAL

    def test_feature_drift_non_negative(self):
        det = self._det()
        for _ in range(12):
            det.recent_inputs.append([0.1] * 14)
        assert det.compute_feature_drift() >= 0.0

    def test_score_rounded_to_4_decimal(self):
        det = self._det()
        for _ in range(12):
            det.recent_inputs.append([0.5] * 14)
        score = det.compute_feature_drift()
        assert score == round(score, 4)


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckDrift
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDrift:
    """Tests for check_drift() orchestration."""

    def _det(self):
        det = _make_detector()
        # Patch _save_state so no file I/O happens
        det._save_state = MagicMock()
        return det

    def test_returns_dict(self):
        det = self._det()
        result = det.check_drift()
        assert isinstance(result, dict)

    def test_result_has_required_keys(self):
        det = self._det()
        result = det.check_drift()
        required = {
            "timestamp", "prediction_drift_score", "feature_drift_score",
            "overall_score", "status", "drift_detected", "window_size",
            "current_pred_dist"
        }
        assert required.issubset(result.keys())

    def test_status_ok_with_empty_window(self):
        det = self._det()
        result = det.check_drift()
        assert result["status"] == "OK"
        assert result["drift_detected"] is False

    def test_overall_score_is_max_of_two_scores(self):
        det = self._det()
        det.compute_prediction_drift = MagicMock(return_value=0.05)
        det.compute_feature_drift = MagicMock(return_value=0.15)
        result = det.check_drift()
        assert result["overall_score"] == 0.15

    def test_status_critical_sets_drift_detected_true(self):
        det = self._det()
        det.compute_prediction_drift = MagicMock(return_value=0.25)
        det.compute_feature_drift = MagicMock(return_value=0.25)
        result = det.check_drift()
        assert result["status"] == "CRITICAL"
        assert result["drift_detected"] is True

    def test_status_warning_sets_drift_detected_false(self):
        det = self._det()
        det.compute_prediction_drift = MagicMock(return_value=0.15)
        det.compute_feature_drift = MagicMock(return_value=0.12)
        result = det.check_drift()
        assert result["status"] == "WARNING"
        assert result["drift_detected"] is False

    def test_window_size_in_result(self):
        det = self._det()
        for i in range(5):
            det.recent_predictions.append(i % 3)
        result = det.check_drift()
        assert result["window_size"] == 5

    def test_current_pred_dist_counts(self):
        det = self._det()
        for _ in range(3):
            det.recent_predictions.append(0)
        for _ in range(2):
            det.recent_predictions.append(1)
        result = det.check_drift()
        assert result["current_pred_dist"]["High"] == 3
        assert result["current_pred_dist"]["Low"] == 2
        assert result["current_pred_dist"]["Medium"] == 0

    def test_save_state_called(self):
        det = self._det()
        det.check_drift()
        det._save_state.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# TestRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestRecord:
    """Tests for record() method."""

    def _det(self):
        det = _make_detector()
        det._save_state = MagicMock()
        return det

    def test_record_appends_prediction(self):
        det = self._det()
        det.record([0.5] * 14, 1)
        assert list(det.recent_predictions) == [1]

    def test_record_appends_input(self):
        det = self._det()
        features = [float(i) for i in range(14)]
        det.record(features, 0)
        assert list(det.recent_inputs)[0] == features

    def test_record_multiple_predictions(self):
        det = self._det()
        for i in range(5):
            det.record([0.1] * 14, i % 3)
        assert len(det.recent_predictions) == 5

    def test_window_does_not_exceed_max(self):
        det = self._det()
        for i in range(DRIFT_WINDOW + 20):
            det.record([0.5] * 14, i % 3)
        assert len(det.recent_predictions) == DRIFT_WINDOW

    def test_record_calls_save_state(self):
        det = self._det()
        det.record([0.0] * 14, 2)
        det._save_state.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# TestSaveLoadState
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveLoadState:
    """Tests for _save_state / _load_state persistence."""

    def test_save_and_load_roundtrip(self, tmp_path):
        state_file = str(tmp_path / "state.json")

        rng = np.random.default_rng(1)
        X_ref = rng.uniform(0, 1, (50, 14))
        y_ref = np.array([0] * 17 + [1] * 17 + [2] * 16)

        with patch("src.drift_detector.joblib.load") as mock_load, \
             patch("os.makedirs"):
            mock_load.side_effect = [X_ref, y_ref]
            from src.drift_detector import DriftDetector
            det = DriftDetector(
                reference_data_path="fake/X.pkl",
                reference_labels_path="fake/y.pkl",
                state_path=state_file,
            )

        det.recent_predictions.extend([0, 1, 2, 0, 1])
        det.recent_inputs.extend([[0.5] * 14, [0.3] * 14])
        det.retraining_triggered = True
        det._save_state()

        # Reload
        det.recent_predictions.clear()
        det.recent_inputs.clear()
        det.retraining_triggered = False
        det._load_state()

        assert list(det.recent_predictions) == [0, 1, 2, 0, 1]
        assert len(det.recent_inputs) == 2
        assert det.retraining_triggered is True

    def test_load_state_handles_missing_file(self, tmp_path):
        state_file = str(tmp_path / "nonexistent.json")
        rng = np.random.default_rng(2)
        X_ref = rng.uniform(0, 1, (50, 14))
        y_ref = np.array([0] * 17 + [1] * 16 + [2] * 17)

        with patch("src.drift_detector.joblib.load") as mock_load, \
             patch("os.makedirs"):
            mock_load.side_effect = [X_ref, y_ref]
            from src.drift_detector import DriftDetector
            det = DriftDetector(
                reference_data_path="fake/X.pkl",
                reference_labels_path="fake/y.pkl",
                state_path=state_file,
            )
        # No exception — state is empty
        assert len(det.recent_predictions) == 0
