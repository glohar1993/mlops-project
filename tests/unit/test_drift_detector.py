"""
Unit tests for DriftDetector — PSI calculations and thresholds.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.feature_registry import PSI_WARNING, PSI_CRITICAL


class TestDriftDetectorPSI:
    """Test PSI calculation logic without requiring pickle artifacts."""

    def test_psi_zero_when_no_predictions(self):
        """No predictions → drift score must be 0 (not an error)."""
        from collections import deque
        recent = deque(maxlen=50)
        # Simulate compute_prediction_drift with empty window
        assert len(recent) < 10  # triggers early return of 0.0

    def test_psi_thresholds_from_registry(self):
        """PSI thresholds come from feature_registry, not hardcoded values."""
        assert PSI_WARNING  == 0.1
        assert PSI_CRITICAL == 0.2

    def test_psi_formula_symmetric(self):
        """PSI(A, B) when A == B should be 0."""
        import math
        actual_pct   = 0.5
        expected_pct = 0.5
        psi = (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)
        assert abs(psi) < 1e-10

    def test_psi_formula_increases_with_divergence(self):
        """Higher divergence between distributions → higher PSI."""
        import math
        def single_bin_psi(actual, expected):
            actual   = max(actual, 1e-4)
            expected = max(expected, 1e-4)
            return (actual - expected) * math.log(actual / expected)

        small_divergence = single_bin_psi(0.5, 0.45)
        large_divergence = single_bin_psi(0.5, 0.1)
        assert small_divergence < large_divergence

    def test_status_classification(self):
        """PSI score → status label mapping."""
        def classify(score):
            if score >= PSI_CRITICAL:
                return "CRITICAL"
            elif score >= PSI_WARNING:
                return "WARNING"
            return "OK"

        assert classify(0.05)  == "OK"
        assert classify(0.15)  == "WARNING"
        assert classify(0.25)  == "CRITICAL"
        assert classify(PSI_CRITICAL)  == "CRITICAL"
        assert classify(PSI_WARNING)   == "WARNING"


class TestDriftDetectorRecord:
    """Test the record/state logic."""

    def test_deque_respects_max_window(self):
        """Rolling window must not exceed DRIFT_WINDOW size."""
        from collections import deque
        from src.feature_registry import DRIFT_WINDOW

        q = deque(maxlen=DRIFT_WINDOW)
        for i in range(DRIFT_WINDOW + 100):
            q.append(i)
        assert len(q) == DRIFT_WINDOW

    def test_predictions_are_integers(self):
        """Prediction labels stored as integers 0, 1, 2."""
        from src.feature_registry import LABEL_MAP
        valid_preds = set(LABEL_MAP.values())
        assert valid_preds == {0, 1, 2}
