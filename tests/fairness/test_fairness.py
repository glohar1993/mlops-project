"""
Fairness Tests — Model Bias Detection
=======================================
Tests: no bias baseline, bias injection detection,
       all operation modes, disparate impact, accuracy gap,
       full fairness report generation
"""

import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.fairness_monitor import FairnessMonitor


@pytest.fixture
def monitor():
    return FairnessMonitor()


# ════════════════════════════════════════════════════════════════
#  No-Bias Baseline
# ════════════════════════════════════════════════════════════════

class TestNoBiasBaseline:

    def test_equal_positive_rates_no_bias(self, monitor):
        """If all operation modes have same positive rate → DI = 1.0."""
        for _ in range(30):
            for mode in ["Mode_A", "Mode_B", "Mode_C"]:
                monitor.record({"Operation_Mode": mode}, prediction=0,
                               ground_truth=0)
        report = monitor.full_report()
        assert report["disparate_impact"]["disparate_impact"] == pytest.approx(1.0)
        assert report["disparate_impact"]["status"] == "OK"

    def test_equal_accuracy_no_bias(self, monitor):
        """All modes with 100% accuracy → accuracy gap = 0."""
        for mode in ["Mode_A", "Mode_B"]:
            for _ in range(20):
                monitor.record({"Operation_Mode": mode}, prediction=0, ground_truth=0)
        report = monitor.full_report()
        if report["accuracy_gap"].get("reason") != "insufficient_ground_truth_data":
            assert report["accuracy_gap"]["accuracy_gap"] == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════
#  Bias Injection
# ════════════════════════════════════════════════════════════════

class TestBiasDetection:

    def test_disparate_impact_bias_detected(self, monitor):
        """Mode_A gets 90% positive; Mode_B gets 20% positive → bias."""
        for _ in range(50):
            monitor.record({"Operation_Mode": "Mode_A"}, prediction=0)   # 100% High
        for _ in range(50):
            monitor.record({"Operation_Mode": "Mode_B"}, prediction=1)   # 100% Low

        report = monitor.full_report()
        di     = report["disparate_impact"]
        # Mode_B positive_rate = 0.0, Mode_A = 1.0 → DI = 0.0 < 0.8
        assert di["disparate_impact"] < 0.8
        assert di["status"] == "BIAS_DETECTED"
        assert report["overall_status"] == "BIAS_DETECTED"

    def test_accuracy_gap_bias_detected(self, monitor):
        """Mode_A 95% accurate, Mode_B 70% accurate → 25% gap → bias."""
        for i in range(20):
            # Mode_A: always correct
            monitor.record({"Operation_Mode": "Mode_A"},
                           prediction=0, ground_truth=0)
        for i in range(20):
            # Mode_B: mostly wrong
            pred = 0 if i < 3 else 1    # only 15% correct
            monitor.record({"Operation_Mode": "Mode_B"},
                           prediction=pred, ground_truth=0)

        report = monitor.full_report()
        if "reason" not in report["accuracy_gap"]:
            assert report["accuracy_gap"]["accuracy_gap"] > 0.1


# ════════════════════════════════════════════════════════════════
#  All Operation Modes
# ════════════════════════════════════════════════════════════════

class TestAllOperationModes:

    @pytest.mark.parametrize("mode", ["Mode_A", "Mode_B", "Mode_C", "0", "1", "2"])
    def test_all_modes_tracked(self, mode):
        monitor = FairnessMonitor()
        for _ in range(10):
            monitor.record({"Operation_Mode": mode}, prediction=0)
        report = monitor.full_report()
        assert str(mode) in report["subgroup_metrics"]

    def test_three_operation_modes_report(self, monitor):
        for mode in ["Mode_A", "Mode_B", "Mode_C"]:
            for pred in [0, 1, 2, 0, 1]:
                monitor.record({"Operation_Mode": mode}, prediction=pred)
        report = monitor.full_report()
        assert len(report["subgroup_metrics"]) == 3

    def test_prediction_distribution_in_subgroup(self, monitor):
        for _ in range(5):
            monitor.record({"Operation_Mode": "Mode_A"}, prediction=0)
        for _ in range(3):
            monitor.record({"Operation_Mode": "Mode_A"}, prediction=1)
        for _ in range(2):
            monitor.record({"Operation_Mode": "Mode_A"}, prediction=2)

        metrics = monitor.compute_subgroup_metrics(monitor._history, "Operation_Mode")
        dist    = metrics["Mode_A"]["predictions"]
        assert dist["High"]   == 5
        assert dist["Low"]    == 3
        assert dist["Medium"] == 2


# ════════════════════════════════════════════════════════════════
#  Thresholds
# ════════════════════════════════════════════════════════════════

class TestFairnessThresholds:

    def test_disparate_impact_threshold_is_08(self):
        assert FairnessMonitor.DISPARATE_IMPACT_THRESHOLD == 0.8

    def test_accuracy_gap_threshold_is_010(self):
        assert FairnessMonitor.ACCURACY_GAP_THRESHOLD == 0.1

    @pytest.mark.parametrize("di,expected_status", [
        (0.9, "OK"),
        (0.8, "OK"),      # at threshold — borderline OK
        (0.79, "BIAS_DETECTED"),
        (0.5,  "BIAS_DETECTED"),
        (0.0,  "BIAS_DETECTED"),
    ])
    def test_di_threshold_boundary(self, di, expected_status):
        monitor  = FairnessMonitor()
        metrics  = {
            "Mode_A": {"positive_rate": 1.0},
            "Mode_B": {"positive_rate": di},
        }
        result = monitor.compute_disparate_impact(metrics)
        assert result["status"] == expected_status
