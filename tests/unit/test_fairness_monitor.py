"""
Unit Tests — Fairness Monitor (src/fairness_monitor.py)
==========================================================
Tests: record, subgroup metrics, disparate impact, accuracy gap,
       full report, bias detection, edge cases
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.fairness_monitor import FairnessMonitor


@pytest.fixture
def monitor():
    return FairnessMonitor()


# ════════════════════════════════════════════════════════════════
#  Recording
# ════════════════════════════════════════════════════════════════

class TestRecording:

    def test_record_without_ground_truth(self, monitor):
        monitor.record({"Operation_Mode": "Mode_A"}, prediction=0)
        assert len(monitor._history) == 1

    def test_record_with_ground_truth(self, monitor):
        monitor.record({"Operation_Mode": "Mode_A"}, prediction=0, ground_truth=0)
        assert monitor._history[0]["ground_truth"] == 0

    def test_multiple_records(self, monitor):
        for i in range(50):
            monitor.record({"Operation_Mode": "Mode_A"}, prediction=i % 3)
        assert len(monitor._history) == 50


# ════════════════════════════════════════════════════════════════
#  Subgroup Metrics
# ════════════════════════════════════════════════════════════════

class TestSubgroupMetrics:

    def _make_records(self, group_preds):
        """Helper: group_preds = {"Mode_A": [0,1,0], "Mode_B": [1,1,2]}"""
        records = []
        for group, preds in group_preds.items():
            for p in preds:
                records.append({
                    "features":     {"Operation_Mode": group},
                    "prediction":   p,
                    "ground_truth": None,
                })
        return records

    def test_single_group(self, monitor):
        records = self._make_records({"Mode_A": [0, 0, 1, 2]})
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert "Mode_A" in m
        assert m["Mode_A"]["count"] == 4

    def test_two_groups(self, monitor):
        records = self._make_records({"Mode_A": [0, 0], "Mode_B": [1, 1]})
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert set(m.keys()) == {"Mode_A", "Mode_B"}

    def test_positive_rate_all_high(self, monitor):
        records = self._make_records({"Mode_A": [0, 0, 0]})   # all class-0 = High
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert m["Mode_A"]["positive_rate"] == pytest.approx(1.0)

    def test_positive_rate_none_high(self, monitor):
        records = self._make_records({"Mode_A": [1, 2, 1, 2]})  # no class-0
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert m["Mode_A"]["positive_rate"] == pytest.approx(0.0)

    def test_accuracy_computed_when_gt_available(self, monitor):
        records = [
            {"features": {"Operation_Mode": "Mode_A"}, "prediction": 0, "ground_truth": 0},  # correct
            {"features": {"Operation_Mode": "Mode_A"}, "prediction": 1, "ground_truth": 0},  # wrong
        ]
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert m["Mode_A"]["accuracy"] == pytest.approx(0.5)

    def test_accuracy_100_pct(self, monitor):
        records = [
            {"features": {"Operation_Mode": "Mode_B"}, "prediction": 2, "ground_truth": 2},
            {"features": {"Operation_Mode": "Mode_B"}, "prediction": 2, "ground_truth": 2},
        ]
        m = monitor.compute_subgroup_metrics(records, "Operation_Mode")
        assert m["Mode_B"]["accuracy"] == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════
#  Disparate Impact
# ════════════════════════════════════════════════════════════════

class TestDisparateImpact:

    def test_no_bias_equal_rates(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.8, "count": 100},
            "Mode_B": {"positive_rate": 0.8, "count": 100},
        }
        result = monitor.compute_disparate_impact(metrics)
        assert result["disparate_impact"] == pytest.approx(1.0)
        assert result["status"] == "OK"

    def test_bias_detected_below_threshold(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.9, "count": 100},
            "Mode_B": {"positive_rate": 0.5, "count": 100},  # 0.5/0.9 ≈ 0.56 < 0.8
        }
        result = monitor.compute_disparate_impact(metrics)
        assert result["disparate_impact"] < 0.8
        assert result["status"] == "BIAS_DETECTED"

    def test_di_above_threshold_is_ok(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.9, "count": 100},
            "Mode_B": {"positive_rate": 0.85, "count": 100},  # 0.85/0.9 ≈ 0.94 > 0.8
        }
        result = monitor.compute_disparate_impact(metrics)
        assert result["status"] == "OK"

    def test_single_group_insufficient(self, monitor):
        metrics = {"Mode_A": {"positive_rate": 0.5, "count": 50}}
        result = monitor.compute_disparate_impact(metrics)
        assert result["status"] == "OK"   # can't detect DI with 1 group

    def test_zero_positive_rate_no_crash(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.0, "count": 50},
            "Mode_B": {"positive_rate": 0.0, "count": 50},
        }
        result = monitor.compute_disparate_impact(metrics)
        assert result["status"] == "OK"


# ════════════════════════════════════════════════════════════════
#  Accuracy Gap
# ════════════════════════════════════════════════════════════════

class TestAccuracyGap:

    def test_no_gap_no_bias(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.5, "accuracy": 0.90},
            "Mode_B": {"positive_rate": 0.5, "accuracy": 0.90},
        }
        result = monitor.compute_accuracy_gap(metrics)
        assert result["status"] == "OK"
        assert result["accuracy_gap"] == pytest.approx(0.0)

    def test_gap_over_threshold_bias_detected(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.5, "accuracy": 0.95},
            "Mode_B": {"positive_rate": 0.5, "accuracy": 0.80},  # 15% gap
        }
        result = monitor.compute_accuracy_gap(metrics)
        assert result["accuracy_gap"] == pytest.approx(0.15)
        assert result["status"] == "BIAS_DETECTED"

    def test_gap_under_threshold_is_ok(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.5, "accuracy": 0.90},
            "Mode_B": {"positive_rate": 0.5, "accuracy": 0.85},  # 5% gap
        }
        result = monitor.compute_accuracy_gap(metrics)
        assert result["status"] == "OK"

    def test_identifies_best_and_worst_group(self, monitor):
        metrics = {
            "Mode_A": {"positive_rate": 0.5, "accuracy": 0.95},
            "Mode_B": {"positive_rate": 0.5, "accuracy": 0.70},
        }
        result = monitor.compute_accuracy_gap(metrics)
        assert result["best_group"]  == "Mode_A"
        assert result["worst_group"] == "Mode_B"


# ════════════════════════════════════════════════════════════════
#  Full Report
# ════════════════════════════════════════════════════════════════

class TestFullReport:

    def test_empty_history_returns_no_data(self, monitor):
        report = monitor.full_report()
        assert report["overall_status"] == "NO_DATA"

    def test_full_report_structure(self, monitor):
        for i in range(30):
            mode = "Mode_A" if i % 2 == 0 else "Mode_B"
            monitor.record(
                {"Operation_Mode": mode},
                prediction=i % 3,
                ground_truth=i % 3,
            )
        report = monitor.full_report()
        assert "overall_status"   in report
        assert "records_analyzed" in report
        assert "subgroup_metrics" in report
        assert "disparate_impact" in report
        assert "accuracy_gap"     in report
        assert report["records_analyzed"] == 30

    def test_report_detects_no_bias(self, monitor):
        for _ in range(20):
            for mode in ["Mode_A", "Mode_B", "Mode_C"]:
                monitor.record({"Operation_Mode": mode}, prediction=0,
                               ground_truth=0)
        report = monitor.full_report()
        assert report["disparate_impact"]["status"] == "OK"
