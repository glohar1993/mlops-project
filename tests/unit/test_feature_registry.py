"""
Unit tests for feature_registry.py — the single source of truth.
These tests MUST pass before any deployment.
"""
import pytest
import pandas as pd
from src.feature_registry import (
    FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAP, LABEL_MAP_REVERSE,
    apply_label_map, encode_operation_mode, MIN_ACCURACY, MIN_ACCURACY_GAIN,
    PSI_CRITICAL, PSI_WARNING, DRIFT_WINDOW,
)


class TestFeatureColumns:
    def test_exactly_14_features(self):
        assert len(FEATURE_COLUMNS) == 14

    def test_temporal_features_last_four(self):
        assert FEATURE_COLUMNS[-4:] == ["Year", "Month", "Day", "Hour"]

    def test_no_target_in_features(self):
        assert TARGET_COLUMN not in FEATURE_COLUMNS

    def test_no_duplicates(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_operation_mode_is_first(self):
        assert FEATURE_COLUMNS[0] == "Operation_Mode"


class TestLabelMap:
    def test_three_classes(self):
        assert set(LABEL_MAP.keys()) == {"High", "Low", "Medium"}

    def test_numeric_values_0_1_2(self):
        assert set(LABEL_MAP.values()) == {0, 1, 2}

    def test_reverse_map_is_inverse(self):
        for k, v in LABEL_MAP.items():
            assert LABEL_MAP_REVERSE[v] == k

    def test_apply_label_map_happy_path(self):
        s = pd.Series(["High", "Low", "Medium", "High"])
        result = apply_label_map(s)
        assert list(result) == [0, 1, 2, 0]

    def test_apply_label_map_raises_on_unknown(self):
        s = pd.Series(["High", "Unknown"])
        with pytest.raises(ValueError, match="Unknown"):
            apply_label_map(s)


class TestOperationModeEncoding:
    def test_numeric_passthrough(self):
        assert encode_operation_mode(0) == 0
        assert encode_operation_mode(1) == 1
        assert encode_operation_mode(2) == 2

    def test_string_numeric_passthrough(self):
        assert encode_operation_mode("0") == 0
        assert encode_operation_mode("1") == 1

    def test_unknown_value_returns_zero(self):
        assert encode_operation_mode("unknown_mode") == 0

    def test_none_returns_zero(self):
        assert encode_operation_mode(None) == 0


class TestThresholds:
    def test_accuracy_threshold_range(self):
        assert 0.5 <= MIN_ACCURACY <= 1.0

    def test_gain_threshold_is_positive(self):
        assert MIN_ACCURACY_GAIN > 0

    def test_psi_warning_less_than_critical(self):
        assert PSI_WARNING < PSI_CRITICAL

    def test_drift_window_positive(self):
        assert DRIFT_WINDOW >= 10
