"""
Unit Tests — Batch Predictor (src/batch_predictor.py)
========================================================
Tests: DataFrame prediction, CSV prediction, chunking, progress callback,
       error handling, inject_model (no disk artifacts needed)
"""

import os
import pytest
import numpy as np
import pandas as pd
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.batch_predictor import BatchPredictor


# ── Minimal mock model / scaler ──────────────────────────────────────────────

class MockScaler:
    def transform(self, X):
        return X   # no-op


class MockModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)   # always predict class 0 = "High"

    def predict_proba(self, X):
        proba = np.zeros((len(X), 3))
        proba[:, 0] = 0.9
        proba[:, 1] = 0.05
        proba[:, 2] = 0.05
        return proba


@pytest.fixture
def predictor():
    bp = BatchPredictor(chunk_size=50)
    bp.inject_model(MockModel(), MockScaler())
    return bp


@pytest.fixture
def sample_df():
    """DataFrame with all required feature columns."""
    n = 200
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Operation_Mode":                   rng.choice([0, 1, 2], n),
        "Temperature_C":                    rng.uniform(60, 100, n),
        "Vibration_Hz":                     rng.uniform(1, 10, n),
        "Power_Consumption_kW":             rng.uniform(30, 90, n),
        "Network_Latency_ms":               rng.uniform(5, 100, n),
        "Packet_Loss_%":                    rng.uniform(0, 5, n),
        "Quality_Control_Defect_Rate_%":    rng.uniform(0, 15, n),
        "Production_Speed_units_per_hr":    rng.uniform(100, 400, n),
        "Predictive_Maintenance_Score":     rng.uniform(0, 1, n),
        "Error_Rate_%":                     rng.uniform(0, 10, n),
        "Year":  [2026] * n,
        "Month": [3]    * n,
        "Day":   [15]   * n,
        "Hour":  [10]   * n,
    })


# ════════════════════════════════════════════════════════════════
#  DataFrame Prediction
# ════════════════════════════════════════════════════════════════

class TestPredictDataframe:

    def test_returns_dataframe(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_prediction_column(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert "prediction" in result.columns

    def test_adds_confidence_column(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert "confidence" in result.columns

    def test_adds_prediction_id_column(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert "prediction_id" in result.columns

    def test_output_row_count_matches_input(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert len(result) == len(sample_df)

    def test_prediction_values_are_valid(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        valid  = {"High", "Low", "Medium", "Unknown", "ERROR"}
        assert set(result["prediction"].unique()).issubset(valid)

    def test_confidence_between_0_and_1(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert result["confidence"].between(0, 1).all()

    def test_mock_model_predicts_high(self, predictor, sample_df):
        result = predictor.predict_dataframe(sample_df)
        assert (result["prediction"] == "High").all()

    def test_chunking_produces_same_result(self, sample_df):
        """Chunk size should not affect final results."""
        bp1 = BatchPredictor(chunk_size=10)
        bp2 = BatchPredictor(chunk_size=200)
        bp1.inject_model(MockModel(), MockScaler())
        bp2.inject_model(MockModel(), MockScaler())
        r1 = bp1.predict_dataframe(sample_df)
        r2 = bp2.predict_dataframe(sample_df)
        assert list(r1["prediction"]) == list(r2["prediction"])

    def test_progress_callback_called(self, predictor, sample_df):
        calls = []
        def callback(processed, total, errors):
            calls.append((processed, total, errors))
        predictor.predict_dataframe(sample_df, progress_callback=callback)
        assert len(calls) > 0
        # Last call should have processed == total
        assert calls[-1][1] == len(sample_df)

    def test_empty_dataframe(self, predictor):
        df     = pd.DataFrame(columns=["Operation_Mode", "Temperature_C"])
        result = predictor.predict_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.parametrize("n_rows", [1, 10, 100, 500])
    def test_various_row_counts(self, predictor, n_rows):
        rng = np.random.default_rng(n_rows)
        df  = pd.DataFrame({
            "Operation_Mode": rng.choice([0, 1, 2], n_rows),
            "Temperature_C":  rng.uniform(60, 100, n_rows),
            "Vibration_Hz":   rng.uniform(1, 10, n_rows),
            "Power_Consumption_kW":          rng.uniform(30, 90, n_rows),
            "Network_Latency_ms":            rng.uniform(5, 100, n_rows),
            "Packet_Loss_%":                 rng.uniform(0, 5, n_rows),
            "Quality_Control_Defect_Rate_%": rng.uniform(0, 15, n_rows),
            "Production_Speed_units_per_hr": rng.uniform(100, 400, n_rows),
            "Predictive_Maintenance_Score":  rng.uniform(0, 1, n_rows),
            "Error_Rate_%":                  rng.uniform(0, 10, n_rows),
        })
        result = predictor.predict_dataframe(df)
        assert len(result) == n_rows


# ════════════════════════════════════════════════════════════════
#  CSV Prediction
# ════════════════════════════════════════════════════════════════

class TestPredictCSV:

    def test_predict_csv_creates_output(self, predictor, sample_df, tmp_path):
        in_path  = str(tmp_path / "input.csv")
        out_path = str(tmp_path / "output.csv")
        sample_df.to_csv(in_path, index=False)
        summary = predictor.predict_csv(in_path, output_path=out_path)
        assert os.path.exists(out_path)
        assert summary["input_rows"] == len(sample_df)

    def test_summary_has_required_fields(self, predictor, sample_df, tmp_path):
        in_path = str(tmp_path / "input.csv")
        sample_df.to_csv(in_path, index=False)
        summary = predictor.predict_csv(in_path)
        for field in ["input_rows", "output_rows", "output_path",
                      "elapsed_seconds", "rows_per_second",
                      "prediction_distribution", "errors"]:
            assert field in summary

    def test_prediction_distribution_populated(self, predictor, sample_df, tmp_path):
        in_path = str(tmp_path / "input.csv")
        sample_df.to_csv(in_path, index=False)
        summary = predictor.predict_csv(in_path)
        assert len(summary["prediction_distribution"]) > 0
