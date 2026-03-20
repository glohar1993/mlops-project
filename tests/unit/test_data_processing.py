"""
Unit tests for DataProcessing — validates transforms are deterministic.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from src.feature_registry import FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAP


class TestDataProcessingTransforms:

    def test_label_encoding_deterministic(self, sample_raw_df):
        """Same input → same numeric labels, regardless of run order."""
        from src.feature_registry import apply_label_map
        s = sample_raw_df[TARGET_COLUMN]
        result1 = apply_label_map(s)
        result2 = apply_label_map(s)
        assert list(result1) == list(result2)

    def test_timestamp_extraction_correct(self, sample_raw_df):
        """Timestamp → Year/Month/Day/Hour extracted correctly."""
        df = sample_raw_df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Year"]  = df["Timestamp"].dt.year
        df["Month"] = df["Timestamp"].dt.month
        df["Day"]   = df["Timestamp"].dt.day
        df["Hour"]  = df["Timestamp"].dt.hour
        assert df["Year"].min()  >= 2026
        assert df["Month"].between(1, 12).all()
        assert df["Day"].between(1, 31).all()
        assert df["Hour"].between(0, 23).all()

    def test_all_feature_columns_present_after_processing(self, sample_raw_df, tmp_path):
        """After processing, all 14 model features must exist in the dataframe."""
        from src.data_processing import DataProcessing
        input_file = tmp_path / "data.csv"
        sample_raw_df.to_csv(input_file, index=False)
        output_dir = str(tmp_path / "processed")

        processor = DataProcessing(str(input_file), output_dir)
        processor.load_data()
        processor.preprocess()

        for col in FEATURE_COLUMNS:
            assert col in processor.df.columns, f"Missing feature column: {col}"

    def test_no_nan_in_feature_columns_after_scaling(self, sample_raw_df, tmp_path):
        """No NaN values in output feature matrices."""
        from src.data_processing import DataProcessing
        input_file = tmp_path / "data.csv"
        sample_raw_df.to_csv(input_file, index=False)
        output_dir = str(tmp_path / "processed")

        processor = DataProcessing(str(input_file), output_dir)
        processor.run()

        import joblib
        X_train = joblib.load(os.path.join(output_dir, "X_train.pkl"))
        X_test  = joblib.load(os.path.join(output_dir, "X_test.pkl"))
        assert not np.isnan(X_train).any(), "NaN in X_train"
        assert not np.isnan(X_test).any(),  "NaN in X_test"

    def test_train_test_split_ratio(self, sample_raw_df, tmp_path):
        """Test set should be ~20% of total data."""
        from src.data_processing import DataProcessing
        input_file = tmp_path / "data.csv"
        sample_raw_df.to_csv(input_file, index=False)
        output_dir = str(tmp_path / "processed")

        processor = DataProcessing(str(input_file), output_dir)
        processor.run()

        import joblib
        X_train = joblib.load(os.path.join(output_dir, "X_train.pkl"))
        X_test  = joblib.load(os.path.join(output_dir, "X_test.pkl"))
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.18 <= test_ratio <= 0.22, f"Unexpected test ratio: {test_ratio:.2f}"
